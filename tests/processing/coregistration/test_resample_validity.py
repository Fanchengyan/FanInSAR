"""P0078 validity through production complex resampling consumers."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coregistration.offsets import (
    _resample_complex_deramped_reramp_with_validity,
    _resample_complex_with_validity,
    resample_complex,
    resample_complex_deramped_reramp,
)
from faninsar.processing.coregistration.tops.deramp import TOPSCarrierModel

torch = pytest.importorskip("torch")

HEIGHT = WIDTH = 32
LANCZOS_A = 4
STRIPE = slice(16, 20)
COMPLEX_NAN = np.complex64(np.nan + 1j * np.nan)


def _ones() -> np.ndarray:
    """Return a 32x32 complex-ones source."""
    return np.ones((HEIGHT, WIDTH), dtype=np.complex64)


def _resample(
    samples: np.ndarray,
    valid: np.ndarray,
    *,
    range_offset_px: np.ndarray | float = 0.5,
    azimuth_offset_px: np.ndarray | float = 0.5,
    row_chunk: int = 8,
    chunk_size: int = 32,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Call the private mask-aware production worker on CPU."""
    return _resample_complex_with_validity(
        samples,
        valid,
        range_offset_px=range_offset_px,
        azimuth_offset_px=azimuth_offset_px,
        lanczos_a=LANCZOS_A,
        row_chunk=row_chunk,
        device=device,
        chunk_size=chunk_size,
    )


def _is_complex_nan(values: np.ndarray) -> np.ndarray:
    """Return True where both real and imaginary parts are NaN."""
    return np.isnan(values.real) & np.isnan(values.imag)


def _carrier() -> TOPSCarrierModel:
    """Return a minimal TOPS carrier for deramp/reramp tests."""
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.005,
        range_sampling_rate_hz=32.0e6,
        azimuth_time_interval_s=0.002,
        doppler_centroid_hz=(20.0, 0.0, 0.0),
        doppler_t0_s=0.005,
        fm_rate_hz_s=(-2200.0, 0.0, 0.0),
        fm_t0_s=0.005,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.005,
        azimuth_steering_rate_hz_s=800.0,
    )


def test_masked_zero_filled_stripe_is_nan_at_fractional_offsets() -> None:
    """Outputs whose Lanczos support meets an invalid stripe must be complex NaN."""
    samples = _ones()
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    valid[:, STRIPE] = False
    filled = samples.copy()
    filled[:, STRIPE] = 0
    output, mask = _resample(filled, valid, row_chunk=5, chunk_size=24)
    assert output.shape == samples.shape
    assert mask.dtype == np.bool_
    assert mask.shape == samples.shape
    # At offset 0.5 every 8-tap neighbourhood that overlaps columns 16:20 is
    # incomplete. Column 18 is interior to that neighbourhood.
    assert _is_complex_nan(output[16, 18])
    assert not mask[16, 18]
    # Complete-support baseline of all-valid ones stays finite and unmasked.
    baseline, baseline_mask = _resample(_ones(), np.ones((HEIGHT, WIDTH), dtype=bool))
    assert np.isfinite(baseline[16, 10])
    assert baseline_mask[16, 10]
    assert not _is_complex_nan(baseline[16, 10])


def test_valid_complex_zero_stays_valid() -> None:
    """A genuine zero-valued echo with a true source mask remains valid."""
    samples = _ones()
    samples[10, 10] = np.complex64(0.0)
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    output, mask = _resample(
        samples,
        valid,
        range_offset_px=0.0,
        azimuth_offset_px=0.0,
    )
    assert output[10, 10] == np.complex64(0.0)
    assert mask[10, 10]
    assert np.isfinite(output[10, 10])


def test_identical_zero_stripe_is_valid_when_source_mask_is_true() -> None:
    """Zero-filled samples that are declared valid stay finite."""
    samples = _ones()
    samples[:, STRIPE] = 0
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    output, mask = _resample(
        samples,
        valid,
        range_offset_px=0.0,
        azimuth_offset_px=0.0,
    )
    assert mask[:, STRIPE].all()
    assert np.isfinite(output[:, STRIPE]).all()
    assert np.isfinite(output[16, 18])
    assert output[16, 18] == np.complex64(0.0)


def test_exact_integer_ignores_invalid_zero_weight_neighbor() -> None:
    """Exact integer mapping stays valid when only a zero-weight neighbor is missing."""
    samples = _ones()
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    valid[8, 15] = False
    samples[8, 15] = 0
    output, mask = _resample(
        samples,
        valid,
        range_offset_px=0.0,
        azimuth_offset_px=0.0,
    )
    assert mask[8, 16]
    assert np.isfinite(output[8, 16])
    np.testing.assert_allclose(output[8, 16], 1.0 + 0.0j, atol=1e-5)


def test_nextafter_integer_invalidates_nonzero_neighbor() -> None:
    """The next representable coordinate above an integer uses the neighbor tap."""
    samples = _ones()
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    valid[8, 15] = False
    samples[8, 15] = 0
    source_col = np.nextafter(16.0, np.inf)
    range_offset = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    range_offset[8, 16] = 16.0 - source_col
    output, mask = _resample(
        samples,
        valid,
        range_offset_px=range_offset,
        azimuth_offset_px=0.0,
    )
    assert not mask[8, 16]
    assert _is_complex_nan(output[8, 16])


def test_negative_weight_invalid_tap_is_not_ignored() -> None:
    """A negative Lanczos weight at 16.5 still requires that tap to be valid."""
    samples = _ones()
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    # Source column 16.5, a=4: taps 13..20; column 15 is a signed contributor.
    valid[:, 15] = False
    samples[:, 15] = 0
    output, mask = _resample(
        samples,
        valid,
        range_offset_px=0.5,
        azimuth_offset_px=0.0,
    )
    # Output column 17 maps to source 16.5.
    assert not mask[8, 17]
    assert _is_complex_nan(output[8, 17])
    complete, complete_mask = _resample(
        _ones(),
        np.ones((HEIGHT, WIDTH), dtype=bool),
        range_offset_px=0.5,
        azimuth_offset_px=0.0,
    )
    assert complete_mask[8, 17]
    assert np.isfinite(complete[8, 17])


def test_zero_weight_nan_neighbor_does_not_poison_integer_sample() -> None:
    """Inactive taps must be selected to zero before multiply so 0*NaN cannot leak."""
    samples = _ones()
    samples[8, 15] = np.complex64(np.nan + 1j * np.nan)
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    valid[8, 15] = False
    output, mask = _resample(
        samples,
        valid,
        range_offset_px=0.0,
        azimuth_offset_px=0.0,
    )
    assert mask[8, 16]
    assert np.isfinite(output[8, 16])
    np.testing.assert_allclose(output[8, 16], 1.0 + 0.0j, atol=1e-5)


def test_nan_and_inf_coordinates_are_invalid_not_clamped_valid() -> None:
    """Non-finite mapped coordinates cannot publish a clamped edge sample."""
    samples = _ones()
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    range_offset = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    azimuth_offset = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    range_offset[4, 4] = np.nan
    azimuth_offset[5, 5] = np.inf
    output, mask = _resample(
        samples,
        valid,
        range_offset_px=range_offset,
        azimuth_offset_px=azimuth_offset,
    )
    assert not mask[4, 4]
    assert not mask[5, 5]
    assert _is_complex_nan(output[4, 4])
    assert _is_complex_nan(output[5, 5])
    assert mask[6, 6]
    assert np.isfinite(output[6, 6])


def test_out_of_bounds_contributing_tap_is_invalid() -> None:
    """Original in-bounds predicates win over clamped gathers at native edges."""
    samples = _ones()
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    output, mask = _resample(
        samples,
        valid,
        range_offset_px=0.5,
        azimuth_offset_px=0.5,
    )
    assert not mask[0, 0]
    assert _is_complex_nan(output[0, 0])
    assert mask[16, 10]
    assert np.isfinite(output[16, 10])


def test_nonfinite_source_sample_invalidates_contributing_output() -> None:
    """Finiteness of the actual interpolation inputs is part of source validity."""
    samples = _ones()
    samples[16, 16] = np.complex64(np.nan + 1j * np.nan)
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    output, mask = _resample(
        samples,
        valid,
        range_offset_px=0.5,
        azimuth_offset_px=0.5,
    )
    assert not mask[16, 16]
    assert _is_complex_nan(output[16, 16])


def test_public_resample_complex_keeps_array_return_and_complete_support() -> None:
    """The public unmasked function still returns an array and matches itself."""
    samples = _ones()
    public = resample_complex(
        samples,
        range_offset_px=0.5,
        azimuth_offset_px=0.5,
        row_chunk=8,
        device="cpu",
    )
    assert isinstance(public, np.ndarray)
    assert public.shape == samples.shape
    masked, mask = _resample(
        samples,
        np.ones((HEIGHT, WIDTH), dtype=bool),
        row_chunk=8,
    )
    assert mask[8:24, 8:24].all()
    np.testing.assert_allclose(public[mask], masked[mask], atol=1e-5)


def test_deramped_reramp_carries_source_mask_and_preserves_valid_zero() -> None:
    """Raw deramp/resample/reramp consumers persist NaN and valid zeros."""
    samples = _ones()
    samples[12, 12] = np.complex64(0.0)
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    valid[:, STRIPE] = False
    filled = samples.copy()
    filled[:, STRIPE] = 0
    output, mask = _resample_complex_deramped_reramp_with_validity(
        filled,
        valid,
        secondary_carrier=_carrier(),
        range_offset_px=0.5,
        azimuth_offset_px=0.5,
        row_chunk=8,
        device="cpu",
    )
    assert output.shape == samples.shape
    assert not mask[16, 18]
    assert _is_complex_nan(output[16, 18])
    integer, integer_mask = _resample_complex_deramped_reramp_with_validity(
        samples,
        np.ones((HEIGHT, WIDTH), dtype=bool),
        secondary_carrier=_carrier(),
        range_offset_px=0.0,
        azimuth_offset_px=0.0,
        device="cpu",
    )
    assert integer_mask[12, 12]
    assert integer[12, 12] == np.complex64(0.0)
    public = resample_complex_deramped_reramp(
        samples,
        secondary_carrier=_carrier(),
        range_offset_px=0.0,
        azimuth_offset_px=0.0,
        device="cpu",
    )
    assert isinstance(public, np.ndarray)
    np.testing.assert_allclose(public, integer, atol=1e-4)
