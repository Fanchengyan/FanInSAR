"""Tests for coregistration offsets and complex resampling."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coreg import (
    combine_offset_fields,
    estimate_global_shift,
    geometry_shift_offsets,
    refine_peak_subpixel,
    resample_complex,
)


def test_estimate_global_shift_recovers_injected_offset() -> None:
    """Cross-correlation recovers an injected integer pixel shift."""
    rng = np.random.default_rng(1)
    reference = (rng.normal(size=(48, 48)) + 1j * rng.normal(size=(48, 48))).astype(
        np.complex64
    )
    # secondary is reference shifted by +3 range, -2 azimuth
    secondary = np.roll(np.roll(reference, shift=3, axis=1), shift=-2, axis=0)
    rg_shift, az_shift = estimate_global_shift(reference, secondary, max_shift=8)
    assert rg_shift == pytest.approx(3.0, abs=1e-3)
    assert az_shift == pytest.approx(-2.0, abs=1e-3)


def test_estimate_global_shift_subpixel_recovers_fractional_shift() -> None:
    """Subpixel cross-correlation recovers a fractional pixel shift."""
    from scipy.ndimage import shift

    # Use a Gaussian blob so the cross-correlation peak is sharp
    y, x = np.mgrid[-32:32, -32:32]
    blob = np.exp(-(x**2 + y**2) / 200.0)
    reference = (blob + 1j * blob).astype(np.complex64)
    # Apply a real fractional shift to both real and imaginary parts
    secondary = (
        shift(reference.real, (0.0, 0.4), order=3)
        + 1j * shift(reference.imag, (0.0, 0.4), order=3)
    ).astype(np.complex64)
    rg_shift, az_shift = estimate_global_shift(
        reference, secondary, max_shift=8, subpixel=True
    )
    assert rg_shift == pytest.approx(0.4, abs=0.05)
    assert az_shift == pytest.approx(0.0, abs=0.05)


def test_refine_peak_subpixel_parabolic_recovery() -> None:
    """Parabolic refinement recovers the true peak on a synthetic parabola."""
    x = np.arange(-2, 3)
    y = -((x - 0.3) ** 2)  # peak at +0.3
    sub = refine_peak_subpixel(y[:, None], (2, 0))[0]
    assert sub == pytest.approx(0.3, abs=1e-3)


def test_resample_complex_applies_constant_offset() -> None:
    """Resampling with a constant offset moves an impulse to the expected pixel."""
    samples = np.zeros((16, 16), dtype=np.complex64)
    samples[8, 8] = 1.0 + 2.0j
    offsets = geometry_shift_offsets(
        samples.shape,
        range_shift_px=2.0,
        azimuth_shift_px=-1.0,
    )
    # source = output - offset => impulse at (8,8) appears at output (7,10)
    out = resample_complex(
        samples,
        range_offset_px=offsets.range_offset_px,
        azimuth_offset_px=offsets.azimuth_offset_px,
        order=0,
    )
    peak = np.unravel_index(int(np.argmax(np.abs(out))), out.shape)
    assert peak == (7, 10)


def test_combine_offset_fields_adds_residuals() -> None:
    """Combined field adds ESD and amplitude residuals to geometry."""
    geometry = geometry_shift_offsets(
        (8, 8),
        range_shift_px=1.0,
        azimuth_shift_px=2.0,
    )
    combined = combine_offset_fields(
        geometry,
        esd_azimuth_shift_px=0.3,
        amplitude_residual_rg=0.1,
        amplitude_residual_az=-0.2,
    )
    assert np.allclose(combined.range_offset_px, 1.1)
    assert np.allclose(combined.azimuth_offset_px, 2.1)
    assert combined.coverage.all()
    assert np.all(combined.uncertainty_px >= geometry.uncertainty_px)
