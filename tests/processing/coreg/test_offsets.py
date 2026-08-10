"""Tests for coregistration offsets and complex resampling."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coreg import (
    combine_offset_fields,
    estimate_global_shift,
    estimate_patch_amplitude_shift,
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


def _sar_like_amplitude(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Band-limited random amplitude with sparse bright scatterers (SAR-like)."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(size=shape), sigma=1.5)
    # Sparse bright peaks improve local uniqueness for Ampcor
    n_peaks = max(20, shape[0] * shape[1] // 2000)
    for _ in range(n_peaks):
        az = int(rng.integers(10, shape[0] - 10))
        rg = int(rng.integers(10, shape[1] - 10))
        field[az - 1 : az + 2, rg - 1 : rg + 2] += float(rng.uniform(3.0, 8.0))
    return np.abs(field).astype(np.float32)


def test_estimate_patch_amplitude_shift_recovers_injected_offset() -> None:
    """Multi-window Ampcor recovers an injected integer residual."""
    texture = _sar_like_amplitude((256, 512), seed=2)
    reference = texture.astype(np.complex64)
    secondary = np.roll(np.roll(reference, shift=2, axis=1), shift=-1, axis=0)
    result = estimate_patch_amplitude_shift(
        reference,
        secondary,
        window_az=32,
        window_rg=64,
        search_az=8,
        search_rg=8,
        n_az=8,
        n_rg=12,
        snr_threshold=3.0,
        max_abs_residual=4.0,
        margin_rg=40,
        margin_az=40,
    )
    assert result.n_valid > 0
    assert result.range_shift_px == pytest.approx(2.0, abs=0.15)
    assert result.azimuth_shift_px == pytest.approx(-1.0, abs=0.15)


def test_estimate_patch_amplitude_shift_zero_when_aligned() -> None:
    """Aligned scenes yield near-zero residual after SNR cull."""
    texture = _sar_like_amplitude((200, 400), seed=3)
    reference = texture.astype(np.complex64)
    result = estimate_patch_amplitude_shift(
        reference,
        reference.copy(),
        window_az=32,
        window_rg=48,
        search_az=6,
        search_rg=6,
        n_az=6,
        n_rg=8,
        snr_threshold=3.0,
        max_abs_residual=1.2,
        margin_rg=40,
        margin_az=40,
    )
    assert result.n_valid > 0
    assert abs(result.range_shift_px) < 0.15
    assert abs(result.azimuth_shift_px) < 0.15


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
