"""Tests for residual azimuth phase ramp estimation and removal."""

from __future__ import annotations

import numpy as np

from faninsar.processing.interferometry.flatten import (
    estimate_residual_azimuth_ramp,
    remove_azimuth_phase_ramp,
)


def test_estimate_and_remove_linear_azimuth_ramp() -> None:
    """Known linear az ramp on top of reference phase is recovered and removed."""
    height, width = 40, 80
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    # Smooth geometric-like reference (range ramp + mild 2D).
    ref_phase = 0.08 * xx + 0.01 * yy * (xx / width)
    true_c = -0.17
    ph = np.angle(np.exp(1j * (ref_phase + true_c * yy)))
    ifg = np.exp(1j * ph).astype(np.complex64)
    coh = np.ones((height, width), dtype=np.float32)

    est = estimate_residual_azimuth_ramp(ifg, ref_phase, coherence=coh)
    assert abs(est - true_c) < 0.02

    fixed = remove_azimuth_phase_ramp(ifg, est)
    residual = np.angle(fixed * np.exp(-1j * ref_phase))
    assert float(np.sqrt(np.mean(residual**2))) < 0.1
    assert float(np.abs(np.mean(np.exp(1j * residual)))) > 0.95


def test_remove_azimuth_ramp_zero_is_noop() -> None:
    """Zero ramp coefficient leaves the array unchanged."""
    ifg = np.ones((8, 8), dtype=np.complex64) * (1 + 1j)
    out = remove_azimuth_phase_ramp(ifg, 0.0)
    np.testing.assert_array_equal(out, ifg)
