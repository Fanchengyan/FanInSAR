"""Tests for invalid multilook masking (no solid phase=0 black edges)."""

from __future__ import annotations

import numpy as np

from faninsar.processing.interferometry.pair import (
    form_interferogram,
    mask_invalid_looks,
)


def test_form_interferogram_masks_zero_power_looks_as_nan() -> None:
    """Looks with zero primary or secondary power must be NaN, not phase 0."""
    h, w = 16, 32
    primary = np.ones((h, w), dtype=np.complex64)
    secondary = np.ones((h, w), dtype=np.complex64)
    # Zero-fill the last two full-res rows → last ML row (looks=2) invalid.
    primary[-2:, :] = 0
    secondary[-2:, :] = 0
    # Also zero-fill first two columns of secondary.
    secondary[:, :2] = 0

    prod = form_interferogram(primary, secondary, multilook=(2, 4))
    assert prod.complex_ifg.shape == (8, 8)
    # Last azimuth look must be fully NaN.
    assert np.all(~np.isfinite(prod.complex_ifg[-1].real))
    assert np.all(np.isnan(prod.wrapped_phase[-1]))
    assert np.all(np.isnan(prod.coherence[-1]))
    # Interior should remain finite.
    assert np.all(np.isfinite(prod.complex_ifg[2:6, 2:6].real))
    assert np.all(np.isfinite(prod.wrapped_phase[2:6, 2:6]))


def test_mask_invalid_looks_converts_zero_amp_to_nan() -> None:
    """mask_invalid_looks replaces zero amplitude with NaN complex/phase."""
    ifg = np.ones((4, 5), dtype=np.complex64)
    ifg[-1, :] = 0
    coh = np.ones((4, 5), dtype=np.float32)
    out_ifg, out_coh, phase = mask_invalid_looks(ifg, coh)
    assert np.all(np.isnan(out_ifg[-1].real))
    assert np.all(np.isnan(phase[-1]))
    assert out_coh is not None
    assert np.all(np.isnan(out_coh[-1]))
    assert np.all(np.isfinite(phase[0]))
