"""Tests for clean-room IRLS unwrapping."""

from __future__ import annotations

import numpy as np

from faninsar.processing.unwrap import irls_unwrap, wrap_phase


def test_irls_unwraps_linear_ramp_modulo_offset() -> None:
    """IRLS recovers a linear ramp up to one global 2π offset."""
    y, x = np.mgrid[0:24, 0:24]
    true = 0.35 * x + 0.15 * y
    wrapped = wrap_phase(true)
    result = irls_unwrap(wrapped, max_iter=30, tol=1e-4)
    residual = wrap_phase(result.unwrapped_phase - true)
    assert float(np.sqrt(np.mean(residual**2))) < 0.1
    assert result.iterations >= 1


def test_irls_rewrap_residual_is_small_for_smooth_field() -> None:
    """Rewrap residual stays small for a smooth synthetic field."""
    _, x = np.mgrid[0:16, 0:16]
    true = 0.2 * x
    wrapped = wrap_phase(true)
    result = irls_unwrap(wrapped)
    assert float(np.max(np.abs(result.rewrap_residual))) < 0.5


def test_irls_preserves_nan_mask_and_labels_disconnected_components() -> None:
    """IRLS must solve valid islands without bridging NaN gaps."""
    y, x = np.mgrid[0:18, 0:24]
    truth = 0.45 * x + 0.2 * y
    wrapped = wrap_phase(truth)
    wrapped[:, 11:13] = np.nan
    coherence = np.ones_like(wrapped, dtype=np.float32)
    coherence[:, 11:13] = np.nan

    result = irls_unwrap(
        wrapped,
        coherence,
        device="cpu",
        max_iter=8,
        cg_max_iter=30,
        conncomp_size=2,
    )

    assert np.isnan(result.unwrapped_phase[:, 11:13]).all()
    assert set(np.unique(result.connected_components)) == {0, 1, 2}
    valid = np.isfinite(wrapped)
    residual = wrap_phase(result.unwrapped_phase[valid] - truth[valid])
    assert float(np.sqrt(np.mean(residual**2))) < 0.1
