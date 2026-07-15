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
