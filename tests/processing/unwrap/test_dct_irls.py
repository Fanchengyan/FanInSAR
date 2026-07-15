"""Tests for clean-room DCT + IRLS unwrapping."""

from __future__ import annotations

import numpy as np

from faninsar.processing.unwrap import dct_irls_unwrap, unwrap, wrap_phase


def test_dct_irls_unwraps_linear_ramp() -> None:
    """DCT-IRLS recovers a smooth linear ramp (rewrap residual small)."""
    y, x = np.mgrid[0:24, 0:24]
    true = 0.35 * x + 0.15 * y
    wrapped = wrap_phase(true)
    result = dct_irls_unwrap(wrapped, max_iter=30, tol=1e-4, device="cpu")
    residual = wrap_phase(result.unwrapped_phase - true)
    assert float(np.sqrt(np.mean(residual**2))) < 0.15
    assert result.iterations >= 1
    assert result.unwrapped_phase.shape == true.shape


def test_dct_irls_api_dispatch() -> None:
    """API method='dct_irls' returns normalized CommonUnwrapResult."""
    _, x = np.mgrid[0:16, 0:16]
    true = 0.2 * x
    wrapped = wrap_phase(true)
    out = unwrap(
        wrapped,
        method="dct_irls",
        irls_kwargs={"device": "cpu", "max_iter": 15},
    )
    assert out.method == "dct_irls"
    assert out.unwrapped_phase.shape == wrapped.shape
    assert float(np.max(np.abs(out.unwrapped_phase - true))) < 2.0


def test_dct_irls_memory_bounded_medium_grid() -> None:
    """Medium grid completes without allocating sparse operators (smoke)."""
    rng = np.random.default_rng(0)
    h, w = 128, 256
    true = 0.05 * np.arange(w)[None, :] + 0.02 * np.arange(h)[:, None]
    true = true + 0.01 * rng.standard_normal((h, w))
    wrapped = wrap_phase(true)
    result = dct_irls_unwrap(wrapped, max_iter=5, device="cpu")
    assert result.unwrapped_phase.shape == (h, w)
    assert np.isfinite(result.unwrapped_phase).all()
