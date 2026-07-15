"""Tests for chunked Lanczos resampling and memory-bounded behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coreg.offsets import resample_complex
from faninsar.processing.resampling import (
    DEFAULT_LANCZOS_CHUNK,
    lanczos_resample,
    lanczos_weights,
)


def test_lanczos_weights_normalise_at_integer() -> None:
    """Integer-aligned taps sum to one (DC preservation)."""
    w = lanczos_weights(np.array([0.0]), a=4)
    assert w.shape == (1, 8)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-12)


def test_lanczos_chunked_matches_monolithic() -> None:
    """Chunked path matches a single-chunk (monolithic) gather."""
    rng = np.random.default_rng(0)
    data = (rng.normal(size=(48, 64)) + 1j * rng.normal(size=(48, 64))).astype(
        np.complex64
    )
    rows, cols = np.mgrid[0:48:1.0, 0:64:1.0]
    # Fractional sub-pixel shift so the kernel is non-trivial
    coords = np.array([(rows + 0.35).ravel(), (cols - 0.2).ravel()])
    mono = lanczos_resample(data, coords, a=4, chunk_size=None)
    chunked = lanczos_resample(data, coords, a=4, chunk_size=257)
    assert mono.shape == chunked.shape
    np.testing.assert_allclose(chunked.real, mono.real, rtol=0, atol=1e-5)
    np.testing.assert_allclose(chunked.imag, mono.imag, rtol=0, atol=1e-5)


def test_lanczos_default_chunk_is_positive() -> None:
    """Production default chunk size is a positive bound."""
    assert DEFAULT_LANCZOS_CHUNK > 0


def test_resample_complex_row_chunk_matches_full() -> None:
    """Row-tiled resample_complex matches a single large-row chunk."""
    rng = np.random.default_rng(1)
    samples = (rng.normal(size=(40, 50)) + 1j * rng.normal(size=(40, 50))).astype(
        np.complex64
    )
    rg = np.full(samples.shape, 0.4, dtype=np.float64)
    az = np.full(samples.shape, -0.3, dtype=np.float64)
    full = resample_complex(
        samples, range_offset_px=rg, azimuth_offset_px=az, row_chunk=40
    )
    tiled = resample_complex(
        samples, range_offset_px=rg, azimuth_offset_px=az, row_chunk=7
    )
    np.testing.assert_allclose(tiled.real, full.real, rtol=0, atol=1e-5)
    np.testing.assert_allclose(tiled.imag, full.imag, rtol=0, atol=1e-5)


def test_lanczos_large_coordinate_count_stays_bounded() -> None:
    """Large coordinate count runs without allocating a full-frame patch.

    Uses a medium synthetic grid (not a full S1 burst) and asserts the
    chunked path completes and returns the right length. Peak-RSS of a
    true full burst is covered by the offline memory profile, not CI.
    """
    h, w = 256, 1024
    data = np.ones((h, w), dtype=np.complex64)
    n = h * w
    rows = np.arange(n, dtype=np.float64) // w + 0.25
    cols = np.arange(n, dtype=np.float64) % w + 0.1
    coords = np.array([rows, cols])
    out = lanczos_resample(data, coords, a=4, chunk_size=32_768)
    assert out.shape == (n,)
    assert out.dtype == np.complex64
    # Interior samples of a constant field should stay near 1+0j
    assert float(np.mean(np.abs(out))) == pytest.approx(1.0, abs=0.05)
