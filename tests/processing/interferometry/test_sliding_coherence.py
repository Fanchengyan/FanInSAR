"""Regression: the stride-one coherence MLE stays exact after vectorization."""

from __future__ import annotations

import numpy as np

from faninsar.processing.interferometry.pair import _sliding_coherence


def _brute_force(
    primary: np.ndarray,
    secondary: np.ndarray,
    window: tuple[int, int],
) -> np.ndarray:
    """Compute the reference double-loop clipped-window MLE."""
    half_az, half_rg = window[0] // 2, window[1] // 2
    height, width = primary.shape
    output = np.full((height, width), np.nan, dtype=np.float64)
    finite = np.isfinite(primary.real) & np.isfinite(primary.imag)
    finite &= np.isfinite(secondary.real) & np.isfinite(secondary.imag)
    for row in range(height):
        r0, r1 = max(0, row - half_az), min(height, row + half_az + 1)
        for col in range(width):
            c0, c1 = max(0, col - half_rg), min(width, col + half_rg + 1)
            valid = finite[r0:r1, c0:c1]
            if np.count_nonzero(valid) < 2:
                continue
            p = primary[r0:r1, c0:c1][valid]
            s = secondary[r0:r1, c0:c1][valid]
            numerator = abs(np.sum(p * np.conjugate(s)))
            denominator = np.sqrt(np.sum(abs(p) ** 2) * np.sum(abs(s) ** 2))
            if np.isfinite(denominator) and denominator > 0:
                output[row, col] = numerator / denominator
    return output


def test_sliding_coherence_matches_bruteforce_reference() -> None:
    """The boxcar-sum estimator equals the clipped-window loop exactly."""
    rng = np.random.default_rng(11)
    shape = (23, 19)
    primary = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    secondary = primary + rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    primary[3, 5] = np.nan
    secondary[7, 2] = np.inf + 0j
    fast = _sliding_coherence(primary, secondary, (5, 3))
    slow = _brute_force(primary, secondary, (5, 3))
    np.testing.assert_allclose(fast, slow, rtol=1e-8, atol=1e-12, equal_nan=True)
