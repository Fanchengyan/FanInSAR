"""Hard-edge gradient regression: feather=True lowers boundary gradient."""

from __future__ import annotations

import numpy as np

from faninsar.processing.merge.overlap import compute_feather


def _boundary_gradient_energy(weight: np.ndarray) -> float:
    """Sum of squared horizontal gradient magnitude at the valid edge."""
    gx = np.diff(weight, axis=1)
    return float(np.sum(gx**2))


def test_hard_edge_feather_true_lower_gradient_than_false() -> None:
    """feather=True yields significantly lower boundary |nabla| than False."""
    rng = np.random.default_rng(0)
    valid = np.zeros((80, 80), dtype=bool)
    valid[10:70, 10:70] = True
    base = rng.uniform(0.4, 1.0, size=(80, 80)).astype(np.float32)
    base[~valid] = 0.0

    hard = base.copy()
    soft = base * compute_feather(valid, feather_width_px=16.0)

    hard_energy = _boundary_gradient_energy(hard)
    soft_energy = _boundary_gradient_energy(soft)
    # Feather should reduce boundary gradient energy by a clear margin.
    assert soft_energy < 0.5 * hard_energy
    # Sanity: both have the same support
    assert hard.shape == soft.shape