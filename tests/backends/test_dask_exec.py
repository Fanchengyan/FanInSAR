"""Tests for locality-aware Dask execution wrappers."""

from __future__ import annotations

import numpy as np

from faninsar.backends.dask_exec import (
    compute_with_profile,
    map_elementwise,
    map_finite_halo,
)
from faninsar.backends.execution import ExecutionProfile


def test_map_elementwise_matches_numpy() -> None:
    """Elementwise Dask map matches eager NumPy for a simple kernel."""
    array = np.arange(16, dtype=np.float32).reshape(4, 4)
    lazy = map_elementwise(lambda block: block * 2.0, array, name="scale")
    result = compute_with_profile(lazy, profile=ExecutionProfile.local_cpu())
    np.testing.assert_allclose(result, array * 2.0)


def test_map_finite_halo_preserves_interior() -> None:
    """Finite-halo map_overlap preserves interior values for identity kernel."""
    array = np.arange(36, dtype=np.float32).reshape(6, 6)

    def identity(block: np.ndarray) -> np.ndarray:
        return block

    lazy = map_finite_halo(identity, array, depth=1, name="identity-halo")
    result = compute_with_profile(lazy)
    np.testing.assert_allclose(result[1:-1, 1:-1], array[1:-1, 1:-1])
