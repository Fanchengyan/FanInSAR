"""NumPy-only ComputeBackend implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class NumpyBackend:
    """Eager NumPy compute backend (reference / CPU path)."""

    name: str = "numpy"

    def to_device_array(self, x: np.ndarray) -> np.ndarray:
        """Return a contiguous NumPy copy of *x*."""
        return np.ascontiguousarray(x)

    def map_blocks(
        self,
        fn: Callable[..., Any],
        *args: Any,
        chunks: tuple[int, ...] | None = None,
        dtype: np.dtype | type | None = None,
        resources: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Apply *fn* to whole arrays eagerly (chunks ignored for day-1).

        Parameters
        ----------
        fn : callable
            Function applied to array arguments.
        *args : Any
            Positional array-like arguments.
        chunks : tuple of int, optional
            Ignored on the eager path (reserved for Dask backend).
        dtype : dtype, optional
            Cast result when provided.
        resources : dict, optional
            Ignored (resource tags are Dask-only).

        """
        del chunks, resources  # eager path
        result = fn(*args)
        arr = np.asarray(result)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def compute(
        self,
        *arrays: Any,
        sync: bool = True,
    ) -> tuple[np.ndarray, ...]:
        """Materialize arrays as NumPy ndarrays."""
        del sync
        return tuple(np.asarray(a) for a in arrays)


__all__ = ["NumpyBackend"]
