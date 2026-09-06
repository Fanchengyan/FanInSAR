"""Dask + Torch ComputeBackend implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class DaskTorchBackend:
    """Compute backend that prefers Dask arrays and Torch inside map_blocks.

    Cluster construction is **not** performed here — inject a Client externally
    when running distributed workloads.
    """

    name: str = "dask_torch"

    def __init__(self, *, device: str = "auto") -> None:
        """Configure the preferred Torch device string."""
        self.device = device

    def to_device_array(self, x: np.ndarray) -> Any:
        """Wrap *x* as a Dask array when dask is available, else NumPy."""
        try:
            import dask.array as da
        except ImportError:
            return np.ascontiguousarray(x)
        arr = np.ascontiguousarray(x)
        # Single-chunk wrap so map_blocks can partition later if rechunked.
        return da.from_array(arr, chunks=arr.shape)

    def map_blocks(
        self,
        fn: Callable[..., Any],
        *args: Any,
        chunks: tuple[int, ...] | None = None,
        dtype: np.dtype | type | None = None,
        resources: dict[str, int] | None = None,
    ) -> Any:
        """Map *fn* over Dask blocks or apply eagerly for NumPy inputs."""
        try:
            import dask.array as da
        except ImportError:
            result = fn(*(np.asarray(a) for a in args))
            arr = np.asarray(result)
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            return arr

        dask_args = []
        for a in args:
            if isinstance(a, da.Array):
                dask_args.append(a)
            else:
                na = np.asarray(a)
                dask_args.append(da.from_array(na, chunks=chunks or na.shape))

        meta_dtype = np.dtype(dtype) if dtype is not None else np.result_type(
            *[getattr(a, "dtype", np.float64) for a in dask_args]
        )
        # resources tag is recorded for injected Client scheduling; day-1
        # map_blocks does not construct a cluster.
        del resources
        return da.map_blocks(fn, *dask_args, dtype=meta_dtype, meta=np.array([], dtype=meta_dtype))

    def compute(
        self,
        *arrays: Any,
        sync: bool = True,
    ) -> tuple[np.ndarray, ...]:
        """Compute Dask arrays (or pass through NumPy)."""
        try:
            import dask
            import dask.array as da
        except ImportError:
            return tuple(np.asarray(a) for a in arrays)

        to_compute = []
        for a in arrays:
            if isinstance(a, da.Array):
                to_compute.append(a)
            else:
                to_compute.append(np.asarray(a))

        if not sync:
            return tuple(to_compute)  # type: ignore[return-value]

        results = dask.compute(*to_compute)
        return tuple(np.asarray(r) for r in results)


__all__ = ["DaskTorchBackend"]
