"""Locality-aware Dask wrappers for validated CPU operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import dask.array as da
import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.runtime.execution import (
    Backend,
    ChunkLayout,
    DTypePolicy,
    ExecutionProfile,
    Locality,
    OperationPlan,
    validate_execution,
)

logger = setup_logger(__name__)


def _normalize_chunks(
    chunks: tuple[Any, ...],
) -> tuple[tuple[int, ...], ...]:
    """Normalize Dask chunk tuples to nested integer tuples."""
    normalized: list[tuple[int, ...]] = []
    for axis in chunks:
        if isinstance(axis, tuple):
            normalized.append(tuple(int(value) for value in axis))
        else:
            normalized.append((int(axis),))
    return tuple(normalized)


def _as_dask_array(
    array: np.ndarray | da.Array,
    chunks: tuple[tuple[int, ...], ...] | tuple[int, ...] | str = "auto",
) -> da.Array:
    """Convert a NumPy array to Dask or rechunk an existing Dask array."""
    if isinstance(array, da.Array):
        if chunks != "auto":
            return array.rechunk(chunks)
        return array
    return da.from_array(np.asarray(array), chunks=chunks)


def map_elementwise(
    func: Callable[..., np.ndarray],
    *arrays: np.ndarray | da.Array,
    name: str = "elementwise",
    dtype: np.dtype | None = None,
    profile: ExecutionProfile | None = None,
    **kwargs: Any,
) -> da.Array:
    """Apply an elementwise kernel with locality preflight validation."""
    if not arrays:
        message = "map_elementwise requires at least one input array"
        logger.error(message)
        raise ValueError(message)
    dask_arrays = [_as_dask_array(array) for array in arrays]
    sample = dask_arrays[0]
    plan = OperationPlan(
        name=name,
        locality=Locality.ELEMENTWISE,
        backend=Backend.NUMPY,
        dtype_policy=DTypePolicy.SCIENTIFIC,
        layout=ChunkLayout(
            shape=tuple(int(size) for size in sample.shape),
            chunks=_normalize_chunks(sample.chunks),
        ),
        itemsize=max(int(np.dtype(sample.dtype).itemsize), 1),
        temporary_arrays=1,
    )
    validate_execution(plan, profile or ExecutionProfile.local_cpu())
    out_dtype = dtype or sample.dtype
    return da.map_blocks(func, *dask_arrays, dtype=out_dtype, **kwargs)


def map_finite_halo(
    func: Callable[..., np.ndarray],
    array: np.ndarray | da.Array,
    *,
    depth: int | tuple[int, ...],
    name: str = "finite-halo",
    dtype: np.dtype | None = None,
    profile: ExecutionProfile | None = None,
    **kwargs: Any,
) -> da.Array:
    """Apply a finite-halo kernel via ``map_overlap`` after preflight checks."""
    darr = _as_dask_array(array)
    if isinstance(depth, int):
        halo = tuple(depth for _ in darr.shape)
    else:
        halo = tuple(int(value) for value in depth)
    plan = OperationPlan(
        name=name,
        locality=Locality.FINITE_HALO,
        backend=Backend.NUMPY,
        dtype_policy=DTypePolicy.SCIENTIFIC,
        layout=ChunkLayout(
            shape=tuple(int(size) for size in darr.shape),
            chunks=_normalize_chunks(darr.chunks),
        ),
        itemsize=max(int(np.dtype(darr.dtype).itemsize), 1),
        temporary_arrays=2,
        halo=halo,
    )
    validate_execution(plan, profile or ExecutionProfile.local_cpu())
    return da.map_overlap(
        func,
        darr,
        depth=depth,
        dtype=dtype or darr.dtype,
        boundary="none",
        **kwargs,
    )


def compute_with_profile(
    array: da.Array,
    *,
    profile: ExecutionProfile | None = None,
) -> np.ndarray:
    """Compute a Dask array after logging the active execution profile."""
    active = profile or ExecutionProfile.local_cpu()
    logger.info(
        "Computing Dask graph with %s workers (memory budget %s bytes)",
        len(active.workers),
        active.memory.usable_bytes,
    )
    return np.asarray(array.compute())
