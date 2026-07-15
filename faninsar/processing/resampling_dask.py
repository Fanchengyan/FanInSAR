"""Dask-array Lanczos resampler (B6-E): map_blocks over coords, shared source.

CPU-parallel path for phase-preserving Lanczos resampling. Coordinate axis
``N`` is tiled with :func:`dask.array.map_blocks`; the 2-D source array is
**not** serialised into the task graph — workers share a single process-local
read-only reference (threads scheduler).

This is the memory-safe evolution of the legacy ``use_dask=True`` delayed path
in :mod:`faninsar.processing.resampling_torch`, which captured the full source
in every delayed task and could explode RSS under process schedulers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.resampling import _lanczos_resample_chunk

logger = setup_logger(__name__)

__all__ = [
    "DaskArrayLanczosConfig",
    "estimate_lanczos_chunk_size",
    "lanczos_resample_dask_array",
]

# Process-local worker state set for the duration of one compute() call.
# Threads share this reference; the source is never pickled into the graph.
_WORKER_DATA: np.ndarray | None = None
_WORKER_A: int = 4
_WORKER_MODE: str = "constant"
_WORKER_CVAL: float = 0.0
_WORKER_INNER_CHUNK: int = 128 * 1024

_MIN_CHUNK = 64 * 1024
_MAX_CHUNK = 128 * 1024
_DEFAULT_OUTER_CHUNK = 1_048_576
_DEFAULT_BUDGET_MIB = 4096.0


@dataclass(frozen=True)
class DaskArrayLanczosConfig:
    """Resolved runtime parameters for one :func:`lanczos_resample_dask_array` call."""

    chunk_size: int
    num_workers: int
    memory_budget_mib: float
    estimated_peak_mib: float


def estimate_lanczos_chunk_size(
    n: int,
    *,
    a: int = 4,
    itemsize: int = 8,
    num_workers: int = 1,
    memory_budget_mib: float = _DEFAULT_BUDGET_MIB,
) -> tuple[int, float]:
    """Choose a coordinate chunk size under a memory budget.

    Peak temporary storage per in-flight chunk is dominated by the
    ``(chunk, 2a, 2a)`` gather patch plus index/weight buffers inside
    :func:`~faninsar.processing.resampling._lanczos_resample_chunk`.

    Parameters
    ----------
    n : int
        Total number of output samples.
    a : int, optional
        Lanczos half-width. Default 4.
    itemsize : int, optional
        Source element size in bytes (8 for complex64). Default 8.
    num_workers : int, optional
        Concurrent thread workers. Default 1.
    memory_budget_mib : float, optional
        Soft budget for concurrent compute temps (MiB). Default 4096.

    Returns
    -------
    chunk_size : int
        Samples per coordinate chunk, clamped to ``[_MIN_CHUNK, _MAX_CHUNK]``.
    estimated_peak_mib : float
        Rough peak for ``num_workers`` concurrent patches (MiB), excluding
        source / coords / output.

    """
    if n <= 0:
        return _MIN_CHUNK, 0.0
    taps = 2 * a
    # patch + row/col taps (int64) + weights (float64) + small overhead
    per_sample = (
        taps * taps * itemsize
        + 2 * taps * 8
        + 2 * taps * 8
        + 64
    )
    workers = max(int(num_workers), 1)
    budget_bytes = float(memory_budget_mib) * 1024.0 * 1024.0
    # Reserve half the budget for source + coords + output + dask overhead.
    compute_budget = 0.5 * budget_bytes
    per_worker = compute_budget / workers
    raw = int(per_worker / max(per_sample, 1))
    chunk = max(_MIN_CHUNK, min(raw, _MAX_CHUNK, n))
    peak_mib = (workers * chunk * per_sample) / (1024.0 * 1024.0)
    return chunk, peak_mib


def _block_fn(coords_block: np.ndarray) -> np.ndarray:
    """Resample one outer coords block ``(2, c)`` with inner micro-chunks (S3)."""
    data = _WORKER_DATA
    if data is None:
        msg = "lanczos worker source is not initialised"
        raise RuntimeError(msg)
    block = np.asarray(coords_block)
    if block.ndim != 2 or block.shape[0] != 2:
        msg = f"expected coords block shape (2, c), got {block.shape}"
        raise ValueError(msg)
    rows = np.asarray(block[0], dtype=np.float64).reshape(-1)
    cols = np.asarray(block[1], dtype=np.float64).reshape(-1)
    n_loc = int(rows.shape[0])
    if n_loc == 0:
        return np.empty(0, dtype=data.dtype)
    inner = max(int(_WORKER_INNER_CHUNK), 1)
    if n_loc <= inner:
        return _lanczos_resample_chunk(
            data,
            rows,
            cols,
            a=_WORKER_A,
            mode=_WORKER_MODE,
            cval=_WORKER_CVAL,
        )
    out = np.empty(n_loc, dtype=data.dtype)
    for s in range(0, n_loc, inner):
        e = min(s + inner, n_loc)
        out[s:e] = _lanczos_resample_chunk(
            data,
            rows[s:e],
            cols[s:e],
            a=_WORKER_A,
            mode=_WORKER_MODE,
            cval=_WORKER_CVAL,
        )
    return out


def lanczos_resample_dask_array(
    data: np.ndarray,
    coords: np.ndarray,
    *,
    a: int = 4,
    mode: str = "constant",
    cval: float = 0.0,
    chunk_size: int | None = None,
    outer_chunk_size: int | None = None,
    num_workers: int | None = None,
    memory_budget_mib: float = _DEFAULT_BUDGET_MIB,
    scheduler: Literal["threads", "synchronous"] = "threads",
) -> np.ndarray:
    """Resample a 2-D array at fractional coords via dask.array map_blocks.

    Parameters
    ----------
    data : numpy.ndarray
        2-D source array (real or complex). Held as a single process-local
        read-only reference shared by all thread workers.
    coords : numpy.ndarray
        Output sample coordinates of shape ``(2, N)``.
    a : int, optional
        Lanczos half-width. Default 4.
    mode : str, optional
        Boundary handling: ``"constant"`` or ``"nearest"``. Default ``"constant"``.
    cval : float, optional
        Fill value for out-of-bounds taps. Default 0.0.
    chunk_size : int or None, optional
        Inner micro-chunk size (patch peak control, S3/S6). ``None`` selects
        a size from :func:`estimate_lanczos_chunk_size` under
        ``memory_budget_mib``.
    outer_chunk_size : int or None, optional
        Dask task granularity along ``N``. Larger values reduce scheduler
        overhead (S3). Default ``max(1_048_576, chunk_size)``.
    num_workers : int or None, optional
        Thread-pool size for the dask threads scheduler. ``None`` uses
        ``min(8, os.cpu_count() or 1)``.
    memory_budget_mib : float, optional
        Soft budget used when ``chunk_size`` is ``None``. Default 4096.
    scheduler : {"threads", "synchronous"}, optional
        Dask scheduler. Only in-process schedulers are supported so the
        shared source reference stays valid. Default ``"threads"``.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``N`` (dtype matches ``data``).

    Raises
    ------
    ValueError
        If shapes/modes are invalid or ``scheduler`` is not in-process.
    ImportError
        If ``dask`` is not installed.

    Notes
    -----
    The source array is **never** placed in the task graph. Workers read
    ``_WORKER_DATA`` set for the duration of ``compute()``. This keeps peak
    RSS near ``source + coords + output + num_workers * patch_temps``.

    For CUDA acceleration use
    :func:`faninsar.processing.resampling_torch.lanczos_resample_dask_torch`
    with ``device="cuda"`` instead of this CPU map_blocks path.

    Examples
    --------
    >>> import numpy as np
    >>> from faninsar.processing.resampling_dask import lanczos_resample_dask_array
    >>> rng = np.random.default_rng(0)
    >>> data = (rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))).astype(
    ...     np.complex64
    ... )
    >>> coords = np.vstack([rng.uniform(0, 31, 100), rng.uniform(0, 31, 100)])
    >>> out = lanczos_resample_dask_array(data, coords, chunk_size=50, num_workers=2)
    >>> out.shape
    (100,)

    """
    if data.ndim != 2:
        msg = "lanczos_resample_dask_array requires a 2-D source array"
        raise ValueError(msg)
    if mode not in ("constant", "nearest"):
        msg = f"unsupported mode {mode!r}; use 'constant' or 'nearest'"
        raise ValueError(msg)
    if scheduler not in ("threads", "synchronous"):
        msg = (
            f"scheduler={scheduler!r} is not supported; "
            "use 'threads' or 'synchronous' so the source is not pickled"
        )
        raise ValueError(msg)

    coords_arr = np.asarray(coords)
    if coords_arr.ndim != 2 or coords_arr.shape[0] != 2:
        msg = "coords must have shape (2, N)"
        raise ValueError(msg)

    rows = np.asarray(coords_arr[0], dtype=np.float64).reshape(-1)
    cols = np.asarray(coords_arr[1], dtype=np.float64).reshape(-1)
    n = int(rows.shape[0])
    if n == 0:
        return np.empty(0, dtype=data.dtype)

    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    num_workers = max(int(num_workers), 1)

    if chunk_size is None:
        chunk_size, est_peak = estimate_lanczos_chunk_size(
            n,
            a=a,
            itemsize=int(np.dtype(data.dtype).itemsize),
            num_workers=num_workers,
            memory_budget_mib=memory_budget_mib,
        )
    else:
        chunk_size = max(int(chunk_size), 1)
        chunk_size = min(chunk_size, n)
        _, est_peak = estimate_lanczos_chunk_size(
            n,
            a=a,
            itemsize=int(np.dtype(data.dtype).itemsize),
            num_workers=num_workers,
            memory_budget_mib=memory_budget_mib,
        )

    if outer_chunk_size is None:
        outer_chunk_size = min(n, max(_DEFAULT_OUTER_CHUNK, chunk_size))
    else:
        outer_chunk_size = max(int(outer_chunk_size), chunk_size)
        outer_chunk_size = min(outer_chunk_size, n)

    cfg = DaskArrayLanczosConfig(
        chunk_size=chunk_size,
        num_workers=num_workers,
        memory_budget_mib=float(memory_budget_mib),
        estimated_peak_mib=float(est_peak),
    )
    logger.info(
        "lanczos_resample_dask_array n=%d outer=%d inner=%d workers=%d "
        "scheduler=%s est_patch_peak=%.0f MiB",
        n,
        outer_chunk_size,
        cfg.chunk_size,
        cfg.num_workers,
        scheduler,
        cfg.estimated_peak_mib,
    )

    import dask
    import dask.array as da

    global _WORKER_DATA, _WORKER_A, _WORKER_MODE, _WORKER_CVAL, _WORKER_INNER_CHUNK
    coords2 = np.stack([rows, cols], axis=0)
    coords_da = da.from_array(coords2, chunks=(2, outer_chunk_size))

    _WORKER_DATA = data
    _WORKER_A = int(a)
    _WORKER_MODE = mode
    _WORKER_CVAL = float(cval)
    _WORKER_INNER_CHUNK = int(chunk_size)
    try:
        out_da = coords_da.map_blocks(
            _block_fn,
            dtype=data.dtype,
            drop_axis=0,
            meta=np.array([], dtype=data.dtype),
        )
        with dask.config.set(scheduler=scheduler, num_workers=cfg.num_workers):
            out = out_da.compute()
    finally:
        _WORKER_DATA = None

    return np.asarray(out, dtype=data.dtype)
