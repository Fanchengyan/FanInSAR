"""dask + torch + numpy Lanczos resampler (portable, optional GPU).

Implements the dask-torch-numpy contract from the project skill:

- Dask layer: task graph, tiling with halo, scheduling
- Torch layer: GPU/CPU computation, internal only
- Numpy layer: worker boundary contract

The kernel is the same phase-preserving separable Lanczos-``a`` as
:func:`faninsar.processing.resampling.lanczos_resample`. Results match the
NumPy reference within float rounding of the separable weighted sum.

Torch is optional. When ``device="cpu"`` and torch is unavailable, the path
falls back to the NumPy reference. When ``device="auto"`` and no CUDA/MPS is
available, it also uses CPU (NumPy or torch CPU).
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.resampling import DEFAULT_LANCZOS_CHUNK, lanczos_weights

logger = setup_logger(__name__)

__all__ = [
    "DaskTorchLanczosConfig",
    "lanczos_resample_dask_torch",
]

DeviceName = Literal["auto", "cpu", "cuda", "mps"]

# S1: process-local shared source for delayed threads (never pickled into tasks).
_DELAYED_DATA: np.ndarray | None = None
_DELAYED_ROWS: np.ndarray | None = None
_DELAYED_COLS: np.ndarray | None = None
_DELAYED_OUT: np.ndarray | None = None
_DELAYED_A: int = 4
_DELAYED_MODE: str = "constant"
_DELAYED_CVAL: float = 0.0
_DELAYED_DEV: object | None = None


def _delayed_write_range(start: int, stop: int) -> int:
    """Worker for S1 delayed path: write ``out[start:stop]``, return stop."""
    assert _DELAYED_DATA is not None
    assert _DELAYED_ROWS is not None
    assert _DELAYED_COLS is not None
    assert _DELAYED_OUT is not None
    _DELAYED_OUT[start:stop] = _torch_lanczos_block(
        _DELAYED_DATA,
        _DELAYED_ROWS[start:stop],
        _DELAYED_COLS[start:stop],
        a=_DELAYED_A,
        mode=_DELAYED_MODE,
        cval=_DELAYED_CVAL,
        dev=_DELAYED_DEV,
    )
    return stop


def _resolve_torch_device(device: DeviceName) -> object | None:
    """Return a torch device or None for the NumPy fallback path."""
    if device == "cpu":
        return None
    try:
        import torch
    except ImportError:
        if device in ("cuda", "mps"):
            from faninsar.processing.errors import reject_invalid_state

            reject_invalid_state(f"torch is required for device={device!r}")
        return None

    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()
    resolved: object | None = None
    if device == "auto":
        if cuda_ok:
            resolved = torch.device("cuda")
    elif device == "cuda":
        if not cuda_ok:
            from faninsar.processing.errors import reject_invalid_state

            reject_invalid_state("CUDA requested but torch.cuda is unavailable")
        resolved = torch.device("cuda")
    elif device == "mps":
        if not mps_ok:
            from faninsar.processing.errors import reject_invalid_state

            reject_invalid_state("MPS requested but torch.backends.mps is unavailable")
        resolved = torch.device("mps")
    return resolved


def _cleanup_gpu(dev: object | None) -> None:
    """Release GPU caches after a torch-assisted block."""
    if dev is None:
        return
    try:
        import torch

        dev_type = getattr(dev, "type", None)
        if dev_type == "cuda":
            torch.cuda.empty_cache()
        elif dev_type == "mps":
            torch.mps.empty_cache()
    except Exception:  # pragma: no cover - defensive
        pass


def _torch_lanczos_block(
    data_np: np.ndarray,
    rows_np: np.ndarray,
    cols_np: np.ndarray,
    *,
    a: int,
    mode: str,
    cval: float,
    dev: object | None,
) -> np.ndarray:
    """Resample one coordinate block with torch (or numpy fallback).

    numpy in, numpy out. Torch is an implementation detail.
    """
    n = int(rows_np.shape[0])
    if n == 0:
        return np.empty(0, dtype=data_np.dtype)

    if dev is None:
        # NumPy fallback path: reuse the verified reference chunk implementation.
        from faninsar.processing.resampling import _lanczos_resample_chunk

        return _lanczos_resample_chunk(
            data_np,
            rows_np,
            cols_np,
            a=a,
            mode=mode,
            cval=cval,
        )

    import torch

    h, w = data_np.shape
    is_complex = np.iscomplexobj(data_np)
    out_dtype = data_np.dtype

    # Torch does not have a native sinc-based Lanczos; build weights in NumPy
    # (small, float64) and move to device as float32 for gather + weighted sum.
    row_base = np.floor(rows_np).astype(np.int64)
    col_base = np.floor(cols_np).astype(np.int64)
    row_frac = rows_np - row_base
    col_frac = cols_np - col_base

    tap_offsets = np.arange(2 * a, dtype=np.int64) - (a - 1)
    row_taps = row_base[:, None] + tap_offsets[None, :]
    col_taps = col_base[:, None] + tap_offsets[None, :]

    if mode == "nearest":
        row_taps_clipped = np.clip(row_taps, 0, h - 1)
        col_taps_clipped = np.clip(col_taps, 0, w - 1)
        row_valid = None
        col_valid = None
    else:
        row_taps_clipped = np.clip(row_taps, 0, h - 1)
        col_taps_clipped = np.clip(col_taps, 0, w - 1)
        row_valid = (row_taps >= 0) & (row_taps <= h - 1)
        col_valid = (col_taps >= 0) & (col_taps <= w - 1)

    wr_np = lanczos_weights(row_frac, a=a)
    wc_np = lanczos_weights(col_frac, a=a)

    # Move source array and index buffers to device.
    if is_complex:
        data_t = torch.from_numpy(data_np).to(dev)
    else:
        data_t = torch.from_numpy(data_np).to(dev)

    row_taps_t = torch.from_numpy(row_taps_clipped).to(dev)
    col_taps_t = torch.from_numpy(col_taps_clipped).to(dev)
    wr_t = torch.from_numpy(wr_np.astype(np.float32)).to(dev)
    wc_t = torch.from_numpy(wc_np.astype(np.float32)).to(dev)

    # Gather (n, 2a, 2a). For chunked n this stays bounded.
    # Use advanced indexing: data_t[row_taps_t[:, :, None], col_taps_t[:, None, :]]
    if is_complex:
        patch = data_t[row_taps_t[:, :, None], col_taps_t[:, None, :]]
    else:
        patch = data_t[row_taps_t[:, :, None], col_taps_t[:, None, :]]

    if mode == "constant" and row_valid is not None and col_valid is not None:
        tap_mask_np = row_valid[:, :, None] & col_valid[:, None, :]
        if not np.all(tap_mask_np):
            tap_mask_t = torch.from_numpy(tap_mask_np).to(dev)
            if is_complex:
                fill = cval + 0j
                fill_t = torch.tensor(fill, dtype=patch.dtype, device=dev)
            else:
                fill_t = torch.tensor(cval, dtype=patch.dtype, device=dev)
            patch = torch.where(tap_mask_t, patch, fill_t)

    # Separable weighted sum on the gathered patch.
    # patch: (n, 2a, 2a). Real weights (n, 2a) broadcast over row taps.
    # Keep complex patch in complex64 so real weights multiply correctly
    # without losing the imaginary part.
    if is_complex:
        patch_f = patch  # complex64
    else:
        patch_f = patch.to(torch.float32)
    wc_t_b = wc_t.unsqueeze(1)  # (n, 1, 2a) real
    # patch_f * wc_t_b broadcasts real wc over complex patch → complex.
    col_reduced = patch_f * wc_t_b
    col_reduced = col_reduced.sum(dim=2)  # (n, 2a)
    # wr_t: (n, 2a) real; multiply broadcast → complex.
    out_t = (col_reduced * wr_t).sum(dim=1)  # (n,)

    out_np = out_t.detach().cpu().numpy().astype(out_dtype, copy=False)

    # Cleanup device tensors before next chunk.
    del data_t, row_taps_t, col_taps_t, wr_t, wc_t, patch, out_t
    _cleanup_gpu(dev)
    return out_np


def _lanczos_resample_device_persistent(
    data_np: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    a: int,
    mode: str,
    cval: float,
    dev: object,
    chunk_size: int,
    data_t: object | None = None,
) -> np.ndarray:
    """CUDA P7 path: source resident; per-chunk device prep via torch.sinc.

    Host only uploads ``data`` once (unless ``data_t`` is already on device)
    and ``rows``/``cols`` once. Each chunk builds taps and float32 Lanczos
    weights on device, gathers ``(chunk, 2a, 2a)``, and reduces — peak temps
    scale with ``chunk_size``, not full ``N``.

    Parameters
    ----------
    data_np : numpy.ndarray
        2-D source array (used for shape/dtype; also uploaded when ``data_t``
        is None).
    rows, cols : numpy.ndarray
        1-D fractional sample coordinates.
    a, mode, cval, dev, chunk_size
        Kernel and device controls (see :func:`lanczos_resample_dask_torch`).
    data_t : torch.Tensor or None, optional
        Pre-uploaded source tensor on ``dev``. When provided, this function
        does **not** re-upload ``data_np`` and does **not** free ``data_t``
        on exit — the caller owns residency across multiple coordinate tiles.
        Default None (upload once for this call, free on exit).

    """
    import torch

    n = int(rows.shape[0])
    h, w = data_np.shape
    is_complex = np.iscomplexobj(data_np)
    out_dtype = data_np.dtype
    chunk = max(int(chunk_size), 1)

    own_data = data_t is None
    if own_data:
        data_t = torch.from_numpy(np.ascontiguousarray(data_np)).to(
            dev, non_blocking=True
        )
    rows_t = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float64)).to(
        dev, non_blocking=True
    )
    cols_t = torch.from_numpy(np.ascontiguousarray(cols, dtype=np.float64)).to(
        dev, non_blocking=True
    )
    tap_offsets = torch.arange(2 * a, device=dev, dtype=torch.float64) - float(a - 1)
    tap_offsets_i = tap_offsets.to(dtype=torch.int64)
    a_f = float(a)

    out_t = torch.empty(n, dtype=data_t.dtype, device=dev)
    if is_complex:
        fill_t = torch.tensor(cval + 0j, dtype=data_t.dtype, device=dev)
    else:
        fill_t = torch.tensor(cval, dtype=data_t.dtype, device=dev)

    if own_data and getattr(dev, "type", None) == "cuda":
        torch.cuda.synchronize()

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rk = rows_t[s:e]
        ck = cols_t[s:e]
        row_base = torch.floor(rk)
        col_base = torch.floor(ck)
        row_frac = rk - row_base
        col_frac = ck - col_base

        row_taps = row_base[:, None].to(dtype=torch.int64) + tap_offsets_i[None, :]
        col_taps = col_base[:, None].to(dtype=torch.int64) + tap_offsets_i[None, :]
        row_taps_cl = torch.clamp(row_taps, 0, h - 1)
        col_taps_cl = torch.clamp(col_taps, 0, w - 1)

        x_r = row_frac[:, None] - tap_offsets[None, :]
        wr = torch.sinc(x_r) * torch.sinc(x_r / a_f)
        wr = (wr / wr.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(
            dtype=torch.float32
        )
        x_c = col_frac[:, None] - tap_offsets[None, :]
        wc = torch.sinc(x_c) * torch.sinc(x_c / a_f)
        wc = (wc / wc.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(
            dtype=torch.float32
        )

        patch = data_t[row_taps_cl[:, :, None], col_taps_cl[:, None, :]]
        if mode == "constant":
            row_ok = (row_taps >= 0) & (row_taps <= h - 1)
            col_ok = (col_taps >= 0) & (col_taps <= w - 1)
            tap_mask = row_ok[:, :, None] & col_ok[:, None, :]
            if not bool(torch.all(tap_mask).item()):
                patch = torch.where(tap_mask, patch, fill_t)

        col_red = (patch * wc.unsqueeze(1)).sum(dim=2)
        out_t[s:e] = (col_red * wr).sum(dim=1)
        del patch, col_red, wr, wc, row_taps, col_taps, row_taps_cl, col_taps_cl

    if getattr(dev, "type", None) == "cuda":
        torch.cuda.synchronize()
    out_np = out_t.detach().cpu().numpy().astype(out_dtype, copy=False)
    del rows_t, cols_t, out_t
    if own_data:
        del data_t
        _cleanup_gpu(dev)
    return out_np


def lanczos_resample_dask_torch(
    data: np.ndarray,
    coords: np.ndarray,
    *,
    a: int = 4,
    mode: str = "constant",
    cval: float = 0.0,
    chunk_size: int | None = DEFAULT_LANCZOS_CHUNK,
    device: DeviceName = "auto",
    use_dask: bool = False,
    dask_chunks: tuple[int, int] | None = None,
) -> np.ndarray:
    """Resample a 2-D array at fractional coords with Lanczos via dask-torch-numpy.

    Parameters
    ----------
    data : numpy.ndarray
        2-D source array (real or complex).
    coords : numpy.ndarray
        Output sample coordinates of shape ``(2, N)``.
    a : int, optional
        Lanczos half-width. Default 4.
    mode : str, optional
        Boundary handling. ``"constant"`` or ``"nearest"``. Default ``"constant"``.
    cval : float, optional
        Fill value for out-of-bounds taps. Default 0.0.
    chunk_size : int or None, optional
        Maximum output samples per coordinate chunk. Default :data:`DEFAULT_LANCZOS_CHUNK`.
    device : {"auto","cpu","cuda","mps"}, optional
        Torch device for the gather + weighted sum. ``"auto"`` picks cuda > mps > cpu.
    use_dask : bool, optional
        When True, schedule the chunk loop via ``dask.array.map_blocks`` so the
        resample can run on a Dask cluster / multi-threaded scheduler. When False,
        run a local serial loop over chunks (still using torch inside each chunk).
    dask_chunks : tuple of int, optional
        Output chunk shape for the Dask array. Defaults to ``(chunk_size,)``.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``N`` holding resampled values (dtype matches ``data``).

    Notes
    -----
    When ``device="cpu"`` and torch is not installed, this transparently falls
    back to the NumPy reference (:func:`_lanczos_resample_chunk`). This keeps the
    core path installable with numpy-only dependencies.

    """
    if data.ndim != 2:
        msg = "lanczos_resample_dask_torch requires a 2-D source array"
        raise ValueError(msg)
    if mode not in ("constant", "nearest"):
        msg = f"unsupported mode {mode!r}; use 'constant' or 'nearest'"
        raise ValueError(msg)

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[0] != 2:
        msg = "coords must have shape (2, N)"
        raise ValueError(msg)

    rows = np.asarray(coords[0], dtype=np.float64).reshape(-1)
    cols = np.asarray(coords[1], dtype=np.float64).reshape(-1)
    n = int(rows.shape[0])
    if n == 0:
        return np.empty(0, dtype=data.dtype)

    if chunk_size is None or chunk_size <= 0:
        chunk_size = max(n, 1)
    chunk_size = min(chunk_size, n)

    dev = _resolve_torch_device(device)
    if dev is not None:
        logger.info(
            "lanczos_resample_dask_torch device=%s chunk_size=%d n=%d",
            dev,
            chunk_size,
            n,
        )
    else:
        logger.info(
            "lanczos_resample_dask_torch numpy-fallback chunk_size=%d n=%d",
            chunk_size,
            n,
        )

    if use_dask:
        import dask

        global _DELAYED_DATA, _DELAYED_ROWS, _DELAYED_COLS, _DELAYED_OUT
        global _DELAYED_A, _DELAYED_MODE, _DELAYED_CVAL, _DELAYED_DEV

        ranges = [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]
        out = np.empty(n, dtype=data.dtype)
        _DELAYED_DATA = data
        _DELAYED_ROWS = rows
        _DELAYED_COLS = cols
        _DELAYED_OUT = out
        _DELAYED_A = int(a)
        _DELAYED_MODE = mode
        _DELAYED_CVAL = float(cval)
        _DELAYED_DEV = dev
        try:
            tasks = [dask.delayed(_delayed_write_range)(s, e) for s, e in ranges]
            n_workers = min(8, os.cpu_count() or 1)
            with dask.config.set(scheduler="threads", num_workers=n_workers):
                dask.compute(*tasks)
        finally:
            _DELAYED_DATA = None
            _DELAYED_ROWS = None
            _DELAYED_COLS = None
            _DELAYED_OUT = None
            _DELAYED_DEV = None
        return out

    # Local serial loop over chunks (torch inside each chunk).
    if dev is not None and not use_dask:
        # Source stays on device across all chunks (avoids per-chunk 245 MB copy).
        return _lanczos_resample_device_persistent(
            data,
            rows,
            cols,
            a=a,
            mode=mode,
            cval=cval,
            dev=dev,
            chunk_size=chunk_size,
        )
    out = np.empty(n, dtype=data.dtype)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        out[start:stop] = _torch_lanczos_block(
            data,
            rows[start:stop],
            cols[start:stop],
            a=a,
            mode=mode,
            cval=cval,
            dev=dev,
        )
    return out
