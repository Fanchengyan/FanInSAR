"""Phase-preserving resampling kernels for complex SAR data.

Complex SLCs and wrapped interferograms are sampled bandlimited signals.
Reconstructing them at fractional coordinates requires a sinc-family kernel;
bilinear (triangular, ``sinc²`` response) attenuates in-band signal and leaks
residual aliasing, producing a sub-pixel-offset-dependent *phase bias* that is
invisible in amplitude but catastrophic downstream (burst seams, decorrelation,
cm-level phantom deformation). See the ``sar-resampling-kernels`` skill for the
full evidence chain.

This module provides a small, dependency-light Lanczos resampler that operates
on plain ``numpy`` arrays (real or complex) at arbitrary fractional output
coordinates, matching the interface used by
:func:`scipy.ndimage.map_coordinates` for the resampling paths in the
processing layer.

Full-burst S1 SLCs are ~30M samples. Materialising a single ``(N, 2a, 2a)``
gather patch for all samples at once requires tens of GB and will OOM a
typical workstation. :func:`lanczos_resample` therefore processes coordinates
in bounded chunks while preserving bit-identical results relative to a
monolithic gather (within float rounding of the separable sum).
"""

from __future__ import annotations

import threading

import numpy as np

__all__ = ["DEFAULT_LANCZOS_CHUNK", "lanczos_resample", "lanczos_weights"]

# ~0.5M output samples → peak patch temps ≈ 0.5e6 * 64 * 8 B ≈ 256 MB for a=4
DEFAULT_LANCZOS_CHUNK = 512 * 1024

_TLS = threading.local()


def _tls_workbufs(
    n: int,
    a: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return thread-local (row_taps, col_taps, wr, wc) views of length ``n``."""
    taps = 2 * a
    state = getattr(_TLS, "lanczos", None)
    need = (
        state is None
        or state["cap"] < n
        or state["a"] != a
    )
    if need:
        cap = max(n, 64 * 1024)
        _TLS.lanczos = {
            "cap": cap,
            "a": a,
            "row_taps": np.empty((cap, taps), dtype=np.int64),
            "col_taps": np.empty((cap, taps), dtype=np.int64),
            "wr": np.empty((cap, taps), dtype=np.float64),
            "wc": np.empty((cap, taps), dtype=np.float64),
        }
        state = _TLS.lanczos
    return (
        state["row_taps"][:n],
        state["col_taps"][:n],
        state["wr"][:n],
        state["wc"][:n],
    )


def lanczos_weights(
    frac: np.ndarray,
    a: int = 4,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return Lanczos-``a`` interpolation weights for a 1-D offset.

    Parameters
    ----------
    frac : numpy.ndarray
        Fractional offset in ``[0, 1)`` of the target position relative to the
        nearest integer sample to its left. Any real array shape is accepted.
    a : int, optional
        Lanczos half-width (``a=4`` gives an 8-tap kernel). Default 4.
    out : numpy.ndarray, optional
        Optional preallocated output of shape ``frac.shape + (2 * a,)``.

    Returns
    -------
    numpy.ndarray
        Weights of shape ``frac.shape + (2 * a,)``. The tap at index ``k``
        corresponds to the source sample at relative position ``k - a + 1``.

    """
    frac = np.asarray(frac, dtype=np.float64)
    taps = np.arange(2 * a, dtype=np.float64) - (a - 1)
    x = frac[..., None] - taps
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc_x = np.sinc(x)
        sinc_xa = np.sinc(x / a)
    weights = sinc_x * sinc_xa
    total = weights.sum(axis=-1, keepdims=True)
    total = np.where(total == 0.0, 1.0, total)
    weights = weights / total
    if out is not None:
        np.copyto(out, weights)
        return out
    return weights


def _lanczos_resample_chunk(
    data: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    a: int,
    mode: str,
    cval: float,
) -> np.ndarray:
    """Resample one coordinate chunk (peak memory O(chunk * a²))."""
    h, w = data.shape
    n = int(rows.shape[0])
    if n == 0:
        return np.empty(0, dtype=data.dtype)

    row_base = np.floor(rows).astype(np.int64, copy=False)
    col_base = np.floor(cols).astype(np.int64, copy=False)
    row_frac = rows - row_base
    col_frac = cols - col_base

    tap_offsets = np.arange(2 * a, dtype=np.int64) - (a - 1)
    row_taps_buf, col_taps_buf, wr_buf, wc_buf = _tls_workbufs(n, a)
    np.add(row_base[:, None], tap_offsets[None, :], out=row_taps_buf)
    np.add(col_base[:, None], tap_offsets[None, :], out=col_taps_buf)
    row_taps = row_taps_buf
    col_taps = col_taps_buf

    row_taps_clipped = np.clip(row_taps, 0, h - 1)
    col_taps_clipped = np.clip(col_taps, 0, w - 1)
    if mode == "constant":
        row_valid = (row_taps >= 0) & (row_taps <= h - 1)
        col_valid = (col_taps >= 0) & (col_taps <= w - 1)
    else:
        row_valid = None
        col_valid = None

    wr = lanczos_weights(row_frac, a=a, out=wr_buf)
    wc = lanczos_weights(col_frac, a=a, out=wc_buf)

    patch = data[row_taps_clipped[:, :, None], col_taps_clipped[:, None, :]]

    if mode == "constant" and row_valid is not None and col_valid is not None:
        tap_mask = row_valid[:, :, None] & col_valid[:, None, :]
        if not np.all(tap_mask):
            fill = np.asarray(cval, dtype=patch.dtype)
            patch = np.where(tap_mask, patch, fill)

    col_reduced = (patch * wc[:, None, :]).sum(axis=2)
    out = (col_reduced * wr).sum(axis=1)
    return out.astype(data.dtype, copy=False)


def lanczos_resample(
    data: np.ndarray,
    coords: np.ndarray,
    *,
    a: int = 4,
    mode: str = "constant",
    cval: float = 0.0,
    chunk_size: int | None = DEFAULT_LANCZOS_CHUNK,
) -> np.ndarray:
    """Resample a 2-D array at fractional coordinates with a Lanczos kernel.

    This is a phase-preserving replacement for
    ``scipy.ndimage.map_coordinates(data, coords, order=1)`` when ``data`` is a
    complex SLC or interferogram. For complex input the real and imaginary
    parts are convolved with the *same* kernel, which is equivalent to
    convolving the complex field directly and preserves phase statistics.

    Large coordinate arrays are processed in chunks of ``chunk_size`` samples
    so peak temporary memory stays O(``chunk_size * a²``) rather than
    O(``N * a²``). Results match a monolithic gather within floating-point
    rounding of the separable weighted sum.

    Parameters
    ----------
    data : numpy.ndarray
        2-D source array, real or complex.
    coords : numpy.ndarray
        Output sample coordinates of shape ``(2, N)``: ``coords[0]`` are row
        (azimuth) indices and ``coords[1]`` are column (range) indices, as in
        :func:`scipy.ndimage.map_coordinates`.
    a : int, optional
        Lanczos half-width. ``a=4`` (8-tap) is the production default for SLC
        resampling; ``a=6`` (12-tap) for highest-precision demands. Default 4.
    mode : str, optional
        Boundary handling. Only ``"constant"`` and ``"nearest"`` are
        implemented; ``"constant"`` fills out-of-bounds taps with ``cval``.
        Default ``"constant"``.
    cval : float, optional
        Fill value for out-of-bounds taps when ``mode="constant"``. Default 0.0.
    chunk_size : int or None, optional
        Maximum number of output samples processed per chunk. ``None`` or a
        non-positive value disables chunking (useful for tiny arrays / tests).
        Default :data:`DEFAULT_LANCZOS_CHUNK`.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``N`` holding the resampled values. The dtype
        matches ``data`` (complex inputs stay complex).

    """
    if data.ndim != 2:
        msg = "lanczos_resample requires a 2-D source array"
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

    if chunk_size is None or chunk_size <= 0 or n <= chunk_size:
        return _lanczos_resample_chunk(
            data, rows, cols, a=a, mode=mode, cval=cval
        )

    out = np.empty(n, dtype=data.dtype)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        out[start:stop] = _lanczos_resample_chunk(
            data,
            rows[start:stop],
            cols[start:stop],
            a=a,
            mode=mode,
            cval=cval,
        )
    return out
