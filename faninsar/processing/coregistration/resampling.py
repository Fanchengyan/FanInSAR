"""Phase-preserving Lanczos resampling for complex SAR data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "DEFAULT_LANCZOS_CHUNK",
    "DeviceName",
    "LanczosBoundaryMode",
    "lanczos_resample",
]

DEFAULT_LANCZOS_CHUNK = 128 * 1024
DeviceName: TypeAlias = Literal["auto", "cpu", "cuda", "mps"]
LanczosBoundaryMode: TypeAlias = Literal["constant", "nearest"]


def lanczos_resample(
    data: np.ndarray,
    coords: np.ndarray,
    *,
    a: int = 4,
    mode: LanczosBoundaryMode = "constant",
    cval: float = 0.0,
    chunk_size: int | None = DEFAULT_LANCZOS_CHUNK,
    device: DeviceName = "auto",
) -> np.ndarray:
    """Resample a two-dimensional array with the unified Torch Lanczos kernel.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional real or complex source array.
    coords : numpy.ndarray
        Fractional output coordinates with shape ``(2, N)``.
    a : int, optional
        Lanczos half-width. The default produces an eight-tap kernel.
    mode : {"constant", "nearest"}, optional
        Boundary treatment.
    cval : float, optional
        Constant boundary value.
    chunk_size : int or None, optional
        Maximum number of output samples per device chunk.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch execution device. ``"auto"`` selects CUDA, then MPS, then CPU.

    Returns
    -------
    numpy.ndarray
        One-dimensional resampled values with the source dtype.

    """
    from faninsar.processing.coregistration.resampling_torch import (
        lanczos_resample_torch,
    )

    return lanczos_resample_torch(
        data,
        coords,
        a=a,
        mode=mode,
        cval=cval,
        chunk_size=chunk_size,
        device=device,
    )
