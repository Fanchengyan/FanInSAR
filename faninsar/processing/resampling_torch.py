"""Torch Lanczos resampling shared by CPU, CUDA, and MPS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.resampling import (
    DEFAULT_LANCZOS_CHUNK,
    DeviceName,
    LanczosBoundaryMode,
)

if TYPE_CHECKING:
    import torch

logger = setup_logger(__name__)

__all__ = ["lanczos_resample_torch"]


def _resolve_torch_device(device: DeviceName) -> torch.device:
    """Resolve an available Torch execution device via ``parse_device``."""
    try:
        from faninsar.processing.runtime.device import parse_device
    except ImportError as error:
        message = "Lanczos resampling requires torch; install FanInSAR dependencies"
        logger.exception(message)
        raise ImportError(message) from error
    return parse_device(device)


def _cleanup_device(device: torch.device) -> None:
    """Drop no allocator slabs from a resampling kernel (PROPOSAL-0034).

    Callers still ``del`` tensor references. Reclaim is orchestrated by
    :func:`faninsar.processing.runtime.device.reclaim_checkpoint`.
    """
    del device


def _lanczos_resample_device_persistent(
    data: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    a: int,
    mode: LanczosBoundaryMode,
    cval: float,
    device: torch.device,
    chunk_size: int,
    source_tensor: torch.Tensor | None = None,
) -> np.ndarray:
    """Run the bounded Lanczos kernel while keeping the source device-resident."""
    import torch

    sample_count = int(rows.size)
    if sample_count == 0:
        return np.empty(0, dtype=data.dtype)

    height, width = data.shape
    owns_source = source_tensor is None
    if owns_source:
        source_tensor = torch.from_numpy(np.ascontiguousarray(data)).to(
            device,
            non_blocking=device.type == "cuda",
        )
    assert source_tensor is not None

    coordinate_dtype = torch.float32 if device.type == "mps" else torch.float64
    numpy_coordinate_dtype = np.float32 if device.type == "mps" else np.float64
    rows_tensor = torch.from_numpy(
        np.ascontiguousarray(rows, dtype=numpy_coordinate_dtype)
    ).to(device, non_blocking=device.type == "cuda")
    columns_tensor = torch.from_numpy(
        np.ascontiguousarray(cols, dtype=numpy_coordinate_dtype)
    ).to(device, non_blocking=device.type == "cuda")
    tap_offsets = torch.arange(
        2 * a,
        dtype=coordinate_dtype,
        device=device,
    ) - float(a - 1)
    integer_tap_offsets = tap_offsets.to(dtype=torch.int64)
    half_width = float(a)
    output_tensor = torch.empty(
        sample_count,
        dtype=source_tensor.dtype,
        device=device,
    )
    fill_value = cval + 0j if np.iscomplexobj(data) else cval
    fill_tensor = torch.tensor(
        fill_value,
        dtype=source_tensor.dtype,
        device=device,
    )

    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        row_coordinates = rows_tensor[start:stop]
        column_coordinates = columns_tensor[start:stop]
        row_base = torch.floor(row_coordinates)
        column_base = torch.floor(column_coordinates)
        row_taps = row_base[:, None].to(torch.int64) + integer_tap_offsets[None, :]
        column_taps = (
            column_base[:, None].to(torch.int64) + integer_tap_offsets[None, :]
        )
        clipped_rows = torch.clamp(row_taps, 0, height - 1)
        clipped_columns = torch.clamp(column_taps, 0, width - 1)

        row_fraction = row_coordinates - row_base
        column_fraction = column_coordinates - column_base
        row_distance = row_fraction[:, None] - tap_offsets[None, :]
        column_distance = column_fraction[:, None] - tap_offsets[None, :]
        row_weights = torch.sinc(row_distance) * torch.sinc(
            row_distance / half_width
        )
        column_weights = torch.sinc(column_distance) * torch.sinc(
            column_distance / half_width
        )
        row_weights = (
            row_weights
            / row_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        ).to(torch.float32)
        column_weights = (
            column_weights
            / column_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        ).to(torch.float32)

        patch = source_tensor[
            clipped_rows[:, :, None],
            clipped_columns[:, None, :],
        ]
        if mode == "constant":
            valid_rows = (row_taps >= 0) & (row_taps < height)
            valid_columns = (column_taps >= 0) & (column_taps < width)
            valid_taps = valid_rows[:, :, None] & valid_columns[:, None, :]
            if not bool(torch.all(valid_taps).item()):
                patch = torch.where(valid_taps, patch, fill_tensor)
        column_reduced = (patch * column_weights[:, None, :]).sum(dim=2)
        output_tensor[start:stop] = (column_reduced * row_weights).sum(dim=1)
        del (
            patch,
            column_reduced,
            row_weights,
            column_weights,
            row_taps,
            column_taps,
            clipped_rows,
            clipped_columns,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    output = output_tensor.detach().cpu().numpy().astype(data.dtype, copy=False)
    del rows_tensor, columns_tensor, output_tensor
    if owns_source:
        del source_tensor
        _cleanup_device(device)
    return output


def lanczos_resample_torch(
    data: np.ndarray,
    coords: np.ndarray,
    *,
    a: int = 4,
    mode: LanczosBoundaryMode = "constant",
    cval: float = 0.0,
    chunk_size: int | None = DEFAULT_LANCZOS_CHUNK,
    device: DeviceName = "auto",
) -> np.ndarray:
    """Resample fractional coordinates with one Torch CPU/GPU implementation.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional real or complex source array.
    coords : numpy.ndarray
        Fractional coordinates with shape ``(2, N)``.
    a : int, optional
        Lanczos half-width.
    mode : {"constant", "nearest"}, optional
        Boundary treatment.
    cval : float, optional
        Constant boundary value.
    chunk_size : int or None, optional
        Maximum output samples per device chunk.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch device.

    Returns
    -------
    numpy.ndarray
        One-dimensional resampled values with the source dtype.

    """
    source = np.asarray(data)
    coordinates = np.asarray(coords)
    if source.ndim != 2:
        message = "lanczos_resample_torch requires a two-dimensional source"
        logger.error(message)
        raise ValueError(message)
    if coordinates.ndim != 2 or coordinates.shape[0] != 2:
        message = "coords must have shape (2, N)"
        logger.error(message)
        raise ValueError(message)
    if mode not in {"constant", "nearest"}:
        message = f"unsupported boundary mode: {mode!r}"
        logger.error(message)
        raise ValueError(message)
    if a < 1:
        message = "Lanczos half-width must be positive"
        logger.error(message)
        raise ValueError(message)

    rows = np.asarray(coordinates[0], dtype=np.float64).reshape(-1)
    columns = np.asarray(coordinates[1], dtype=np.float64).reshape(-1)
    bounded_chunk_size = (
        max(rows.size, 1)
        if chunk_size is None or chunk_size <= 0
        else min(int(chunk_size), max(rows.size, 1))
    )
    resolved_device = _resolve_torch_device(device)
    logger.info(
        "Lanczos device=%s samples=%d chunk_size=%d",
        resolved_device,
        rows.size,
        bounded_chunk_size,
    )
    return _lanczos_resample_device_persistent(
        source,
        rows,
        columns,
        a=a,
        mode=mode,
        cval=cval,
        device=resolved_device,
        chunk_size=bounded_chunk_size,
    )
