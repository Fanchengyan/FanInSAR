"""Bounded real-valued spatial block reduction."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Literal, overload

from faninsar.core.algorithms._validation import (
    _as_tensor,
    _checked_size,
    _execution_device,
    _inspect_array,
    _require_torch,
)
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    import numpy as np
    import torch

    Array = np.ndarray | torch.Tensor

logger = setup_logger(__name__)
_WORK_ITEMS = 262_144
_OUTPUT_BYTES = 8 * 1024**3


@overload
def block_reduce2d(
    source: Array,
    block_shape: tuple[int, int],
    *,
    method: Literal["mean", "sum", "min", "max"] = "mean",
    valid: Array | None = None,
    invalid: Literal["propagate", "omit"] = "propagate",
    remainder: Literal["drop", "partial", "error"] = "drop",
    return_valid: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def block_reduce2d(
    source: Array,
    block_shape: tuple[int, int],
    *,
    method: Literal["mean", "sum", "min", "max"] = "mean",
    valid: Array | None = None,
    invalid: Literal["propagate", "omit"] = "propagate",
    remainder: Literal["drop", "partial", "error"] = "drop",
    return_valid: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


def block_reduce2d(
    source: Array,
    block_shape: tuple[int, int],
    *,
    method: Literal["mean", "sum", "min", "max"] = "mean",
    valid: Array | None = None,
    invalid: Literal["propagate", "omit"] = "propagate",
    remainder: Literal["drop", "partial", "error"] = "drop",
    return_valid: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reduce origin-aligned blocks independently on the final two axes.

    Parameters
    ----------
    source : numpy.ndarray, numpy.ma.MaskedArray, or torch.Tensor
        Real float32/float64 observations shaped ``(..., height, width)``.
    block_shape : tuple of int
        Positive block height and width.
    method : {'mean', 'sum', 'min', 'max'}, default 'mean'
        Reduction of the admitted block members.
    valid : array of bool, optional
        Additional admission mask with spatial or exact source shape.
    invalid : {'propagate', 'omit'}, default 'propagate'
        Reject a block with any unavailable member, or omit those members.
    remainder : {'drop', 'partial', 'error'}, default 'drop'
        Discard tails, reduce available tail members, or reject tails.
    return_valid : bool, default False
        Also return the Boolean output admission mask.

    Returns
    -------
    values : torch.Tensor
        Float64 CPU/CUDA reductions; empty effective blocks are NaN.
    valid_output : torch.Tensor
        Boolean mask, returned only when ``return_valid=True``.

    Raises
    ------
    ValueError
        If shapes, options, devices or allocation sizes are unsupported.
    TypeError
        If input dtypes or block dimensions are unsupported.

    Notes
    -----
    Blocks have no physical look origin. Spatial tails contain only actual
    observations, not padded members. Working tiles are bounded independently
    of block size; outputs above 8 GiB are rejected before conversion.

    """
    torch = _require_torch()
    info = _inspect_array(source, name="source")
    device = _execution_device(source, valid)
    if len(info.shape) < 2:
        _fail("source must have at least two dimensions")
    if (
        not isinstance(block_shape, (tuple, list))
        or len(block_shape) != 2
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in block_shape
        )
    ):
        _fail("block_shape must contain two positive non-Boolean integers")
    if method not in ("mean", "sum", "min", "max"):
        _fail("method must be mean, sum, min, or max")
    if invalid not in ("propagate", "omit"):
        _fail("invalid must be propagate or omit")
    if remainder not in ("drop", "partial", "error"):
        _fail("remainder must be drop, partial, or error")
    by, bx = block_shape
    height, width = info.shape[-2:]
    if remainder == "error" and (height % by or width % bx):
        _fail("source spatial shape is not divisible by block_shape")
    rows = (height + by - 1) // by if remainder == "partial" else height // by
    cols = (width + bx - 1) // bx if remainder == "partial" else width // bx
    output_shape = (*info.shape[:-2], rows, cols)
    _checked_size(
        output_shape, itemsize=9, name="block output", limit_bytes=_OUTPUT_BYTES
    )
    if valid is not None:
        validity = _inspect_array(valid, name="valid", allow_bool=True)
        if str(validity.dtype) not in ("bool", "torch.bool"):
            _fail("valid must have Boolean dtype")
        if validity.shape not in (info.shape[-2:], info.shape):
            _fail("valid must have spatial shape or exactly source shape")
    output = torch.full(output_shape, math.nan, dtype=torch.float64, device=device)
    admission = torch.zeros(output_shape, dtype=torch.bool, device=device)
    if output.numel() == 0:
        return (output, admission) if return_valid else output
    data, native = _as_tensor(source, device=device, name="source")
    extra, extra_native = (
        (None, None)
        if valid is None
        else _as_tensor(valid, device=device, name="valid", allow_bool=True)
    )
    if isinstance(native, torch.Tensor):
        native = native.expand_as(data)
    if isinstance(extra_native, torch.Tensor):
        extra_native = extra_native.expand_as(extra)
    for batch in itertools.product(*(range(size) for size in info.shape[:-2])):
        image = data[batch]
        image_native = (
            native[batch] if hasattr(native, "ndim") and native.ndim > 2 else native
        )
        image_extra = extra[batch] if extra is not None and extra.ndim > 2 else extra
        image_extra_native = (
            extra_native[batch]
            if hasattr(extra_native, "ndim") and extra_native.ndim > 2
            else extra_native
        )
        full_rows, full_cols = height // by, width // bx
        vectorized = by * bx <= _WORK_ITEMS and full_rows > 0 and full_cols > 0
        if vectorized:
            tile_cols = min(full_cols, max(1, _WORK_ITEMS // (by * bx)))
            tile_rows = min(full_rows, max(1, _WORK_ITEMS // (by * bx * tile_cols)))
            for row in range(0, full_rows, tile_rows):
                stop_row = min(full_rows, row + tile_rows)
                for col in range(0, full_cols, tile_cols):
                    stop_col = min(full_cols, col + tile_cols)
                    index = (
                        slice(row * by, stop_row * by),
                        slice(col * bx, stop_col * bx),
                    )
                    values = image[index].to(torch.float64)
                    allowed = torch.isfinite(values)
                    for mask, invert in (
                        (image_native, True),
                        (image_extra, False),
                        (image_extra_native, True),
                    ):
                        if mask is not None:
                            chosen = mask[index] if hasattr(mask, "ndim") else mask
                            if invert:
                                chosen = (
                                    not chosen if isinstance(chosen, bool) else ~chosen
                                )
                            allowed &= chosen
                    shape = (stop_row - row, by, stop_col - col, bx)
                    grouped = values.reshape(shape)
                    selected = allowed.reshape(shape)
                    count = selected.sum(dim=(1, 3))
                    if method == "mean" and invalid == "propagate":
                        reduced = _complete_block_mean(values, by, bx)
                    elif method == "min":
                        reduced = torch.where(selected, grouped, math.inf).amin(
                            dim=(1, 3)
                        )
                    elif method == "max":
                        reduced = torch.where(selected, grouped, -math.inf).amax(
                            dim=(1, 3)
                        )
                    else:
                        reduced = torch.where(selected, grouped, 0.0).sum(dim=(1, 3))
                        if method == "mean":
                            reduced = reduced / count
                    supported = (count > 0) & (
                        (count == by * bx) if invalid == "propagate" else True
                    )
                    target = (*batch, slice(row, stop_row), slice(col, stop_col))
                    output[target] = torch.where(supported, reduced, math.nan)
                    admission[target] = supported
        if vectorized and rows == full_rows and cols == full_cols:
            continue
        for row in range(rows):
            first_col = full_cols if vectorized and row < full_rows else 0
            for col in range(first_col, cols):
                y0, x0 = row * by, col * bx
                y1, x1 = min(y0 + by, height), min(x0 + bx, width)
                initial = (
                    math.inf
                    if method == "min"
                    else -math.inf
                    if method == "max"
                    else 0.0
                )
                total = torch.full((), initial, dtype=torch.float64, device=device)
                count = torch.zeros((), dtype=torch.int64, device=device)
                for start in range(y0, y1, max(1, _WORK_ITEMS // max(1, x1 - x0))):
                    stop = min(y1, start + max(1, _WORK_ITEMS // max(1, x1 - x0)))
                    for left in range(x0, x1, _WORK_ITEMS):
                        index = (
                            slice(start, stop),
                            slice(left, min(left + _WORK_ITEMS, x1)),
                        )
                        values = image[index].to(torch.float64)
                        allowed = torch.isfinite(values)
                        for mask, invert in (
                            (image_native, True),
                            (image_extra, False),
                            (image_extra_native, True),
                        ):
                            if mask is not None:
                                chosen = mask[index] if hasattr(mask, "ndim") else mask
                                allowed &= (
                                    (
                                        not chosen
                                        if isinstance(chosen, bool)
                                        else ~chosen
                                    )
                                    if invert
                                    else chosen
                                )
                        if method == "min":
                            total = torch.minimum(
                                total, torch.where(allowed, values, math.inf).amin()
                            )
                        elif method == "max":
                            total = torch.maximum(
                                total, torch.where(allowed, values, -math.inf).amax()
                            )
                        else:
                            total += torch.where(allowed, values, 0.0).sum()
                        count += allowed.sum()
                supported = (count > 0) & (
                    (count == (y1 - y0) * (x1 - x0)) if invalid == "propagate" else True
                )
                value = total / count if method == "mean" else total
                output[(*batch, row, col)] = torch.where(supported, value, math.nan)
                admission[(*batch, row, col)] = supported
    return (output, admission) if return_valid else output


def _complete_block_mean(source: torch.Tensor, by: int, bx: int) -> torch.Tensor:
    """Reduce admitted complete blocks, retaining ordinary IEEE mean semantics."""
    height, width = source.shape[-2:]
    rows, cols = height // by, width // bx
    if source.numel() <= 4_194_304 and height % by == 0 and width % bx == 0:
        return source.reshape(*source.shape[:-2], rows, by, cols, bx).mean(dim=(-3, -1))
    torch = _require_torch()
    cropped = source[..., : rows * by, : cols * bx]
    shape = (*source.shape[:-2], rows, by, cols, bx)
    if source.numel() <= 4_194_304 or not source.is_contiguous():
        return cropped.reshape(shape).mean(dim=(-3, -1))
    output = torch.empty(
        (*source.shape[:-2], rows, cols), dtype=source.dtype, device=source.device
    )
    step = max(1, 4_194_304 // max(1, width * by))
    for row in range(0, rows, step):
        stop = min(rows, row + step)
        tile = source[..., row * by : stop * by, : cols * bx]
        output[..., row:stop, :] = tile.reshape(
            *source.shape[:-2], stop - row, by, cols, bx
        ).mean(dim=(-3, -1))
    return output


def _fail(message: str) -> None:
    logger.error(message)
    raise ValueError(message)
