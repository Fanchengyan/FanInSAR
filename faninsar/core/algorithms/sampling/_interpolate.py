"""Public real-valued array-index interpolation with bounded Torch workspaces."""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Literal, TypeAlias, overload

from faninsar.core.algorithms._validation import (
    _as_tensor,
    _checked_size,
    _execution_device,
    _inspect_array,
    _require_torch,
)
from faninsar.logging import reject, setup_logger

from ._kernels import _QUERY_BATCH_SIZE, _query_chunk, _sample_chunk

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    _Array: TypeAlias = Tensor | np.ndarray

_Method: TypeAlias = Literal["nearest", "bilinear", "biquintic"]
_Boundary: TypeAlias = Literal["nan", "clamp"]
_MAX_INPUT_BYTES = 8 * 1024**3
_MAX_OUTPUT_BYTES = 1024**3
logger = setup_logger(__name__)


@overload
def interpolate2d(
    source: _Array,
    rows: _Array,
    cols: _Array,
    *,
    method: _Method = "bilinear",
    valid: _Array | None = None,
    boundary: _Boundary = "nan",
    return_valid: Literal[False] = False,
) -> Tensor: ...


@overload
def interpolate2d(
    source: _Array,
    rows: _Array,
    cols: _Array,
    *,
    method: _Method = "bilinear",
    valid: _Array | None = None,
    boundary: _Boundary = "nan",
    return_valid: Literal[True],
) -> tuple[Tensor, Tensor]: ...


@overload
def interpolate2d(
    source: _Array,
    rows: _Array,
    cols: _Array,
    *,
    method: _Method = "bilinear",
    valid: _Array | None = None,
    boundary: _Boundary = "nan",
    return_valid: bool,
) -> Tensor | tuple[Tensor, Tensor]: ...


def interpolate2d(
    source: _Array,
    rows: _Array,
    cols: _Array,
    *,
    method: _Method = "bilinear",
    valid: _Array | None = None,
    boundary: _Boundary = "nan",
    return_valid: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Interpolate real image arrays at shared array-index coordinates.

    Parameters
    ----------
    source : torch.Tensor or numpy.ndarray
        Float32/float64 data shaped ``(..., H, W)``. NumPy masked arrays
        retain their masks. Integer, complex and requires-grad inputs fail.
    rows, cols : torch.Tensor or numpy.ndarray
        Float32/float64 coordinates broadcasting to query shape ``Q``. Integer
        coordinates name sample centers directly, without half-pixel shifts.
        These queries are shared across every leading source image slice.
    method : {"nearest", "bilinear", "biquintic"}, default "bilinear"
        Nearest uses ``floor(u + 0.5)``. Bilinear requires every nonzero-weight
        tap. Biquintic is the six-sample natural spline, evaluated along columns
        then rows, requiring the complete ``floor(u)-1`` through ``floor(u)+4``
        stencil even at integer coordinates.
    valid : torch.Tensor or numpy.ndarray, optional
        Boolean additional source admission, shaped ``(H, W)`` or exactly like
        source. True never overrides native masks or nonfinite values. A native
        mask on this argument also excludes its corresponding observations.
    boundary : {"nan", "clamp"}, default "nan"
        Missing support produces NaN. Clamp first clips finite coordinates to
        ``[0, H-1]`` and ``[0, W-1]``; it is unavailable for biquintic.
    return_valid : bool, default False
        Return ``(values, valid_output)`` instead of values alone.

    Returns
    -------
    values : torch.Tensor
        Float64 values shaped ``(..., *Q)``, on the common input Tensor device,
        or CPU for NumPy-only input. Invalid results are NaN.
    valid_output : torch.Tensor
        Boolean admission with the same shape/device; returned only when
        ``return_valid=True``.

    Raises
    ------
    TypeError
        Inputs are not supported arrays or have unsupported dtypes/layouts.
    ValueError
        Shapes, source support, method, placement, gradients or allocation
        requirements violate the operation contract. Only CPU/CUDA is admitted;
        all existing Tensors must already share a device.

    Notes
    -----
    Nearest and bilinear permit singleton spatial axes when their strict active
    support exists. Spatial axes cannot be empty; biquintic requires both >=6.
    Native/nonfinite query coordinates are invalid before floor/index conversion.
    No inputs are mutated. Source finite masks and promoted 6x6 patches are
    gathered in bounded query chunks per image, not expanded over the raster.
    Metadata preflight limits each adapted input to 8 GiB and unavoidable output
    values plus optional validity to 1 GiB before transfer/allocation.

    """
    torch = _require_torch()
    device = _execution_device(source, rows, cols, valid)
    if method not in {"nearest", "bilinear", "biquintic"}:
        reject(
            f"Unsupported interpolation method {method!r}.", ValueError, logger=logger
        )
    if boundary not in {"nan", "clamp"}:
        reject(f"Unsupported boundary {boundary!r}.", ValueError, logger=logger)
    if boundary == "clamp" and method == "biquintic":
        reject(
            "Biquintic interpolation does not support clamp.", ValueError, logger=logger
        )
    if not isinstance(return_valid, bool):
        reject("return_valid must be Boolean.", TypeError, logger=logger)
    source_info = _inspect_array(source, name="source")
    row_info = _inspect_array(rows, name="rows")
    col_info = _inspect_array(cols, name="cols")
    if len(source_info.shape) < 2:
        reject("source requires at least two dimensions.", ValueError, logger=logger)
    height, width = source_info.shape[-2:]
    minimum = 6 if method == "biquintic" else 1
    if min(height, width) < minimum:
        reject(
            f"{method} requires source support of at least {minimum}x{minimum}.",
            ValueError,
            logger=logger,
        )
    try:
        query_shape = tuple(torch.broadcast_shapes(row_info.shape, col_info.shape))
    except RuntimeError as exc:
        reject(
            "rows and cols shapes must broadcast.", ValueError, logger=logger, cause=exc
        )
    inspections = [("source", source_info), ("rows", row_info), ("cols", col_info)]
    if valid is not None:
        valid_info = _inspect_array(valid, name="valid", allow_bool=True)
        if str(valid_info.dtype) not in {"bool", "torch.bool"}:
            reject("valid must have Boolean dtype.", TypeError, logger=logger)
        if valid_info.shape not in {source_info.shape[-2:], source_info.shape}:
            reject(
                "valid shape must be spatial or exactly source shape.",
                ValueError,
                logger=logger,
            )
        inspections.append(("valid", valid_info))
    for name, info in inspections:
        _checked_size(info.shape, itemsize=9, name=name, limit_bytes=_MAX_INPUT_BYTES)
    output_shape = (*source_info.shape[:-2], *query_shape)
    _checked_size(
        output_shape,
        itemsize=9 if return_valid else 8,
        name="interpolation output",
        limit_bytes=_MAX_OUTPUT_BYTES,
    )
    # Check the broadcast grid even if an empty image batch makes output empty.
    _checked_size(
        query_shape, itemsize=16, name="query size", limit_bytes=_MAX_INPUT_BYTES
    )
    with torch.inference_mode():
        output = torch.full(output_shape, torch.nan, dtype=torch.float64, device=device)
        output_valid = (
            torch.zeros(output_shape, dtype=torch.bool, device=device)
            if return_valid
            else None
        )
        if output.numel() == 0:
            return (output, output_valid) if return_valid else output
        values, source_invalid = _as_tensor(source, device=device, name="source")
        row, row_invalid = _as_tensor(rows, device=device, name="rows")
        col, col_invalid = _as_tensor(cols, device=device, name="cols")
        row, col = torch.broadcast_tensors(row, col)
        allowed, allowed_invalid = (
            (None, None)
            if valid is None
            else _as_tensor(valid, device=device, name="valid", allow_bool=True)
        )
        if allowed is not None:
            allowed = allowed.expand(source_info.shape)
        source_invalid = _expand_invalid(source_invalid, source_info.shape)
        allowed_invalid = _expand_invalid(allowed_invalid, source_info.shape)
        query_count = math.prod(query_shape)
        # Image slicing preserves noncontiguous source storage without reshape copies.
        for image in product(*(range(size) for size in source_info.shape[:-2])):
            image_output = output[image].reshape(-1)
            image_valid = (
                None if output_valid is None else output_valid[image].reshape(-1)
            )
            for start in range(0, query_count, _QUERY_BATCH_SIZE):
                stop = min(start + _QUERY_BATCH_SIZE, query_count)
                chunk_row, chunk_col, coordinate_valid = _query_chunk(
                    row, col, row_invalid, col_invalid, start, stop
                )
                sampled, admitted = _sample_chunk(
                    values[image],
                    _image_invalid(source_invalid, image),
                    None if allowed is None else allowed[image],
                    _image_invalid(allowed_invalid, image),
                    chunk_row,
                    chunk_col,
                    coordinate_valid,
                    method=method,
                    boundary=boundary,
                )
                image_output[start:stop] = torch.where(admitted, sampled, torch.nan)
                if image_valid is not None:
                    image_valid[start:stop] = admitted
    return (output, output_valid) if return_valid else output


def _expand_invalid(
    invalid: Tensor | bool | None, shape: tuple[int, ...]
) -> Tensor | bool | None:
    """Expand native mask metadata as a view, leaving scalar masks scalar."""
    return (
        invalid
        if invalid is None or isinstance(invalid, bool)
        else invalid.expand(shape)
    )


def _image_invalid(
    invalid: Tensor | bool | None, image: tuple[int, ...]
) -> Tensor | bool | None:
    """Select one leading image from a native mask without changing scalars."""
    return invalid if invalid is None or isinstance(invalid, bool) else invalid[image]
