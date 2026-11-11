"""Tiled separable normalized convolution with explicit boundary support."""

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
_TILE = 256
_MAX_RADIUS = 4096
_WORK_ITEMS = 4_194_304
_OUTPUT_BYTES = 8 * 1024**3


@overload
def filter2d(
    source: Array,
    *,
    method: Literal["boxcar", "gaussian"],
    window: tuple[int, int] | None = None,
    sigma: tuple[float, float] | None = None,
    min_radius: int | None = None,
    weights: Array | None = None,
    valid: Array | None = None,
    boundary: Literal["constant", "reflect"] = "constant",
    return_valid: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def filter2d(
    source: Array,
    *,
    method: Literal["boxcar", "gaussian"],
    window: tuple[int, int] | None = None,
    sigma: tuple[float, float] | None = None,
    min_radius: int | None = None,
    weights: Array | None = None,
    valid: Array | None = None,
    boundary: Literal["constant", "reflect"] = "constant",
    return_valid: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


def filter2d(
    source: Array,
    *,
    method: Literal["boxcar", "gaussian"],
    window: tuple[int, int] | None = None,
    sigma: tuple[float, float] | None = None,
    min_radius: int | None = None,
    weights: Array | None = None,
    valid: Array | None = None,
    boundary: Literal["constant", "reflect"] = "constant",
    return_valid: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Smooth using nonnegative normalized convolution on the final two axes.

    Parameters
    ----------
    source : numpy.ndarray, numpy.ma.MaskedArray, or torch.Tensor
        Real float32/float64 observations of shape ``(..., height, width)``.
    method : {'boxcar', 'gaussian'}
        Separable smoothing kernel.
    window : tuple of int, optional
        Boxcar-only positive odd spatial window lengths.
    sigma : tuple of float, optional
        Gaussian-only finite positive spatial standard deviations.
    min_radius : int, optional
        Gaussian-only nonnegative, non-Boolean minimum radius; None means one.
        Actual radii are ``max(min_radius, floor(3*sigma + 0.5))``.
    weights : array, optional
        Nonnegative finite weights, spatial or exact source shape. Native
        masked weights exclude observations, even with invalid hidden payload.
    valid : array of bool, optional
        Additional admission mask of spatial or exact source shape.
    boundary : {'constant', 'reflect'}, default 'constant'
        Zero outside support or no-edge-repeat reflection (SciPy ``mirror``).
    return_valid : bool, default False
        Also return the Boolean positive-support output mask.

    Returns
    -------
    values : torch.Tensor
        Float64 CPU/CUDA smoothed estimates; zero support yields NaN.
    valid_output : torch.Tensor
        Boolean positive-support mask when ``return_valid=True``.

    Raises
    ------
    ValueError
        If options, masks, weights, devices or derived support are invalid.
    TypeError
        If an input dtype is unsupported.

    Notes
    -----
    Invalid centers may be reconstructed from positive-weight neighbors.
    Singleton axes reflect safely for arbitrary admitted support. Tiles retain
    the true image halo, not artificial internal boundaries. Support is checked
    before tensor conversion: each radius is at most 4096, one padded working
    tile at most 4194304 samples, conservative whole-image convolution
    expansions at most 33554432 samples, and combined outputs at most 8 GiB.
    These are operation bounds, not a process-wide memory or device policy.

    """
    torch = _require_torch()
    info = _inspect_array(source, name="source")
    device = _execution_device(source, weights, valid)
    if len(info.shape) < 2:
        _fail("source must have at least two dimensions")
    if boundary not in ("constant", "reflect"):
        _fail("boundary must be constant or reflect")
    radii = _support(method, window, sigma, min_radius)
    ry, rx = radii
    height, width = info.shape[-2:]
    _checked_size(
        info.shape, itemsize=9, name="filter output", limit_bytes=_OUTPUT_BYTES
    )
    tile_y, tile_x = min(_TILE, max(1, height)), min(_TILE, max(1, width))
    while (
        max(
            2 * (2 * ry + 1) * tile_y * (tile_x + 2 * rx),
            2 * (2 * rx + 1) * tile_y * tile_x,
        )
        > _WORK_ITEMS
    ):
        if tile_y >= tile_x and tile_y > 1:
            tile_y = max(1, tile_y // 2)
        elif tile_x > 1:
            tile_x = max(1, tile_x // 2)
        else:
            _fail("derived convolution workspace exceeds the supported bound")
    _checked_size(
        (tile_y + 2 * ry, tile_x + 2 * rx),
        itemsize=1,
        name="filter padded workspace",
        limit_bytes=_WORK_ITEMS,
    )
    for value, name, boolean in ((weights, "weights", False), (valid, "valid", True)):
        if value is not None:
            companion = _inspect_array(value, name=name, allow_bool=boolean)
            if boolean and str(companion.dtype) not in ("bool", "torch.bool"):
                _fail("valid must have Boolean dtype")
            if companion.shape not in (info.shape[-2:], info.shape):
                _fail(f"{name} must have spatial shape or exactly source shape")
    data, native = _as_tensor(source, device=device, name="source")
    weight, weight_native = (
        (None, None)
        if weights is None
        else _as_tensor(weights, device=device, name="weights")
    )
    extra, extra_native = (
        (None, None)
        if valid is None
        else _as_tensor(valid, device=device, name="valid", allow_bool=True)
    )
    if isinstance(native, torch.Tensor):
        native = native.expand_as(data)
    if isinstance(weight_native, torch.Tensor):
        weight_native = weight_native.expand_as(weight)
    if isinstance(extra_native, torch.Tensor):
        extra_native = extra_native.expand_as(extra)
    if weight is not None:
        _check_weights(weight, weight_native)
    output = torch.full(info.shape, math.nan, dtype=torch.float64, device=device)
    admission = torch.zeros(info.shape, dtype=torch.bool, device=device)
    if output.numel() == 0:
        return (output, admission) if return_valid else output
    kernels = []
    for axis, radius in enumerate(radii):
        if method == "gaussian":
            coordinates = torch.arange(
                -radius, radius + 1, dtype=torch.float64, device=device
            )
            kernel = torch.exp(-0.5 * (coordinates / sigma[axis]).square())
            kernel = kernel / kernel.sum()
        else:
            kernel = torch.full(
                (2 * radius + 1,),
                1.0 / (2 * radius + 1),
                dtype=torch.float64,
                device=device,
            )
        kernels.append(kernel)
    ky, kx = kernels
    whole_items = math.prod(info.shape)
    if (
        boundary == "constant"
        and whole_items * max(ky.numel(), kx.numel()) <= 33_554_432
    ):
        values = data.to(torch.float64)
        allowed = torch.isfinite(values)
        for mask, invert in (
            (native, True),
            (extra, False),
            (extra_native, True),
            (weight_native, True),
        ):
            if mask is not None:
                chosen = (
                    (not mask if isinstance(mask, bool) else ~mask) if invert else mask
                )
                allowed &= chosen
        w = torch.ones_like(values) if weight is None else weight.to(torch.float64)
        w = torch.where(allowed, w, 0.0)
        numerator = torch.where(allowed, values, 0.0) * w
        return _normalized_convolution(numerator, w, ky, kx, return_valid=return_valid)
    for batch in itertools.product(*(range(size) for size in info.shape[:-2])):
        for row in range(0, height, tile_y):
            for col in range(0, width, tile_x):
                stop_y, stop_x = min(row + tile_y, height), min(col + tile_x, width)
                ys = torch.arange(row - ry, stop_y + ry, device=device)
                xs = torch.arange(col - rx, stop_x + rx, device=device)
                outside = (
                    (ys[:, None] >= 0)
                    & (ys[:, None] < height)
                    & (xs[None, :] >= 0)
                    & (xs[None, :] < width)
                )
                if boundary == "reflect":
                    ys = _mirror(ys, height)
                    xs = _mirror(xs, width)
                else:
                    ys = ys.clamp(0, height - 1)
                    xs = xs.clamp(0, width - 1)
                values = _gather(data, batch, ys, xs).to(torch.float64)
                allowed = torch.isfinite(values)
                for mask, invert in (
                    (native, True),
                    (extra, False),
                    (extra_native, True),
                    (weight_native, True),
                ):
                    if mask is not None:
                        chosen = _gather(mask, batch, ys, xs)
                        allowed &= (
                            (not chosen if isinstance(chosen, bool) else ~chosen)
                            if invert
                            else chosen
                        )
                if boundary == "constant":
                    allowed &= outside
                w = (
                    torch.ones_like(values)
                    if weight is None
                    else _gather(weight, batch, ys, xs).to(torch.float64)
                )
                w = torch.where(allowed, w, 0.0)
                numerator = torch.where(allowed, values, 0.0) * w
                filtered, supported = _normalized_convolution(
                    numerator, w, ky, kx, padded=False, return_valid=True
                )
                index = (*batch, slice(row, stop_y), slice(col, stop_x))
                output[index] = filtered
                admission[index] = supported
    return (output, admission) if return_valid else output


def _support(
    method: str,
    window: tuple[int, int] | None,
    sigma: tuple[float, float] | None,
    minimum: int | None,
) -> tuple[int, int]:
    if method == "gaussian":
        if (
            window is not None
            or not isinstance(sigma, (tuple, list))
            or len(sigma) != 2
        ):
            _fail("gaussian requires two positive finite sigmas and no window")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in sigma
        ):
            _fail("sigma must contain two positive finite non-Boolean numbers")
        if minimum is None:
            minimum = 1
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            _fail("min_radius must be a nonnegative non-Boolean integer or None")
        if minimum > _MAX_RADIUS or any(value > _MAX_RADIUS / 3 for value in sigma):
            _fail("derived filter radius exceeds the supported maximum of 4096")
        return tuple(max(minimum, int(3 * value + 0.5)) for value in sigma)
    if method != "boxcar" or sigma is not None or minimum is not None:
        _fail("boxcar requires a window, no sigma, and no explicit min_radius")
    if (
        not isinstance(window, (tuple, list))
        or len(window) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value % 2 == 0
            for value in window
        )
    ):
        _fail("window must contain two positive odd non-Boolean integers")
    radii = (window[0] // 2, window[1] // 2)
    if max(radii) > _MAX_RADIUS:
        _fail("derived filter radius exceeds the supported maximum of 4096")
    return radii


def _normalized_convolution(
    numerator: torch.Tensor,
    weight: torch.Tensor,
    ky: torch.Tensor,
    kx: torch.Tensor,
    *,
    padded: bool = True,
    return_valid: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute admitted same-device arithmetic, bounding convolution workspaces."""
    torch = _require_torch()

    def separable(image: torch.Tensor, kernel: torch.Tensor, axis: int) -> torch.Tensor:
        radius = (kernel.numel() - 1) // 2
        shape = image.shape
        moved = image.reshape(-1, 1, *shape[-2:])
        kernel4d = (
            kernel.reshape(1, 1, -1, 1) if axis == 0 else kernel.reshape(1, 1, 1, -1)
        )
        output_items = moved.numel()
        if not padded:
            axis_length = shape[-2 + axis]
            output_items = (
                output_items // axis_length * (axis_length - kernel.numel() + 1)
            )
        if output_items * kernel.numel() <= 33_554_432:
            if padded:
                pads = (0, 0, radius, radius) if axis == 0 else (radius, radius, 0, 0)
                moved = torch.nn.functional.pad(moved, pads)
            result = torch.nn.functional.conv2d(moved, kernel4d)
            return result.reshape(*shape[:-2], *result.shape[-2:])
        # Only large admitted images take the bounded stripe path. A stripe
        # includes its true halo; internal stripe boundaries are never padded.
        result = torch.empty_like(image)
        length = shape[-2 + axis]
        orthogonal = shape[-1 if axis == 0 else -2]
        step = max(1, 33_554_432 // (max(1, orthogonal) * kernel.numel()))
        for start in range(0, length, step):
            stop = min(length, start + step)
            left, right = max(0, start - radius), min(length, stop + radius)
            index = [slice(None)] * image.ndim
            index[-2 + axis] = slice(left, right)
            stripe = image[tuple(index)].reshape(-1, 1, *image[tuple(index)].shape[-2:])
            pad = (
                (0, 0, max(0, radius - start), max(0, stop + radius - length))
                if axis == 0
                else (max(0, radius - start), max(0, stop + radius - length), 0, 0)
            )
            stripe = torch.nn.functional.pad(stripe, pad)
            filtered = torch.nn.functional.conv2d(stripe, kernel4d)
            index[-2 + axis] = slice(start, stop)
            result[tuple(index)] = filtered.reshape(*shape[:-2], *filtered.shape[-2:])
        return result

    num = separable(separable(numerator, ky, 0), kx, 1)
    den = separable(separable(weight, ky, 0), kx, 1)
    supported = den > 0
    output = torch.where(supported, num / den, math.nan)
    return (output, supported) if return_valid else output


def _mirror(indices: torch.Tensor, length: int) -> torch.Tensor:
    torch = _require_torch()
    if length == 1:
        return torch.zeros_like(indices)
    period = 2 * (length - 1)
    folded = indices.remainder(period)
    return torch.minimum(folded, period - folded)


def _gather(
    value: torch.Tensor | bool,
    batch: tuple[int, ...],
    ys: torch.Tensor,
    xs: torch.Tensor,
) -> torch.Tensor | bool:
    if isinstance(value, bool):
        return value
    image = value[batch] if value.ndim > 2 else value
    return image[ys[:, None], xs[None, :]]


def _check_weights(weight: torch.Tensor, native: torch.Tensor | bool | None) -> None:
    torch = _require_torch()
    for batch in itertools.product(*(range(size) for size in weight.shape[:-2])):
        for row in range(0, weight.shape[-2], _TILE):
            for col in range(0, weight.shape[-1], _TILE):
                index = (*batch, slice(row, row + _TILE), slice(col, col + _TILE))
                values = weight[index]
                bad = (~torch.isfinite(values)) | (values < 0)
                if native is not None:
                    mask = native[index] if not isinstance(native, bool) else native
                    bad &= not mask if isinstance(mask, bool) else ~mask
                if bool(bad.any()):
                    _fail("unmasked weights must be finite and nonnegative")


def _fail(message: str) -> None:
    logger.error(message)
    raise ValueError(message)
