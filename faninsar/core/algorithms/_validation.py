"""Private metadata admission and container adaptation for numerical operations.

Callers inspect all inputs, choose their execution device, and preflight derived
shapes/workspaces before calling ``_as_tensor``. This module never constructs a
finite-value mask or promotes an entire float32 source to float64.
"""

from __future__ import annotations

import sys
from operator import index
from typing import TYPE_CHECKING, NamedTuple

from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

    from torch import Tensor, device

logger = setup_logger(__name__)


class _ArrayInfo(NamedTuple):
    """Array storage and native invalidity inspected without numerical work."""

    data: object
    invalid: object
    shape: tuple[int, ...]
    dtype: object


def _require_torch() -> ModuleType:
    """Import Torch only when numerical execution is requested."""
    try:
        import torch
    except ImportError as exc:
        reject(
            "faninsar.core.algorithms requires Torch for numerical execution; "
            "install the torch package.",
            ImportError,
            logger=logger,
            cause=exc,
        )
    return torch


def _execution_device(*inputs: object) -> device:
    """Require all supplied tensors, including masks, to share CPU/CUDA placement."""
    torch = _require_torch()
    selected = None
    for value in inputs:
        if not isinstance(value, torch.Tensor):
            continue
        current = value.device
        if current.type not in {"cpu", "cuda"}:
            reject(
                f"Tensor execution requires a CPU or CUDA device, got {current}.",
                ValueError,
                logger=logger,
            )
        if selected is not None and current != selected:
            reject(
                "All Tensor inputs must share one device; "
                f"got {selected} and {current}.",
                ValueError,
                logger=logger,
            )
        selected = current
    return torch.device("cpu") if selected is None else selected


def _inspect_array(value: object, *, name: str, allow_bool: bool = False) -> _ArrayInfo:
    """Inspect supported storage, extracting native masks before conversion.

    ``allow_bool`` admits Boolean arrays in addition to float32/float64 arrays;
    callers requiring an actual Boolean mask must check the returned dtype.
    Native invalidity uses True for excluded values and retains scalar/view
    storage. Nonfinite-value checks belong to bounded numerical kernels.
    """
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            reject(f"{name} must be a dense strided Tensor.", TypeError, logger=logger)
        if value.requires_grad:
            reject(f"{name} must not require gradients.", ValueError, logger=logger)
        allowed = (
            (torch.float32, torch.float64, torch.bool)
            if allow_bool
            else (
                torch.float32,
                torch.float64,
            )
        )
        if value.dtype not in allowed:
            reject(
                f"{name} must have float32/float64 dtype"
                f"{' or Boolean mask dtype' if allow_bool else ''}; got {value.dtype}.",
                TypeError,
                logger=logger,
            )
        return _ArrayInfo(value, None, tuple(value.shape), value.dtype)

    try:
        import numpy as np
    except ImportError as exc:
        reject(
            f"{name} must be a Torch Tensor; NumPy array input requires numpy.",
            ImportError,
            logger=logger,
            cause=exc,
        )
    if not isinstance(value, np.ndarray):
        reject(
            f"{name} must be a NumPy array or Torch Tensor, "
            f"not {type(value).__name__}.",
            TypeError,
            logger=logger,
        )
    if isinstance(value, np.ma.MaskedArray):
        invalid = value.mask
        data = value.data
        if np.ndim(invalid) == 0:
            invalid = bool(invalid)
        elif any(stride == 0 for stride in invalid.strides):
            # Retain one element along repeated axes before a possible transfer.
            # Full-mask expansion, including a device copy, would defeat preflight.
            invalid = invalid[
                tuple(
                    slice(0, 1) if stride == 0 and length > 1 else slice(None)
                    for length, stride in zip(
                        invalid.shape, invalid.strides, strict=True
                    )
                )
            ]
    else:
        invalid = None
        data = value
    dtype = data.dtype
    if not (
        (dtype.kind == "f" and dtype.itemsize in {4, 8})
        or (allow_bool and dtype.kind == "b")
    ):
        reject(
            f"{name} must have float32/float64 dtype"
            f"{' or Boolean mask dtype' if allow_bool else ''}; got {dtype}.",
            TypeError,
            logger=logger,
        )
    return _ArrayInfo(data, invalid, tuple(data.shape), dtype)


def _checked_size(
    shape: Iterable[int], *, itemsize: int, name: str, limit_bytes: int
) -> int:
    """Return required bytes, rejecting invalid/oversized metadata before allocation."""
    raw_dimensions = tuple(shape)
    if any(
        isinstance(value, bool)
        or str(getattr(value, "dtype", "")) in {"bool", "torch.bool"}
        for value in (*raw_dimensions, itemsize, limit_bytes)
    ):
        reject(
            f"{name} shape, itemsize, and limit_bytes must be integers, not Boolean.",
            ValueError,
            logger=logger,
        )
    try:
        dimensions = tuple(index(value) for value in raw_dimensions)
        size = index(itemsize)
        limit = index(limit_bytes)
    except TypeError as exc:
        reject(
            f"{name} shape, itemsize, and limit_bytes must be integers.",
            ValueError,
            logger=logger,
            cause=exc,
        )
    if size <= 0 or limit < 0 or any(value < 0 for value in dimensions):
        reject(
            f"{name} requires nonnegative dimensions/limit and a positive itemsize.",
            ValueError,
            logger=logger,
        )
    count = 1
    for dimension in dimensions:
        count *= dimension
    nbytes = count * size
    if nbytes > min(limit, sys.maxsize) or any(
        dimension > sys.maxsize for dimension in dimensions
    ):
        reject(
            f"{name} requires {nbytes} bytes, exceeding limit_bytes={limit} "
            "or the platform allocation range.",
            ValueError,
            logger=logger,
        )
    return nbytes


def _as_tensor(
    value: object, *, device: device, name: str, allow_bool: bool = False
) -> tuple[Tensor, Tensor | bool | None]:
    """Adapt inspected input after caller preflight, preserving float32 storage.

    The second result is native invalidity only: None, a Boolean scalar, or a
    Boolean Tensor on ``device`` retaining its unexpanded mask shape. Callers
    must not mutate either returned storage because NumPy may share it.
    """
    info = _inspect_array(value, name=name, allow_bool=allow_bool)
    torch = _require_torch()
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"} or (
        target.type == "cuda" and target.index is None
    ):
        reject(
            f"{name} requires a concrete CPU/CUDA device; got {target}.",
            ValueError,
            logger=logger,
        )
    if target.type == "cpu":
        target = torch.device("cpu")
    if isinstance(info.data, torch.Tensor):
        if info.data.device != target:
            reject(
                f"{name} is on {info.data.device}, not {target}; Tensor transfer "
                "must be explicit at the call site.",
                ValueError,
                logger=logger,
            )
        return info.data, None
    # NumPy is only a storage adapter here, not a numerical executor. Copies
    # needed for unsupported storage layouts occur after the caller's preflight.
    raw = info.data
    if (
        any(stride < 0 for stride in raw.strides)
        or not raw.dtype.isnative
        or not raw.flags.writeable
    ):
        raw = raw.astype(raw.dtype.newbyteorder("="), order="C", copy=True)
    data = torch.as_tensor(raw, device=target)
    invalid = info.invalid
    if invalid is not None and not isinstance(invalid, bool):
        if any(stride < 0 for stride in invalid.strides) or not invalid.flags.writeable:
            invalid = invalid.copy(order="C")
        invalid = torch.as_tensor(invalid, dtype=torch.bool, device=target)
    return data, invalid
