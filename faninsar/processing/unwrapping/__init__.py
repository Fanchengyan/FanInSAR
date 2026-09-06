"""Phase unwrapping backends and reconciliation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .api import UnwrapBackend, unwrap
from .common import (
    SpatialUnwrapper,
    SpatialUnwrapResult,
)
from .errors import NoValidSupportError, UnwrapFailedError
from .irls import SpatialIRLS, wrap_phase

if TYPE_CHECKING:
    from .snaphu_backend import (
        Snaphu,
        SnaphuConfig,
        SnaphuNotAvailableError,
        snaphu_available,
        snaphu_unwrap,
    )

__all__ = [
    "NoValidSupportError",
    "Snaphu",
    "SnaphuConfig",
    "SnaphuNotAvailableError",
    "SpatialIRLS",
    "SpatialUnwrapResult",
    "SpatialUnwrapper",
    "UnwrapBackend",
    "UnwrapFailedError",
    "snaphu_available",
    "snaphu_unwrap",
    "unwrap",
    "wrap_phase",
]

_SNAPHU_EXPORTS = {
    "Snaphu",
    "SnaphuConfig",
    "SnaphuNotAvailableError",
    "snaphu_available",
    "snaphu_unwrap",
}


def __getattr__(name: str) -> Any:
    """Load the optional SNAPHU adapter only when its API is requested."""
    if name in _SNAPHU_EXPORTS:
        backend = import_module("faninsar.processing.unwrapping.snaphu_backend")
        return getattr(backend, name)
    raise AttributeError(name)
