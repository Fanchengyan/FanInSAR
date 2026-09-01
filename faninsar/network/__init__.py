"""Explicit Network reader boundary and canonical Network facade."""

from __future__ import annotations

from typing import Any

from .protocols import NetworkReader
from .registry import (
    ENTRY_POINT_GROUP,
    DuplicateReaderError,
    InvalidReaderError,
    ReaderNotFoundError,
    ReaderRegistry,
    ReaderRegistryError,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "DuplicateReaderError",
    "InvalidReaderError",
    "Network",
    "NetworkReader",
    "ReaderNotFoundError",
    "ReaderRegistry",
    "ReaderRegistryError",
]


def __getattr__(name: str) -> Any:
    """Resolve the data-backed Network lazily to avoid import cycles."""
    if name == "Network":
        from .network import Network

        return Network
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
