"""Explicit Network product and reader boundary."""

from __future__ import annotations

from typing import Any

from .products import (
    AcquisitionKey,
    AssetKind,
    AssetTransform,
    AssetTransformOperation,
    Interferogram,
    NetworkProduct,
    NetworkProductIndex,
    NetworkProductKey,
    NetworkProductRecord,
    PhaseConvention,
)

_NETWORK_EXPORTS = {
    "Network",
}
_READER_EXPORTS = {
    "InterferogramCollection",
    "NetworkReader",
    "ReaderRegistry",
    "ENTRY_POINT_GROUP",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve path-based Network and reader implementations."""
    if name in _NETWORK_EXPORTS:
        from . import network as module

        return getattr(module, name)
    if name in _READER_EXPORTS:
        if name == "InterferogramCollection":
            from .readers.interferogram import InterferogramCollection

            return InterferogramCollection
        if name == "NetworkReader":
            from .protocols import NetworkReader

            return NetworkReader
        from .registry import (
            ENTRY_POINT_GROUP,
            ReaderRegistry,
        )

        return locals()[name]
    raise AttributeError(name)


__all__ = [
    "ENTRY_POINT_GROUP",
    "AcquisitionKey",
    "AssetKind",
    "AssetTransform",
    "AssetTransformOperation",
    "Interferogram",
    "InterferogramCollection",
    "Network",
    "NetworkProduct",
    "NetworkProductIndex",
    "NetworkProductKey",
    "NetworkProductRecord",
    "NetworkReader",
    "PhaseConvention",
    "ReaderRegistry",
]
