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
from .network import (
    ExternalNetworkLayoutError,
    GAMMANetwork,
    IncompleteNetworkError,
    IncompleteNetworkProductError,
    ISCE2Network,
    ISCE3Network,
    LegacyLayoutError,
    LegacyNetworkLayoutError,
    Network,
    NetworkAnalysisError,
    NetworkConstructionError,
    NetworkCurrentError,
    NetworkGenerationError,
    NetworkLayoutError,
    NetworkManifestError,
    NetworkPathError,
    SNAPNetwork,
    UnknownNetworkIndexTypeError,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "DuplicateReaderError",
    "InvalidReaderError",
    "ExternalNetworkLayoutError",
    "GAMMANetwork",
    "IncompleteNetworkError",
    "IncompleteNetworkProductError",
    "ISCE2Network",
    "ISCE3Network",
    "LegacyLayoutError",
    "LegacyNetworkLayoutError",
    "Network",
    "NetworkAnalysisError",
    "NetworkConstructionError",
    "NetworkCurrentError",
    "NetworkGenerationError",
    "NetworkLayoutError",
    "NetworkManifestError",
    "NetworkReader",
    "NetworkPathError",
    "ReaderNotFoundError",
    "ReaderRegistry",
    "ReaderRegistryError",
    "SNAPNetwork",
    "UnknownNetworkIndexTypeError",
]
