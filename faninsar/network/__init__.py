"""Explicit Network reader boundary and canonical Network facade."""

from __future__ import annotations

from typing import Any

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
from faninsar.core.interferogram import Interferogram
from .readers.interferogram import InterferogramCollection
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
    "ExternalNetworkLayoutError",
    "GAMMANetwork",
    "ISCE2Network",
    "ISCE3Network",
    "IncompleteNetworkError",
    "IncompleteNetworkProductError",
    "Interferogram",
    "InterferogramCollection",
    "InvalidReaderError",
    "LegacyLayoutError",
    "LegacyNetworkLayoutError",
    "Network",
    "NetworkAnalysisError",
    "NetworkConstructionError",
    "NetworkCurrentError",
    "NetworkGenerationError",
    "NetworkLayoutError",
    "NetworkManifestError",
    "NetworkPathError",
    "NetworkReader",
    "ReaderNotFoundError",
    "ReaderRegistry",
    "ReaderRegistryError",
    "SNAPNetwork",
    "UnknownNetworkIndexTypeError",
]
