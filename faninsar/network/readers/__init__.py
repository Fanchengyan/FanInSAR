"""Readers for persisted Network product collections.

The reader package contains the concrete file-backed implementations used by
``Network.open``. It is separate from the public Network workflow and does not
expose a second product hierarchy.
"""

from __future__ import annotations

from .exceptions import (
    COGValidationError,
    GridMismatchError,
    MetadataError,
    MissingGeometryAssetError,
    MissingInterferogramAssetError,
    NetworkGeometryError,
    NetworkInterferogramError,
    NetworkProductError,
    PairNotFoundError,
)
from .geometry import NetworkGeometry
from .interferogram import InterferogramCollection
from .metadata import (
    GEOMETRY_ASSETS,
    INTERFEROGRAM_ASSETS,
    GeometryAssetName,
    InterferogramAssetName,
)
from .timeseries import NetworkTimeSeries

__all__ = [
    "COGValidationError",
    "GEOMETRY_ASSETS",
    "INTERFEROGRAM_ASSETS",
    "GridMismatchError",
    "GeometryAssetName",
    "InterferogramAssetName",
    "InterferogramCollection",
    "MetadataError",
    "MissingGeometryAssetError",
    "MissingInterferogramAssetError",
    "NetworkGeometry",
    "NetworkGeometryError",
    "NetworkInterferogramError",
    "NetworkProductError",
    "NetworkTimeSeries",
    "PairNotFoundError",
]
