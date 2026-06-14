"""FanInSAR frame-level InSAR products.

This subpackage provides standardized frame-level geometry and interferogram
assets built on top of existing FanInSAR geospatial primitives.
"""

from __future__ import annotations

from .exceptions import (
    COGValidationError,
    FrameGeometryError,
    FrameInterferogramError,
    FrameProductError,
    GridMismatchError,
    MetadataError,
    MissingGeometryAssetError,
    MissingInterferogramAssetError,
    PairNotFoundError,
)
from .frame import Frame
from .geometry import FrameGeometry
from .interferogram import FrameInterferogramCollection
from .metadata import (
    GEOMETRY_ASSETS,
    INTERFEROGRAM_ASSETS,
    GeometryAssetName,
    InterferogramAssetName,
)

__all__ = [
    "GEOMETRY_ASSETS",
    "INTERFEROGRAM_ASSETS",
    "COGValidationError",
    "Frame",
    "FrameGeometry",
    "FrameGeometryError",
    "FrameInterferogramCollection",
    "FrameInterferogramError",
    "FrameProductError",
    "GeometryAssetName",
    "GridMismatchError",
    "InterferogramAssetName",
    "MetadataError",
    "MissingGeometryAssetError",
    "MissingInterferogramAssetError",
    "PairNotFoundError",
]
