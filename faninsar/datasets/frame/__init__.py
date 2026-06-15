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
from .publish import build_upload_plan, iter_frame_assets, publish_to_huggingface
from .remote import RemoteFrame
from .timeseries import FrameTimeSeries

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
    "FrameTimeSeries",
    "GeometryAssetName",
    "GridMismatchError",
    "InterferogramAssetName",
    "MetadataError",
    "MissingGeometryAssetError",
    "MissingInterferogramAssetError",
    "PairNotFoundError",
    "RemoteFrame",
    "build_upload_plan",
    "iter_frame_assets",
    "publish_to_huggingface",
]
