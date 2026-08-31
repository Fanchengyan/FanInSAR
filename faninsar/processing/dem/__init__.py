"""Unified DEM, grid, and geoid resource public API."""

from __future__ import annotations

from .api import (
    DEM,
    ConstantDEM,
    DEMProduct,
    GridSpec,
    RasterDEM,
    SourceDEM,
    VerticalDatum,
)
from .cache import ArtifactValidationError, CachePathError, validate_artifact
from .datum import conversion_models, fetch_required, requires_fetch, validate_datum
from .fetch import (
    DEFAULT_RESOURCES,
    EGM96,
    EGM2008_2_5,
    Fetch,
    GeoidArtifactError,
    GeoidOfflineError,
    GeoidResource,
    GeoidResourceError,
)
from .geoid import GeoidSampler, load_geoid

__all__ = [
    "DEFAULT_RESOURCES",
    "DEM",
    "EGM96",
    "EGM2008_2_5",
    "ArtifactValidationError",
    "CachePathError",
    "ConstantDEM",
    "DEMProduct",
    "Fetch",
    "GeoidArtifactError",
    "GeoidOfflineError",
    "GeoidResource",
    "GeoidResourceError",
    "GeoidSampler",
    "GridSpec",
    "RasterDEM",
    "SourceDEM",
    "VerticalDatum",
    "conversion_models",
    "fetch_required",
    "load_geoid",
    "requires_fetch",
    "validate_artifact",
    "validate_datum",
]
