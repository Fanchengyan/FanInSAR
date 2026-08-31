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
    "DEM",
    "ConstantDEM",
    "DEMProduct",
    "GridSpec",
    "RasterDEM",
    "SourceDEM",
    "VerticalDatum",
    "ArtifactValidationError",
    "CachePathError",
    "DEFAULT_RESOURCES",
    "EGM96",
    "EGM2008_2_5",
    "Fetch",
    "GeoidArtifactError",
    "GeoidOfflineError",
    "GeoidResource",
    "GeoidResourceError",
    "GeoidSampler",
    "conversion_models",
    "fetch_required",
    "load_geoid",
    "requires_fetch",
    "validate_artifact",
    "validate_datum",
]
