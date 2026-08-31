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
from .datum import (
    conversion_models,
    convert_heights,
    fetch_required,
    requires_fetch,
    validate_datum,
)
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
from .providers import (
    GLO30_PC,
    GLO90_PC,
    PC_REGISTRY,
    PC_STAC_URL,
    PcStacSource,
    ProviderUnavailableError,
    SourceResource,
    get_provider,
    materialize_source,
    parse_selection,
)
from .resources import ResourceBudget, ResourcePreflightError, preflight_grid
from .seam import ExplicitAntimeridianError, SeamAwareSourceSampler, plan_query_windows
from .transport import (
    BoundedTransferError,
    download,
    download_ftp,
    resolve_cache_path,
    stream_response_to_cache,
    stream_to_cache,
    validate_https_origin,
    validate_redirect,
)

__all__ = [
    "DEFAULT_RESOURCES",
    "DEM",
    "EGM96",
    "EGM2008_2_5",
    "GLO30_PC",
    "GLO90_PC",
    "PC_REGISTRY",
    "PC_STAC_URL",
    "ArtifactValidationError",
    "BoundedTransferError",
    "CachePathError",
    "ConstantDEM",
    "DEMProduct",
    "ExplicitAntimeridianError",
    "Fetch",
    "GeoidArtifactError",
    "GeoidOfflineError",
    "GeoidResource",
    "GeoidResourceError",
    "GeoidSampler",
    "GridSpec",
    "PcStacSource",
    "ProviderUnavailableError",
    "RasterDEM",
    "ResourceBudget",
    "ResourcePreflightError",
    "SeamAwareSourceSampler",
    "SourceDEM",
    "SourceResource",
    "VerticalDatum",
    "conversion_models",
    "convert_heights",
    "download",
    "download_ftp",
    "fetch_required",
    "get_provider",
    "load_geoid",
    "materialize_source",
    "parse_selection",
    "plan_query_windows",
    "preflight_grid",
    "requires_fetch",
    "resolve_cache_path",
    "stream_response_to_cache",
    "stream_to_cache",
    "validate_artifact",
    "validate_datum",
    "validate_https_origin",
    "validate_redirect",
]
