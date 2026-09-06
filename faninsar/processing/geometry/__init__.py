"""Python reference geometry for orbit, ellipsoid, Doppler and baseline."""

from __future__ import annotations

from .api import DEM, ConstantDEM, DEMProduct, RasterDEM, SourceDEM, VerticalDatum
from .baseline import BaselineComponents, geometric_baseline, zero_doppler_residual_hz
from .boundary import (
    BoundaryDecision,
    evaluate_canonical_boundary,
    normalize_result_boundary,
)
from .cache import ArtifactValidationError, CachePathError, validate_artifact
from .converters import GeoDataFormatConverter
from .coordinates import (
    bounds_from_xy,
    geoinfo_from_xy,
    transform_from_xy,
    xy_from_profile,
    xy_from_transform,
)
from .datum import (
    conversion_models,
    convert_heights,
    fetch_required,
    requires_fetch,
    validate_datum,
)
from .ellipsoid import (
    WGS84_A_M,
    WGS84_E2,
    WGS84_F,
    ecef_to_llh,
    llh_to_ecef,
    local_earth_radius_m,
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
from .grids import GeoGrid, GeoGridMixin, GridSpec, format_bounds_and_crs
from .lut_cache import TransformCacheKey, read_transform_cache, write_transform_cache
from .orbit import (
    OrbitInterpolationError,
    OrbitInterpolator,
    OrbitState,
    interpolate_orbit,
)
from .prepare_production import (
    prepare_production_geometry,
    run_geo2rdr,
    run_rdr2geo,
    run_rdr2geo_chunked,
)
from .prepared_provider import (
    LocalPreparedGeometryProvider,
    PreparedGeometryArrayPayload,
    PreparedLutArrayPayload,
    PreparedScenePayload,
    ScenePreparationCallback,
)
from .prepared_store import (
    PreparedGenerationLease,
    PreparedGenerationReader,
    PreparedGenerationRecord,
    PreparedGenerationStore,
)
from .profiles import Profile
from .public import (
    BackendSelector,
    NativeContextInputs,
    PreparedGeometry,
    execute_geometry,
    execute_geometry_v2,
    prepare_geometry,
    prepare_geometry_v2,
)
from .raster_ops import match_to_raster
from .resources import ResourceBudget, ResourcePreflightError, preflight_grid
from .seam import (
    SOURCE_KERNEL_RADIUS,
    SOURCE_KERNEL_SIZE,
    ExplicitAntimeridianError,
    SeamAwareSourceSampler,
    plan_query_windows,
)
from .transforms import RadarGeometryModel, TransformResult
from .v2 import (
    INT32_MAX,
    ArraySpan,
    CandidateKey,
    DeviceKey,
    ExecutionProfile,
    GeometryValidationError,
    NativeSpan,
    Operation,
    OperationSettings,
    RawSpan,
    SolverSettings,
    TransformResultV2,
    validate_array_span,
    validate_input_span,
    validate_native_spans,
    validate_span,
    validate_spans,
)

_PROVIDER_EXPORTS = {
    "GLO30_PC",
    "GLO90_PC",
    "PC_REGISTRY",
    "PC_STAC_URL",
    "PcStacSource",
    "ProviderUnavailableError",
    "SourceConflictError",
    "SourceResource",
    "get_provider",
    "materialize_source",
    "parse_selection",
}


def __getattr__(name: str) -> object:
    """Load provider adapters only when a caller requests one."""
    if name in _PROVIDER_EXPORTS:
        from . import providers

        value = getattr(providers, name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


__all__ = [
    "DEFAULT_RESOURCES",
    "DEM",
    "EGM96",
    "EGM2008_2_5",
    "GLO30_PC",
    "GLO90_PC",
    "INT32_MAX",
    "PC_REGISTRY",
    "PC_STAC_URL",
    "SOURCE_KERNEL_RADIUS",
    "SOURCE_KERNEL_SIZE",
    "WGS84_A_M",
    "WGS84_E2",
    "WGS84_F",
    "ArraySpan",
    "ArtifactValidationError",
    "BackendSelector",
    "BaselineComponents",
    "BoundaryDecision",
    "CachePathError",
    "CandidateKey",
    "ConstantDEM",
    "DEMProduct",
    "DeviceKey",
    "ExecutionProfile",
    "ExplicitAntimeridianError",
    "Fetch",
    "GeoDataFormatConverter",
    "GeoGrid",
    "GeoGridMixin",
    "GeoidArtifactError",
    "GeoidOfflineError",
    "GeoidResource",
    "GeoidResourceError",
    "GeoidSampler",
    "GeometryValidationError",
    "GridSpec",
    "LocalPreparedGeometryProvider",
    "NativeContextInputs",
    "NativeSpan",
    "Operation",
    "OperationSettings",
    "OrbitInterpolationError",
    "OrbitInterpolator",
    "OrbitState",
    "PcStacSource",
    "PreparedGenerationLease",
    "PreparedGenerationReader",
    "PreparedGenerationRecord",
    "PreparedGenerationStore",
    "PreparedGeometry",
    "PreparedGeometryArrayPayload",
    "PreparedLutArrayPayload",
    "PreparedScenePayload",
    "Profile",
    "ProviderUnavailableError",
    "RadarGeometryModel",
    "RasterDEM",
    "RawSpan",
    "ResourceBudget",
    "ResourcePreflightError",
    "ScenePreparationCallback",
    "SeamAwareSourceSampler",
    "SolverSettings",
    "SourceConflictError",
    "SourceDEM",
    "SourceResource",
    "TransformCacheKey",
    "TransformResult",
    "TransformResultV2",
    "VerticalDatum",
    "bounds_from_xy",
    "conversion_models",
    "convert_heights",
    "ecef_to_llh",
    "evaluate_canonical_boundary",
    "execute_geometry",
    "execute_geometry_v2",
    "fetch_required",
    "format_bounds_and_crs",
    "geoinfo_from_xy",
    "geometric_baseline",
    "get_provider",
    "interpolate_orbit",
    "llh_to_ecef",
    "load_geoid",
    "local_earth_radius_m",
    "match_to_raster",
    "materialize_source",
    "normalize_result_boundary",
    "parse_selection",
    "plan_query_windows",
    "preflight_grid",
    "prepare_geometry",
    "prepare_geometry_v2",
    "prepare_production_geometry",
    "read_transform_cache",
    "requires_fetch",
    "run_geo2rdr",
    "run_rdr2geo",
    "run_rdr2geo_chunked",
    "transform_from_xy",
    "validate_array_span",
    "validate_artifact",
    "validate_datum",
    "validate_input_span",
    "validate_native_spans",
    "validate_span",
    "validate_spans",
    "write_transform_cache",
    "xy_from_profile",
    "xy_from_transform",
    "zero_doppler_residual_hz",
]
