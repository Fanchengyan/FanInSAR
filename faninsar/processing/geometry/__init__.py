"""Python reference geometry for orbit, ellipsoid, Doppler and baseline."""

from __future__ import annotations

from .api import (
    DEM,
    ConstantDEM,
    DEMProduct,
    RasterDEM,
    SourceDEM,
    VerticalDatum,
    dem_content_identity,
)
from .baseline import BaselineComponents, geometric_baseline, zero_doppler_residual_hz
from .boundary import (
    BoundaryDecision,
    evaluate_canonical_boundary,
    normalize_result_boundary,
)
from .cache import ArtifactValidationError, validate_artifact
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
    GeoidResource,
    GeoidResourceError,
)
from .geoid import GeoidSampler, load_geoid
from .grids import GeoGrid, GeoGridMixin, GridSpec, format_bounds_and_crs
from .lut_cache import TransformCacheKey, read_transform_cache, write_transform_cache
from .orbit import (
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
from .prepared import (
    PreparedGeometryArrayPayload,
    PreparedLutArrayPayload,
)
from .profiles import Profile
from .public import (
    BackendSelector,
    NativeContextInputs,
    PreparedGeometry,
    execute_geometry,
    prepare_geometry,
)
from .raster_ops import match_to_raster
from .resources import ResourceBudget, preflight_grid
from .seam import (
    SOURCE_KERNEL_RADIUS,
    SOURCE_KERNEL_SIZE,
    ExplicitAntimeridianError,
    SeamAwareSourceSampler,
    plan_query_windows,
)
from .transforms import (
    GEOM_FLAG_LAYOVER,
    GEOM_FLAG_SHADOW,
    INT32_MAX,
    ArraySpan,
    CandidateKey,
    DeviceKey,
    ExecutionProfile,
    GeometryValidationError,
    NativeSpan,
    Operation,
    OperationSettings,
    RadarGeometryModel,
    RawSpan,
    SolverSettings,
    TransformResult,
    geom_flags,
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
    "GEOM_FLAG_LAYOVER",
    "GEOM_FLAG_SHADOW",
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
    "GeoidResource",
    "GeoidResourceError",
    "GeoidSampler",
    "GeometryValidationError",
    "GridSpec",
    "NativeContextInputs",
    "NativeSpan",
    "Operation",
    "OperationSettings",
    "OrbitInterpolator",
    "OrbitState",
    "PcStacSource",
    "PreparedGeometry",
    "PreparedGeometryArrayPayload",
    "PreparedLutArrayPayload",
    "Profile",
    "RadarGeometryModel",
    "RasterDEM",
    "RawSpan",
    "ResourceBudget",
    "SeamAwareSourceSampler",
    "SolverSettings",
    "SourceDEM",
    "SourceResource",
    "TransformCacheKey",
    "TransformResult",
    "VerticalDatum",
    "bounds_from_xy",
    "conversion_models",
    "convert_heights",
    "dem_content_identity",
    "ecef_to_llh",
    "evaluate_canonical_boundary",
    "execute_geometry",
    "fetch_required",
    "format_bounds_and_crs",
    "geoinfo_from_xy",
    "geom_flags",
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
