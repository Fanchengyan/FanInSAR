"""Python reference geometry for orbit, ellipsoid, Doppler and baseline."""

from __future__ import annotations

from .baseline import BaselineComponents, geometric_baseline, zero_doppler_residual_hz
from .boundary import (
    BoundaryDecision,
    evaluate_canonical_boundary,
    normalize_result_boundary,
)
from .converters import GeoDataFormatConverter
from .coordinates import (
    bounds_from_xy,
    geoinfo_from_xy,
    transform_from_xy,
    xy_from_profile,
    xy_from_transform,
)
from .ellipsoid import (
    WGS84_A_M,
    WGS84_E2,
    WGS84_F,
    ecef_to_llh,
    llh_to_ecef,
    local_earth_radius_m,
)
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

__all__ = [
    "INT32_MAX",
    "WGS84_A_M",
    "WGS84_E2",
    "WGS84_F",
    "ArraySpan",
    "BackendSelector",
    "BaselineComponents",
    "BoundaryDecision",
    "CandidateKey",
    "DeviceKey",
    "ExecutionProfile",
    "GeoDataFormatConverter",
    "GeoGrid",
    "GeoGridMixin",
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
    "PreparedGenerationLease",
    "PreparedGenerationReader",
    "PreparedGenerationRecord",
    "PreparedGenerationStore",
    "PreparedGeometry",
    "PreparedGeometryArrayPayload",
    "PreparedLutArrayPayload",
    "PreparedScenePayload",
    "Profile",
    "RadarGeometryModel",
    "RawSpan",
    "ScenePreparationCallback",
    "SolverSettings",
    "TransformCacheKey",
    "TransformResult",
    "TransformResultV2",
    "bounds_from_xy",
    "ecef_to_llh",
    "evaluate_canonical_boundary",
    "execute_geometry",
    "execute_geometry_v2",
    "format_bounds_and_crs",
    "geoinfo_from_xy",
    "geometric_baseline",
    "interpolate_orbit",
    "llh_to_ecef",
    "local_earth_radius_m",
    "match_to_raster",
    "normalize_result_boundary",
    "prepare_geometry",
    "prepare_geometry_v2",
    "prepare_production_geometry",
    "read_transform_cache",
    "run_geo2rdr",
    "run_rdr2geo",
    "run_rdr2geo_chunked",
    "transform_from_xy",
    "validate_array_span",
    "validate_input_span",
    "validate_native_spans",
    "validate_span",
    "validate_spans",
    "write_transform_cache",
    "xy_from_profile",
    "xy_from_transform",
    "zero_doppler_residual_hz",
]
