"""Python reference geometry for orbit, ellipsoid, Doppler and baseline."""

from __future__ import annotations

from .baseline import BaselineComponents, geometric_baseline, zero_doppler_residual_hz
from .boundary import (
    BoundaryDecision,
    evaluate_canonical_boundary,
    normalize_result_boundary,
)
from .dem import ConstantHeightDEM, GeoidAdjustedDEM, NetCDFGeoid, RasterDEM
from .dem_manager import (
    DEMManager,
    copernicus_tile_name,
    default_dem_name,
    get_dem_manager,
)
from .ellipsoid import (
    WGS84_A_M,
    WGS84_E2,
    WGS84_F,
    ecef_to_llh,
    llh_to_ecef,
    local_earth_radius_m,
)
from .lut_cache import TransformCacheKey, read_transform_cache, write_transform_cache
from .orbit import (
    OrbitInterpolationError,
    OrbitInterpolator,
    OrbitState,
    interpolate_orbit,
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
from .transforms import (
    RadarGeometryModel,
    TransformResult,
    geo2rdr,
    rdr2geo_ellipsoid,
    rdr2geo_with_dem,
    rdr2geo_with_dem_chunked,
)
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
    "BaselineComponents",
    "BoundaryDecision",
    "CandidateKey",
    "ConstantHeightDEM",
    "DEMManager",
    "DeviceKey",
    "ExecutionProfile",
    "GeoidAdjustedDEM",
    "GeometryValidationError",
    "LocalPreparedGeometryProvider",
    "NativeSpan",
    "NetCDFGeoid",
    "Operation",
    "OperationSettings",
    "OrbitInterpolationError",
    "OrbitInterpolator",
    "OrbitState",
    "PreparedGenerationLease",
    "PreparedGenerationReader",
    "PreparedGenerationRecord",
    "PreparedGenerationStore",
    "PreparedGeometryArrayPayload",
    "PreparedLutArrayPayload",
    "PreparedScenePayload",
    "RadarGeometryModel",
    "RasterDEM",
    "RawSpan",
    "ScenePreparationCallback",
    "SolverSettings",
    "TransformCacheKey",
    "TransformResult",
    "TransformResultV2",
    "copernicus_tile_name",
    "default_dem_name",
    "ecef_to_llh",
    "evaluate_canonical_boundary",
    "geo2rdr",
    "geometric_baseline",
    "get_dem_manager",
    "interpolate_orbit",
    "llh_to_ecef",
    "local_earth_radius_m",
    "normalize_result_boundary",
    "rdr2geo_ellipsoid",
    "rdr2geo_with_dem",
    "rdr2geo_with_dem_chunked",
    "read_transform_cache",
    "validate_array_span",
    "validate_input_span",
    "validate_native_spans",
    "validate_span",
    "validate_spans",
    "write_transform_cache",
    "zero_doppler_residual_hz",
]
