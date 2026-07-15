"""Python reference geometry for orbit, ellipsoid, Doppler and baseline."""

from __future__ import annotations

from .baseline import BaselineComponents, geometric_baseline, zero_doppler_residual_hz
from .dem import ConstantHeightDEM, RasterDEM
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
from .transforms import (
    RadarGeometryModel,
    TransformResult,
    geo2rdr,
    rdr2geo_ellipsoid,
    rdr2geo_with_dem,
    rdr2geo_with_dem_chunked,
)

__all__ = [
    "WGS84_A_M",
    "WGS84_E2",
    "WGS84_F",
    "BaselineComponents",
    "ConstantHeightDEM",
    "OrbitInterpolationError",
    "OrbitInterpolator",
    "OrbitState",
    "RadarGeometryModel",
    "RasterDEM",
    "TransformCacheKey",
    "TransformResult",
    "ecef_to_llh",
    "geo2rdr",
    "geometric_baseline",
    "interpolate_orbit",
    "llh_to_ecef",
    "local_earth_radius_m",
    "rdr2geo_ellipsoid",
    "rdr2geo_with_dem",
    "rdr2geo_with_dem_chunked",
    "read_transform_cache",
    "write_transform_cache",
    "zero_doppler_residual_hz",
]
