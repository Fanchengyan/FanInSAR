"""WGS84 ellipsoid conversions between geodetic and ECEF coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

# WGS84 parameters (SI metres)
WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True, slots=True)
class Geodetic:
    """Geodetic latitude, longitude, and ellipsoidal height."""

    latitude_deg: float
    longitude_deg: float
    height_m: float


def llh_to_ecef(
    latitude_deg: float | np.ndarray,
    longitude_deg: float | np.ndarray,
    height_m: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic LLH to ECEF XYZ.

    Parameters
    ----------
    latitude_deg, longitude_deg : float or array
        Geodetic latitude and longitude in degrees.
    height_m : float or array
        Ellipsoidal height in metres.

    Returns
    -------
    tuple of numpy.ndarray
        ECEF ``x``, ``y``, ``z`` in metres.

    """
    lat = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    height = np.asarray(height_m, dtype=np.float64)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    radius_n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (radius_n + height) * cos_lat * cos_lon
    y = (radius_n + height) * cos_lat * sin_lon
    z = (radius_n * (1.0 - WGS84_E2) + height) * sin_lat
    return x, y, z


def ecef_to_llh(
    x_m: float | np.ndarray,
    y_m: float | np.ndarray,
    z_m: float | np.ndarray,
    *,
    iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF XYZ to geodetic LLH with Bowring-style iteration.

    Parameters
    ----------
    x_m, y_m, z_m : float or array
        ECEF coordinates in metres.
    iterations : int, optional
        Fixed iteration count for latitude refinement.

    Returns
    -------
    tuple of numpy.ndarray
        Latitude (deg), longitude (deg), height (m).

    """
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    longitude = np.arctan2(y, x)
    radius_p = np.hypot(x, y)
    latitude = np.arctan2(z, radius_p * (1.0 - WGS84_E2))
    for _ in range(iterations):
        sin_lat = np.sin(latitude)
        radius_n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        latitude = np.arctan2(z + WGS84_E2 * radius_n * sin_lat, radius_p)
    sin_lat = np.sin(latitude)
    cos_lat = np.cos(latitude)
    radius_n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    height = np.where(
        np.abs(cos_lat) > 1e-12,
        radius_p / cos_lat - radius_n,
        np.abs(z) / np.maximum(np.abs(sin_lat), 1e-16) - radius_n * (1.0 - WGS84_E2),
    )
    return np.rad2deg(latitude), np.rad2deg(longitude), height


def local_earth_radius_m(latitude_deg: float) -> float:
    """Return the WGS84 Gaussian mean local Earth radius at a latitude."""
    lat = math.radians(latitude_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    radius_m = WGS84_A_M * (1.0 - WGS84_E2) / ((1.0 - WGS84_E2 * sin_lat**2) ** 1.5)
    radius_n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    if abs(cos_lat) < 1e-12:
        return radius_n
    # Gaussian mean radius
    return math.sqrt(radius_m * radius_n)
