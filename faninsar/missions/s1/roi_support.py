"""Exact Polygon scientific support for Sentinel-1 scene publication (P0074).

Support is the intersection of (1) the unbuffered ROI, including holes, evaluated
at actual destination coordinates and (2) native sample validity. A rectangular
crop/window may remain as storage; samples outside support are not scientific
output. Amplitude is not used as a validity proxy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from faninsar.data.query import BoundingBox, Polygons
    from faninsar.processing.geometry import DEM, RadarGeometryModel
    from faninsar.processing.mosaicking.grid import GeoGridSpec

logger = setup_logger(__name__)

_INVALID_SAMPLE = np.complex64(np.nan + 1j * np.nan)


def roi_lonlat_geometry(roi: BoundingBox | Polygons) -> object:
    """Return the ROI as a lon/lat shapely geometry.

    Parameters
    ----------
    roi : BoundingBox or Polygons
        Apply-window ROI.

    Returns
    -------
    shapely geometry
        Union of desired regions in EPSG:4326.

    """
    from faninsar.missions.s1.processing import _roi_geometry

    return _roi_geometry(roi)


def points_inside_roi(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    roi_geometry: object,
) -> np.ndarray:
    """Return whether lon/lat samples lie in the ROI, excluding holes.

    Parameters
    ----------
    longitude_deg, latitude_deg : numpy.ndarray
        Destination coordinates in degrees, broadcast to one shape.
    roi_geometry : shapely geometry
        Polygon or MultiPolygon in lon/lat.

    Returns
    -------
    numpy.ndarray
        Boolean mask; False for holes, exterior, and non-finite coordinates.

    """
    from shapely import contains_xy

    from faninsar.processing.geocoding.geo_lut import polygon_parts

    longitude = np.asarray(longitude_deg, dtype=np.float64)
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    longitude, latitude = np.broadcast_arrays(longitude, latitude)
    inside = np.zeros(longitude.shape, dtype=bool)
    finite = np.isfinite(longitude) & np.isfinite(latitude)
    if not np.any(finite):
        return inside
    hits = np.zeros(int(np.count_nonzero(finite)), dtype=bool)
    xs = longitude[finite]
    ys = latitude[finite]
    parts = polygon_parts(roi_geometry)
    if not parts:
        hits = contains_xy(roi_geometry, xs, ys)
    else:
        for part in parts:
            hits |= np.asarray(contains_xy(part, xs, ys), dtype=bool)
    inside[finite] = hits
    return inside


def radar_scientific_support(
    geometry: RadarGeometryModel | object,
    dem: DEM | None,
    *,
    crop: tuple[int, int, int, int],
    native_valid: np.ndarray | None,
    roi: BoundingBox | Polygons | None,
    device: str = "auto",
) -> np.ndarray:
    """Return exact Radar support on a native crop using per-pixel rdr2geo.

    Parameters
    ----------
    geometry : RadarGeometryModel
        Destination burst geometry. A non-model stand-in is rejected when
        ``roi`` is set rather than publishing the full crop as support.
    dem : DEM or None
        Height sampler forwarded to rdr2geo.
    crop : tuple of int
        Native ``(row0, row1, col0, col1)`` of the published window.
    native_valid : numpy.ndarray or None
        Burst-native validity on the crop; canvas fill is never promoted.
    roi : BoundingBox or Polygons or None
        Unbuffered scientific ROI.
    device : str, optional
        Geometry device.

    Returns
    -------
    numpy.ndarray
        Boolean support with shape ``(row1 - row0, col1 - col0)``.

    """
    from faninsar.processing.geometry.prepare_production import run_rdr2geo
    from faninsar.processing.geometry.transforms import RadarGeometryModel

    row0, row1, col0, col1 = (int(value) for value in crop)
    shape = (row1 - row0, col1 - col0)
    if any(size <= 0 for size in shape):
        reject_invalid_state("radar ROI crop must have positive shape")
    if roi is None:
        support = np.ones(shape, dtype=bool)
    elif not isinstance(geometry, RadarGeometryModel):
        reject_invalid_state(
            "exact radar ROI support requires RadarGeometryModel; "
            "unknown geometry cannot publish full-window support"
        )
    else:
        rows = np.arange(row0, row1, dtype=np.float64)
        cols = np.arange(col0, col1, dtype=np.float64)
        azimuth, slant = np.meshgrid(rows, cols, indexing="ij")
        result = run_rdr2geo(geometry, azimuth, slant, dem, device=device)
        converged = np.asarray(result.converged, dtype=bool)
        if converged.shape != shape:
            reject_invalid_state("rdr2geo support shape must match the radar crop")
        inside = points_inside_roi(
            result.longitude_deg,
            result.latitude_deg,
            roi_lonlat_geometry(roi),
        )
        support = converged & inside
    support = _and_native_valid(support, native_valid)
    if roi is not None and isinstance(geometry, RadarGeometryModel):
        logger.info(
            "radar ROI support %d/%d samples on crop %s",
            int(np.count_nonzero(support)),
            support.size,
            crop,
        )
    return support


def geo_scientific_support(
    geo_grid: GeoGridSpec,
    *,
    row0: int,
    col0: int,
    shape: tuple[int, int],
    native_valid: np.ndarray | None,
    roi: BoundingBox | Polygons | None,
) -> np.ndarray:
    """Return exact Geo support at destination pixel centres.

    Parameters
    ----------
    geo_grid : GeoGridSpec
        Committed geographic grid (transform and CRS).
    row0, col0 : int
        Origin of the published window on ``geo_grid``.
    shape : tuple of int
        Window shape ``(rows, cols)``.
    native_valid : numpy.ndarray or None
        LUT/native validity on the window.
    roi : BoundingBox or Polygons or None
        Unbuffered scientific ROI.

    Returns
    -------
    numpy.ndarray
        Boolean support matching ``shape``.

    """
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        reject_invalid_state("geo ROI window must have positive shape")
    if roi is None:
        support = np.ones((height, width), dtype=bool)
    else:
        from faninsar.processing.geocoding.geo_lut import grid_lonlat_rows

        row1 = int(row0) + height
        col1 = int(col0) + width
        latitude, longitude = grid_lonlat_rows(geo_grid, int(row0), row1)
        inside = points_inside_roi(
            longitude[:, int(col0) : col1],
            latitude[:, int(col0) : col1],
            roi_lonlat_geometry(roi),
        )
        if inside.shape != (height, width):
            reject_invalid_state("geo pixel-centre support must match the window")
        support = inside
    support = _and_native_valid(support, native_valid)
    if roi is not None:
        logger.info(
            "geo ROI support %d/%d samples at origin (%d, %d)",
            int(np.count_nonzero(support)),
            support.size,
            int(row0),
            int(col0),
        )
    return support


def apply_scientific_support(payload: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Return a complex64 payload with non-support samples set to NaN.

    Parameters
    ----------
    payload : numpy.ndarray
        Published complex samples.
    support : numpy.ndarray
        Boolean scientific support. Interior zeros on support are preserved.

    Returns
    -------
    numpy.ndarray
        Copy of ``payload`` as complex64.

    """
    samples = np.array(payload, dtype=np.complex64, copy=True)
    mask = np.asarray(support, dtype=bool)
    if samples.shape != mask.shape:
        reject_invalid_state("ROI support shape must match the published payload")
    samples[~mask] = _INVALID_SAMPLE
    return samples


def _and_native_valid(
    support: np.ndarray,
    native_valid: np.ndarray | None,
) -> np.ndarray:
    """Intersect geometry support with native sample validity."""
    if native_valid is None:
        return support
    native = np.asarray(native_valid, dtype=bool)
    if native.shape != support.shape:
        reject_invalid_state("native validity shape must match the published window")
    return support & native
