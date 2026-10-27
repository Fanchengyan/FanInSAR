"""Build geographic-to-radar lookup tables."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry.prepare_production import run_geo2rdr, run_rdr2geo
from faninsar.processing.runtime.memory import release_memmap_pages

if TYPE_CHECKING:
    from faninsar.processing.geometry import DEM, RadarGeometryModel
    from faninsar.processing.mosaicking.grid import GeoGridSpec
    from faninsar.processing.runtime.types import DeviceLike

logger = setup_logger(__name__)


def _grid_axes(grid: GeoGridSpec) -> tuple[float, float, float, float]:
    """Return canonical affine ``(x0, dx, y0, dy)`` components."""
    from affine import Affine

    affine = Affine(*grid.transform)
    return affine.c, affine.a, affine.f, affine.e


def _add_timing(timings: dict[str, float] | None, key: str, started: float) -> None:
    """Accumulate one exclusive substage into ``timings`` when provided."""
    if timings is None:
        return
    timings[key] = timings.get(key, 0.0) + (time.perf_counter() - started)


__all__ = [
    "Geo2RdrLUT",
    "build_geo2rdr_lut",
    "burst_geo_footprint_lonlat",
    "burst_geo_quad_lonlat",
    "grid_lonlat",
    "grid_lonlat_rows",
    "polygon_parts",
    "roi_geo_bbox",
    "roi_geo_mask",
]


@dataclass(frozen=True, slots=True)
class Geo2RdrLUT:
    """Full-resolution radar indices sampled on a geographic grid.

    Attributes
    ----------
    az_full, rg_full : numpy.ndarray
        Full-resolution azimuth and range indices.
    valid : numpy.ndarray
        Mask of converged indices inside the radar image.
    full_radar_shape : tuple[int, int]
        Shape of the full-resolution radar image.
    height_m : float
        Mean ellipsoidal height used to build the lookup table.
    height_full : numpy.ndarray, optional
        Per-pixel DEM samples reused by exact geometric flattening.
    row0, col0 : int
        Offset of this LUT inside the full geographic grid. When the LUT is
        built with row_range/col_range the arrays have bbox shape
        ``(row1-row0, col1-col0)`` and these fields locate the bbox.

    """

    az_full: np.ndarray
    rg_full: np.ndarray
    valid: np.ndarray
    full_radar_shape: tuple[int, int]
    height_m: float
    height_full: np.ndarray | None = None
    row0: int = 0
    col0: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        """Return the geographic grid shape."""
        return self.valid.shape


def derive_burst_geo_bbox(
    geometry: RadarGeometryModel,
    radar_shape: tuple[int, int],
    grid: GeoGridSpec,
    *,
    margin_px: int = 32,
    dem: DEM | None = None,
    device: DeviceLike,
) -> tuple[int, int, int, int]:
    """Return ``(row0, row1, col0, col1)`` bounding the burst footprint.

    The burst radar frame corners (plus edge midpoints for safety) are mapped
    with rdr2geo and converted to geo-grid pixel coordinates. A margin is
    added so the returned rectangle fully covers the burst's valid pixels,
    even where the DEM height shifts the ground track. This is used to crop
    geo2rdr LUT construction to the burst's footprint instead of scanning the
    whole shared study-area grid.

    Parameters
    ----------
    geometry : RadarGeometryModel
        Reference-scene radar geometry.
    radar_shape : tuple[int, int]
        Radar image shape ``(height, width)``.
    grid : GeoGridSpec
        Shared geographic grid.
    margin_px : int, optional
        Extra rows/cols added around the footprint bbox.
    dem : DEM, optional
        DEM used by rdr2geo. When omitted an ellipsoid is used.
    device : DeviceLike
        Required production device (``auto`` resolves to cpu or cuda).

    Returns
    -------
    tuple[int, int, int, int]
        ``(row0, row1, col0, col1)`` half-open bounding box.

    """
    height, width = radar_shape
    # corners + edge midpoints
    points = np.array(
        [
            [0.0, 0.0],
            [0.0, width - 1],
            [height - 1, 0.0],
            [height - 1, width - 1],
            [0.0, (width - 1) / 2],
            [height - 1, (width - 1) / 2],
            [(height - 1) / 2, 0.0],
            [(height - 1) / 2, width - 1],
        ],
        dtype=np.float64,
    )
    res = run_rdr2geo(
        geometry,
        points[:, 0],
        points[:, 1],
        dem,
        device=device,
    )
    lat = np.asarray(res.latitude_deg)
    lon = np.asarray(res.longitude_deg)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(ok):
        return (0, grid.height, 0, grid.width)

    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    xs, ys = transformer.transform(lon[ok], lat[ok])
    x0, dx, y0, dy = _grid_axes(grid)
    cols = (np.asarray(xs) - x0) / dx - 0.5
    rows = (y0 - np.asarray(ys)) / (-dy) - 0.5  # dy is negative north-up
    col0 = max(0, int(np.floor(np.min(cols))) - margin_px)
    col1 = min(grid.width, int(np.ceil(np.max(cols))) + 1 + margin_px)
    row0 = max(0, int(np.floor(np.min(rows))) - margin_px)
    row1 = min(grid.height, int(np.ceil(np.max(rows))) + 1 + margin_px)
    return (row0, row1, col0, col1)


def burst_geo_quad_lonlat(
    geometry: RadarGeometryModel,
    radar_shape: tuple[int, int],
    dem: DEM | None,
    *,
    device: DeviceLike,
) -> np.ndarray | None:
    """Return the burst frame's 4-corner ground quad in (lon, lat).

    The four radar-frame corners are mapped with rdr2geo (DEM-based when
    available, ellipsoid fallback per corner). Corners that still fail are
    dropped; at least three finite corners are required.

    Parameters
    ----------
    geometry : RadarGeometryModel
        Reference-scene radar geometry for the burst.
    radar_shape : tuple[int, int]
        Radar image shape ``(height, width)``.
    dem : DEM, optional
        DEM for rdr2geo; when omitted an ellipsoid is used.
    device : DeviceLike
        Required production device (``auto`` resolves to cpu or cuda).

    Returns
    -------
    numpy.ndarray or None
        ``(N, 2)`` ring of (lon, lat) corners (``N >= 3``) or None.

    """
    height, width = radar_shape
    points = np.array(
        [
            [0.0, 0.0],
            [0.0, width - 1],
            [height - 1, width - 1],
            [height - 1, 0.0],
        ],
        dtype=np.float64,
    )
    res = run_rdr2geo(
        geometry,
        points[:, 0],
        points[:, 1],
        dem,
        device=device,
    )
    lat = np.asarray(res.latitude_deg, dtype=np.float64)
    lon = np.asarray(res.longitude_deg, dtype=np.float64)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if dem is not None and not bool(ok.all()):
        # Stack DEMs are ROI-bounded while the quad corners are full-burst
        # edges; corners the DEM cannot bracket fall back to the ellipsoid.
        fallback = run_rdr2geo(
            geometry,
            points[~ok, 0],
            points[~ok, 1],
            None,
            device=device,
        )
        lat[~ok] = np.asarray(fallback.latitude_deg, dtype=np.float64)
        lon[~ok] = np.asarray(fallback.longitude_deg, dtype=np.float64)
        ok = np.isfinite(lat) & np.isfinite(lon)
    if int(ok.sum()) < 3:
        return None
    return np.column_stack([lon[ok], lat[ok]])


def burst_geo_footprint_lonlat(
    geometry: RadarGeometryModel,
    radar_shape: tuple[int, int],
    dem: DEM | None,
    grid: GeoGridSpec,
    *,
    device: DeviceLike,
) -> np.ndarray | None:
    """Return an eight-point convex hull in (lon, lat) for the burst footprint.

    Vertices are the same rdr2geo corners and edge midpoints used by
    :func:`derive_burst_geo_bbox`. The hull is formed in destination-grid
    pixel space so a lon/lat convex hull cannot drop a midpoint that is
    extreme after the map projection.
    """
    from scipy.spatial import ConvexHull

    height, width = radar_shape
    points = np.array(
        [
            [0.0, 0.0],
            [0.0, width - 1],
            [height - 1, 0.0],
            [height - 1, width - 1],
            [0.0, (width - 1) / 2],
            [height - 1, (width - 1) / 2],
            [(height - 1) / 2, 0.0],
            [(height - 1) / 2, width - 1],
        ],
        dtype=np.float64,
    )
    res = run_rdr2geo(
        geometry,
        points[:, 0],
        points[:, 1],
        dem,
        device=device,
    )
    lat = np.asarray(res.latitude_deg, dtype=np.float64)
    lon = np.asarray(res.longitude_deg, dtype=np.float64)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if int(ok.sum()) < 3:
        return None
    px = _lonlat_ring_to_grid_px(np.column_stack([lon[ok], lat[ok]]), grid)
    finite = np.isfinite(px).all(axis=1)
    px = px[finite]
    if px.shape[0] < 3:
        return None
    hull = ConvexHull(px)
    ring_px = px[hull.vertices]
    x0, dx, y0, dy = _grid_axes(grid)
    xs = (ring_px[:, 0] + 0.5) * dx + x0
    ys = y0 - (ring_px[:, 1] + 0.5) * (-dy)
    from pyproj import Transformer

    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    ring_lon, ring_lat = transformer.transform(xs, ys)
    return np.column_stack([np.asarray(ring_lon), np.asarray(ring_lat)])


def footprint_polygon_mask(
    grid: GeoGridSpec,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    footprint_lonlat: np.ndarray,
    dilate_px: int = 16,
) -> np.ndarray:
    """Return a bool mask of grid pixels inside the burst footprint polygon.

    The inside test and the dilation are evaluated on a context box derived
    only from the polygon bounding box, the dilation radius, and the grid —
    never on the request window — so the mask for a window is exactly the
    corresponding slice of the mask of any larger window. Cached LUT entries
    and fresh small-window builds therefore agree bitwise at the hull edge.

    Parameters
    ----------
    grid : GeoGridSpec
        Geographic grid.
    row0, row1, col0, col1 : int
        Bounding box to mask (half-open). This window only selects the
        returned slice of the window-independent mask.
    footprint_lonlat : numpy.ndarray
        ``(N, 2)`` polygon vertices in (lon, lat).
    dilate_px : int, optional
        Morphological dilation radius applied to the inside mask so edge
        pixels that lie just outside the axis-aligned burst footprint (due to
        DEM height or geometry rounding) are still processed and the crop
        stays lossless.

    Returns
    -------
    numpy.ndarray
        Boolean mask with shape ``(row1-row0, col1-col0)``.

    """
    from matplotlib.path import Path
    from pyproj import Transformer

    window = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    polygon_lonlat = np.asarray(footprint_lonlat, dtype=np.float64)
    if polygon_lonlat.size == 0:
        return window
    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    xs, ys = transformer.transform(
        polygon_lonlat[:, 0],
        polygon_lonlat[:, 1],
    )
    x0, dx, y0, dy = _grid_axes(grid)
    polygon = np.column_stack(
        [
            (np.asarray(xs) - x0) / dx - 0.5,
            (y0 - np.asarray(ys)) / (-dy) - 0.5,
        ]
    )
    dilate = int(dilate_px)
    # Context covering the polygon plus its full dilation halo, clipped only
    # to the grid. Clipping cannot truncate the halo: any grid pixel outside
    # the context is farther than ``dilate_px`` from the polygon bounding box,
    # so the context distance transform equals the full-grid one everywhere.
    context_row0 = max(0, int(np.floor(polygon[:, 1].min())) - dilate)
    context_row1 = min(
        int(grid.height),
        int(np.ceil(polygon[:, 1].max())) + 1 + dilate,
    )
    context_col0 = max(0, int(np.floor(polygon[:, 0].min())) - dilate)
    context_col1 = min(
        int(grid.width),
        int(np.ceil(polygon[:, 0].max())) + 1 + dilate,
    )
    path = Path(polygon)
    rows = np.arange(context_row0, context_row1)
    cols = np.arange(context_col0, context_col1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    pts = np.column_stack([cc.ravel(), rr.ravel()])
    mask = path.contains_points(pts).reshape(rr.shape)
    if dilate > 0:
        from scipy.ndimage import distance_transform_cdt

        distance = distance_transform_cdt(~mask, metric="chessboard")
        mask = distance <= float(dilate)
    intersect_row0 = max(row0, context_row0)
    intersect_row1 = min(row1, context_row1)
    intersect_col0 = max(col0, context_col0)
    intersect_col1 = min(col1, context_col1)
    if intersect_row0 < intersect_row1 and intersect_col0 < intersect_col1:
        window[
            intersect_row0 - row0 : intersect_row1 - row0,
            intersect_col0 - col0 : intersect_col1 - col0,
        ] = mask[
            intersect_row0 - context_row0 : intersect_row1 - context_row0,
            intersect_col0 - context_col0 : intersect_col1 - context_col0,
        ]
    return window


def _lonlat_ring_to_grid_px(
    ring_coords: np.ndarray,
    grid: GeoGridSpec,
) -> np.ndarray:
    """Convert an (N, 2) (lon, lat) ring to grid pixel coordinates."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    xs, ys = transformer.transform(ring_coords[:, 0], ring_coords[:, 1])
    x0, dx, y0, dy = _grid_axes(grid)
    return np.column_stack(
        [
            (np.asarray(xs) - x0) / dx - 0.5,
            (y0 - np.asarray(ys)) / (-dy) - 0.5,
        ]
    )


def polygon_parts(geometry: object) -> list[object]:
    """Return the Polygon members of a shapely geometry as a list."""
    geom_type = getattr(geometry, "geom_type", None)
    if geom_type == "Polygon":
        return [geometry]
    if geom_type == "MultiPolygon":
        return list(getattr(geometry, "geoms", []))
    if geom_type == "GeometryCollection":
        parts: list[object] = []
        for part in getattr(geometry, "geoms", []):
            parts.extend(polygon_parts(part))
        return parts
    return []


def roi_geo_bbox(
    roi_geometry: object,
    grid: GeoGridSpec,
    margin_px: int = 32,
) -> tuple[int, int, int, int]:
    """Return ``(row0, row1, col0, col1)`` bounding a shapely lon/lat ROI.

    Parameters
    ----------
    roi_geometry : shapely Polygon or MultiPolygon
        ROI polygon in (lon, lat) degrees.
    grid : GeoGridSpec
        Destination geographic grid.
    margin_px : int, optional
        Extra grid pixels added around the polygon bounds.

    Returns
    -------
    tuple[int, int, int, int]
        Half-open bounding box clipped to the grid.

    """
    rings = []
    for polygon in polygon_parts(roi_geometry):
        rings.append(np.asarray(polygon.exterior.coords))
        rings.extend(np.asarray(interior.coords) for interior in polygon.interiors)
    if not rings:
        return (0, grid.height, 0, grid.width)
    px = np.concatenate([_lonlat_ring_to_grid_px(ring, grid) for ring in rings])
    col0 = max(0, int(np.floor(np.min(px[:, 0]))) - margin_px)
    col1 = min(grid.width, int(np.ceil(np.max(px[:, 0]))) + 1 + margin_px)
    row0 = max(0, int(np.floor(np.min(px[:, 1]))) - margin_px)
    row1 = min(grid.height, int(np.ceil(np.max(px[:, 1]))) + 1 + margin_px)
    return (row0, row1, col0, col1)


def roi_geo_mask(
    roi_geometry: object,
    grid: GeoGridSpec,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    dilate_px: int = 0,
) -> np.ndarray:
    """Return a bool mask of grid pixels inside a shapely lon/lat ROI.

    Supports arbitrary polygons with holes and MultiPolygon geometries.

    Parameters
    ----------
    roi_geometry : shapely Polygon or MultiPolygon
        ROI polygon in (lon, lat) degrees.
    grid : GeoGridSpec
        Destination geographic grid.
    row0, row1, col0, col1 : int
        Bounding box to mask (half-open).
    dilate_px : int, optional
        Morphological dilation radius applied to the inside mask.

    Returns
    -------
    numpy.ndarray
        Boolean mask with shape ``(row1-row0, col1-col0)``.

    """
    from matplotlib.path import Path

    polygons = polygon_parts(roi_geometry)
    rows = np.arange(row0, row1)
    cols = np.arange(col0, col1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    pts = np.column_stack([cc.ravel(), rr.ravel()])
    inside = np.zeros(pts.shape[0], dtype=bool)
    for polygon in polygons:
        outer = _lonlat_ring_to_grid_px(np.asarray(polygon.exterior.coords), grid)
        poly_inside = Path(outer).contains_points(pts)
        for interior in polygon.interiors:
            hole = _lonlat_ring_to_grid_px(np.asarray(interior.coords), grid)
            poly_inside &= ~Path(hole).contains_points(pts)
        inside |= poly_inside
    mask = inside.reshape(rr.shape)
    if dilate_px > 0:
        from scipy.ndimage import distance_transform_cdt

        distance = distance_transform_cdt(~mask, metric="chessboard")
        mask = distance <= float(dilate_px)
    return mask


def grid_lonlat(grid: GeoGridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return geographic coordinates at destination pixel centers."""
    return grid_lonlat_rows(grid, 0, grid.height)


def grid_lonlat_rows(
    grid: GeoGridSpec,
    row_start: int,
    row_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return geographic pixel-centre coordinates for selected grid rows.

    Parameters
    ----------
    grid : GeoGridSpec
        Destination geographic grid.
    row_start, row_stop : int
        Half-open destination row interval.

    Returns
    -------
    latitude, longitude : tuple[numpy.ndarray, numpy.ndarray]
        Coordinate arrays with shape ``(row_stop - row_start, grid.width)``.

    """
    from pyproj import Transformer

    if row_start < 0 or row_stop > grid.height or row_start >= row_stop:
        reject_invalid_state("invalid geographic grid row interval")
    x0, dx, y0, dy = _grid_axes(grid)
    x_coordinates = x0 + dx * (0.5 + np.arange(grid.width, dtype=np.float64))
    y_coordinates = y0 + dy * (0.5 + np.arange(row_start, row_stop, dtype=np.float64))
    x, y = np.meshgrid(x_coordinates, y_coordinates)
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(x.ravel(), y.ravel())
    return (
        np.asarray(latitude, dtype=np.float64).reshape(x.shape),
        np.asarray(longitude, dtype=np.float64).reshape(x.shape),
    )


_LUT_CACHE_ARRAY_FILES = (
    "reference_azimuth.float64",
    "reference_range.float64",
    "reference_valid.bool",
    "height.float64",
)
_LUT_CACHE_ARRAY_DTYPES = {
    "reference_azimuth.float64": np.float64,
    "reference_range.float64": np.float64,
    "reference_valid.bool": np.bool_,
    "height.float64": np.float64,
}
_LUT_CACHE_META_NAME = "meta.json"
# Schema pins every convention the stored content depends on: the pixel-centre
# coordinate convention, the converged/in-bounds validity rule, the
# window-independent eight-point hull prefilter constants and its
# generation-per-entry publish layout, the float64/bool array dtypes, and the
# fallback-filled mean-height rule used to recompute ``height_m`` on load.
_LUT_CACHE_SCHEMA = "faninsar.geo2rdr-lut.v3:8pt-hull-v2:fallback-filled-mean-v1"
# geo2rdr solver settings fixed by this builder, passed explicitly to
# ``run_geo2rdr``, and folded into the content key so changed defaults can
# never silently hit a stale entry. The values here are exactly what the
# solver runs with on every (re)build.
_LUT_GEO2RDR_SOLVER = {
    "max_iter": 20,
    "extra_iter": 0,
    "range_tol_m": 0.01,
    "doppler_tol_hz": 0.1,
}
_LUT_WINDOW_DIR_PATTERN = re.compile(r"^w(\d+)-(\d+)-(\d+)-(\d+)$")
_LUT_GENERATION_DIR_PATTERN = re.compile(r"^g(\d+)$")


def _canonical_digest(payload: object) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible values."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_cache_component(value: str) -> str:
    """Sanitize one caller-supplied path component (readable namespace)."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _window_dir_name(window: tuple[int, int, int, int]) -> str:
    """Return the entry subdirectory name for a ``(row0, col0, row1, col1)`` window."""
    row0, col0, row1, col1 = (int(value) for value in window)
    return f"w{row0}-{col0}-{row1}-{col1}"


def _parse_window_dir_name(name: str) -> tuple[int, int, int, int] | None:
    """Parse a ``w{row0}-{col0}-{row1}-{col1}`` entry name, or return ``None``."""
    match = _LUT_WINDOW_DIR_PATTERN.match(name)
    if match is None:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )


def _generation_dir(entry: Path, generation: int) -> Path:
    """Return the array directory of one entry generation."""
    return entry / f"g{int(generation)}"


def _referenced_generation(entry: Path) -> int:
    """Return the highest generation any on-disk state could reference.

    Scans both the published ``meta.json`` and the generation directories so
    a partially written generation from a crashed publisher is never reused:
    the next publish always allocates a strictly higher number.
    """
    highest = 0
    try:
        published = json.loads((entry / _LUT_CACHE_META_NAME).read_text())
        generation = (
            published.get("generation") if isinstance(published, dict) else None
        )
        if isinstance(generation, int) and not isinstance(generation, bool):
            highest = max(highest, int(generation))
    except (OSError, ValueError, TypeError):
        pass
    if entry.is_dir():
        for child in entry.iterdir():
            match = _LUT_GENERATION_DIR_PATTERN.match(child.name)
            if match is not None and child.is_dir():
                highest = max(highest, int(match.group(1)))
    return highest


def _lut_identity_payload(
    *,
    geometry: RadarGeometryModel,
    grid: GeoGridSpec,
    full_radar_shape: tuple[int, int],
    height_m: float | np.ndarray,
    dem: DEM | None,
    footprint_lonlat: np.ndarray | None,
    footprint_dilate_px: int,
) -> tuple[dict[str, object], str]:
    """Return the full content identity and its digest for one LUT build.

    The payload covers the schema version (coordinate convention, validity
    rule, hull constants, dtypes, height rule), the geometry model and orbit
    content, the DEM content identity, the fallback height, the grid CRS and
    full-precision affine coefficients (shape excluded on purpose), the
    pinned solver settings, the full radar shape, and the footprint-hull
    prefilter marker. Request-specific choices (ROI polygon, ROI dilation,
    chunking, and the destination window) are deliberately excluded: masks
    are overlaid at load time and windows slice existing entries.
    """
    from faninsar.processing.geometry.api import dem_content_identity
    from faninsar.processing.geometry.torch_backends_v2 import (
        _array_digest,
        _model_digests,
    )

    model_digest, orbit_digest = _model_digests(geometry)
    if np.isscalar(height_m):
        fallback_identity: dict[str, object] = {
            "mode": "scalar",
            "height_m": float(height_m),
        }
    else:
        fallback_identity = {
            "mode": "array",
            "digest": _array_digest(np.asarray(height_m, dtype=np.float64)),
        }
    prefilter: dict[str, object] | None = None
    if footprint_lonlat is not None:
        prefilter = {
            "polygon": _array_digest(np.asarray(footprint_lonlat, dtype=np.float64)),
            "dilate_px": int(footprint_dilate_px),
        }
    payload: dict[str, object] = {
        "schema": _LUT_CACHE_SCHEMA,
        "geometry_model": model_digest,
        "geometry_orbit": orbit_digest,
        "dem": dem_content_identity(dem),
        "fallback_height": fallback_identity,
        "grid": {
            "crs": str(grid.crs),
            "transform": [float(value) for value in tuple(grid.transform)[:6]],
        },
        "solver": dict(_LUT_GEO2RDR_SOLVER),
        "full_radar_shape": [int(full_radar_shape[0]), int(full_radar_shape[1])],
        "footprint_prefilter": prefilter,
    }
    return payload, _canonical_digest(payload)


def _lut_entry_meta(
    *,
    key_digest: str,
    identity: dict[str, object],
    window: tuple[int, int, int, int],
    full_radar_shape: tuple[int, int],
    fallback_height: float,
) -> dict[str, object]:
    """Return the published metadata for one cache entry."""
    return {
        "schema": _LUT_CACHE_SCHEMA,
        "key_digest": key_digest,
        "identity": identity,
        "window": [int(value) for value in window],
        "full_radar_shape": [
            int(full_radar_shape[0]),
            int(full_radar_shape[1]),
        ],
        "fallback_height": float(fallback_height),
    }


def _find_cached_lut_entry(
    key_dir: Path,
    window: tuple[int, int, int, int],
) -> Path | None:
    """Return the tightest entry directory whose window contains ``window``."""
    if not key_dir.is_dir():
        return None
    row0, col0, row1, col1 = window
    best: Path | None = None
    best_area: int | None = None
    for child in sorted(key_dir.iterdir()):
        entry_window = _parse_window_dir_name(child.name)
        if entry_window is None or not child.is_dir():
            continue
        entry_row0, entry_col0, entry_row1, entry_col1 = entry_window
        if (
            entry_row0 <= row0
            and entry_col0 <= col0
            and entry_row1 >= row1
            and entry_col1 >= col1
        ):
            area = (entry_row1 - entry_row0) * (entry_col1 - entry_col0)
            if best_area is None or area < best_area:
                best = child
                best_area = area
    return best


def _load_cached_lut_entry(
    entry: Path,
    *,
    key_digest: str,
    identity: dict[str, object],
    full_radar_shape: tuple[int, int],
) -> tuple[dict[str, object], dict[str, np.memmap]] | None:
    """Load one entry read-only when its metadata validates; else ``None``.

    Corrupted metadata (including ``JSONDecodeError``, a ``ValueError``
    subclass), a digest/identity disagreement with the recomputed identity,
    and any array disagreement — including a generation directory whose
    files were cleaned up or never finished publishing — are treated as a
    cache miss so the caller rebuilds and republishes the entry. The reader
    opens only the files of the one generation named by the meta it read, so
    it can never mix arrays across generations.
    """
    try:
        meta = json.loads((entry / _LUT_CACHE_META_NAME).read_text())
        if not isinstance(meta, dict):
            reason = "meta is not a JSON object"
        elif meta.get("schema") != _LUT_CACHE_SCHEMA:
            reason = "schema mismatch"
        elif meta.get("key_digest") != key_digest:
            reason = "key_digest mismatch"
        elif meta.get("identity") != identity:
            reason = "stored identity mismatch"
        elif tuple(
            int(value) for value in meta.get("window", ())
        ) != _parse_window_dir_name(entry.name):
            reason = "window mismatch"
        elif tuple(int(value) for value in meta.get("full_radar_shape", ())) != (
            int(full_radar_shape[0]),
            int(full_radar_shape[1]),
        ):
            reason = "full_radar_shape mismatch"
        elif (
            not isinstance(meta.get("generation"), int)
            or isinstance(meta.get("generation"), bool)
            or int(meta["generation"]) < 1
        ):
            reason = "invalid generation"
        else:
            reason = None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # json.JSONDecodeError is a ValueError: truncated or corrupt entries
        # are misses, never exceptions escaping into the build path.
        logger.warning(
            "geo2rdr LUT cache entry %s unreadable (%s); treating as miss",
            entry,
            exc,
        )
        return None
    if reason is not None:
        logger.warning(
            "geo2rdr LUT cache entry %s unusable (%s); rebuilding",
            entry,
            reason,
        )
        return None
    stored_window = tuple(int(value) for value in meta["window"])
    rows = stored_window[2] - stored_window[0]
    cols = stored_window[3] - stored_window[1]
    generation_dir = _generation_dir(entry, int(meta["generation"]))
    arrays: dict[str, np.memmap] = {}
    for name in _LUT_CACHE_ARRAY_FILES:
        path = generation_dir / name
        expected_bytes = rows * cols * np.dtype(_LUT_CACHE_ARRAY_DTYPES[name]).itemsize
        try:
            if path.stat().st_size != expected_bytes:
                logger.warning(
                    "geo2rdr LUT cache entry %s unusable (array size mismatch: %s); "
                    "rebuilding",
                    entry,
                    name,
                )
                return None
            arrays[name] = np.memmap(
                path,
                mode="r",
                dtype=_LUT_CACHE_ARRAY_DTYPES[name],
                shape=(rows, cols),
            )
        except OSError as exc:
            logger.warning(
                "geo2rdr LUT cache entry %s unreadable (%s); treating as miss",
                entry,
                exc,
            )
            return None
    return meta, arrays


def _publish_lut_entry(
    entry: Path,
    arrays: dict[str, np.ndarray],
    meta: dict[str, object],
) -> None:
    """Publish one cache entry as a new generation; ``meta.json`` lands last.

    Arrays are written into a fresh ``g<N>`` generation directory that no
    ``meta.json`` has ever referenced, so a concurrent reader can never see
    partially replaced arrays: it either reads the previous generation named
    by the old meta or the new one named by the new meta, both complete.
    The generation number is allocated strictly above every number that any
    on-disk meta or directory ever used, so a published generation is never
    overwritten. Superseded generations are removed best-effort, keeping the
    immediately previous one as a grace window for readers that read the old
    meta but have not opened its files yet; POSIX readers holding open mmaps
    of deleted files keep reading them, and readers that lose the race get a
    miss and rebuild.
    """
    entry.mkdir(parents=True, exist_ok=True)
    generation = _referenced_generation(entry) + 1
    generation_dir = _generation_dir(entry, generation)
    token = f".tmp-{os.getpid()}-{time.time_ns()}"
    created = False
    try:
        generation_dir.mkdir()
        created = True
        for name, array in arrays.items():
            np.ascontiguousarray(array).tofile(generation_dir / name)
        published = {**meta, "generation": int(generation)}
        temporary = entry / f"{_LUT_CACHE_META_NAME}{token}"
        temporary.write_text(json.dumps(published, sort_keys=True))
        temporary.replace(entry / _LUT_CACHE_META_NAME)
    except BaseException:
        if created:
            shutil.rmtree(generation_dir, ignore_errors=True)
        (entry / f"{_LUT_CACHE_META_NAME}{token}").unlink(missing_ok=True)
        raise
    # Best-effort retention of the current and immediately previous
    # generations; failures never fail the publish.
    keep = {generation, generation - 1}
    for child in entry.iterdir():
        match = _LUT_GENERATION_DIR_PATTERN.match(child.name)
        if match is None or not child.is_dir():
            continue
        if int(match.group(1)) in keep:
            continue
        shutil.rmtree(child, ignore_errors=True)


def _allocate_lut_arrays(
    storage_dir: Path | None,
    crop_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate fresh writable LUT arrays, memmap-backed when ``storage_dir`` is set."""
    if storage_dir is None:
        return (
            np.full(crop_shape, np.nan, dtype=np.float64),
            np.full(crop_shape, np.nan, dtype=np.float64),
            np.zeros(crop_shape, dtype=bool),
            np.full(crop_shape, np.nan, dtype=np.float64),
        )
    directory = Path(storage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return (
        np.memmap(
            directory / "reference_azimuth.float64",
            mode="w+",
            dtype=np.float64,
            shape=crop_shape,
        ),
        np.memmap(
            directory / "reference_range.float64",
            mode="w+",
            dtype=np.float64,
            shape=crop_shape,
        ),
        np.memmap(
            directory / "reference_valid.bool",
            mode="w+",
            dtype=np.bool_,
            shape=crop_shape,
        ),
        np.memmap(
            directory / "height.float64",
            mode="w+",
            dtype=np.float64,
            shape=crop_shape,
        ),
    )


def _apply_request_mask(
    azimuth: np.ndarray,
    range_index: np.ndarray,
    valid: np.ndarray,
    height_lookup: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Overlay a request-specific validity mask on LUT arrays in place."""
    outside = ~np.asarray(mask, dtype=bool)
    azimuth[outside] = np.nan
    range_index[outside] = np.nan
    height_lookup[outside] = np.nan
    valid &= mask


def _filled_mean_height(height_lookup: np.ndarray, fallback_height: float) -> float:
    """Mean height after fallback-filling unsampled pixels (schema-pinned rule)."""
    if height_lookup.size == 0:
        return float(fallback_height)
    filled = np.where(np.isfinite(height_lookup), height_lookup, fallback_height)
    return float(np.mean(filled))


def build_geo2rdr_lut(
    *,
    geometry: RadarGeometryModel,
    grid: GeoGridSpec,
    full_radar_shape: tuple[int, int],
    height_m: float | np.ndarray = 0.0,
    dem: DEM | None = None,
    chunk_size: int = 128,
    storage_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cache_key: str | None = None,
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
    footprint_lonlat: np.ndarray | None = None,
    roi_geometry: object | None = None,
    polygon_dilate_px: int = 64,
    footprint_dilate_px: int = 64,
    device: DeviceLike,
    timings_out: dict[str, float] | None = None,
) -> Geo2RdrLUT:
    """Build a reusable geographic-to-radar lookup table.

    The disk cache is content-addressed: ``cache_key`` is only a readable
    namespace, while the on-disk identity is derived inside this builder
    from the schema version, geometry model and orbit content, DEM content,
    fallback height, grid CRS and full-precision affine coefficients, pinned
    solver settings, ``full_radar_shape``, and the footprint-hull prefilter.
    The ROI polygon and its dilation are request-specific and never enter
    the identity: entries are solved without them and the mask is overlaid
    on the returned arrays at load time. Entries are stored per destination
    window; a request whose window is contained in an existing entry is
    served by slicing that entry, and a non-contained request publishes a
    new entry instead of overwriting.

    Parameters
    ----------
    geometry : RadarGeometryModel
        Reference-scene radar geometry.
    grid : GeoGridSpec
        Destination geographic grid.
    full_radar_shape : tuple[int, int]
        Full-resolution radar image shape.
    height_m : float or numpy.ndarray, optional
        Fallback ellipsoidal height.
    dem : DEM, optional
        Per-pixel ellipsoidal height source.
    device : DeviceLike
        Required production device (``auto`` resolves to cpu or cuda).
    chunk_size : int, optional
        Destination rows processed per geometry call.
    storage_dir : str or pathlib.Path, optional
        Directory for disk-backed LUT arrays. In-memory arrays are used when
        omitted. Required when caching is enabled: the returned arrays always
        live in this directory and cache entries are published as copies, so
        work files and cache files never share storage.
    cache_dir : str or pathlib.Path, optional
        Shared directory for reusable LUT caches. Requires ``cache_key``.
    cache_key : str, optional
        Readable namespace (scene, swath, burst). Sanitized into one path
        component; the entry identity itself is content-derived.
    row_range : tuple[int, int], optional
        Half-open row interval of the geographic grid to process. Rows outside
        are left NaN/invalid. When omitted the full grid is processed.
    col_range : tuple[int, int], optional
        Half-open column interval to process. Columns outside are left
        NaN/invalid.
    footprint_lonlat : numpy.ndarray, optional
        ``(N, 2)`` burst footprint polygon in (lon, lat). Pixels outside the
        polygon are skipped without calling geo2rdr. This hull prefilter is
        part of the cached content and its digest plus ``footprint_dilate_px``
        enter the identity.
    roi_geometry : shapely Polygon or MultiPolygon, optional
        ROI polygon in (lon, lat). When given (and ``footprint_lonlat`` is
        omitted) it is overlaid on the returned arrays as a request mask and
        is excluded from the cache identity.
    polygon_dilate_px : int, optional
        Dilation radius applied to the request ROI mask.
    footprint_dilate_px : int, optional
        Dilation applied to the eight-point hull mask; part of the identity.
    timings_out : dict, optional
        Exclusive LUT substages accumulated in seconds (``polygon_mask``,
        ``dem_sample``, ``geo2rdr``, ``store``). Nested observations of the
        parent ``geo2rdr_lut`` wall, not an independent public-call boundary.

    Returns
    -------
    Geo2RdrLUT
        Full-resolution radar coordinates on the destination grid.

    """
    row0, row1 = (
        (0, grid.height)
        if row_range is None
        else (int(row_range[0]), int(row_range[1]))
    )
    col0, col1 = (
        (0, grid.width) if col_range is None else (int(col_range[0]), int(col_range[1]))
    )
    row0 = max(0, min(row0, grid.height))
    row1 = max(row0, min(row1, grid.height))
    col0 = max(0, min(col0, grid.width))
    col1 = max(col0, min(col1, grid.width))

    full_height = int(full_radar_shape[0])
    full_width = int(full_radar_shape[1])
    crop_shape = (row1 - row0, col1 - col0)
    full_crop = (row_range is not None) or (col_range is not None)
    window = (row0, col0, row1, col1)

    if (cache_dir is None) != (cache_key is None):
        reject_invalid_state("geo2rdr LUT cache requires both cache_dir and cache_key")
    if cache_dir is not None and storage_dir is None:
        reject_invalid_state(
            "geo2rdr LUT cache requires storage_dir for downstream reopen"
        )

    fallback_height = (
        float(height_m) if np.isscalar(height_m) else float(np.nanmean(height_m))
    )
    height_array = None if np.isscalar(height_m) else np.asarray(height_m)
    if height_array is not None and height_array.shape != crop_shape:
        reject_invalid_state(
            f"height_m shape {height_array.shape} does not match crop {crop_shape}"
        )

    # The ROI polygon is request-specific: it never bakes into the cached
    # solve and is overlaid on the returned arrays on both the fresh and the
    # cached path. The footprint hull stays inside the solve because it is
    # content-determined and folded into the identity above.
    mask_started = time.perf_counter()
    request_mask: np.ndarray | None = None
    if footprint_lonlat is None and roi_geometry is not None:
        request_mask = roi_geo_mask(
            roi_geometry,
            grid,
            row0,
            row1,
            col0,
            col1,
            dilate_px=polygon_dilate_px,
        )
    _add_timing(timings_out, "polygon_mask", mask_started)

    key_dir: Path | None = None
    if cache_dir is not None:
        assert cache_key is not None
        identity, key_digest = _lut_identity_payload(
            geometry=geometry,
            grid=grid,
            full_radar_shape=(full_height, full_width),
            height_m=height_m,
            dem=dem,
            footprint_lonlat=footprint_lonlat,
            footprint_dilate_px=footprint_dilate_px,
        )
        key_dir = Path(cache_dir) / _safe_cache_component(str(cache_key)) / key_digest
        entry = _find_cached_lut_entry(key_dir, window)
        if entry is not None:
            loaded = _load_cached_lut_entry(
                entry,
                key_digest=key_digest,
                identity=identity,
                full_radar_shape=(full_height, full_width),
            )
            if loaded is not None:
                meta, stored = loaded
                stored_window = [
                    int(value)
                    for value in meta["window"]  # type: ignore[union-attr]
                ]
                row_slice = slice(row0 - stored_window[0], row1 - stored_window[0])
                col_slice = slice(col0 - stored_window[1], col1 - stored_window[1])
                azimuth, range_index, valid, height_lookup = _allocate_lut_arrays(
                    None if storage_dir is None else Path(storage_dir),
                    crop_shape,
                )
                azimuth[:] = stored["reference_azimuth.float64"][row_slice, col_slice]
                range_index[:] = stored["reference_range.float64"][row_slice, col_slice]
                valid[:] = stored["reference_valid.bool"][row_slice, col_slice]
                height_lookup[:] = stored["height.float64"][row_slice, col_slice]
                del stored
                if request_mask is not None:
                    _apply_request_mask(
                        azimuth, range_index, valid, height_lookup, request_mask
                    )
                for array in (azimuth, range_index, valid, height_lookup):
                    if isinstance(array, np.memmap):
                        array.flush()
                mean_height = _filled_mean_height(height_lookup, fallback_height)
                logger.info(
                    "Reused cached geo2rdr LUT %s: %d/%d valid, radar_shape=%s",
                    key_digest[:12],
                    int(np.asarray(valid).sum()),
                    valid.size,
                    (full_height, full_width),
                )
                return Geo2RdrLUT(
                    az_full=azimuth,
                    rg_full=range_index,
                    valid=valid,
                    full_radar_shape=(full_height, full_width),
                    height_m=mean_height,
                    height_full=height_lookup,
                    row0=row0 if full_crop else 0,
                    col0=col0 if full_crop else 0,
                )

    azimuth, range_index, valid, height_lookup = _allocate_lut_arrays(
        None if storage_dir is None else Path(storage_dir),
        crop_shape,
    )

    mask_started = time.perf_counter()
    hull_mask: np.ndarray | None = None
    if footprint_lonlat is not None:
        hull_mask = footprint_polygon_mask(
            grid,
            row0,
            row1,
            col0,
            col1,
            np.asarray(footprint_lonlat),
            dilate_px=footprint_dilate_px,
        )
    _add_timing(timings_out, "polygon_mask", mask_started)
    for row_start in range(row0, row1, chunk_size):
        row_stop = min(row_start + chunk_size, row1)
        latitude_chunk, longitude_chunk = grid_lonlat_rows(
            grid,
            row_start,
            row_stop,
        )
        finite_geo = np.isfinite(latitude_chunk) & np.isfinite(longitude_chunk)
        if col0 > 0 or col1 < grid.width:
            latitude_chunk = latitude_chunk[:, col0:col1]
            longitude_chunk = longitude_chunk[:, col0:col1]
            finite_geo = finite_geo[:, col0:col1]
        grid_ok = finite_geo.copy()
        if hull_mask is not None:
            poly_rows = slice(row_start - row0, row_stop - row0)
            finite_geo &= hull_mask[poly_rows, :]
        safe_latitude = np.where(finite_geo, latitude_chunk, 0.0)
        safe_longitude = np.where(finite_geo, longitude_chunk, 0.0)
        dem_started = time.perf_counter()
        if dem is not None:
            sampled_height = np.full(finite_geo.shape, np.nan, dtype=np.float64)
            sample_mask = finite_geo if hull_mask is not None else grid_ok
            if sample_mask.any():
                packed = np.asarray(
                    dem.sample(
                        latitude_chunk[sample_mask],
                        longitude_chunk[sample_mask],
                    ),
                    dtype=np.float64,
                )
                sampled_height[sample_mask] = packed
            raw_height = sampled_height
            height_chunk = np.where(
                np.isfinite(sampled_height),
                sampled_height,
                fallback_height,
            )
        elif height_array is not None:
            selected_height = np.asarray(
                height_array[row_start:row_stop],
                dtype=np.float64,
            )
            raw_height = selected_height
            height_chunk = np.where(
                np.isfinite(selected_height),
                selected_height,
                fallback_height,
            )
        else:
            raw_height = fallback_height
            height_chunk = fallback_height
        height_lookup[row_start - row0 : row_stop - row0, :] = raw_height
        _add_timing(timings_out, "dem_sample", dem_started)

        geo2rdr_started = time.perf_counter()
        result = run_geo2rdr(
            geometry,
            safe_latitude.reshape(-1),
            safe_longitude.reshape(-1),
            (
                np.asarray(height_chunk).reshape(-1)
                if not np.isscalar(height_chunk)
                else np.full(finite_geo.shape, height_chunk).reshape(-1)
            ),
            device=device,
            max_iter=_LUT_GEO2RDR_SOLVER["max_iter"],
            extra_iter=_LUT_GEO2RDR_SOLVER["extra_iter"],
            range_tol_m=_LUT_GEO2RDR_SOLVER["range_tol_m"],
            doppler_tol_hz=_LUT_GEO2RDR_SOLVER["doppler_tol_hz"],
        )
        _add_timing(timings_out, "geo2rdr", geo2rdr_started)
        result_converged = np.asarray(result.converged).reshape(finite_geo.shape)
        result_azimuth = np.asarray(result.azimuth_index).reshape(finite_geo.shape)
        result_range = np.asarray(result.range_index).reshape(finite_geo.shape)
        chunk_valid = (
            finite_geo
            & result_converged
            & np.isfinite(result_azimuth)
            & np.isfinite(result_range)
            & (result_azimuth >= 0.0)
            & (result_azimuth <= full_height - 1.0)
            & (result_range >= 0.0)
            & (result_range <= full_width - 1.0)
        )
        azimuth[row_start - row0 : row_stop - row0, :] = np.where(
            chunk_valid,
            result_azimuth,
            np.nan,
        )
        range_index[row_start - row0 : row_stop - row0, :] = np.where(
            chunk_valid,
            result_range,
            np.nan,
        )
        valid[row_start - row0 : row_stop - row0, :] = chunk_valid
        store_started = time.perf_counter()
        for array in (azimuth, range_index, valid, height_lookup):
            if isinstance(array, np.memmap):
                release_memmap_pages(array)
        _add_timing(timings_out, "store", store_started)

    flush_started = time.perf_counter()
    for array in (azimuth, range_index, valid, height_lookup):
        if isinstance(array, np.memmap):
            array.flush()
    if key_dir is not None:
        assert storage_dir is not None
        entry = key_dir / _window_dir_name(window)
        entry_meta = _lut_entry_meta(
            key_digest=key_digest,
            identity=identity,
            window=window,
            full_radar_shape=(full_height, full_width),
            fallback_height=fallback_height,
        )
        try:
            _publish_lut_entry(
                entry,
                {
                    "reference_azimuth.float64": azimuth,
                    "reference_range.float64": range_index,
                    "reference_valid.bool": valid,
                    "height.float64": height_lookup,
                },
                entry_meta,
            )
        except OSError as exc:
            logger.warning(
                "geo2rdr LUT cache publish %s failed (%s); continuing uncached",
                entry,
                exc,
            )
    _add_timing(timings_out, "store", flush_started)
    if request_mask is not None:
        _apply_request_mask(azimuth, range_index, valid, height_lookup, request_mask)
    mean_height = _filled_mean_height(height_lookup, fallback_height)
    logger.info(
        "Built geo2rdr LUT: %d/%d valid, radar_shape=%s, mean_height=%.1f m",
        int(np.asarray(valid).sum()),
        valid.size,
        (full_height, full_width),
        mean_height,
    )
    return Geo2RdrLUT(
        az_full=azimuth,
        rg_full=range_index,
        valid=valid,
        full_radar_shape=(full_height, full_width),
        height_m=mean_height,
        height_full=height_lookup,
        row0=0 if not full_crop else row0,
        col0=0 if not full_crop else col0,
    )
