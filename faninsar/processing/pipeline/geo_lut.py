"""Build geographic-to-radar lookup tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry import geo2rdr
from faninsar.processing.memory import release_memmap_pages

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.processing.geometry import RadarGeometryModel
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.merge.grid import GeoGridSpec

logger = setup_logger(__name__)

__all__ = [
    "Geo2RdrLUT",
    "build_geo2rdr_lut",
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
    dem: DEMSampler | None = None,
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
    dem : DEMSampler, optional
        DEM used by rdr2geo. When omitted an ellipsoid is used.

    Returns
    -------
    tuple[int, int, int, int]
        ``(row0, row1, col0, col1)`` half-open bounding box.

    """
    from faninsar.processing.geometry import rdr2geo_ellipsoid, rdr2geo_with_dem

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
    if dem is not None:
        res = rdr2geo_with_dem(
            geometry,
            points[:, 0],
            points[:, 1],
            dem,
            height_seed_m=0.0,
        )
        lat = np.asarray(res.latitude_deg)
        lon = np.asarray(res.longitude_deg)
        ok = np.isfinite(lat) & np.isfinite(lon)
    else:
        res = rdr2geo_ellipsoid(
            geometry,
            points[:, 0],
            points[:, 1],
            height_m=0.0,
        )
        lat = np.asarray(res.latitude_deg)
        lon = np.asarray(res.longitude_deg)
        ok = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(ok):
        return (0, grid.height, 0, grid.width)

    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    xs, ys = transformer.transform(lon[ok], lat[ok])
    x0, dx, _, y0, _, dy = grid.transform
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
    dem: DEMSampler | None,
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
    dem : DEMSampler, optional
        DEM for rdr2geo; when omitted an ellipsoid is used.

    Returns
    -------
    numpy.ndarray or None
        ``(N, 2)`` ring of (lon, lat) corners (``N >= 3``) or None.

    """
    from faninsar.processing.geometry import rdr2geo_ellipsoid, rdr2geo_with_dem

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
    if dem is not None:
        res = rdr2geo_with_dem(
            geometry,
            points[:, 0],
            points[:, 1],
            dem,
            height_seed_m=0.0,
        )
        lat = np.asarray(res.latitude_deg, dtype=np.float64)
        lon = np.asarray(res.longitude_deg, dtype=np.float64)
        ok = np.isfinite(lat) & np.isfinite(lon)
        if not np.all(ok):
            fallback = rdr2geo_ellipsoid(
                geometry,
                points[~ok, 0],
                points[~ok, 1],
                height_m=0.0,
            )
            lat[~ok] = np.asarray(fallback.latitude_deg, dtype=np.float64)
            lon[~ok] = np.asarray(fallback.longitude_deg, dtype=np.float64)
            ok = np.isfinite(lat) & np.isfinite(lon)
    else:
        res = rdr2geo_ellipsoid(
            geometry,
            points[:, 0],
            points[:, 1],
            height_m=0.0,
        )
        lat = np.asarray(res.latitude_deg, dtype=np.float64)
        lon = np.asarray(res.longitude_deg, dtype=np.float64)
        ok = np.isfinite(lat) & np.isfinite(lon)
    if int(ok.sum()) < 3:
        return None
    return np.column_stack([lon[ok], lat[ok]])


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

    Parameters
    ----------
    grid : GeoGridSpec
        Geographic grid.
    row0, row1, col0, col1 : int
        Bounding box to mask (half-open).
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

    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    xs, ys = transformer.transform(
        footprint_lonlat[:, 0],
        footprint_lonlat[:, 1],
    )
    x0, dx, _, y0, _, dy = grid.transform
    polygon = np.column_stack(
        [
            (np.asarray(xs) - x0) / dx - 0.5,
            (y0 - np.asarray(ys)) / (-dy) - 0.5,
        ]
    )
    path = Path(polygon)
    rows = np.arange(row0, row1)
    cols = np.arange(col0, col1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    pts = np.column_stack([cc.ravel(), rr.ravel()])
    inside = path.contains_points(pts)
    mask = inside.reshape(rr.shape)
    if dilate_px > 0:
        from scipy.ndimage import distance_transform_cdt

        distance = distance_transform_cdt(~mask, metric="chessboard")
        mask = distance <= float(dilate_px)
    return mask


def _lonlat_ring_to_grid_px(
    ring_coords: np.ndarray,
    grid: GeoGridSpec,
) -> np.ndarray:
    """Convert an (N, 2) (lon, lat) ring to grid pixel coordinates."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    xs, ys = transformer.transform(ring_coords[:, 0], ring_coords[:, 1])
    x0, dx, _, y0, _, dy = grid.transform
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
    x0, dx, _, y0, _, dy = grid.transform
    x_coordinates = x0 + dx * (0.5 + np.arange(grid.width, dtype=np.float64))
    y_coordinates = y0 + dy * (0.5 + np.arange(row_start, row_stop, dtype=np.float64))
    x, y = np.meshgrid(x_coordinates, y_coordinates)
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(x.ravel(), y.ravel())
    return (
        np.asarray(latitude, dtype=np.float64).reshape(x.shape),
        np.asarray(longitude, dtype=np.float64).reshape(x.shape),
    )


def build_geo2rdr_lut(
    *,
    geometry: RadarGeometryModel,
    grid: GeoGridSpec,
    full_radar_shape: tuple[int, int],
    height_m: float | np.ndarray = 0.0,
    dem: DEMSampler | None = None,
    chunk_size: int = 128,
    storage_dir: str | Path | None = None,
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
    footprint_lonlat: np.ndarray | None = None,
    roi_geometry: object | None = None,
    polygon_dilate_px: int = 64,
) -> Geo2RdrLUT:
    """Build a reusable geographic-to-radar lookup table.

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
    dem : DEMSampler, optional
        Per-pixel ellipsoidal height source.
    chunk_size : int, optional
        Destination rows processed per geometry call.
    storage_dir : str or pathlib.Path, optional
        Directory for disk-backed LUT arrays. In-memory arrays are used when
        omitted.
    row_range : tuple[int, int], optional
        Half-open row interval of the geographic grid to process. Rows outside
        are left NaN/invalid. When omitted the full grid is processed.
    col_range : tuple[int, int], optional
        Half-open column interval to process. Columns outside are left
        NaN/invalid.
    footprint_lonlat : numpy.ndarray, optional
        ``(N, 2)`` burst footprint polygon in (lon, lat). Pixels outside the
        polygon are skipped without calling geo2rdr, saving most of the
        wasted iterations in the bbox corners.
    roi_geometry : shapely Polygon or MultiPolygon, optional
        ROI polygon in (lon, lat). When given (and ``footprint_lonlat`` is
        omitted) it replaces the footprint polygon as the prefilter mask,
        allowing arbitrary polygons with holes and multi-part ROIs.
    polygon_dilate_px : int, optional
        Dilation radius applied to the footprint mask so edge pixels stay
        inside the crop and the optimization remains lossless.

    Returns
    -------
    Geo2RdrLUT
        Full-resolution radar coordinates on the destination grid.

    """
    from pathlib import Path

    row0, row1 = (
        (0, grid.height)
        if row_range is None
        else (int(row_range[0]), int(row_range[1]))
    )
    col0, col1 = (
        (0, grid.width)
        if col_range is None
        else (int(col_range[0]), int(col_range[1]))
    )
    row0 = max(0, min(row0, grid.height))
    row1 = max(row0, min(row1, grid.height))
    col0 = max(0, min(col0, grid.width))
    col1 = max(col0, min(col1, grid.width))

    full_height, full_width = full_radar_shape
    crop_shape = (row1 - row0, col1 - col0)
    full_crop = (row_range is not None) or (col_range is not None)
    if storage_dir is None:
        azimuth = np.full(crop_shape, np.nan, dtype=np.float64)
        range_index = np.full(crop_shape, np.nan, dtype=np.float64)
        valid = np.zeros(crop_shape, dtype=bool)
        height_lookup = np.full(crop_shape, np.nan, dtype=np.float64)
    else:
        directory = Path(storage_dir)
        directory.mkdir(parents=True, exist_ok=True)
        azimuth = np.memmap(
            directory / "reference_azimuth.float64",
            mode="w+",
            dtype=np.float64,
            shape=crop_shape,
        )
        range_index = np.memmap(
            directory / "reference_range.float64",
            mode="w+",
            dtype=np.float64,
            shape=crop_shape,
        )
        valid = np.memmap(
            directory / "reference_valid.bool",
            mode="w+",
            dtype=np.bool_,
            shape=crop_shape,
        )
        height_lookup = np.memmap(
            directory / "height.float64",
            mode="w+",
            dtype=np.float64,
            shape=crop_shape,
        )

    fallback_height = (
        float(height_m) if np.isscalar(height_m) else float(np.nanmean(height_m))
    )
    height_array = None if np.isscalar(height_m) else np.asarray(height_m)
    if height_array is not None and height_array.shape != crop_shape:
        reject_invalid_state(
            f"height_m shape {height_array.shape} does not match crop {crop_shape}"
        )

    mean_heights: list[float] = []
    polygon_mask: np.ndarray | None = None
    if footprint_lonlat is not None:
        polygon_mask = footprint_polygon_mask(
            grid,
            row0,
            row1,
            col0,
            col1,
            np.asarray(footprint_lonlat),
            dilate_px=polygon_dilate_px,
        )
    elif roi_geometry is not None:
        polygon_mask = roi_geo_mask(
            roi_geometry,
            grid,
            row0,
            row1,
            col0,
            col1,
            dilate_px=polygon_dilate_px,
        )
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
        if polygon_mask is not None:
            poly_rows = slice(row_start - row0, row_stop - row0)
            finite_geo &= polygon_mask[poly_rows, :]
        safe_latitude = np.where(finite_geo, latitude_chunk, 0.0)
        safe_longitude = np.where(finite_geo, longitude_chunk, 0.0)
        if dem is not None:
            sampled_height = np.asarray(
                dem.sample(
                    np.where(grid_ok, latitude_chunk, 0.0),
                    np.where(grid_ok, longitude_chunk, 0.0),
                ),
                dtype=np.float64,
            )
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
        if not np.isscalar(height_chunk):
            mean_heights.append(float(np.nanmean(height_chunk)))

        result = geo2rdr(
            geometry,
            safe_latitude,
            safe_longitude,
            height_chunk,
        )
        chunk_valid = (
            finite_geo
            & result.converged
            & np.isfinite(result.azimuth_index)
            & np.isfinite(result.range_index)
            & (result.azimuth_index >= 0.0)
            & (result.azimuth_index <= full_height - 1.0)
            & (result.range_index >= 0.0)
            & (result.range_index <= full_width - 1.0)
        )
        azimuth[row_start - row0 : row_stop - row0, :] = np.where(
            chunk_valid,
            result.azimuth_index,
            np.nan,
        )
        range_index[row_start - row0 : row_stop - row0, :] = np.where(
            chunk_valid,
            result.range_index,
            np.nan,
        )
        valid[row_start - row0 : row_stop - row0, :] = chunk_valid
        for array in (azimuth, range_index, valid, height_lookup):
            if isinstance(array, np.memmap):
                release_memmap_pages(array)

    if isinstance(azimuth, np.memmap):
        azimuth.flush()
    if isinstance(range_index, np.memmap):
        range_index.flush()
    if isinstance(valid, np.memmap):
        valid.flush()
    if isinstance(height_lookup, np.memmap):
        height_lookup.flush()
    mean_height = float(np.mean(mean_heights)) if mean_heights else fallback_height
    logger.info(
        "Built geo2rdr LUT: %d/%d valid, radar_shape=%s, mean_height=%.1f m",
        int(valid.sum()),
        valid.size,
        full_radar_shape,
        mean_height,
    )
    return Geo2RdrLUT(
        az_full=azimuth,
        rg_full=range_index,
        valid=valid,
        full_radar_shape=(int(full_height), int(full_width)),
        height_m=mean_height,
        height_full=height_lookup,
        row0=0 if not full_crop else row0,
        col0=0 if not full_crop else col0,
    )
