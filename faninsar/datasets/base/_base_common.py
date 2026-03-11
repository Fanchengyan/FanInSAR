"""Shared imports and helper utilities for dataset base modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from os import PathLike
from typing import TypeAlias, cast

import numpy as np
import rioxarray  # noqa: F401
from rasterio.dtypes import dtype_ranges

from faninsar._core.sar.pairs import Pairs
from faninsar.logging import setup_logger
from faninsar.query import BoundingBox, GeoQuery, Points, Polygons

logger = setup_logger(__name__)

lat_names = ["latitude", "lat", "latitudes", "y", "lats", "ny"]
lon_names = ["longitude", "lon", "long", "lng", "longitudes", "longs", "nx", "x"]

PairParser: TypeAlias = Callable[[Iterable[str | PathLike]], Pairs]


# Serialization helper functions
def _serialize_points(points: Points) -> dict:
    """Serialize points to dict for saving in dataset attrs."""
    crs_str = str(points.crs) if points.crs is not None else None
    return {
        "type": "Points",
        "crs": crs_str,
        "coords": points.values.tolist(),
    }


def _serialize_bbox(bbox: BoundingBox) -> dict:
    """Serialize bbox to dict for saving in dataset attrs."""
    crs_str = str(bbox.crs) if bbox.crs is not None else None
    return {
        "type": "BoundingBox",
        "crs": crs_str,
        "left": float(bbox.left),
        "bottom": float(bbox.bottom),
        "right": float(bbox.right),
        "top": float(bbox.top),
    }


def _serialize_polygons(polygons: Polygons) -> dict:
    """Serialize polygons to dict for saving in dataset attrs."""
    crs_str = str(polygons.crs) if polygons.crs is not None else None
    wkts = []
    try:
        wkts = [geom.wkt for geom in polygons.geodataframe.geometry]
    except Exception:
        wkts = [str(g) for g in polygons.geodataframe.geometry]
    return {"type": "Polygons", "crs": crs_str, "wkt": wkts}


def get_nodata(
    arr: np.ndarray,
    nodata: float | None,
    dtype: str | np.dtype,
) -> float:
    """Get a proper no data value for the array."""
    if nodata is None:
        if np.issubdtype(arr.dtype, np.floating):
            nodata = np.nan
        else:
            rng = dtype_ranges[str(dtype)]
            nodata = rng[1] if np.any(arr == rng[0]) else rng[0] - 1
    return cast("float", nodata)


def parse_1d_dims(
    values_1d: np.ndarray,
    multi_files: bool = True,
) -> tuple[list[tuple[str, int]], np.ndarray]:
    """Parse the dimensions of 1D array. (used by points)."""
    if multi_files:
        if values_1d.ndim == 2:
            n_files, n_points = values_1d.shape
            dims = [("files", n_files), ("points", n_points)]
        elif values_1d.ndim == 3:
            n_files, n_points, n_bands = values_1d.shape
            values_1d = values_1d.transpose(0, 2, 1)
            dims = [("files", n_files), ("bands", n_bands), ("points", n_points)]
        else:
            msg = f"values_1d must be 2D or 3D, got {values_1d.ndim}"
            raise ValueError(msg)
    elif values_1d.ndim == 1:
        n_points = values_1d.shape[0]
        dims = [("points", n_points)]
    elif values_1d.ndim == 2:
        n_points, n_bands = values_1d.shape
        values_1d = values_1d.T
        dims = [("bands", n_bands), ("points", n_points)]
    return dims, values_1d


def format_dims_as_string(dims: list[tuple[str, int]] | list[tuple[str, str]]) -> str:
    """Format dims list as string for backward compatibility."""
    return ", ".join([f"{name}:{size}" for name, size in dims])


def parse_2d_dims(
    values_2d: np.ndarray,
    multi_files: bool = True,
) -> list[tuple[str, int]]:
    """Parse the dimensions of 2D array. (used by bbox, polygons)."""
    if multi_files:
        if values_2d.ndim == 4:
            n_files, n_bands, height, width = values_2d.shape
            dims = [
                ("files", n_files),
                ("bands", n_bands),
                ("height", height),
                ("width", width),
            ]
        elif values_2d.ndim == 3:
            n_files, height, width = values_2d.shape
            dims = [("files", n_files), ("height", height), ("width", width)]
        else:
            msg = f"values_2d must be 3D or 4D, got {values_2d.ndim}"
            raise ValueError(msg)
    elif values_2d.ndim == 3:
        n_bands, height, width = values_2d.shape
        values_2d = values_2d.transpose(1, 2, 0)
        dims = [("bands", n_bands), ("height", height), ("width", width)]
    elif values_2d.ndim == 2:
        height, width = values_2d.shape
        dims = [("height", height), ("width", width)]
    else:
        msg = f"values_2d must be 2D or 3D, got {values_2d.ndim}"
        raise ValueError(msg)
    return dims


def ensure_geo_query(query: GeoQuery | Points | BoundingBox | Polygons) -> GeoQuery:
    """Ensure the query is a GeoQuery object."""
    if isinstance(query, GeoQuery):
        return query
    if isinstance(query, Points):
        query = GeoQuery(points=query)
    if isinstance(query, BoundingBox):
        query = GeoQuery(boxes=query)
    if isinstance(query, Polygons):
        query = GeoQuery(polygons=query)
    return query


def _serialize_points(points: Points) -> dict:
    """Serialize points to dict for saving in dataset attrs."""
    crs_str = str(points.crs) if points.crs is not None else None
    return {
        "type": "Points",
        "crs": crs_str,
        "coords": points.values.tolist(),
    }


def _serialize_bbox(bbox: BoundingBox) -> dict:
    """Serialize bbox to dict for saving in dataset attrs."""
    crs_str = str(bbox.crs) if bbox.crs is not None else None
    return {
        "type": "BoundingBox",
        "crs": crs_str,
        "left": float(bbox.left),
        "bottom": float(bbox.bottom),
        "right": float(bbox.right),
        "top": float(bbox.top),
    }


def _serialize_polygons(polygons: Polygons) -> dict:
    """Serialize polygons to dict for saving in dataset attrs."""
    crs_str = str(polygons.crs) if polygons.crs is not None else None
    try:
        wkts = [geom.wkt for geom in polygons.geodataframe.geometry]
    except Exception:
        wkts = [str(g) for g in polygons.geodataframe.geometry]
    return {"type": "Polygons", "crs": crs_str, "wkt": wkts}
