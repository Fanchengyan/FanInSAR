"""Coordinate and affine-transform helpers for geospatial rasters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from rasterio import Affine, transform

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from rasterio.profiles import Profile as RasterioProfile

    from faninsar.core.types import CrsLike
    from faninsar.data.query.bbox import BoundingBox

logger = setup_logger(__name__)

OFFSET_LOCATIONS: dict[
    Literal["center", "ul", "ur", "ll", "lr"], tuple[float, float]
] = {
    "center": (0.5, 0.5),
    "ul": (0, 0),
    "ur": (1, 0),
    "ll": (0, 1),
    "lr": (1, 1),
}


def _offset_from_loc(
    loc: Literal["center", "ul", "ur", "ll", "lr"],
) -> tuple[float, float]:
    """Get the offset from pixel location."""
    if loc not in OFFSET_LOCATIONS:
        msg = f"loc should be one of {tuple(OFFSET_LOCATIONS.keys())}, but got {loc}"
        logger.error(msg)
        raise ValueError(msg)
    return OFFSET_LOCATIONS[loc]


def transform_from_xy(
    x: ArrayLike,
    y: ArrayLike,
    *,
    loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
) -> Affine:
    """Get the :class:`rasterio.Affine` from x and y coordinates.

    Parameters
    ----------
    x, y: ArrayLike
        x and y coordinates
    loc: Literal["center", "ul", "ur", "ll", "lr"], optional
        The pixel location that the coordinates refer to. Supported values are
        "center", "ul", "ur", "ll", and "lr". Default is "center".

    """
    west, south, east, north = (
        np.nanmin(x),
        np.nanmin(y),
        np.nanmax(x),
        np.nanmax(y),
    )
    width, height = len(x), len(y)

    xsize = (east - west) / width
    ysize = (north - south) / height

    offset = _offset_from_loc(loc)

    return transform.from_origin(
        west - offset[0] * xsize,  # center to left
        north + offset[1] * ysize,  # center to top
        xsize,
        ysize,
    )


def bounds_from_xy(
    x: ArrayLike,
    y: ArrayLike,
    *,
    loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
    crs: CrsLike = "WGS84",
) -> BoundingBox:
    """Get the bounds from x and y coordinates.

    Parameters
    ----------
    x, y: ArrayLike
        x and y coordinates
    loc: Literal["center", "ul", "ur", "ll", "lr"], optional
        The pixel location that the coordinates refer to. Supported values are
        "center", "ul", "ur", "ll", and "lr". Default is "center".
    crs: CrsLike, optional
        the coordinate reference system. Could be any type that accepted by
        :meth:`pyproj.CRS.from_user_input`. Default is "WGS84".

    """
    from faninsar.data.query.bbox import BoundingBox

    width, height = len(x), len(y)
    tf = transform_from_xy(x, y, loc=loc)
    left, top = tf * (0, 0)
    right, bottom = tf * (width, height)
    return BoundingBox(left, bottom, right, top, crs=crs)


def geoinfo_from_xy(
    x: ArrayLike,
    y: ArrayLike,
    *,
    crs: CrsLike = "WGS84",
    loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
) -> tuple[BoundingBox, Affine, tuple, tuple]:
    """Evaluate the geoinformation from x and y coordinates.

    Parameters
    ----------
    x, y: numpy.ndarray or list
        x and y coordinates
    loc: Literal["center", "ul", "ur", "ll", "lr"], optional
        The pixel location that the coordinates refer to. Supported values are
        "center", "ul", "ur", "ll", and "lr". Default is "center".
    crs: CrsLike, optional
        the coordinate reference system. Could be any type that accepted by
        :meth:`pyproj.CRS.from_user_input`. Default is "WGS84".

    Returns
    -------
    bounds: BoundingBox
        the bounding box of the raster.
    transform: Affine
        the affine transform of the raster.
    res: tuple[xsize, ysize]
        the resolution of the raster
    shape: tuple[height, width]
        the shape of the raster

    """
    from faninsar.data.query.bbox import BoundingBox

    tf = transform_from_xy(x, y, loc=loc)
    res = (abs(tf.a), abs(tf.e))

    width, height = len(x), len(y)
    shape = (height, width)

    left, top = tf * (0, 0)
    right, bottom = tf * (width, height)
    bounds = BoundingBox(left, bottom, right, top, crs=crs)

    return bounds, tf, res, shape


def xy_from_transform(
    tf: Affine | None,
    width: int,
    height: int,
    *,
    loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
) -> tuple[np.ndarray, np.ndarray]:
    """Get the x and y coordinates from transform and shape.

    Parameters
    ----------
    tf: Affine | None
        the transform of the raster. If tf is None, the x and y coordinates will
        be range(width) and range(height).
    width, height: int
        the width and height of the raster
    loc: Literal["center", "ul", "ur", "ll", "lr"], optional
        the pixel location that the coordinates refer to. Supported values are
        "center", "ul", "ur", "ll", and "lr". Default is "center".

    Returns
    -------
    x, y: numpy.ndarray

    """
    if tf is None:
        return np.arange(width), np.arange(height)
    offset = _offset_from_loc(loc)
    x = tf.xoff + tf.a * (np.arange(width) + offset[0])
    y = tf.yoff + tf.e * (np.arange(height) + offset[1])
    return x, y


def xy_from_profile(profile: RasterioProfile) -> tuple[np.ndarray, np.ndarray]:
    """Get the x and y coordinates from rasterio profile data.

    Parameters
    ----------
    profile: Profile
        the profile data of rasterio dataset. It can be get from
        rasterio.open().profile

    Returns
    -------
    x, y: numpy.ndarray

    """
    tf = profile["transform"]
    width = profile["width"]
    height = profile["height"]
    return xy_from_transform(tf, width, height)
