"""Coordinate and affine-transform helpers for geospatial rasters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

import numpy as np
from rasterio import Affine, transform

from faninsar.logging import setup_logger
from faninsar.processing.errors import (
    GridMismatchError,
    reject_grid_mismatch,
    reject_invalid_state,
)

if TYPE_CHECKING:
    from datetime import datetime

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


Shape2D: TypeAlias = tuple[int, int]
AffineTransform: TypeAlias = tuple[float, float, float, float, float, float]
Vector3: TypeAlias = tuple[float, float, float]


class CoordinateSystem(StrEnum):
    """Coordinate systems supported by processing products."""

    RADAR = "radar"
    GEO = "geo"


class ArrayRepresentation(StrEnum):
    """Physical representation stored by an array asset."""

    COMPLEX = "complex"
    PHASE = "phase"
    AMPLITUDE = "amplitude"
    COORDINATE = "coordinate"
    OFFSET = "offset"
    MASK = "mask"


@dataclass(frozen=True, slots=True)
class ArrayDescriptor:
    """Location and semantics of a lazily readable two-dimensional array."""

    uri: str
    shape: Shape2D
    dtype: str
    representation: ArrayRepresentation

    def __post_init__(self) -> None:
        """Validate the array locator and its two-dimensional shape."""
        if not self.uri:
            reject_invalid_state("array URI must not be empty")
        if len(self.shape) != 2 or min(self.shape) <= 0:
            reject_invalid_state("array shape must contain two positive dimensions")


class GridContract(Protocol):
    """Common shape and coordinate identity exposed by processing grids."""

    @property
    def shape(self) -> Shape2D:
        """Return the grid height and width."""

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Return the coordinate-system identity."""


@dataclass(frozen=True, slots=True)
class RadarGrid:
    """Regular zero-Doppler radar grid metadata."""

    shape: Shape2D
    starting_slant_range_m: float
    range_spacing_m: float
    sensing_start: datetime
    azimuth_time_interval_s: float
    wavelength_m: float
    look_direction: Literal["left", "right"]

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Return the radar coordinate-system identity."""
        return CoordinateSystem.RADAR

    def __post_init__(self) -> None:
        """Validate radar-grid dimensions, timing, and physical spacing."""
        if min(self.shape) <= 0:
            reject_invalid_state("radar grid shape must be positive")
        if (
            min(
                self.starting_slant_range_m,
                self.range_spacing_m,
                self.azimuth_time_interval_s,
                self.wavelength_m,
            )
            <= 0
        ):
            reject_invalid_state("radar grid spacings and wavelength must be positive")
        if self.sensing_start.tzinfo is None:
            reject_invalid_state("radar sensing start must include a timezone")


@dataclass(frozen=True, slots=True)
class GeoGrid:
    """Regular projected or geographic affine grid metadata."""

    shape: Shape2D
    crs: str
    transform: AffineTransform

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Return the geographic coordinate-system identity."""
        return CoordinateSystem.GEO

    def __post_init__(self) -> None:
        """Validate geographic-grid dimensions, CRS, and affine spacing."""
        if min(self.shape) <= 0:
            reject_invalid_state("geo grid shape must be positive")
        if not self.crs:
            reject_invalid_state("geo grid CRS must not be empty")
        if self.transform[0] == 0.0 or self.transform[4] == 0.0:
            reject_invalid_state("geo grid pixel sizes must be non-zero")


ProcessingGrid: TypeAlias = RadarGrid | GeoGrid


class TransformDirection(StrEnum):
    """Direction represented by a coordinate lookup table."""

    RADAR_TO_GEO = "rdr2geo"
    GEO_TO_RADAR = "geo2rdr"


@dataclass(frozen=True, slots=True)
class TransformLUT:
    """Lookup arrays mapping a source grid onto a target grid."""

    source: ProcessingGrid
    target: ProcessingGrid
    direction: TransformDirection
    range_coordinates: ArrayDescriptor
    azimuth_coordinates: ArrayDescriptor

    def __post_init__(self) -> None:
        """Validate direction, coordinate representation, and target shape."""
        expected_shape = self.target.shape
        if (
            self.range_coordinates.shape != expected_shape
            or self.azimuth_coordinates.shape != expected_shape
        ):
            reject_grid_mismatch("transform coordinates must match the target grid")
        if (
            self.range_coordinates.representation is not ArrayRepresentation.COORDINATE
            or self.azimuth_coordinates.representation
            is not ArrayRepresentation.COORDINATE
        ):
            reject_invalid_state("transform lookup arrays must store coordinates")
        expected_systems = {
            TransformDirection.RADAR_TO_GEO: (
                CoordinateSystem.RADAR,
                CoordinateSystem.GEO,
            ),
            TransformDirection.GEO_TO_RADAR: (
                CoordinateSystem.GEO,
                CoordinateSystem.RADAR,
            ),
        }[self.direction]
        actual_systems = (
            self.source.coordinate_system,
            self.target.coordinate_system,
        )
        if actual_systems != expected_systems:
            reject_invalid_state(
                "transform direction does not match its source and target"
            )

    def validate_resampling_source(self, source: ArrayDescriptor) -> None:
        """Validate that resampling preserves complex SAR samples."""
        if source.representation is not ArrayRepresentation.COMPLEX:
            reject_invalid_state(
                "resampling requires complex samples, not phase-only data"
            )
        if source.shape != self.source.shape:
            reject_grid_mismatch(
                "resampling source must match the transform source grid"
            )


@dataclass(frozen=True, slots=True)
class OffsetField:
    """Range and azimuth offsets defined on one processing grid."""

    grid: ProcessingGrid
    range_offsets: ArrayDescriptor
    azimuth_offsets: ArrayDescriptor

    def __post_init__(self) -> None:
        """Validate offset representations and their shared grid shape."""
        if (
            self.range_offsets.shape != self.grid.shape
            or self.azimuth_offsets.shape != self.grid.shape
        ):
            reject_grid_mismatch("offset arrays must match their grid")
        if (
            self.range_offsets.representation is not ArrayRepresentation.OFFSET
            or self.azimuth_offsets.representation is not ArrayRepresentation.OFFSET
        ):
            reject_invalid_state("offset arrays must use the offset representation")


__all__ = [
    "AffineTransform",
    "ArrayDescriptor",
    "ArrayRepresentation",
    "CoordinateSystem",
    "GeoGrid",
    "GridContract",
    "GridMismatchError",
    "OffsetField",
    "ProcessingGrid",
    "RadarGrid",
    "Shape2D",
    "TransformDirection",
    "TransformLUT",
    "Vector3",
]
