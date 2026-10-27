"""Grid reference models for geospatial rasters."""

from __future__ import annotations

import pprint
from typing import TYPE_CHECKING, Literal

import numpy as np
from pyproj.crs import CRS
from rasterio import Affine, transform
from rasterio.transform import array_bounds, rowcol, xy
from rasterio.warp import calculate_default_transform

from faninsar.logging import setup_logger

from .coordinates import geoinfo_from_xy, xy_from_transform

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from numpy.typing import ArrayLike
    from odc.geo import GeoBox

    from faninsar.query.bbox import BoundingBox
    from faninsar.typing import CrsLike

logger = setup_logger(__name__)


class GeoGridMixin:
    """A class to manage GeoGrid information of a raster image.

    A GeoGrid is a fixed pixel grid reference system for geospatial data. It
    represents a grid system with fixed pixel resolution, alignment, and coordinate
    reference system (CRS). Different spatial extents can be derived from the
    same grid, ensuring consistent pixel grid alignment across all views.
    """

    def _refresh_bounds(self) -> None:
        """Refresh cached bounds after geometry updates."""
        if (
            hasattr(self, "_transform")
            and hasattr(self, "_shape")
            and hasattr(self, "_crs")
        ):
            self._bounds = self._parse_bounds()

    def _parse_bounds(self) -> BoundingBox:
        """Parse the bounds from geogrid data."""
        from faninsar.query.bbox import BoundingBox

        west, south, east, north = array_bounds(self.height, self.width, self.transform)
        left = min(west, east)
        right = max(west, east)
        bottom = min(south, north)
        top = max(south, north)
        return BoundingBox(left, bottom, right, top, crs=self.crs)

    @property
    def transform(self) -> Affine:
        """The transform of raster image."""
        return self._transform

    @transform.setter
    def transform(self, value: Affine) -> None:
        """Set the transform of raster image."""
        self._transform = value
        self._refresh_bounds()

    @property
    def north_up(self) -> bool:
        """Whether the raster image is north-up."""
        return self.transform.e < 0

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of raster image in (height, width) order."""
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int, int]) -> None:
        """Set the shape of raster image in (height, width) order."""
        if len(value) != 2:
            msg = "shape must be a tuple of (height, width)"
            logger.error(msg)
            raise ValueError(msg)
        self._shape = (int(value[0]), int(value[1]))
        self._refresh_bounds()

    @property
    def res(self) -> tuple[float, float]:
        """The resolution of raster image in x and y direction."""
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def width(self) -> int:
        """The width of raster image."""
        return self.shape[1]

    @width.setter
    def width(self, value: int) -> None:
        """Set the width of raster image."""
        self._shape = (self.shape[0], int(value))
        self._refresh_bounds()

    @property
    def height(self) -> int:
        """The height of raster image."""
        return self.shape[0]

    @height.setter
    def height(self, value: int) -> None:
        """Set the height of raster image."""
        self._shape = (int(value), self.shape[1])
        self._refresh_bounds()

    @property
    def crs(self) -> CRS | None:
        """The coordinate reference system of raster image."""
        return self._crs

    @crs.setter
    def crs(self, value: CrsLike | None) -> None:
        """Set the coordinate reference system of raster image."""
        self._crs = CRS.from_user_input(value) if value is not None else None
        self._refresh_bounds()

    @property
    def bounds(self) -> BoundingBox:
        """The bounds of the GeoGrid in (west, south, east, north) order."""
        return self._bounds

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """The extent of the GeoGrid in (west, east, south, north) order."""
        b = self.bounds
        return (b.left, b.right, b.bottom, b.top)


class GeoGrid(GeoGridMixin):
    """A fixed pixel grid reference system for geospatial data.

    GeoGrid represents a grid system with fixed resolution, pixel alignment,
    and coordinate reference system (CRS). Different spatial extents can be
    derived from the same grid, ensuring consistent pixel grid alignment
    across all views.

    Parameters
    ----------
    transform: Affine
        Affine transformation matrix mapping pixel coordinates to map coordinates.
    shape: tuple[int, int]
        Grid dimensions as (height, width) in pixels.
    crs: CrsLike, optional
        Coordinate reference system. Accepts any type compatible with
        :meth:`pyproj.CRS.from_user_input`. Default is None (unset).

    """

    def __init__(
        self,
        transform: Affine,
        shape: tuple[int, int],
        crs: CrsLike | None = None,
    ) -> None:
        """Initialize the GeoBox class."""
        self._transform = transform
        self.crs = crs
        self.shape = shape

    def __repr__(self) -> str:
        """Get the string representation of the GeoGrid."""
        info = {
            "bounds": self.bounds.to_tuple(),
            "transform": self.transform,
            "shape": self.shape,
            "crs": self.crs.to_string() if self.crs else None,
        }
        repr_str = f" {pprint.pformat(info, indent=2, sort_dicts=False).strip('{}')}"
        return f"GeoGrid(\n{repr_str}\n)"

    @classmethod
    def from_bounds(
        cls,
        bounds: BoundingBox | tuple[float, float, float, float],
        *,
        res: float | tuple[float, float] | None = None,
        shape: tuple[int, int] | None = None,
        crs: CrsLike | None = None,
    ) -> Self:
        """Create a GeoGrid from bounds and shape.

        Parameters
        ----------
        bounds: BoundingBox or tuple[float, float, float, float]
            the bounds of the raster image in [west, south, east, north] order.
            The bounds will be converted to the :param:`crs` if crs is not None,
            otherwise the crs of bounds will be used.
        res: float or tuple[float, float] | None, optional
            the resolution of the raster image in x and y direction. If res is a
            single float, it will be used for both x and y direction. If None,
            the resolution will be calculated from bounds and shape.
        shape: tuple[int, int] | None, optional
            the shape of the raster image in (height, width) order, which will
            be ignored if res is not None.
        crs: CrsLike | None, optional
            the coordinate reference system of the raster image. Could be any
            type that :meth:`pyproj.CRS.from_user_input` accepts. If None, the
            crs of bounds will be used.

        Returns
        -------
        GeoGrid
             the GeoGrid object created from bounds and shape.

        Raises
        ------
        ValueError
            if neither res nor shape is provided

        """
        bounds, crs = format_bounds_and_crs(bounds, crs)
        left, bottom, right, top = bounds
        if res is not None:
            if isinstance(res, (int, float)):
                res = (res, res)
            xsize = abs(float(res[0]))
            ysize = abs(float(res[1]))
            width = int(np.ceil((right - left) / xsize))
            height = int(np.ceil((top - bottom) / ysize))
            shape = (height, width)
        elif shape is not None:
            width, height = shape[1], shape[0]
            xsize = (right - left) / width
            ysize = (top - bottom) / height
        else:
            msg = "either res or shape must be provided"
            logger.error(msg)
            raise ValueError(msg)
        tf = transform.from_origin(left, top, xsize, ysize)

        return cls(tf, shape, crs)

    @classmethod
    def from_xy(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        crs: CrsLike = "WGS84",
        loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
    ) -> Self:
        """Create a GeoGrid from x and y coordinates.

        Parameters
        ----------
        x, y: ArrayLike
            x and y coordinates of pixel centers in raster image.
        crs: CrsLike, optional
            the coordinate reference system of the input coordinates. Could be
            any type that accepted by :meth:`pyproj.CRS.from_user_input`.
            Default is "WGS84".
        loc: str, optional
            the location of the coordinates in pixel. It can be "center", "ul",
            "ur", "ll" or "lr". Default is "center".

        Returns
        -------
        GeoGrid
            the GeoGrid object created from x and y coordinates.

        """
        bounds, tf, _, shape = geoinfo_from_xy(x, y, crs=crs, loc=loc)
        return cls(tf, shape, bounds.crs)

    def to_crs(
        self,
        crs: CrsLike,
        *,
        res: float | tuple[float, float] | None = None,
        shape: tuple[int, int] | None = None,
    ) -> GeoGrid:
        """Get a new GeoGrid reprojected to the destination CRS.

        Parameters
        ----------
        crs: CrsLike
            the destination coordinate reference system. Could be any type that
            :meth:`pyproj.CRS.from_user_input` accepts.
        res: float | tuple[float, float] | None, optional
            Target resolution, in units of target coordinate reference system.
            Default is None.
        shape: tuple[x resolution, y resolution] | None, optional
            the shape of the new GeoGrid in (height, width) order. Cannot be set
            if res is not None. Default is None.

        Returns
        -------
        GeoGrid
            New GeoGrid reprojected to the destination CRS with aligned pixel grid.

        """
        left, bottom, right, top = self.bounds
        dst_width, dst_height = None, None
        if shape is not None:
            dst_width, dst_height = shape[1], shape[0]
        tf, width, height = calculate_default_transform(
            self.crs,
            crs,
            self.width,
            self.height,
            left,
            bottom,
            right,
            top,
            dst_width=dst_width,
            dst_height=dst_height,
            resolution=res,
        )

        # Create a new GeoGrid with the same resolution and shape but new bounds and CRS
        return GeoGrid(tf, (height, width), crs)

    def to_view(self, roi: BoundingBox) -> GeoGrid:
        """Create a new GeoGrid with the view focused on the roi region.

        The new GeoGrid shares the same pixel resolution and alignment as the
        original, only view window (bounds) change.

        Parameters
        ----------
        roi: BoundingBox
            Region of interest in BoundingBox format. The bounds will be
            converted to the GeoGrid's CRS if needed.

        Returns
        -------
        GeoGrid
            New GeoGrid windowed to the roi region with aligned pixel grid.

        """
        roi, _ = format_bounds_and_crs(roi, self.crs)
        if roi == self.bounds:
            return self
        xsize = abs(float(self.res[0]))
        ysize = abs(float(self.res[1]))

        left, bottom, right, top = roi
        rows, cols = rowcol(
            self.transform,
            [left, right, right, left],
            [top, top, bottom, bottom],
            op=float,
        )
        w, n = self.transform * (min(cols), min(rows))
        tf = transform.from_origin(w, n, xsize, ysize)
        shape = (max(rows) - min(rows), max(cols) - min(cols))

        return GeoGrid(tf, shape, self.crs)

    def get_xy(
        self, loc: Literal["center", "ul", "ur", "ll", "lr"] = "center"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the x and y coordinates of pixel centers of the raster image."""
        return xy_from_transform(self.transform, self.width, self.height, loc=loc)

    def to_geobox(self) -> GeoBox:
        """Convert the GeoGrid to an odc.geo.GeoBox."""
        from odc.geo import GeoBox

        return GeoBox(self.shape, self.transform, self.crs)

    def row_col(
        self, x: ArrayLike, y: ArrayLike, op: Callable | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the row and column indices for the given x and y coordinates.

        Parameters
        ----------
        x, y: ArrayLike
            x and y coordinates to be converted to row and column indices.
        op: Callable, optional
            Function to convert fractional pixels to whole numbers (floor,
            ceiling, round). If None, numpy.floor will be used. Default is None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            row and column indices corresponding to the input x and y coordinates.

        """
        return rowcol(self.transform, x, y, op=op)

    def xy(
        self,
        row: ArrayLike,
        col: ArrayLike,
        offset: Literal["center", "ul", "ur", "ll", "lr"] = "center",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the x and y coordinates for the given row and column indices.

        Parameters
        ----------
        row, col: ArrayLike
            row and column indices to be converted to x and y coordinates.
        offset: str, optional
            Determines if the returned coordinates are for the center of the
            pixel or for a corner. It can be "center", "ul", "ll" or "lr".
            Default is "center".

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            x and y coordinates corresponding to the input row and column indices.

        """
        return xy(self.transform, row, col, offset=offset)


def format_bounds_and_crs(
    bounds: BoundingBox | tuple[float, float, float, float],
    crs: CrsLike | None = None,
) -> tuple[BoundingBox, CRS | None]:
    """Get the formatted bounds and crs from the input."""
    from faninsar.query.bbox import BoundingBox

    if not isinstance(bounds, BoundingBox):
        left, bottom, right, top = bounds
        bounds = BoundingBox(left, bottom, right, top, crs=crs)
    if crs is not None:
        crs = CRS.from_user_input(crs)
        if bounds.crs is not None and bounds.crs != crs:
            bounds = bounds.to_crs(crs)
    else:
        crs = bounds.crs
    return bounds, crs
