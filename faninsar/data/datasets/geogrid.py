"""GeoGrid utilities for FanInSAR lazy geoprocessing.

Notes
-----
This module provides a thin extension of odc-geo's GeoBox with FanInSAR-specific
compatibility methods for BoundingBox-based APIs. The class is named
:class:`GeoGrid` to align naming with the sibling package ``geosam``.

See Also
--------
odc.geo.GeoBox : Base class with full geobox functionality
faninsar.data.query.bbox : BoundingBox primitives

"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from odc.geo import GeoBox as OdcGeoBox
from odc.geo.geobox import GeoboxTiles as OdcGeoboxTiles
from odc.geo.geobox import pixel_translation
from odc.geo.math import is_almost_int
from rasterio.transform import from_bounds, rowcol
from rasterio.windows import Window
from rasterio.windows import transform as _window_transform

from faninsar.data.query.bbox import BoundingBox
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from affine import Affine

TileIndex: TypeAlias = tuple[int, int]

logger = setup_logger(__name__)


class GeoGrid(OdcGeoBox):
    """Extended GeoGrid with FanInSAR-specific compatibility methods.

    This class extends odc-geo's GeoBox, adding compatibility methods for
    FanInSAR's existing codebase. It keeps the ``odc.geo`` foundation
    (``.coords``, ``.dims``, xarray integration) so downstream code is
    unaffected; only the class name and a few factory aliases are aligned
    with ``geosam.datasets.geogrid.GeoGrid``.

    See Also
    --------
    odc.geo.GeoBox : Base class with full documentation

    """

    def __init__(
        self,
        shape: tuple[int, int] | Any | None = None,
        affine: Affine | None = None,
        crs: Any | None = None,
        *,
        transform: Affine | None = None,
        bounds: BoundingBox | None = None,
        width: int | None = None,
        height: int | None = None,
        res: float | tuple[float, float] | None = None,
        dtype: Any | None = None,
        nodata: float | int | None = None,
    ) -> None:
        """Construct a grid from odc-geo or raster geoinfo fields.

        Parameters
        ----------
        shape, affine, crs
            Native :class:`odc.geo.GeoBox` constructor fields.
        transform, bounds, width, height, res, dtype, nodata
            Raster geoinfo fields accepted by FanInSAR data readers.

        """
        if affine is None:
            affine = transform
        if shape is None and width is not None and height is not None:
            shape = (height, width)
        if affine is None and bounds is not None and shape is not None:
            affine = from_bounds(
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                int(shape[1]),
                int(shape[0]),
            )
        if shape is None or affine is None:
            raise TypeError("GeoGrid requires shape and affine/transform")
        super().__init__(shape=shape, affine=affine, crs=crs)
        self._dtype = None if dtype is None else np.dtype(dtype)
        self._nodata = nodata

    @classmethod
    def from_geoinfo(
        cls,
        *,
        crs: Any,
        bounds: BoundingBox,
        res: float | tuple[float, float],
        dtype: Any | None = None,
        nodata: float | int | None = None,
    ) -> GeoGrid:
        """Construct a grid from CRS, bounds, resolution, and raster metadata."""
        if isinstance(res, (int, float)):
            res = (float(res), float(res))
        xres, yres = abs(float(res[0])), abs(float(res[1]))
        width = int(round((bounds.right - bounds.left) / xres))
        height = int(round((bounds.top - bounds.bottom) / yres))
        if width <= 0 or height <= 0:
            raise ValueError("bounds and resolution must define a non-empty grid")
        return cls(
            shape=(height, width),
            affine=from_bounds(
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                width,
                height,
            ),
            crs=crs,
            dtype=dtype,
            nodata=nodata,
        )

    @classmethod
    def from_odc_geobox(cls, geobox: OdcGeoBox) -> GeoGrid:
        """Create a GeoGrid from an odc-geo GeoBox."""
        return cls(
            shape=geobox.shape,
            affine=geobox.affine,
            crs=geobox.crs,
            dtype=getattr(geobox, "dtype", None),
            nodata=getattr(geobox, "nodata", None),
        )

    @property
    def dtype(self) -> np.dtype | None:
        """Return the optional raster sample dtype."""
        return self._dtype

    @property
    def nodata(self) -> float | int | None:
        """Return the optional raster nodata value."""
        return self._nodata

    @property
    def res(self) -> tuple[float, float]:
        """Return positive pixel resolution in ``(x, y)`` order."""
        return (abs(float(self.resolution.x)), abs(float(self.resolution.y)))

    def align_to(self, other: GeoGrid) -> GeoGrid:
        """Snap this grid to another grid's pixel alignment."""
        return self.snap_to(other)

    def resample(self, res: float | tuple[float, float]) -> GeoGrid:
        """Create a grid with the same bounds at a new resolution."""
        if isinstance(res, (int, float)):
            res = (float(res), float(res))
        xres, yres = abs(float(res[0])), abs(float(res[1]))
        width = max(1, int(round((self.bounds.right - self.bounds.left) / xres)))
        height = max(1, int(round((self.bounds.top - self.bounds.bottom) / yres)))
        return type(self).from_geoinfo(
            crs=self.crs,
            bounds=self.bounds,
            res=(xres, yres),
            dtype=self.dtype,
            nodata=self.nodata,
        )

    def intersect_tiles(
        self,
        other: GeoGrid,
        self_chunk_shape: tuple[int, int],
        other_chunk_shape: tuple[int, int],
    ) -> dict[tuple[int, int], list[tuple[int, int]]]:
        """Map tiles in this grid to overlapping tiles in another grid."""
        self_rows = range(ceil(self.height / self_chunk_shape[0]))
        self_cols = range(ceil(self.width / self_chunk_shape[1]))
        other_rows = range(ceil(other.height / other_chunk_shape[0]))
        other_cols = range(ceil(other.width / other_chunk_shape[1]))
        mapping: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for row in self_rows:
            for col in self_cols:
                self_tile = self[
                    BoundingBox(
                        self.bounds.left + col * self_chunk_shape[1] * self.res[0],
                        self.bounds.top
                        - min((row + 1) * self_chunk_shape[0], self.height)
                        * self.res[1],
                        self.bounds.left
                        + min((col + 1) * self_chunk_shape[1], self.width)
                        * self.res[0],
                        self.bounds.top - row * self_chunk_shape[0] * self.res[1],
                        crs=self.crs,
                    )
                ]
                overlaps: list[tuple[int, int]] = []
                for other_row in other_rows:
                    for other_col in other_cols:
                        other_tile = other[
                            BoundingBox(
                                other.bounds.left
                                + other_col * other_chunk_shape[1] * other.res[0],
                                other.bounds.top
                                - min(
                                    (other_row + 1) * other_chunk_shape[0],
                                    other.height,
                                )
                                * other.res[1],
                                other.bounds.left
                                + min(
                                    (other_col + 1) * other_chunk_shape[1], other.width
                                )
                                * other.res[0],
                                other.bounds.top
                                - other_row * other_chunk_shape[0] * other.res[1],
                                crs=other.crs,
                            )
                        ]
                        if self_tile.bounds.intersects(other_tile.bounds):
                            overlaps.append((other_row, other_col))
                mapping[(row, col)] = overlaps
        return mapping

    # ==================== Class Factory Methods ====================
    # Wrap all @staticmethod factory methods to return the subclass

    @classmethod
    def from_bbox(cls, *args, **kwargs) -> GeoGrid:
        """Construct GeoGrid from a bounding box.

        Returns the FanInSAR GeoGrid subclass instead of odc.geo.GeoBox.
        See odc.geo.GeoBox.from_bbox for full documentation.

        """
        odc_geobox = OdcGeoBox.from_bbox(*args, **kwargs)
        return cls.from_odc_geobox(odc_geobox)

    @classmethod
    def from_bounds(cls, *args, **kwargs) -> GeoGrid:
        """Alias for :meth:`from_bbox` (aligned with geosam API)."""
        return cls.from_bbox(*args, **kwargs)

    @classmethod
    def from_geopolygon(cls, *args, **kwargs) -> GeoGrid:
        """Construct GeoGrid from a polygon.

        Returns the FanInSAR GeoGrid subclass instead of odc.geo.GeoBox.
        See odc.geo.GeoBox.from_geopolygon for full documentation.

        """
        odc_geobox = OdcGeoBox.from_geopolygon(*args, **kwargs)
        return cls.from_odc_geobox(odc_geobox)

    @classmethod
    def from_rio(cls, *args, **kwargs) -> GeoGrid:
        """Construct GeoGrid from rasterio DatasetReader.

        Returns the FanInSAR GeoGrid subclass instead of odc.geo.GeoBox.
        See odc.geo.GeoBox.from_rio for full documentation.

        """
        odc_geobox = OdcGeoBox.from_rio(*args, **kwargs)
        return cls.from_odc_geobox(odc_geobox)

    @classmethod
    def from_dataset(cls, dataset: Any) -> GeoGrid:
        """Build a GeoGrid from a rasterio dataset (alias of :meth:`from_rio`).

        Aligned with the geosam API.

        Parameters
        ----------
        dataset : rasterio.io.DatasetReader
            Open rasterio dataset.

        """
        return cls.from_rio(dataset)

    @property
    def bounds(self) -> BoundingBox:
        """Get the BoundingBox representation of the bounds of GeoGrid."""
        return BoundingBox(*self.boundingbox.bbox, crs=self.crs)

    @property
    def y(self) -> np.ndarray:
        """Get the y coordinates of the GeoGrid."""
        return self.coords[self.dims[0]].values

    @property
    def x(self) -> np.ndarray:
        """Get the x coordinates of the GeoGrid."""
        return self.coords[self.dims[1]].values

    def get_overlap(self, other: GeoGrid) -> BoundingBox | None:
        """Get the overlap between this GeoGrid and another GeoGrid.

        Parameters
        ----------
        other : GeoGrid
            GeoGrid to test against.

        Returns
        -------
        BoundingBox or None
            The overlap bounds in ``self.crs`` if present, otherwise None.

        """
        # Convert both to BoundingBox format
        self_bb = self.boundingbox
        self_bbox = BoundingBox(
            self_bb.left, self_bb.bottom, self_bb.right, self_bb.top, crs=self.crs
        )

        other_bb = other.boundingbox
        other_bbox = BoundingBox(
            other_bb.left, other_bb.bottom, other_bb.right, other_bb.top, crs=other.crs
        )

        # Transform to same CRS
        if other.crs != self.crs:
            other_bbox = other_bbox.to_crs(self.crs)

        if not self_bbox.intersects(other_bbox):
            return None

        left = max(self_bbox.left, other_bbox.left)
        bottom = max(self_bbox.bottom, other_bbox.bottom)
        right = min(self_bbox.right, other_bbox.right)
        top = min(self_bbox.top, other_bbox.top)

        if left >= right or bottom >= top:
            return None

        return BoundingBox(left, bottom, right, top, crs=self.crs)

    def is_aligned(self, other: GeoGrid, tol: float = 1e-6) -> bool:
        """Check if two grids share CRS, resolution, and grid alignment.

        Parameters
        ----------
        other : GeoGrid
            Grid to compare against.
        tol : float, optional
            Numerical tolerance for alignment.

        Returns
        -------
        bool
            True when both GeoGrids share CRS, resolution, and are aligned.

        """
        if self.crs != other.crs:
            return False

        # Check resolution
        res_self = (abs(self.resolution.x), abs(self.resolution.y))
        res_other = (abs(other.resolution.x), abs(other.resolution.y))
        if not np.allclose(res_self, res_other, atol=tol):
            return False

        # Check pixel alignment
        try:
            tx, ty = pixel_translation(other, self).xy
            return is_almost_int(tx, tol) and is_almost_int(ty, tol)
        except (ValueError, AttributeError):
            # Fallback
            dx = abs(self.transform.c - other.transform.c)
            dy = abs(self.transform.f - other.transform.f)
            return bool(
                np.isclose(dx % res_self[0], 0.0, atol=tol)
                and np.isclose(dy % res_self[1], 0.0, atol=tol)
            )

    # ==================== Automatic Method Wrapping ====================
    # Use __init_subclass__ to automatically wrap all methods that return GeoBox

    @classmethod
    def _wrap_instance_method(cls, method_name: str) -> None:
        """Wrap an instance method to return subclass instances."""
        original_method = getattr(OdcGeoBox, method_name)

        def wrapped_method(self, *args, **kwargs) -> GeoGrid:  # noqa: ANN001
            result = original_method(self, *args, **kwargs)
            if isinstance(result, OdcGeoBox) and not isinstance(result, cls):
                return cls.from_odc_geobox(result)
            return result

        wrapped_method.__name__ = method_name
        wrapped_method.__doc__ = (
            f"Wrapped {method_name} returning {cls.__name__} subclass instance.\n\n"
            f"See odc.geo.GeoBox.{method_name} for full documentation."
        )
        setattr(cls, method_name, wrapped_method)

    @classmethod
    def _wrap_property(cls, prop_name: str) -> None:
        """Wrap a property to return subclass instances."""
        original_property = getattr(OdcGeoBox, prop_name)

        def wrapped_getter(self) -> GeoGrid:  # noqa: ANN001
            result = original_property.fget(self)
            if isinstance(result, OdcGeoBox) and not isinstance(result, cls):
                return cls.from_odc_geobox(result)
            return result

        wrapped_property = property(
            wrapped_getter, doc=f"Wrapped {prop_name} property (see odc.geo.GeoBox)"
        )
        setattr(cls, prop_name, wrapped_property)

    def __init_subclass__(cls, **kwargs) -> None:
        """Automatically wrap parent methods for further subclasses."""
        super().__init_subclass__(**kwargs)
        cls._apply_method_wrapping()

    @classmethod
    def _apply_method_wrapping(cls) -> None:
        """Apply method wrapping to the current class."""
        # Methods that return GeoBox instances
        methods_to_wrap = [
            # Buffering and padding
            "buffered",
            "enclosing",
            "pad",
            "pad_wh",
            # Scaling and cropping
            "crop",
            "zoom_out",
            "zoom_to",
            # Projection
            "to_crs",
            # Rotation and flip
            "flipy",
            "flipx",
            "rotate",
            # Translation
            "translate_pix",
            "snap_to",
        ]

        # Properties that return GeoBox instances
        properties_to_wrap = ["left", "right", "top", "bottom", "center_pixel"]

        # Special methods that return GeoBox instances
        special_methods_to_wrap = ["__or__", "__and__", "__mul__", "__rmul__"]

        # Wrap instance methods
        for method_name in methods_to_wrap:
            if hasattr(OdcGeoBox, method_name):
                cls._wrap_instance_method(method_name)

        # Wrap properties
        for prop_name in properties_to_wrap:
            if hasattr(OdcGeoBox, prop_name):
                cls._wrap_property(prop_name)

        # Wrap special methods
        for method_name in special_methods_to_wrap:
            if hasattr(OdcGeoBox, method_name):
                cls._wrap_instance_method(method_name)

    # Alias for crop (will be auto-wrapped)
    expand = property(lambda self: self.crop)

    # ==================== Special Methods ====================

    def __getitem__(self, roi: BoundingBox) -> GeoGrid:
        """Index/slice operation returning a cropped GeoGrid."""
        if not isinstance(roi, BoundingBox):
            shape_, affine_ = self.compute_crop(roi)
            return self.__class__(shape=shape_, affine=affine_, crs=self._crs)
        window = self.get_window_for_bounds(roi)
        shape = (int(window.height), int(window.width))
        affine = _window_transform(window, self.transform)
        return self.__class__(shape=shape, affine=affine, crs=self._crs)

    # Helper methods for compatibility with existing code
    def get_window_for_bounds(self, bounds: BoundingBox) -> Window:
        """Convert bounds to a pixel window on this grid."""
        target = bounds if bounds.crs == self.crs else bounds.to_crs(self.crs)
        left = max(self.boundingbox.left, target.left)
        right = min(self.boundingbox.right, target.right)
        bottom = max(self.boundingbox.bottom, target.bottom)
        top = min(self.boundingbox.top, target.top)

        if left >= right or bottom >= top:
            return Window(0, 0, 0, 0)

        r0, c0 = rowcol(self.transform, left, top, op=float)
        r1, c1 = rowcol(self.transform, right, bottom, op=float)
        row_off = int(np.floor(min(r0, r1)))
        col_off = int(np.floor(min(c0, c1)))
        row_end = int(np.ceil(max(r0, r1)))
        col_end = int(np.ceil(max(c0, c1)))
        row_off = max(0, min(row_off, self.height))
        col_off = max(0, min(col_off, self.width))
        row_end = max(row_off, min(row_end, self.height))
        col_end = max(col_off, min(col_end, self.width))
        return Window(col_off, row_off, col_end - col_off, row_end - row_off)

    def get_window_transform(
        self, window: tuple[int, int, int, int] | Window
    ) -> Affine:
        """Get the affine transform for a window."""
        if not isinstance(window, Window):
            window = Window(*window)
        return _window_transform(window, self.transform)


# Apply method wrapping to the GeoGrid class
GeoGrid._apply_method_wrapping()


# ==================== Backwards-compatibility aliases ====================
# ``GeoBox`` is kept as a plain alias for ``GeoGrid`` so external callers and
# isinstance checks keep working. The class identity is identical.

GeoBox = GeoGrid


class GeoBoxTileGrid(OdcGeoboxTiles):
    """Tile grid wrapper for GeoGrid."""

    def __init__(self, geogrid: GeoGrid, chunk_shape: tuple[int, int]) -> None:
        """Initialize tile grid."""
        if len(chunk_shape) != 2 or any(size <= 0 for size in chunk_shape):
            raise ValueError("chunk_shape must contain two positive integers")
        super().__init__(geogrid, chunk_shape)
        self.geogrid = geogrid
        # backwards-compat attribute
        self.geobox = geogrid

    def tile_bounds(self, ij: TileIndex) -> BoundingBox:
        """Return bounds of a tile."""
        tile_geogrid = self[ij]
        bb = tile_geogrid.boundingbox
        return BoundingBox(bb.left, bb.bottom, bb.right, bb.top, crs=self.geogrid.crs)


__all__ = ["GeoBox", "GeoBoxTileGrid", "GeoGrid", "TileIndex"]
