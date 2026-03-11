"""GeoBox utilities for FanInSAR lazy geoprocessing.

Notes
-----
This module provides a thin extension of odc-geo's GeoBox with FanInSAR-specific
compatibility methods for BoundingBox-based APIs.

See Also
--------
odc.geo.GeoBox : Base class with full geobox functionality
faninsar.query.bbox : BoundingBox primitives

"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from odc.geo import GeoBox as OdcGeoBox
from odc.geo.geobox import GeoboxTiles as OdcGeoboxTiles
from odc.geo.geobox import pixel_translation
from odc.geo.math import is_almost_int
from rasterio.transform import rowcol
from rasterio.windows import Window
from rasterio.windows import transform as _window_transform

from faninsar.logging import setup_logger
from faninsar.query.bbox import BoundingBox

if TYPE_CHECKING:
    from affine import Affine

TileIndex: TypeAlias = tuple[int, int]

logger = setup_logger(__name__)


class GeoBox(OdcGeoBox):
    """Extended GeoBox with FanInSAR-specific compatibility methods.

    This class extends odc-geo's GeoBox, adding compatibility methods for
    FanInSAR's existing codebase.

    See Also
    --------
    odc.geo.GeoBox : Base class with full documentation

    """

    @classmethod
    def from_odc_geobox(cls, geobox: OdcGeoBox) -> GeoBox:
        """Create a GeoBox from an odc-geo GeoBox."""
        return cls(shape=geobox.shape, affine=geobox.affine, crs=geobox.crs)

    # ==================== Class Factory Methods ====================
    # Wrap all @staticmethod factory methods to return the subclass

    @classmethod
    def from_bbox(cls, *args, **kwargs) -> GeoBox:
        """Construct GeoBox from a bounding box.

        Returns the FanInSAR GeoBox subclass instead of odc.geo.GeoBox.
        See odc.geo.GeoBox.from_bbox for full documentation.
        """
        odc_geobox = OdcGeoBox.from_bbox(*args, **kwargs)
        return cls.from_odc_geobox(odc_geobox)

    @classmethod
    def from_geopolygon(cls, *args, **kwargs) -> GeoBox:
        """Construct GeoBox from a polygon.

        Returns the FanInSAR GeoBox subclass instead of odc.geo.GeoBox.
        See odc.geo.GeoBox.from_geopolygon for full documentation.
        """
        odc_geobox = OdcGeoBox.from_geopolygon(*args, **kwargs)
        return cls.from_odc_geobox(odc_geobox)

    @classmethod
    def from_rio(cls, *args, **kwargs) -> GeoBox:
        """Construct GeoBox from rasterio DatasetReader.

        Returns the FanInSAR GeoBox subclass instead of odc.geo.GeoBox.
        See odc.geo.GeoBox.from_rio for full documentation.
        """
        odc_geobox = OdcGeoBox.from_rio(*args, **kwargs)
        return cls.from_odc_geobox(odc_geobox)

    @property
    def bounds(self) -> BoundingBox:
        """Get the BoundingBox representation of the bounds of GeoBox."""
        return BoundingBox(*self.boundingbox.bbox, crs=self.crs)

    @property
    def y(self) -> np.ndarray:
        """Get the y coordinates of the GeoBox."""
        y = "y" if "y" in self.dims else "lat"
        return self.coords[y].values

    @property
    def x(self) -> np.ndarray:
        """Get the x coordinates of the GeoBox."""
        x = "x" if "x" in self.dims else "lon"
        return self.coords[x].values

    def get_overlap(self, other: GeoBox) -> BoundingBox | None:
        """Get the overlap between this GeoBox and another GeoBox.

        Parameters
        ----------
        other : GeoBox
            GeoBox to test against.

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

    def is_aligned(self, other: GeoBox, tol: float = 1e-6) -> bool:
        """Check if two grids share CRS, resolution, and grid alignment.

        Parameters
        ----------
        other : GeoBox
            Grid to compare against.
        tol : float, optional
            Numerical tolerance for alignment.

        Returns
        -------
        bool
            True when both GeoBoxes share CRS, resolution, and are aligned.

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

        def wrapped_method(self, *args, **kwargs) -> GeoBox:  # noqa: ANN001
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

        def wrapped_getter(self) -> GeoBox:  # noqa: ANN001
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

    def __getitem__(self, roi: BoundingBox) -> GeoBox:
        """Index/slice operation returning a cropped GeoBox."""
        shape_, affine_ = self.compute_crop(roi)
        return self.__class__(shape=shape_, affine=affine_, crs=self._crs)

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


# Apply method wrapping to the GeoBox class
GeoBox._apply_method_wrapping()


# Re-export for compatibility
class GeoBoxTileGrid(OdcGeoboxTiles):
    """Tile grid wrapper for GeoBox."""

    def __init__(self, geobox: GeoBox, chunk_shape: tuple[int, int]) -> None:
        """Initialize tile grid."""
        super().__init__(geobox, chunk_shape)
        self.geobox = geobox

    def tile_bounds(self, ij: TileIndex) -> BoundingBox:
        """Return bounds of a tile."""
        tile_geobox = self[ij]
        bb = tile_geobox.boundingbox
        return BoundingBox(bb.left, bb.bottom, bb.right, bb.top, crs=self.geobox.crs)


__all__ = ["GeoBox", "GeoBoxTileGrid", "TileIndex"]
