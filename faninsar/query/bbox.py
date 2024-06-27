from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import overload

import geopandas as gpd
from rasterio.crs import CRS
from rasterio.warp import transform_bounds


class BoundingBox:
    """a class used for indexing datasets using a spatial bounding box.

    Note:
        This class is a modified version of the BoundingBox class from the torchgeo package.
        The main modifications include:

        - Removal of ``date bounds``
        - Change of the bounding box to (left, bottom, right, top), which aligns with the :class:`rasterio.coords.BoundingBox` class.
        - Addition of the CRS attribute to automatically convert the bounding box to the CRS of the dataset.
    """

    def __init__(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
        crs: CRS | str | None = None,
    ) -> None:
        """Initialize a BoundingBox.

        Parameters
        ----------
        left : float
            The western boundary.
        bottom : float
            The southern boundary.
        right : float
            The eastern boundary.
        top : float
            The northern boundary.
        crs : CRS | str | None, optional
            The coordinate reference system of the bounding box. Can be any object
            that can be passed to :meth:`pyproj.crs.CRS.from_user_input`.
            Default is None.
        """
        if left > right:
            raise ValueError(
                f"Bounding box is invalid: 'left={left}' > 'right={right}'"
            )
        if bottom > top:
            raise ValueError(
                f"Bounding box is invalid: 'bottom={bottom}' > 'top={top}'"
            )

        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.crs = crs

    def __str__(self) -> str:
        return f"BoundingBox(left={self.left}, bottom={self.bottom}, right={self.right}, top={self.top}, crs={self.crs})"

    def __repr__(self) -> str:
        return self.__str__()

    # https://github.com/PyCQA/pydocstyle/issues/525
    @overload
    def __getitem__(self, key: int) -> float:  # noqa: D105
        pass

    @overload
    def __getitem__(self, key: slice) -> list[float]:  # noqa: D105
        pass

    def __getitem__(self, key: int | slice) -> float | list[float]:
        """Index the (left, bottom, right,  top) tuple.

        Args:
            key: integer or slice object

        Returns:
            the value(s) at that index

        Raises:
            IndexError: if key is out of bounds
        """
        return [self.left, self.bottom, self.right, self.top][key]

    def __iter__(self) -> Iterator[float]:
        """Container iterator.

        Returns:
            iterator object that iterates over all objects in the container
        """
        yield from [self.left, self.bottom, self.right, self.top]

    def __contains__(self, other: "BoundingBox") -> bool:
        """Whether or not other is within the bounds of this bounding box.

        Args:
            other: another bounding box

        Returns:
            True if other is within this bounding box, else False
        """
        return (
            (self.left <= other.left <= self.right)
            and (self.left <= other.right <= self.right)
            and (self.bottom <= other.bottom <= self.top)
            and (self.bottom <= other.top <= self.top)
        )

    def __or__(self, other: "BoundingBox") -> "BoundingBox":
        """The union operator.

        Args:
            other: another bounding box

        Returns:
            the minimum bounding box that contains both self and other
        """
        other, crs_new = self._ensure_points_crs(other)
        return BoundingBox(
            min(self.left, other.left),
            max(self.right, other.right),
            min(self.bottom, other.bottom),
            max(self.top, other.top),
            crs=crs_new,
        )

    def __and__(self, other: "BoundingBox") -> "BoundingBox":
        """The intersection operator.

        Args:
            other: another bounding box

        Returns:
            the intersection of self and other

        Raises:
            ValueError: if self and other do not intersect
        """
        try:
            other, crs_new = self._ensure_points_crs(other)
            return BoundingBox(
                max(self.left, other.left),
                min(self.right, other.right),
                max(self.bottom, other.bottom),
                min(self.top, other.top),
                crs=crs_new,
            )
        except ValueError:
            raise ValueError(f"Bounding boxes {self} and {other} do not overlap")

    def _ensure_points_crs(self, other: "BoundingBox"):
        """Ensure the coordinate reference system of the bbox are the same."""
        if self.crs != other.crs:
            if self.crs is None or other.crs is None:
                crs_new = self.crs or other.crs
                warnings.warn(
                    "Cannot find the coordinate reference system of the bbox. "
                    "The crs of two bbox will assume to be the same. "
                )
            else:
                other = other.to_crs(self.crs)
                crs_new = self.crs
        else:
            crs_new = self.crs
        return other, crs_new

    @property
    def area(self) -> float:
        """Area of bounding box.

        Area is defined as spatial area.

        Returns:
            area
        """
        return (self.right - self.left) * (self.top - self.bottom)

    def to_crs(self, crs: CRS | str) -> "BoundingBox":
        """Convert the bounding box to a new coordinate reference system.

        Parameters
        ----------
        crs : CRS | str
            The new coordinate reference system. Can be any object that can be
            passed to :meth:`pyproj.crs.CRS.from_user_input`.
        """
        if self.crs is None:
            raise ValueError(
                "The current coordinate reference system is None. "
                "Please set the crs using set_crs() first."
            )
        crs = CRS.from_user_input(crs)
        if self.crs == crs:
            return self
        else:
            left, bottom, right, top = transform_bounds(
                self.crs, crs, self.left, self.bottom, self.right, self.top
            )
            return BoundingBox(left, bottom, right, top, crs=crs)

    def set_crs(self, crs: CRS | str) -> None:
        """Set the coordinate reference system of the bounding box.

        Parameters
        ----------
        crs : CRS | str
            The new coordinate reference system. Can be any object that can be
            passed to :meth:`pyproj.crs.CRS.from_user_input`.

            .. warning::

                This method will only set the crs attribute without converting the
                bounding box to a new coordinate reference system. If you want to convert
                the bounding box values to a new coordinate, please use :meth:`to_crs`
        """
        self.crs = CRS.from_user_input(crs)

    def to_dict(self) -> dict[str, float]:
        """Convert the bounding box to a dictionary.

        Returns:
            dictionary with keys 'left', 'bottom', 'right', 'top'
        """
        return {
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
            "top": self.top,
        }

    def to_GeoDataFrame(self) -> gpd.GeoDataFrame:
        """Convert the bounding box to a GeoDataFrame.

        Returns:
            GeoDataFrame with the bounding box as a polygon
        """
        import geopandas as gpd
        from shapely.geometry import box

        gdf = gpd.GeoDataFrame(
            geometry=[box(self.left, self.bottom, self.right, self.top)]
        )
        gdf.crs = self.crs
        return gdf

    def intersects(self, other: "BoundingBox") -> bool:
        """Whether or not two bounding boxes intersect.

        Args:
            other: another bounding box

        Returns:
            True if bounding boxes intersect, else False
        """
        return (
            self.left <= other.right
            and self.right >= other.left
            and self.bottom <= other.top
            and self.top >= other.bottom
        )

    def split(
        self, proportion: float, horizontal: bool = True
    ) -> tuple["BoundingBox", "BoundingBox"]:
        """Split BoundingBox in two.

        Args:
            proportion: split proportion in range (0,1)
            horizontal: whether the split is horizontal or vertical

        Returns:
            A tuple with the resulting BoundingBoxes
        """
        if not (0.0 < proportion < 1.0):
            raise ValueError("Input proportion must be between 0 and 1.")

        if horizontal:
            w = self.right - self.left
            splitx = self.left + w * proportion
            bbox1 = BoundingBox(self.left, splitx, self.bottom, self.top)
            bbox2 = BoundingBox(splitx, self.right, self.bottom, self.top)
        else:
            h = self.top - self.bottom
            splity = self.bottom + h * proportion
            bbox1 = BoundingBox(self.left, self.right, self.bottom, splity)
            bbox2 = BoundingBox(self.left, self.right, splity, self.top)

        return bbox1, bbox2

    def buffer(self, distance: float) -> "BoundingBox":
        """Buffer the bounding box.

        Parameters
        ----------
        distance: float
            the buffer distance in the units of the bounding box

        Returns:
            the buffered bounding box
        """
        return BoundingBox(
            self.left - distance,
            self.bottom - distance,
            self.right + distance,
            self.top + distance,
            crs=self.crs,
        )
