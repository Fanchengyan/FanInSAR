"""A combined query for geo-spatial data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .bbox import BoundingBox
from .points import Points
from .polygons import Polygons

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np


class GeoQuery:
    """A combined query for geo-spatial data.

    the :class:`Points`, :class:`BoundingBox`, and :class:`Polygons` queries.
    This class is used to sample data from a GeoDataset using multiple points,
    bounding boxes, and polygons at the same time.
    """

    _points: Points | None
    _boxes: BoundingBox | list[BoundingBox] | None
    _polygons: Polygons | None
    _indexes: int | Iterable[int] | np.ndarray | None

    __slots__ = ["_boxes", "_indexes", "_points", "_polygons"]

    def __init__(
        self,
        points: Points | None = None,
        boxes: BoundingBox | list[BoundingBox] | None = None,
        polygons: Polygons | None = None,
        indexes: int | Iterable[int] | np.ndarray | None = None,
    ) -> None:
        """Initialize a GeoQuery instance.

        Parameters
        ----------
        points : Points | None, optional
            The Points instance used to retrieve point values from the dataset.
            Default is None.
        boxes : BoundingBox | list[BoundingBox] | None, optional
            The BoundingBox or a list of BoundingBox instances used to retrieve
            bounding box values from the dataset. Default is None.
        polygons: Polygons | None, optional
            The Polygons instance used to retrieve polygon values from the dataset.
            Default is None.
        indexes : int | Iterable[int] | np.ndarray | None, optional
            Indexes of files to query. If None, all valid files are queried.
            Default is None.

        Raises
        ------
        ValueError:
            If neither points, boxes, nor polygons are provided.
        TypeError:
            If points is not a Points instance, boxes is not a BoundingBox or a
            list of BoundingBox instances, or polygons is not a Polygons instance.

        """
        if boxes is None and points is None and polygons is None:
            msg = "One of boxes, points or polygons must be provided."
            raise ValueError(msg)

        if points is not None and not isinstance(points, Points):
            msg = f"points must be a Points. Got {type(points)}"
            raise TypeError(msg)
        if boxes is not None and not isinstance(boxes, (list, BoundingBox)):
            try:
                boxes = list(boxes)
            except TypeError as e:
                msg = (
                    "boxes must be a BoundingBox or a list of "
                    f"BoundingBox. Got {type(boxes)}"
                )
                raise TypeError(msg) from e
        if polygons is not None:
            if not isinstance(polygons, Polygons):
                msg = f"polygons must be a Polygons. Got {type(polygons)}"
                raise TypeError(msg)
            if polygons.is_mixed:
                polygons = polygons.to_desired()

        self._points = points
        self._boxes = boxes
        self._polygons = polygons
        self._indexes = indexes

    def __str__(self) -> str:
        """Return the string representation of the GeoQuery instance."""
        boxes = None
        if self.boxes is not None:
            if isinstance(self.boxes, BoundingBox):
                boxes = "1 BoundingBox"
            else:
                boxes = f"[{len(self.boxes)} BoundingBox]"
        points = self.points if self.points is not None else None
        return (
            f"GeoQuery(points={points}, boxes={boxes}, "
            f"polygons={self.polygons}, indexes={self.indexes})"
        )

    def __repr__(self) -> str:
        """Return the string representation of the GeoQuery instance."""
        boxes = None
        if self.boxes is not None:
            if isinstance(self.boxes, BoundingBox):
                boxes = "1 BoundingBox"
            else:
                boxes = f"[{len(self.boxes)} BoundingBox]"
        points = self.points if self.points is not None else None
        return (
            "GeoQuery("
            f"\n    points={points}"
            f"\n    boxes={boxes}"
            f"\n    polygons={self.polygons}"
            f"\n    indexes={self.indexes}"
            f"\n)"
        )

    @property
    def points(self) -> Points | None:
        """The points used to sample the dataset."""
        return self._points

    @property
    def boxes(self) -> list[BoundingBox] | BoundingBox | None:
        """The bounding boxes used to sample the dataset."""
        return self._boxes

    @property
    def polygons(self) -> Polygons | None:
        """The polygons used to sample the dataset."""
        return self._polygons

    @property
    def indexes(self) -> int | Iterable[int] | np.ndarray | None:
        """Indexes of files to query. None means all valid files."""
        return self._indexes
