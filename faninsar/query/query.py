from __future__ import annotations

from .bbox import BoundingBox
from .points import Points
from .polygons import Polygons


class GeoQuery:
    """A combined query of the :class:`Points`, :class:`BoundingBox`, and
    :class:`Polygons` queries. This class is used to sample data from a
    GeoDataset using multiple points, bounding boxes, and polygons at the same
    time.
    """

    _points: Points | None
    _boxes: BoundingBox | list[BoundingBox] | None
    _polygons: Polygons | None

    __slots__ = ["_points", "_boxes", "_polygons"]

    def __init__(
        self,
        points: Points | None = None,
        boxes: BoundingBox | list[BoundingBox] | None = None,
        polygons: Polygons | None = None,
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

        Raises
        ------
        ValueError:
            If neither points, boxes, nor polygons are provided.
        TypeError:
            If points is not a Points instance, boxes is not a BoundingBox or a
            list of BoundingBox instances, or polygons is not a Polygons instance.
        """
        if boxes is None and points is None and polygons is None:
            raise ValueError("One of boxes, points or polygons must be provided.")

        if points is not None and not isinstance(points, Points):
            raise TypeError(f"points must be a Points. Got {type(points)}")
        if boxes is not None:
            if isinstance(boxes, BoundingBox):
                boxes = [boxes]
            if not isinstance(boxes, list):
                try:
                    boxes = list(boxes)
                except TypeError:
                    raise TypeError(
                        f"boxes must be a BoundingBox or a list of BoundingBox. Got {type(boxes)}"
                    )
        if polygons is not None:
            if not isinstance(polygons, Polygons):
                raise TypeError(f"polygons must be a Polygons. Got {type(polygons)}")
            if polygons.is_mixed:
                polygons = polygons.to_desired()

        self._points = points
        self._boxes = boxes
        self._polygons = polygons

    def __str__(self) -> str:
        boxes = f"[{len(self.boxes)} BoundingBox]" if self.boxes is not None else None
        points = self.points if self.points is not None else None
        return f"GeoQuery(points={points}, boxes={boxes}, polygons={self.polygons})"

    def __repr__(self) -> str:
        boxes = f"[{len(self.boxes)} BoundingBox]" if self.boxes is not None else None
        points = self.points if self.points is not None else None
        return (
            "GeoQuery("
            f"\n    points={points}"
            f"\n    boxes={boxes}"
            f"\n    polygons={self.polygons}"
            f"\n)"
        )

    @property
    def points(self) -> Points | None:
        """Return the points of the samples."""
        return self._points

    @property
    def boxes(self) -> list[BoundingBox] | None:
        """Return the bounding boxes of the samples."""
        return self._boxes

    @property
    def polygons(self) -> Points | None:
        """Return the polygons of the samples."""
        return self._polygons
