from __future__ import annotations

from .bbox import BoundingBox
from .points import Points
from .polygons import Polygons


class GeoQuery:
    """A class to represent a collection of bbox(es) or/and points queries that
    will be used to sample data from a GeoDataset.

    .. Note::
        This class is designed to sample data using multiple bounding boxes and
        points at the same time to improve the efficiency and reduce the IO
        overhead. You are recommended to use this class to sample data from a
        GeoDataset instead of using BoundingBox(es) and Points separately.
    """

    _points: Points | None
    _bbox: BoundingBox | list[BoundingBox] | None
    _polygons: Polygons | None

    __slots__ = ["_points", "_bbox", "_polygons"]

    def __init__(
        self,
        points: Points | None = None,
        bbox: BoundingBox | list[BoundingBox] | None = None,
        polygons: Polygons | None = None,
    ) -> None:
        """Initialize a sampler.

        Parameters
        ----------
        points : Points | None, optional
            The :class:`Points` for querying the samples. Default is None.
        bbox : BoundingBox | list[BoundingBox] | None, optional
            The :class:`BoundingBox` or a list of :class:`BoundingBox`. for querying
            the samples. Default is None.
        polygons: Polygons | None, optional
            The :class:`Polygons` for querying the samples. If None, the samples
            will be queried from the bbox. Default is None.

        Raises
        ------
        ValueError:
            If both bbox and points are None.
        TypeError:
            If bbox is not a BoundingBox or a list of BoundingBox.
        """
        if bbox is None and points is None and polygons is None:
            raise ValueError("One of bbox, points or polygons must be provided.")

        if points is not None and not isinstance(points, Points):
            raise TypeError(f"points must be a Points. Got {type(points)}")
        if bbox is not None:
            if isinstance(bbox, BoundingBox):
                bbox = [bbox]
            if not isinstance(bbox, list):
                try:
                    bbox = list(bbox)
                except TypeError:
                    raise TypeError(
                        f"bbox must be a BoundingBox or a list of BoundingBox. Got {type(bbox)}"
                    )
        if polygons is not None:
            if not isinstance(polygons, Polygons):
                raise TypeError(f"polygons must be a Polygons. Got {type(polygons)}")
            if polygons.is_mixed:
                polygons = polygons.to_desired()

        self._points = points
        self._bbox = bbox
        self._polygons = polygons

    def __str__(self) -> str:
        bbox = f"[{len(self.bbox)} BoundingBox]" if self.bbox is not None else None
        points = self.points if self.points is not None else None
        return f"GeoQuery(points={points}, bbox={bbox}, polygons={self.polygons})"

    def __repr__(self) -> str:
        bbox = f"[{len(self.bbox)} BoundingBox]" if self.bbox is not None else None
        points = self.points if self.points is not None else None
        return (
            "GeoQuery("
            f"\n    points={points}"
            f"\n    bbox={bbox}"
            f"\n    polygons={self.polygons}"
            f"\n)"
        )

    @property
    def points(self) -> Points | None:
        """Return the points of the samples."""
        return self._points

    @property
    def bbox(self) -> list[BoundingBox] | None:
        """Return the bounding boxes of the samples."""
        return self._bbox

    @property
    def polygons(self) -> Points | None:
        """Return the polygons of the samples."""
        return self._polygons
