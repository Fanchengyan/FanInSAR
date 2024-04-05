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

    _bbox: BoundingBox | list[BoundingBox]
    _points: Points | None

    __slots__ = ["_bbox", "_points"]

    def __init__(
        self,
        bbox: BoundingBox | list[BoundingBox] | None = None,
        points: Points | None = None,
    ) -> None:
        """Initialize a sampler.

        Parameters
        ----------
        bbox : BoundingBox | list[BoundingBox] | None, optional
            The :class:`BoundingBox` or a list of :class:`BoundingBox`. for querying
            the samples.If None, the samples will be queried from the points.
            default: None
        points : Optional[Points], optional
            The :class:`Points` for querying the samples. If None, the samples
            will be queried from the bbox. default: None

        Raises
        ------
        ValueError:
            If both bbox and points are None.
        TypeError:
            If bbox is not a BoundingBox or a list of BoundingBox.
        """
        if bbox is None and points is None:
            raise ValueError("bbox and points cannot be both None.")
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

        self._bbox = bbox
        self._points = points

    def __str__(self) -> str:
        bbox = f"[{len(self.bbox)} BoundingBox]" if self.bbox is not None else None
        points = self.points if self.points is not None else None
        return f"GeoQuery(bbox={bbox}, points={points})"

    def __repr__(self) -> str:
        bbox = f"[{len(self.bbox)} BoundingBox]" if self.bbox is not None else None
        points = self.points if self.points is not None else None
        return f"GeoQuery(\n    bbox={bbox}\n    points={points}\n)"

    @property
    def bbox(self) -> list[BoundingBox] | None:
        """Return the bounding boxes of the samples."""
        return self._bbox

    @property
    def points(self) -> Points | None:
        """Return the points of the samples."""
        return self._points
