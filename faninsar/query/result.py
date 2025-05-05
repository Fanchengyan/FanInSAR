"""Result classes for the queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import numpy as np
    from rasterio.transform import Affine

    from .query import GeoQuery


class BaseResult:
    """Base class for the result of the queries."""

    def __init__(
        self,
        result: dict,
    ) -> None:
        """Initialize the BaseResult instance.

        Parameters
        ----------
        result : dict
            The result of the query.

        """
        self.result = result

    def __repr__(self) -> str:
        """Return the string representation of the instance."""
        return f"{self.__class__.__name__}{self.dims}"

    def __str__(self) -> str:
        """Return the string representation of the instance."""
        return f"{self.__class__.__name__}{self.dims}"

    def __getitem__(self, item: int | slice) -> pd.Series | pd.DataFrame:
        """Get the item from the result."""
        return self.frame.iloc[item, :]

    def __iter__(self) -> iter:
        """Iterate over the result."""
        return iter(self.frame.iterrows())

    def __len__(self) -> int:
        """Return the length of the result."""
        return len(self.data)

    @property
    def data(self) -> list[np.ndarray] | None:
        """List of numpy arrays."""
        if self.result is None:
            return []
        return self.result["data"]

    @property
    def dims(self) -> str | None:
        """Description of the dimensions."""
        if self.result is None:
            return (None,)
        return self.result["dims"]

    @property
    def frame(self) -> pd.DataFrame | None:
        """DataFrame of the result."""
        if self.result is None:
            return None
        return pd.DataFrame(
            {
                "data": self.data,
                "transforms": self.transforms,
            },
            dtype="O",
        )

    @property
    def is_empty(self) -> bool:
        """If the result is empty."""
        return len(self.data) == 0


class PointsResult(BaseResult):
    """A class to manage the result of :class:`~faninsar.query.Points` query."""

    def __getitem__(self, item: int | slice) -> pd.Series:
        """Get the item from the result."""
        return self.result[item]


class BBoxesResult(BaseResult):
    """A class to manage the result of :class:`~faninsar.query.BoundingBox` query."""

    @property
    def transforms(self) -> list[Affine] | None:
        """List of affine transforms."""
        if self.result is None:
            return None
        return self.result["transforms"]


class PolygonsResult(BBoxesResult):
    """A class to manage the result of :class:`~faninsar.query.Polygons` query."""

    @property
    def transforms(self) -> list[Affine] | None:
        """List of affine transforms."""
        if self.result is None:
            return None
        return self.result["transforms"]

    @property
    def masks(self) -> list[np.ndarray] | None:
        """List of masks."""
        if self.result is None:
            return None
        return self.result["masks"]


class QueryResult:
    """A combined result of Queries.

    the :class:`PointsResult`, :class:`BBoxesResult`, and
    :class:`PolygonsResult` queries. This class is the default return type of the
    :ref:`query` results for the datasets.
    """

    _points: PointsResult | None
    _boxes: BBoxesResult | None
    _polygons: PolygonsResult | None
    _query: GeoQuery | None

    __slots__ = ["_boxes", "_points", "_polygons", "_query"]

    def __init__(
        self,
        points: PointsResult | dict | None = None,
        boxes: BBoxesResult | dict | None = None,
        polygons: PolygonsResult | dict | None = None,
        query: GeoQuery = None,
    ) -> None:
        """Initialize the QueryResult instance.

        Parameters
        ----------
        points : PointsResult, optional
            Result of the :class:`~faninsar.query.Points` query.
        boxes : BBoxesResult, optional
            Result of the :class:`~faninsar.query.BoundingBox` query.
        polygons : PolygonsResult, optional
            Result of the :class:`~faninsar.query.Polygons` query.
        query : GeoQuery, optional
            The :class:`~faninsar.query.GeoQuery` instance used to generate results.

        """
        if isinstance(points, dict):
            points = PointsResult(points)
        if isinstance(boxes, dict):
            boxes = BBoxesResult(boxes)
        if isinstance(polygons, dict):
            polygons = PolygonsResult(polygons)

        self._points = points
        self._boxes = boxes
        self._polygons = polygons
        self._query = query

    def __repr__(self) -> str:
        """Return the string representation of the instance."""
        return (
            "QueryResult("
            f"\n    points={self.points},"
            f"\n    boxes={self.boxes},"
            f"\n    polygons={self.polygons},"
            f"\n    query={self.query}"
            "\n)"
        )

    def __str__(self) -> str:
        """Return the string representation of the instance."""
        return (
            f"QueryResult(points={self.points}, boxes={self.boxes}, "
            f"polygons={self.polygons})"
        )

    @property
    def points(self) -> PointsResult | None:
        """Result of the :class:`~faninsar.query.Points` query."""
        return self._points

    @property
    def boxes(self) -> BBoxesResult | None:
        """Result of the :class:`~faninsar.query.BoundingBox` query."""
        return self._boxes

    @property
    def polygons(self) -> PolygonsResult | None:
        """Result of the :class:`~faninsar.query.Polygons` query."""
        return self._polygons

    @property
    def query(self) -> GeoQuery | None:
        """The :class:`~faninsar.query.GeoQuery` instance used to generate results."""
        return self._query
