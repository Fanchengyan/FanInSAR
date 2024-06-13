from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio.transform import Affine

from .query import GeoQuery


class BaseResult:
    """Base class for the result of the queries."""

    def __init__(self, result):
        self.result = result

    def __getitem__(self, item: int | slice) -> pd.Series | pd.DataFrame:
        return self.frame.iloc[item, :]

    def __iter__(self):
        return iter(self.frame.iterrows())

    def __len__(self):
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
            return None
        return self.result["dims"]

    @property
    def frame(self) -> pd.DataFrame | None:
        """DataFrame of the result."""
        if self.result is None:
            return None
        df = pd.DataFrame(
            {
                "data": self.data,
                "transforms": self.transforms,
            },
            dtype="O",
        )
        return df

    @property
    def is_empty(self) -> bool:
        """if the result is empty."""
        return len(self.data) == 0


class PointsResult(BaseResult):
    """A class to manage the result of :class:`~faninsar.query.Points` query."""

    def __getitem__(self, item):
        return self.result[item]

    def __repr__(self):
        if self.result is None:
            return "PointsResult(None)"
        return (
            "PointsResult("
            f"\n    n_files={self.data.shape[0]},"
            f"\n    n_points={self.data.shape[1]},"
            f"\n    dims={self.dims}"
            "\n)"
        )

    def __str__(self):
        if self.result is None:
            return "PointsResult(None)"
        return f"PointsResult(n_files={self.data.shape[0]}, n_points={self.data.shape[1]}, dims={self.dims})"


class BBoxesResult(BaseResult):
    """A class to manage the result of :class:`~faninsar.query.BoundingBox` query."""

    def __repr__(self):
        if self.result is None:
            return "BBoxesResult(None)"
        return (
            "BBoxesResult("
            f"\n    n_boxes={len(self.data)},"
            f"\n    n_files={self.data[0].shape[0]},"
            f"\n    dims={self.dims}\n)"
        )

    def __str__(self):
        if self.result is None:
            return "BBoxesResult(None)"
        return f"BBoxesResult(n_boxes={len(self.data)}, n_files={self.data[0].shape[0]}, dims={self.dims})"

    @property
    def transforms(self) -> list[Affine] | None:
        if self.result is None:
            return None
        return self.result["transforms"]


class PolygonsResult(BBoxesResult):
    """A class to manage the result of :class:`~faninsar.query.Polygons` query."""

    def __repr__(self):
        if self.result is None:
            return "PolygonsResult(None)"
        return (
            "PolygonsResult("
            f"\n    n_polygons={len(self.data)},"
            f"\n    n_files={self.data[0].shape[0]},"
            f"\n    dims={self.dims}\n)"
        )

    def __str__(self):
        if self.result is None:
            return "PolygonsResult(None)"
        return f"PolygonsResult(n_boxes={len(self.data)}, n_files={self.data[0].shape[0]}, dims={self.dims})"

    @property
    def transforms(self) -> list[Affine] | None:
        if self.result is None:
            return None
        return self.result["transforms"]

    @property
    def masks(self) -> list[np.ndarray] | None:
        if self.result is None:
            return None
        return self.result["masks"]


class QueryResult:
    """A combined result of the :class:`PointsResult`, :class:`BBoxesResult`, and
    :class:`PolygonsResult` queries. This class is the default return type of the
    :ref:`query` results for the datasets.
    """

    _points: PointsResult | None
    _boxes: BBoxesResult | None
    _polygons: PolygonsResult | None
    _query: GeoQuery | None

    __slots__ = ["_points", "_boxes", "_polygons", "_query"]

    def __init__(
        self,
        points=None,
        boxes=None,
        polygons=None,
        query=None,
    ):
        self._points = PointsResult(points)
        self._boxes = BBoxesResult(boxes)
        self._polygons = PolygonsResult(polygons)
        self._query = query

    def __repr__(self):
        return (
            "QueryResult("
            f"\n    points={self.points},"
            f"\n    boxes={self.boxes},"
            f"\n    polygons={self.polygons},"
            f"\n    query={self.query}"
            "\n)"
        )

    def __str__(self):
        return f"QueryResult(points={self.points}, boxes={self.boxes}, polygons={self.polygons})"

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
