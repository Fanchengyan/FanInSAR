import numpy as np
import pandas as pd
from rasterio.transform import Affine


class BaseResult:
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
    def __getitem__(self, item):
        return self.result[item]

    def __repr__(self):
        if self.result is None:
            return "Points(None)"
        return (
            "Points("
            f"\n    n_files={self.data.shape[0]},"
            f"\n    n_points={self.data.shape[1]},"
            f"\n    dims={self.dims}"
            "\n)"
        )

    def __str__(self):
        if self.result is None:
            return "Points(None)"
        return f"Points(n_files={self.data.shape[0]}, n_points={self.data.shape[1]}, dims={self.dims})"


class BBoxesResult(BaseResult):

    def __repr__(self):
        if self.result is None:
            return "BoundingBox(None)"
        return (
            "BoundingBox("
            f"\n    n_boxes={len(self.data)},"
            f"\n    n_files={self.data[0].shape[0]},"
            f"\n    dims={self.dims}\n)"
        )

    def __str__(self):
        if self.result is None:
            return "BoundingBox(None)"
        return f"BoundingBox(n_boxes={len(self.data)}, n_files={self.data[0].shape[0]}, dims={self.dims})"

    @property
    def transforms(self) -> list[Affine] | None:
        if self.result is None:
            return None
        return self.result["transforms"]


class PolygonsResult(BBoxesResult):

    def __repr__(self):
        if self.result is None:
            return "Polygons(None)"
        return (
            "Polygons("
            f"\n    n_polygons={len(self.data)},"
            f"\n    n_files={self.data[0].shape[0]},"
            f"\n    dims={self.dims}\n)"
        )

    def __str__(self):
        if self.result is None:
            return "Polygons(None)"
        return f"Polygons(n_boxes={len(self.data)}, n_files={self.data[0].shape[0]}, dims={self.dims})"


class QueryResult:
    _points = None
    _boxes = None
    _polygons = None

    def __init__(
        self,
        points=None,
        boxes=None,
        polygons=None,
    ):
        self._points = PointsResult(points)
        self._boxes = BBoxesResult(boxes)
        self._polygons = PolygonsResult(polygons)

    def __repr__(self):
        return (
            "QueryResult("
            f"\n    points={self.points},"
            f"\n    boxes={self.boxes},"
            f"\n    polygons={self.polygons}"
            "\n)"
        )

    def __str__(self):
        return f"QueryResult(points={self.points}, boxes={self.boxes}, polygons={self.polygons})"

    @property
    def points(self) -> PointsResult | None:
        return self._points

    @property
    def boxes(self) -> BBoxesResult | None:
        return self._boxes

    @property
    def polygons(self) -> PolygonsResult | None:
        return self._polygons
