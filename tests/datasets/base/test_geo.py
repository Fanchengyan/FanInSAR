"""Tests for GeoDataset base mixin."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj.crs import CRS
from rasterio.transform import from_bounds

from faninsar._core.geo import Profile
from faninsar.data.datasets.base import GeoDataset
from faninsar.data.query import BoundingBox, Points


class ToyGeoDataset(GeoDataset):
    """Minimal GeoDataset implementation used to exercise mixins."""

    def __init__(self) -> None:
        """Initialize with a fixed 10x10 WGS84 ROI."""
        super().__init__()
        self._crs = CRS.from_epsg(4326)
        self._res = (1.0, 1.0)
        self._dtype = np.dtype("float32")
        self._count = 1
        self._nodata = 0.0
        bbox = BoundingBox(0.0, 10.0, 0.0, 10.0, crs=self._crs)
        self._roi = bbox
        self._valid = np.array([True])
        self.index.insert(0, tuple(bbox), "toy")

    def get_profile(
        self,
        bbox: BoundingBox | str = "roi",
    ) -> Profile:
        """Return a mock profile for the given bbox."""
        bbox_obj = (
            self._ensure_bbox(bbox) if isinstance(bbox, BoundingBox) else self.roi
        )
        width = int(abs(bbox_obj.right - bbox_obj.left)) or 1
        height = int(abs(bbox_obj.top - bbox_obj.bottom)) or 1
        transform = from_bounds(
            bbox_obj.left,
            bbox_obj.bottom,
            bbox_obj.right,
            bbox_obj.top,
            width,
            height,
        )
        return Profile(
            width=width,
            height=height,
            transform=transform,
            crs=self._crs,
            nodata=self._nodata,
            count=self._count,
            dtype=self._dtype,
        )


def test_geo_dataset_roi_and_query_crs() -> None:
    """GeoDataset mixin logic should normalize ROI and query CRS."""
    ds = ToyGeoDataset()
    assert ds.bounds.left == 0.0
    new_roi = BoundingBox(-1.0, 5.0, -1.0, 5.0, crs=CRS.from_epsg(4326))
    ds.roi = new_roi
    assert ds.roi.left == pytest.approx(-1.0)

    mercator = CRS.from_epsg(3857)
    points = Points([(0.0, 0.0)], crs=mercator)
    converted = ds._ensure_query_crs(points)
    assert converted.crs == ds.crs
