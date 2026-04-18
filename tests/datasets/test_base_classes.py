"""Unit tests targeting the refactored dataset base modules."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from lxml import etree
from pyproj.crs import CRS
from rasterio.transform import from_bounds

from faninsar._core.geo import Profile
from faninsar._core.sar.acquisition import Acquisition
from faninsar._core.sar.pairs import Pairs
from faninsar.datasets.base import (
    GeoDataset,
    PairDataset,
    RasterDataset,
    TimeSeriesDataset,
)
from faninsar.query import BoundingBox, Points


def _write_tile(
    path: Path,
    bounds: tuple[float, float, float, float],
    value: float,
    *,
    crs: CRS | None = None,
) -> None:
    """Create a small GeoTIFF tile for testing."""
    height = width = 4
    transform = from_bounds(*bounds, width, height)
    data = np.full((height, width), value, dtype=np.float32)
    crs = CRS.from_epsg(4326) if crs is None else crs
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(data, 1)


class ToyGeoDataset(GeoDataset):
    """Minimal GeoDataset implementation used to exercise mixins."""

    def __init__(self) -> None:
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


class SampleRasterDataset(RasterDataset):
    """RasterDataset variant with permissive filename matching."""

    pattern = "*.tif"


class SampleTimeSeriesDataset(TimeSeriesDataset):
    """Time-series dataset that parses dates from filenames."""

    pattern = "*.tif"

    @classmethod
    def parse_dates(cls, paths: list[str | Path]) -> Acquisition:
        dates = [
            np.datetime64(
                datetime.strptime(Path(path).stem.split("_")[0], "%Y%m%d"), "ns"
            )
            for path in paths
        ]
        return Acquisition(dates)


class SamplePairDataset(PairDataset):
    """Pair dataset that parses primary/secondary dates from filenames."""

    pattern = "*.tif"

    @classmethod
    def parse_pairs(cls, paths: Iterable[str | Path]) -> Pairs:
        parsed: list[tuple[np.datetime64, np.datetime64]] = []
        for path in paths:
            parts = Path(path).stem.split("_")
            primary = np.datetime64(datetime.strptime(parts[0], "%Y%m%d"), "ns")
            secondary = np.datetime64(datetime.strptime(parts[1], "%Y%m%d"), "ns")
            parsed.append((primary, secondary))
        return Pairs(parsed)


@pytest.fixture
def raster_root(tmp_path: Path) -> Path:
    """Populate a temporary directory with basic tiles."""
    bounds = (0.0, 0.0, 4.0, 4.0)
    for idx in range(3):
        _write_tile(tmp_path / f"tile_{idx:02d}.tif", bounds, value=idx)
    return tmp_path


@pytest.fixture
def time_series_root(tmp_path: Path) -> Path:
    """Create tiles with YYYYMMDD prefixes for time-series parsing."""
    bounds = (0.0, 0.0, 4.0, 4.0)
    for date, value in [("20210101", 1.0), ("20210113", 2.0)]:
        _write_tile(tmp_path / f"{date}_scene.tif", bounds, value=value)
    return tmp_path


@pytest.fixture
def pair_root(tmp_path: Path) -> Path:
    """Create tiles encoding interferometric pair names."""
    bounds = (0.0, 0.0, 4.0, 4.0)
    stems = ["20210101_20210113_pair", "20210113_20210125_pair"]
    for stem in stems:
        _write_tile(tmp_path / f"{stem}.tif", bounds, value=1.0)
    return tmp_path


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


def test_raster_dataset_scans_files(raster_root: Path) -> None:
    """RasterDataset should discover GeoTIFF tiles under the root directory."""
    ds = SampleRasterDataset(root_dir=raster_root, verbose=False)
    assert len(ds) == 3
    assert ds.files.valid.sum() == 3


def test_time_series_dataset_parses_dates(time_series_root: Path) -> None:
    """TimeSeriesDataset attaches parsed Acquisition metadata."""
    ds = SampleTimeSeriesDataset(root_dir=time_series_root, verbose=False)
    expected = [
        np.datetime64("2021-01-01T00:00:00"),
        np.datetime64("2021-01-13T00:00:00"),
    ]
    assert list(ds.dates.values) == expected
    assert ds.file_dim_name == "date"


def test_pair_dataset_parses_pairs(pair_root: Path) -> None:
    """PairDataset should store primary and secondary date metadata."""
    ds = SamplePairDataset(root_dir=pair_root, verbose=False)
    assert list(ds.pairs.to_names()) == [
        "20210101_20210113",
        "20210113_20210125",
    ]
    assert ds.file_dim_name == "pair"


def test_raster_dataset_array2kmz_tiled_reprojects_to_wgs84(
    tmp_path: Path,
) -> None:
    """RasterDataset tiled KMZ export should reproject to WGS84."""
    mercator = CRS.from_epsg(3857)
    bounds = (0.0, 0.0, 4000.0, 4000.0)
    _write_tile(tmp_path / "mercator_tile.tif", bounds, value=1.0, crs=mercator)

    dataset = SampleRasterDataset(root_dir=tmp_path, verbose=False)
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    out_file = tmp_path / "dataset_tiled_kmz.kmz"

    dataset.array2kmz(arr, out_file, render_scale=1, verbose=False)

    with zipfile.ZipFile(out_file) as kmz:
        names = set(kmz.namelist())
        root_tile = etree.fromstring(kmz.read("tiles/0/0/0.kml"))

    assert "doc.kml" in names
    assert "legend/colorbar.png" in names
    assert "tiles/0/0/0.kml" in names
    assert "tiles/0/0/0.png" in names

    west = float(root_tile.xpath("string(.//*[local-name()='west'][1])"))
    south = float(root_tile.xpath("string(.//*[local-name()='south'][1])"))
    east = float(root_tile.xpath("string(.//*[local-name()='east'][1])"))
    north = float(root_tile.xpath("string(.//*[local-name()='north'][1])"))

    assert west < east
    assert south < north
    assert -180.0 <= west <= 180.0
    assert -180.0 <= east <= 180.0
    assert -90.0 <= south <= 90.0
    assert -90.0 <= north <= 90.0
