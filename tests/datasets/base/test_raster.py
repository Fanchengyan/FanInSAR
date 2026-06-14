"""Tests for RasterDataset: file discovery, queries, and dask integration."""

from __future__ import annotations

import tempfile
import time
import zipfile
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from lxml import etree
from pyproj.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from faninsar.datasets.base import RasterDataset
from faninsar.query import BoundingBox, Points, Polygons

if TYPE_CHECKING:
    from pathlib import Path


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


class SampleRasterDataset(RasterDataset):
    """RasterDataset variant with permissive filename matching."""

    pattern = "*.tif"


@pytest.fixture
def raster_root(tmp_path: Path) -> Path:
    """Populate a temporary directory with basic tiles."""
    bounds = (0.0, 0.0, 4.0, 4.0)
    for idx in range(3):
        _write_tile(tmp_path / f"tile_{idx:02d}.tif", bounds, value=idx)
    return tmp_path


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with test GeoTIFF files."""
    bounds_list = [
        (0, 0, 10, 10),
        (5, 5, 15, 15),
        (10, 10, 20, 20),
        (15, 15, 25, 25),
        (20, 20, 30, 30),
    ]
    for i, bounds in enumerate(bounds_list):
        file_path = tmp_path / f"test_file_{i:02d}.tif"
        _write_tile(file_path, bounds, value=float(i))
    return tmp_path


def test_raster_dataset_scans_files(raster_root: Path) -> None:
    """RasterDataset should discover GeoTIFF tiles under the root directory."""
    ds = SampleRasterDataset(root_dir=raster_root, verbose=False)
    assert len(ds) == 3
    assert ds.files.valid.sum() == 3


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


class TestRasterDatasetDask:
    """Test RasterDataset with dask functionality."""

    def test_init_with_chunks_none(self, temp_dataset_dir: Path) -> None:
        """Eager mode when chunks is None."""
        ds = RasterDataset(root_dir=temp_dataset_dir, chunks=None, verbose=False)
        assert ds.is_lazy is False
        assert len(ds) == 5

    def test_init_with_chunks_dict(self, temp_dataset_dir: Path) -> None:
        """Lazy mode with explicit chunk sizes."""
        ds = RasterDataset(
            root_dir=temp_dataset_dir,
            chunks={"y": 256, "x": 256},
            verbose=False,
        )
        assert ds.is_lazy is True
        assert len(ds) == 5

    def test_init_with_chunks_auto(self, temp_dataset_dir: Path) -> None:
        """Lazy mode with auto-detected chunks."""
        ds = RasterDataset(root_dir=temp_dataset_dir, chunks="auto", verbose=False)
        assert ds.is_lazy is True
        assert ds._chunks == "auto"

    def test_init_with_chunks_int(self, temp_dataset_dir: Path) -> None:
        """Lazy mode with integer chunk size."""
        ds = RasterDataset(root_dir=temp_dataset_dir, chunks=512, verbose=False)
        assert ds.is_lazy is True
        assert ds._chunks == {"y": 512, "x": 512}

    def test_points_query_without_chunks(self, temp_dataset_dir: Path) -> None:
        """Points query in eager mode."""
        ds = RasterDataset(root_dir=temp_dataset_dir, chunks=None, verbose=False)
        points = Points([(5, 5), (15, 15), (25, 25)])
        result = ds.points_query(points)
        assert "data" in result
        assert result["data"].data.shape[1] == 3

    def test_points_query_with_chunks(self, temp_dataset_dir: Path) -> None:
        """Points query in lazy mode."""
        ds = RasterDataset(root_dir=temp_dataset_dir, chunks="auto", verbose=False)
        points = Points([(5, 5), (15, 15), (25, 25)])
        result = ds.points_query(points)
        assert "data" in result
        assert result["data"].data.shape[1] == 3

    def test_box_query_without_chunks(self, temp_dataset_dir: Path) -> None:
        """Bbox query in eager mode."""
        ds = RasterDataset(root_dir=temp_dataset_dir, chunks=None, verbose=False)
        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))
        result = ds.boxes_query(bbox)
        assert result.dataset is not None
        assert result.dataset["data"].ndim >= 2

    def test_box_query_with_chunks(self, temp_dataset_dir: Path) -> None:
        """Bbox query in lazy mode."""
        ds = RasterDataset(root_dir=temp_dataset_dir, chunks="auto", verbose=False)
        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))
        result = ds.boxes_query(bbox)
        assert result.dataset is not None
        assert result.dataset["data"].ndim >= 2

    def test_getitem_with_chunks_default(self, temp_dataset_dir: Path) -> None:
        """__getitem__ respects dataset chunk settings."""
        ds_no_dask = RasterDataset(
            root_dir=temp_dataset_dir, chunks=None, verbose=False,
        )
        points = Points([(5, 5), (15, 15)])
        result_no_dask = ds_no_dask[points]
        assert isinstance(result_no_dask, dict)
        assert "data" in result_no_dask

        ds_with_dask = RasterDataset(
            root_dir=temp_dataset_dir, chunks="auto", verbose=False,
        )
        result_with_dask = ds_with_dask[points]
        assert isinstance(result_with_dask, dict)
        assert "data" in result_with_dask

    def test_query_consistency(self, temp_dataset_dir: Path) -> None:
        """Lazy and eager queries produce consistent results."""
        ds_eager = RasterDataset(
            root_dir=temp_dataset_dir, chunks=None, verbose=False,
        )
        ds_lazy = RasterDataset(
            root_dir=temp_dataset_dir, chunks="auto", verbose=False,
        )

        bbox = BoundingBox(0, 30, 0, 30, crs=CRS.from_epsg(4326))
        res_eager = ds_eager.boxes_query(bbox).dataset["data"]
        res_lazy = ds_lazy.boxes_query(bbox).dataset["data"]

        assert res_eager.shape == res_lazy.shape
        np.testing.assert_allclose(
            res_eager.values, res_lazy.compute().values, rtol=1e-5, atol=1e-8,
        )

    def test_performance_comparison(self, temp_dataset_dir: Path) -> None:
        """Compare performance between lazy and eager queries."""
        ds_eager = RasterDataset(
            root_dir=temp_dataset_dir, chunks=None, verbose=False,
        )
        ds_lazy = RasterDataset(
            root_dir=temp_dataset_dir, chunks="auto", verbose=False,
        )
        bbox = BoundingBox(0, 30, 0, 30, crs=CRS.from_epsg(4326))

        start = time.time()
        tree_no = ds_eager.boxes_query(bbox)
        time_eager = time.time() - start

        start = time.time()
        tree_yes = ds_lazy.boxes_query(bbox)
        time_lazy = time.time() - start

        assert hasattr(tree_no, "children")
        assert hasattr(tree_yes, "children")
        assert time_eager >= 0
        assert time_lazy >= 0

    def test_mixed_query_types(self, temp_dataset_dir: Path) -> None:
        """Mixed query types with chunks enabled."""
        ds = RasterDataset(root_dir=temp_dataset_dir, chunks="auto", verbose=False)

        points = Points([(10, 10), (20, 20)])
        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))

        result = ds[points]
        assert isinstance(result, dict)
        assert "data" in result
        tree_bbox = ds.boxes_query(bbox)
        assert hasattr(tree_bbox, "children")

    def test_error_handling_without_dask_installed(
        self, temp_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Graceful error when dask is unavailable."""
        import faninsar.datasets.base.raster as raster_mod

        class _Dummy:
            def __init__(self, *_: object, **__: object) -> None:
                msg = "Lazy loading requires dask"
                raise ImportError(msg)

        monkeypatch.setattr(
            raster_mod, "LazyMultiFileReader", _Dummy, raising=False,
        )

        ds = RasterDataset(root_dir=temp_dataset_dir, chunks="auto", verbose=False)
        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))
        with pytest.raises(ImportError, match="Lazy loading requires dask"):
            ds.boxes_query(bbox)

    def test_single_file_query(self, temp_dataset_dir: Path) -> None:
        """Single-file query preserves file dimension."""
        ds = RasterDataset(root_dir=temp_dataset_dir, chunks=None, verbose=False)
        points = Points([(5, 5), (15, 15), (25, 25)])
        ds.points_query(points)

        bbox = BoundingBox(0, 10, 0, 10, crs=CRS.from_epsg(4326))
        result_single = ds.boxes_query(bbox)
        assert result_single.dataset["data"].shape[0] >= 1

    def test_polygons_query(self, temp_dataset_dir: Path) -> None:
        """Polygons query returns expected structure."""
        ds = RasterDataset(root_dir=temp_dataset_dir, chunks=None, verbose=False)

        polygon1 = box(2, 2, 8, 8)
        polygon2 = box(12, 12, 18, 18)
        gdf_multi = gpd.GeoDataFrame(geometry=[polygon1, polygon2], crs=ds.crs)
        multi_polygons = Polygons(gdf_multi, types="desired")

        tree = ds.polygons_query(multi_polygons)
        assert hasattr(tree, "children")
