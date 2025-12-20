"""Tests for :mod:`faninsar.datasets.xarray_dataset`."""

from __future__ import annotations

import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from faninsar.datasets.xarray_dataset import XarrayDataset
from faninsar.query.bbox import BoundingBox


def _write_tiff(path: Path, bounds: tuple[float, float, float, float], data: np.ndarray) -> None:
    """Helper to write a single-band GeoTIFF."""
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def temp_raster_dir() -> Path:
    """Create a temporary directory with two small GeoTIFF tiles."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tile_a = np.arange(25, dtype=np.float32).reshape(5, 5)
        tile_b = np.arange(25, dtype=np.float32).reshape(5, 5) + 100

        _write_tiff(tmp_path / "tile_a.tif", (0, 0, 5, 5), tile_a)
        _write_tiff(tmp_path / "tile_b.tif", (5, 0, 10, 5), tile_b)
        yield tmp_path


def test_dataset_metadata(temp_raster_dir: Path) -> None:
    """XarrayDataset should expose combined metadata."""
    dataset = XarrayDataset(paths=[temp_raster_dir / "tile_a.tif", temp_raster_dir / "tile_b.tif"])
    assert dataset.file_count == 2
    assert dataset.crs == CRS.from_epsg(4326)
    assert dataset.res == (1.0, 1.0)
    assert dataset.bounds.left == 0
    assert dataset.bounds.right == 10


def test_box_query_full_extent(temp_raster_dir: Path) -> None:
    """A bbox covering both tiles should return stacked data."""
    dataset = XarrayDataset(paths=[temp_raster_dir / "tile_a.tif", temp_raster_dir / "tile_b.tif"])
    bbox = BoundingBox(left=0, bottom=0, right=10, top=5, crs=CRS.from_epsg(4326))
    result = dataset.box_query(bbox)
    assert result.shape == (2, 5, 10)
    # First tile occupies first 5 columns
    np.testing.assert_array_equal(result.sel(file=0).values[:, :5], np.arange(25, dtype=np.float32).reshape(5, 5))
    # Second tile occupies last 5 columns
    expected_b = (np.arange(25, dtype=np.float32).reshape(5, 5) + 100)
    np.testing.assert_array_equal(result.sel(file=1).values[:, 5:], expected_b)


def test_box_query_partial_overlap(temp_raster_dir: Path) -> None:
    """Partial overlap should fill NoData outside the source extent."""
    dataset = XarrayDataset(paths=[temp_raster_dir / "tile_a.tif"])
    bbox = BoundingBox(left=2, bottom=0, right=4, top=2, crs=CRS.from_epsg(4326))
    result = dataset.box_query(bbox)
    assert result.shape == (1, 2, 2)
    expected = np.array([[17, 18], [22, 23]], dtype=np.float32)
    np.testing.assert_array_equal(result.values[0], expected)


def test_box_query_no_overlap(temp_raster_dir: Path) -> None:
    """When there is no spatial overlap, the array should be filled with nodata."""
    dataset = XarrayDataset(paths=[temp_raster_dir / "tile_a.tif"])
    bbox = BoundingBox(left=20, bottom=20, right=25, top=25, crs=CRS.from_epsg(4326))
    result = dataset.box_query(bbox)
    assert np.all(result.values == dataset.nodata)


def test_box_query_lazy_matches_eager(temp_raster_dir: Path) -> None:
    """Lazy queries should build equivalent results once computed."""
    bbox = BoundingBox(left=0, bottom=0, right=10, top=5, crs=CRS.from_epsg(4326))
    eager_dataset = XarrayDataset(paths=[temp_raster_dir / "tile_a.tif", temp_raster_dir / "tile_b.tif"])
    lazy_dataset = XarrayDataset(
        paths=[temp_raster_dir / "tile_a.tif", temp_raster_dir / "tile_b.tif"],
        lazy_loading=True,
        chunks=(3, 3),
    )
    eager = eager_dataset.box_query(bbox)
    lazy = lazy_dataset.box_query(bbox)
    assert isinstance(lazy.data, da.Array)
    np.testing.assert_array_equal(lazy.data.compute(), eager.data)


def test_lazy_no_overlap_returns_nodata(temp_raster_dir: Path) -> None:
    """Lazy queries with no overlap should avoid creating source tasks."""
    dataset = XarrayDataset(
        paths=[temp_raster_dir / "tile_a.tif"],
        lazy_loading=True,
        chunks=(2, 2),
    )
    bbox = BoundingBox(left=50, bottom=50, right=60, top=60, crs=CRS.from_epsg(4326))
    result = dataset.box_query(bbox)
    assert isinstance(result.data, da.Array)
    computed = result.data.compute()
    expected_height = int(round((bbox.top - bbox.bottom) / dataset.res[1]))
    expected_width = int(round((bbox.right - bbox.left) / dataset.res[0]))
    assert computed.shape == (1, expected_height, expected_width)
    assert np.all(computed == dataset.nodata)
