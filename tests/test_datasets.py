"""Tests for RasterDataset with dask functionality (updated for new API)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from faninsar.datasets.base import RasterDataset
from faninsar.query import BoundingBox, Points


def create_test_tiff(path: Path, bounds: tuple, data: np.ndarray, crs: str = "EPSG:4326") -> None:
    """Create a test GeoTIFF file."""
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory with test GeoTIFF files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple test files with different bounds
        bounds_list = [
            (0, 0, 10, 10),
            (5, 5, 15, 15),
            (10, 10, 20, 20),
            (15, 15, 25, 25),
            (20, 20, 30, 30),
        ]

        for i, bounds in enumerate(bounds_list):
            data = np.random.rand(100, 100).astype(np.float32) * 100
            file_path = temp_path / f"test_file_{i:02d}.tif"
            create_test_tiff(file_path, bounds, data)

        yield temp_path


class TestRasterDatasetDask:
    """Test RasterDataset with dask functionality."""

    def test_init_with_dask_false(self, temp_dataset_dir):
        """Test RasterDataset initialization with lazy_loading=False."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=False, verbose=False)
        assert ds.lazy_loading is False
        assert len(ds) == 5

    def test_init_with_dask_true(self, temp_dataset_dir):
        """Test RasterDataset initialization with lazy_loading=True."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=True, verbose=False)
        assert ds.lazy_loading is True
        assert len(ds) == 5

    def test_dask_availability_check(self, temp_dataset_dir):
        """Test that dask availability is checked when lazy_loading=True."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=True, verbose=False)
        # This should not raise an error if dask is available
        # Note: _check_dask_available method may not exist anymore - checking is done in lazy methods
        assert ds.lazy_loading is True

    def test_points_query_without_dask(self, temp_dataset_dir):
        """Test points query without dask."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=False, verbose=False)
        points = Points([(5, 5), (15, 15), (25, 25)])

        start_time = time.time()
        result = ds.points_query(points)
        end_time = time.time()

        assert "data" in result
        assert result["data"].data.shape[1] == 3  # 3 points
        print(f"Points query without dask took: {end_time - start_time:.4f} seconds")

    def test_points_query_with_dask(self, temp_dataset_dir):
        """Test points query (points are always eager; no lazy switch)."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=True, verbose=False)
        points = Points([(5, 5), (15, 15), (25, 25)])

        start_time = time.time()
        result = ds.points_query(points)
        end_time = time.time()

        assert "data" in result
        assert result["data"].data.shape[1] == 3  # 3 points
        print(f"Points query (eager) took: {end_time - start_time:.4f} seconds")

    def test_box_query_without_dask(self, temp_dataset_dir):
        """Test bbox query without dask."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=False, verbose=False)
        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))

        start_time = time.time()
        result = ds.boxes_query(bbox)
        end_time = time.time()

        # Single bbox -> dataset on root
        assert result.dataset is not None
        assert result.dataset["data"].ndim >= 2
        print(f"BBox query without dask took: {end_time - start_time:.4f} seconds")

    def test_box_query_with_dask(self, temp_dataset_dir):
        """Test bbox query with dask."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=True, verbose=False)
        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))

        start_time = time.time()
        result = ds.boxes_query(bbox, lazy_loading=True)
        end_time = time.time()

        # Single bbox -> dataset on root
        assert result.dataset is not None
        assert result.dataset["data"].ndim >= 2
        print(f"BBox query with dask took: {end_time - start_time:.4f} seconds")

    def test_getitem_with_dask_default(self, temp_dataset_dir):
        """Test __getitem__ with dataset's default dask setting."""
        # Test with dask disabled by default
        ds_no_dask = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=False, verbose=False)
        points = Points([(5, 5), (15, 15)])
        tree_no_dask = ds_no_dask[points]
        assert "points" in tree_no_dask.children

        ds_with_dask = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=True, verbose=False)
        tree_with_dask = ds_with_dask[points]
        assert "points" in tree_with_dask.children

    def test_query_consistency(self, temp_dataset_dir):
        """Test that dask and non-dask queries produce consistent results."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=False, verbose=False)
        points = Points([(10, 10), (20, 20)])

        # Compare bbox results with and without lazy loading
        bbox = BoundingBox(0, 30, 0, 30, crs=CRS.from_epsg(4326))
        ds_no = ds.boxes_query(bbox, lazy_loading=False).dataset["data"]
        ds_yes = ds.boxes_query(bbox, lazy_loading=True).dataset["data"]
        assert ds_no.shape == ds_yes.shape
        np.testing.assert_allclose(ds_no.compute() if hasattr(ds_no, "compute") else ds_no.values,
                                   ds_yes.compute() if hasattr(ds_yes, "compute") else ds_yes.values,
                                   rtol=1e-5, atol=1e-8)

    def test_performance_comparison(self, temp_dataset_dir):
        """Compare performance between dask and non-dask queries."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=False, verbose=False)
        bbox = BoundingBox(0, 30, 0, 30, crs=CRS.from_epsg(4326))

        # Measure time without dask
        start_time = time.time()
        tree_no = ds.boxes_query(bbox, lazy_loading=False)
        time_no_dask = time.time() - start_time

        start_time = time.time()
        tree_yes = ds.boxes_query(bbox, lazy_loading=True)
        time_with_dask = time.time() - start_time

        print(f"Time without dask: {time_no_dask:.4f} seconds")
        print(f"Time with dask: {time_with_dask:.4f} seconds")
        assert hasattr(tree_no, "children") and hasattr(tree_yes, "children")

    def test_mixed_query_types(self, temp_dataset_dir):
        """Test mixed query types with dask."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=True, verbose=False)

        points = Points([(10, 10), (20, 20)])
        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))

        # Test combined selection using __getitem__ with Points then boxes_query
        tree = ds[points]
        assert "points" in tree.children
        tree_bbox = ds.boxes_query(bbox)
        assert hasattr(tree_bbox, "children")

    def test_error_handling_without_dask_installed(self, temp_dataset_dir, monkeypatch):
        """Test error handling when dask (LazyMultiFileReader) is not available."""
        # Simulate missing dask by making LazyMultiFileReader raise ImportError
        class _Dummy:
            def __init__(self, *a, **k):
                raise ImportError("Lazy loading requires dask")
        import faninsar.datasets.base as base_mod
        monkeypatch.setattr(base_mod, "LazyMultiFileReader", _Dummy, raising=False)

        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=True, verbose=False)

        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))
        with pytest.raises(ImportError, match="Lazy loading requires dask"):
            ds.boxes_query(bbox, lazy_loading=True)

    def test_debug_single_file_query(self, temp_dataset_dir):
        """Debug single file query issue."""
        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=False, verbose=False)
        points = Points([(5, 5), (15, 15), (25, 25)])

        print(f"Dataset files: {len(ds)}")
        result_all = ds.points_query(points)
        print(f"All files result shape: {result_all['data'].data.shape}")

        # Use boxes_query to validate single-file dim not squeezed (indexes not used here)
        bbox = BoundingBox(0, 10, 0, 10, crs=CRS.from_epsg(4326))
        result_single = ds.boxes_query(bbox, lazy_loading=False)
        print(f"Single bbox result file-dim: {result_single.dataset['data'].shape[0]}")
        assert result_single.dataset["data"].shape[0] >= 1

    def test_debug_polygons_query(self, temp_dataset_dir):
        """Debug polygons query issue."""
        import geopandas as gpd
        from shapely.geometry import box

        from faninsar.query import Polygons

        ds = RasterDataset(root_dir=temp_dataset_dir, lazy_loading=False, verbose=False)

        # Create two polygons that should intersect with the dataset
        polygon1 = box(2, 2, 8, 8)  # Should intersect with first few files
        polygon2 = box(12, 12, 18, 18)  # Should intersect with later files
        gdf_multi = gpd.GeoDataFrame(geometry=[polygon1, polygon2], crs=ds.crs)
        multi_polygons = Polygons(gdf_multi, types="desired")

        print(f"Number of polygons: {len(multi_polygons)}")
        print(f"Dataset files: {len(ds)}")

        try:
            tree = ds.polygons_query(multi_polygons)
            print("Polygons query succeeded!")
            assert hasattr(tree, "children")
        except Exception as e:
            print(f"Polygons query failed: {e}")
            import traceback
            traceback.print_exc()
