"""Tests for RasterDataset with dask functionality."""

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
from faninsar.query import BoundingBox, GeoQuery, Points


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
        """Test RasterDataset initialization with use_dask=False."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        assert ds.use_dask is False
        assert len(ds) == 5

    def test_init_with_dask_true(self, temp_dataset_dir):
        """Test RasterDataset initialization with use_dask=True."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=True)
        assert ds.use_dask is True
        assert len(ds) == 5

    def test_dask_availability_check(self, temp_dataset_dir):
        """Test that dask availability is checked when use_dask=True."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=True)
        # This should not raise an error if dask is available
        ds._check_dask_available()

    def test_points_query_without_dask(self, temp_dataset_dir):
        """Test points query without dask."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        points = Points([(5, 5), (15, 15), (25, 25)])

        start_time = time.time()
        result = ds.points_query(points)
        end_time = time.time()

        assert result.data is not None
        assert result.data.shape[1] == 3  # 3 points
        print(f"Points query without dask took: {end_time - start_time:.4f} seconds")

    def test_points_query_with_dask(self, temp_dataset_dir):
        """Test points query with dask."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=True)
        points = Points([(5, 5), (15, 15), (25, 25)])

        start_time = time.time()
        result = ds.points_query(points, use_dask=True)
        end_time = time.time()

        assert result.data is not None
        assert result.data.shape[1] == 3  # 3 points
        print(f"Points query with dask took: {end_time - start_time:.4f} seconds")

    def test_bbox_query_without_dask(self, temp_dataset_dir):
        """Test bbox query without dask."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))

        start_time = time.time()
        result = ds.bbox_query(bbox)
        end_time = time.time()

        assert result.data is not None
        assert result.data.ndim >= 2
        print(f"BBox query without dask took: {end_time - start_time:.4f} seconds")

    def test_bbox_query_with_dask(self, temp_dataset_dir):
        """Test bbox query with dask."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=True)
        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))

        start_time = time.time()
        result = ds.bbox_query(bbox, use_dask=True)
        end_time = time.time()

        assert result.data is not None
        assert result.data.ndim >= 2
        print(f"BBox query with dask took: {end_time - start_time:.4f} seconds")

    def test_getitem_with_dask_default(self, temp_dataset_dir):
        """Test __getitem__ with dataset's default dask setting."""
        # Test with dask disabled by default
        ds_no_dask = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        points = Points([(5, 5), (15, 15)])
        result_no_dask = ds_no_dask[points]
        assert result_no_dask.points.data is not None

        # Test with dask enabled by default
        ds_with_dask = RasterDataset(root_dir=temp_dataset_dir, use_dask=True)
        result_with_dask = ds_with_dask[points]
        assert result_with_dask.points.data is not None

    def test_query_consistency(self, temp_dataset_dir):
        """Test that dask and non-dask queries produce consistent results."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        points = Points([(10, 10), (20, 20)])

        # Query without dask
        result_no_dask = ds.points_query(points, use_dask=False)

        # Query with dask
        result_with_dask = ds.points_query(points, use_dask=True)

        # Results should be similar (allowing for small numerical differences)
        assert result_no_dask.data.shape == result_with_dask.data.shape
        np.testing.assert_allclose(
            result_no_dask.data, result_with_dask.data, rtol=1e-5, atol=1e-8
        )

    def test_performance_comparison(self, temp_dataset_dir):
        """Compare performance between dask and non-dask queries."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        bbox = BoundingBox(0, 30, 0, 30, crs=CRS.from_epsg(4326))

        # Measure time without dask
        start_time = time.time()
        result_no_dask = ds.bbox_query(bbox, use_dask=False)
        time_no_dask = time.time() - start_time

        # Measure time with dask
        start_time = time.time()
        result_with_dask = ds.bbox_query(bbox, use_dask=True)
        time_with_dask = time.time() - start_time

        print(f"Time without dask: {time_no_dask:.4f} seconds")
        print(f"Time with dask: {time_with_dask:.4f} seconds")
        print(f"Speedup ratio: {time_no_dask / time_with_dask:.2f}x")

        # Both should produce valid results
        assert result_no_dask.data is not None
        assert result_with_dask.data is not None

    def test_mixed_query_types(self, temp_dataset_dir):
        """Test mixed query types with dask."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=True)

        points = Points([(10, 10), (20, 20)])
        bbox = BoundingBox(5, 15, 5, 15, crs=CRS.from_epsg(4326))

        # Test GeoQuery with multiple query types
        query = GeoQuery(points=points, boxes=bbox)
        result = ds[query]

        assert result.points is not None
        assert result.boxes is not None
        assert result.points.data is not None
        assert result.boxes.data is not None

    def test_error_handling_without_dask_installed(self, temp_dataset_dir, monkeypatch):
        """Test error handling when dask is not available."""
        # Mock dask as not available
        monkeypatch.setattr("faninsar.datasets.base.HAS_DASK", False)

        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=True)

        with pytest.raises(ImportError, match="dask is required"):
            ds._check_dask_available()

    def test_debug_single_file_query(self, temp_dataset_dir):
        """Debug single file query issue."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        points = Points([(5, 5), (15, 15), (25, 25)])

        print(f"Dataset files: {len(ds)}")
        result_all = ds.points_query(points)
        print(f"All files result shape: {result_all.data.shape}")

        result_single = ds.points_query(points, indexes=0)
        print(f"Single file result shape: {result_single.data.shape}")

        # For single file query, data should be 1D with shape (n_points,) - points query still squeezes
        assert result_single.data.shape == (3,)  # 3 points, single file squeezed

    def test_debug_polygons_query(self, temp_dataset_dir):
        """Debug polygons query issue."""
        import geopandas as gpd
        from shapely.geometry import box

        from faninsar.query import Polygons

        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)

        # Create two polygons that should intersect with the dataset
        polygon1 = box(2, 2, 8, 8)  # Should intersect with first few files
        polygon2 = box(12, 12, 18, 18)  # Should intersect with later files
        gdf_multi = gpd.GeoDataFrame(geometry=[polygon1, polygon2], crs=ds.crs)
        multi_polygons = Polygons(gdf_multi, types="desired")

        print(f"Number of polygons: {len(multi_polygons)}")
        print(f"Dataset files: {len(ds)}")

        try:
            result = ds.polygons_query(multi_polygons)
            print("Polygons query succeeded!")
            print(f"Result data type: {type(result.data)}")
            if isinstance(result.data, list):
                print(f"Number of polygon results: {len(result.data)}")
                for i, poly_data in enumerate(result.data):
                    print(f"Polygon {i} shape: {poly_data.shape}")
        except Exception as e:
            print(f"Polygons query failed: {e}")
            import traceback
            traceback.print_exc()
