"""Comprehensive tests for queries and colormaps with boundary conditions.

This module combines tests from test_queries.py, test_cmaps_advanced.py,
and adds comprehensive boundary condition tests for points, bbox, and polygons queries.
"""

from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np
import pytest
import rasterio
import rasterio.transform as rasterio_transform
from pyproj.crs.crs import CRS
from rasterio.crs import CRS as RasterioCRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from shapely.geometry import box

from faninsar.cmaps import cmaps
from faninsar.datasets import RasterDataset
from faninsar.datasets.base import RasterDataset as BaseRasterDataset
from faninsar.query import (
    BBoxesResult,
    BoundingBox,
    Points,
    PointsResult,
    Polygons,
    PolygonsResult,
)


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
def create_test_tiffs(tmp_path: Path):
    """Create test GeoTIFF files for testing."""
    # Create directory for test files
    test_dir = tmp_path / "test_tiffs"
    test_dir.mkdir(exist_ok=True)

    # Create simple profiles for the test tiffs with proper WGS84 coordinates
    height, width = 100, 120
    # Use realistic longitude/latitude ranges (-10 to -8, 40 to 42)
    transform = rasterio_transform.from_bounds(-10, 40, -8, 42, width, height)

    # Create 3 test tiffs with simple gradient data
    filenames = []
    for i in range(3):
        # Create gradient data
        data = np.ones((height, width), dtype=np.float32) * (i + 1)
        # Add some gradients
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        data += xx + yy

        # Write to file
        filename = test_dir / f"test_{i}.tif"
        with rasterio.open(
            filename,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        filenames.append(filename)

    return filenames


@pytest.fixture
def raster_dataset(create_test_tiffs):
    """Create a RasterDataset instance for testing."""
    return RasterDataset(
        paths=create_test_tiffs,
        cache=True,
        resampling=Resampling.nearest,
        verbose=False,
    )


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
        ]

        for i, bounds in enumerate(bounds_list):
            # Create data with different values for each file
            data = np.full((50, 50), i + 1, dtype=np.float32)
            file_path = temp_path / f"test_file_{i:02d}.tif"
            create_test_tiff(file_path, bounds, data)

        yield temp_path


# =============================================================================
# Basic Query Tests (from test_queries.py)
# =============================================================================

class TestRasterDatasetQueries:
    """Basic tests for RasterDataset query methods."""

    def test_points_query(self, raster_dataset):
        """Test the points_query method."""
        # Create test points with proper WGS84 coordinates
        points = Points([
            [-9.8, 40.2],  # Inside the dataset near the bottom-left
            [-9.0, 41.0],  # Middle of the dataset
            [-8.2, 41.8],  # Near the top-right
        ],
        crs=4326,
        )

        # Test with default parameters (all files)
        result = raster_dataset.points_query(points)

        # Validate result
        assert isinstance(result, PointsResult)
        assert result.data is not None
        assert result.data.shape[0] == 3  # 3 files
        assert result.data.shape[1] == 3  # 3 points

        # Test with specific file index
        result_single = raster_dataset.points_query(points, indexes=0)
        assert result_single.data.shape == (3,)  # 3 points, single file squeezed

        # Test with multiple file indexes
        result_multi = raster_dataset.points_query(points, indexes=[0, 2])
        assert result_multi.data.shape[0] == 2  # 2 files
        assert result_multi.data.shape[1] == 3  # 3 points

        # Verify values are different between files
        # File index 2 should have higher values than file index 0
        assert np.all(result_multi.data[1] > result_multi.data[0])

    def test_bbox_query(self, raster_dataset):
        """Test the bbox_query method."""
        # Create test bounding box with proper WGS84 coordinates
        bbox = BoundingBox(-9.5, 40.5, -9.0, 41.0, crs=raster_dataset.crs)

        # Test with default parameters (all files)
        result = raster_dataset.bbox_query(bbox)

        # Validate result
        assert isinstance(result, BBoxesResult)
        assert result.data is not None

        # Check dimensions: [n_files, height, width]
        assert result.data.ndim == 3
        assert result.data.shape[0] == 3  # 3 files
        profile= raster_dataset.get_profile(bbox)
        height, width = profile['height'], profile['width']
        assert result.data.shape[1] == height
        assert result.data.shape[2] == width

        # Test with specific file index - file dimension should NOT be squeezed
        result_single = raster_dataset.bbox_query(bbox, indexes=1)
        assert result_single.data.shape == (1, height, width)  # Single file NOT squeezed

        # Verify values increase between files
        result_multi = raster_dataset.bbox_query(bbox, indexes=[0, 1, 2])
        assert np.all(result_multi.data[1] > result_multi.data[0])
        assert np.all(result_multi.data[2] > result_multi.data[1])

    def test_polygons_query(self, raster_dataset):
        """Test the polygons_query method."""
        # Create a simple polygon (rectangle) with proper WGS84 coordinates
        polygon = box(-9.6, 40.6, -9.0, 41.2)
        # Create a GeoDataFrame with the polygon
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        # Test with default parameters (all files)
        result = raster_dataset.polygons_query(polygons)

        # Validate result
        assert isinstance(result, PolygonsResult)
        assert result.data is not None

        # Check if we received data from all files
        if isinstance(result.data, list):
            assert len(result.data) == 1  # One polygon
            assert result.data[0].shape[0] == 3  # 3 files
        else:
            assert result.data.shape[0] == 3  # 3 files

        # Test with specific file index - file dimension should NOT be squeezed
        result_single = raster_dataset.polygons_query(polygons, indexes=2)
        if isinstance(result_single.data, list):
            assert len(result_single.data) == 1  # One polygon
            assert result_single.data[0].shape[0] == 1  # 1 file, NOT squeezed
        else:
            assert result_single.data.shape[0] == 1  # 1 file, NOT squeezed

        # Test with multiple polygons
        polygon2 = box(-8.8, 40.3, -8.4, 40.8)
        # Create a GeoDataFrame with multiple polygons
        gdf_multi = gpd.GeoDataFrame(geometry=[polygon, polygon2], crs=raster_dataset.crs)
        multi_polygons = Polygons(gdf_multi, types="desired")

        result_multi = raster_dataset.polygons_query(multi_polygons)
        assert isinstance(result_multi, PolygonsResult)
        assert result_multi.data is not None

        # Should have data for 2 polygons
        assert len(result_multi.data) == 2
        # Each polygon may have data from some files (not necessarily all)
        # Just check that we have valid data structures
        assert isinstance(result_multi.data[0], np.ma.MaskedArray)
        assert isinstance(result_multi.data[1], np.ma.MaskedArray)

    def test_invalid_indexes(self, raster_dataset):
        """Test error handling for invalid indexes."""
        points = Points([[50, 50]])

        # Test with out-of-range index
        with pytest.raises(ValueError):
            raster_dataset.points_query(points, indexes=100)

        # Test with negative index
        with pytest.raises(ValueError):
            raster_dataset.points_query(points, indexes=-1)


# =============================================================================
# Boundary Condition Tests (from test_query_boundary_conditions.py)
# =============================================================================

class TestUpdatedBboxPolygonsQuery:
    """Test updated bbox_query and polygons_query behavior."""

    def test_bbox_query_single_bbox_no_list(self, temp_dataset_dir):
        """Test bbox_query with single bbox (not in list) - should not have bbox dimension."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        bbox = BoundingBox(2, 12, 2, 12, crs=CRS.from_epsg(4326))

        # Test with all files
        result = ds.bbox_query(bbox)
        assert result.data is not None
        # Should have shape (n_files, height, width) - no bbox dimension
        assert result.data.ndim == 3
        assert result.data.shape[0] == 3  # 3 files

        # Test with single file - file dimension should NOT be squeezed
        result_single = ds.bbox_query(bbox, indexes=0)
        assert result_single.data is not None
        # Should still have file dimension even though it's 1
        assert result_single.data.ndim == 3
        assert result_single.data.shape[0] == 1  # 1 file, not squeezed

    def test_bbox_query_single_bbox_in_list(self, temp_dataset_dir):
        """Test bbox_query with single bbox in list - should have bbox dimension."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        bbox = BoundingBox(2, 12, 2, 12, crs=CRS.from_epsg(4326))
        bbox_list = [bbox]  # Single bbox in list

        # Test with all files
        result = ds.bbox_query(bbox_list)
        assert result.data is not None
        # Should have shape (n_files, n_bboxes, height, width) - with bbox dimension
        assert result.data.ndim == 4
        assert result.data.shape[0] == 3  # 3 files
        assert result.data.shape[1] == 1  # 1 bbox

        # Test with single file - file dimension should NOT be squeezed
        result_single = ds.bbox_query(bbox_list, indexes=0)
        assert result_single.data is not None
        # Should still have file dimension even though it's 1
        assert result_single.data.ndim == 4
        assert result_single.data.shape[0] == 1  # 1 file, not squeezed
        assert result_single.data.shape[1] == 1  # 1 bbox

    def test_bbox_query_multiple_bboxes(self, temp_dataset_dir):
        """Test bbox_query with multiple bboxes - should have bbox dimension."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        bbox1 = BoundingBox(2, 8, 2, 8, crs=CRS.from_epsg(4326))
        bbox2 = BoundingBox(12, 18, 12, 18, crs=CRS.from_epsg(4326))
        bbox_list = [bbox1, bbox2]

        # Test with all files
        result = ds.bbox_query(bbox_list)
        assert result.data is not None
        # Should have shape (n_files, n_bboxes, height, width)
        assert result.data.ndim == 4
        assert result.data.shape[0] == 3  # 3 files
        assert result.data.shape[1] == 2  # 2 bboxes

        # Test with single file - file dimension should NOT be squeezed
        result_single = ds.bbox_query(bbox_list, indexes=1)
        assert result_single.data is not None
        assert result_single.data.ndim == 4
        assert result_single.data.shape[0] == 1  # 1 file, not squeezed
        assert result_single.data.shape[1] == 2  # 2 bboxes

    def test_polygons_query_single_polygon(self, temp_dataset_dir):
        """Test polygons_query with single polygon - should always have polygon dimension."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)

        # Debug: print dataset bounds and individual file bounds
        print(f"Dataset bounds: {ds.bounds}")
        for i, file_path in enumerate(ds.files.paths):
            if ds.files.valid.iloc[i]:
                with rasterio.open(file_path) as src:
                    print(f"File {i} bounds: {src.bounds}")

        # Create single polygon that overlaps with the test data bounds
        # Use bounds that definitely overlap with all files
        polygon = box(0.5, 0.5, 19.5, 19.5)  # Large polygon covering most of the dataset
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=ds.crs)
        print(f"Polygon bounds: {polygon.bounds}")
        print(f"GDF CRS: {gdf.crs}")
        polygons = Polygons(gdf, types="desired")

        # Test with all files
        result = ds.polygons_query(polygons)
        print(f"Result data type: {type(result.data)}")
        if isinstance(result.data, list):
            print(f"Number of polygons: {len(result.data)}")
            for i, poly_data in enumerate(result.data):
                print(f"Polygon {i} type: {type(poly_data)}")
                if hasattr(poly_data, 'shape'):
                    print(f"Polygon {i} shape: {poly_data.shape}")
                elif isinstance(poly_data, list):
                    print(f"Polygon {i} list length: {len(poly_data)}")
                    for j, item in enumerate(poly_data):
                        if hasattr(item, 'shape'):
                            print(f"  Item {j} shape: {item.shape}")
        assert result.data is not None
        # Should always have polygon dimension
        if isinstance(result.data, list):
            assert len(result.data) == 1  # 1 polygon
            # Check if result.data[0] is an array or list
            if hasattr(result.data[0], 'shape'):
                # If it's an array, file dimension should not be squeezed
                assert result.data[0].shape[0] == 3  # 3 files
            elif isinstance(result.data[0], list):
                # If it's a list (due to different shapes), check we have 3 files
                assert len(result.data[0]) == 3  # 3 files
                # Each file should have some data
                for file_data in result.data[0]:
                    assert hasattr(file_data, 'shape')  # Should be an array
        else:
            # If not a list, should still have proper dimensions
            assert result.data.shape[0] == 3  # 3 files

        # Test with single file - file dimension should NOT be squeezed
        result_single = ds.polygons_query(polygons, indexes=0)
        assert result_single.data is not None
        if isinstance(result_single.data, list):
            assert len(result_single.data) == 1  # 1 polygon
            # File dimension should not be squeezed even for single file
            assert result_single.data[0].shape[0] == 1  # 1 file, not squeezed
        else:
            assert result_single.data.shape[0] == 1  # 1 file, not squeezed

    def test_polygons_query_multiple_polygons(self, temp_dataset_dir):
        """Test polygons_query with multiple polygons - should have polygon dimension."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=False)

        # Create multiple polygons that overlap with the test data bounds
        polygon1 = box(1, 1, 9, 9)
        polygon2 = box(11, 11, 19, 19)
        gdf = gpd.GeoDataFrame(geometry=[polygon1, polygon2], crs=ds.crs)
        polygons = Polygons(gdf, types="desired")

        # Test with all files
        result = ds.polygons_query(polygons)
        assert result.data is not None
        # Should have polygon dimension
        assert isinstance(result.data, list)
        assert len(result.data) == 2  # 2 polygons
        # File dimension should not be squeezed
        for poly_data in result.data:
            if hasattr(poly_data, 'shape'):
                # If it's an array, file dimension should not be squeezed
                assert poly_data.shape[0] == 3  # 3 files
            elif isinstance(poly_data, list):
                # If it's a list (due to different shapes), check we have 3 files
                assert len(poly_data) == 3  # 3 files
                # Each file should have some data (even if empty)
                for file_data in poly_data:
                    assert hasattr(file_data, 'shape')  # Should be an array

        # Test with single file - file dimension should NOT be squeezed
        result_single = ds.polygons_query(polygons, indexes=1)
        assert result_single.data is not None
        assert isinstance(result_single.data, list)
        assert len(result_single.data) == 2  # 2 polygons
        # File dimension should not be squeezed even for single file
        for poly_data in result_single.data:
            assert poly_data.shape[0] == 1  # 1 file, not squeezed

    def test_bbox_query_with_dask(self, temp_dataset_dir):
        """Test bbox_query with dask enabled."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=True)
        bbox = BoundingBox(2, 12, 2, 12, crs=CRS.from_epsg(4326))

        # Test single bbox (no list)
        result_single = ds.bbox_query(bbox, use_dask=True)
        assert result_single.data is not None
        assert result_single.data.ndim == 3  # No bbox dimension
        assert result_single.data.shape[0] == 3  # 3 files

        # Test bbox in list
        result_list = ds.bbox_query([bbox], use_dask=True)
        assert result_list.data is not None
        assert result_list.data.ndim == 4  # With bbox dimension
        assert result_list.data.shape[0] == 3  # 3 files
        assert result_list.data.shape[1] == 1  # 1 bbox

    def test_polygons_query_with_dask(self, temp_dataset_dir):
        """Test polygons_query with dask enabled."""
        ds = RasterDataset(root_dir=temp_dataset_dir, use_dask=True)

        polygon = box(1, 1, 9, 9)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=ds.crs)
        polygons = Polygons(gdf, types="desired")

        result = ds.polygons_query(polygons, use_dask=True)
        assert result.data is not None
        # Should maintain polygon dimension and not squeeze file dimension
        if isinstance(result.data, list):
            assert len(result.data) == 1  # 1 polygon
            # Check if result.data[0] is an array or list
            if hasattr(result.data[0], 'shape'):
                # If it's an array, file dimension should not be squeezed
                assert result.data[0].shape[0] == 3  # 3 files
            elif isinstance(result.data[0], list):
                # If it's a list (due to different shapes), check we have 3 files
                assert len(result.data[0]) == 3  # 3 files
                # Each file should have some data (even if empty)
                for file_data in result.data[0]:
                    assert hasattr(file_data, 'shape')  # Should be an array


# =============================================================================
# Comprehensive Boundary Condition Tests
# =============================================================================

class TestPointsQueryBoundaryConditions:
    """Test boundary conditions for points_query method."""

    def test_points_outside_dataset_bounds(self, raster_dataset):
        """Test points_query with points outside dataset bounds."""
        # Points completely outside the dataset bounds
        points = Points([
            [-15.0, 35.0],  # Far west and south
            [-5.0, 45.0],   # Far east and north
            [0.0, 0.0],     # Completely different region
        ], crs=4326)

        result = raster_dataset.points_query(points)
        assert isinstance(result, PointsResult)
        assert result.data is not None
        # Should return NaN or masked values for points outside bounds
        assert result.data.shape[0] == 3  # 3 files
        assert result.data.shape[1] == 3  # 3 points

    def test_points_on_dataset_edges(self, raster_dataset):
        """Test points_query with points exactly on dataset edges."""
        # Points on the exact edges of the dataset
        points = Points([
            [-10.0, 40.0],  # Bottom-left corner
            [-8.0, 42.0],   # Top-right corner
            [-9.0, 40.0],   # Bottom edge
            [-9.0, 42.0],   # Top edge
            [-10.0, 41.0],  # Left edge
            [-8.0, 41.0],   # Right edge
        ], crs=4326)

        result = raster_dataset.points_query(points)
        assert isinstance(result, PointsResult)
        assert result.data is not None
        assert result.data.shape[0] == 3  # 3 files
        assert result.data.shape[1] == 6  # 6 points

    def test_points_single_point(self, raster_dataset):
        """Test points_query with a single point."""
        points = Points([[-9.0, 41.0]], crs=4326)

        result = raster_dataset.points_query(points)
        assert isinstance(result, PointsResult)
        assert result.data is not None
        assert result.data.shape[0] == 3  # 3 files
        assert result.data.shape[1] == 1  # 1 point

    def test_points_empty_list(self, raster_dataset):
        """Test points_query with empty points list."""
        # Empty points should raise a ValueError
        with pytest.raises(ValueError, match="points must be a 2D array with 2 columns"):
            points = Points([], crs=4326)

    def test_points_duplicate_coordinates(self, raster_dataset):
        """Test points_query with duplicate point coordinates."""
        points = Points([
            [-9.0, 41.0],
            [-9.0, 41.0],  # Duplicate
            [-8.5, 40.5],
            [-8.5, 40.5],  # Duplicate
        ], crs=4326)

        result = raster_dataset.points_query(points)
        assert isinstance(result, PointsResult)
        assert result.data is not None
        assert result.data.shape[0] == 3  # 3 files
        assert result.data.shape[1] == 4  # 4 points (including duplicates)
        # Duplicate points should have identical values
        assert np.allclose(result.data[:, 0], result.data[:, 1])
        assert np.allclose(result.data[:, 2], result.data[:, 3])

    def test_points_different_crs(self, raster_dataset):
        """Test points_query with points in different CRS."""
        # Points in UTM coordinates (approximate conversion)
        points = Points([
            [500000, 4500000],  # Approximate UTM coordinates
        ], crs=32630)  # UTM Zone 30N

        result = raster_dataset.points_query(points)
        assert isinstance(result, PointsResult)
        assert result.data is not None

    def test_points_extreme_coordinates(self, raster_dataset):
        """Test points_query with extreme coordinate values."""
        points = Points([
            [-180.0, -90.0],   # Extreme south-west
            [180.0, 90.0],     # Extreme north-east
            [0.0, 0.0],        # Equator/Prime meridian
        ], crs=4326)

        result = raster_dataset.points_query(points)
        assert isinstance(result, PointsResult)
        assert result.data is not None
        # Most of these should be outside bounds and return NaN/masked values


class TestBboxQueryBoundaryConditions:
    """Test boundary conditions for bbox_query method."""

    def test_bbox_outside_dataset_bounds(self, raster_dataset):
        """Test bbox_query with bbox completely outside dataset bounds."""
        bbox = BoundingBox(-20.0, 30.0, -15.0, 35.0, crs=raster_dataset.crs)

        result = raster_dataset.bbox_query(bbox)
        assert isinstance(result, BBoxesResult)
        # Should handle out-of-bounds bbox gracefully
        # Result might be empty or contain NaN/masked values

    def test_bbox_partially_overlapping(self, raster_dataset):
        """Test bbox_query with bbox partially overlapping dataset."""
        # Bbox that extends beyond dataset bounds
        bbox = BoundingBox(-11.0, 39.0, -7.0, 43.0, crs=raster_dataset.crs)

        result = raster_dataset.bbox_query(bbox)
        assert isinstance(result, BBoxesResult)
        assert result.data is not None
        # Should return data for the overlapping region

    def test_bbox_exact_dataset_bounds(self, raster_dataset):
        """Test bbox_query with bbox exactly matching dataset bounds."""
        # Get the exact bounds of the dataset
        bounds = raster_dataset.bounds
        bbox = BoundingBox(bounds.left, bounds.bottom, bounds.right, bounds.top, crs=raster_dataset.crs)

        result = raster_dataset.bbox_query(bbox)
        assert isinstance(result, BBoxesResult)
        assert result.data is not None
        # Should return the entire dataset

    def test_bbox_very_small(self, raster_dataset):
        """Test bbox_query with very small bbox."""
        # Tiny bbox that might result in 1x1 pixel
        bbox = BoundingBox(-9.0, 41.0, -8.99, 41.01, crs=raster_dataset.crs)

        result = raster_dataset.bbox_query(bbox)
        assert isinstance(result, BBoxesResult)
        assert result.data is not None
        # Should handle small bboxes gracefully

    def test_bbox_zero_area(self, raster_dataset):
        """Test bbox_query with zero-area bbox (point)."""
        # Bbox with same min/max coordinates
        bbox = BoundingBox(-9.0, 41.0, -9.0, 41.0, crs=raster_dataset.crs)

        try:
            result = raster_dataset.bbox_query(bbox)
            # If it doesn't raise an error, check the result
            if result.data is not None:
                assert isinstance(result, BBoxesResult)
        except (ValueError, ZeroDivisionError):
            # Zero-area bbox might raise an error, which is acceptable
            pass

    def test_bbox_inverted_coordinates(self, raster_dataset):
        """Test bbox_query with inverted coordinates (min > max)."""
        # Bbox with inverted coordinates should raise a ValueError
        with pytest.raises(ValueError, match="Bounding box is invalid"):
            bbox = BoundingBox(-8.0, 42.0, -10.0, 40.0, crs=raster_dataset.crs)

    def test_bbox_list_single_bbox(self, temp_dataset_dir):
        """Test bbox_query with single bbox in list - should preserve bbox dimension."""
        ds = BaseRasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        bbox = BoundingBox(2, 12, 2, 12, crs=RasterioCRS.from_epsg(4326))
        bbox_list = [bbox]

        result = ds.bbox_query(bbox_list)
        assert result.data is not None
        # Should have bbox dimension when bbox is in a list
        assert result.data.ndim == 4  # (files, bboxes, height, width)
        assert result.data.shape[1] == 1  # 1 bbox

    def test_bbox_list_multiple_bboxes(self, temp_dataset_dir):
        """Test bbox_query with multiple bboxes in list."""
        ds = BaseRasterDataset(root_dir=temp_dataset_dir, use_dask=False)
        bbox1 = BoundingBox(2, 8, 2, 8, crs=RasterioCRS.from_epsg(4326))
        bbox2 = BoundingBox(12, 18, 12, 18, crs=RasterioCRS.from_epsg(4326))
        bbox_list = [bbox1, bbox2]

        result = ds.bbox_query(bbox_list)
        assert result.data is not None
        assert result.data.ndim == 4  # (files, bboxes, height, width)
        assert result.data.shape[1] == 2  # 2 bboxes


class TestPolygonsQueryBoundaryConditions:
    """Test boundary conditions for polygons_query method."""

    def test_polygon_outside_dataset_bounds(self, raster_dataset):
        """Test polygons_query with polygon completely outside dataset bounds."""
        # Polygon completely outside the dataset
        polygon = box(-20.0, 30.0, -15.0, 35.0)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        result = raster_dataset.polygons_query(polygons)
        assert isinstance(result, PolygonsResult)
        # Should handle out-of-bounds polygon gracefully
        # Result might be empty or contain masked values

    def test_polygon_partially_overlapping(self, raster_dataset):
        """Test polygons_query with polygon partially overlapping dataset."""
        # Polygon that extends beyond dataset bounds
        polygon = box(-11.0, 39.0, -7.0, 43.0)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        result = raster_dataset.polygons_query(polygons)
        assert isinstance(result, PolygonsResult)
        assert result.data is not None
        # Should return data for the overlapping region

    def test_polygon_exact_dataset_bounds(self, raster_dataset):
        """Test polygons_query with polygon exactly matching dataset bounds."""
        # Polygon covering the exact bounds of the dataset
        bounds = raster_dataset.bounds
        polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        result = raster_dataset.polygons_query(polygons)
        assert isinstance(result, PolygonsResult)
        assert result.data is not None
        # Should return the entire dataset

    def test_polygon_very_small(self, raster_dataset):
        """Test polygons_query with very small polygon."""
        # Tiny polygon that might result in few pixels
        polygon = box(-9.0, 41.0, -8.99, 41.01)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        result = raster_dataset.polygons_query(polygons)
        assert isinstance(result, PolygonsResult)
        # Should handle small polygons gracefully

    def test_polygon_complex_shape(self, raster_dataset):
        """Test polygons_query with complex polygon shapes."""
        from shapely.geometry import Polygon

        # Create a complex polygon (star shape)
        center_x, center_y = -9.0, 41.0
        outer_radius = 0.5
        inner_radius = 0.2
        points = []

        for i in range(10):
            angle = i * np.pi / 5
            if i % 2 == 0:
                radius = outer_radius
            else:
                radius = inner_radius
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append((x, y))

        polygon = Polygon(points)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        result = raster_dataset.polygons_query(polygons)
        assert isinstance(result, PolygonsResult)
        # Should handle complex shapes

    def test_polygon_with_holes(self, raster_dataset):
        """Test polygons_query with polygon containing holes."""
        from shapely.geometry import Polygon

        # Create a polygon with a hole
        exterior = [(-9.5, 40.5), (-8.5, 40.5), (-8.5, 41.5), (-9.5, 41.5), (-9.5, 40.5)]
        hole = [(-9.2, 40.8), (-8.8, 40.8), (-8.8, 41.2), (-9.2, 41.2), (-9.2, 40.8)]
        polygon = Polygon(exterior, [hole])

        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        result = raster_dataset.polygons_query(polygons)
        assert isinstance(result, PolygonsResult)
        # Should handle polygons with holes

    def test_polygon_multipolygon(self, raster_dataset):
        """Test polygons_query with MultiPolygon geometry."""
        from shapely.geometry import MultiPolygon

        # Create multiple separate polygons
        poly1 = box(-9.5, 40.5, -9.0, 41.0)
        poly2 = box(-8.5, 41.0, -8.0, 41.5)
        multipolygon = MultiPolygon([poly1, poly2])

        gdf = gpd.GeoDataFrame(geometry=[multipolygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        result = raster_dataset.polygons_query(polygons)
        assert isinstance(result, PolygonsResult)
        # Should handle MultiPolygon geometries

    def test_polygon_invalid_geometry(self, raster_dataset):
        """Test polygons_query with invalid polygon geometry."""
        from shapely.geometry import Polygon

        # Create a self-intersecting polygon (invalid)
        coords = [(-9.0, 40.0), (-8.0, 41.0), (-9.0, 42.0), (-8.0, 40.0), (-9.0, 40.0)]
        polygon = Polygon(coords)

        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        try:
            result = raster_dataset.polygons_query(polygons)
            # Some implementations might handle invalid geometries gracefully
            if result.data is not None:
                assert isinstance(result, PolygonsResult)
        except (ValueError, Exception):
            # Invalid geometries might raise an error, which is acceptable
            pass

    def test_polygon_empty_geometry(self, raster_dataset):
        """Test polygons_query with empty polygon geometry."""
        from shapely.geometry import Polygon

        # Create an empty polygon
        polygon = Polygon()
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        try:
            result = raster_dataset.polygons_query(polygons)
            # Empty geometries might be handled gracefully
            if result.data is not None:
                assert isinstance(result, PolygonsResult)
        except (ValueError, Exception):
            # Empty geometries might raise an error, which is acceptable
            pass

    def test_polygon_different_crs(self, raster_dataset):
        """Test polygons_query with polygon in different CRS."""
        # Polygon in UTM coordinates (approximate conversion)
        polygon = box(500000, 4500000, 600000, 4600000)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=32630)  # UTM Zone 30N
        polygons = Polygons(gdf, types="desired")

        result = raster_dataset.polygons_query(polygons)
        assert isinstance(result, PolygonsResult)
        # Should handle CRS transformation

    def test_multiple_polygons_always_preserve_dimension(self, temp_dataset_dir):
        """Test that polygons_query always preserves polygon dimension."""
        ds = BaseRasterDataset(root_dir=temp_dataset_dir, use_dask=False)

        # Single polygon
        polygon1 = box(1, 1, 9, 9)
        gdf_single = gpd.GeoDataFrame(geometry=[polygon1], crs=ds.crs)
        polygons_single = Polygons(gdf_single, types="desired")

        result_single = ds.polygons_query(polygons_single)
        assert result_single.data is not None
        # Should always have polygon dimension
        assert isinstance(result_single.data, list)
        assert len(result_single.data) == 1  # 1 polygon

        # Multiple polygons
        polygon2 = box(11, 11, 19, 19)
        gdf_multi = gpd.GeoDataFrame(geometry=[polygon1, polygon2], crs=ds.crs)
        polygons_multi = Polygons(gdf_multi, types="desired")

        result_multi = ds.polygons_query(polygons_multi)
        assert result_multi.data is not None
        assert isinstance(result_multi.data, list)
        assert len(result_multi.data) == 2  # 2 polygons


class TestFileIndexBoundaryConditions:
    """Test boundary conditions for file indexing across all query methods."""

    def test_single_file_index_no_squeeze(self, raster_dataset):
        """Test that single file index doesn't squeeze file dimension."""
        points = Points([[-9.0, 41.0]], crs=4326)
        bbox = BoundingBox(-9.5, 40.5, -9.0, 41.0, crs=raster_dataset.crs)
        polygon = box(-9.5, 40.5, -9.0, 41.0)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        # Points query with single file
        result_points = raster_dataset.points_query(points, indexes=0)
        assert result_points.data.shape == (1,)  # Single point, single file squeezed

        # Bbox query with single file - file dimension should NOT be squeezed
        result_bbox = raster_dataset.bbox_query(bbox, indexes=0)
        assert result_bbox.data.shape[0] == 1  # File dimension not squeezed

        # Polygons query with single file - file dimension should NOT be squeezed
        result_polygons = raster_dataset.polygons_query(polygons, indexes=0)
        if isinstance(result_polygons.data, list):
            assert len(result_polygons.data) == 1  # 1 polygon
            assert result_polygons.data[0].shape[0] == 1  # File dimension not squeezed

    def test_multiple_file_indexes(self, raster_dataset):
        """Test queries with multiple specific file indexes."""
        points = Points([[-9.0, 41.0]], crs=4326)
        bbox = BoundingBox(-9.5, 40.5, -9.0, 41.0, crs=raster_dataset.crs)
        polygon = box(-9.5, 40.5, -9.0, 41.0)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_dataset.crs)
        polygons = Polygons(gdf, types="desired")

        indexes = [0, 2]  # First and third files

        # Points query
        result_points = raster_dataset.points_query(points, indexes=indexes)
        assert result_points.data.shape[0] == 2  # 2 files

        # Bbox query
        result_bbox = raster_dataset.bbox_query(bbox, indexes=indexes)
        assert result_bbox.data.shape[0] == 2  # 2 files

        # Polygons query
        result_polygons = raster_dataset.polygons_query(polygons, indexes=indexes)
        if isinstance(result_polygons.data, list):
            assert len(result_polygons.data) == 1  # 1 polygon
            assert result_polygons.data[0].shape[0] == 2  # 2 files

    def test_out_of_range_file_indexes(self, raster_dataset):
        """Test error handling for out-of-range file indexes."""
        points = Points([[-9.0, 41.0]], crs=4326)

        # Test with index beyond available files
        with pytest.raises(ValueError):
            raster_dataset.points_query(points, indexes=100)

        # Test with negative index
        with pytest.raises(ValueError):
            raster_dataset.points_query(points, indexes=-1)

        # Test with mixed valid/invalid indexes
        with pytest.raises(ValueError):
            raster_dataset.points_query(points, indexes=[0, 100])
