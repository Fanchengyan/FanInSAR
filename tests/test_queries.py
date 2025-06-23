from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj.crs.crs import CRS
from rasterio.enums import Resampling
import rasterio.transform as rasterio_transform
from shapely.geometry import box

from faninsar.datasets import RasterDataset
from faninsar.query import (BBoxesResult, BoundingBox, Points, PointsResult,
                            Polygons, PolygonsResult)


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


class TestRasterDatasetQueries:

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
        assert result_single.data.shape[0] == 1  # 1 file
        assert result_single.data.shape[1] == 3  # 3 points

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

        # Test with specific file index
        result_single = raster_dataset.bbox_query(bbox, indexes=1)
        assert result_single.data.shape[0] == 1  # 1 file
        assert result_single.data.shape[1] == height
        assert result_single.data.shape[2] == width

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

        # Test with specific file index
        result_single = raster_dataset.polygons_query(polygons, indexes=2)
        if isinstance(result_single.data, list):
            assert len(result_single.data) == 1  # One polygon
            assert result_single.data[0].shape[0] == 1  # 1 file
        else:
            assert result_single.data.shape[0] == 1  # 1 file

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
        # Each polygon should have data from 3 files
        assert result_multi.data[0].shape[0] == 3
        assert result_multi.data[1].shape[0] == 3

    def test_invalid_indexes(self, raster_dataset):
        """Test error handling for invalid indexes."""
        points = Points([[50, 50]])

        # Test with out-of-range index
        with pytest.raises(ValueError):
            raster_dataset.points_query(points, indexes=100)

        # Test with negative index
        with pytest.raises(ValueError):
            raster_dataset.points_query(points, indexes=-1)
