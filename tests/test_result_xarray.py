"""Tests for xarray-based query result classes."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from rasterio.transform import Affine

from faninsar.query.result import (
    BBoxesResult,
    PointsResult,
    PolygonsResult,
    QueryResult,
)


class TestPointsResult:
    """Test PointsResult with xarray DataArray."""

    def test_points_result_1d(self):
        """Test PointsResult with 1D data (single file, multiple points)."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_dict = {
            "data": data,
            "dims": "points:3"
        }

        result = PointsResult(data=result_dict)

        # Test that result is directly a DataArray
        assert isinstance(result, xr.DataArray)
        assert super(PointsResult, result).dims == ("points",)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result.values, data)

        # Test data access
        np.testing.assert_array_equal(result.values, data)
        assert result.dims == ("points",)
        assert len(result) == 3

    def test_points_result_2d_files_points(self):
        """Test PointsResult with 2D data (multiple files, multiple points)."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result_dict = {
            "data": data,
            "dims": "(files:3, points:2)"
        }

        result = PointsResult(data=result_dict)

        # Test that result is directly a DataArray
        assert isinstance(result, xr.DataArray)
        assert super(PointsResult, result).dims == ("files", "points")
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.values, data)

        # Test coordinates
        assert list(result.coords["files"]) == [0, 1, 2]
        assert list(result.coords["points"]) == [0, 1]

    def test_points_result_2d_bands_points(self):
        """Test PointsResult with 2D data (multiple bands, multiple points)."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result_dict = {
            "data": data,
            "dims": "(bands:2, points:2)"
        }

        result = PointsResult(data=result_dict)

        # Test that result is directly a DataArray
        assert isinstance(result, xr.DataArray)
        assert super(PointsResult, result).dims == ("bands", "points")
        assert result.shape == (2, 2)

    def test_points_result_3d(self):
        """Test PointsResult with 3D data (files, bands, points)."""
        data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
        result_dict = {
            "data": data,
            "dims": "(files:2, bands:2, points:2)"
        }

        result = PointsResult(data=result_dict)

        # Test that result is directly a DataArray
        assert isinstance(result, xr.DataArray)
        assert super(PointsResult, result).dims == ("files", "bands", "points")
        assert result.shape == (2, 2, 2)

    def test_points_result_new_dims_format(self):
        """Test PointsResult with new dims format (list of tuples)."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result_dict = {
            "data": data,
            "dims": [("files", 3), ("points", 2)]
        }

        result = PointsResult(data=result_dict)

        # Test that result is directly a DataArray
        assert isinstance(result, xr.DataArray)
        # Use super() to access the original xarray dims
        assert super(PointsResult, result).dims == ("files", "points")
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.values, data)

        # Test dims property returns tuple format
        assert result.dims == ("files", "points")

    def test_points_result_empty(self):
        """Test PointsResult with empty data."""
        data = np.array([], dtype=np.float32)
        result_dict = {
            "data": data,
            "dims": "points:0"
        }

        result = PointsResult(data=result_dict)
        assert result.size == 0


class TestBBoxesResult:
    """Test BBoxesResult with xarray Dataset."""

    def test_bboxes_result_3d(self):
        """Test BBoxesResult with 3D data (files, height, width)."""
        data = np.random.rand(2, 10, 10).astype(np.float32)
        transform = Affine.identity()
        result_dict = {
            "data": data,
            "dims": "files:2, height:10, width:10",
            "transforms": [transform]
        }

        result = BBoxesResult(data_vars=result_dict)

        # Test that result is directly a Dataset
        assert isinstance(result, xr.Dataset)
        assert "values" in result.data_vars
        assert result["values"].dims == ("files", "y", "x")
        assert result["values"].shape == (2, 10, 10)

        # Test transforms
        assert result.transforms == [transform]
        assert "transforms" in result.attrs

    def test_bboxes_result_4d_bands(self):
        """Test BBoxesResult with 4D data (files, bands, height, width)."""
        data = np.random.rand(2, 3, 10, 10).astype(np.float32)
        result_dict = {
            "data": data,
            "dims": "files:2, bands:3, height:10, width:10",
            "transforms": []
        }

        result = BBoxesResult(data_vars=result_dict)

        # Test that result is directly a Dataset
        assert isinstance(result, xr.Dataset)
        assert result["values"].dims == ("files", "bands", "y", "x")
        assert result["values"].shape == (2, 3, 10, 10)

    def test_bboxes_result_4d_multiple_bboxes(self):
        """Test BBoxesResult with 4D data (bbox, files, height, width)."""
        data = np.random.rand(3, 2, 10, 10).astype(np.float32)
        transforms = [Affine.identity(), Affine.identity(), Affine.identity()]
        result_dict = {
            "data": data,
            "dims": "bbox:3, files:2, height:10, width:10",
            "transforms": transforms
        }

        result = BBoxesResult(data_vars=result_dict)

        # Test that result is directly a Dataset
        assert isinstance(result, xr.Dataset)
        assert result["values"].dims == ("bbox", "files", "y", "x")
        assert result["values"].shape == (3, 2, 10, 10)

        # Test transforms
        assert result.transforms is not None
        assert len(result.transforms) == 3

    def test_bboxes_result_new_dims_format(self):
        """Test BBoxesResult with new dims format (list of tuples)."""
        data = np.random.rand(2, 10, 10).astype(np.float32)
        transform = Affine.identity()
        result_dict = {
            "data": data,
            "dims": [("files", 2), ("height", 10), ("width", 10)],
            "transforms": [transform]
        }

        result = BBoxesResult(data_vars=result_dict)

        # Test that result is directly a Dataset
        assert isinstance(result, xr.Dataset)
        assert "values" in result.data_vars
        assert result["values"].dims == ("files", "y", "x")
        assert result["values"].shape == (2, 10, 10)

        # Test dims property
        assert result["values"].dims == ("files", "y", "x")


class TestPolygonsResult:
    """Test PolygonsResult with xarray Dataset."""

    def test_polygons_result_single_polygon(self):
        """Test PolygonsResult with single polygon."""
        data_array = np.random.rand(2, 10, 10).astype(np.float32)  # (files, height, width)
        data = [data_array]  # List with one polygon
        transform = Affine.identity()
        mask = np.ones((10, 10), dtype=bool)

        result_dict = {
            "data": data,
            "dims": "(n_polygons:1, (files:2, height:10, width:10))",
            "transforms": [transform],
            "masks": [mask]
        }

        result = PolygonsResult(data_vars=result_dict)

        # Test that result is directly a Dataset
        assert isinstance(result, xr.Dataset)
        assert "values" in result.data_vars
        assert result["values"].dims == ("polygon", "files", "y", "x")
        assert result["values"].shape == (1, 2, 10, 10)

        # Test transforms and masks
        assert result.transforms is not None
        assert len(result.transforms) == 1
        assert result.masks is not None
        assert len(result.masks) == 1

    def test_polygons_result_multiple_polygons(self):
        """Test PolygonsResult with multiple polygons."""
        data_array1 = np.random.rand(2, 10, 10).astype(np.float32)
        data_array2 = np.random.rand(2, 10, 10).astype(np.float32)
        data = [data_array1, data_array2]
        transforms = [Affine.identity(), Affine.identity()]
        masks = [np.ones((10, 10), dtype=bool), np.ones((10, 10), dtype=bool)]

        result_dict = {
            "data": data,
            "dims": "(n_polygons:2, (files:2, height:10, width:10))",
            "transforms": transforms,
            "masks": masks
        }

        result = PolygonsResult(data_vars=result_dict)

        # Test that result is directly a Dataset
        assert isinstance(result, xr.Dataset)
        assert result["values"].shape == (2, 2, 10, 10)

        # Test transforms and masks
        assert result.transforms is not None
        assert len(result.transforms) == 2
        assert result.masks is not None
        assert len(result.masks) == 2


class TestQueryResult:
    """Test QueryResult with xarray DataTree."""

    def test_query_result_creation(self):
        """Test QueryResult creation with different result types."""
        # Create sample data
        points_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        points_dict = {"data": points_data, "dims": "points:3"}

        bbox_data = np.random.rand(2, 10, 10).astype(np.float32)
        bbox_dict = {
            "data": bbox_data,
            "dims": "files:2, height:10, width:10",
            "transforms": [Affine.identity()]
        }

        # Create QueryResult
        query_result = QueryResult(
            points=points_dict,
            boxes=bbox_dict,
            polygons=None
        )

        # Test individual results
        assert isinstance(query_result.points, PointsResult)
        assert isinstance(query_result.boxes, BBoxesResult)
        assert query_result.polygons is None

        # Test DataTree (QueryResult is now a DataTree itself)
        if hasattr(query_result, "children") and query_result.children:
            assert "points" in query_result.children
            assert "boxes" in query_result.children

    def test_query_result_empty(self):
        """Test QueryResult with no data."""
        query_result = QueryResult()

        assert query_result.points is None
        assert query_result.boxes is None
        assert query_result.polygons is None
        assert query_result.query is None


class TestBackwardCompatibility:
    """Test backward compatibility with legacy API."""

    def test_legacy_api_points(self):
        """Test that legacy API still works for PointsResult."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_dict = {"data": data, "dims": "points:3"}

        result = PointsResult(data=result_dict)

        # Test properties
        np.testing.assert_array_equal(result.values, data)
        assert result.dims == ("points",)
        assert len(result) == 3
        assert result.size > 0

    def test_legacy_api_bboxes(self):
        """Test that legacy API still works for BBoxesResult."""
        data = np.random.rand(2, 10, 10).astype(np.float32)
        transform = Affine.identity()
        result_dict = {
            "data": data,
            "dims": "files:2, height:10, width:10",
            "transforms": [transform]
        }

        result = BBoxesResult(data_vars=result_dict)

        # Test properties
        np.testing.assert_array_equal(result["values"].values, data)
        assert result.transforms == [transform]
        assert result.sizes["files"] == 2

    def test_legacy_api_polygons(self):
        """Test that legacy API still works for PolygonsResult."""
        data_array = np.random.rand(2, 10, 10).astype(np.float32)
        data = [data_array]
        transform = Affine.identity()
        mask = np.ones((10, 10), dtype=bool)

        result_dict = {
            "data": data,
            "dims": "(n_polygons:1, (files:2, height:10, width:10))",
            "transforms": [transform],
            "masks": [mask]
        }

        result = PolygonsResult(data_vars=result_dict)

        # Test properties
        assert result.transforms is not None
        assert len(result.transforms) == 1
        assert result.masks is not None
        assert len(result.masks) == 1
        assert result.sizes["polygon"] == 1
