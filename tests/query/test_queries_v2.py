"""New tests for RasterDataset queries using xarray DataArray/DataTree."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from shapely.geometry import box
import xarray as xr

from faninsar.datasets import RasterDataset
from faninsar.query import BoundingBox, Points, Polygons


def _create_test_tiff(path: Path, bounds: tuple[float, float, float, float], data: np.ndarray, crs: str = "EPSG:4326") -> None:
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1, dtype=data.dtype, crs=crs, transform=transform
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    tmp = tmp_path / "tiffs"
    tmp.mkdir(exist_ok=True)
    h, w = 64, 80
    bounds = (-10, 40, -8, 42)
    for i in range(3):
        arr = (np.ones((h, w), dtype=np.float32) * (i + 1)) + np.linspace(0, 1, w)[None, :] + np.linspace(0, 1, h)[:, None]
        _create_test_tiff(tmp / f"f{i}.tif", bounds, arr)
    return tmp


@pytest.fixture
def ds(sample_dir: Path) -> RasterDataset:
    return RasterDataset(root_dir=sample_dir, verbose=False, resampling=Resampling.nearest)


def test_points_query_returns_dataset(ds: RasterDataset) -> None:
    pts = Points([(-9.5, 40.5), (-9.0, 41.0), (-8.2, 41.8)], crs=4326)
    ds_points = ds.points_query(pts)
    assert isinstance(ds_points, xr.Dataset)
    assert "data" in ds_points
    assert ds_points["data"].dims[0] == "file" and ds_points["data"].dims[-1] == "point"
    assert ds_points["data"].shape[0] == 3 and ds_points["data"].shape[-1] == 3


def test_box_query_returns_datatree(ds: RasterDataset) -> None:
    bbox = BoundingBox(-9.5, 40.5, -9.0, 41.0, crs=ds.crs)
    tree = ds.boxes_query(bbox)
    # Single bbox -> dataset on root
    assert tree.dataset is not None
    ds_root = tree.dataset
    assert isinstance(ds_root, xr.Dataset)
    assert "data" in ds_root
    assert ds_root["data"].dims[0] == "file"


def test_polygons_query_returns_datatree(ds: RasterDataset) -> None:
    poly = box(-9.6, 40.6, -9.0, 41.2)
    gdf = gpd.GeoDataFrame(geometry=[poly], crs=ds.crs)
    polygons = Polygons(gdf, types="desired")
    tree = ds.polygons_query(polygons)
    # Single polygon -> dataset on root (or ragged children if shapes differ)
    if tree.dataset is not None and "data" in tree.dataset:
        assert tree.dataset["data"].dims[0] == "file"
    else:
        # per-file children indexed by 0..N-1
        files = list(tree.children.keys())
        assert len(files) == 3
