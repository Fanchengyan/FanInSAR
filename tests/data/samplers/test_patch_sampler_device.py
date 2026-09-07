"""Tests for sampler tensor collation behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from pyproj.crs import CRS
from rasterio.transform import from_bounds

from faninsar.data.datasets import RasterDataset
from faninsar.data.samplers import RowSampler, tensor_collate


def _write_tile(path: Path, bounds: tuple[float, float, float, float], value: float) -> None:
    """Create a small GeoTIFF tile for testing."""
    height = width = 4
    transform = from_bounds(*bounds, width, height)
    data = np.full((height, width), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def raster_root(tmp_path: Path) -> Path:
    """Create a temporary directory with a single raster tile."""
    bounds = (0.0, 0.0, 4.0, 4.0)
    _write_tile(tmp_path / "tile_00.tif", bounds, value=1.0)
    return tmp_path


@pytest.fixture
def raster_dataset(raster_root: Path) -> RasterDataset:
    """Build a RasterDataset for sampler integration tests."""
    paths = sorted(raster_root.glob("*.tif"))
    return RasterDataset(paths=paths, verbose=False)


def test_tensor_collate_converts_numpy_to_torch_all() -> None:
    """tensor_collate with scope=all should convert numpy arrays into tensors."""
    sample = {
        "data": np.arange(6, dtype=np.float32).reshape(2, 3),
        "indexes": np.array([0, 1], dtype=np.int64),
        "meta": {"data": np.arange(3, dtype=np.float32)},
        "paths": ["a.tif", "b.tif"],
    }

    result = tensor_collate([sample], tensor_scope="all")

    assert isinstance(result["data"], torch.Tensor)
    assert result["data"].device.type == "cpu"
    assert isinstance(result["indexes"], torch.Tensor)
    assert result["paths"] == ["a.tif", "b.tif"]


def test_tensor_collate_converts_only_data_by_default() -> None:
    """tensor_collate should convert only data entries by default."""
    sample = {
        "data": np.arange(6, dtype=np.float32).reshape(2, 3),
        "indexes": np.array([0, 1], dtype=np.int64),
        "meta": {"data": np.arange(3, dtype=np.float32)},
        "paths": ["a.tif", "b.tif"],
    }

    result = tensor_collate([sample])

    assert isinstance(result["data"], torch.Tensor)
    assert isinstance(result["indexes"], np.ndarray)
    assert isinstance(result["meta"]["data"], torch.Tensor)
    assert result["paths"] == ["a.tif", "b.tif"]


def test_to_dataloader_defaults_to_tensor(
    raster_dataset: RasterDataset,
) -> None:
    """Samplers should return tensors by default."""
    sampler = RowSampler(raster_dataset, row_num=1, verbose=False)
    loader = sampler.to_dataloader(num_workers=0)

    sample = next(iter(loader))

    assert isinstance(sample["data"], torch.Tensor)


def test_to_dataloader_tensor_false_returns_numpy(
    raster_dataset: RasterDataset,
) -> None:
    """Samplers should keep numpy arrays when tensor=False."""
    sampler = RowSampler(raster_dataset, row_num=1, verbose=False)
    loader = sampler.to_dataloader(num_workers=0, tensor=False)

    sample = next(iter(loader))

    assert isinstance(sample["data"], np.ndarray)
