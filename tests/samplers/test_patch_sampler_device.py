"""Tests for sampler device-aware tensor collation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from pyproj.crs import CRS
from rasterio.transform import from_bounds

from faninsar.datasets import RasterDataset
from faninsar.samplers import RowSampler, tensor_collate


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


def test_tensor_collate_converts_numpy_to_torch() -> None:
    """tensor_collate should convert numpy arrays into torch tensors."""
    sample = {
        "data": np.arange(6, dtype=np.float32).reshape(2, 3),
        "indexes": np.array([0, 1], dtype=np.int64),
        "paths": ["a.tif", "b.tif"],
    }

    result = tensor_collate([sample], device=torch.device("cpu"))

    assert isinstance(result["data"], torch.Tensor)
    assert result["data"].device.type == "cpu"
    assert isinstance(result["indexes"], torch.Tensor)
    assert result["paths"] == ["a.tif", "b.tif"]


def test_to_dataloader_uses_identity_collate_when_device_is_none(
    raster_dataset: RasterDataset,
) -> None:
    """Samplers without a device should keep numpy arrays."""
    sampler = RowSampler(raster_dataset, row_num=1, device=None, verbose=False)
    loader = sampler.to_dataloader(num_workers=0)

    sample = next(iter(loader))

    assert isinstance(sample["data"], np.ndarray)


def test_to_dataloader_uses_tensor_collate_when_device_set(
    raster_dataset: RasterDataset,
) -> None:
    """Samplers with a device should return torch tensors."""
    sampler = RowSampler(raster_dataset, row_num=1, device="cpu", verbose=False)
    loader = sampler.to_dataloader(num_workers=0)

    sample = next(iter(loader))

    assert isinstance(sample["data"], torch.Tensor)
    assert sample["data"].device.type == "cpu"
