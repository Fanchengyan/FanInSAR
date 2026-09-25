"""Regression tests for EGM2008 geoid sampling on multi-dimensional grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from faninsar.processing.geometry.geoid import GeoidSampler

if TYPE_CHECKING:
    from pathlib import Path


def _write_tiny_geoid(path: Path) -> np.ndarray:
    from rasterio.transform import from_origin

    data = (np.arange(36 * 36, dtype=np.float32).reshape(36, 36) % 100) * 0.01
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=36,
        width=36,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-180.0, 90.0, 10.0, 10.0),
    ) as dataset:
        dataset.write(data, 1)
    return data


def _mesh() -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(10.0, 30.0, 7)[:, None] * np.ones((1, 5))
    lon = np.linspace(-20.0, 20.0, 5)[None, :] * np.ones((7, 1))
    return lat, lon


def test_egm2008_sample_preserves_2d_shape(tmp_path: Path) -> None:
    """A 2-D coordinate grid must return undulations on the same grid."""
    _write_tiny_geoid(tmp_path / "egm2008.tif")
    sampler = GeoidSampler("egm2008", tmp_path / "egm2008.tif")
    lat, lon = _mesh()
    values = sampler.sample(lat, lon)
    assert values.shape == lat.shape
    assert np.all(np.isfinite(values))


def test_egm2008_sample_preserves_1d_shape(tmp_path: Path) -> None:
    """A 1-D coordinate vector must return undulations on the same vector."""
    _write_tiny_geoid(tmp_path / "egm2008.tif")
    sampler = GeoidSampler("egm2008", tmp_path / "egm2008.tif")
    lat, lon = _mesh()
    values = sampler.sample(lat.ravel(), lon.ravel())
    assert values.shape == lat.ravel().shape
    assert np.all(np.isfinite(values))
