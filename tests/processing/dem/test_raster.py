"""Tests for RasterDEM array, identity, regridding, and persistence."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.dem import GridSpec, RasterDEM


def _grid(size: int = 8) -> GridSpec:
    return GridSpec(
        crs="EPSG:4326",
        transform=Affine(0.1, 0, 10, 0, -0.1, 2),
        height=size,
        width=size,
    )


def test_array_conversion_and_identity() -> None:
    raster = RasterDEM(array=np.ones((8, 8), dtype=np.float32), grid=_grid())
    assert np.asarray(raster).shape == (8, 8)
    assert len(raster.identity) == 64
    assert raster.provenance["resampling"] == "isce-p0032-biquintic-6x6-v1"
    target = raster.to_raster(_grid(4))
    assert target.grid.shape == (4, 4)


def test_save_requires_new_geotiff(tmp_path) -> None:
    raster = RasterDEM(array=np.ones((8, 8), dtype=np.float32), grid=_grid())
    path = tmp_path / "dem.tif"
    raster.save(path)
    assert path.exists()
    with pytest.raises(FileExistsError):
        raster.save(path)
    with pytest.raises(ValueError):
        raster.save(tmp_path / "dem.xyz")


def test_raster_dem_rejects_undersized_source_without_bilinear_fallback() -> None:
    """The public DEM sampler requires the qualified P0032 6x6 support."""
    grid = _grid(4)
    raster = RasterDEM(array=np.ones((4, 4), dtype=np.float32), grid=grid)
    with pytest.raises(ValueError, match="6x6 source support"):
        raster.sample(np.array([1.5]), np.array([10.15]))
