"""Tests for public DEM category factories."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.dem import DEM, ConstantDEM, GridSpec, RasterDEM, SourceDEM


def _grid() -> GridSpec:
    return GridSpec(
        crs="EPSG:4326",
        transform=Affine(0.1, 0, 10, 0, -0.1, 2),
        height=4,
        width=4,
    )


def test_factories_return_public_categories_without_io() -> None:
    assert isinstance(DEM.from_source("glo30:pc"), SourceDEM)
    assert isinstance(DEM.from_constant(4), ConstantDEM)
    with pytest.raises(ValueError):
        DEM.from_constant(float("nan"))
    with pytest.raises(ValueError):
        DEM.from_source("unknown")


def test_constant_materializes_and_applies_target_validity() -> None:
    validity = np.ones((4, 4), dtype=bool)
    validity[0, 0] = False
    grid = GridSpec(
        crs="EPSG:4326",
        transform=Affine(0.1, 0, 10, 0, -0.1, 2),
        height=4,
        width=4,
        validity=validity,
    )
    raster = DEM.from_constant(4).to_raster(grid)
    assert isinstance(raster, RasterDEM)
    assert raster.array.shape == grid.shape
    assert raster.array.flags.writeable is False
    assert raster.array[0, 0] != raster.array[0, 0]
    assert np.all(raster.array[1:] == 4)


def test_source_without_cache_fails_at_materialization() -> None:
    with pytest.raises(ValueError, match="cache_dir"):
        DEM.from_source("auto").to_raster(_grid())
