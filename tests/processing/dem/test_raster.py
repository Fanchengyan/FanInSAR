"""Tests for RasterDEM array, identity, regridding, and persistence."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.dem import GridSpec, RasterDEM
from faninsar.processing.dem import api as dem_api


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


def test_same_grid_datum_conversion_does_not_require_6x6_support(monkeypatch) -> None:
    """Convert a small authoritative raster pointwise without terrain sampling."""
    grid = GridSpec(
        crs="EPSG:4326",
        transform=Affine(1, 0, 100, 0, -1, 10),
        shape=(2, 2),
        validity=np.array([[True, False], [True, True]]),
    )
    source = RasterDEM(
        array=np.array([[10.0, 20.0], [30.0, np.nan]], dtype=np.float32),
        grid=grid,
        vertical_datum="egm2008",
        nodata=-9999.0,
        provenance={"source": "fixture"},
    )

    def fail_sampler(*args, **kwargs):
        raise AssertionError("same-grid datum conversion must not resample terrain")

    monkeypatch.setattr(dem_api, "_sample_biquintic", fail_sampler)

    from faninsar.processing.dem import datum

    def fake_convert(heights, longitude, latitude, source_datum, target_datum, **kwargs):
        assert source_datum == "egm2008"
        assert target_datum == "ellipsoidal"
        return np.asarray(heights, dtype=np.float64) + 7.5

    monkeypatch.setattr(datum, "convert_heights", fake_convert)
    converted = source.to_raster(grid, vertical_datum="ellipsoidal")

    np.testing.assert_allclose(converted.array[0, 0], 17.5)
    assert np.isnan(converted.array[0, 1])
    assert np.isnan(converted.array[1, 1])
    assert converted.provenance["resampling"] == "none"
    assert converted.provenance["source_datum"] == "egm2008"
    assert converted.identity != source.identity
    assert converted.nodata == source.nodata


def test_same_grid_same_datum_is_identity() -> None:
    """A same-grid, same-datum request returns the existing materialization."""
    grid = _grid(2)
    source = RasterDEM(
        array=np.ones(grid.shape, dtype=np.float32),
        grid=grid,
        vertical_datum="egm2008",
    )
    assert source.to_raster(grid, vertical_datum="egm2008") is source


def test_changed_grid_still_requires_p0032_support() -> None:
    """A real grid change remains strict and has no bilinear fallback."""
    source_grid = _grid(2)
    target_grid = GridSpec(
        crs="EPSG:4326",
        transform=Affine(0.2, 0, 10, 0, -0.2, 2),
        shape=(2, 2),
    )
    source = RasterDEM(
        array=np.ones(source_grid.shape, dtype=np.float32), grid=source_grid
    )
    with pytest.raises(ValueError, match="6x6 source support"):
        source.to_raster(target_grid)
