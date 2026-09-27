"""Automatic DEM unpublished Copernicus water tiles (PROPOSAL-0053).

The Stack ``auto`` path is :func:`p0030_adapter._materialize_auto` via
``DEM.from_source("auto").to_raster``.  These tests mock HTTP HEAD/GET through
``dem_transport.thread_local_session`` and serve fake GeoTIFF bodies the same
way as :mod:`tests.processing.geometry.test_dem_manager`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.geometry import DEM, GridSpec, RasterDEM
from faninsar.processing.geometry.dem_sources import get_dem_source
from faninsar.processing.geometry.prepare_production import (
    _public_raster_dem_native_arrays,
)
from faninsar.processing.geometry.providers.p0030_adapter import _bounds
from tests.processing.geometry.test_dem_manager import (
    HttpFake,
    _glo30_url_at,
    _glo90_url,
    _seed_body,
)

if TYPE_CHECKING:
    from pathlib import Path


def _lonlat_grid(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    width: int = 16,
    height: int = 24,
) -> GridSpec:
    """Build a north-up EPSG:4326 grid covering the given bounds."""
    transform = Affine(
        (east - west) / width,
        0.0,
        west,
        0.0,
        -(north - south) / height,
        north,
    )
    return GridSpec("EPSG:4326", transform, height=height, width=width)


def _pixel_centers(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return longitude and latitude of pixel centres."""
    columns, rows = np.meshgrid(
        np.arange(grid.width, dtype=np.float64) + 0.5,
        np.arange(grid.height, dtype=np.float64) + 0.5,
    )
    transform = Affine(*grid.transform)
    longitude, latitude = transform * (columns, rows)
    return np.asarray(longitude), np.asarray(latitude)


def _materialize_auto(grid: GridSpec, cache_dir: Path) -> RasterDEM:
    """Materialize the Stack auto DEM on ``grid`` in the source datum."""
    return DEM.from_source("auto", cache_dir=cache_dir).to_raster(
        grid, vertical_datum="egm2008"
    )


def test_land_and_unpublished_ocean_fill_zero_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Land 200 plus ocean 404/404 yields finite land, 0 m ocean, and a warning."""
    fake = HttpFake().install(monkeypatch)
    fake.register_ok(_glo30_url_at(37, 15), _seed_body(tmp_path, 37, 15, 123.0))
    fake.register_missing(_glo30_url_at(36, 15))
    fake.register_missing(_glo90_url(36, 15))

    grid = _lonlat_grid(15.3, 36.4, 15.7, 37.6)
    with caplog.at_level(
        logging.WARNING,
        logger="faninsar.processing.geometry.providers.p0030_adapter",
    ):
        raster = _materialize_auto(grid, tmp_path / "cache")

    assert raster.grid.crs == grid.crs
    assert raster.array.shape == grid.shape
    longitude, latitude = _pixel_centers(grid)
    interior = (longitude >= 15.4) & (longitude <= 15.6)
    land = interior & (latitude >= 37.25) & (latitude <= 37.5)
    ocean = interior & (latitude >= 36.5) & (latitude <= 36.75)
    assert land.any()
    assert ocean.any()
    assert np.isfinite(raster.array[land]).all()
    assert np.allclose(raster.array[land], 123.0, atol=1.0)
    assert np.isfinite(raster.array[ocean]).all()
    assert np.allclose(raster.array[ocean], 0.0, atol=1e-6)
    warning_text = " ".join(record.getMessage() for record in caplog.records)
    assert "unpublished" in warning_text.lower()
    assert "N36_E015" in warning_text
    assert not any(
        method == "GET" and "copernicus-dem-90m" in url for method, url in fake.calls
    )
    assert not any(
        method == "GET" and "N36_00_E015" in url for method, url in fake.calls
    )
    assert not any("nasadem" in url or "skadi" in url for _, url in fake.calls)


def test_whole_roi_unpublished_raises_and_does_not_mint_zero_dem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetch window with no published GLO-30/GLO-90 tiles must fail closed."""
    fake = HttpFake().install(monkeypatch)
    fake.register_missing(_glo30_url_at(36, 13))
    fake.register_missing(_glo90_url(36, 13))

    grid = _lonlat_grid(13.3, 36.3, 13.7, 36.7, width=8, height=8)
    with pytest.raises(RuntimeError, match="no published GLO-30/GLO-90"):
        _materialize_auto(grid, tmp_path / "cache")
    assert not any(method == "GET" for method, _url in fake.calls)


def test_one_km_buffer_expands_lonlat_tile_list() -> None:
    """A grid hugging a 1° corner must list neighbouring cells after the 1 km pad."""
    grid = _lonlat_grid(15.001, 37.001, 15.02, 37.02, width=8, height=8)
    unpadded_west, unpadded_south, unpadded_east, unpadded_north = (
        15.001,
        37.001,
        15.02,
        37.02,
    )
    padded = _bounds(grid)
    assert padded[0] < unpadded_west
    assert padded[1] < unpadded_south
    assert padded[2] > unpadded_east
    assert padded[3] > unpadded_north
    # ~1 km, not a 10 km-scale pad.
    assert padded[0] > unpadded_west - 0.03
    assert padded[1] > unpadded_south - 0.03

    unpadded_tags = {
        tile.cache_path.parts[0]
        for tile in get_dem_source("auto")
        .plan((unpadded_west, unpadded_south, unpadded_east, unpadded_north))
        .tiles
    }
    padded_tags = {
        tile.cache_path.parts[0] for tile in get_dem_source("auto").plan(padded).tiles
    }
    assert unpadded_tags == {"N37_E015"}
    assert "N36_E015" in padded_tags
    assert "N37_E014" in padded_tags


def test_projected_fetch_window_pads_one_km_in_grid_crs() -> None:
    """Projected grids expand 1 km in metres before the WGS84 tile listing."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    origin_x, origin_y = transformer.transform(15.001, 37.5)
    size = 8
    resolution = 12.5
    grid = GridSpec(
        "EPSG:32633",
        Affine(
            resolution,
            0.0,
            origin_x,
            0.0,
            -resolution,
            origin_y + size * resolution,
        ),
        height=size,
        width=size,
    )
    left, bottom, right, top = grid.bounds
    to_lonlat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    corners_x = np.array([left, left, right, right])
    corners_y = np.array([bottom, top, bottom, top])
    unpadded_lon, unpadded_lat = to_lonlat.transform(corners_x, corners_y)
    padded = _bounds(grid)
    assert padded[0] < float(np.min(unpadded_lon))
    assert padded[2] > float(np.max(unpadded_lon))
    padded_tags = {
        tile.cache_path.parts[0] for tile in get_dem_source("auto").plan(padded).tiles
    }
    unpadded_tags = {
        tile.cache_path.parts[0]
        for tile in get_dem_source("auto")
        .plan(
            (
                float(np.min(unpadded_lon)),
                float(np.min(unpadded_lat)),
                float(np.max(unpadded_lon)),
                float(np.max(unpadded_lat)),
            )
        )
        .tiles
    }
    assert unpadded_tags == {"N37_E015"}
    assert len(padded_tags) > len(unpadded_tags)


def test_glo90_twin_is_rescued_when_head_returns_200(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A glo30 404 with a published glo90 twin still uses GLO-90 rescue."""
    fake = HttpFake().install(monkeypatch)
    fake.register_ok(_glo30_url_at(39, 45), _seed_body(tmp_path, 39, 45, 55.0))
    fake.register_missing(_glo30_url_at(38, 45))
    fake.register_ok(_glo90_url(38, 45), _seed_body(tmp_path, 38, 45, 77.0))

    grid = _lonlat_grid(45.3, 38.4, 45.7, 39.6)
    raster = _materialize_auto(grid, tmp_path / "cache")
    longitude, latitude = _pixel_centers(grid)
    interior = (longitude >= 45.4) & (longitude <= 45.6)
    primary = interior & (latitude >= 39.25) & (latitude <= 39.5)
    rescued = interior & (latitude >= 38.5) & (latitude <= 38.75)
    assert primary.any()
    assert rescued.any()
    assert np.isfinite(raster.array[primary]).all()
    assert np.allclose(raster.array[primary], 55.0, atol=1.0)
    assert np.isfinite(raster.array[rescued]).all()
    assert np.allclose(raster.array[rescued], 77.0, atol=1.0)
    assert any(
        method == "GET" and "copernicus-dem-90m" in url and "N38_00_E045" in url
        for method, url in fake.calls
    )


def test_prepare_production_view_bulge_still_fills_zero() -> None:
    """Projected lon/lat view corners outside the footprint still fill 0 m."""
    size = 256
    grid = GridSpec(
        "EPSG:32633",
        (30.0, 0.0, 320067.0, 0.0, -30.0, 4284852.0),
        shape=(size, size),
    )
    dem = RasterDEM(
        array=np.full((size, size), 123.0, dtype=np.float32),
        grid=grid,
        vertical_datum="ellipsoidal",
    )
    values, metadata, bounds = _public_raster_dem_native_arrays(dem)
    assert np.isfinite(values).all()
    assert np.isfinite(metadata).all()
    assert np.isfinite(bounds).all()
    assert np.any(values == 0.0)
    assert np.any(np.isclose(values, 123.0, atol=1e-3))
