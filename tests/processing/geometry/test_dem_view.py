"""Tests for the shared lon/lat rdr2geo DEM view."""

from __future__ import annotations

import numpy as np
from affine import Affine

from faninsar.processing.geometry.api import RasterDEM
from faninsar.processing.geometry.dem_view import (
    clear_lonlat_dem_views,
    geodetic_crop_grid,
    lonlat_dem_view,
    native_spacing_deg,
)
from faninsar.processing.geometry.grids import GridSpec


def setup_function() -> None:
    """Isolate the process-level view cache."""
    clear_lonlat_dem_views()


def test_native_spacing_follows_source_product() -> None:
    """GLO-30 stays on a 1 arc-second lattice."""
    assert native_spacing_deg("glo30") == 1.0 / 3600.0
    assert native_spacing_deg("glo90") == 1.0 / 1200.0


def test_geodetic_crop_snaps_utm_bounds_to_source_pixels() -> None:
    """The geographic crop uses native spacing, not the UTM 30 m grid."""
    utm = GridSpec(
        "EPSG:32633",
        Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4200000.0),
        shape=(20, 20),
    )
    grid = geodetic_crop_grid(utm, 1.0 / 3600.0)
    assert grid.crs == "EPSG:4326"
    assert abs(grid.transform[0] - 1.0 / 3600.0) < 1.0e-12
    assert abs(grid.transform[4] + 1.0 / 3600.0) < 1.0e-12
    assert grid.width >= 6
    assert grid.height >= 6


def test_lonlat_view_keeps_geodetic_affine_in_degrees() -> None:
    """An EPSG:4326 RasterDEM is used as-is."""
    grid = GridSpec(
        "EPSG:4326",
        Affine(0.01, 0.0, 14.0, 0.0, -0.01, 38.0),
        shape=(12, 12),
    )
    dem = RasterDEM(
        array=np.full((12, 12), 123.0, dtype=np.float32),
        grid=grid,
        vertical_datum="ellipsoidal",
    )
    view = lonlat_dem_view(dem)
    assert view.longitude_start_deg == 14.0
    assert view.latitude_start_deg == 38.0
    assert view.longitude_spacing_deg == 0.01
    np.testing.assert_allclose(view.values, 123.0)


def test_projected_view_metadata_is_degrees_not_metres() -> None:
    """Fallback conversion must not pass UTM metres into the kernel affine."""
    size = 32
    grid = GridSpec(
        "EPSG:32633",
        Affine(30.0, 0.0, 320067.0, 0.0, -30.0, 4284852.0),
        shape=(size, size),
    )
    dem = RasterDEM(
        array=np.full((size, size), 123.0, dtype=np.float32),
        grid=grid,
        vertical_datum="ellipsoidal",
    )
    view = lonlat_dem_view(dem)
    assert abs(view.latitude_start_deg) <= 90.0
    assert abs(view.longitude_start_deg) <= 180.0
    assert abs(view.latitude_spacing_deg) < 1.0
    assert abs(view.longitude_spacing_deg) < 1.0
    assert np.isfinite(view.values).all()
    assert np.any(view.values == 0.0)
    assert np.any(np.isclose(view.values, 123.0, atol=1e-3))
