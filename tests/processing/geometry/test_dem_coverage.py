"""Admission tests: a raster DEM must over-cover the target grid footprint."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Transformer

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.api import (
    GridSpec,
    RasterDEM,
    ensure_dem_covers_grid,
)


def _target_grid() -> GridSpec:
    return GridSpec(
        "EPSG:32633",
        (30.0, 0.0, 320000.0, 0.0, -30.0, 4290000.0),
        shape=(200, 200),
    )


def _corner_bbox_lonlat(grid: GridSpec) -> tuple[float, float, float, float]:
    west, south, east, north = grid.bounds
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    xs, ys = transformer.transform(
        [west, east, west, east], [south, south, north, north]
    )
    return min(xs), min(ys), max(xs), max(ys)


def _geographic_dem(
    west: float, south: float, east: float, north: float, spacing: float = 0.001
) -> RasterDEM:
    shape = (
        round((north - south) / spacing),
        round((east - west) / spacing),
    )
    return RasterDEM(
        array=np.zeros(shape, dtype=np.float32),
        grid=GridSpec(
            "EPSG:4326",
            (spacing, 0.0, west, 0.0, -spacing, north),
            shape=shape,
        ),
        vertical_datum="ellipsoidal",
    )


def test_dem_larger_than_footprint_is_admitted() -> None:
    """A geographic DEM clearly larger than the target passes admission."""
    west, south, east, north = _corner_bbox_lonlat(_target_grid())
    dem = _geographic_dem(west - 0.5, south - 0.5, east + 0.5, north + 0.5)
    ensure_dem_covers_grid(dem, _target_grid())


def test_exact_footprint_without_margin_is_rejected() -> None:
    """Admission requires a sample margin beyond the corner-poked bbox."""
    west, south, east, north = _corner_bbox_lonlat(_target_grid())
    dem = _geographic_dem(west, south, east, north)
    with pytest.raises(InvalidProcessingStateError, match="deg"):
        ensure_dem_covers_grid(dem, _target_grid())


def test_same_crs_shortfall_names_missing_sides_in_km() -> None:
    """A projected DEM slightly smaller than the grid fails per side."""
    target = _target_grid()
    smaller = RasterDEM(
        array=np.zeros((200, 200), dtype=np.float32),
        grid=GridSpec(
            "EPSG:32633",
            (30.0, 0.0, 320100.0, 0.0, -30.0, 4290000.0),
            shape=(200, 200),
        ),
        vertical_datum="ellipsoidal",
    )
    with pytest.raises(InvalidProcessingStateError, match="west short by"):
        ensure_dem_covers_grid(smaller, target)
