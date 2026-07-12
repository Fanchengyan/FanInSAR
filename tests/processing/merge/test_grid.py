"""Tests for the common merge grid spec and builder."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.merge.grid import (
    GeoGridSpec,
    build_geo_grid,
)


def test_geo_grid_spec_round_trip() -> None:
    """GeoGridSpec stores crs, transform, shape, resolution, bbox."""
    grid = GeoGridSpec(
        crs="EPSG:32633",
        transform=(300000.0, 15.0, 0.0, 5000000.0, 0.0, -5.0),
        width=100,
        height=200,
        resolution_m=(15.0, 5.0),
        bbox=(300000.0, 4999000.0, 301500.0, 5000000.0),
    )
    assert grid.crs == "EPSG:32633"
    assert grid.width == 100
    assert grid.height == 200
    assert grid.resolution_m == (15.0, 5.0)
    assert grid.shape == (200, 100)


def test_geo_grid_spec_rejects_non_positive_shape() -> None:
    """Width/height must be positive."""
    with pytest.raises(ValueError):
        GeoGridSpec(
            crs="EPSG:32633",
            transform=(300000.0, 15.0, 0.0, 5000000.0, 0.0, -5.0),
            width=0,
            height=10,
            resolution_m=(15.0, 5.0),
            bbox=(0.0, 0.0, 1.0, 1.0),
        )


def test_build_geo_grid_auto_utm_single_footprint() -> None:
    """build_geo_grid picks a UTM zone from a lon/lat footprint."""
    # A small box near Rome (lon ~12.5, lat ~41.9) -> UTM zone 33N
    footprints = [((12.4, 41.8), (12.6, 42.0))]
    grid = build_geo_grid(
        footprints,
        resolution_m=(15.0, 15.0),
        crs="auto_utm",
        margin_m=0.0,
    )
    assert grid.crs == "EPSG:32633"
    assert grid.resolution_m == (15.0, 15.0)
    assert grid.width > 0
    assert grid.height > 0
    # bbox covers the footprint with some margin in CRS units
    assert grid.bbox[0] < grid.bbox[2]
    assert grid.bbox[1] < grid.bbox[3]


def test_build_geo_grid_union_of_multiple_footprints() -> None:
    """build_geo_grid takes the union of several footprints."""
    footprints = [
        ((12.4, 41.8), (12.6, 42.0)),
        ((12.5, 41.9), (12.7, 42.1)),
    ]
    grid = build_geo_grid(
        footprints,
        resolution_m=(20.0, 20.0),
        crs="auto_utm",
        margin_m=0.0,
    )
    # Union is wider/taller than either single footprint grid
    single = build_geo_grid(
        [((12.4, 41.8), (12.6, 42.0))],
        resolution_m=(20.0, 20.0),
        crs="auto_utm",
        margin_m=0.0,
    )
    assert grid.width >= single.width
    assert grid.height >= single.height


def test_build_geo_grid_applies_margin() -> None:
    """Margin expands the grid extent."""
    footprints = [((12.4, 41.8), (12.6, 42.0))]
    small = build_geo_grid(
        footprints, resolution_m=(15.0, 15.0), crs="auto_utm", margin_m=0.0
    )
    large = build_geo_grid(
        footprints, resolution_m=(15.0, 15.0), crs="auto_utm", margin_m=1000.0
    )
    assert large.width >= small.width
    assert large.height >= small.height


def test_build_geo_grid_explicit_crs() -> None:
    """An explicit EPSG code is honored directly."""
    footprints = [((12.4, 41.8), (12.6, 42.0))]
    grid = build_geo_grid(
        footprints,
        resolution_m=(15.0, 15.0),
        crs="EPSG:3857",
        margin_m=0.0,
    )
    assert grid.crs == "EPSG:3857"


def test_geo_grid_spec_bbox_property_matches_transform() -> None:
    """bbox derived from transform + shape when not given explicitly."""
    grid = GeoGridSpec(
        crs="EPSG:32633",
        transform=(300000.0, 15.0, 0.0, 5000000.0, 0.0, -5.0),
        width=100,
        height=200,
        resolution_m=(15.0, 5.0),
    )
    # bbox defaults from transform
    west, south, east, north = grid.bbox
    assert east == pytest.approx(300000.0 + 100 * 15.0)
    assert north == pytest.approx(5000000.0)
    assert south == pytest.approx(5000000.0 - 200 * 5.0)
    assert west == pytest.approx(300000.0)