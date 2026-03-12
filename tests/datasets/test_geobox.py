"""Tests for :mod:`faninsar.datasets.geobox` utilities."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj.crs import CRS
from rasterio.transform import Affine

from faninsar.datasets.geobox import GeoBox, GeoBoxTileGrid
from faninsar.query.bbox import BoundingBox


def make_geobox_dataset() -> GeoBox:
    """Helper to construct a simple GeoBox."""
    bbox = BoundingBox(0, 0, 100, 100, crs=CRS.from_epsg(32633))
    return GeoBox.from_geoinfo(
        crs=CRS.from_epsg(32633),
        bounds=bbox,
        res=(10, 10),
        dtype=np.float32,
        nodata=-9999,
    )


def test_geobox_dataset_basic_properties() -> None:
    """Ensure GeoBox captures grid metadata correctly."""
    vd = make_geobox_dataset()
    assert vd.width == 10
    assert vd.height == 10
    assert vd.res == (10, 10)
    assert vd.nodata == -9999


def test_alignment_check() -> None:
    """GeoBox alignment detection should be robust."""
    vd = make_geobox_dataset()
    snapped = vd.align_to(vd)
    assert vd.is_aligned(snapped)

    shifted_transform = vd.transform * Affine.translation(0.5, 0.5)
    shifted_bounds = BoundingBox(5, 5, 105, 105, crs=vd.crs)
    shifted = GeoBox(
        crs=vd.crs,
        transform=shifted_transform,
        bounds=shifted_bounds,
        width=vd.width,
        height=vd.height,
        dtype=vd.dtype,
        nodata=vd.nodata,
        res=vd.res,
    )
    assert not vd.is_aligned(shifted)


def test_iter_tiles_and_mapping() -> None:
    """Iterating tiles must cover the full grid and compute overlaps."""
    vd = make_geobox_dataset()
    other = vd.resample((5, 5))

    mapping = vd.intersect_tiles(other, (5, 5), (10, 10))
    assert mapping  # there should be overlaps
    any_refs = next(iter(mapping.values()))
    assert isinstance(any_refs, list)
    assert any_refs  # non-empty reference list


def test_geobox_tile_grid_errors() -> None:
    """GeoBoxTileGrid should validate chunk shapes."""
    vd = make_geobox_dataset()
    with pytest.raises(ValueError):
        GeoBoxTileGrid(vd, (0, 256))
