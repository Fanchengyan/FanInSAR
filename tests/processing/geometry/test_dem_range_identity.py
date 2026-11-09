"""PROPOSAL-0079: DEM identity is a cheap range/grid/datum descriptor."""

from __future__ import annotations

import numpy as np
from affine import Affine

from faninsar.processing.geometry import GridSpec, RasterDEM, dem_content_identity
from faninsar.processing.geometry.dem_view import clear_lonlat_dem_views, lonlat_dem_view


def _grid(size: int = 8, west: float = 10.0) -> GridSpec:
    return GridSpec(
        crs="EPSG:4326",
        transform=Affine(0.1, 0.0, west, 0.0, -0.1, 2.0),
        height=size,
        width=size,
    )


class _ForbiddenArray(np.ndarray):
    """Array that refuses pixel enumeration used by content hashing."""

    def tobytes(self, *args: object, **kwargs: object) -> bytes:
        raise AssertionError("DEM identity must not enumerate pixels")


def test_identity_does_not_hash_pixels_and_ignores_values() -> None:
    """Same grid and datum share identity even when heights differ.

    Identity must be computable from header metadata; hashing the DEM array
    would call ``tobytes`` on the stored samples.
    """
    grid = _grid()
    zeros = np.zeros((8, 8), dtype=np.float32).view(_ForbiddenArray)
    peaks = np.full((8, 8), 250.0, dtype=np.float32).view(_ForbiddenArray)
    first = RasterDEM(array=zeros, grid=grid, vertical_datum="ellipsoidal")
    second = RasterDEM(array=peaks, grid=grid, vertical_datum="ellipsoidal")
    assert first.identity == second.identity
    assert dem_content_identity(first) == dem_content_identity(second)
    first._array = zeros
    assert first.identity == second.identity


def test_identity_changes_with_bounds_grid_or_datum() -> None:
    """The descriptor distinguishes coverage, sampling grid, and datum."""
    base = RasterDEM(
        array=np.ones((8, 8), dtype=np.float32),
        grid=_grid(),
        vertical_datum="ellipsoidal",
    )
    shifted = RasterDEM(
        array=np.ones((8, 8), dtype=np.float32),
        grid=_grid(west=11.0),
        vertical_datum="ellipsoidal",
    )
    denser = RasterDEM(
        array=np.ones((12, 12), dtype=np.float32),
        grid=_grid(size=12),
        vertical_datum="ellipsoidal",
    )
    orthometric = RasterDEM(
        array=np.ones((8, 8), dtype=np.float32),
        grid=_grid(),
        vertical_datum="egm2008",
    )
    assert base.identity != shifted.identity
    assert base.identity != denser.identity
    assert base.identity != orthometric.identity


def test_equal_range_independent_dems_do_not_share_live_views() -> None:
    """Live array caches are owner-scoped, not merged by equal bounds."""
    clear_lonlat_dem_views()
    grid = _grid(size=12)
    low = RasterDEM(
        array=np.full((12, 12), 10.0, dtype=np.float32),
        grid=grid,
        vertical_datum="ellipsoidal",
    )
    high = RasterDEM(
        array=np.full((12, 12), 90.0, dtype=np.float32),
        grid=grid,
        vertical_datum="ellipsoidal",
    )
    assert low.identity == high.identity
    first = lonlat_dem_view(low)
    second = lonlat_dem_view(high)
    assert first.values is not second.values
    np.testing.assert_allclose(first.values, 10.0)
    np.testing.assert_allclose(second.values, 90.0)
    again = lonlat_dem_view(low)
    assert again.values is first.values
