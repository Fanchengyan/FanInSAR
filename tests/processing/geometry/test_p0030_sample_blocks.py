"""P0030 native-tile sampling must keep 128-row block edges finite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.io import MemoryFile

from faninsar.processing.geometry import GridSpec
from faninsar.processing.geometry.dem_sources import get_dem_source
from faninsar.processing.geometry.providers.p0030_adapter import (
    _sample_dataset,
    _sample_paths,
)

if TYPE_CHECKING:
    from pathlib import Path


def _point_source(*, height: int = 200, width: int = 80) -> MemoryFile:
    """In-memory Point GLO-30-like raster whose value is the source row index."""
    spacing = 1.0 / 3600.0
    transform = Affine(spacing, 0.0, -91.0, 0.0, -spacing, 0.0)
    values = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, None],
        (height, width),
    ).copy()
    memory = MemoryFile()
    with memory.open(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.update_tags(AREA_OR_POINT="Point")
        dataset.write(values, 1)
    return memory


def test_sample_dataset_keeps_128_block_last_row_when_source_row_is_integer() -> None:
    """A 128-row block whose max source index is an integer must still sample.

    ``_sample_biquintic`` reads taps ``floor(row)-1 .. floor(row)+4``. The
    per-block read window used ``ceil(max)+3``, which is one pixel short when
    ``max`` is an integer — the last target row of every 128-row chunk became
    NaN even though the Copernicus tile was finite. That is the 1-row stripe
    baked into managed ``dem.tif`` mosaics.
    """
    spacing = 1.0 / 3600.0
    target = Affine(spacing, 0.0, -91.0 - 0.5 * spacing, 0.0, -spacing, 0.5 * spacing)
    height, width = 130, 40
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
    )
    xs, ys = target * (columns, rows)
    with _point_source() as memory, memory.open() as dataset:
        sampled = _sample_dataset(
            dataset,
            np.asarray(xs),
            np.asarray(ys),
            target_crs="EPSG:4326",
            source_crs="EPSG:4326",
            source_nodata=None,
        )
    # Target row r maps to source row r exactly; row 127 is the 128-block edge.
    interior = sampled[:, 2:-2]
    assert np.isfinite(interior[126]).all()
    assert np.isfinite(interior[127]).all()
    assert np.isfinite(interior[128]).all()
    np.testing.assert_allclose(interior[127], 127.0, atol=1e-4)


def test_adjacent_tiles_share_the_spline_support(tmp_path: Path) -> None:
    """Four tiles form one continuous affine ramp, including their corner."""
    paths = []
    for row in range(2):
        for col in range(2):
            path = tmp_path / f"{row}-{col}.tif"
            transform = Affine(0.01, 0, col * 0.12, 0, -0.01, 1 - row * 0.12)
            yy, xx = np.indices((12, 12))
            values = (100 + 2 * (xx + col * 12) + 3 * (yy + row * 12)).astype(
                np.float32
            )
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=12,
                width=12,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
            ) as dataset:
                dataset.write(values, 1)
            paths.append(path)
    grid = GridSpec("EPSG:4326", Affine(0.01, 0, 0.075, 0, -0.01, 0.925), shape=(8, 8))
    sampled = _sample_paths(paths, get_dem_source("glo30"), grid)
    yy, xx = np.indices(grid.shape)
    np.testing.assert_allclose(sampled, 100 + 2 * (xx + 8) + 3 * (yy + 8), atol=1e-4)


@pytest.mark.parametrize("void_in_left", [True, False])
def test_partial_tile_overlap_valid_sample_fills_other_tile_void(
    tmp_path: Path, void_in_left: bool
) -> None:
    """Distinct partially overlapping tiles can complement missing source cells."""
    paths = []
    for col in range(2):
        path = tmp_path / f"tile-{col}.tif"
        yy, xx = np.indices((12, 12))
        values = (100 + 2 * (xx + col * 8) + 3 * yy).astype(np.float32)
        if (col == 0) == void_in_left:
            values[5, 10 if col == 0 else 2] = -9999
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=12,
            width=12,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            nodata=-9999,
            transform=Affine(0.01, 0, col * 0.08, 0, -0.01, 1),
        ) as dataset:
            dataset.write(values, 1)
        paths.append(path)
    grid = GridSpec("EPSG:4326", Affine(0.01, 0, 0.095, 0, -0.01, 0.955), shape=(1, 1))
    result = _sample_paths(paths, get_dem_source("glo30"), grid)
    np.testing.assert_allclose(result, [[135.0]], atol=1e-4)


def test_partial_tile_overlap_conflicting_valid_cells_raise(tmp_path: Path) -> None:
    """Distinct tile overlaps still reject conflicting valid elevations."""
    paths = []
    for col in range(2):
        path = tmp_path / f"tile-{col}.tif"
        yy, xx = np.indices((12, 12))
        values = (100 + 2 * (xx + col * 8) + 3 * yy).astype(np.float32)
        if col == 1:
            values[5, 2] += 10
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=12,
            width=12,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=Affine(0.01, 0, col * 0.08, 0, -0.01, 1),
        ) as dataset:
            dataset.write(values, 1)
        paths.append(path)
    grid = GridSpec("EPSG:4326", Affine(0.01, 0, 0.095, 0, -0.01, 0.955), shape=(1, 1))
    with pytest.raises(RuntimeError, match="overlapping DEM source files disagree"):
        _sample_paths(paths, get_dem_source("glo30"), grid)


def test_overlapping_native_cells_must_agree(tmp_path: Path) -> None:
    """Conflicting cells are detected before interpolation can hide them."""
    paths = []
    for number in range(2):
        path = tmp_path / f"duplicate-{number}.tif"
        values = np.ones((12, 12), dtype=np.float32)
        values[5, 5] += number
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=12,
            width=12,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=Affine(0.01, 0, 0, 0, -0.01, 1),
        ) as dataset:
            dataset.write(values, 1)
        paths.append(path)
    grid = GridSpec("EPSG:4326", Affine(0.01, 0, 0.045, 0, -0.01, 0.955), shape=(1, 1))
    with pytest.raises(RuntimeError, match="overlapping DEM source files disagree"):
        _sample_paths(paths, get_dem_source("glo30"), grid)


def test_single_tile_matches_existing_sampler(tmp_path: Path) -> None:
    """The native-lattice sampler preserves single-tile results."""
    path = tmp_path / "single.tif"
    transform = Affine(0.01, 0, 0, 0, -0.01, 1)
    values = np.arange(400, dtype=np.float32).reshape(20, 20)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(values, 1)
    grid = GridSpec("EPSG:4326", Affine(0.01, 0, 0.045, 0, -0.01, 0.955), shape=(8, 8))
    cols, rows = np.meshgrid(np.arange(8) + 0.5, np.arange(8) + 0.5)
    x, y = Affine(*grid.transform) * (cols, rows)
    with rasterio.open(path) as dataset:
        expected = _sample_dataset(
            dataset,
            x,
            y,
            target_crs=grid.crs,
            source_crs=dataset.crs,
            source_nodata=None,
        )
    np.testing.assert_allclose(
        _sample_paths([path], get_dem_source("glo30"), grid), expected, atol=1e-4
    )


def test_projected_tile_seam_uses_native_support(tmp_path: Path) -> None:
    """Adjacent projected source cells are sampled on their own lattice."""
    paths = []
    for col in range(2):
        path = tmp_path / f"projected-{col}.tif"
        yy, xx = np.indices((12, 12))
        values = (100 + 2 * (xx + col * 12) + 3 * yy).astype(np.float32)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=12,
            width=12,
            count=1,
            dtype="float32",
            crs="EPSG:32632",
            transform=Affine(30, 0, 500000 + col * 360, 0, -30, 4500000),
        ) as dataset:
            dataset.write(values, 1)
        paths.append(path)
    grid = GridSpec("EPSG:32632", Affine(30, 0, 500225, 0, -30, 4499865), shape=(1, 8))
    sampled = _sample_paths(paths, get_dem_source("glo30"), grid)
    np.testing.assert_allclose(
        sampled[0], 100 + 2 * (np.arange(8) + 8) + 3 * 5, atol=1e-4
    )


def test_antimeridian_tiles_share_support(tmp_path: Path) -> None:
    """Longitude wrapping joins adjacent native tiles rather than extrapolating."""
    paths = []
    for col, origin in enumerate((179.88, -180.0)):
        path = tmp_path / f"seam-{col}.tif"
        yy, xx = np.indices((12, 12))
        values = (100 + 2 * (xx + col * 12) + 3 * yy).astype(np.float32)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=12,
            width=12,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=Affine(0.01, 0, origin, 0, -0.01, 1),
        ) as dataset:
            dataset.write(values, 1)
        paths.append(path)
    grid = GridSpec(
        "EPSG:4326", Affine(0.01, 0, 179.955, 0, -0.01, 0.955), shape=(1, 8)
    )
    sampled = _sample_paths(paths, get_dem_source("glo30"), grid)
    np.testing.assert_allclose(
        sampled[0], 100 + 2 * (np.arange(8) + 8) + 3 * 5, atol=1e-4
    )


def test_nodata_support_remains_missing(tmp_path: Path) -> None:
    """A real void is not replaced with terrain from outside the tile."""
    path = tmp_path / "void.tif"
    values = np.full((12, 12), 42, dtype=np.float32)
    values[5, 5] = -9999
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=12,
        width=12,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        nodata=-9999,
        transform=Affine(0.01, 0, 0, 0, -0.01, 1),
    ) as dataset:
        dataset.write(values, 1)
    grid = GridSpec("EPSG:4326", Affine(0.01, 0, 0.045, 0, -0.01, 0.955), shape=(1, 1))
    assert np.isnan(_sample_paths([path], get_dem_source("glo30"), grid)).all()
