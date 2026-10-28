"""Stack source-DEM handoff: prepare materializes, open reloads."""

# The repository intentionally keeps tests as a namespace package.
# ruff: noqa: INP001

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from affine import Affine

if TYPE_CHECKING:
    from pathlib import Path

from faninsar.processing.geometry import DEM, GridSpec, RasterDEM, SourceDEM
from faninsar.stack.session import Stack


def _grid() -> GridSpec:
    return GridSpec(
        crs="EPSG:4326",
        transform=Affine(0.1, 0.0, 10.0, 0.0, -0.1, 50.0),
        height=8,
        width=8,
    )


def _dummy_safes(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / "S1_20240101.SAFE", tmp_path / "S1_20240113.SAFE"]
    for path in paths:
        path.mkdir(exist_ok=True)
    return paths


def test_prepare_promotes_source_dem_and_reopen_reloads(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A SourceDEM becomes stack.dem after prepare and survives reopen."""
    grid = _grid()
    raster = RasterDEM(
        array=np.ones((8, 8), dtype=np.float32),
        grid=grid,
        vertical_datum="ellipsoidal",
    )
    cache = tmp_path / "dem-cache"
    cache.mkdir()
    source = DEM.from_source("auto", cache_dir=cache)

    def _fake_materialize(_self: Stack, _dem: object | None = None) -> RasterDEM:
        return raster

    monkeypatch.setattr(Stack, "materialize_dem", _fake_materialize)

    work = tmp_path / "work"
    stack = Stack._from_safes(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=grid,
        dem_cache_dir=cache,
    )
    assert isinstance(stack.config.dem, SourceDEM)
    assert stack.dem is None

    stack.prepare()
    assert isinstance(stack.dem, RasterDEM)
    assert (work / "dem" / "stack.tif").is_file()
    assert (work / "dem" / "stack.json").is_file()
    assert stack.dem.path == work / "dem" / "stack.tif"

    reopened = Stack._from_safes(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=grid,
        dem_cache_dir=cache,
    )
    reopened.prepare()
    assert isinstance(reopened.dem, RasterDEM)
    assert reopened.dem.grid == grid


def _utm_grid() -> GridSpec:
    return GridSpec(
        crs="EPSG:32633",
        transform=Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4200000.0),
        height=12,
        width=12,
    )


def test_prepare_writes_native_wgs84_from_source(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Projected Stack grids persist a native 1 arc-second geographic crop."""
    utm = _utm_grid()
    cache = tmp_path / "dem-cache"
    cache.mkdir()
    source = DEM.from_source("auto", cache_dir=cache)
    grids: list[str] = []

    def _fake_to_raster(
        _self: SourceDEM,
        grid: GridSpec,
        **_kwargs: object,
    ) -> RasterDEM:
        grids.append(str(grid.crs))
        return RasterDEM(
            array=np.full(grid.shape, 100.0, dtype=np.float32),
            grid=grid,
            vertical_datum="ellipsoidal",
            provenance={"product": "glo30", "provider": "pc"},
        )

    monkeypatch.setattr(SourceDEM, "to_raster", _fake_to_raster)

    work = tmp_path / "work"
    stack = Stack._from_safes(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=utm,
        dem_cache_dir=cache,
    )
    stack.prepare()

    assert str(stack.dem.crs) == "EPSG:32633"
    assert (work / "dem" / "stack.tif").is_file()
    assert (work / "dem" / "stack_wgs84.tif").is_file()
    wgs84 = stack._dem_wgs84
    assert wgs84 is not None
    assert str(wgs84.crs) == "EPSG:4326"
    assert abs(wgs84.grid.transform[0] - 1.0 / 3600.0) < 1.0e-12
    assert "EPSG:32633" in grids
    assert "EPSG:4326" in grids
    n_materialize = len(grids)

    reopened = Stack._from_safes(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=utm,
        dem_cache_dir=cache,
    )
    reopened.prepare()
    assert str(reopened.dem.crs) == "EPSG:32633"
    assert reopened._dem_wgs84 is not None
    assert str(reopened._dem_wgs84.crs) == "EPSG:4326"
    assert len(grids) == n_materialize


def _fake_source_to_raster(
    grids: list[GridSpec],
) -> object:
    """Record every materialization grid and return a matching raster."""

    def _fake_to_raster(
        _self: SourceDEM,
        grid: GridSpec,
        **_kwargs: object,
    ) -> RasterDEM:
        grids.append(grid)
        return RasterDEM(
            array=np.full(grid.shape, 100.0, dtype=np.float32),
            grid=grid,
            vertical_datum="ellipsoidal",
            provenance={"product": "glo30", "provider": "pc"},
        )

    return _fake_to_raster


def test_prepare_wgs84_dem_covers_working_frame_not_just_roi(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Processing-domain DEM covers selected footprints; public DEM stays on the ROI."""
    from faninsar.data.query.bbox import BoundingBox

    roi = BoundingBox(10.0, 40.0, 10.05, 40.05, crs="EPSG:4326")
    frame = (9.5, 39.5, 11.0, 41.0)
    cache = tmp_path / "dem-cache"
    cache.mkdir()
    source = DEM.from_source("auto", cache_dir=cache)
    grids: list[GridSpec] = []
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_source_to_raster(grids))

    work = tmp_path / "work-frame"
    stack = Stack._from_safes(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        roi=roi,
        dem_cache_dir=cache,
    )
    stack.config.extra["selected_footprints"] = (frame,)
    stack.prepare()

    public = stack.dem
    assert isinstance(public, RasterDEM)
    public_west, public_south, public_east, public_north = public.grid.bounds
    if str(public.grid.crs) != "EPSG:4326":
        from pyproj import Transformer

        transformer = Transformer.from_crs(public.grid.crs, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(
            [public_west, public_east, public_west, public_east],
            [public_south, public_south, public_north, public_north],
        )
        public_west, public_east = min(lons), max(lons)
        public_south, public_north = min(lats), max(lats)
    assert public_east - public_west < 0.2
    assert public_north - public_south < 0.2

    wgs84 = stack._dem_wgs84
    assert wgs84 is not None
    west, south, east, north = wgs84.grid.bounds
    assert west <= 9.5
    assert south <= 39.5
    assert east >= 11.0
    assert north >= 41.0
    assert (work / "dem" / "stack.tif").is_file()
    assert (work / "dem" / "stack_wgs84.tif").is_file()
    assert any(str(grid.crs) == "EPSG:4326" and grid.bounds[0] <= 9.5 for grid in grids)


def test_geographic_roi_grid_builds_larger_processing_dem(
    tmp_path: Path, monkeypatch: object
) -> None:
    """An EPSG:4326 ROI product grid still gets a working-frame geometry DEM."""
    roi_grid = GridSpec(
        crs="EPSG:4326",
        transform=Affine(0.001, 0.0, 10.0, 0.0, -0.001, 40.05),
        height=50,
        width=50,
    )
    frame = (9.5, 39.5, 11.0, 41.0)
    cache = tmp_path / "dem-cache"
    cache.mkdir()
    source = DEM.from_source("auto", cache_dir=cache)
    grids: list[GridSpec] = []
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_source_to_raster(grids))

    work = tmp_path / "work-geo-frame"
    stack = Stack._from_safes(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=roi_grid,
        dem_cache_dir=cache,
    )
    stack.config.extra["selected_footprints"] = (frame,)
    stack.prepare()

    public = stack.dem
    assert isinstance(public, RasterDEM)
    assert public.grid == roi_grid
    wgs84 = stack._dem_wgs84
    assert wgs84 is not None
    west, south, east, north = wgs84.grid.bounds
    assert west <= 9.5
    assert south <= 39.5
    assert east >= 11.0
    assert north >= 41.0
    assert wgs84.grid != roi_grid
    n_materialize = len(grids)

    reopened = Stack._from_safes(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=roi_grid,
        dem_cache_dir=cache,
    )
    reopened.prepare()
    restored = reopened._dem_wgs84
    assert restored is not None
    west, south, east, north = restored.grid.bounds
    assert west <= 9.5
    assert south <= 39.5
    assert east >= 11.0
    assert north >= 41.0
    assert reopened.dem.grid == roi_grid
    assert len(grids) == n_materialize
