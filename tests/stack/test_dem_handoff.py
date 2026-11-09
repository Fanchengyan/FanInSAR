"""Stack source-DEM handoff: prepare materializes, open reloads."""

# The repository intentionally keeps tests as a namespace package.
# ruff: noqa: INP001

from __future__ import annotations

from tests.support.local_stack import LocalStack

from typing import TYPE_CHECKING

import numpy as np
import pytest
from affine import Affine

if TYPE_CHECKING:
    from pathlib import Path

from faninsar.processing.geometry import DEM, GridSpec, RasterDEM, SourceDEM
from faninsar.processing.geometry.managed_dem import (
    DEMCoverageInsufficientError,
    raster_covers_bounds,
)
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
    stack = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=grid,
        dem_cache_dir=cache,
    )
    assert isinstance(stack.config.dem, SourceDEM)
    assert stack.dem is None

    stack.prepare_data()
    assert isinstance(stack.dem, RasterDEM)
    assert (work / "dem.tif").is_file()
    assert (work / "dem.json").is_file()
    assert stack.dem.path == work / "dem.tif"

    reopened = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=grid,
        dem_cache_dir=cache,
    )
    reopened.prepare_data()
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
    """Projected Stack grids persist one geographic root DEM, not a UTM raster."""
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
    stack = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=utm,
        dem_cache_dir=cache,
    )
    stack.prepare_data()

    assert str(stack.dem.crs) == "EPSG:4326"
    assert (work / "dem.tif").is_file()
    assert not (work / "dem_wgs84.tif").exists()
    assert not (work / "dem" / "rgv_stack.tif").exists()
    assert abs(stack.dem.grid.transform[0] - 1.0 / 3600.0) < 1.0e-12
    assert grids == ["EPSG:4326"]
    n_materialize = len(grids)

    reopened = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=utm,
        dem_cache_dir=cache,
    )
    reopened.prepare_data()
    assert str(reopened.dem.crs) == "EPSG:4326"
    assert reopened.dem.grid == stack.dem.grid
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
    """Prepare covers the explicit ROI only; selected footprints are not prefetched."""
    from faninsar.data.query.bbox import BoundingBox

    roi = BoundingBox(10.0, 40.0, 10.05, 40.05, crs="EPSG:4326")
    frame = (9.5, 39.5, 11.0, 41.0)
    cache = tmp_path / "dem-cache"
    cache.mkdir()
    source = DEM.from_source("auto", cache_dir=cache)
    grids: list[GridSpec] = []
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_source_to_raster(grids))

    work = tmp_path / "work-frame"
    stack = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        roi=roi,
        dem_cache_dir=cache,
    )
    stack.config.extra["selected_footprints"] = (frame,)
    stack.prepare_data()

    public = stack.dem
    assert isinstance(public, RasterDEM)
    assert str(public.grid.crs) == "EPSG:4326"
    west, south, east, north = public.grid.bounds
    assert west <= 10.0
    assert south <= 40.0
    assert east >= 10.05
    assert north >= 40.05
    assert east - west < 0.5
    assert north - south < 0.5
    assert west > 9.5
    assert east < 11.0
    assert (work / "dem.tif").is_file()
    assert not (work / "dem_wgs84.tif").exists()
    assert all(str(grid.crs) == "EPSG:4326" for grid in grids)


def test_geographic_roi_grid_builds_larger_processing_dem(
    tmp_path: Path, monkeypatch: object
) -> None:
    """An EPSG:4326 product grid does not force a second DEM or footprint prefetch."""
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
    stack = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=roi_grid,
        dem_cache_dir=cache,
    )
    stack.config.extra["selected_footprints"] = (frame,)
    stack.prepare_data()

    public = stack.dem
    assert isinstance(public, RasterDEM)
    assert str(public.grid.crs) == "EPSG:4326"
    assert public.grid != roi_grid
    west, south, east, north = public.grid.bounds
    assert west <= roi_grid.bounds[0]
    assert south <= roi_grid.bounds[1]
    assert east >= roi_grid.bounds[2]
    assert north >= roi_grid.bounds[3]
    assert west > 9.5
    assert east < 11.0
    assert not (work / "dem_wgs84.tif").exists()
    n_materialize = len(grids)

    reopened = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        grid=roi_grid,
        dem_cache_dir=cache,
    )
    reopened.prepare_data()
    assert str(reopened.dem.crs) == "EPSG:4326"
    assert reopened.dem.grid == public.grid
    assert reopened.config.grid == roi_grid
    assert len(grids) == n_materialize


def test_use_time_shortage_grows_same_file_and_failed_expand_preserves_bytes(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A later uncovered request rewrites dem.tif in place; a failed fetch keeps it."""
    from faninsar.data.query.bbox import BoundingBox

    roi = BoundingBox(10.0, 40.0, 10.05, 40.05, crs="EPSG:4326")
    cache = tmp_path / "dem-cache"
    cache.mkdir()
    source = DEM.from_source("auto", cache_dir=cache)
    grids: list[GridSpec] = []
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_source_to_raster(grids))
    work = tmp_path / "work-expand"
    stack = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        roi=roi,
        dem_cache_dir=cache,
    )
    stack.prepare_data()
    managed = work / "dem.tif"
    assert managed.is_file()
    before = managed.read_bytes()
    first_grid = stack.dem.grid
    n_materialize = len(grids)

    def _boom(_self: SourceDEM, _grid: GridSpec, **_kwargs: object) -> RasterDEM:
        raise OSError("injected source outage")

    monkeypatch.setattr(SourceDEM, "to_raster", _boom)
    with pytest.raises(OSError, match="injected source outage"):
        stack._ensure_dem_covers_request((9.5, 39.5, 11.0, 41.0))
    assert managed.read_bytes() == before
    assert stack.dem.grid == first_grid

    monkeypatch.setattr(SourceDEM, "to_raster", _fake_source_to_raster(grids))
    stack._ensure_dem_covers_request((9.5, 39.5, 11.0, 41.0))
    assert managed.is_file()
    assert not (work / "dem_wgs84.tif").exists()
    west, south, east, north = stack.dem.grid.bounds
    assert west <= 9.5
    assert south <= 39.5
    assert east >= 11.0
    assert north >= 41.0
    assert len(grids) == n_materialize + 1


_USE_TIME_BOUNDS = (9.5, 39.5, 11.0, 41.0)


def _roi_bounded_stack(
    tmp_path: Path, monkeypatch: object
) -> tuple[LocalStack, list[GridSpec]]:
    """Prepare one ROI-bounded managed DEM; later use-time requests may grow it."""
    from faninsar.data.query.bbox import BoundingBox

    roi = BoundingBox(10.0, 40.0, 10.05, 40.05, crs="EPSG:4326")
    cache = tmp_path / "dem-cache"
    cache.mkdir()
    source = DEM.from_source("auto", cache_dir=cache)
    grids: list[GridSpec] = []
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_source_to_raster(grids))
    work = tmp_path / "work-use-time"
    stack = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        roi=roi,
        dem_cache_dir=cache,
    )
    stack.prepare_data()
    west, _south, east, _north = stack.dem.grid.bounds
    assert west > _USE_TIME_BOUNDS[0]
    assert east < _USE_TIME_BOUNDS[2]
    return stack, grids


def _drop_explicit_roi(stack: LocalStack) -> None:
    """Leave the prepared DEM in place and drop the ROI so prechecks become no-ops."""
    stack.config.roi = None
    stack._effective_roi_resolution = None
    stack._s1_roi_burst_selection_cache = {}


def _raise_if_short(dem: object, required: tuple[float, float, float, float]) -> None:
    """Raise the production coverage error when ``dem`` does not cover ``required``."""
    if not isinstance(dem, RasterDEM) or not raster_covers_bounds(dem, required):
        actual = None if not isinstance(dem, RasterDEM) else tuple(dem.grid.bounds)
        raise DEMCoverageInsufficientError(
            "actual geometry request is outside the managed DEM",
            required_bounds=required,
            actual_bounds=actual,
        )


def test_noroi_reference_prepare_retries_actual_callback_with_refreshed_dem(
    tmp_path: Path, monkeypatch: object
) -> None:
    """noROI reuse stays small until an actual reference prepare needs more terrain."""
    stack, grids = _roi_bounded_stack(tmp_path, monkeypatch)
    _drop_explicit_roi(stack)
    n_prepare = len(grids)
    dem_ids: list[int] = []

    def _fake_prepare(*_args: object, dem: object | None = None, **_kwargs: object) -> object:
        dem_ids.append(id(dem))
        _raise_if_short(dem, _USE_TIME_BOUNDS)
        return object()

    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.prepare_s1_reference",
        _fake_prepare,
    )
    stack._prepare_reference_scene(overwrite=True)
    assert dem_ids[0] != dem_ids[1]
    assert len(dem_ids) == 2
    west, south, east, north = stack.dem.grid.bounds
    assert west <= _USE_TIME_BOUNDS[0]
    assert south <= _USE_TIME_BOUNDS[1]
    assert east >= _USE_TIME_BOUNDS[2]
    assert north >= _USE_TIME_BOUNDS[3]
    assert id(stack._geometry_dem()) == dem_ids[1]
    assert len(grids) == n_prepare + 1


def test_produce_pair_retries_actual_callback_with_refreshed_dem(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Pair production expands once and retries with a live refreshed DEM."""
    stack, grids = _roi_bounded_stack(tmp_path, monkeypatch)
    _drop_explicit_roi(stack)
    n_prepare = len(grids)
    dem_ids: list[int] = []

    def _provider(
        _primary: object,
        _secondary: object,
        *,
        output_dir: Path,
        options: dict[str, object],
    ) -> object:
        del output_dir
        dem = options.get("dem")
        dem_ids.append(id(dem))
        _raise_if_short(dem, _USE_TIME_BOUNDS)
        return object()

    stack.scene_provider = _provider
    dates = list(stack.catalog.dates)
    stack._produce_pair(
        stack.catalog.paths_for(dates[0]),
        stack.catalog.paths_for(dates[1]),
        output_dir=tmp_path / "pair-out",
    )
    assert dem_ids[0] != dem_ids[1]
    assert len(dem_ids) == 2
    west, _south, east, _north = stack.dem.grid.bounds
    assert west <= _USE_TIME_BOUNDS[0]
    assert east >= _USE_TIME_BOUNDS[2]
    assert len(grids) == n_prepare + 1


def _instrument_managed_dem_reads(
    monkeypatch: object, managed: Path
) -> tuple[list[int], list[Path]]:
    """Count full-array loads of the canonical managed DEM file."""
    from pathlib import Path as PathType

    import rasterio

    from faninsar.processing.geometry.api import RasterDEM as LiveRaster

    cells_read: list[int] = []
    path_loads: list[PathType] = []
    real_open = rasterio.open
    real_init = LiveRaster.__init__
    managed_key = managed.resolve()

    def _open(*args: object, **kwargs: object) -> object:
        dataset = real_open(*args, **kwargs)
        opened = args[0] if args else kwargs.get("fp")
        try:
            same = PathType(str(opened)).resolve() == managed_key
        except OSError:
            same = False
        if not same:
            return dataset
        real_read = dataset.read

        def _read(*r_args: object, **r_kwargs: object) -> object:
            array = real_read(*r_args, **r_kwargs)
            cells_read.append(int(np.size(array)))
            return array

        dataset.read = _read  # type: ignore[method-assign]
        return dataset

    def _init(self: LiveRaster, *args: object, **kwargs: object) -> None:
        path = kwargs.get("path")
        if path is not None and PathType(str(path)).resolve() == managed_key:
            path_loads.append(PathType(str(path)))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", _open)
    monkeypatch.setattr(LiveRaster, "__init__", _init)
    return cells_read, path_loads


def test_repeated_ensure_reuses_live_canonical_dem_without_disk_reread(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A live sufficient RasterDEM stays in memory across production ensures."""
    from types import SimpleNamespace

    from faninsar.processing.geometry import prepare_production as prepare_mod
    from faninsar.processing.geometry.dem_view import lonlat_dem_view

    stack, grids = _roi_bounded_stack(tmp_path, monkeypatch)
    _drop_explicit_roi(stack)
    live = stack.dem
    assert isinstance(live, RasterDEM)
    managed = stack._stack_dem_raster_path()
    assert managed.is_file()
    n_materialize = len(grids)

    view = lonlat_dem_view(live)
    sentinel = object()
    prepare_mod._prepared_store(live, ("live-reuse",), sentinel)
    prepare_mod._GPU_DEM_TABLE[live] = {"cpu": {"token": sentinel}}

    cells_read, path_loads = _instrument_managed_dem_reads(monkeypatch, managed)
    first = stack._ensure_raster_dem()
    second = stack._ensure_raster_dem()
    assert first is live
    assert second is live
    assert stack.dem is live
    assert cells_read == []
    assert path_loads == []
    assert lonlat_dem_view(live) is view
    assert prepare_mod._prepared_lookup(live, ("live-reuse",)) is sentinel
    assert prepare_mod._GPU_DEM_TABLE[live]["cpu"]["token"] is sentinel
    assert len(grids) == n_materialize

    stack.config.roi = SimpleNamespace(bounds=_USE_TIME_BOUNDS, crs="EPSG:4326")
    grown = stack._ensure_raster_dem()
    assert grown is not live
    west, south, east, north = stack.dem.grid.bounds
    assert west <= _USE_TIME_BOUNDS[0]
    assert south <= _USE_TIME_BOUNDS[1]
    assert east >= _USE_TIME_BOUNDS[2]
    assert north >= _USE_TIME_BOUNDS[3]
    assert len(grids) == n_materialize + 1
    assert lonlat_dem_view(live) is not view
    assert prepare_mod._prepared_lookup(live, ("live-reuse",)) is None


def test_bind_live_dem_replacement_drops_previous_and_new_owner_state(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A true live-DEM replacement still invalidates previous and new views."""
    from faninsar.processing.geometry import prepare_production as prepare_mod
    from faninsar.processing.geometry.dem_view import lonlat_dem_view

    stack, _grids = _roi_bounded_stack(tmp_path, monkeypatch)
    previous = stack.dem
    assert isinstance(previous, RasterDEM)
    previous_view = lonlat_dem_view(previous)
    previous_token = object()
    prepare_mod._prepared_store(previous, ("live-reuse",), previous_token)

    replacement = RasterDEM(
        array=np.full(previous.grid.shape, 50.0, dtype=np.float32),
        grid=previous.grid,
        vertical_datum="ellipsoidal",
        provenance={"product": "glo30", "provider": "pc"},
    )
    replacement_view = lonlat_dem_view(replacement)
    replacement_token = object()
    prepare_mod._prepared_store(replacement, ("live-reuse",), replacement_token)

    bound = stack._bind_live_dem(replacement)
    assert bound is replacement
    assert stack.dem is replacement
    assert lonlat_dem_view(previous) is not previous_view
    assert lonlat_dem_view(replacement) is not replacement_view
    assert prepare_mod._prepared_lookup(previous, ("live-reuse",)) is None
    assert prepare_mod._prepared_lookup(replacement, ("live-reuse",)) is None
