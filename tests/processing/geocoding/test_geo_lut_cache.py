"""Unit tests for the content-addressed geo2rdr LUT disk cache.

The cache identity is derived inside ``build_geo2rdr_lut`` from geometry,
orbit, DEM, grid, solver settings, radar shape, and hull-prefilter content;
callers only pass a readable namespace. These tests exercise the cache
through the public builder with synthetic geometry and pin the regression
findings from the read-only ``tmp/geocode-wf`` experiments (false hits on
DEM/orbit/ROI changes, store truncation, corrupt metadata).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.processing.geocoding import geo_lut
from faninsar.processing.geocoding.geo_lut import (
    build_geo2rdr_lut,
    burst_geo_footprint_lonlat,
    footprint_polygon_mask,
)
from faninsar.processing.geometry import ConstantDEM, RadarGeometryModel, RasterDEM
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.prepare_production import run_geo2rdr, run_rdr2geo
from faninsar.processing.mosaicking.grid import GeoGridSpec

if TYPE_CHECKING:
    from pathlib import Path

RADAR_SHAPE = (64, 128)
UTM = "EPSG:32631"


def _toy_geometry(
    shape: tuple[int, int] = RADAR_SHAPE,
    orbit_shift_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RadarGeometryModel:
    """Return a synthetic right-looking geometry with a straight orbit."""
    epoch = datetime(2020, 1, 1, tzinfo=UTC)
    sx, sy, sz = orbit_shift_m
    vectors = tuple(
        OrbitStateVector(
            time=epoch + timedelta(seconds=float(i)),
            position_m=(7000_000.0 + sx, 0.0 + sy, 0.0 + 100.0 * i + sz),
            velocity_m_s=(0.0, 7500.0, 0.0),
        )
        for i in range(40)
    )
    orbit = OrbitMetadata(reference_frame="ECFT", source="test", vectors=vectors)
    rgrid = RadarGrid(
        shape=shape,
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=epoch,
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(rgrid, orbit)


def _scene_grid(
    model: RadarGeometryModel,
    dx: float = 10.0,
    margin: float = 600.0,
) -> GeoGridSpec:
    """Return a UTM grid covering the toy burst footprint plus margin."""
    from pyproj import Transformer

    az = np.array([0.0, 0.0, 63.0, 63.0, 0.0, 63.0, 31.5, 31.5], dtype=np.float64)
    rg = np.array([0.0, 127.0, 0.0, 127.0, 63.5, 63.5, 0.0, 127.0], dtype=np.float64)
    geo = run_rdr2geo(model, az, rg, ConstantDEM(0.0), device="cpu")
    ok = np.isfinite(geo.latitude_deg) & np.isfinite(geo.longitude_deg)
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    ux, uy = to_utm.transform(geo.longitude_deg[ok], geo.latitude_deg[ok])
    west, east = float(np.min(ux)) - margin, float(np.max(ux)) + margin
    south, north = float(np.min(uy)) - margin, float(np.max(uy)) + margin
    return GeoGridSpec(
        crs=UTM,
        transform=(west, dx, 0.0, north, 0.0, -dx),
        width=int(np.ceil((east - west) / dx)),
        height=int(np.ceil((north - south) / dx)),
        resolution_m=(dx, dx),
    )


def _polygon_lonlat(
    grid: GeoGridSpec,
    corners_rc: list[tuple[float, float]],
) -> object:
    """Return a lon/lat shapely polygon from grid ``(row, col)`` vertices."""
    from affine import Affine
    from pyproj import Transformer
    from shapely.geometry import Polygon

    affine = Affine(*grid.transform)
    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    xy = [affine * (c + 0.5, r + 0.5) for r, c in corners_rc]
    lon, lat = to_ll.transform([p[0] for p in xy], [p[1] for p in xy])
    return Polygon(list(zip(lon, lat, strict=True)))


def _build_lut(
    model: RadarGeometryModel,
    grid: GeoGridSpec,
    *,
    cache_dir: Path | None = None,
    cache_key: str = "scene_IW1_b0",
    storage_dir: Path | None = None,
    **overrides: object,
) -> geo_lut.Geo2RdrLUT:
    """Call :func:`build_geo2rdr_lut` with the shared synthetic defaults."""
    options: dict[str, object] = {
        "full_radar_shape": RADAR_SHAPE,
        "height_m": 0.0,
        "dem": ConstantDEM(0.0),
        "chunk_size": 64,
        **overrides,
    }
    return build_geo2rdr_lut(
        geometry=model,
        grid=grid,
        device="cpu",
        storage_dir=storage_dir,
        cache_dir=cache_dir,
        cache_key=None if cache_dir is None else cache_key,
        **options,  # type: ignore[arg-type]
    )


@pytest.fixture
def solve_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count real geo2rdr solves; a cache hit performs zero of them."""
    calls = {"n": 0}
    original = geo_lut.run_geo2rdr

    def _counted(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(geo_lut, "run_geo2rdr", _counted)
    return calls


def _assert_lut_equal(
    cached: geo_lut.Geo2RdrLUT,
    fresh: geo_lut.Geo2RdrLUT,
) -> None:
    """Assert bitwise array equality (NaN positions included)."""
    for name in ("az_full", "rg_full", "valid", "height_full"):
        np.testing.assert_array_equal(
            np.asarray(getattr(cached, name)),
            np.asarray(getattr(fresh, name)),
            err_msg=name,
        )
    assert cached.height_m == fresh.height_m


def test_second_identical_build_hits_cache_and_matches_fresh(
    tmp_path: Path,
    solve_calls: dict[str, int],
) -> None:
    """A rebuilt LUT with identical inputs hits the cache bit-for-bit."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    cache_dir = tmp_path / "cache"

    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w1")
    assert solve_calls["n"] > 0
    solve_calls["n"] = 0
    cached = _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w2")
    assert solve_calls["n"] == 0, "identical rebuild must not re-solve"

    solve_calls["n"] = 0
    fresh = _build_lut(model, grid)
    solve_calls["n"] = 0
    _assert_lut_equal(cached, fresh)


def test_dem_content_change_misses(tmp_path: Path, solve_calls: dict[str, int]) -> None:
    """A different DEM height or raster content must miss and match a fresh solve."""
    from faninsar.processing.geometry import GridSpec

    model = _toy_geometry()
    grid = _scene_grid(model)
    cache_dir = tmp_path / "cache"
    window = {"row_range": (4, grid.height // 2), "col_range": (4, grid.width // 2)}
    raster_grid = GridSpec(
        crs=grid.crs,
        transform=grid.transform,
        height=grid.height,
        width=grid.width,
    )

    _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w1",
        dem=ConstantDEM(0.0),
        **window,
    )
    solve_calls["n"] = 0
    cached = _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w2",
        dem=ConstantDEM(40.0),
        **window,
    )
    assert solve_calls["n"] > 0, "different constant height must miss"
    solve_calls["n"] = 0
    fresh = _build_lut(model, grid, dem=ConstantDEM(40.0), **window)
    _assert_lut_equal(cached, fresh)

    values_a = np.full(raster_grid.shape, 0.0, dtype=np.float32)
    values_b = np.full(raster_grid.shape, 40.0, dtype=np.float32)
    dem_a = RasterDEM(values_a, raster_grid)
    dem_b = RasterDEM(values_b, raster_grid)
    assert dem_a.identity != dem_b.identity
    _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w3",
        dem=dem_a,
        **window,
    )
    solve_calls["n"] = 0
    cached = _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w4",
        dem=dem_b,
        **window,
    )
    assert solve_calls["n"] > 0, "different raster DEM content must miss"
    solve_calls["n"] = 0
    fresh = _build_lut(model, grid, dem=dem_b, **window)
    _assert_lut_equal(cached, fresh)


def test_orbit_change_misses(tmp_path: Path, solve_calls: dict[str, int]) -> None:
    """A shifted orbit (same scene namespace) must miss and match a fresh solve."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    shifted = _toy_geometry(orbit_shift_m=(0.0, 0.0, 50.0))
    cache_dir = tmp_path / "cache"

    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w1")
    solve_calls["n"] = 0
    cached = _build_lut(shifted, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w2")
    assert solve_calls["n"] > 0, "orbit shift must miss"
    fresh = _build_lut(shifted, grid)
    _assert_lut_equal(cached, fresh)


def test_solver_settings_and_radar_shape_enter_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    solve_calls: dict[str, int],
) -> None:
    """Changed pinned solver settings or full radar shape must miss."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    cache_dir = tmp_path / "cache"

    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w1")
    patched = {**geo_lut._LUT_GEO2RDR_SOLVER, "max_iter": 2}
    monkeypatch.setattr(geo_lut, "_LUT_GEO2RDR_SOLVER", patched)
    solve_calls["n"] = 0
    cached = _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w2")
    assert solve_calls["n"] > 0, "changed solver settings must miss"
    fresh = _build_lut(model, grid)
    _assert_lut_equal(cached, fresh)

    monkeypatch.setattr(
        geo_lut, "_LUT_GEO2RDR_SOLVER", dict(geo_lut._LUT_GEO2RDR_SOLVER)
    )
    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w3")
    solve_calls["n"] = 0
    smaller = _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w4",
        full_radar_shape=(60, 128),
    )
    assert solve_calls["n"] > 0, "changed full_radar_shape must miss"
    fresh = _build_lut(model, grid, full_radar_shape=(60, 128))
    _assert_lut_equal(smaller, fresh)


@pytest.mark.parametrize(
    ("field", "value"),
    [("range_tol_m", 0.5), ("doppler_tol_hz", 50.0), ("extra_iter", 1)],
)
def test_pinned_solver_fields_enter_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    solve_calls: dict[str, int],
    field: str,
    value: float,
) -> None:
    """Every pinned solver field enters the identity and reaches the solver.

    The recorded settings and the values handed to ``run_geo2rdr`` come from
    the same pinned mapping, so a changed field must miss and the variant
    must match a fresh solve built under the same patched field.
    """
    model = _toy_geometry()
    grid = _scene_grid(model, dx=25.0, margin=150.0)
    cache_dir = tmp_path / "cache"

    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "base")
    patched = {**geo_lut._LUT_GEO2RDR_SOLVER, field: value}
    monkeypatch.setattr(geo_lut, "_LUT_GEO2RDR_SOLVER", patched)
    solve_calls["n"] = 0
    cached = _build_lut(
        model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "variant"
    )
    assert solve_calls["n"] > 0, f"changed {field} must miss the base entry"
    fresh = _build_lut(model, grid)
    _assert_lut_equal(cached, fresh)


def test_fallback_height_change_misses(
    tmp_path: Path,
    solve_calls: dict[str, int],
) -> None:
    """A changed scalar or per-window fallback height array must miss."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    cache_dir = tmp_path / "cache"

    _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w1",
        height_m=0.0,
    )
    solve_calls["n"] = 0
    cached = _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w2",
        height_m=30.0,
    )
    assert solve_calls["n"] > 0, "changed scalar fallback height must miss"
    fresh = _build_lut(model, grid, height_m=30.0)
    _assert_lut_equal(cached, fresh)

    window = {"row_range": (2, grid.height - 2), "col_range": (2, grid.width - 2)}
    shape = (
        window["row_range"][1] - window["row_range"][0],
        window["col_range"][1] - window["col_range"][0],
    )
    _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w3",
        height_m=np.full(shape, 10.0),
        **window,
    )
    solve_calls["n"] = 0
    cached = _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w4",
        height_m=np.full(shape, 70.0),
        **window,
    )
    assert solve_calls["n"] > 0, "changed fallback height array must miss"
    fresh = _build_lut(model, grid, height_m=np.full(shape, 70.0), **window)
    _assert_lut_equal(cached, fresh)


def test_roi_and_dilate_changes_hit_and_match_fresh(
    tmp_path: Path,
    solve_calls: dict[str, int],
) -> None:
    """ROI polygon/dilation are request masks: they hit and match fresh solves."""
    model = _toy_geometry()
    grid = _scene_grid(model, dx=10.0, margin=200.0)
    cache_dir = tmp_path / "cache"
    rect = [
        (4, 4),
        (4, grid.width - 5),
        (grid.height - 5, grid.width - 5),
        (grid.height - 5, 4),
    ]
    half = [(4, 4), (grid.height - 5, 4), (4, grid.width - 5)]
    rect_lonlat = _polygon_lonlat(grid, rect)
    half_lonlat = _polygon_lonlat(grid, half)

    _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w1",
        roi_geometry=rect_lonlat,
        polygon_dilate_px=2,
    )
    solve_calls["n"] = 0
    cached = _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w2",
        roi_geometry=half_lonlat,
        polygon_dilate_px=2,
    )
    assert solve_calls["n"] == 0, "ROI polygon change must hit the same entry"
    fresh = _build_lut(model, grid, roi_geometry=half_lonlat, polygon_dilate_px=2)
    _assert_lut_equal(cached, fresh)
    solve_calls["n"] = 0
    cached = _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        storage_dir=tmp_path / "w3",
        roi_geometry=half_lonlat,
        polygon_dilate_px=64,
    )
    assert solve_calls["n"] == 0, "ROI dilation change must hit the same entry"
    fresh = _build_lut(model, grid, roi_geometry=half_lonlat, polygon_dilate_px=64)
    _assert_lut_equal(cached, fresh)
    assert int(np.asarray(cached.valid).sum()) > int(
        np.asarray(
            _build_lut(model, grid, roi_geometry=half_lonlat, polygon_dilate_px=2).valid
        ).sum()
    )


def test_request_masks_never_change_solved_values(tmp_path: Path) -> None:
    """ROI and footprint-hull masks only hide solved pixels, never move them.

    Masked builds return a validity subset of the unmasked solve and keep
    the az/rg/height values bitwise identical on the surviving pixels.
    """
    model = _toy_geometry()
    grid = _scene_grid(model, dx=10.0, margin=200.0)
    base = _build_lut(model, grid)
    base_valid = np.asarray(base.valid, dtype=bool)

    ring = burst_geo_footprint_lonlat(
        model, RADAR_SHAPE, ConstantDEM(0.0), grid, device="cpu"
    )
    assert ring is not None
    hull = _build_lut(model, grid, footprint_lonlat=ring, footprint_dilate_px=4)
    triangle = _polygon_lonlat(
        grid,
        [(2, 2), (2, grid.width - 3), (grid.height - 3, grid.width - 3)],
    )
    masked = _build_lut(model, grid, roi_geometry=triangle, polygon_dilate_px=2)
    assert int(np.asarray(masked.valid).sum()) < int(base_valid.sum())

    for variant in (hull, masked):
        valid = np.asarray(variant.valid, dtype=bool)
        assert not bool((valid & ~base_valid).any()), (
            "a request mask may only remove pixels from the unmasked solve"
        )
        for name in ("az_full", "rg_full", "height_full"):
            np.testing.assert_array_equal(
                np.asarray(getattr(variant, name))[valid],
                np.asarray(getattr(base, name))[valid],
                err_msg=name,
            )


def test_contained_window_hits_and_slices_bitwise(
    tmp_path: Path,
    solve_calls: dict[str, int],
) -> None:
    """A request inside a stored window is served by slicing, bit-for-bit."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    cache_dir = tmp_path / "cache"
    big = {
        "row_range": (grid.height // 8, 7 * grid.height // 8),
        "col_range": (grid.width // 8, 7 * grid.width // 8),
    }
    small = {
        "row_range": (grid.height // 4, grid.height // 2),
        "col_range": (grid.width // 4, grid.width // 2),
    }

    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w1", **big)
    solve_calls["n"] = 0
    cached = _build_lut(
        model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w2", **small
    )
    assert solve_calls["n"] == 0, "contained window must slice the stored entry"
    fresh = _build_lut(model, grid, **small)
    _assert_lut_equal(cached, fresh)


def test_diagonal_hull_subwindow_slice_matches_fresh_build(
    tmp_path: Path,
    solve_calls: dict[str, int],
) -> None:
    """A hull-masked entry slices bitwise even where the hull cuts a corner.

    The window sits on the hypotenuse of a diagonal footprint hull: the
    nearest inside pixels of some window pixels lie outside the window, so
    the pre-fix window-local dilation truncated the halo and the sliced
    entry disagreed with a fresh small-window build (verified: this exact
    window differed at 1 pixel for ``footprint_dilate_px=1``).
    """
    model = _toy_geometry()
    grid = _scene_grid(model, dx=10.0, margin=200.0)
    ring = _grid_px_lonlat_ring(
        grid,
        [
            (0.0, 0.0),
            (float(grid.height) - 4.5, 0.0),
            (0.0, float(grid.width) - 4.5),
        ],
    )
    # Window straddling the hypotenuse inside the solved (converged) band.
    # Found by exhaustive scan over the synthetic geometry: the pre-fix
    # window-local dilation disagreed with the full-grid mask at 132 pixels
    # of this window for ``footprint_dilate_px=1``.
    window = {
        "row_range": (18, min(30, int(grid.height))),
        "col_range": (1112, min(1124, int(grid.width))),
    }
    hull = {"footprint_lonlat": ring, "footprint_dilate_px": 1}
    _build_lut(
        model, grid, cache_dir=tmp_path / "cache", storage_dir=tmp_path / "w1", **hull
    )
    solve_calls["n"] = 0
    sliced = _build_lut(
        model,
        grid,
        cache_dir=tmp_path / "cache",
        storage_dir=tmp_path / "w2",
        **window,
        **hull,
    )
    assert solve_calls["n"] == 0, "contained window must slice the stored entry"
    fresh = _build_lut(model, grid, **window, **hull)
    window_valid = np.asarray(fresh.valid, dtype=bool)
    assert window_valid.any(), "the scenario must keep valid pixels"
    assert not window_valid.all(), "the scenario must keep hull-masked pixels"
    _assert_lut_equal(sliced, fresh)


def test_windowed_solve_matches_full_solve_slice(tmp_path: Path) -> None:
    """A cropped solve is bitwise the corresponding block of the full solve."""
    model = _toy_geometry()
    grid = _scene_grid(model, dx=25.0, margin=150.0)
    full = _build_lut(model, grid)
    row_range, col_range = (2, grid.height - 3), (3, grid.width - 4)
    cropped = _build_lut(model, grid, row_range=row_range, col_range=col_range)

    assert (cropped.row0, cropped.col0) == (row_range[0], col_range[0])
    assert cropped.shape == (
        row_range[1] - row_range[0],
        col_range[1] - col_range[0],
    )
    for name in ("az_full", "rg_full", "valid", "height_full"):
        np.testing.assert_array_equal(
            np.asarray(getattr(cropped, name)),
            np.asarray(getattr(full, name))[
                row_range[0] : row_range[1], col_range[0] : col_range[1]
            ],
            err_msg=name,
        )


def test_chunk_size_does_not_change_solved_values() -> None:
    """The solve is pointwise: chunking never changes any solved value."""
    model = _toy_geometry()
    grid = _scene_grid(model, dx=25.0, margin=150.0)
    base = _build_lut(model, grid, chunk_size=7)
    for chunk in (13, 1 << 20):
        _assert_lut_equal(_build_lut(model, grid, chunk_size=chunk), base)


def test_disjoint_windows_publish_separate_entries(
    tmp_path: Path,
    solve_calls: dict[str, int],
) -> None:
    """Two windows under one key must not overwrite each other's entry."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    cache_dir = tmp_path / "cache"
    left = {"row_range": (0, grid.height), "col_range": (0, grid.width // 2)}
    right = {"row_range": (0, grid.height), "col_range": (grid.width // 2, grid.width)}

    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w1", **left)
    solve_calls["n"] = 0
    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w2", **right)
    assert solve_calls["n"] > 0, "a disjoint window is not contained and must rebuild"
    entries = sorted(path for path in cache_dir.rglob("meta.json"))
    assert len(entries) == 2, "each window owns its own entry directory"

    solve_calls["n"] = 0
    cached_left = _build_lut(
        model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w3", **left
    )
    cached_right = _build_lut(
        model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w4", **right
    )
    assert solve_calls["n"] == 0, "both windows must hit their own entry afterwards"
    _assert_lut_equal(cached_left, _build_lut(model, grid, **left))
    _assert_lut_equal(cached_right, _build_lut(model, grid, **right))


def _digest_entry(cache_dir: Path) -> dict[Path, str]:
    """Return md5 per cached array file for truncation checks."""
    return {
        path: hashlib.md5(path.read_bytes()).hexdigest()
        for path in sorted(cache_dir.rglob("*.float64"))
    } | {
        path: hashlib.md5(path.read_bytes()).hexdigest()
        for path in sorted(cache_dir.rglob("*.bool"))
    }


def test_two_builds_do_not_truncate_cache(
    tmp_path: Path,
    solve_calls: dict[str, int],
) -> None:
    """Repeated builds with separate work dirs leave the cache intact."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    cache_dir = tmp_path / "cache"
    storage_a = tmp_path / "work" / "a"
    storage_b = tmp_path / "work" / "b"

    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=storage_a)
    before = _digest_entry(cache_dir)
    assert before

    solve_calls["n"] = 0
    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=storage_b)
    assert solve_calls["n"] == 0
    solve_calls["n"] = 0
    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=storage_a)
    assert solve_calls["n"] == 0

    assert _digest_entry(cache_dir) == before, "cache bytes must never be rewritten"
    for path in sorted(cache_dir.rglob("*.float64")):
        assert path.stat().st_nlink == 1, "cache must not be hardlinked into work dirs"
    for name in (
        "reference_azimuth.float64",
        "reference_range.float64",
        "reference_valid.bool",
        "height.float64",
    ):
        storage_a_inode = (storage_a / name).stat().st_ino
        storage_b_inode = (storage_b / name).stat().st_ino
        assert storage_a_inode != storage_b_inode
    solve_calls["n"] = 0
    fresh = _build_lut(model, grid)
    loaded = _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w9")
    _assert_lut_equal(loaded, fresh)


def test_publish_generations_are_reader_safe_and_clean_up(tmp_path: Path) -> None:
    """Publishing never rewrites arrays a reader may still hold.

    Each publish allocates a fresh generation directory and only then
    switches ``meta.json``; superseded generations are removed best-effort
    one publish later, while a reader holding the previous generation's
    mmaps keeps reading its original bytes.
    """
    entry = tmp_path / "entry" / "w0-0-4-4"
    shape = (4, 4)
    meta = {"schema": geo_lut._LUT_CACHE_SCHEMA, "key_digest": "a" * 64}

    def arrays(value: float) -> dict[str, np.ndarray]:
        return {
            "reference_azimuth.float64": np.full(shape, value),
            "reference_range.float64": np.full(shape, value * 2.0),
            "reference_valid.bool": np.ones(shape, dtype=bool),
            "height.float64": np.full(shape, value * 3.0),
        }

    geo_lut._publish_lut_entry(entry, arrays(1.0), dict(meta))
    published = json.loads((entry / "meta.json").read_text())
    assert published["generation"] == 1

    reader = {
        name: np.memmap(
            geo_lut._generation_dir(entry, 1) / name,
            mode="r",
            dtype=geo_lut._LUT_CACHE_ARRAY_DTYPES[name],
            shape=shape,
        )
        for name in geo_lut._LUT_CACHE_ARRAY_FILES
    }
    original = {name: np.array(view) for name, view in reader.items()}

    geo_lut._publish_lut_entry(entry, arrays(2.0), dict(meta))
    published = json.loads((entry / "meta.json").read_text())
    assert published["generation"] == 2
    assert geo_lut._generation_dir(entry, 2).is_dir()
    for name, view in reader.items():
        np.testing.assert_array_equal(np.asarray(view), original[name])

    # The superseded generation is cleaned up; the held mmaps stay valid.
    geo_lut._publish_lut_entry(entry, arrays(3.0), dict(meta))
    published = json.loads((entry / "meta.json").read_text())
    assert published["generation"] == 3
    assert not geo_lut._generation_dir(entry, 1).exists()
    assert geo_lut._generation_dir(entry, 2).is_dir()
    assert geo_lut._generation_dir(entry, 3).is_dir()
    for name, view in reader.items():
        np.testing.assert_array_equal(np.asarray(view), original[name])

    reused = np.fromfile(
        geo_lut._generation_dir(entry, 3) / "reference_azimuth.float64",
        dtype=np.float64,
    )
    np.testing.assert_array_equal(reused.reshape(shape), np.full(shape, 3.0))


def test_deleted_generation_files_are_miss_and_republish(
    tmp_path: Path,
    solve_calls: dict[str, int],
) -> None:
    """A meta whose generation files are gone is a miss that republishes."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    cache_dir = tmp_path / "cache"
    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w1")
    meta_path = next(cache_dir.rglob("meta.json"))
    entry = meta_path.parent
    published = json.loads(meta_path.read_text())
    shutil.rmtree(geo_lut._generation_dir(entry, int(published["generation"])))

    solve_calls["n"] = 0
    rebuilt = _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w2")
    assert solve_calls["n"] > 0, "missing generation files must be a miss"
    republished = json.loads(meta_path.read_text())
    assert republished["generation"] > published["generation"], (
        "the republished generation must be strictly newer"
    )
    assert geo_lut._generation_dir(entry, int(republished["generation"])).is_dir()
    fresh = _build_lut(model, grid)
    solve_calls["n"] = 0
    reloaded = _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w3")
    assert solve_calls["n"] == 0, "the republished entry must hit again"
    _assert_lut_equal(rebuilt, fresh)
    _assert_lut_equal(reloaded, fresh)


def test_corrupt_meta_is_miss_and_self_heals(
    tmp_path: Path,
    solve_calls: dict[str, int],
) -> None:
    """Truncated JSON, tampered digests, and bad schema are misses that heal."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    cache_dir = tmp_path / "cache"
    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w1")
    meta_path = next(cache_dir.rglob("meta.json"))

    meta_path.write_text('{"schema": "faninsar.geo2rdr-lut')
    solve_calls["n"] = 0
    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w2")
    assert solve_calls["n"] > 0, "truncated meta JSON must be a miss, never a crash"

    meta = json.loads(meta_path.read_text())
    meta["key_digest"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    solve_calls["n"] = 0
    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w3")
    assert solve_calls["n"] > 0, "tampered key digest must be a miss"

    meta = json.loads(meta_path.read_text())
    assert meta["key_digest"] != "0" * 64, "republished meta must carry the real digest"
    meta["identity"]["fallback_height"]["height_m"] = 999.0
    meta_path.write_text(json.dumps(meta))
    solve_calls["n"] = 0
    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w5")
    assert solve_calls["n"] > 0, "tampered stored identity must be a miss"

    solve_calls["n"] = 0
    _build_lut(model, grid, cache_dir=cache_dir, storage_dir=tmp_path / "w4")
    assert solve_calls["n"] == 0, "self-healed entry must hit again"


def test_cache_namespace_is_sanitized(tmp_path: Path) -> None:
    """Caller namespaces with separators still map to one safe directory."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    cache_dir = tmp_path / "cache"
    _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        cache_key="S1A/Eu_T124 b3//",
        storage_dir=tmp_path / "w1",
    )
    _build_lut(
        model,
        grid,
        cache_dir=cache_dir,
        cache_key="S1A/Eu_T124 b3//",
        storage_dir=tmp_path / "w2",
    )
    assert len(list(cache_dir.iterdir())) == 1


def test_dem_content_identity_rejects_unsupported_types() -> None:
    """DEM types without a content rule are rejected, not name-mangled."""
    from faninsar.processing.errors import InvalidProcessingStateError
    from faninsar.processing.geometry import dem_content_identity

    assert dem_content_identity(None) == "none"
    assert dem_content_identity(ConstantDEM(12.5)).startswith("constant:")
    with pytest.raises(InvalidProcessingStateError):
        dem_content_identity(object())


def test_dem_content_identity_tracks_content_and_never_drifts() -> None:
    """Content identities separate DEMs by content and stay fixed on use.

    Different constant heights and different raster contents must not share
    a cache identity, two rasters built from the same array must, and the
    identity of a raster must not change after it has been sampled (the
    retired digests drifted once the lazy backend materialized).
    """
    from faninsar.processing.geometry import GridSpec, dem_content_identity

    assert dem_content_identity(ConstantDEM(0.0)) != dem_content_identity(
        ConstantDEM(100.0)
    )

    raster_grid = GridSpec(
        crs="EPSG:4326",
        transform=(-1.0, 0.05, 0.0, 42.0, 0.0, -0.05),
        height=8,
        width=8,
        resolution_m=(0.05, 0.05),
    )
    values = np.linspace(0.0, 400.0, 64, dtype=np.float32).reshape(8, 8)
    first = RasterDEM(values, raster_grid)
    assert first.identity == RasterDEM(values, raster_grid).identity
    assert RasterDEM(values + 25.0, raster_grid).identity != first.identity

    before = first.identity
    sampled = first.sample(np.array([[41.825]]), np.array([[-0.825]]))
    assert np.isfinite(sampled).all()
    assert first.identity == before


def _lonlat_footprint_grid() -> GeoGridSpec:
    """Return a geographic grid covering a small lon/lat rectangle."""
    return GeoGridSpec(
        crs="EPSG:4326",
        transform=(-1.0, 0.05, 0.0, 42.0, 0.0, -0.05),
        width=80,
        height=80,
        resolution_m=(0.05, 0.05),
    )


def _grid_px_lonlat_ring(
    grid: GeoGridSpec,
    corners_rc: list[tuple[float, float]],
) -> np.ndarray:
    """Return an ``(N, 2)`` lon/lat ring for grid ``(row, col)`` vertices."""
    from affine import Affine
    from pyproj import Transformer

    affine = Affine(*grid.transform)
    to_lonlat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    xy = [affine * (col + 0.5, row + 0.5) for row, col in corners_rc]
    lon, lat = to_lonlat.transform(
        [point[0] for point in xy], [point[1] for point in xy]
    )
    return np.column_stack([np.asarray(lon), np.asarray(lat)])


def test_burst_geo_footprint_all_nan_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-NaN rdr2geo samples yield no burst footprint."""

    def _all_nan_rdr2geo(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            latitude_deg=np.full(8, np.nan),
            longitude_deg=np.full(8, np.nan),
        )

    monkeypatch.setattr(geo_lut, "run_rdr2geo", _all_nan_rdr2geo)
    result = burst_geo_footprint_lonlat(
        object(),  # type: ignore[arg-type]
        (12, 20),
        None,
        _lonlat_footprint_grid(),
        device="cpu",
    )
    assert result is None


def test_burst_geo_footprint_convex_hull_has_vertices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eight finite rdr2geo lon/lat samples form a hull with at least three vertices."""

    def _finite_rdr2geo(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            latitude_deg=np.array(
                [40.0, 40.0, 41.0, 41.0, 40.0, 41.0, 40.5, 40.5],
                dtype=np.float64,
            ),
            longitude_deg=np.array(
                [0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0],
                dtype=np.float64,
            ),
        )

    monkeypatch.setattr(geo_lut, "run_rdr2geo", _finite_rdr2geo)
    result = burst_geo_footprint_lonlat(
        object(),  # type: ignore[arg-type]
        (12, 20),
        None,
        _lonlat_footprint_grid(),
        device="cpu",
    )
    assert result is not None
    assert result.ndim == 2
    assert result.shape[1] == 2
    assert result.shape[0] >= 3
    assert np.isfinite(result).all()


def test_footprint_mask_keeps_hull_interior_unmasked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pixels well inside the eight-point hull stay unmasked after dilation."""

    def _finite_rdr2geo(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            latitude_deg=np.array(
                [40.2, 40.2, 41.8, 41.8, 40.2, 41.8, 41.0, 41.0],
                dtype=np.float64,
            ),
            longitude_deg=np.array(
                [0.2, 1.8, 0.2, 1.8, 1.0, 1.0, 0.2, 1.8],
                dtype=np.float64,
            ),
        )

    monkeypatch.setattr(geo_lut, "run_rdr2geo", _finite_rdr2geo)
    grid = _lonlat_footprint_grid()
    ring = burst_geo_footprint_lonlat(
        object(),  # type: ignore[arg-type]
        (12, 20),
        None,
        grid,
        device="cpu",
    )
    assert ring is not None
    undilated = footprint_polygon_mask(
        grid,
        0,
        grid.height,
        0,
        grid.width,
        ring,
        dilate_px=0,
    )
    dilated = footprint_polygon_mask(
        grid,
        0,
        grid.height,
        0,
        grid.width,
        ring,
        dilate_px=64,
    )
    # Pixel-area: lon=1, lat=41 → col≈39.5, row≈19.5 on this grid.
    hull_row, hull_col = 20, 40
    assert bool(undilated[hull_row, hull_col])
    assert bool(dilated[hull_row, hull_col])
    assert not bool(undilated[0, 0])
    assert dilated[undilated].all()


def test_footprint_mask_is_window_independent_at_diagonal_hull_edge() -> None:
    """A hull mask for a window equals the slice of any larger-window mask.

    The dilation radius must reach across a request-window edge: a diagonal
    hull clipping the window corner has its nearest inside pixels partly
    outside the window, so the pre-fix window-local distance transform
    truncated the halo (this exact scenario differs at 100 pixels for
    ``dilate_px=1`` under the pre-fix implementation).
    """
    grid = _lonlat_footprint_grid()
    ring = _grid_px_lonlat_ring(
        grid,
        [(0.0, 0.0), (38.5, 0.0), (0.0, 38.5)],
    )  # Triangle hugging the top/left grid edges; the hypotenuse is r+c=38.5.
    for dilate_px in (0, 1, 16):
        big = footprint_polygon_mask(
            grid,
            0,
            grid.height,
            0,
            grid.width,
            ring,
            dilate_px=dilate_px,
        )
        for window in ((30, 40, 2, 12), (28, 40, 24, 36), (0, 10, 30, 40)):
            row0, row1, col0, col1 = window
            small = footprint_polygon_mask(
                grid,
                row0,
                row1,
                col0,
                col1,
                ring,
                dilate_px=dilate_px,
            )
            np.testing.assert_array_equal(
                small,
                big[row0:row1, col0:col1],
                err_msg=f"dilate_px={dilate_px} window={window}",
            )
    corner = footprint_polygon_mask(grid, 30, 40, 2, 12, ring, dilate_px=1)
    assert bool(corner.any()), "the scenario must keep inside pixels"
    assert not bool(corner.all()), "the scenario must keep outside pixels"


def test_masked_bbox_pixels_geo2rdr_outside_burst_window() -> None:
    """FINDING-db7509427bd3911f: masked lon/lat must geo2rdr outside the burst.

    Uses the toy UTM geometry from geo-mode tests so the eight-point hull is
    two-dimensional. AABB corners are not the only sample: also probe a
    masked pixel adjacent to a valid cell.
    """
    from pyproj import Transformer

    from faninsar.processing.geocoding.geo_lut import grid_lonlat
    from faninsar.processing.geometry import ConstantDEM

    radar_shape = (64, 128)
    model = _toy_geometry(radar_shape)
    dem = ConstantDEM(0.0)
    az = np.array([0.0, 0.0, 63.0, 63.0, 0.0, 63.0, 31.5, 31.5], dtype=np.float64)
    rg = np.array([0.0, 127.0, 0.0, 127.0, 63.5, 63.5, 0.0, 127.0], dtype=np.float64)
    geo = run_rdr2geo(model, az, rg, dem, device="cpu")
    ok = np.isfinite(geo.latitude_deg) & np.isfinite(geo.longitude_deg)
    if int(ok.sum()) < 3:
        pytest.fail("toy orbit did not map burst corners")
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    ux, uy = to_utm.transform(geo.longitude_deg[ok], geo.latitude_deg[ok])
    west, east = float(np.min(ux)) - 400.0, float(np.max(ux)) + 400.0
    south, north = float(np.min(uy)) - 400.0, float(np.max(uy)) + 400.0
    dx = 50.0
    width = int(np.ceil((east - west) / dx))
    height = int(np.ceil((north - south) / dx))
    grid = GeoGridSpec(
        crs="EPSG:32631",
        transform=(west, dx, 0.0, north, 0.0, -dx),
        width=width,
        height=height,
        resolution_m=(dx, dx),
    )
    ring = burst_geo_footprint_lonlat(model, radar_shape, dem, grid, device="cpu")
    assert ring is not None
    lut = build_geo2rdr_lut(
        geometry=model,
        grid=grid,
        full_radar_shape=radar_shape,
        height_m=0.0,
        dem=dem,
        device="cpu",
        footprint_lonlat=ring,
        footprint_dilate_px=64,
        chunk_size=32,
    )
    latitude, longitude = grid_lonlat(grid)
    assert bool(lut.valid.any())
    masked = ~lut.valid
    assert bool(masked.any())

    def _in_window(lat: float, lon: float) -> bool:
        result = run_geo2rdr(
            model,
            np.array([[lat]]),
            np.array([[lon]]),
            np.array([[0.0]]),
            device="cpu",
        )
        az_i = float(result.azimuth_index[0, 0])
        rg_i = float(result.range_index[0, 0])
        return bool(
            result.converged[0, 0]
            and np.isfinite(az_i)
            and np.isfinite(rg_i)
            and 0.0 <= az_i <= radar_shape[0] - 1.0
            and 0.0 <= rg_i <= radar_shape[1] - 1.0
        )

    neighbor = None
    valid_idx = np.argwhere(lut.valid)
    for row, col in valid_idx:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(row) + dr, int(col) + dc
            in_bounds = 0 <= rr < lut.valid.shape[0] and 0 <= cc < lut.valid.shape[1]
            if in_bounds and not lut.valid[rr, cc]:
                neighbor = (rr, cc)
                break
        if neighbor is not None:
            break
    probes = [(0, 0), (lut.valid.shape[0] - 1, lut.valid.shape[1] - 1)]
    if neighbor is not None:
        probes.append(neighbor)
    for row, col in probes:
        if not lut.valid[row, col]:
            assert not _in_window(float(latitude[row, col]), float(longitude[row, col]))
    center = tuple(int(v) for v in valid_idx[len(valid_idx) // 2])
    assert _in_window(float(latitude[center]), float(longitude[center]))
