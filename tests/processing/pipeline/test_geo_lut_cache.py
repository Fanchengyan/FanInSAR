"""Unit tests for the shared geo2rdr LUT disk cache."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.pipeline import geo_lut
from faninsar.processing.pipeline.geo_lut import (
    _LUT_CACHE_ARRAY_FILES,
    _load_cached_lut,
    _lut_cache_mismatch_reason,
    _lut_cache_path,
    _materialize_work_lut_files,
    _store_lut_cache,
    burst_geo_footprint_lonlat,
    footprint_polygon_mask,
    geo_grid_hash,
)


class _Grid:
    """Minimal stand-in with the fields the cache identity needs."""

    def __init__(
        self,
        transform: tuple[float, ...] = (1.0, 20.0, 0.0, 5.0e6, 0.0, -8.0),
    ) -> None:
        self.crs = "EPSG:32633"
        self.transform = transform
        self.width = 64
        self.height = 32


def _write_arrays(directory: Path, crop_shape: tuple[int, int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "reference_azimuth.float64").write_bytes(b"")
    np.zeros(crop_shape).tofile(directory / "reference_azimuth.float64")
    np.zeros(crop_shape).tofile(directory / "reference_range.float64")
    np.ones(crop_shape, dtype=bool).tofile(directory / "reference_valid.bool")
    np.full(crop_shape, 900.0).tofile(directory / "height.float64")


def test_cache_key_sanitized() -> None:
    """Cache keys must be filesystem-safe."""
    path = _lut_cache_path(Path("/tmp/lut"), "S1A/Eu_T124 b3//")
    assert path is not None
    assert "/" not in str(path.relative_to(Path("/tmp/lut")))
    assert path.name.startswith("S1A_Eu_T124_b3_")


def test_store_load_round_trip(tmp_path: Path) -> None:
    """Stored LUT entries load back with matching values."""
    grid = _Grid()
    storage = tmp_path / "work" / "lut"
    cache_root = tmp_path / "cache"
    key = "scene_IW1_b0_" + geo_grid_hash(grid)
    entry = _lut_cache_path(cache_root, key)
    assert entry is not None

    crop_shape = (8, 16)
    _write_arrays(storage, crop_shape)
    _store_lut_cache(
        entry,
        storage,
        crop_shape=crop_shape,
        full_radar_shape=(1500, 20000),
        grid=grid,
        row0=100,
        col0=200,
        mean_height=912.5,
    )

    meta = json.loads((entry / "meta.json").read_text())
    assert meta["mask_algorithm"] == "8pt-hull-v1"

    assert _lut_cache_mismatch_reason(
        entry,
        crop_shape=crop_shape,
        full_radar_shape=(1500, 20000),
        grid=grid,
        row0=100,
        col0=200,
    ) is None

    loaded = _load_cached_lut(
        entry,
        crop_shape=crop_shape,
        full_radar_shape=(1500, 20000),
        grid=grid,
        row0=100,
        col0=200,
    )
    assert loaded is not None
    mean_height, arrays = loaded
    assert mean_height == pytest.approx(912.5)
    azimuth, range_index, valid, height_full = arrays
    assert azimuth.shape == crop_shape
    assert valid.dtype == np.bool_
    assert bool(valid.all())
    assert float(height_full.mean()) == pytest.approx(900.0)
    del azimuth, range_index, valid, height_full


def test_materialize_hardlinks_into_work_dir(tmp_path: Path) -> None:
    """Work-dir files appear via hardlink for downstream reopen."""
    grid = _Grid()
    storage = tmp_path / "pair" / "lut"
    cache_root = tmp_path / "cache"
    key = "k1"
    entry = _lut_cache_path(cache_root, key)
    assert entry is not None
    crop_shape = (4, 4)
    _write_arrays(storage.parent / "seed", crop_shape)
    # seed the cache from the seed dir directly
    _store_lut_cache(
        entry,
        storage.parent / "seed",
        crop_shape=crop_shape,
        full_radar_shape=(10, 10),
        grid=grid,
        row0=0,
        col0=0,
        mean_height=1.0,
    )
    meta = json.loads((entry / "meta.json").read_text())
    assert meta["mean_height"] == 1.0

    _materialize_work_lut_files(storage, entry)
    for name in _LUT_CACHE_ARRAY_FILES:
        assert (storage / name).exists()


def test_identity_mismatches_detected(tmp_path: Path) -> None:
    """Identity mismatches force a rebuild."""
    grid = _Grid()
    other_grid = _Grid(transform=(2.0, 20.0, 0.0, 5.0e6, 0.0, -8.0))
    storage = tmp_path / "seed"
    entry = _lut_cache_path(tmp_path / "cache", "k2")
    assert entry is not None
    crop_shape = (4, 8)
    _write_arrays(storage, crop_shape)
    _store_lut_cache(
        entry,
        storage,
        crop_shape=crop_shape,
        full_radar_shape=(100, 200),
        grid=grid,
        row0=5,
        col0=7,
        mean_height=10.0,
    )

    assert (
        _lut_cache_mismatch_reason(
            entry,
            crop_shape=(6, 8),
            full_radar_shape=(100, 200),
            grid=grid,
            row0=5,
            col0=7,
        )
        == "crop_shape mismatch"
    )
    assert (
        _lut_cache_mismatch_reason(
            entry,
            crop_shape=crop_shape,
            full_radar_shape=(101, 200),
            grid=grid,
            row0=5,
            col0=7,
        )
        == "full_radar_shape mismatch"
    )
    assert (
        _lut_cache_mismatch_reason(
            entry,
            crop_shape=crop_shape,
            full_radar_shape=(100, 200),
            grid=other_grid,
            row0=5,
            col0=7,
        )
        == "grid_hash mismatch"
    )


def test_mask_identity_misses_pre_change_and_opposite_flag(
    tmp_path: Path,
) -> None:
    """Mask-on and mask-off must not share a cache entry (PROPOSAL-0032 §6)."""
    grid = _Grid()
    storage = tmp_path / "seed"
    entry = _lut_cache_path(tmp_path / "cache", "k-mask")
    assert entry is not None
    crop_shape = (4, 8)
    _write_arrays(storage, crop_shape)
    _store_lut_cache(
        entry,
        storage,
        crop_shape=crop_shape,
        full_radar_shape=(100, 200),
        grid=grid,
        row0=0,
        col0=0,
        mean_height=10.0,
        footprint_applied=True,
        footprint_dilate_px=64,
    )
    assert (
        _lut_cache_mismatch_reason(
            entry,
            crop_shape=crop_shape,
            full_radar_shape=(100, 200),
            grid=grid,
            row0=0,
            col0=0,
            footprint_applied=False,
            footprint_dilate_px=64,
        )
        == "footprint_applied mismatch"
    )
    assert (
        _lut_cache_mismatch_reason(
            entry,
            crop_shape=crop_shape,
            full_radar_shape=(100, 200),
            grid=grid,
            row0=0,
            col0=0,
            footprint_applied=True,
            footprint_dilate_px=2,
        )
        == "footprint_dilate_px mismatch"
    )
    meta = json.loads((entry / "meta.json").read_text())
    del meta["footprint_applied"]
    (entry / "meta.json").write_text(json.dumps(meta))
    assert (
        _lut_cache_mismatch_reason(
            entry,
            crop_shape=crop_shape,
            full_radar_shape=(100, 200),
            grid=grid,
            row0=0,
            col0=0,
            footprint_applied=True,
            footprint_dilate_px=64,
        )
        == "footprint_applied missing"
    )


def test_missing_entry_reports_missing(tmp_path: Path) -> None:
    """Absent entries report missing rather than raising."""
    entry = tmp_path / "cache" / "nope"
    assert (
        _lut_cache_mismatch_reason(
            entry,
            crop_shape=(1, 1),
            full_radar_shape=(2, 2),
            grid=_Grid(),
            row0=0,
            col0=0,
        )
        == "missing"
    )


def _lonlat_footprint_grid() -> GeoGridSpec:
    """Return a geographic grid covering a small lon/lat rectangle."""
    return GeoGridSpec(
        crs="EPSG:4326",
        transform=(-1.0, 0.05, 0.0, 42.0, 0.0, -0.05),
        width=80,
        height=80,
        resolution_m=(0.05, 0.05),
    )


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


def test_masked_bbox_pixels_geo2rdr_outside_burst_window() -> None:
    """FINDING-db7509427bd3911f: masked lon/lat must geo2rdr outside the burst.

    Uses the toy UTM geometry from geo-mode tests so the eight-point hull is
    two-dimensional. AABB corners are not the only sample: also probe a
    masked pixel adjacent to a valid cell.
    """
    from pyproj import Transformer

    from faninsar.processing.geometry.dem import ConstantHeightDEM
    from faninsar.processing.geometry.prepare_production import run_geo2rdr, run_rdr2geo
    from faninsar.processing.pipeline.geo_lut import (
        build_geo2rdr_lut,
        burst_geo_footprint_lonlat,
        grid_lonlat,
    )
    from tests.processing.pipeline.test_geo_modes import _toy_geometry

    radar_shape = (64, 128)
    model = _toy_geometry(radar_shape)
    dem = ConstantHeightDEM(0.0)
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
            assert not _in_window(
                float(latitude[row, col]), float(longitude[row, col])
            )
    center = tuple(int(v) for v in valid_idx[len(valid_idx) // 2])
    assert _in_window(float(latitude[center]), float(longitude[center]))
