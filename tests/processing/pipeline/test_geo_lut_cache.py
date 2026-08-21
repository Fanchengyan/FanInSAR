"""Unit tests for the shared geo2rdr LUT disk cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from faninsar.processing.pipeline.geo_lut import (
    _LUT_CACHE_ARRAY_FILES,
    _load_cached_lut,
    _lut_cache_mismatch_reason,
    _lut_cache_path,
    _materialize_work_lut_files,
    _store_lut_cache,
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
