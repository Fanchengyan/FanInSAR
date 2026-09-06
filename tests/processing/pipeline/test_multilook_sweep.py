"""Tests for the unified Stack interferogram sweep and geo integration."""

from __future__ import annotations

import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import zarr

from faninsar.missions.s1.errors import Sentinel1ProductError
from faninsar.processing.dem import ConstantDEM
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.mosaicking.grid import GeoGridSpec
from faninsar.processing.stages import ProductionPairState
import faninsar.processing.stages as production_mod
from faninsar.processing.stages import (
    SharedPairResources,
    _is_multilook_pair,
    looks_dir,
    normalize_multilook_sweep,
    produce_interferogram_pair,
)

SHAPE = (8, 16)
WAVELENGTH_M = 0.05546576

SLC_ROOT_RAW = Path("/Volumes/DATA2/TEST_sentinel-1/Raw Data/sentinel-slc")
SCENES = sorted(SLC_ROOT_RAW.glob("S1A_IW_SLC*.zip")) if SLC_ROOT_RAW.exists() else []


def _make_scene(shape: tuple[int, int], scene_id: str) -> MagicMock:
    """Build a mock scene with the attributes the production stages read."""
    scene = MagicMock()
    scene.array.samples = np.ones(shape, dtype=np.complex64)
    scene.array.valid_mask = np.ones(shape, dtype=bool)
    scene.array.row0 = 0
    scene.array.col0 = 0
    scene.scene_id = scene_id
    scene.swath.swath = "IW1"
    scene.burst.index = 0
    scene.burst.azimuth_time = datetime(2016, 12, 7, tzinfo=UTC)
    scene.geometry.wavelength_m = WAVELENGTH_M
    return scene


def _origin_state(shape: tuple[int, int] = SHAPE) -> ProductionPairState:
    """Build a minimal post-coreg origin state for sweep finalization."""
    return ProductionPairState(
        pair_id="20161207_20161231_pair",
        primary=_make_scene(shape, "20161207"),
        secondary=_make_scene(shape, "20161231"),
        dem=ConstantDEM(0.0),
        coregistration_grid="radar",
        coreg_executor="torch",
        coreg_device="cpu",
        unwrap_method="snaphu",
    )


def _write_unit(
    work: Path,
    tag: str,
    ifg: np.ndarray,
    pri: np.ndarray,
    sec: np.ndarray,
    *,
    height: np.ndarray | None = None,
    azimuth_offset: int = 0,
    burst_row0: int = 0,
    burst_col0: int = 0,
    row0: int = 0,
    col0: int = 0,
    mode: str = "radar",
) -> dict[str, object]:
    """Persist one archived IFG unit and return its unit record."""
    base = work / tag
    ifg_path = str(base) + ".complex64"
    pri_path = str(base) + ".pri.f64"
    sec_path = str(base) + ".sec.f64"
    ifg.astype(np.complex64, copy=False).tofile(ifg_path)
    pri.astype(np.float64, copy=False).tofile(pri_path)
    sec.astype(np.float64, copy=False).tofile(sec_path)
    unit: dict[str, object] = {
        "tag": tag,
        "swath": "IW1",
        "frame_index": 0,
        "ifg_path": ifg_path,
        "pri_path": pri_path,
        "sec_path": sec_path,
        "rows": int(ifg.shape[0]),
        "cols": int(ifg.shape[1]),
    }
    if mode == "geo":
        height_path = str(base) + ".height.f64"
        assert height is not None
        height.astype(np.float64, copy=False).tofile(height_path)
        unit.update(
            {
                "height_path": height_path,
                "row0": row0,
                "col0": col0,
                "mode": "geo",
            }
        )
    else:
        unit.update(
            {
                "azimuth_offset": azimuth_offset,
                "burst_row0": burst_row0,
                "burst_col0": burst_col0,
            }
        )
    return unit


def _radar_archive(work: Path) -> dict[str, object]:
    """Build a single-unit radar archive with a deterministic phase ramp."""
    ifg_dir = work / "ifgs"
    ifg_dir.mkdir(parents=True, exist_ok=True)
    azimuth = np.arange(SHAPE[0], dtype=np.float32)[:, None]
    rg = np.arange(SHAPE[1], dtype=np.float32)[None, :]
    phase = 0.15 * azimuth + 0.05 * rg
    ifg = (25.0 * np.exp(1j * phase)).astype(np.complex64)
    pri = np.full(SHAPE, 25.0, dtype=np.float64)
    sec = np.full(SHAPE, 25.0, dtype=np.float64)
    unit = _write_unit(ifg_dir, "f0_IW1_b0", ifg, pri, sec)
    return {
        "units": [unit],
        "origin_state": _origin_state(),
        "per_burst_timings": {unit["tag"]: {"deramp": 0.1, "coregister": 0.2}},
        "prefix_started": time.perf_counter(),
        "grid_mode": "radar",
        "geo_grid": None,
    }


def _geo_archive(work: Path) -> dict[str, object]:
    """Build a two-unit geo archive on one shared 32x16 grid."""
    ifg_dir = work / "ifgs"
    ifg_dir.mkdir(parents=True, exist_ok=True)
    azimuth = np.arange(8, dtype=np.float32)[:, None]
    rg = np.arange(16, dtype=np.float32)[None, :]
    phase = 0.1 * azimuth + 0.05 * rg
    ifg_a = (25.0 * np.exp(1j * phase)).astype(np.complex64)
    ifg_b = (25.0 * np.exp(1j * (phase + 0.3))).astype(np.complex64)
    pri = np.full((8, 16), 25.0, dtype=np.float64)
    sec = np.full((8, 16), 25.0, dtype=np.float64)
    height = np.full((8, 16), 42.0, dtype=np.float64)
    unit_a = _write_unit(
        ifg_dir,
        "f0_IW1_b0",
        ifg_a,
        pri,
        sec,
        height=height,
        row0=0,
        col0=0,
        mode="geo",
    )
    unit_b = _write_unit(
        ifg_dir,
        "f0_IW1_b1",
        ifg_b,
        pri,
        sec,
        height=height,
        row0=0,
        col0=16,
        mode="geo",
    )
    return {
        "units": [unit_a, unit_b],
        "origin_state": _origin_state(),
        "per_burst_timings": {},
        "prefix_started": time.perf_counter(),
        "grid_mode": "geo",
        "geo_grid": GeoGridSpec(
            crs="EPSG:32647",
            transform=(446_120.0, 10.0, 0.0, 4_133_680.0, 0.0, 10.0),
            width=32,
            height=8,
            resolution_m=(10.0, 10.0),
        ),
    }


def test_normalize_multilook_sweep_accepts_valid_configs() -> None:
    """Valid pairs normalize and preserve input order."""
    assert normalize_multilook_sweep([(2, 4), (1, 1)]) == [(2, 4), (1, 1)]
    assert normalize_multilook_sweep([[2, 4]]) == [(2, 4)]


def test_normalize_multilook_sweep_rejects_invalid() -> None:
    """Empty, duplicate, bool/float/str, and malformed configs are rejected."""
    with pytest.raises(TypeError):
        normalize_multilook_sweep("1x1")
    with pytest.raises(TypeError):
        normalize_multilook_sweep(42)
    with pytest.raises(ValueError, match="at least one"):
        normalize_multilook_sweep([])
    with pytest.raises(ValueError, match="pair"):
        normalize_multilook_sweep([(2,)])
    with pytest.raises(TypeError, match="pair"):
        normalize_multilook_sweep([2, 4])
    with pytest.raises(TypeError, match="integers"):
        normalize_multilook_sweep([(True, 4)])
    with pytest.raises(TypeError, match="integers"):
        normalize_multilook_sweep([(2.0, 4)])
    with pytest.raises(TypeError, match="integers"):
        normalize_multilook_sweep([("2", 4)])
    with pytest.raises(ValueError, match=">= 1"):
        normalize_multilook_sweep([(0, 4)])
    with pytest.raises(ValueError, match="duplicate"):
        normalize_multilook_sweep([(2, 4), (2, 4)])


def test_is_multilook_pair_dispatch_semantics() -> None:
    """Single pairs and iterable-of-pairs route to different paths."""
    assert _is_multilook_pair((2, 4))
    assert _is_multilook_pair([2, 4])
    assert not _is_multilook_pair([(2, 4)])
    assert not _is_multilook_pair((2, 4, 1))
    assert not _is_multilook_pair((2.0, 4))
    assert not _is_multilook_pair((True, 4))
    assert not _is_multilook_pair("24")


def test_looks_dir_layout() -> None:
    """Per-config subtree names are deterministic."""
    assert looks_dir(2, 4) == "looks_2x4"
    assert looks_dir(1, 1) == "looks_1x1"


def test_merge_burst_ifgs_accumulates_and_multilooks(tmp_path: Path) -> None:
    """Radar merge replays archived IFGs into the multilooked frame."""
    archive = _radar_archive(tmp_path)
    merged = production_mod._merge_burst_ifgs(
        archive,
        swath_tuple=("IW1",),
        range_offsets={"IW1": 0},
        burst_lines={"IW1": SHAPE[0]},
        burst_width={"IW1": SHAPE[1]},
        frame_rows=SHAPE[0],
        frame_cols=SHAPE[1],
        az_looks=2,
        rg_looks=4,
        geo_grid=None,
    )
    assert merged["merged_ifg"].shape == (SHAPE[0] // 2, SHAPE[1] // 4)
    assert merged["wrapped"].shape == merged["merged_ifg"].shape
    assert not merged["invalid"].any()
    assert float(np.nanmean(merged["coherence"])) > 0.99


def test_finalize_sweep_config_writes_subtree_and_manifest(tmp_path: Path) -> None:
    """Sweep finalize writes per-config zarr, STAC, and run.json."""
    archive = _radar_archive(tmp_path / "prefix")
    merged = production_mod._merge_burst_ifgs(
        archive,
        swath_tuple=("IW1",),
        range_offsets={"IW1": 0},
        burst_lines={"IW1": SHAPE[0]},
        burst_width={"IW1": SHAPE[1]},
        frame_rows=SHAPE[0],
        frame_cols=SHAPE[1],
        az_looks=2,
        rg_looks=4,
        geo_grid=None,
    )
    out = tmp_path / "out"
    outcome, state = production_mod._finalize_sweep_config(
        merged,
        config=(2, 4),
        pair_id="20161207_20161231_pair",
        output_root=out,
        goldstein_alpha=0.0,
        snaphu_config=None,
        unwrap_method=None,
        irls_kwargs=None,
        unwrap=False,
        geo_height_m=0.0,
    )
    subtree = out / looks_dir(2, 4)
    assert outcome.zarr_path == subtree / "20161207_20161231_pair.zarr"
    assert outcome.zarr_path.is_dir()
    item = json.loads(outcome.stac_path.read_text(encoding="utf-8"))
    assert item["id"] == "20161207_20161231_pair__l2x4"
    manifest = json.loads((subtree / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["looks"] == [2, 4]
    root = zarr.open_group(str(outcome.zarr_path), mode="r")
    assert tuple(np.asarray(root["wrapped_phase"]).shape) == (4, 4)
    assert outcome.metadata["multilook"] == [2, 4]
    assert outcome.metadata["multilook_sweep"] == [[2, 4]]
    assert outcome.metadata["wavelength_m"] == pytest.approx(WAVELENGTH_M)
    assert state.zarr_path == outcome.zarr_path
    assert state.stac_path == outcome.stac_path
    assert outcome.shape == (4, 4)


def test_sweep_preflight_refuses_existing_output(tmp_path: Path) -> None:
    """Existing looks_* subtrees are refused unless overwrite=True."""
    (tmp_path / looks_dir(2, 4)).mkdir(parents=True)
    with pytest.raises(InvalidProcessingStateError, match="already exist"):
        production_mod._produce_interferogram_sweep(
            "reference.SAFE",
            "secondary.SAFE",
            output_dir=tmp_path,
            roi=None,
            swaths=("IW1",),
            bursts=None,
            dem=None,
            multilook=[(2, 4)],
            overwrite=False,
            goldstein_alpha=0.0,
            dead_pixel_amp_threshold=3.0,
            esd_enabled=False,
            amplitude_refinement_enabled=False,
            control_spacing=None,
            executor="torch",
            device="cpu",
            dask_client=None,
            coregistration_grid="radar",
            geo_grid=None,
            geo_height_m=0.0,
            geo_chunk_size=128,
            geo_work_dir=None,
            snaphu_config=None,
            unwrap_method=None,
            irls_kwargs=None,
            primary_orbit_path=None,
            secondary_orbit_path=None,
            unwrap=False,
            geoid_correction=True,
        )


def test_geo_merge_and_finalize_produce_geocoded_product(tmp_path: Path) -> None:
    """Geo sweep finalize emits the geocoded product grid and bbox STAC."""
    archive = _geo_archive(tmp_path / "prefix")
    geo_grid = archive["geo_grid"]
    merged = production_mod._merge_burst_ifgs(
        archive,
        swath_tuple=("IW1",),
        range_offsets={"IW1": 0},
        burst_lines={"IW1": SHAPE[0]},
        burst_width={"IW1": SHAPE[1]},
        frame_rows=SHAPE[0],
        frame_cols=SHAPE[1],
        az_looks=2,
        rg_looks=4,
        geo_grid=geo_grid,
    )
    assert merged["merged_ifg"].shape == (4, 8)
    assert merged["height_field"].shape == (4, 8)
    np.testing.assert_allclose(merged["height_field"], 42.0)
    out = tmp_path / "out"
    outcome, state = production_mod._finalize_sweep_config(
        merged,
        config=(2, 4),
        pair_id="20161207_20161231_pair",
        output_root=out,
        goldstein_alpha=0.0,
        snaphu_config=None,
        unwrap_method=None,
        irls_kwargs=None,
        unwrap=False,
        geo_height_m=42.0,
    )
    root = zarr.open_group(str(outcome.zarr_path), mode="r")
    assert "geocoded" in root
    geo = root["geocoded"]
    for name in (
        "unwrapped_phase",
        "coherence",
        "wrapped_phase",
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "converged",
    ):
        assert name in geo
        assert tuple(np.asarray(geo[name]).shape) == (4, 8)
    item = json.loads(outcome.stac_path.read_text(encoding="utf-8"))
    assert item["geometry"] is not None
    assert item["bbox"] is not None
    manifest = json.loads(
        (out / looks_dir(2, 4) / "run.json").read_text(encoding="utf-8")
    )
    assert manifest["grid"]["crs"] == "EPSG:32647"
    assert manifest["grid"]["transform"] is not None
    assert outcome.metadata["multilook_sweep"] == [[2, 4]]
    assert outcome.metadata["wavelength_m"] == pytest.approx(WAVELENGTH_M)
    assert state.zarr_path == outcome.zarr_path
    assert state.stac_path == outcome.stac_path


def test_finalize_geo_products_builds_multilooked_grid() -> None:
    """Geo products assemble on the pixel-centre-aligned multilooked grid."""
    grid = GeoGridSpec(
        crs="EPSG:32647",
        transform=(446_120.0, 10.0, 0.0, 4_133_680.0, 0.0, 10.0),
        width=32,
        height=16,
        resolution_m=(10.0, 10.0),
    )
    shape = (16, 32)
    product_shape = (8, 8)
    state = ProductionPairState(
        pair_id="GEO_ML",
        primary=_make_scene(shape, "20161207"),
        secondary=_make_scene(shape, "20161231"),
        dem=ConstantDEM(0.0),
        coregistration_grid="geo",
        unwrapped_phase=np.full(product_shape, 0.25, dtype=np.float32),
        coherence=np.ones(product_shape, dtype=np.float32),
        wrapped_phase=np.zeros(product_shape, dtype=np.float32),
        geo_height_field=np.full(shape, 42.0, dtype=np.float64),
    )
    result = production_mod._finalize_geo_products(
        state,
        geo_grid=grid,
        multilook=(2, 4),
        geo_height_m=0.0,
    )
    assert result.geo_grid_meta is not None
    assert result.geo_grid_meta["width"] == 8
    assert result.geo_grid_meta["height"] == 8
    assert result.geocoded is not None
    assert result.geocoded["unwrapped_phase"].shape == product_shape
    np.testing.assert_allclose(result.geocoded["height_m"], 42.0)


def test_shared_resources_cleanup_is_idempotent(tmp_path: Path) -> None:
    """cleanup() removes the geo work dir once and is safe to call twice."""
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT

    class CountingTemporaryDirectory(tempfile.TemporaryDirectory):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.cleanup_calls = 0

        def cleanup(self) -> None:
            self.cleanup_calls += 1
            super().cleanup()

    owner = CountingTemporaryDirectory(dir=tmp_path)
    work = Path(owner.name)
    (work / "slc").mkdir()
    (work / "lut").mkdir()
    shape = (4, 4)

    def _memmap(
        relative: str, dtype: object, array_shape: tuple[int, int]
    ) -> np.memmap:
        path = work / relative
        array = np.memmap(path, mode="w+", dtype=dtype, shape=array_shape)
        array[:] = 0
        return array

    reference_slc = _memmap("slc/reference_geo.complex64", np.complex64, shape)
    secondary_slc = _memmap("slc/secondary_geo.complex64", np.complex64, shape)
    valid_slc = _memmap("slc/geo_valid.bool", np.bool_, shape)
    topo_phase = _memmap("topographic_phase.float32", np.float32, shape)
    height_full = _memmap("lut/height.float64", np.float64, shape)
    state = ProductionPairState(
        pair_id="GEO_CLEANUP",
        primary=_make_scene(shape, "20161207"),
        secondary=_make_scene(shape, "20161231"),
        dem=ConstantDEM(0.0),
        coregistration_grid="geo",
        primary_geocoded_slc=reference_slc,
        secondary_geocoded_slc=secondary_slc,
        geocoded_slc_valid=valid_slc,
        topo_phase=topo_phase,
        geo_height_field=height_full,
        primary_deramped=reference_slc,
        secondary_aligned=secondary_slc,
        geo_work_dir=work,
        geo_grid=GeoGridSpec(
            crs="EPSG:4326",
            transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            width=4,
            height=4,
            resolution_m=(1.0, 1.0),
        ),
        geo2rdr_lut=Geo2RdrLUT(
            az_full=_memmap("lut/reference_azimuth.float64", np.float64, shape),
            rg_full=_memmap("lut/reference_range.float64", np.float64, shape),
            valid=_memmap("lut/reference_valid.bool", np.bool_, shape),
            full_radar_shape=shape,
            height_m=0.0,
            height_full=height_full,
        ),
    )
    resources = SharedPairResources(prefix_state=state, temporary_directory=owner)
    assert work.is_dir()
    resources.cleanup()
    assert not work.exists()
    assert owner.cleanup_calls == 1
    assert resources.temporary_directory is None
    assert state.geo2rdr_lut is None
    assert state.primary_geocoded_slc is None
    resources.cleanup()
    assert owner.cleanup_calls == 1


@pytest.mark.slow
@pytest.mark.skipif(len(SCENES) < 2, reason="need two local S1 ZIP scenes")
def test_sweep_matches_single_config_run_array_for_array(tmp_path: Path) -> None:
    """A one-config radar sweep is array-identical to the single-config run."""
    from faninsar.missions.s1.safe import open_safe_product
    from faninsar.processing.stages import _common_burst_indices

    reference, secondary = SCENES[0], SCENES[1]
    ref_swath = open_safe_product(reference).swath("IW1")
    sec_swath = open_safe_product(secondary).swath("IW1")
    if not _common_burst_indices(ref_swath, sec_swath):
        pytest.skip("no common IW1 burst between the first two scenes")
    single = produce_interferogram_pair(
        reference,
        secondary,
        output_dir=tmp_path / "single",
        swaths=("IW1",),
        bursts={"IW1": [0]},
        multilook=(2, 10),
        goldstein_alpha=0.0,
        unwrap=False,
    )
    sweep = produce_interferogram_pair(
        reference,
        secondary,
        output_dir=tmp_path / "sweep",
        swaths=("IW1",),
        bursts={"IW1": [0]},
        multilook=[(2, 10)],
        goldstein_alpha=0.0,
        unwrap=False,
    )
    outcome = sweep.per_config[(2, 10)]
    single_root = zarr.open_group(str(single.zarr_path), mode="r")
    sweep_root = zarr.open_group(str(outcome.zarr_path), mode="r")
    for layer in ("unwrapped_phase", "coherence", "wrapped_phase", "complex_ifg"):
        np.testing.assert_allclose(
            np.asarray(single_root[layer]),
            np.asarray(sweep_root[layer]),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


def test_snaphu_config_nlooks_overridden_per_config_radar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit SnaphuConfig nlooks is replaced by az * rg in radar finalize."""
    from faninsar.processing.unwrapping import SnaphuConfig

    archive = _radar_archive(tmp_path / "prefix")
    merged = production_mod._merge_burst_ifgs(
        archive,
        swath_tuple=("IW1",),
        range_offsets={"IW1": 0},
        burst_lines={"IW1": SHAPE[0]},
        burst_width={"IW1": SHAPE[1]},
        frame_rows=SHAPE[0],
        frame_cols=SHAPE[1],
        az_looks=2,
        rg_looks=4,
        geo_grid=None,
    )
    captured: dict[str, object] = {}

    def _fake_stage_unwrap(
        state: ProductionPairState,
        *,
        method: object,
        config: SnaphuConfig,
        irls_kwargs: object,
    ) -> ProductionPairState:
        del method, irls_kwargs
        captured["nlooks"] = config.nlooks
        state.unwrapped_phase = np.zeros_like(state.wrapped_phase, dtype=np.float32)
        state.connected_components = np.zeros_like(state.wrapped_phase, dtype=np.uint8)
        return state

    monkeypatch.setattr(production_mod, "stage_unwrap", _fake_stage_unwrap)
    production_mod._finalize_sweep_config(
        merged,
        config=(2, 4),
        pair_id="20161207_20161231_pair",
        output_root=tmp_path / "out",
        goldstein_alpha=0.0,
        snaphu_config=SnaphuConfig(nlooks=99.0),
        unwrap_method="snaphu",
        irls_kwargs=None,
        unwrap=True,
        geo_height_m=0.0,
    )
    assert captured["nlooks"] == 8.0


def test_snaphu_config_nlooks_overridden_per_config_geo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit SnaphuConfig nlooks is replaced by az * rg in geo finalize."""
    from faninsar.processing.unwrapping import SnaphuConfig

    archive = _geo_archive(tmp_path / "prefix")
    merged = production_mod._merge_burst_ifgs(
        archive,
        swath_tuple=("IW1",),
        range_offsets={"IW1": 0},
        burst_lines={"IW1": SHAPE[0]},
        burst_width={"IW1": SHAPE[1]},
        frame_rows=SHAPE[0],
        frame_cols=SHAPE[1],
        az_looks=2,
        rg_looks=4,
        geo_grid=archive["geo_grid"],
    )
    captured: dict[str, object] = {}

    def _fake_stage_unwrap(
        state: ProductionPairState,
        *,
        method: object,
        config: SnaphuConfig,
        irls_kwargs: object,
    ) -> ProductionPairState:
        del method, irls_kwargs
        captured["nlooks"] = config.nlooks
        state.unwrapped_phase = np.zeros_like(state.wrapped_phase, dtype=np.float32)
        state.connected_components = np.zeros_like(state.wrapped_phase, dtype=np.uint8)
        return state

    monkeypatch.setattr(production_mod, "stage_unwrap", _fake_stage_unwrap)
    production_mod._finalize_sweep_config(
        merged,
        config=(2, 4),
        pair_id="20161207_20161231_pair",
        output_root=tmp_path / "out",
        goldstein_alpha=0.0,
        snaphu_config=SnaphuConfig(nlooks=99.0),
        unwrap_method="snaphu",
        irls_kwargs=None,
        unwrap=True,
        geo_height_m=42.0,
    )
    assert captured["nlooks"] == 8.0


def test_overwrite_true_removes_stale_subtree(tmp_path: Path) -> None:
    """overwrite=True deletes existing config subtrees before processing."""
    subtree = tmp_path / looks_dir(2, 4)
    subtree.mkdir()
    (subtree / "stale.bin").write_bytes(b"stale")
    with pytest.raises(Sentinel1ProductError):
        production_mod._produce_interferogram_sweep(
            "reference.SAFE",
            "secondary.SAFE",
            output_dir=tmp_path,
            roi=None,
            swaths=("IW1",),
            bursts=None,
            dem=None,
            multilook=[(2, 4)],
            overwrite=True,
            goldstein_alpha=0.0,
            dead_pixel_amp_threshold=3.0,
            esd_enabled=False,
            amplitude_refinement_enabled=False,
            control_spacing=None,
            executor="torch",
            device="cpu",
            dask_client=None,
            coregistration_grid="radar",
            geo_grid=None,
            geo_height_m=0.0,
            geo_chunk_size=128,
            geo_work_dir=None,
            snaphu_config=None,
            unwrap_method=None,
            irls_kwargs=None,
            primary_orbit_path=None,
            secondary_orbit_path=None,
            unwrap=False,
            geoid_correction=True,
        )
    assert not subtree.exists()


def test_geo_sweep_wires_prefix_state_and_closes_memmaps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Geo sweeps hand the last unit state to SharedPairResources cleanup."""
    from faninsar.missions.s1 import safe as safe_module
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT

    work = tmp_path / "geo-memmaps"
    work.mkdir()
    shape = (4, 4)

    def _memmap(name: str, dtype: object) -> np.memmap:
        array = np.memmap(work / name, mode="w+", dtype=dtype, shape=shape)
        array[:] = 0
        return array

    lut = Geo2RdrLUT(
        az_full=_memmap("az.f64", np.float64),
        rg_full=_memmap("rg.f64", np.float64),
        valid=_memmap("valid.bool", np.bool_),
        full_radar_shape=shape,
        height_m=0.0,
        height_full=_memmap("height.f64", np.float64),
    )
    geo_state = ProductionPairState(
        pair_id="GEO_WIRE",
        primary=_make_scene(shape, "20161207"),
        secondary=_make_scene(shape, "20161231"),
        dem=ConstantDEM(0.0),
        coregistration_grid="geo",
        primary_geocoded_slc=_memmap("ref.c64", np.complex64),
        secondary_geocoded_slc=_memmap("sec.c64", np.complex64),
        geocoded_slc_valid=_memmap("slc_valid.bool", np.bool_),
        topo_phase=_memmap("topo.f32", np.float32),
        geo_height_field=_memmap("geo_h.f64", np.float64),
        primary_deramped=_memmap("deramp.c64", np.complex64),
        secondary_aligned=_memmap("aligned.c64", np.complex64),
        geo_work_dir=work,
        geo2rdr_lut=lut,
    )
    geo_grid = GeoGridSpec(
        crs="EPSG:32647",
        transform=(446_120.0, 10.0, 0.0, 4_133_680.0, 0.0, 10.0),
        width=16,
        height=8,
        resolution_m=(10.0, 10.0),
    )
    archive = {
        "units": [],
        "origin_state": _origin_state(),
        "per_burst_timings": {},
        "prefix_started": time.perf_counter(),
        "grid_mode": "geo",
        "geo_grid": geo_grid,
        "geo_prefix_state": geo_state,
    }
    burst = MagicMock()
    burst.azimuth_time = datetime(2016, 12, 7, tzinfo=UTC)
    swath = MagicMock()
    swath.swath = "IW1"
    swath.bursts = [burst]
    swath.lines_per_burst = SHAPE[0]
    swath.samples_per_burst = SHAPE[1]
    swath.azimuth_time_interval_s = 1.0
    swath.slant_range_time_s = 0.0
    swath.range_sampling_rate_hz = 1.0
    product = MagicMock()
    product.swaths = [swath]
    product.swath.return_value = swath
    monkeypatch.setattr(safe_module, "open_safe_product", lambda _path: product)
    monkeypatch.setattr(
        production_mod, "_archive_burst_ifgs", lambda *_args, **_kwargs: archive
    )
    result = production_mod._produce_interferogram_sweep(
        "reference.SAFE",
        "secondary.SAFE",
        output_dir=tmp_path / "out",
        roi=None,
        swaths=("IW1",),
        bursts=None,
        dem=ConstantDEM(0.0),
        multilook=[(1, 1)],
        overwrite=False,
        goldstein_alpha=0.0,
        dead_pixel_amp_threshold=3.0,
        esd_enabled=False,
        amplitude_refinement_enabled=False,
        control_spacing=None,
        executor="torch",
        device="cpu",
        dask_client=None,
        coregistration_grid="geo",
        geo_grid=geo_grid,
        geo_height_m=0.0,
        geo_chunk_size=128,
        geo_work_dir=None,
        snaphu_config=None,
        unwrap_method=None,
        irls_kwargs=None,
        primary_orbit_path=None,
        secondary_orbit_path=None,
        unwrap=False,
        geoid_correction=True,
    )
    outcome = result.per_config[(1, 1)]
    assert outcome.metadata["multilook_sweep"] == [[1, 1]]
    assert outcome.metadata["wavelength_m"] == pytest.approx(WAVELENGTH_M)
    assert geo_state.geo2rdr_lut is None
    assert geo_state.primary_geocoded_slc is None
    assert geo_state.geocoded_slc_valid is None
    assert geo_state.geo_height_field is None
    assert lut.az_full._mmap.closed
    single = produce_interferogram_pair(
        "reference.SAFE",
        "secondary.SAFE",
        output_dir=tmp_path / "single-out",
        roi=None,
        swaths=("IW1",),
        bursts=None,
        dem=ConstantDEM(0.0),
        multilook=(1, 1),
        overwrite=False,
        goldstein_alpha=0.0,
        dead_pixel_amp_threshold=3.0,
        esd_enabled=False,
        amplitude_refinement_enabled=False,
        control_spacing=None,
        executor="torch",
        device="cpu",
        coregistration_grid="geo",
        geo_grid=geo_grid,
        geo_height_m=0.0,
        geo_chunk_size=128,
        geo_work_dir=None,
        snaphu_config=None,
        unwrap_method=None,
        irls_kwargs=None,
        primary_orbit_path=None,
        secondary_orbit_path=None,
        unwrap=False,
        geoid_correction=True,
    )
    assert isinstance(single, ProductionPairState)
    assert single.zarr_path is not None
    assert single.stac_path is not None
    assert single.coregistration_grid == "geo"
    assert single.multilook == (1, 1)
