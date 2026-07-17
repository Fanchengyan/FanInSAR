"""Tests for the production S1 full-burst pair pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import zarr

from faninsar.processing.geometry import ConstantHeightDEM
from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.pipeline import (
    ProductionPairState,
    load_production_scene,
    run_production_pair,
    stage_deramp,
    stage_flatten,
    stage_interferogram,
    stage_unwrap,
    stage_write,
)
from faninsar.processing.tops.deramp import TOPSCarrierModel
from faninsar.processing.unwrap import SnaphuConfig

SLC_ROOT = Path("/Volumes/DATA2/TEST_sentinel-1/sentinel-slc")
SCENES = sorted(SLC_ROOT.glob("S1A_IW_SLC*.zip")) if SLC_ROOT.exists() else []


def _make_carrier() -> TOPSCarrierModel:
    """Return a minimal valid TOPS carrier model for testing."""
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.002,
        range_sampling_rate_hz=64.348e6,
        azimuth_time_interval_s=0.002,
        doppler_centroid_hz=(0.0, 0.0),
        doppler_t0_s=0.0,
        fm_rate_hz_s=(-2300.0, 0.0),
        fm_t0_s=0.0,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.002,
        azimuth_steering_rate_hz_s=6500.0,
    )


def _make_mock_scene(shape: tuple[int, int]) -> MagicMock:
    """Build a mock ProductionScene with the attributes stages need."""
    scene = MagicMock()
    scene.array.samples = np.ones(shape, dtype=np.complex64)
    scene.array.valid_mask = np.ones(shape, dtype=bool)
    scene.array.row0 = 0
    scene.array.col0 = 0
    scene.carrier = _make_carrier()
    scene.scene_id = "test_scene"
    scene.swath.swath = "IW1"
    scene.burst.index = 0
    return scene


def test_production_pair_state_note() -> None:
    """note() appends messages and logs them."""
    ref = _make_mock_scene((8, 8))
    sec = _make_mock_scene((8, 8))
    state = ProductionPairState(
        pair_id="TEST",
        reference=ref,
        secondary=sec,
        dem=ConstantHeightDEM(0.0),
    )
    state.note("hello")
    assert state.log == ["hello"]
    state.note("world")
    assert state.log == ["hello", "world"]


def test_stage_deramp_with_synthetic() -> None:
    """stage_deramp deramps both scenes and masks invalid samples."""
    ref = _make_mock_scene((16, 16))
    sec = _make_mock_scene((16, 16))
    state = ProductionPairState(
        pair_id="TEST",
        reference=ref,
        secondary=sec,
        dem=ConstantHeightDEM(0.0),
    )
    result = stage_deramp(state)
    assert result.reference_deramped is not None
    assert result.secondary_deramped is not None
    assert result.reference_deramped.shape == (16, 16)
    assert "DERAMP" in " ".join(result.log)


def test_stage_interferogram_with_synthetic() -> None:
    """stage_interferogram forms a multilooked interferogram from aligned arrays."""
    ref = _make_mock_scene((16, 16))
    sec = _make_mock_scene((16, 16))
    state = ProductionPairState(
        pair_id="TEST",
        reference=ref,
        secondary=sec,
        dem=ConstantHeightDEM(0.0),
    )
    state.reference_deramped = np.ones((16, 16), dtype=np.complex64)
    state.secondary_aligned = np.ones((16, 16), dtype=np.complex64)
    result = stage_interferogram(state, multilook=(2, 4), goldstein_alpha=0.0)
    assert result.complex_ifg is not None
    assert result.coherence is not None
    assert result.wrapped_phase is not None
    assert "IFG" in " ".join(result.log)


def test_stage_interferogram_masks_zero_power_edge_as_nan() -> None:
    """Shipped ifg stage must not paint zero-power looks as phase=0 black edge.

    Zero-filled full-res rows (burst margin / coreg shift) multilook to invalid
    looks that must be NaN in complex_ifg, coherence and wrapped_phase.
    """
    ref = _make_mock_scene((16, 32))
    sec = _make_mock_scene((16, 32))
    state = ProductionPairState(
        pair_id="edge",
        reference=ref,
        secondary=sec,
        dem=ConstantHeightDEM(0.0),
    )
    primary = np.ones((16, 32), dtype=np.complex64)
    secondary = np.ones((16, 32), dtype=np.complex64)
    # Last two full-res rows → last ML look row (az_looks=2) has zero power.
    primary[-2:, :] = 0
    secondary[-2:, :] = 0
    # First two range samples of secondary → first ML range look invalid.
    secondary[:, :2] = 0
    state.reference_deramped = primary
    state.secondary_aligned = secondary
    result = stage_interferogram(state, multilook=(2, 4), goldstein_alpha=0.0)
    assert result.complex_ifg is not None
    assert result.wrapped_phase is not None
    assert result.coherence is not None
    assert result.complex_ifg.shape == (8, 8)
    # Bottom ML row fully invalid → NaN, not phase 0.
    assert np.all(~np.isfinite(result.complex_ifg[-1].real))
    assert np.all(np.isnan(result.wrapped_phase[-1]))
    assert np.all(np.isnan(result.coherence[-1]))
    # Interior remains finite.
    assert np.all(np.isfinite(result.complex_ifg[2:6, 2:6].real))
    assert np.all(np.isfinite(result.wrapped_phase[2:6, 2:6]))


def test_stage_flatten_wrapped_phase_matches_flat_not_unflat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stage_flatten must set wrapped_phase from the flattened ifg (NaN-masked).

    stage_write archives complex_ifg_flat as the product complex layer; the
    matching wrapped_phase must be angle(flat), not angle(unflattened), and
    zero-power edge looks must remain NaN (not phase=0).
    """
    from faninsar.processing.pipeline import production as production_mod

    h, w = 8, 8
    ref = _make_mock_scene((16, 32))  # full-res shape for look-index scaling
    sec = _make_mock_scene((16, 32))
    state = ProductionPairState(
        pair_id="flat_edge",
        reference=ref,
        secondary=sec,
        dem=ConstantHeightDEM(0.0),
    )
    # Unflattened phase ≈ 0.5 rad; zero-power bottom row (black-edge risk).
    ifg = (np.exp(1j * 0.5) * np.ones((h, w))).astype(np.complex64)
    ifg[-1, :] = 0
    state.complex_ifg = ifg
    state.coherence = np.ones((h, w), dtype=np.float32)
    state.coherence[-1, :] = 0

    topo = np.full((h, w), 0.3, dtype=np.float64)
    monkeypatch.setattr(
        production_mod,
        "compute_topographic_phase",
        lambda *_args, **_kwargs: topo,
    )
    monkeypatch.setattr(
        production_mod,
        "estimate_residual_azimuth_ramp",
        lambda *_args, **_kwargs: 0.0,
    )

    result = stage_flatten(state)
    assert result.complex_ifg_flat is not None
    assert result.wrapped_phase is not None
    complex_ifg_flat = result.complex_ifg_flat
    wrapped_phase = result.wrapped_phase
    # Edge: invalid looks are NaN on both flat product and wrapped phase.
    assert np.all(~np.isfinite(complex_ifg_flat[-1].real))
    assert np.all(np.isnan(wrapped_phase[-1]))
    # Interior: wrapped_phase == angle(flat), not unflattened phase.
    flat_ph = np.angle(complex_ifg_flat[0, 0])
    unflat_ph = np.angle(result.complex_ifg[0, 0])
    assert np.isclose(wrapped_phase[0, 0], flat_ph, atol=1e-5)
    assert abs(float(wrapped_phase[0, 0]) - float(unflat_ph)) > 0.1
    # Topo removal shifts phase by ~0.3 rad relative to unflattened.
    assert np.isclose(float(unflat_ph - flat_ph), 0.3, atol=1e-4)


def test_staged_ifg_only_never_enters_unwrap(tmp_path: Path) -> None:
    """Composed deramp→ifg path yields finite wrapped products without unwrap.

    Mirrors the production stop-gate used by ``scripts/run_three_slc_ifg_compare.py``:
    stages are composed explicitly so ``stage_unwrap`` is never called.
    Coregistration is stubbed by supplying already-aligned secondaries so the
    unit under test remains the staged ifg stop-gate (not the geometry engine).
    """
    import zarr

    from faninsar.processing.memory import MemoryWatchdog

    shape = (32, 64)
    ref = _make_mock_scene(shape)
    sec = _make_mock_scene(shape)
    # Known phase ramp between scenes → finite non-trivial ifg after multilook
    az = np.arange(shape[0], dtype=np.float32)[:, None]
    rg = np.arange(shape[1], dtype=np.float32)[None, :]
    phase = 0.15 * az + 0.05 * rg
    ref.array.samples = np.ones(shape, dtype=np.complex64)
    sec.array.samples = np.exp(1j * phase).astype(np.complex64)

    state = ProductionPairState(
        pair_id="IFG_ONLY",
        reference=ref,
        secondary=sec,
        dem=ConstantHeightDEM(0.0),
    )
    wd = MemoryWatchdog(limit_mib=2048.0)
    state = wd.run_stage("deramp", stage_deramp, state)
    # Stub coreg: aligned secondary equals deramped secondary (identity shift)
    state.secondary_aligned = state.secondary_deramped
    state = wd.run_stage("interferogram", stage_interferogram, state, multilook=(2, 4))

    assert state.wrapped_phase is not None
    assert state.coherence is not None
    assert state.complex_ifg is not None
    assert state.unwrapped_phase is None
    assert state.connected_components is None
    assert np.isfinite(state.wrapped_phase).all()
    assert np.isfinite(state.coherence).all()
    assert state.wrapped_phase.size > 0
    assert float(np.nanmean(state.coherence)) > 0.5
    log_text = " ".join(state.log)
    assert "DERAMP" in log_text
    assert "IFG" in log_text
    assert "UNWRAP" not in log_text
    assert not wd.killed

    # Persist ifg-only layers the same way the production driver does
    zpath = tmp_path / "IFG_ONLY.zarr"
    root = zarr.open_group(str(zpath), mode="w")
    root.create_array("wrapped_phase", data=state.wrapped_phase, overwrite=True)
    root.create_array("coherence", data=state.coherence, overwrite=True)
    root.create_array("complex_ifg", data=state.complex_ifg, overwrite=True)
    root.attrs["unwrap_executed"] = "False"
    root.attrs["stop_after"] = "interferogram"
    reopened = zarr.open_group(str(zpath), mode="r")
    assert "unwrapped_phase" not in reopened
    assert reopened.attrs["unwrap_executed"] == "False"


def test_stage_unwrap_with_synthetic() -> None:
    """stage_unwrap unwraps a flattened interferogram."""
    ref = _make_mock_scene((8, 8))
    sec = _make_mock_scene((8, 8))
    state = ProductionPairState(
        pair_id="TEST",
        reference=ref,
        secondary=sec,
        dem=ConstantHeightDEM(0.0),
    )
    # Create a simple ramp phase for reliable unwrapping
    az = np.arange(8)
    rg = np.arange(8)
    phase = (az[:, None] * 0.5 + rg[None, :] * 0.3).astype(np.float32)
    state.complex_ifg_flat = np.exp(1j * phase).astype(np.complex64)
    state.coherence = np.ones((8, 8), dtype=np.float32)
    result = stage_unwrap(state, config=SnaphuConfig(nlooks=1.0))
    assert result.unwrapped_phase is not None
    assert result.connected_components is not None
    assert "method=snaphu" in " ".join(result.log)
    assert "UNWRAP" in " ".join(result.log)


def test_stage_write_with_synthetic(tmp_path: Path) -> None:
    """stage_write persists radar products and metadata to Zarr/STAC."""
    ref = _make_mock_scene((8, 8))
    sec = _make_mock_scene((8, 8))
    state = ProductionPairState(
        pair_id="TEST_PAIR",
        reference=ref,
        secondary=sec,
        dem=ConstantHeightDEM(0.0),
    )
    state.complex_ifg = np.ones((8, 8), dtype=np.complex64)
    state.complex_ifg_flat = np.ones((8, 8), dtype=np.complex64)
    state.coherence = np.ones((8, 8), dtype=np.float32)
    state.wrapped_phase = np.zeros((8, 8), dtype=np.float32)
    state.unwrapped_phase = np.zeros((8, 8), dtype=np.float32)
    state.connected_components = np.zeros((8, 8), dtype=np.int32)
    state.range_shift_px = 1.0
    state.azimuth_shift_px = 2.0

    result = stage_write(state, tmp_path)
    assert result.zarr_path is not None
    assert result.zarr_path.exists()
    assert result.stac_path is not None
    assert result.stac_path.exists()
    assert "WRITE" in " ".join(result.log)

    root = zarr.open_group(str(result.zarr_path), mode="r")
    assert "complex_ifg" in root
    assert "coherence" in root
    assert "unwrapped_phase" in root


def test_stage_write_persists_geocoded_slcs(tmp_path: Path) -> None:
    """Geo products retain both coregistered SLCs on their native grid."""
    ref = _make_mock_scene((8, 8))
    sec = _make_mock_scene((8, 8))
    state = ProductionPairState(
        pair_id="GEO_PAIR",
        reference=ref,
        secondary=sec,
        dem=ConstantHeightDEM(0.0),
        coregistration_grid="geo",
    )
    state.complex_ifg = np.ones((4, 4), dtype=np.complex64)
    state.complex_ifg_flat = np.ones((4, 4), dtype=np.complex64)
    state.coherence = np.ones((4, 4), dtype=np.float32)
    state.wrapped_phase = np.zeros((4, 4), dtype=np.float32)
    state.unwrapped_phase = np.zeros((4, 4), dtype=np.float32)
    state.connected_components = np.zeros((4, 4), dtype=np.int32)
    state.reference_geocoded_slc = np.ones((8, 8), dtype=np.complex64)
    state.secondary_geocoded_slc = np.full((8, 8), 2 + 1j, dtype=np.complex64)
    state.geocoded_slc_valid = np.ones((8, 8), dtype=bool)
    state.geo_grid = GeoGridSpec(
        crs="EPSG:32647",
        transform=(446_120.0, 10.0, 0.0, 4_133_680.0, 0.0, 40.0),
        width=8,
        height=8,
        resolution_m=(10.0, 40.0),
    )

    result = stage_write(state, tmp_path)

    assert result.zarr_path is not None
    root = zarr.open_group(str(result.zarr_path), mode="r")
    assert np.asarray(root["slc/reference"]).shape == (8, 8)
    assert np.asarray(root["slc/secondary"]).shape == (8, 8)
    assert np.asarray(root["slc/valid"]).shape == (8, 8)
    assert root["slc"].attrs["crs"] == "EPSG:32647"
    np.testing.assert_array_equal(root["slc/x"][:], 446_125.0 + 10.0 * np.arange(8))
    np.testing.assert_array_equal(root["slc/y"][:], 4_133_700.0 + 40.0 * np.arange(8))


@pytest.mark.skipif(len(SCENES) < 1, reason="local S1 ZIP corpus unavailable")
def test_load_production_scene_burst() -> None:
    """load_production_scene returns a full burst with geometry and carrier."""
    scene = load_production_scene(SCENES[0], swath="IW1", scope="burst", burst_index=0)
    assert scene.scene_id is not None
    assert scene.array.samples.ndim == 2
    assert scene.array.samples.dtype == np.complex64
    assert scene.carrier is not None
    assert scene.geometry is not None


@pytest.mark.slow
@pytest.mark.skipif(len(SCENES) < 2, reason="need two local S1 ZIP scenes")
def test_run_production_pair_full_burst(tmp_path: Path) -> None:
    """Full-burst production pair workflow writes products and logs stages."""
    state = run_production_pair(
        SCENES[0],
        SCENES[1],
        output_dir=tmp_path / "pair",
        swath="IW1",
        scope="burst",
        burst_index=0,
        multilook=(4, 20),
        esd_enabled=True,
        control_spacing=64,
    )
    assert state.zarr_path is not None
    assert state.zarr_path.exists()
    assert state.stac_path is not None
    assert state.stac_path.exists()
    assert state.range_shift_px is not None
    assert state.azimuth_shift_px is not None
    assert state.unwrapped_phase is not None
    assert state.geocoded is None
    assert state.coregistration_grid == "radar"

    log_text = " ".join(state.log)
    assert "DERAMP" in log_text
    assert "COREG" in log_text
    assert "IFG" in log_text
    assert "FLATTEN" in log_text
    assert "UNWRAP" in log_text
    assert "BASELINE" in log_text
    assert "GEO direct" not in log_text
    assert "WRITE" in log_text
    assert "DONE" in log_text

    root = zarr.open_group(str(state.zarr_path), mode="r")
    assert "complex_ifg" in root
    assert "coherence" in root
    assert "geocoded" not in root
