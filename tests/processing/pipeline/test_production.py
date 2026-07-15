"""Tests for the production S1 full-burst pair pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import zarr

from faninsar.processing.geometry import ConstantHeightDEM
from faninsar.processing.pipeline import (
    ProductionPairState,
    load_production_scene,
    run_production_pair,
    stage_deramp,
    stage_interferogram,
    stage_unwrap,
    stage_write,
)
from faninsar.processing.tops.deramp import TOPSCarrierModel

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
        fm_rate_hz_s=(0.0, 0.0),
        fm_t0_s=0.0,
        burst_sensing_time_s=0.0,
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
    result = stage_interferogram(state, multilook=(2, 4))
    assert result.complex_ifg is not None
    assert result.coherence is not None
    assert result.wrapped_phase is not None
    assert "IFG" in " ".join(result.log)


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
    result = stage_unwrap(state, method="irls")
    assert result.unwrapped_phase is not None
    assert result.connected_components is not None
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
    assert state.geocoded is not None

    log_text = " ".join(state.log)
    assert "DERAMP" in log_text
    assert "COREG" in log_text
    assert "IFG" in log_text
    assert "FLATTEN" in log_text
    assert "UNWRAP" in log_text
    assert "BASELINE" in log_text
    assert "GEOCODE" in log_text
    assert "WRITE" in log_text
    assert "DONE" in log_text

    root = zarr.open_group(str(state.zarr_path), mode="r")
    assert "complex_ifg" in root
    assert "coherence" in root
    assert "geocoded" in root
    assert "latitude_deg" in root["geocoded"]
    assert "longitude_deg" in root["geocoded"]
