"""Tests for the explicit production pair workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from faninsar.processing.pipeline import PairWorkflowState, run_pair_workflow
from faninsar.processing.pipeline.workflow import (
    stage_coregister,
    stage_deramp,
    stage_geocode,
    stage_interferogram,
    stage_read_scene,
    stage_unwrap,
    stage_write,
)
from faninsar.processing.tops import carrier_from_swath, deramp_reramp_roundtrip_error

SLC_ROOT = Path("/Volumes/DATA2/TEST_sentinel-1/sentinel-slc")
SCENES = sorted(SLC_ROOT.glob("S1A_IW_SLC*.zip")) if SLC_ROOT.exists() else []


@pytest.mark.skipif(len(SCENES) < 1, reason="local S1 ZIP corpus unavailable")
def test_carrier_from_annotation_and_deramp_roundtrip() -> None:
    """Annotation-built carrier supports deramp→reramp on a real window."""
    scene = stage_read_scene(SCENES[0], height=64, width=64)
    carrier = carrier_from_swath(scene.swath, scene.burst)
    error = deramp_reramp_roundtrip_error(scene.samples, carrier)
    assert error <= 1e-5


@pytest.mark.skipif(len(SCENES) < 2, reason="need two local S1 ZIP scenes")
def test_full_pair_workflow_stages(tmp_path: Path) -> None:
    """Full production workflow writes radar and geocoded products."""
    state = run_pair_workflow(
        SCENES[0],
        SCENES[1],
        output_dir=tmp_path / "pair",
        height=64,
        width=64,
        multilook=(2, 2),
        geocode_stride=4,
    )
    assert state.zarr_path is not None
    assert state.zarr_path.exists()
    assert state.stac_path is not None
    assert state.stac_path.exists()
    assert state.range_shift_px is not None
    assert state.unwrapped_phase is not None
    assert state.geocoded_unwrapped is not None
    assert "DERAMP" in " ".join(state.log)
    assert "COREG" in " ".join(state.log)
    assert "GEOCODE" in " ".join(state.log)

    root = zarr.open_group(str(state.zarr_path), mode="r")
    assert "unwrapped_phase" in root
    assert "geocoded" in root
    assert "latitude_deg" in root["geocoded"]
    assert int(np.count_nonzero(state.geocoded_unwrapped.converged)) >= 0


@pytest.mark.skipif(len(SCENES) < 2, reason="need two local S1 ZIP scenes")
def test_step_by_step_workflow(tmp_path: Path) -> None:
    """Each stage can be called explicitly in order."""
    ref = stage_read_scene(SCENES[0], height=64, width=64)
    sec = stage_read_scene(SCENES[1], height=64, width=64)
    state = PairWorkflowState(
        pair_id=f"{ref.scene_id}_{sec.scene_id}",
        reference=ref,
        secondary=sec,
    )
    state = stage_deramp(state)
    assert state.reference_deramped is not None
    state = stage_coregister(state)
    assert state.secondary_aligned is not None
    state = stage_interferogram(state, multilook=(2, 2))
    assert state.complex_ifg is not None
    state = stage_unwrap(state)
    assert state.unwrapped_phase is not None
    state = stage_geocode(state, stride=4)
    assert state.geocoded_unwrapped is not None
    state = stage_write(state, tmp_path / "steps")
    assert state.zarr_path is not None
