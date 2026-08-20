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
    state = stage_geocode(state, stride=4, device="cpu")
    assert state.geocoded_unwrapped is not None
    state = stage_write(state, tmp_path / "steps")
    assert state.zarr_path is not None


def test_deprecated_workflow_keeps_numpy_ampcor_rollback_with_torch_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The portable Ampcor rollback does not disable phase-preserving remapping."""
    from faninsar.processing.pipeline import workflow as workflow_mod

    shape = (8, 12)
    reference = type("Scene", (), {})()
    secondary = type("Scene", (), {})()
    reference.swath = object()
    secondary.swath = object()
    reference.burst = object()
    secondary.burst = object()
    reference.carrier = object()
    secondary.carrier = object()
    state = workflow_mod.PairWorkflowState(
        pair_id="rollback",
        reference=reference,
        secondary=secondary,
    )
    state.reference_deramped = np.ones(shape, dtype=np.complex64)
    state.secondary_deramped = np.ones(shape, dtype=np.complex64)
    ampcor_kwargs: dict[str, object] = {}
    resample_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        workflow_mod,
        "geometry_coarse_shift",
        lambda *_args, **_kwargs: (0.0, 0.0),
    )

    def fake_refine(*_args: object, **kwargs: object) -> tuple[float, float]:
        ampcor_kwargs.update(kwargs)
        return 0.0, 0.0

    monkeypatch.setattr(workflow_mod, "refine_shift_with_correlation", fake_refine)
    monkeypatch.setattr(workflow_mod, "reramp", lambda samples, *_args: samples)

    def fake_resample(samples: np.ndarray, **kwargs: object) -> np.ndarray:
        resample_kwargs.update(kwargs)
        return samples.copy()

    monkeypatch.setattr(workflow_mod, "resample_complex", fake_resample)

    result = workflow_mod.stage_coregister(
        state,
        executor="numpy",
        device="auto",
    )

    assert result.secondary_aligned is not None
    assert ampcor_kwargs["executor"] == "numpy"
    assert ampcor_kwargs["device"] == "cpu"
    assert resample_kwargs["executor"] == "torch"
    assert resample_kwargs["device"] == "auto"
