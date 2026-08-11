"""Real SAFE tests using local scenes as fixtures for the production workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faninsar.missions.sentinel1 import open_safe_product, read_burst_window
from faninsar.processing.pipeline import run_pair_workflow, run_stack_pipeline

SLC_ROOT = Path("/Volumes/DATA2/TEST_sentinel-1/sentinel-slc")
SCENES = sorted(SLC_ROOT.glob("S1A_IW_SLC*.zip")) if SLC_ROOT.exists() else []


@pytest.mark.skipif(len(SCENES) < 1, reason="local S1 ZIP corpus unavailable")
def test_read_burst_window_from_real_safe() -> None:
    """Read a non-empty complex window from a real SAFE measurement."""
    product = open_safe_product(SCENES[0])
    window = read_burst_window(
        product.swath("IW1"),
        burst_index=0,
        height=64,
        width=64,
    )
    assert window.samples.shape == (64, 64)
    assert np.iscomplexobj(window.samples)
    assert float(np.mean(np.abs(window.samples) > 0)) > 0.5


@pytest.mark.skipif(len(SCENES) < 2, reason="need two local S1 ZIP scenes")
def test_pair_workflow_on_two_real_scenes(tmp_path: Path) -> None:
    """Two real SAFE scenes complete deramp→coreg→ifg→unwrap→geocode→write."""
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
    assert state.geocoded_unwrapped is not None
    assert any("DERAMP" in line for line in state.log)
    assert any("GEOCODE" in line for line in state.log)


@pytest.mark.skipif(len(SCENES) < 3, reason="need three local S1 ZIP scenes")
def test_stack_pipeline_uses_full_workflow(tmp_path: Path) -> None:
    """Three real scenes exercise the general stack API as fixtures only."""
    result = run_stack_pipeline(
        SCENES[:3],
        output_dir=tmp_path / "stack",
        activation_mode="reference",
        height=64,
        width=64,
        multilook=(2, 2),
        invert_timeseries=True,
    )
    assert len(result.scene_ids) == 3
    assert len(result.pair_results) == 3
    for pair_result in result.pair_results:
        assert pair_result.artifact_root.exists()
        assert (pair_result.artifact_root / "manifest.json").is_file()
        assert len(pair_result.manifest_digest) == 64
    assert result.timeseries_zarr is not None
    assert result.timeseries_zarr.exists()
