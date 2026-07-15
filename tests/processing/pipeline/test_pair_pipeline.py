"""Compatibility tests for run_pair_pipeline path-based API."""

from __future__ import annotations

from pathlib import Path

import pytest

from faninsar.processing.pipeline import run_pair_pipeline

SLC_ROOT = Path("/Volumes/DATA2/TEST_sentinel-1/sentinel-slc")
SCENES = sorted(SLC_ROOT.glob("S1A_IW_SLC*.zip")) if SLC_ROOT.exists() else []


@pytest.mark.skipif(len(SCENES) < 2, reason="need two local S1 ZIP scenes")
def test_run_pair_pipeline_requires_safe_paths(tmp_path: Path) -> None:
    """Path-based API runs the full workflow and writes products."""
    result = run_pair_pipeline(
        SCENES[0],
        SCENES[1],
        output_dir=tmp_path / "pair",
        height=64,
        width=64,
        multilook=(2, 2),
    )
    assert result.zarr_path.exists()
    assert result.stac_path.exists()
    assert result.state.geocoded_unwrapped is not None


def test_run_pair_pipeline_rejects_bare_arrays(tmp_path: Path) -> None:
    """Bare arrays without SAFE paths are rejected."""
    import numpy as np

    arr = np.ones((8, 8), dtype=np.complex64)
    with pytest.raises(TypeError, match="SAFE paths"):
        run_pair_pipeline(arr, arr, pair_id="x", output_dir=tmp_path)
