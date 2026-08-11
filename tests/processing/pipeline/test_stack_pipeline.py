"""Tests for persisted-product Stack pipeline orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.pipeline.stack_pipeline import run_stack_pipeline
from faninsar.processing.stack.ifg_store import write_ifg_artifact
from faninsar.processing.stack.session import Stack
from faninsar.processing.timeseries.inversion import TimeSeriesResult
from faninsar.query import BoundingBox

if TYPE_CHECKING:
    from pathlib import Path


def test_stack_constructor_preserves_roi_in_resume_identity(tmp_path: Path) -> None:
    """A Stack ROI reaches Pair calls and distinguishes resumable products."""
    paths = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000.SAFE"
        path.mkdir()
        paths.append(path)
    western_roi = BoundingBox(80.0, 20.0, 81.0, 21.0, crs="EPSG:4326")
    eastern_roi = BoundingBox(81.0, 20.0, 82.0, 21.0, crs="EPSG:4326")

    western_stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "western",
        activation_mode="reference",
        roi=western_roi,
    )
    eastern_stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "eastern",
        activation_mode="reference",
        roi=eastern_roi,
    )

    assert western_stack.config.roi is western_roi
    assert western_stack._burst_kwargs()["roi"] is western_roi
    assert western_stack._coreg_resume_identity(
        "20160113", misreg_az_px=0.0, misreg_rg_px=0.0
    ) != eastern_stack._coreg_resume_identity(
        "20160113", misreg_az_px=0.0, misreg_rg_px=0.0
    )


def test_stack_pipeline_runs_persisted_unwrap_then_sbas(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """Requested inversion uses Stack artifacts and publishes time-series Zarr."""
    calls: list[str] = []
    roi = BoundingBox(80.0, 20.0, 81.0, 21.0, crs="EPSG:4326")

    def prepare(stack: Stack) -> Stack:
        calls.append("prepare")
        assert stack.config.coregistration_grid == "geo"
        assert stack.config.geo_grid is not None
        assert stack.config.swaths == ("IW1", "IW2")
        assert stack.config.bursts == {"IW1": [0, 1], "IW2": [2]}
        assert stack.config.roi is roi
        stack._prepared = True
        return stack

    def coregister(stack: Stack) -> Stack:
        calls.append("coregister")
        return stack

    def form(stack: Stack) -> Stack:
        calls.append("form")
        artifact_root = stack.config.work_dir / "ifg" / "ml_2x8" / ("20240101_20240113")
        phase = np.zeros((2, 2), dtype=np.float32)
        write_ifg_artifact(
            artifact_root,
            pair=("20240101", "20240113"),
            looks=(2, 8),
            domain="geo",
            grid_identity="b" * 64,
            filter_name="none",
            filter_parameters={},
            source_manifest_digests={"scenes": "a" * 64},
            complex_ifg=np.ones((2, 2), dtype=np.complex64),
            coherence=np.ones((2, 2), dtype=np.float32),
            wrapped_phase=phase,
            amplitude=np.ones((2, 2), dtype=np.float32),
        )
        stack.ifg_dirs = [artifact_root]
        return stack

    def unwrap(stack: Stack) -> Stack:
        calls.append("unwrap")
        return stack

    def invert(_stack: Stack, *, device: str | None = None) -> TimeSeriesResult:
        calls.append(f"invert:{device}")
        result = TimeSeriesResult(
            pair_ids=("20240101_20240113",),
            dates=("20240101", "20240113"),
            increments=np.ones((1, 2, 2), dtype=np.float32),
            residual_pairs=np.zeros((1, 2, 2), dtype=np.float32),
            cumulative=np.stack(
                [
                    np.zeros((2, 2), dtype=np.float32),
                    np.ones((2, 2), dtype=np.float32),
                ]
            ),
            metadata={"method": "sbas"},
        )
        _stack.timeseries = result
        return result

    def publish(_stack: Stack, timeseries_root: Path) -> SimpleNamespace:
        calls.append("publish")
        assert timeseries_root == tmp_path / "out" / "timeseries.zarr"
        return SimpleNamespace(
            generation_id="stack-generation",
            manifest_digest="a" * 64,
        )

    monkeypatch.setattr(Stack, "prepare_scenes", prepare)  # type: ignore[attr-defined]
    monkeypatch.setattr(Stack, "coregister_scenes", coregister)  # type: ignore[attr-defined]
    monkeypatch.setattr(Stack, "form_interferograms", form)  # type: ignore[attr-defined]
    monkeypatch.setattr(Stack, "unwrap", unwrap)  # type: ignore[attr-defined]
    monkeypatch.setattr(Stack, "invert_timeseries", invert)  # type: ignore[attr-defined]
    monkeypatch.setattr(Stack, "publish_generation", publish)  # type: ignore[attr-defined]

    paths = []
    for date_id in ("20240101", "20240113"):
        path = tmp_path / (f"S1A_IW_SLC__1SDV_{date_id}T000000_{date_id}T000001.SAFE")
        path.mkdir()
        paths.append(path)
    result = run_stack_pipeline(
        paths,
        output_dir=tmp_path / "out",
        pairs=[("20240101", "20240113")],
        activation_mode="reference",
        invert_timeseries=True,
        invert_device="cpu",
        coregistration_grid="geo",
        geo_grid=GeoGridSpec(
            crs="EPSG:32647",
            transform=(0.0, 40.0, 0.0, 80.0, 0.0, -40.0),
            width=2,
            height=2,
            resolution_m=(40.0, 40.0),
        ),
        swaths=("IW1", "IW2"),
        bursts={"IW1": [0, 1], "IW2": [2]},
        roi=roi,
    )

    assert calls == [
        "prepare",
        "coregister",
        "form",
        "unwrap",
        "invert:cpu",
        "publish",
    ]
    assert result.timeseries is not None
    assert len(result.pair_results) == 1
    assert result.pair_results[0].multilook == (2, 8)
    assert result.pair_results[0].artifact_root.is_dir()
    assert result.pair_results[0].domain == "geo"
    assert result.timeseries_zarr == tmp_path / "out" / "timeseries.zarr"
    assert result.timeseries_zarr.exists()
    assert result.stack_generation is not None
    assert result.stack_generation.generation_id == "stack-generation"
