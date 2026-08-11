"""Tests for persisted-product Stack pipeline orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.pipeline.stack_pipeline import run_stack_pipeline
from faninsar.processing.stack.ifg_store import write_ifg_artifact
from faninsar.processing.stack.session import Stack
from faninsar.processing.timeseries.inversion import TimeSeriesResult

if TYPE_CHECKING:
    from pathlib import Path


def test_stack_pipeline_runs_persisted_unwrap_then_sbas(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """Requested inversion uses Stack artifacts and publishes time-series Zarr."""
    calls: list[str] = []

    def prepare(stack: Stack) -> Stack:
        calls.append("prepare")
        assert stack.config.coregistration_grid == "geo"
        assert stack.config.geo_grid is not None
        assert stack.config.swaths == ("IW1", "IW2")
        assert stack.config.bursts == {"IW1": [0, 1], "IW2": [2]}
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
        return TimeSeriesResult(
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

    monkeypatch.setattr(Stack, "prepare_scenes", prepare)  # type: ignore[attr-defined]
    monkeypatch.setattr(Stack, "coregister_scenes", coregister)  # type: ignore[attr-defined]
    monkeypatch.setattr(Stack, "form_interferograms", form)  # type: ignore[attr-defined]
    monkeypatch.setattr(Stack, "unwrap", unwrap)  # type: ignore[attr-defined]
    monkeypatch.setattr(Stack, "invert_timeseries", invert)  # type: ignore[attr-defined]

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
    )

    assert calls == ["prepare", "coregister", "form", "unwrap", "invert:cpu"]
    assert result.timeseries is not None
    assert len(result.pair_results) == 1
    assert result.pair_results[0].multilook == (2, 8)
    assert result.pair_results[0].artifact_root.is_dir()
    assert result.pair_results[0].domain == "geo"
    assert result.timeseries_zarr == tmp_path / "out" / "timeseries.zarr"
    assert result.timeseries_zarr.exists()
