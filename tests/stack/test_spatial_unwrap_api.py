"""Tests for the spatial-only Stack unwrap boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from faninsar import Pairs
from faninsar.io.storage.ifg_store import write_ifg_artifact
from faninsar.missions.s1 import S1Stack
from faninsar.processing.runtime import device as device_runtime
from faninsar.processing.runtime.resources import ResourceAdmissionError, ResourceBudget
from faninsar.processing.unwrapping.common import SpatialUnwrapper, SpatialUnwrapResult
from faninsar.stack import Stack

if TYPE_CHECKING:
    from pathlib import Path


class _IdentityUnwrapper(SpatialUnwrapper):
    """Small deterministic strategy used to exercise Stack orchestration."""

    def unwrap(
        self,
        wrapped_phase: torch.Tensor,
        *,
        coherence: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> SpatialUnwrapResult:
        """Return the supported input as a successful spatial result."""
        del coherence
        support = torch.isfinite(wrapped_phase)
        if valid_mask is not None:
            support &= valid_mask
        labels = torch.where(
            support,
            torch.zeros_like(wrapped_phase, dtype=torch.int64),
            torch.full_like(wrapped_phase, -1, dtype=torch.int64),
        )
        return SpatialUnwrapResult(
            phase=torch.where(support, wrapped_phase, torch.nan),
            valid_mask=support,
            component_labels=labels,
            reference_values=torch.zeros(1, device=wrapped_phase.device),
            converged=True,
            iterations=0,
            pcg_iterations=0,
            residual_norm=0.0,
            failure_reason=None,
        )


def _stack_with_ifg(tmp_path: Path) -> Stack:
    """Build one small Stack with one persisted IFG input."""
    dates = ("20240101", "20240113")
    safe_paths = []
    for date in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date}T000000.SAFE"
        path.mkdir()
        safe_paths.append(path)
    stack = S1Stack.from_safes(
        safe_paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        pairs=Pairs.from_names(["20240101_20240113"]),
        multilook=(1, 1),
    ).prepare_scenes()
    phase = np.full((2, 3), 0.2, dtype=np.float32)
    write_ifg_artifact(
        stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113",
        pair=dates,
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"scenes": "a" * 64},
        complex_ifg=np.exp(1j * phase).astype(np.complex64),
        coherence=np.ones_like(phase),
        wrapped_phase=phase,
        amplitude=np.ones_like(phase),
    )
    stack.ifg_dirs = [
        stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113"
    ]
    return stack


def test_stack_unwrap_is_spatial_only_and_publishes_one_root_generation(
    tmp_path: Path,
) -> None:
    """Stack dispatches one spatial strategy and advances UNWRAP_CURRENT."""
    stack = _stack_with_ifg(tmp_path)

    stack.unwrap(_IdentityUnwrapper())

    assert (stack.config.work_dir / "UNWRAP_CURRENT").is_file()
    assert stack.analysis_ready
    assert stack.network_product_index is not None
    assert len(stack.network_product_index.products) == 2


def test_stack_unwrap_accepts_concrete_device_for_generic_cuda_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Device validation compares against the materialized tensor device."""
    stack = _stack_with_ifg(tmp_path)
    stack.config.device = "cuda"
    original_as_tensor = torch.as_tensor

    monkeypatch.setattr(
        device_runtime,
        "parse_device",
        lambda _requested: torch.device("cuda"),
    )

    def materialize_cuda_on_cpu(*args: object, **kwargs: object) -> torch.Tensor:
        """Emulate CUDA allocation while keeping this CPU test portable."""
        requested = kwargs.get("device")
        if requested is not None and torch.device(requested).type == "cuda":
            kwargs["device"] = torch.device("cpu")
        return original_as_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", materialize_cuda_on_cpu)

    stack.unwrap(_IdentityUnwrapper())

    assert (stack.config.work_dir / "UNWRAP_CURRENT").is_file()


def test_stack_unwrap_connects_directly_to_time_series_analysis(tmp_path: Path) -> None:
    """The root unwrap generation is the input to Stack SBAS analysis."""
    stack = _stack_with_ifg(tmp_path)

    stack.unwrap(_IdentityUnwrapper())

    result = stack.analyze_time_series()

    assert result.pair_ids == ("20240101_20240113",)
    assert result.phase_cumulative_rad.shape == (2, 2, 3)


def test_stack_network_exposes_its_interferogram_collection(tmp_path: Path) -> None:
    """An analysis-ready Stack provides readable Network interferograms."""
    stack = _stack_with_ifg(tmp_path)

    stack.unwrap(_IdentityUnwrapper())
    network = stack.network

    assert network is not None
    assert network.interferograms.pairs().to_names().tolist() == [
        "20240101_20240113"
    ]
    unwrapped = network.interferograms.open_stack("unw_phase")
    assert unwrapped.shape == (1, 2, 3)
    np.testing.assert_allclose(unwrapped.values[0], 0.2)


def test_stack_unwrap_public_signature_has_no_temporal_controls() -> None:
    """The Stack boundary accepts a strategy, not temporal orchestration knobs."""
    import inspect

    parameters = inspect.signature(Stack.unwrap).parameters
    assert tuple(parameters)[:2] == ("self", "unwrapper")
    assert not any(name.startswith("temporal") for name in parameters)


def test_stack_unwrap_rejects_decode_before_dataset_materialization(
    tmp_path: Path,
) -> None:
    """A configured budget rejects the pinned Dataset decode before dispatch."""
    stack = _stack_with_ifg(tmp_path)
    stack.config.resource_budget = ResourceBudget(
        max_files=8,
        max_chunks=8,
        max_encoded_bytes=1024,
        max_decoded_bytes=1,
        max_temporary_bytes=1024,
        max_workers=1,
        max_processes=1,
        disk_reserve_bytes=1,
        max_rss_bytes=4 * 1024 * 1024 * 1024,
        max_device_bytes=1024,
    )

    with pytest.raises(ResourceAdmissionError, match="decoded_bytes"):
        stack.unwrap(_IdentityUnwrapper())
