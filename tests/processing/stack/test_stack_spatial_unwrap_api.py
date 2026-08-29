"""Tests for the spatial-only Stack unwrap boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from faninsar import Pairs
from faninsar.processing.resources import ResourceAdmissionError, ResourceBudget
from faninsar.processing.stack import Stack
from faninsar.processing.stack.ifg_store import write_ifg_artifact
from faninsar.processing.unwrap.common import SpatialUnwrapper, SpatialUnwrapResult

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
    stack = Stack.from_safes(
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
