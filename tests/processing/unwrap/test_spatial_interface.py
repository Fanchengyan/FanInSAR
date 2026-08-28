"""Public contract tests for spatial unwrapping."""

from __future__ import annotations

import pytest
import torch

from faninsar.processing.unwrap.common import (
    SpatialUnwrapper,
    SpatialUnwrapResult,
)
from faninsar.processing.unwrap.errors import NoValidSupportError
from faninsar.processing.unwrap.irls import SpatialIRLS, wrap_phase


def test_spatial_irls_returns_torch_result_with_row_major_anchors() -> None:
    """A supported field is labelled deterministically and remains device-local."""
    phase = torch.tensor(
        [[0.0, 0.2, float("nan")], [0.1, 0.3, float("nan")]],
        dtype=torch.float32,
    )
    result = SpatialIRLS().unwrap(phase, valid_mask=torch.isfinite(phase))

    assert isinstance(SpatialIRLS(), SpatialUnwrapper)
    assert isinstance(result, SpatialUnwrapResult)
    assert result.phase.device == phase.device
    assert result.phase.shape == phase.shape
    assert result.component_labels.tolist() == [[0, 0, -1], [0, 0, -1]]
    assert torch.equal(result.reference_values, torch.tensor([0.0]))
    assert torch.isnan(result.phase[0, 2])
    assert result.converged
    assert result.iterations == 0
    assert result.pcg_iterations == 0
    assert result.failure_reason is None


def test_zero_coherence_cuts_edges_but_keeps_supported_pixels() -> None:
    """Zero quality disconnects edges without invalidating either endpoint."""
    phase = torch.zeros((1, 3), dtype=torch.float32)
    coherence = torch.tensor([[1.0, 0.0, 1.0]])
    result = SpatialIRLS().unwrap(phase, coherence=coherence)

    assert result.component_labels.tolist() == [[0, 1, 2]]
    assert result.valid_mask.tolist() == [[True, True, True]]
    assert result.reference_values.tolist() == [0.0, 0.0, 0.0]


def test_spatial_irls_rejects_nonfinite_or_out_of_range_coherence() -> None:
    """Coherence is a finite probability-like quality value."""
    phase = torch.zeros((2, 2))
    with pytest.raises(ValueError, match="coherence"):
        SpatialIRLS().unwrap(phase, coherence=torch.tensor([[1.1, 0.0], [0.0, 0.0]]))
    result = SpatialIRLS().unwrap(
        phase, coherence=torch.tensor([[float("nan"), 0.0], [0.0, 0.0]])
    )
    assert not result.valid_mask[0, 0]


def test_spatial_irls_raises_before_solving_without_support() -> None:
    """An empty authoritative support has a dedicated public exception."""
    phase = torch.zeros((2, 2))
    with pytest.raises(NoValidSupportError):
        SpatialIRLS().unwrap(
            phase, valid_mask=torch.zeros_like(phase, dtype=torch.bool)
        )


def test_wrap_phase_uses_half_open_interval() -> None:
    """Wrapping maps positive pi to the documented half-open interval."""
    result = wrap_phase(torch.tensor([-torch.pi, torch.pi, 3.0 * torch.pi]))
    assert torch.all(result >= -torch.pi)
    assert torch.all(result < torch.pi)
    assert result[1].item() == pytest.approx(-torch.pi)


def test_spatial_irls_solves_a_wrapped_ramp_and_reports_work() -> None:
    """A wrapped ramp is recovered while the solver reports finite work."""
    rows, columns = torch.meshgrid(
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32),
        indexing="ij",
    )
    truth = 0.9 * columns + 0.7 * rows
    result = SpatialIRLS(max_iter=12, cg_max_iter=100).unwrap(wrap_phase(truth))

    residual = wrap_phase(result.phase - truth)
    assert float(torch.linalg.vector_norm(residual)) < 0.2
    assert result.converged
    assert result.failure_reason is None
    assert result.iterations >= 1
    assert result.pcg_iterations >= 1
    assert result.residual_norm < 0.2
