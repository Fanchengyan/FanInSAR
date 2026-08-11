"""Tests for the 2D→1D→batch_lstsq stack unwrap orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.unwrap.irls import wrap_phase
from faninsar.processing.unwrap.quality import StackQualityCriteria
from faninsar.processing.unwrap.stack import unwrap_stack


def _pair_dates() -> list[tuple[str, str]]:
    return [
        ("20161207", "20161231"),
        ("20161231", "20170124"),
        ("20161207", "20170124"),
    ]


def test_unwrap_stack_skip_spatial() -> None:
    """do_spatial=False keeps phase_2d_unw identical to the input stack."""
    pairs = _pair_dates()
    rng = np.random.default_rng(1)
    phase_stack = rng.uniform(-np.pi, np.pi, size=(3, 8, 8)).astype(np.float64)

    result = unwrap_stack(
        phase_stack,
        pairs,
        do_spatial=False,
        do_temporal=False,
        do_invert=False,
    )

    assert result.phase_2d_unw is not None
    np.testing.assert_array_equal(result.phase_2d_unw, phase_stack)
    assert result.temporal_applied is False
    assert result.inverted is False


def test_unwrap_stack_skip_temporal() -> None:
    """do_temporal=False leaves phase_1d_unw as None."""
    pairs = _pair_dates()
    y, x = np.mgrid[0:8, 0:8]
    true = 0.2 * x + 0.1 * y
    phase_stack = np.stack(
        [wrap_phase(true), wrap_phase(0.5 * true), wrap_phase(true)],
        axis=0,
    )

    result = unwrap_stack(
        phase_stack,
        pairs,
        do_spatial=True,
        do_temporal=False,
        do_invert=False,
        spatial_method="irls",
        spatial_kwargs={"max_iter": 15},
    )

    assert result.phase_2d_unw is not None
    assert result.phase_2d_unw.shape == phase_stack.shape
    assert result.phase_1d_unw is None
    assert result.temporal_applied is False
    assert result.corrections_k is None


def test_unwrap_stack_retains_independent_spatial_islands() -> None:
    """Disconnected valid islands remain available for temporal rank checks."""
    pairs = _pair_dates()
    y, x = np.mgrid[0:12, 0:12]
    phase = wrap_phase(0.1 * x + 0.05 * y)
    phase[:, 5:7] = np.nan
    phase_stack = np.stack([phase, 0.5 * phase, 1.5 * phase], axis=0)

    result = unwrap_stack(
        phase_stack,
        pairs,
        do_spatial=True,
        do_temporal=False,
        do_invert=False,
        spatial_kwargs={"max_iter": 15},
    )

    assert result.phase_2d_unw is not None
    assert result.connected_components is not None
    assert np.isfinite(result.phase_2d_unw[:, :, :5]).all()
    assert np.isfinite(result.phase_2d_unw[:, :, 7:]).all()
    assert np.nanmax(result.connected_components) >= 2


def test_unwrap_stack_full_chain() -> None:
    """Full 2D→1D→invert chain returns consistent shapes on a small synthetic stack."""
    pairs = _pair_dates()
    shape = (8, 8)
    true_inc01 = np.full(shape, 0.15, dtype=np.float64)
    true_inc12 = np.full(shape, -0.08, dtype=np.float64)
    # Consistent unwrapped pair phases (skip spatial so we control the network)
    phase_stack = np.stack(
        [true_inc01, true_inc12, true_inc01 + true_inc12],
        axis=0,
    )
    # Plant a 2π jump on the first pair so temporal IRLS has work to do
    phase_stack = phase_stack.copy()
    phase_stack[0] = phase_stack[0] + 2.0 * np.pi

    result = unwrap_stack(
        phase_stack,
        pairs,
        do_spatial=False,
        do_temporal=True,
        do_invert=True,
        temporal_device="cpu",
        lstsq_device="cpu",
        # wrapped residual mode recovers integer 2π network ambiguities
        temporal_kwargs={"max_iter": 20, "wrapped_input": True},
    )

    assert result.phase_2d_unw is not None
    assert result.phase_1d_unw is not None
    assert result.corrections_k is not None
    assert result.timeseries is not None
    assert result.phase_2d_unw.shape == (3, *shape)
    assert result.phase_1d_unw.shape == (3, *shape)
    assert result.corrections_k.shape == (3, *shape)
    # cumulative: n_dates = 3
    assert result.timeseries.shape == (3, *shape)
    assert result.temporal_applied is True
    assert result.inverted is True
    assert result.quality_report is not None
    assert result.quality_report.passed
    assert len(result.pair_ids) == 3
    # Temporal stage should remove the planted 2π jump
    np.testing.assert_allclose(
        result.phase_1d_unw[0],
        true_inc01,
        atol=0.05,
    )


def test_unwrap_stack_does_not_gate_on_multilook_closure() -> None:
    """Non-zero temporal closure remains usable for downstream inversion."""
    phase_stack = np.broadcast_to(
        np.array([1.0, 2.0, 3.3])[:, None, None],
        (3, 2, 2),
    ).copy()

    result = unwrap_stack(
        phase_stack,
        _pair_dates(),
        do_spatial=False,
        do_temporal=True,
        do_invert=True,
        temporal_kwargs={"max_iter": 10},
        quality_criteria=StackQualityCriteria(),
    )

    assert result.quality_report is not None
    assert result.quality_report.passed
