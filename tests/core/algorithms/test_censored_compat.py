"""Freeze historical time-series semantics during private numerical extraction."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from faninsar.timeseries.solver import LstsqResult, batch_lstsq, censored_lstsq


def test_domain_wrappers_delegate_to_private_legacy_kernel(monkeypatch):
    module = importlib.import_module("faninsar.core.algorithms.linalg._censored_compat")
    calls = []
    original = module._censored_lstsq_kernel

    def traced(g, d, *, rcond):
        calls.append((g.shape, d.shape, rcond))
        return original(g, d, rcond=rcond)

    monkeypatch.setattr(module, "_censored_lstsq_kernel", traced)
    a = np.eye(2)
    b = np.ones((2, 3))
    for solver in (censored_lstsq, batch_lstsq):
        kwargs = {"verbose": False} if solver is batch_lstsq else {}
        result = solver(a, b, device="cpu", return_numpy=False, **kwargs)
        assert isinstance(result, LstsqResult)
        torch.testing.assert_close(
            result.solution, torch.ones((2, 3), dtype=torch.float64)
        )
    assert len(calls) == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("return_numpy", [False, True])
def test_old_pivot_condition_and_single_rhs_shape_remain_unchanged(dtype, return_numpy):
    result = censored_lstsq(
        np.diag([1.0, 0.01]),
        np.array([2.0, 3.0]),
        dtype=dtype,
        device="cpu",
        return_numpy=return_numpy,
    )
    assert result.solution.shape == (2, 1)
    assert result.rank[0] == 2
    assert result.condition_number[0] == pytest.approx(0.01, rel=1e-5)
    np.testing.assert_allclose(result.solution, [[2.0], [300.0]], rtol=1e-5)
    assert isinstance(result.solution, np.ndarray if return_numpy else torch.Tensor)


def test_old_gram_rank_cutoff_is_not_new_public_singular_cutoff():
    result = censored_lstsq(
        np.diag([1.0, 1e-7]), np.ones(2), device="cpu", return_numpy=False
    )
    assert result.rank.item() == 1
    assert result.is_constrained[:, 0].tolist() == [True, False]
    torch.testing.assert_close(
        result.solution[:, 0], torch.tensor([1.0, 0.0], dtype=torch.float64)
    )
    assert result.condition_number.item() == 1.0


def test_old_nan_only_selection_and_rank_zero_diagnostics_are_preserved():
    result = censored_lstsq(
        np.eye(2),
        np.array([[1.0, np.nan, np.inf], [2.0, np.nan, 3.0]]),
        device="cpu",
        return_numpy=False,
    )
    assert result.n_observations.tolist() == [2, 0, 2]
    assert result.rank.tolist() == [2, 0, 2]
    assert torch.isnan(result.solution[:, 1]).all()
    zero = censored_lstsq(
        np.zeros((2, 3)), np.ones((2, 1)), device="cpu", return_numpy=False
    )
    assert zero.rank.item() == 0
    assert zero.condition_number.item() == 0.0
    assert not zero.is_constrained.any()
    assert torch.equal(zero.solution, torch.zeros((3, 1), dtype=torch.float64))


def test_per_rhs_design_and_missing_observations_preserve_old_minimum_norm():
    a = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]]])
    b = np.array([[2.0, 2.0], [np.nan, 4.0]])
    result = censored_lstsq(a, b, device="cpu", return_numpy=False)
    torch.testing.assert_close(
        result.solution, torch.tensor([[2.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    )
    assert result.rank.tolist() == [1, 1]
    assert result.n_observations.tolist() == [1, 2]
    assert result.is_constrained.tolist() == [[True, False], [False, False]]
