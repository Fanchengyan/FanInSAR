"""Contract tests for foundational missing-observation least squares."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from tests.core.algorithms.reference.linalg import reference_least_squares


def solve(*args, **kwargs):
    return importlib.import_module("faninsar.core.algorithms.linalg").least_squares(
        *args, **kwargs
    )


def assert_reference(result, expected):
    for name, value in zip(
        ("solution", "rank", "n_observations", "is_constrained", "condition_number"),
        expected,
        strict=True,
    ):
        actual = getattr(result, name)
        assert isinstance(actual, torch.Tensor)
        np.testing.assert_allclose(actual.cpu().numpy(), value, atol=1e-10, rtol=1e-12)
    assert result.solution.dtype == torch.float64
    assert result.condition_number.dtype == torch.float64
    assert result.is_constrained.dtype == torch.bool


def test_full_rank_known_coefficients_float32_storage():
    a = np.array([[1, 0], [1, 1], [1, 2], [1, 4]], dtype=np.float32)
    coefficients = np.array([[2, -4], [3, 0.5]], dtype=np.float32)
    b = a @ coefficients
    result = solve(a, b)
    assert_reference(result, reference_least_squares(a, b))
    np.testing.assert_allclose(result.solution, coefficients, atol=1e-10)
    assert result.solution.device.type == "cpu"
    assert torch.all(result.is_constrained)


@pytest.mark.parametrize("per_rhs", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_each_rhs_selects_its_own_masked_finite_rows(per_rhs, weighted):
    a = np.ma.array(
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]],
        mask=False,
    )
    a.mask[1, 1] = True
    b = np.ma.array(
        [[2.0, -1.0], [5.0, 3.0], [8.0, 7.0], [11.0, 11.0], [14.0, 15.0], [17.0, 19.0]],
        mask=False,
    )
    b.mask[0, 0] = True
    b[2, 1] = np.inf
    b[3, 0] = np.nan
    if per_rhs:
        a = np.ma.stack([a, a.copy()])
        a[1, 4, 0] = -np.inf
    valid = np.ma.array(np.ones(b.shape, dtype=bool), mask=False)
    valid[5, 0] = False
    weights = (
        np.ma.array([1.0, 2.0, 0.0, 4.0, 3.0, 2.0], mask=False) if weighted else None
    )
    if weighted:
        weights.mask[3] = True
    before_a, before_b = a.copy(), b.copy()
    result = solve(
        a, b, valid=valid, weights=weights, method="wls" if weighted else "ols"
    )
    assert_reference(
        result, reference_least_squares(a, b, valid=valid, weights=weights)
    )
    np.testing.assert_equal(a.data, before_a.data)
    np.testing.assert_equal(a.mask, before_a.mask)
    np.testing.assert_equal(b.data, before_b.data)
    np.testing.assert_equal(b.mask, before_b.mask)


@pytest.mark.parametrize(
    "a,b",
    [
        (np.array([[1.0, 0.0]]), np.array([2.0])),
        (np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([2.0, 4.0])),
        (np.zeros((3, 2)), np.array([2.0, 3.0, 4.0])),
        (np.eye(2), np.array([np.nan, np.inf])),
        (np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), np.array([2.0, 3.0])),
        (np.zeros((0, 2)), np.zeros(0)),
    ],
)
def test_deficient_wide_zero_rank_and_no_observation_semantics(a, b):
    result = solve(a, b)
    assert_reference(result, reference_least_squares(a, b))
    assert result.rank.shape == torch.Size([])
    assert result.solution.shape == (a.shape[1],)
    if a.shape == (1, 2):
        torch.testing.assert_close(
            result.solution, torch.tensor([2.0, 0.0], dtype=torch.float64)
        )
        assert result.is_constrained.tolist() == [True, False]


def test_singular_cutoff_and_actual_retained_condition_not_legacy_proxy():
    for diagonal, expected_rank, condition in [
        ([1.0, 1e-7], 2, 1e7),
        ([1.0, 0.01], 2, 100.0),
        ([1.0, 1e-11], 1, 1.0),
    ]:
        result = solve(np.diag(diagonal), np.ones(2))
        assert result.rank.item() == expected_rank
        assert result.condition_number.item() == pytest.approx(condition)


def test_wls_uses_sqrt_weights_not_weights_and_keeps_finite_outlier():
    a = np.ones((3, 1))
    b = np.array([0.0, 2.0, 100.0])
    weights = np.array([1.0, 2.0, 0.5])
    result = solve(a, b, method="wls", weights=weights)
    assert result.solution.item() == pytest.approx(np.average(b, weights=weights))
    assert result.n_observations.item() == 3


@pytest.mark.parametrize("rcond", [-1, np.nan, np.inf, True, "bad"])
def test_invalid_rcond_rejected(rcond):
    with pytest.raises((TypeError, ValueError), match="rcond"):
        solve(np.eye(2), np.ones(2), rcond=rcond)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "unknown"},
        {"weights": np.ones(2)},
        {"method": "wls"},
        {"method": "wls", "weights": np.array([-1.0, 1.0])},
        {"method": "wls", "weights": np.array([np.inf, 1.0])},
        {"method": "wls", "weights": np.array([np.nan, 1.0])},
        {"valid": np.ones(2)},
        {"valid": np.ones((2, 1), dtype=bool)},
    ],
)
def test_method_weight_and_valid_validation(kwargs):
    with pytest.raises((TypeError, ValueError)):
        solve(np.eye(2), np.ones(2), **kwargs)


@pytest.mark.parametrize(
    "a,b",
    [
        (np.ones(2), np.ones(2)),
        (np.ones((2, 2, 2, 2)), np.ones(2)),
        (np.ones((3, 2)), np.ones(2)),
        (np.ones((2, 2, 2)), np.ones(2)),
        (np.ones((2, 0)), np.ones(2)),
        (np.ones((2, 2), dtype=np.int64), np.ones(2)),
        (np.eye(2), np.ones(2, dtype=np.complex128)),
        (torch.eye(2, requires_grad=True), torch.ones(2)),
    ],
)
def test_shape_dtype_gradient_validation(a, b):
    with pytest.raises((TypeError, ValueError)):
        solve(a, b)


def test_native_masked_nonfinite_weights_do_not_raise_or_resurrect():
    weights = np.ma.array([np.inf, np.nan, 2.0], mask=[True, True, False])
    b = np.ma.array([5.0, 6.0, 7.0], mask=False)
    result = solve(
        np.ones((3, 1)), b, method="wls", weights=weights, valid=np.ones(3, dtype=bool)
    )
    assert result.solution.item() == pytest.approx(7, abs=1e-10, rel=1e-12)
    assert result.n_observations.item() == 1


def test_noncontiguous_tensor_data_and_empty_rhs_batch():
    a = torch.arange(24, dtype=torch.float64).reshape(4, 6)[:, ::2]
    b = torch.arange(16, dtype=torch.float64).reshape(4, 4)[:, ::2]
    assert_reference(solve(a, b), reference_least_squares(a.numpy(), b.numpy()))
    result = solve(np.eye(2), np.empty((2, 0)))
    assert result.solution.shape == (2, 0)
    assert result.rank.shape == (0,)


def test_preflight_rejects_unavoidable_workspace_before_conversion():
    a = np.lib.stride_tricks.as_strided(np.ones(1), shape=(2**30, 2), strides=(0, 0))
    b = np.lib.stride_tricks.as_strided(np.ones(1), shape=(2**30,), strides=(0,))
    with pytest.raises((ValueError, MemoryError), match="limit|large|budget|workspace"):
        solve(a, b)


@pytest.mark.parametrize("pattern_count", [1, 4])
def test_shared_design_reuses_one_svd_per_observation_pattern(
    monkeypatch, pattern_count
):
    rng = np.random.default_rng(87)
    a = rng.normal(size=(32, 5))
    b = rng.normal(size=(32, 24))
    for column in range(b.shape[1]):
        if pattern_count > 1:
            b[column % pattern_count, column] = np.nan
    calls = []
    original = torch.linalg.svd

    def counted_svd(matrix, **kwargs):
        calls.append(tuple(matrix.shape))
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted_svd)
    result = solve(a, b)
    assert_reference(result, reference_least_squares(a, b))
    assert len(calls) == pattern_count
    assert all(len(shape) == 2 for shape in calls)


@pytest.mark.parametrize("per_rhs", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_bounded_chunks_preserve_distinct_masks_weights_and_diagnostics(
    monkeypatch, per_rhs, weighted
):
    module = importlib.import_module("faninsar.core.algorithms.linalg._least_squares")
    monkeypatch.setattr(module, "_RHS_BATCH", 3, raising=False)
    rng = np.random.default_rng(88)
    a = rng.normal(size=(11, 7, 4)) if per_rhs else rng.normal(size=(7, 4))
    b = rng.normal(size=(7, 11))
    b[:, 0] = np.nan
    b[1::2, 1] = np.inf
    b[0, 3:] = np.nan
    if per_rhs:
        a[2] = 0
        a[3, 2, 0] = np.nan
    weights = rng.uniform(0.1, 2.0, size=b.shape) if weighted else None
    if weighted:
        weights[2, 4] = 0
    kwargs = {"method": "wls", "weights": weights} if weighted else {}
    assert_reference(
        solve(a, b, **kwargs), reference_least_squares(a, b, weights=weights)
    )


def test_many_distinct_shared_masks_use_bounded_batched_svd(monkeypatch):
    rng = np.random.default_rng(89)
    a = rng.normal(size=(32, 5))
    b = rng.normal(size=(32, 24))
    for column in range(b.shape[1]):
        b[column, column] = np.nan
    shapes = []
    original = torch.linalg.svd

    def counted_svd(matrix, **kwargs):
        shapes.append(tuple(matrix.shape))
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted_svd)
    assert_reference(solve(a, b), reference_least_squares(a, b))
    assert len(shapes) == 1
    assert shapes[0] == (24, 32, 5)


@pytest.mark.parametrize("weighted", [False, True])
def test_batched_wide_deficient_zero_and_no_data_diagnostics(weighted):
    a = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    b = np.array([[2.0, 2.0, 3.0, np.nan], [4.0, 4.0, 5.0, np.nan]])
    w = np.array([[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 1.0, 1.0]]) if weighted else None
    kwargs = {"method": "wls", "weights": w} if weighted else {}
    assert_reference(solve(a, b, **kwargs), reference_least_squares(a, b, weights=w))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_stays_on_requested_device():
    a = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], device="cuda")
    b = torch.tensor([2.0, 5.0, 8.0], device="cuda")
    result = solve(a, b)
    assert result.solution.device == a.device
    torch.testing.assert_close(
        result.solution, torch.tensor([2.0, 3.0], dtype=torch.float64, device="cuda")
    )
    with pytest.raises(ValueError, match="device"):
        solve(a, b.cpu())
