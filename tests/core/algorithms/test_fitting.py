"""Independent-reference polynomial evaluation and fitting contracts."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from tests.core.algorithms.reference.fitting import reference_fit, reference_polyval


def fitting():
    return importlib.import_module("faninsar.core.algorithms.fitting")


def test_polyval_ascending_horner_float64_and_pointwise_invalid_x():
    c = np.array([2.0, -3.0, 0.5], dtype=np.float32)
    x = np.ma.array([[0.0, 1.0, 2.0], [np.inf, np.nan, 5.0]], mask=False)
    x.mask[0, 1] = True
    before = x.copy()
    result = fitting().polyval(c, x)
    np.testing.assert_allclose(result, reference_polyval(c, x), atol=1e-10, rtol=1e-12)
    assert result.dtype == torch.float64
    np.testing.assert_equal(x.data, before.data)
    np.testing.assert_equal(x.mask, before.mask)
    assert fitting().polyval(c, np.array(2.0)).shape == torch.Size([])
    assert fitting().polyval(c, np.empty((0, 3))).shape == (0, 3)


@pytest.mark.parametrize(
    "coefficients",
    [
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.ma.array([1.0, 2.0], mask=[False, True]),
        np.ma.array([1.0, 2.0], mask=True),
    ],
)
def test_unavailable_coefficient_invalidates_whole_polynomial(coefficients):
    result = fitting().polyval(coefficients, np.array([0.0, 1.0, 2.0]))
    assert torch.isnan(result).all()


@pytest.mark.parametrize(
    "coefficients", [np.empty(0), np.ones((2, 2)), np.ones(2, dtype=int), [1.0, 2.0]]
)
def test_polyval_rejects_invalid_coefficient_container_shape_dtype(coefficients):
    with pytest.raises((TypeError, ValueError)):
        fitting().polyval(coefficients, np.array([1.0]))


def test_total_degree_surface_known_coefficients_and_stored_transform():
    rows, cols = np.indices((7, 9), dtype=float)
    x, y = cols / 4 - 1, rows / 3 - 1
    source = 2 + 3 * x - 4 * y + 0.5 * x * x + 2 * x * y - 0.3 * y * y
    surface = fitting().fit_surface2d(source, degree=2)
    assert surface.exponents == ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    np.testing.assert_allclose(
        surface.coefficients, [2, 3, -4, 0.5, 2, -0.3], atol=1e-10
    )
    assert surface.rank.item() == 6
    assert surface.n_observations.item() == 63
    assert surface.row_offset == 3 and surface.row_scale == 1 / 3
    assert surface.column_offset == 4 and surface.column_scale == 1 / 4
    np.testing.assert_allclose(surface.evaluate(rows, cols), source, atol=1e-10)
    query_rows, query_cols = np.array([[-2.0], [10.0]]), np.array([0.0, 3.0, 11.0])
    expected = (
        2
        + 3 * (query_cols / 4 - 1)
        - 4 * (query_rows / 3 - 1)
        + 0.5 * (query_cols / 4 - 1) ** 2
        + 2 * (query_cols / 4 - 1) * (query_rows / 3 - 1)
        - 0.3 * (query_rows / 3 - 1) ** 2
    )
    np.testing.assert_allclose(
        surface.evaluate(query_rows, query_cols), expected, atol=1e-10
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_batched_surface_native_masks_nonfinite_valid_and_custom_basis(weighted):
    rng = np.random.default_rng(87)
    source = np.ma.array(rng.normal(size=(2, 3, 4, 5)), mask=False)
    source.mask[0, 0, 0, 0] = True
    source[1, 0, 1, 2] = np.inf
    source[0, 1, 2, 3] = np.nan
    valid = np.ma.array(np.ones((4, 5), dtype=bool), mask=False)
    valid[1, 1] = False
    valid.mask[2, 0] = True
    weights = np.ma.array(rng.uniform(0.1, 2, (4, 5)), mask=False) if weighted else None
    if weighted:
        weights.mask[0, 1] = True
        weights[3, 3] = 0
    exponents = ((0, 0), (2, 0), (0, 2), (1, 1))
    before = source.copy()
    surface = fitting().fit_surface2d(
        source,
        exponents=exponents,
        valid=valid,
        weights=weights,
        method="wls" if weighted else "ols",
    )
    expected = reference_fit(source, exponents, valid=valid, weights=weights)
    for name, value in zip(
        (
            "coefficients",
            "rank",
            "n_observations",
            "is_constrained",
            "condition_number",
        ),
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(
            getattr(surface, name), value, atol=1e-10, rtol=1e-12
        )
    assert surface.coefficients.shape == (2, 3, 4)
    rows, cols = np.indices((4, 5), dtype=float)
    assert surface.evaluate(rows, cols).shape == (2, 3, 4, 5)
    np.testing.assert_equal(source.data, before.data)
    np.testing.assert_equal(source.mask, before.mask)


def test_singleton_axis_no_observations_and_rank_deficient_fit():
    surface = fitting().fit_surface2d(np.array([[2.0, 3.0, 4.0]]))
    assert surface.rank.item() == 2
    assert surface.row_offset == 0 and surface.row_scale == 0
    assert surface.is_constrained.tolist() == [True, True, False]
    rows, cols = np.indices((1, 3), dtype=float)
    np.testing.assert_allclose(surface.evaluate(rows, cols), [[2, 3, 4]], atol=1e-10)
    empty = fitting().fit_surface2d(np.full((2, 3), np.nan))
    assert empty.rank.item() == 0
    assert empty.n_observations.item() == 0
    assert torch.isnan(empty.evaluate(rows, cols)).all()


def test_surface_query_masks_and_invalid_coordinates_remain_pointwise():
    surface = fitting().fit_surface2d(np.ones((3, 4)))
    rows = np.ma.array([0.0, np.inf, 1.0], mask=[True, False, False])
    result = surface.evaluate(rows, np.zeros(3))
    assert torch.isnan(result[:2]).all()
    assert result[2].item() == pytest.approx(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"degree": True},
        {"degree": -1},
        {"degree": 1.5},
        {"exponents": []},
        {"exponents": [(0, 0), (0, 0)]},
        {"exponents": [(-1, 0)]},
        {"exponents": [(0.5, 1)]},
        {"exponents": [(True, 0)]},
        {"exponents": [(0, 0, 0)]},
        {"exponents": [(0, 0)], "degree": 2},
        {"weights": np.ones((3, 4))},
        {"method": "wls"},
        {"valid": np.ones((3, 1), dtype=bool)},
        {"degree": 10**8},
    ],
)
def test_surface_options_shape_and_preflight_validation(kwargs):
    with pytest.raises((TypeError, ValueError, MemoryError)):
        fitting().fit_surface2d(np.ones((3, 4)), **kwargs)


def test_surface_empty_leading_batch_and_noncontiguous_float32():
    empty = fitting().fit_surface2d(np.empty((0, 3, 4)))
    assert empty.coefficients.shape == (0, 3)
    source = np.arange(48, dtype=np.float32).reshape(6, 8)[::2, ::2]
    surface = fitting().fit_surface2d(source)
    assert surface.coefficients.dtype == torch.float64
    rows, cols = np.indices(source.shape, dtype=float)
    np.testing.assert_allclose(surface.evaluate(rows, cols), source, atol=1e-10)


@pytest.mark.parametrize("batch_size", [7, 65536])
def test_chunked_queries_and_image_slices_match_reference(monkeypatch, batch_size):
    module = importlib.import_module("faninsar.core.algorithms.fitting._polynomial")
    monkeypatch.setattr(module, "_QUERY_BATCH", batch_size)
    source = (
        np.arange(2 * 3 * 4 * 5.0, dtype=float)
        .reshape(2, 3, 4, 5)
        .transpose(1, 0, 3, 2)
    )
    surface = fitting().fit_surface2d(source)
    rows = np.ma.array(np.arange(5.0, dtype=float)[:, None], mask=False)
    rows.mask[2, 0] = True
    cols = np.arange(4.0, dtype=float)
    expected = np.array(source)
    expected[..., 2, :] = np.nan
    np.testing.assert_allclose(
        surface.evaluate(rows, cols), expected, atol=1e-10, rtol=1e-12
    )
    x = np.arange(35.0, dtype=float).reshape(5, 7).T
    c = np.array([1.0, 2.0, -0.5])
    np.testing.assert_allclose(
        fitting().polyval(c, x), reference_polyval(c, x), atol=1e-10, rtol=1e-12
    )
    empty = fitting().fit_surface2d(np.empty((0, 3, 4)))
    assert empty.evaluate(np.empty(0), np.empty(0)).shape == (0, 0)


def test_surface_invalid_shared_weights_rejected_even_for_empty_image_batch():
    with pytest.raises(ValueError, match="weights"):
        fitting().fit_surface2d(
            np.empty((0, 3, 4)), method="wls", weights=np.full((3, 4), -1.0)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_polynomials_and_surface_stay_on_device():
    source = torch.ones((4, 5), device="cuda")
    surface = fitting().fit_surface2d(source)
    rows, cols = torch.zeros(2, device="cuda"), np.zeros(2)
    assert surface.evaluate(rows, cols).device == source.device
    assert (
        fitting().polyval(torch.ones(2, device="cuda"), np.zeros(2)).device
        == source.device
    )
