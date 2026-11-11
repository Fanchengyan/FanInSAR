"""Public normalized-filter tests backed by explicit-radius SciPy oracles."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .reference.filtering import filter_reference


def test_scipy_reference_boxcar_has_analytical_weighted_mean():
    source = np.array([[1.0, np.nan, 5.0]])
    values, valid = filter_reference(
        source, method="boxcar", window=(1, 3), weights=np.array([[1.0, 0.0, 3.0]])
    )
    np.testing.assert_allclose(values, [[1.0, 4.0, 5.0]])
    assert valid.all()


def test_boxcar_reconstructs_missing_centers_with_weighted_neighbors():
    from faninsar.core.algorithms.filtering import filter2d

    source = np.array([[1.0, np.nan, 5.0]])
    actual, valid = filter2d(
        source,
        method="boxcar",
        window=(1, 3),
        weights=np.array([[1.0, 0.0, 3.0]]),
        return_valid=True,
    )
    torch.testing.assert_close(
        actual, torch.tensor([[1.0, 4.0, 5.0]], dtype=torch.float64)
    )
    assert valid.all()


@pytest.mark.parametrize("sigma", [(0.1, 0.16), (0.1, 1.2), (0.7, 1.3)])
@pytest.mark.parametrize("minimum", [None, 0, 1, 3])
@pytest.mark.parametrize("boundary", ["constant", "reflect"])
def test_gaussian_explicit_support_matches_scipy(sigma, minimum, boundary):
    from faninsar.core.algorithms import filter2d

    source = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5) / 10
    source[0, 2, 2] = np.nan
    source[1, 0, 0] = np.inf
    expected, supported = filter_reference(
        source, method="gaussian", sigma=sigma, min_radius=minimum, boundary=boundary
    )
    actual, actual_valid = filter2d(
        source,
        method="gaussian",
        sigma=sigma,
        min_radius=minimum,
        boundary=boundary,
        return_valid=True,
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), equal_nan=True, atol=1e-10, rtol=1e-12
    )
    np.testing.assert_array_equal(actual_valid, supported)


@pytest.mark.parametrize("sigma", [0.1, 0.16])
def test_gaussian_minimum_zero_is_single_tap_but_default_and_one_reconstruct(sigma):
    from faninsar.core.algorithms import filter2d

    source = np.array([[2.0, np.nan, 4.0]])
    zero, supported = filter2d(
        source, method="gaussian", sigma=(sigma, sigma), min_radius=0, return_valid=True
    )
    torch.testing.assert_close(zero, torch.from_numpy(source), equal_nan=True)
    assert not supported[0, 1]
    default = filter2d(source, method="gaussian", sigma=(sigma, sigma))
    one = filter2d(source, method="gaussian", sigma=(sigma, sigma), min_radius=1)
    torch.testing.assert_close(default, one)
    assert default[0, 1].item() == pytest.approx(3.0)
    assert torch.isfinite(default).all()


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1), (2, 2)])
@pytest.mark.parametrize("boundary", ["constant", "reflect"])
def test_arbitrary_padding_singleton_and_short_axes_match_mirror(shape, boundary):
    from faninsar.core.algorithms import filter2d

    source = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 2
    expected, support = filter_reference(
        source, method="boxcar", window=(9, 11), boundary=boundary
    )
    actual, valid = filter2d(
        source, method="boxcar", window=(9, 11), boundary=boundary, return_valid=True
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), atol=1e-10, rtol=1e-12
    )
    np.testing.assert_array_equal(valid, support)


def test_masked_weights_hidden_invalid_payloads_and_masked_valid_do_not_resurrect():
    from faninsar.core.algorithms import filter2d

    source = np.ma.array(np.arange(24, dtype=np.float32).reshape(2, 3, 4), mask=False)
    source.mask[0, 1, 1] = True
    source[1, 1, 2] = np.inf
    weights = np.ma.array(np.ones((3, 4)), mask=False)
    weights.mask[0, :2] = True
    weights.data[0, 0] = -1
    weights.data[0, 1] = np.nan
    weights[2, 3] = 0
    allowed = np.ma.array(np.ones(source.shape, dtype=bool), mask=False)
    allowed[1, 0, 0] = False
    allowed.mask[0, 2, 1] = True
    before = source.copy(), weights.copy(), allowed.copy()
    expected, support = filter_reference(
        source,
        method="gaussian",
        sigma=(0.7, 1.2),
        weights=weights,
        valid=allowed,
        boundary="reflect",
    )
    actual, valid = filter2d(
        source,
        method="gaussian",
        sigma=(0.7, 1.2),
        weights=weights,
        valid=allowed,
        boundary="reflect",
        return_valid=True,
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), equal_nan=True, atol=1e-10, rtol=1e-12
    )
    np.testing.assert_array_equal(valid, support)
    for current, previous in zip((source, weights, allowed), before, strict=True):
        np.testing.assert_array_equal(current.data, previous.data)
        np.testing.assert_array_equal(current.mask, previous.mask)


@pytest.mark.parametrize(
    "source,weights",
    [
        (np.full((2, 3), np.nan), None),
        (np.ma.masked_all((2, 3)), None),
        (np.ones((2, 3)), np.zeros((2, 3))),
        (np.ones((2, 3)), np.ma.masked_all((2, 3))),
    ],
)
def test_zero_support_is_nan_false(source, weights):
    from faninsar.core.algorithms import filter2d

    actual, valid = filter2d(
        source, method="boxcar", window=(3, 3), weights=weights, return_valid=True
    )
    assert torch.isnan(actual).all()
    assert not valid.any()


def test_numerically_zero_gaussian_neighbors_do_not_claim_reconstruction():
    from faninsar.core.algorithms import filter2d

    source = np.array([[2.0, np.nan, 4.0]])
    actual, valid = filter2d(
        source,
        method="gaussian",
        sigma=(1e-200, 1e-200),
        min_radius=1,
        return_valid=True,
    )
    torch.testing.assert_close(actual, torch.from_numpy(source), equal_nan=True)
    assert not valid[0, 1]


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0, 3), (2, 3, 0)])
def test_empty_filter_preserves_exact_shape(shape):
    from faninsar.core.algorithms import filter2d

    actual, valid = filter2d(
        np.empty(shape, dtype=np.float32),
        method="boxcar",
        window=(3, 3),
        return_valid=True,
    )
    assert actual.shape == shape
    assert valid.shape == shape
    assert actual.dtype == torch.float64
    assert valid.dtype == torch.bool


def test_noncontiguous_batched_filter_retains_true_image_halos_across_tiles():
    from faninsar.core.algorithms import filter2d

    source = (
        torch.arange(2 * 259 * 7, dtype=torch.float32)
        .reshape(2, 7, 259)
        .transpose(-1, -2)
    )
    before = source.clone()
    expected, support = filter_reference(
        source.numpy(), method="gaussian", sigma=(0.8, 1.2), boundary="reflect"
    )
    actual, valid = filter2d(
        source,
        method="gaussian",
        sigma=(0.8, 1.2),
        boundary="reflect",
        return_valid=True,
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), atol=1e-10, rtol=1e-12
    )
    np.testing.assert_array_equal(valid, support)
    torch.testing.assert_close(source, before)


@pytest.mark.parametrize(
    "options",
    [
        dict(method="median"),
        dict(method="boxcar"),
        dict(method="boxcar", window=(2, 3)),
        dict(method="boxcar", window=(True, 3)),
        dict(method="boxcar", window=(1.0, 3)),
        dict(method="boxcar", window=(3, 3), sigma=(1.0, 1.0)),
        dict(method="boxcar", window=(3, 3), min_radius=0),
        dict(method="gaussian"),
        dict(method="gaussian", sigma=(0.0, 1.0)),
        dict(method="gaussian", sigma=(-1.0, 1.0)),
        dict(method="gaussian", sigma=(np.inf, 1.0)),
        dict(method="gaussian", sigma=(np.nan, 1.0)),
        dict(method="gaussian", sigma=(True, 1.0)),
        dict(method="gaussian", sigma=(1.0, 1.0), window=(3, 3)),
        *[
            dict(method="gaussian", sigma=(1.0, 1.0), min_radius=value)
            for value in [True, -1, 0.5, np.inf, np.nan, (0, 1)]
        ],
        dict(method="boxcar", window=(3, 3), boundary="mirror"),
        dict(method="boxcar", window=(3, 3), valid=np.ones((3, 4))),
        dict(method="boxcar", window=(3, 3), valid=np.ones((1, 4), dtype=bool)),
        dict(method="boxcar", window=(3, 3), weights=np.ones((1, 4))),
        *[
            dict(method="boxcar", window=(3, 3), weights=np.full((3, 4), value))
            for value in [-1.0, np.inf, np.nan]
        ],
    ],
)
def test_unsupported_parameters_and_weights_rejected(options):
    from faninsar.core.algorithms import filter2d

    with pytest.raises((ValueError, TypeError)):
        filter2d(np.ones((3, 4)), **options)


@pytest.mark.parametrize(
    "options",
    [
        dict(method="gaussian", sigma=(1e300, 1.0)),
        dict(method="gaussian", sigma=(1.0, 1.0), min_radius=10**100),
        dict(method="boxcar", window=(10**100 + 1, 3)),
        dict(method="gaussian", sigma=(1.0, 1.0), min_radius=2048),
    ],
)
def test_absurd_derived_support_fails_before_arange_or_convolution(
    options, monkeypatch
):
    from faninsar.core.algorithms import filter2d

    def forbidden(*args, **kwargs):
        pytest.fail("support must be validated before tensor allocation/convolution")

    monkeypatch.setattr(torch, "arange", forbidden)
    monkeypatch.setattr(torch.nn.functional, "conv2d", forbidden)
    with pytest.raises(ValueError):
        filter2d(np.ones((3, 4)), **options)


@pytest.mark.parametrize(
    "source",
    [
        np.ones((2, 2), dtype=int),
        np.ones((2, 2), dtype=complex),
        np.ones((2, 2), dtype=bool),
        np.ones(3),
        [[1.0, 2.0]],
        torch.ones((2, 2), requires_grad=True),
        torch.ones((2, 2), dtype=torch.float16),
        torch.eye(2).to_sparse(),
        torch.empty((2, 2), device="meta"),
    ],
)
def test_unsupported_filter_inputs_fail_explicitly(source):
    from faninsar.core.algorithms import filter2d

    with pytest.raises((ValueError, TypeError)):
        filter2d(source, method="boxcar", window=(1, 1))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware unavailable; not qualified"
)
@pytest.mark.parametrize(
    "method,options",
    [
        ("boxcar", dict(window=(3, 5))),
        ("gaussian", dict(sigma=(0.1, 0.16), min_radius=0)),
        ("gaussian", dict(sigma=(0.1, 0.16), min_radius=1)),
        ("gaussian", dict(sigma=(0.8, 1.3))),
    ],
)
def test_cuda_filter_matches_cpu_oracle_without_fallback(method, options):
    from faninsar.core.algorithms import filter2d

    source = np.arange(30, dtype=np.float64).reshape(5, 6)
    source[1, 1] = np.nan
    expected, support = filter_reference(
        source, method=method, boundary="reflect", **options
    )
    actual, valid = filter2d(
        torch.from_numpy(source).cuda(),
        method=method,
        boundary="reflect",
        return_valid=True,
        **options,
    )
    assert actual.device.type == "cuda"
    assert valid.device == actual.device
    torch.testing.assert_close(
        actual.cpu(), torch.from_numpy(expected), equal_nan=True, atol=1e-10, rtol=1e-12
    )
    np.testing.assert_array_equal(valid.cpu(), support)
    with pytest.raises(ValueError, match="device"):
        filter2d(
            torch.from_numpy(source).cuda(),
            method=method,
            valid=torch.ones(source.shape, dtype=torch.bool),
            **options,
        )
