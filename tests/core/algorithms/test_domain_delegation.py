"""P0087 domain compatibility and explicitly deferred promotion boundaries."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from faninsar.core import algorithms
from faninsar.processing.atmosphere.filter import smooth_inverse_variance
from faninsar.processing.runtime.torch_kernels import multilook_real_torch


@pytest.mark.parametrize("sigma", [0.1, 0.16, 2.0])
def test_explicit_minimum_one_matches_legacy_smoothing(sigma: float) -> None:
    """Core is compatible only with explicit legacy support and weight clamping."""
    values = torch.arange(63, dtype=torch.float64).reshape(7, 9)
    values[3, 4] = torch.nan
    values[1, 2] = torch.inf
    weights = torch.ones_like(values)
    weights[0, 0] = -1.0
    weights[-1] = 0.0
    before_values, before_weights = values.clone(), weights.clone()
    expected = smooth_inverse_variance(values, weights, sigma_y=sigma, sigma_x=sigma)
    actual, valid = algorithms.filter2d(
        values,
        method="gaussian",
        sigma=(sigma, sigma),
        min_radius=1,
        boundary="constant",
        weights=weights.clamp(min=0.0),
        return_valid=True,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-10, equal_nan=True)
    assert actual.dtype == expected.dtype == torch.float64
    assert torch.equal(valid, ~torch.isnan(expected))
    assert valid[3, 4]
    torch.testing.assert_close(values, before_values, equal_nan=True)
    torch.testing.assert_close(weights, before_weights, equal_nan=True)


@pytest.mark.parametrize(
    ("dtype", "requires_grad", "sigma_y", "sigma_x", "bad_weight"),
    [
        (torch.float64, False, 2.0, 3.0, False),
        (torch.float32, False, 2.0, 3.0, False),
        (torch.float64, True, 2.0, 3.0, False),
        (torch.float64, False, 0.0, 0.0, False),
        (torch.float64, False, 0.0, 1.0, False),
        (torch.float64, False, -1.0, 1.0, False),
        (torch.float64, False, 1.0, 1.0, True),
    ],
)
def test_smoothing_delegates_shared_kernel_preserving_legacy_profile(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    requires_grad: bool,
    sigma_y: float,
    sigma_x: float,
    bad_weight: bool,
) -> None:
    """Domain admission preserves legacy profiles before shared arithmetic."""
    from faninsar.core.algorithms.filtering import _local

    calls = []
    original = _local._normalized_convolution

    def observed(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_local, "_normalized_convolution", observed)
    values = torch.arange(35, dtype=dtype).reshape(5, 7).requires_grad_(requires_grad)
    weights = torch.ones_like(values)
    if bad_weight:
        weights[2, 3] = torch.inf
    actual = smooth_inverse_variance(values, weights, sigma_y=sigma_y, sigma_x=sigma_x)
    assert actual.dtype == dtype
    assert actual.requires_grad == requires_grad
    assert calls


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nonfinite", [None, np.nan, np.inf, -np.inf])
def test_multilook_delegates_shared_arithmetic_preserving_legacy_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type,
    nonfinite: float | None,
) -> None:
    """Shared arithmetic must retain normal-mean Inf rather than strict admission."""
    from faninsar.core.algorithms.reduction import _block

    calls = []
    original = _block._complete_block_mean

    def observed(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_block, "_complete_block_mean", observed)
    array = np.arange(48, dtype=dtype).reshape(6, 8)
    if nonfinite is not None:
        array[0, 0] = nonfinite
    expected = torch.as_tensor(array, dtype=torch.float64).reshape(3, 2, 2, 4)
    expected = expected.mean(dim=(1, 3)).numpy().astype(dtype)
    actual = multilook_real_torch(array, 2, 4, device="cpu")
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
    assert calls


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_core_propagation_is_not_legacy_infinite_mean(value: float) -> None:
    """Nonfinite block profiles cannot be delegated as finite-only core input."""
    source = torch.tensor([[1.0, value], [2.0, 3.0]], dtype=torch.float64)
    core = algorithms.block_reduce2d(
        source, (2, 2), method="mean", invalid="propagate", remainder="drop"
    )
    assert torch.isnan(core).all()
    if np.isinf(value):
        assert torch.isinf(source.mean())
