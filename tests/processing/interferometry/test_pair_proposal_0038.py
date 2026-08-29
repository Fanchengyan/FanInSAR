"""Public formation and phase-filter contracts for Proposal-0038."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    _Factory = Callable[[], object]

from faninsar.processing.interferometry import (
    BoxcarFilter,
    GaussianFilter,
    GoldsteinWerner,
    form_interferogram,
    validate_coherence_window,
)


@pytest.mark.parametrize("value", [(), (1, 5), (2, 5), (4, 5), (3, 0), (3, -1)])
def test_coherence_window_rejects_non_centered_support(value: tuple[int, ...]) -> None:
    """A centered support has two odd axes and each axis is at least 3."""
    with pytest.raises(ValueError, match="odd integers >= 3"):
        validate_coherence_window(value)  # type: ignore[arg-type]


def test_coherence_window_none_is_a_valid_direct_mle_branch() -> None:
    """None computes coherence directly inside each output look block."""
    primary = np.ones((2, 3), dtype=np.complex64)
    secondary = np.ones((2, 3), dtype=np.complex64)
    primary[0, 1] = np.nan + 1j * np.nan
    product = form_interferogram(
        primary,
        secondary,
        multilook=(2, 2),
        coherence_window=None,
    )
    assert product.complex_ifg.shape == (1, 2)
    assert product.valid_mask is not None
    assert product.valid_mask.tolist() == [[True, True]]
    assert np.allclose(product.coherence, 1.0, equal_nan=False)


def test_multilook_retains_partial_tail_with_actual_sample_count() -> None:
    """Output blocks start at zero, retain tails, and divide by valid counts."""
    primary = np.ones((3, 5), dtype=np.complex64)
    secondary = np.ones((3, 5), dtype=np.complex64)
    primary[2, 4] = 3.0 + 0.0j
    product = form_interferogram(
        primary,
        secondary,
        multilook=(2, 2),
        coherence_window=None,
    )
    assert product.complex_ifg.shape == (2, 3)
    assert product.complex_ifg[1, 2] == pytest.approx(3.0 + 0.0j)


def test_direct_mle_marks_singleton_coherence_invalid_but_keeps_complex_support(
) -> None:
    """One valid sample supports an IFG, but not a two-sample MLE coherence."""
    primary = np.full((2, 2), np.nan + 1j * np.nan, dtype=np.complex64)
    secondary = np.full((2, 2), np.nan + 1j * np.nan, dtype=np.complex64)
    primary[0, 0] = 2.0 + 0.0j
    secondary[0, 0] = 1.0 + 0.0j

    product = form_interferogram(
        primary,
        secondary,
        multilook=(2, 2),
        coherence_window=None,
    )

    assert product.valid_mask is not None
    assert bool(product.valid_mask[0, 0])
    assert product.complex_ifg[0, 0] == pytest.approx(2.0 + 0.0j)
    assert np.isnan(product.coherence[0, 0])


def test_two_stage_coherence_uses_full_resolution_sliding_mle() -> None:
    """Tuple windows form HxW gamma first, then average gamma by blocks."""
    primary = np.ones((3, 3), dtype=np.complex64)
    secondary = np.ones((3, 3), dtype=np.complex64)
    primary[0, 0] = 1.0 + 1.0j
    product = form_interferogram(
        primary,
        secondary,
        multilook=(2, 2),
        coherence_window=(3, 3),
    )
    assert product.coherence.shape == (2, 2)
    assert product.coherence[0, 0] < 1.0
    assert product.coherence[1, 1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GoldsteinWerner(),
        lambda: BoxcarFilter(window=(3, 3)),
        lambda: GaussianFilter(sigma=(1.0, 1.0)),
    ],
)
def test_phase_filter_preserves_shape_device_and_support(
    factory: _Factory,
) -> None:
    """Built-in filters preserve the tensor contract and cannot resurrect holes."""
    value = torch.ones((9, 11), dtype=torch.complex64)
    valid = torch.ones((9, 11), dtype=torch.bool)
    valid[0, 0] = False
    value[0, 0] = torch.nan + 0j
    result = factory().apply(value, valid_mask=valid)
    assert result.interferogram.shape == value.shape
    assert result.interferogram.device == value.device
    assert result.valid_mask.device == value.device
    assert result.valid_mask.dtype is torch.bool
    assert not bool(result.valid_mask[0, 0])
    assert torch.isfinite(result.interferogram[result.valid_mask]).all()


def test_goldstein_patch_size_is_scalar_even_and_bounded() -> None:
    """Goldstein-Werner rejects tuple patches and out-of-range values."""
    with pytest.raises(ValueError, match="patch_size"):
        GoldsteinWerner(patch_size=(32, 32))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="patch_size"):
        GoldsteinWerner(patch_size=7)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GoldsteinWerner(alpha=True),
        lambda: GaussianFilter(sigma=(True, 1.0)),
        lambda: GaussianFilter(sigma=(1.0, 1.0), truncate=True),
    ],
)
def test_filter_scalar_parameters_reject_bool(factory: _Factory) -> None:
    """Boolean values must not silently act as numeric filter parameters."""
    with pytest.raises(ValueError, match="finite"):
        factory()
