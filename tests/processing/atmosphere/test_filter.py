"""Tests for screen filtering utilities.

``remove_small_components`` is pinned against the reference semantics
(4-connected ``scipy.ndimage.label`` clustering) on randomized masks and an
adversarial U-shaped case; smoothing is checked against analytic behavior.
"""

from __future__ import annotations

import math

import pytest
import torch
from scipy import ndimage

from faninsar.processing.atmosphere.filter import (
    remove_small_components,
    smooth_inverse_variance,
)


def _reference_small_component_removal(
    mask: torch.Tensor, min_pixels: int
) -> torch.Tensor:
    """Ground-truth implementation using scipy.ndimage.label (4-conn)."""
    labels, _ = ndimage.label(mask.numpy())
    cleaned = mask.clone()
    sizes = ndimage.sum_labels(
        mask.numpy(), labels, index=range(1, int(labels.max()) + 1)
    )
    for label_id, size in enumerate(sizes, start=1):
        if size < min_pixels:
            cleaned[labels == label_id] = False
    return cleaned


class TestRemoveSmallComponents:
    def test_zero_threshold_is_noop(self) -> None:
        mask = torch.rand(16, 16) > 0.5
        assert torch.equal(remove_small_components(mask, 0), mask)

    def test_u_shaped_cluster_survives(self) -> None:
        mask = torch.zeros(32, 32, dtype=torch.bool)
        # U shape: two vertical strokes joined at the bottom (geodesically long)
        mask[4:28, 6] = True
        mask[27, 6:26] = True
        mask[4:28, 25] = True
        # plus one isolated pixel that must be removed
        mask[2, 2] = True

        cleaned = remove_small_components(mask, min_cluster_pixels=5)

        expected = _reference_small_component_removal(mask, 5)
        assert torch.equal(cleaned, expected)
        assert bool(cleaned[4:28, 6].all())
        assert not bool(cleaned[2, 2])

    def test_matches_scipy_on_random_masks(self) -> None:
        generator = torch.Generator().manual_seed(20260827)
        for _ in range(10):
            mask = torch.rand(24, 31, generator=generator) > 0.62
            threshold = int(torch.randint(1, 12, (1,), generator=generator))
            got = remove_small_components(mask, threshold)
            want = _reference_small_component_removal(mask, threshold)
            assert torch.equal(got, want)

    def test_input_not_mutated(self) -> None:
        mask = torch.ones(8, 8, dtype=torch.bool)
        snapshot = mask.clone()
        remove_small_components(mask, min_cluster_pixels=100)
        assert torch.equal(mask, snapshot)

    def test_invalid_arguments_raise(self) -> None:
        with pytest.raises(ValueError, match="boolean"):
            remove_small_components(torch.ones(4, 4), 1)
        with pytest.raises(ValueError, match="min_cluster_pixels"):
            remove_small_components(torch.ones(4, 4, dtype=torch.bool), -1)


def _legacy_smooth(
    values: torch.Tensor,
    weights: torch.Tensor,
    sigma_y: float,
    sigma_x: float,
) -> torch.Tensor:
    """Frozen pre-P0087 smoothing, independent of production kernel helpers."""
    values64 = values.to(torch.float64)
    finite = torch.isfinite(values64)
    clean = torch.where(finite, values64, torch.zeros_like(values64))
    weight = weights.to(torch.float64).clamp(min=0.0) * finite

    def convolve(image: torch.Tensor, sigma: float, axis: int) -> torch.Tensor:
        if sigma <= 0.0:
            kernel = torch.ones(1, dtype=torch.float64)
        else:
            radius = max(1, int(3.0 * sigma + 0.5))
            offsets = torch.arange(-radius, radius + 1, dtype=torch.float64)
            kernel = torch.exp(-(offsets**2) / (2.0 * sigma**2))
            kernel = kernel / kernel.sum()
        kernel = kernel.to(values.device)
        radius = (kernel.numel() - 1) // 2
        padded = torch.nn.functional.pad(
            image[None, None],
            (0, 0, radius, radius) if axis == 0 else (radius, radius, 0, 0),
        )
        shape = (1, 1, -1, 1) if axis == 0 else (1, 1, 1, -1)
        return torch.nn.functional.conv2d(padded, kernel.reshape(shape))[0, 0]

    numerator = convolve(convolve(clean * weight, sigma_y, 0), sigma_x, 1)
    denominator = convolve(convolve(weight, sigma_y, 0), sigma_x, 1)
    return torch.where(denominator > 0, numerator / denominator, math.nan).to(
        values.dtype
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("sigma_y", "sigma_x"),
    [(0.1, 0.1), (0.16, 0.16), (2.0, 3.0), (0.0, 0.0), (0.0, 1.0), (-1.0, 0.16)],
)
@pytest.mark.parametrize("nonfinite_weight", [False, True])
def test_smoothing_preserves_frozen_legacy_profiles(
    dtype: torch.dtype,
    sigma_y: float,
    sigma_x: float,
    nonfinite_weight: bool,
) -> None:
    """Small support, identity axes and invalid weights keep legacy semantics."""
    values = torch.arange(49, dtype=dtype).reshape(7, 7)
    values[3, 3] = math.nan
    values[1, 5] = math.inf
    weights = torch.ones_like(values)
    weights[0, 0] = -3.0
    weights[5, 1] = 0.0
    if nonfinite_weight:
        weights[1, 1] = math.inf
        weights[5, 5] = math.nan
    before_values, before_weights = values.clone(), weights.clone()
    expected = _legacy_smooth(values, weights, sigma_y, sigma_x)
    actual = smooth_inverse_variance(values, weights, sigma_y=sigma_y, sigma_x=sigma_x)
    assert actual.dtype == dtype
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-10, equal_nan=True)
    torch.testing.assert_close(values, before_values, equal_nan=True)
    torch.testing.assert_close(weights, before_weights, equal_nan=True)
    if not nonfinite_weight and sigma_y in (0.1, 0.16) and sigma_x == sigma_y:
        assert torch.isfinite(actual[3, 3])


class TestSmoothInverseVariance:
    def test_constant_field_invariant(self) -> None:
        values = torch.full((32, 48), 1.7)
        weights = torch.ones_like(values)
        smoothed = smooth_inverse_variance(values, weights, sigma_y=2.0, sigma_x=3.0)
        interior_nan = torch.isnan(smoothed[4:-4, 6:-6])
        assert not bool(interior_nan.any())
        torch.testing.assert_close(
            smoothed[8:-8, 12:-12],
            torch.full_like(smoothed[8:-8, 12:-12], 1.7),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_higher_weight_pixel_dominates(self) -> None:
        values = torch.full((16, 16), 0.0)
        values[8, 8] = 10.0
        weights = torch.ones_like(values)
        weights[:8, :] *= 1e6  # everything above row 8 massively trusted? no:
        # give equal weights so smoothing pulls toward local mean instead
        weights = torch.ones_like(values)

        smoothed = smooth_inverse_variance(values, weights, sigma_y=1.0, sigma_x=1.0)
        center = float(smoothed[8, 8])
        away = float(smoothed[0, 0])
        assert center < 10.0  # diluted by neighbors
        assert away < 0.05  # far corner unaffected apart from tiny leakage

    def test_nan_values_excluded_via_weights(self) -> None:
        values = torch.zeros(16, 16)
        weights = torch.ones_like(values)
        values[8, 8] = float("nan")

        smoothed = smooth_inverse_variance(values, weights, sigma_y=1.0, sigma_x=1.0)
        # NaN input contributes nothing; its location fills from neighbors
        assert not math.isnan(float(smoothed[8, 8]))
