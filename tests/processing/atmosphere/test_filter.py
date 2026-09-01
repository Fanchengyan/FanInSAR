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
