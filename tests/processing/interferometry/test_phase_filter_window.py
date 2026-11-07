"""Bounded Goldstein-Werner windows against a full-image oracle (PROPOSAL-0074)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.interferometry import (
    BoxcarFilter,
    GoldsteinWerner,
    PhaseFilter,
)
from faninsar.processing.interferometry.phase_filter import PhaseFilterResult


def _complex_image(shape: tuple[int, int], seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    array = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )
    return torch.from_numpy(array)


def test_apply_window_on_full_image_matches_apply() -> None:
    """A whole-image window is the existing apply contract."""
    image = _complex_image((64, 64), 7402)
    filt = GoldsteinWerner(alpha=0.5, patch_size=16)
    full = filt.apply(image)
    windowed = filt.apply_window(
        image,
        input_window=(0, 64, 0, 64),
        output_window=(0, 64, 0, 64),
        image_shape=(64, 64),
    )
    torch.testing.assert_close(
        windowed.interferogram, full.interferogram, rtol=3e-5, atol=3e-5
    )
    assert torch.equal(windowed.valid_mask, full.valid_mask)


def test_tiled_apply_window_matches_full_image_oracle() -> None:
    """Global patch origins plus halo reproduce the cropped full-image filter."""
    image = _complex_image((64, 64), 7403)
    valid = torch.ones((64, 64), dtype=torch.bool)
    valid[20, 20] = False
    valid[40, 33] = False
    image = image.clone()
    image[~valid] = torch.nan + 0j
    filt = GoldsteinWerner(alpha=0.4, patch_size=16)
    oracle = filt.apply(image, valid_mask=valid)
    output_window = (16, 48, 16, 48)
    input_window = (8, 56, 8, 56)
    tile = image[8:56, 8:56]
    tile_mask = valid[8:56, 8:56]
    windowed = filt.apply_window(
        tile,
        valid_mask=tile_mask,
        input_window=input_window,
        output_window=output_window,
        image_shape=(64, 64),
    )
    expected = oracle.interferogram[16:48, 16:48]
    expected_mask = oracle.valid_mask[16:48, 16:48]
    torch.testing.assert_close(windowed.interferogram, expected, rtol=3e-5, atol=3e-5)
    assert torch.equal(windowed.valid_mask, expected_mask)
    assert not bool(windowed.valid_mask[4, 4])


def test_small_image_bypass_uses_logical_image_shape() -> None:
    """Bypass is decided by the full image, not the tile size."""
    image = _complex_image((16, 16), 7404)
    filt = GoldsteinWerner(alpha=0.5, patch_size=32)
    result = filt.apply_window(
        image[2:10, 2:10],
        input_window=(2, 10, 2, 10),
        output_window=(4, 8, 4, 8),
        image_shape=(16, 16),
    )
    torch.testing.assert_close(result.interferogram, image[4:8, 4:8])


def test_apply_window_rejects_insufficient_halo() -> None:
    """A tile missing a global patch that overlaps the output is rejected."""
    image = _complex_image((64, 64), 7405)
    filt = GoldsteinWerner(alpha=0.5, patch_size=16)
    with pytest.raises(InvalidProcessingStateError, match="halo"):
        filt.apply_window(
            image[16:48, 16:48],
            input_window=(16, 48, 16, 48),
            output_window=(16, 48, 16, 48),
            image_shape=(64, 64),
        )


def test_custom_phase_filter_rejects_bounded_windows() -> None:
    """Unsupported custom filters must fail clearly rather than reset patches."""

    class _Custom(PhaseFilter):
        def apply(
            self,
            interferogram: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None = None,
        ) -> PhaseFilterResult:
            support = interferogram.real == interferogram.real
            mask = support if valid_mask is None else valid_mask
            return PhaseFilterResult(interferogram, mask)

    filt = _Custom()
    image = _complex_image((16, 16), 7406)
    with pytest.raises(InvalidProcessingStateError, match="bounded"):
        filt.apply_window(
            image,
            input_window=(0, 16, 0, 16),
            output_window=(0, 16, 0, 16),
            image_shape=(16, 16),
        )
    boxcar = BoxcarFilter(window=(3, 3))
    with pytest.raises(InvalidProcessingStateError, match="bounded"):
        boxcar.apply_window(
            image,
            input_window=(0, 16, 0, 16),
            output_window=(0, 16, 0, 16),
            image_shape=(16, 16),
        )
