"""Whole-image oracle for selected-field interferogram formation (PROPOSAL-0074)."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.interferometry import form_interferogram_from_fields
from faninsar.processing.interferometry.pair import form_interferogram


def _fields(
    cross: np.ndarray,
    valid: np.ndarray,
    *,
    power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(cross, dtype=np.complex64)
    mask = np.asarray(valid, dtype=bool)
    primary = np.where(mask, np.float32(power), np.float32(0.0))
    secondary = np.where(mask, np.float32(power), np.float32(0.0))
    return samples, primary, secondary, mask


def test_direct_mle_cancellation_keeps_supported_zero_look() -> None:
    """A two-sample look of +1 and -1 is 0j with coherence 0, not nodata."""
    cross = np.zeros((2, 2), dtype=np.complex64)
    valid = np.zeros((2, 2), dtype=bool)
    cross[0, 0] = 1 + 0j
    cross[0, 1] = -1 + 0j
    valid[0, 0] = True
    valid[0, 1] = True
    product = form_interferogram_from_fields(
        *_fields(cross, valid),
        native_origin=(0, 0),
        output_window=(0, 2, 0, 2),
        multilook=(2, 2),
        coherence_window=None,
    )
    assert product.valid_mask is not None
    assert bool(product.valid_mask[0, 0])
    assert product.complex[0, 0] == pytest.approx(0 + 0j)
    assert product.coherence[0, 0] == pytest.approx(0.0)


def test_direct_mle_singleton_keeps_complex_and_nan_coherence() -> None:
    """One valid sample supports the IFG but not a two-sample MLE."""
    cross = np.zeros((2, 2), dtype=np.complex64)
    valid = np.zeros((2, 2), dtype=bool)
    cross[0, 0] = 2 + 0j
    valid[0, 0] = True
    product = form_interferogram_from_fields(
        *_fields(cross, valid, power=4.0),
        native_origin=(0, 0),
        output_window=(0, 2, 0, 2),
        multilook=(2, 2),
        coherence_window=None,
    )
    assert product.valid_mask is not None
    assert bool(product.valid_mask[0, 0])
    assert product.complex[0, 0] == pytest.approx(2 + 0j)
    assert np.isnan(product.coherence[0, 0])


def test_roi_hole_is_excluded_from_look_mean() -> None:
    """Excluded hole samples cannot pull a two-sample look toward the bbox mean."""
    cross = np.full((2, 2), 1 + 0j, dtype=np.complex64)
    valid = np.ones((2, 2), dtype=bool)
    cross[0, 1] = 1000 + 0j
    cross[1, 1] = -1000 + 0j
    valid[0, 1] = False
    valid[1, 1] = False
    product = form_interferogram_from_fields(
        *_fields(cross, valid),
        native_origin=(0, 0),
        output_window=(0, 2, 0, 2),
        multilook=(2, 2),
        coherence_window=None,
    )
    assert product.complex[0, 0] == pytest.approx(1 + 0j)
    assert product.coherence[0, 0] == pytest.approx(1.0)


def test_global_look_origin_drops_partial_blocks() -> None:
    """Looks are aligned to native (0, 0), not the output-window origin."""
    cross = np.ones((5, 5), dtype=np.complex64)
    valid = np.ones((5, 5), dtype=bool)
    cross[0, 0] = 3 + 0j
    product = form_interferogram_from_fields(
        *_fields(cross, valid),
        native_origin=(0, 0),
        output_window=(1, 5, 1, 5),
        multilook=(2, 2),
        coherence_window=None,
    )
    # first complete look inside [1, 5) starts at native 2; [4, 6) is dropped
    assert product.complex.shape == (1, 1)
    assert product.complex[0, 0] == pytest.approx(1 + 0j)


def test_tiled_direct_mle_matches_full_image_oracle() -> None:
    """Bounded tiles reproduce the independent full-image look field."""
    rng = np.random.default_rng(74)
    height, width = 8, 8
    noise = rng.normal(size=(height, width)) + 1j * rng.normal(size=(height, width))
    cross = noise.astype(np.complex64)
    valid = np.ones((height, width), dtype=bool)
    valid[3, 3] = False
    valid[4, 5] = False
    primary = np.abs(cross).astype(np.float32) ** 2 + 0.25
    secondary = np.abs(cross).astype(np.float32) ** 2 + 0.5
    oracle = form_interferogram_from_fields(
        cross,
        primary,
        secondary,
        valid,
        native_origin=(0, 0),
        output_window=(0, height, 0, width),
        multilook=(2, 2),
        coherence_window=None,
    )
    tile = form_interferogram_from_fields(
        cross[2:8, 0:6],
        primary[2:8, 0:6],
        secondary[2:8, 0:6],
        valid[2:8, 0:6],
        native_origin=(2, 0),
        output_window=(2, 8, 0, 6),
        multilook=(2, 2),
        coherence_window=None,
    )
    np.testing.assert_allclose(tile.complex, oracle.complex[1:4, 0:3], atol=1e-6)
    np.testing.assert_array_equal(tile.valid_mask, oracle.valid_mask[1:4, 0:3])
    np.testing.assert_allclose(
        tile.coherence, oracle.coherence[1:4, 0:3], equal_nan=True
    )


def test_sliding_then_look_mean_matches_full_image_with_halo() -> None:
    """Optional sliding MLE is computed on native samples, then look-averaged."""
    rng = np.random.default_rng(7401)
    height, width = 9, 9
    noise = rng.normal(size=(height, width)) + 1j * rng.normal(size=(height, width))
    cross = noise.astype(np.complex64)
    valid = np.ones((height, width), dtype=bool)
    valid[1, 1] = False
    primary = np.full((height, width), 2.0, dtype=np.float32)
    secondary = np.full((height, width), 3.0, dtype=np.float32)
    oracle = form_interferogram_from_fields(
        cross,
        primary,
        secondary,
        valid,
        native_origin=(0, 0),
        output_window=(0, height, 0, width),
        multilook=(3, 3),
        coherence_window=(3, 3),
    )
    # Halo of 1 pixel around looks covering native [3, 6) so the 3x3 window fits.
    tile = form_interferogram_from_fields(
        cross[2:7, 2:7],
        primary[2:7, 2:7],
        secondary[2:7, 2:7],
        valid[2:7, 2:7],
        native_origin=(2, 2),
        output_window=(3, 6, 3, 6),
        multilook=(3, 3),
        coherence_window=(3, 3),
    )
    assert tile.complex.shape == (1, 1)
    np.testing.assert_allclose(tile.complex, oracle.complex[1:2, 1:2], atol=1e-5)
    np.testing.assert_allclose(
        tile.coherence, oracle.coherence[1:2, 1:2], equal_nan=True, atol=1e-5
    )


def test_form_interferogram_delegates_to_selected_fields() -> None:
    """Legacy SLC entry stays numerically equal to the selected-field path."""
    rng = np.random.default_rng(38)
    primary = (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))).astype(
        np.complex64
    )
    secondary = (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))).astype(
        np.complex64
    )
    from_slc = form_interferogram(primary, secondary, multilook=(2, 2))
    valid = np.isfinite(primary.real) & np.isfinite(secondary.real)
    from_fields = form_interferogram_from_fields(
        primary * np.conjugate(secondary),
        (np.abs(primary) ** 2).astype(np.float32),
        (np.abs(secondary) ** 2).astype(np.float32),
        valid,
        native_origin=(0, 0),
        output_window=(0, 6, 0, 6),
        multilook=(2, 2),
        coherence_window=None,
    )
    np.testing.assert_allclose(from_slc.complex, from_fields.complex, atol=1e-5)
    np.testing.assert_array_equal(from_slc.valid_mask, from_fields.valid_mask)


def test_insufficient_input_coverage_is_rejected() -> None:
    """A tile that does not contain a requested look block is not silently padded."""
    cross = np.ones((2, 2), dtype=np.complex64)
    valid = np.ones((2, 2), dtype=bool)
    with pytest.raises(InvalidProcessingStateError, match="coverage"):
        form_interferogram_from_fields(
            *_fields(cross, valid),
            native_origin=(2, 0),
            output_window=(0, 4, 0, 2),
            multilook=(2, 2),
        )
