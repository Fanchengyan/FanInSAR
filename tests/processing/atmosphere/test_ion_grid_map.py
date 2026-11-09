"""Coordinate-aware masked bilinear ion mapping (PROPOSAL-0080 W1)."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.atmosphere.grid_map import (
    map_unwrapped_phase_bilinear,
    native_block_center_anchor,
)
from faninsar.processing.errors import InvalidProcessingStateError


def test_native_block_center_uses_integer_pixel_convention() -> None:
    """A block starting at native s with L looks is centered at s+(L-1)/2."""
    assert native_block_center_anchor((0, 0), (1, 1)) == (0.0, 0.0)
    assert native_block_center_anchor((0, 0), (2, 4)) == (0.5, 1.5)
    assert native_block_center_anchor((7, 3), (3, 5)) == (8.0, 5.0)


def test_identical_grid_is_exact_passthrough() -> None:
    """Matching anchors, spacing, and shape copy finite positive-quality samples."""
    phase = np.array([[0.10, 0.20], [0.30, np.nan]], dtype=np.float64)
    quality = np.array([[1.0, 0.5], [2.0, 1.0]], dtype=np.float64)
    mapped, valid = map_unwrapped_phase_bilinear(
        phase,
        quality,
        source_anchor=(0.5, 1.5),
        source_spacing=(2.0, 4.0),
        target_shape=(2, 2),
        target_anchor=(0.5, 1.5),
        target_spacing=(2.0, 4.0),
    )
    np.testing.assert_allclose(mapped[0, 0], 0.10)
    np.testing.assert_allclose(mapped[0, 1], 0.20)
    np.testing.assert_allclose(mapped[1, 0], 0.30)
    assert np.isnan(mapped[1, 1])
    np.testing.assert_array_equal(valid, np.array([[True, True], [True, False]]))


def test_same_shape_displaced_grid_is_not_passthrough() -> None:
    """Matching array shape does not establish grid compatibility."""
    phase = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float64)
    quality = np.ones((2, 2), dtype=np.float64)
    mapped, valid = map_unwrapped_phase_bilinear(
        phase,
        quality,
        source_anchor=(0.5, 0.5),
        source_spacing=(2.0, 2.0),
        target_shape=(2, 2),
        target_anchor=(2.5, 0.5),
        target_spacing=(2.0, 2.0),
    )
    np.testing.assert_allclose(mapped[0], [20.0, 30.0])
    assert np.all(np.isnan(mapped[1]))
    np.testing.assert_array_equal(valid, np.array([[True, True], [False, False]]))


def test_unequal_looks_bilinear_at_target_centers() -> None:
    """Target centers s+(L-1)/2 interpolate source samples without wrapping."""
    phase = np.array([[0.0, 4.0], [8.0, 12.0]], dtype=np.float64)
    quality = np.ones((2, 2), dtype=np.float64)
    mapped, valid = map_unwrapped_phase_bilinear(
        phase,
        quality,
        source_anchor=(0.5, 0.5),
        source_spacing=(2.0, 2.0),
        target_shape=(1, 1),
        target_anchor=(1.5, 1.5),
        target_spacing=(1.0, 1.0),
    )
    np.testing.assert_allclose(mapped, [[6.0]])
    assert valid[0, 0]


def test_invalid_neighbor_is_not_renormalized() -> None:
    """A hole in a contributing corner invalidates the output sample."""
    phase = np.array([[0.0, 4.0], [8.0, 12.0]], dtype=np.float64)
    quality = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    mapped, valid = map_unwrapped_phase_bilinear(
        phase,
        quality,
        source_anchor=(0.5, 0.5),
        source_spacing=(2.0, 2.0),
        target_shape=(1, 1),
        target_anchor=(1.5, 1.5),
        target_spacing=(1.0, 1.0),
    )
    assert np.isnan(mapped[0, 0])
    assert not valid[0, 0]


def test_zero_weight_neighbor_is_ignored() -> None:
    """A degenerate bilinear axis does not require the unused neighbor."""
    phase = np.array([[1.5, 9.0], [np.nan, np.nan]], dtype=np.float64)
    quality = np.array([[2.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    mapped, valid = map_unwrapped_phase_bilinear(
        phase,
        quality,
        source_anchor=(0.5, 0.5),
        source_spacing=(2.0, 2.0),
        target_shape=(1, 1),
        target_anchor=(0.5, 1.5),
        target_spacing=(2.0, 2.0),
    )
    np.testing.assert_allclose(mapped, [[5.25]])
    assert valid[0, 0]


def test_no_extrapolation_beyond_source_centers() -> None:
    """Target centers outside the source-center box stay invalid."""
    phase = np.ones((2, 2), dtype=np.float64)
    quality = np.ones((2, 2), dtype=np.float64)
    mapped, valid = map_unwrapped_phase_bilinear(
        phase,
        quality,
        source_anchor=(0.5, 0.5),
        source_spacing=(2.0, 2.0),
        target_shape=(1, 2),
        target_anchor=(-0.5, 0.5),
        target_spacing=(2.0, 4.0),
    )
    assert np.all(np.isnan(mapped))
    assert not np.any(valid)


def test_nonpositive_quality_is_unsupported() -> None:
    """Finite-positive source quality is required of every used neighbor."""
    phase = np.array([[0.25]], dtype=np.float64)
    quality = np.array([[0.0]], dtype=np.float64)
    mapped, valid = map_unwrapped_phase_bilinear(
        phase,
        quality,
        source_anchor=(0.0, 0.0),
        source_spacing=(1.0, 1.0),
        target_shape=(1, 1),
        target_anchor=(0.0, 0.0),
        target_spacing=(1.0, 1.0),
    )
    assert np.isnan(mapped[0, 0])
    assert not valid[0, 0]


def test_rejects_inconsistent_source_arrays() -> None:
    """Phase and quality must share one 2-D grid."""
    with pytest.raises(InvalidProcessingStateError):
        map_unwrapped_phase_bilinear(
            np.zeros((2, 2)),
            np.ones((2, 3)),
            source_anchor=(0.0, 0.0),
            source_spacing=(1.0, 1.0),
            target_shape=(2, 2),
            target_anchor=(0.0, 0.0),
            target_spacing=(1.0, 1.0),
        )
