"""Tests for overlap masks and feather weights via Euclidean distance transform."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.mosaicking.overlap import (
    apply_feather,
    compute_feather,
    compute_weight_stack,
    overlap_mask,
)


def test_compute_feather_zero_at_boundary_one_at_center() -> None:
    """Feather weight is small at the valid-region boundary and 1 deep inside."""
    valid = np.zeros((40, 40), dtype=bool)
    valid[5:35, 5:35] = True
    d = compute_feather(valid, feather_width_px=10.0)
    # boundary of valid region (adjacent to invalid) -> small (dist=1 -> 1/W)
    assert d[5, 20] == pytest.approx(0.1, abs=1e-6)
    assert d[34, 20] == pytest.approx(0.1, abs=1e-6)
    # deep inside (>= 10 px from boundary) -> 1
    assert d[20, 20] == pytest.approx(1.0, abs=1e-6)
    # boundary is much smaller than center
    assert d[5, 20] < d[20, 20]


def test_compute_feather_invalid_pixels_are_zero() -> None:
    """Feather is exactly 0 where the valid mask is False."""
    valid = np.zeros((20, 20), dtype=bool)
    valid[5:15, 5:15] = True
    d = compute_feather(valid, feather_width_px=5.0)
    assert np.all(d[~valid] == 0.0)


def test_compute_feather_width_none_returns_ones_on_valid() -> None:
    """feather_width_px=0 (disabled) yields 1 on valid, 0 elsewhere."""
    valid = np.zeros((10, 10), dtype=bool)
    valid[3:7, 3:7] = True
    d = compute_feather(valid, feather_width_px=0.0)
    assert np.allclose(d[valid], 1.0)
    assert np.all(d[~valid] == 0.0)


def test_apply_feather_multiplies_into_weight() -> None:
    """apply_feather multiplies the feather profile into an existing weight."""
    valid = np.zeros((30, 30), dtype=bool)
    valid[10:20, 10:20] = True
    weight = np.ones((30, 30), dtype=np.float32)
    out = apply_feather(weight, valid, feather_width_px=5.0)
    # boundary of valid region -> weight reduced (dist=1 -> 1/5)
    assert out[10, 15] == pytest.approx(0.2, abs=1e-6)
    # deep inside -> weight stays 1
    assert out[15, 15] == pytest.approx(1.0, abs=1e-6)
    # invalid region -> 0
    assert out[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_overlap_mask_and_of_two_valid_thresholds() -> None:
    """overlap_mask is the set of pixels where both weights exceed a threshold."""
    w_i = np.zeros((10, 10), dtype=np.float32)
    w_j = np.zeros((10, 10), dtype=np.float32)
    w_i[0:8, 0:8] = 1.0
    w_j[2:10, 2:10] = 1.0
    om = overlap_mask(w_i, w_j, threshold=0.5)
    # overlap is rows 2..7, cols 2..7
    assert om[5, 5]
    assert not om[0, 0]
    assert not om[9, 9]
    assert om.sum() == 6 * 6


def test_compute_weight_stack_combines_mask_coherence_feather() -> None:
    """Weight = valid_mask * coh^p * feather."""
    valid = np.zeros((20, 20), dtype=bool)
    valid[5:15, 5:15] = True
    coh = np.full((20, 20), 0.5, dtype=np.float32)
    weight = compute_weight_stack(
        valid_mask=valid,
        coherence=coh,
        coherence_exponent=1.0,
        feather_width_px=4.0,
    )
    # boundary of valid region: feather ~1/4 -> weight = 0.5 * 0.25
    assert weight[5, 10] == pytest.approx(0.125, abs=1e-6)
    # deep inside: weight = 1 * 0.5 * 1 = 0.5
    assert weight[10, 10] == pytest.approx(0.5, abs=1e-6)
    # invalid -> 0
    assert weight[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_compute_weight_stack_no_coherence_uses_unity() -> None:
    """When coherence is None, the coherence factor is 1."""
    valid = np.zeros((20, 20), dtype=bool)
    valid[5:15, 5:15] = True
    weight = compute_weight_stack(
        valid_mask=valid,
        coherence=None,
        coherence_exponent=1.0,
        feather_width_px=0.0,
    )
    assert np.allclose(weight[valid], 1.0)


def test_compute_feather_monotonic_from_boundary_inward() -> None:
    """Feather distance increases monotonically from boundary toward center."""
    valid = np.zeros((50, 5), dtype=bool)
    valid[5:45, :] = True
    d = compute_feather(valid, feather_width_px=20.0)
    # along the central column, d should be non-decreasing from row 5 to 25
    center_col = 2
    values = d[5:26, center_col]
    assert np.all(np.diff(values) >= -1e-6)
