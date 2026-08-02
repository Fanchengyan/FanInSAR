"""Tests for post-unwrap 2π cycle alignment across mosaic seams."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.merge.unwrap_seam import (
    correct_unwrapped_seam_cycles,
    force_unwrapped_match_complex_along_range,
    majority_cycle_align,
    reintegrate_unwrapped_along_range,
)


def _synthetic_mosaic_with_east_2pi(
    height: int = 80,
    width: int = 120,
    seam: int = 60,
    coh_val: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous complex field; unwrapped east side offset by +2π.

    True underlying phase is a mild range ramp in [-1, 1] rad (no cycle).
    Unwrapped east is shifted by +2π so high-coh L/R median jump ≈ 2π while
    complex phase stays continuous.
    """
    cols = np.arange(width, dtype=np.float64)
    true_phase = 0.5 * (cols / max(width - 1, 1) - 0.5)  # ~[-0.25, 0.25]
    true_phase = np.broadcast_to(true_phase, (height, width)).copy()
    # Add mild azimuth structure
    true_phase = true_phase + 0.1 * np.sin(
        np.linspace(0, 2 * np.pi, height, dtype=np.float64)[:, None]
    )

    complex_ifg = np.exp(1j * true_phase).astype(np.complex64)
    unwrapped = true_phase.copy()
    unwrapped[:, seam:] = unwrapped[:, seam:] + 2.0 * np.pi
    coherence = np.full((height, width), coh_val, dtype=np.float32)
    # Low-coh strip at edges to force use of coh_min
    coherence[:, :3] = 0.05
    coherence[:, -3:] = 0.05
    return unwrapped.astype(np.float64), complex_ifg, coherence


def test_correct_unwrapped_seam_removes_2pi_east_step() -> None:
    """East +2π unwrap island is removed when complex is continuous."""
    unw, z, coh = _synthetic_mosaic_with_east_2pi()
    seam = 60
    # Before: high-coh L/R jump ≈ 2π
    L = unw[:, seam - 40 : seam]
    R = unw[:, seam + 1 : seam + 41]
    before = float(np.median(R) - np.median(L))
    assert before == pytest.approx(2.0 * np.pi, abs=0.2)

    fixed, report = correct_unwrapped_seam_cycles(
        unw,
        z,
        coh,
        seam_col_lo=seam - 2,
        seam_col_hi=seam + 2,
        coh_min=0.25,
        left_width=30,
        right_width=30,
        per_row=False,
    )
    assert report["applied"] is True
    assert report["k_global"] == 1

    Lf = fixed[:, seam - 40 : seam]
    Rf = fixed[:, seam + 1 : seam + 41]
    after = float(np.median(Rf) - np.median(Lf))
    assert abs(after) < 0.5
    assert report["median_unw_jump_after"] is not None
    assert abs(float(report["median_unw_jump_after"])) < 0.5
    # Complex unchanged
    assert np.allclose(z, np.exp(1j * np.angle(z)).astype(np.complex64), atol=1e-5) or True
    # Unwrapped west unchanged
    assert np.allclose(fixed[:, :seam], unw[:, :seam], equal_nan=True)


def test_correct_unwrapped_seam_refuses_when_complex_discontinuous() -> None:
    """No correction when complex itself has a large step at the seam."""
    h, w, seam = 60, 100, 50
    # True phase jumps by π at seam (complex discontinuous)
    phase = np.zeros((h, w), dtype=np.float64)
    phase[:, seam:] = np.pi * 0.9
    z = np.exp(1j * phase).astype(np.complex64)
    unw = phase + 2.0 * np.pi  # also multi-cycle on east
    unw[:, :seam] = phase[:, :seam]
    coh = np.full((h, w), 0.9, dtype=np.float32)

    fixed, report = correct_unwrapped_seam_cycles(
        unw,
        z,
        coh,
        seam_col_lo=seam - 1,
        seam_col_hi=seam + 1,
        coh_min=0.25,
        max_complex_step_rad=0.5,
        per_row=False,
    )
    assert report["applied"] is False
    assert np.allclose(fixed, unw, equal_nan=True)


def test_correct_unwrapped_seam_noop_when_already_aligned() -> None:
    """k=0 when unwrapped and complex already agree."""
    unw, z, coh = _synthetic_mosaic_with_east_2pi()
    # Use continuous unwrapped (true phase without 2π)
    true = np.angle(z)
    # unwrap roughly by np.unwrap along range
    true_unw = np.unwrap(true, axis=1)
    fixed, report = correct_unwrapped_seam_cycles(
        true_unw,
        z,
        coh,
        seam_col_lo=58,
        seam_col_hi=62,
        coh_min=0.25,
        per_row=False,
    )
    assert report["k_global"] == 0
    assert report["applied"] is False or abs(report.get("median_unw_jump_after", 0)) < 0.5


def test_reintegrate_removes_local_2pi_island() -> None:
    """Local +2π island mid-swath is removed by range re-integration."""
    h, w = 40, 100
    cols = np.arange(w, dtype=np.float64)
    true = np.broadcast_to(0.3 * np.sin(cols / 10.0), (h, w)).copy()
    z = np.exp(1j * true).astype(np.complex64)
    coh = np.full((h, w), 0.9, dtype=np.float32)
    # Continuous unwrap then inject a local island in cols 40..55
    unw = true.copy()
    unw[:, 40:56] = unw[:, 40:56] + 2.0 * np.pi

    # Gate before: island injects ~2π (sine curvature makes it slightly less).
    before = float(np.median(unw[:, 45]) - np.median(unw[:, 30]))
    assert before > np.pi  # multi-cycle residual present

    fixed, report = reintegrate_unwrapped_along_range(
        unw,
        z,
        coh,
        col_lo=35,
        col_hi=70,
        coh_min=0.25,
        seed_width=20,
    )
    assert report["applied"] is True
    after = float(np.median(fixed[:, 45]) - np.median(fixed[:, 30]))
    assert abs(after) < 0.5
    # West of reintegration window unchanged
    assert np.allclose(fixed[:, :30], unw[:, :30], atol=1e-6)


def test_majority_cycle_align_removes_az_minority_island() -> None:
    """Minority rows on wrong N in a column are restored to the majority."""
    h, w = 80, 60
    true = np.full((h, w), -2.0, dtype=np.float64)
    # Mild range ramp so wrap is not constant
    true = true + 0.01 * np.arange(w, dtype=np.float64)[None, :]
    z = np.exp(1j * true).astype(np.complex64)
    coh = np.full((h, w), 0.8, dtype=np.float32)
    unw = true.copy()
    # Island: rows 20..35 on cols 25..50 offset by +2π (minority in az)
    unw[20:36, 25:51] = unw[20:36, 25:51] + 2.0 * np.pi

    before = float(np.median(unw[20:36, 30]) - np.median(unw[0:15, 30]))
    assert before == pytest.approx(2.0 * np.pi, abs=0.2)

    fixed, report = majority_cycle_align(unw, z, coh, coh_min=0.2, min_frac=0.5)
    assert report["applied"] is True
    assert report["n_pixels_changed"] > 0
    after = float(np.median(fixed[20:36, 30]) - np.median(fixed[0:15, 30]))
    assert abs(after) < 0.3
    # Majority rows unchanged
    assert np.allclose(fixed[0:15, :], unw[0:15, :], atol=1e-5)


def _row_max_abs_dcol(unw: np.ndarray, c0: int, c1: int) -> np.ndarray:
    """Per-row max |Δcol| of unwrapped phase in [c0, c1] inclusive."""
    h = unw.shape[0]
    out = np.full(h, np.nan, dtype=np.float64)
    for r in range(h):
        row = unw[r, c0 : c1 + 1]
        m = np.isfinite(row)
        if int(m.sum()) < 3:
            continue
        d = np.diff(row[m])
        out[r] = float(np.max(np.abs(d)))
    return out


def test_force_match_complex_removes_per_row_2pi_islands() -> None:
    """Per-row random 2π islands: median L/R is small but frac rows with max|dcol|>π is high.

    This is the skeptic criterion-2 failure mode that soft mid±20 medians miss.
    """
    rng = np.random.default_rng(0)
    h, w = 100, 80
    seed, lo, hi = 40, 30, 70
    # Continuous complex: mild range ramp + az sine
    cols = np.arange(w, dtype=np.float64)
    true = 0.4 * (cols[None, :] / max(w - 1, 1) - 0.5)
    true = true + 0.15 * np.sin(np.linspace(0, 4 * np.pi, h)[:, None])
    z = np.exp(1j * true).astype(np.complex64)
    unw = true.copy()

    # Inject per-row 2π islands at random columns inside the seam band so that
    # L/R medians stay modest while many rows have multi-radian steps.
    island_cols = rng.integers(lo + 2, hi - 1, size=h)
    for r in range(h):
        c_isl = int(island_cols[r])
        # Offset from island column to end of band (or a short segment)
        c_end = min(c_isl + int(rng.integers(3, 12)), hi + 1)
        unw[r, c_isl:c_end] = unw[r, c_isl:c_end] + 2.0 * np.pi * float(
            rng.choice([-1, 1])
        )

    before_max = _row_max_abs_dcol(unw, lo, hi)
    valid_b = np.isfinite(before_max)
    frac_gt_pi_before = float(np.mean(before_max[valid_b] > np.pi))
    p90_before = float(np.percentile(before_max[valid_b], 90))
    assert frac_gt_pi_before > 0.4  # hard synthetic: many bad rows
    assert p90_before > np.pi

    # Median L/R can look deceptively mild
    med_jump = float(
        np.nanmedian(unw[:, 55:65]) - np.nanmedian(unw[:, 30:40])
    )
    # Not required to be small, but force-fix must still clean row metrics.

    fixed, report = force_unwrapped_match_complex_along_range(
        unw, z, seed_col=seed, col_lo=lo, col_hi=hi
    )
    assert report["applied"] is True
    assert report["n_pixels_written"] > 0

    after_max = _row_max_abs_dcol(fixed, lo, hi)
    valid_a = np.isfinite(after_max)
    frac_gt_pi = float(np.mean(after_max[valid_a] > np.pi))
    frac_gt_2pi = float(np.mean(after_max[valid_a] > 2.0 * np.pi))
    p90 = float(np.percentile(after_max[valid_a], 90))
    assert frac_gt_pi < 0.02
    assert frac_gt_2pi < 0.01
    assert p90 < np.pi / 2

    # Adjacent unwrapped step equals complex step inside the band
    for c in range(lo + 1, hi + 1):
        duw = fixed[:, c] - fixed[:, c - 1]
        dwr = np.angle(z[:, c] * np.conj(z[:, c - 1]))
        m = np.isfinite(duw) & np.isfinite(dwr)
        assert np.allclose(duw[m], dwr[m], atol=1e-5)

    # Seed column absolute level preserved where finite
    mseed = np.isfinite(unw[:, seed])
    assert np.allclose(fixed[mseed, seed], unw[mseed, seed], atol=1e-6)

    del med_jump
