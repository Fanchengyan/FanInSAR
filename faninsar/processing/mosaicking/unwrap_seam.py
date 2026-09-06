"""Post-unwrap 2π cycle alignment across vertical mosaic seams.

Complex-domain phase-network merge can leave the wrapped field continuous
while SNAPHU (or another unwrapper) still inserts multi-cycle steps at weak
swath joins.  This module removes those integer cycles without changing the
complex interferogram.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

__all__ = [
    "correct_unwrapped_seam_cycles",
    "force_unwrapped_match_complex_along_range",
    "majority_cycle_align",
    "reintegrate_highcoh_from_seed",
    "reintegrate_unwrapped_along_range",
]


def force_unwrapped_match_complex_along_range(
    unwrapped_phase: np.ndarray,
    complex_ifg: np.ndarray,
    *,
    seed_col: int,
    col_lo: int,
    col_hi: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove per-row integer 2π unwrap cycles so Δψ matches arg(Δz).

    Walks range both ways from ``seed_col`` over ``[col_lo, col_hi]``.  At
    each adjacent pair the expected complex step is
    ``dwr = arg(z[c] conj(z[c±1]))``.  Only an **integer** cycle correction is
    applied:

    ``k = round((ψ[c] - (ψ[c±1] + dwr)) / 2π)`` → ``ψ[c] -= 2π k``

    so multi-cycle islands vanish while SNAPHU's sub-cycle structure (and
    azimuth continuity) is preserved.  No coherence gate or az smooth.

    Parameters
    ----------
    unwrapped_phase : numpy.ndarray
        2-D unwrapped phase (rad). Absolute level at ``seed_col`` is retained.
    complex_ifg : numpy.ndarray
        Matching complex interferogram (not modified).
    seed_col : int
        Column whose unwrapped values are the absolute anchor.
    col_lo, col_hi : int
        Inclusive column range to correct.

    Returns
    -------
    unwrapped_fixed : numpy.ndarray
        Copy with integer cycle steps removed along range.
    report : dict
        ``applied``, ``seed_col``, ``col_lo``, ``col_hi``, ``n_pixels_changed``,
        ``n_rows_seed_valid``, ``n_cycles_removed``.

    """
    unw = np.asarray(unwrapped_phase, dtype=np.float64)
    z = np.asarray(complex_ifg)
    if unw.ndim != 2 or z.shape != unw.shape:
        message = f"shape mismatch: {unw.shape} vs {z.shape}"
        logger.error(message)
        raise ValueError(message)

    height, width = unw.shape
    seed = int(np.clip(seed_col, 0, width - 1))
    lo = max(0, int(col_lo))
    hi = min(width - 1, int(col_hi))
    if lo > hi:
        message = f"invalid column range col_lo={lo} col_hi={hi}"
        logger.error(message)
        raise ValueError(message)
    if not (lo <= seed <= hi):
        seed = int(np.clip(seed, lo, hi))

    ph = np.angle(z)
    out = unw.copy()
    seed_valid = np.isfinite(out[:, seed])
    n_rows_seed = int(seed_valid.sum())
    n_changed = 0
    n_cycles = 0

    # dwr_fwd[:, c-1] = arg(z[:,c] / z[:,c-1]) for c = 1..W-1
    dwr_fwd = np.angle(np.exp(1j * (ph[:, 1:] - ph[:, :-1])))
    twopi = 2.0 * np.pi

    for c in range(seed + 1, hi + 1):
        step = dwr_fwd[:, c - 1]
        prev = out[:, c - 1]
        cur = out[:, c]
        m = seed_valid & np.isfinite(step) & np.isfinite(prev) & np.isfinite(cur)
        if not np.any(m):
            continue
        expected = prev[m] + step[m]
        k = np.round((cur[m] - expected) / twopi)
        ch = k != 0
        if not np.any(ch):
            continue
        idx = np.where(m)[0][ch]
        out[idx, c] = cur[m][ch] - twopi * k[ch]
        n_changed += int(ch.sum())
        n_cycles += int(np.sum(np.abs(k[ch])))

    for c in range(seed - 1, lo - 1, -1):
        step = dwr_fwd[:, c]
        nxt = out[:, c + 1]
        cur = out[:, c]
        m = seed_valid & np.isfinite(step) & np.isfinite(nxt) & np.isfinite(cur)
        if not np.any(m):
            continue
        # ψ[c] should equal ψ[c+1] - step
        expected = nxt[m] - step[m]
        k = np.round((cur[m] - expected) / twopi)
        ch = k != 0
        if not np.any(ch):
            continue
        idx = np.where(m)[0][ch]
        out[idx, c] = cur[m][ch] - twopi * k[ch]
        n_changed += int(ch.sum())
        n_cycles += int(np.sum(np.abs(k[ch])))

    report: dict[str, Any] = {
        "applied": bool(n_changed > 0),
        "seed_col": seed,
        "col_lo": lo,
        "col_hi": hi,
        "n_pixels_changed": int(n_changed),
        "n_pixels_written": int(n_changed),  # alias for campaign logs
        "n_cycles_removed": int(n_cycles),
        "n_rows_seed_valid": n_rows_seed,
    }
    if report["applied"]:
        logger.info(
            "force_unwrapped_match_complex_along_range: seed=%d cols=[%d,%d] "
            "n_changed=%d n_cycles=%d n_rows_seed=%d",
            seed,
            lo,
            hi,
            n_changed,
            n_cycles,
            n_rows_seed,
        )
    return out, report


def majority_cycle_align(
    unwrapped_phase: np.ndarray,
    complex_ifg: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    coh_min: float = 0.2,
    min_frac: float = 0.5,
    min_samples: int = 30,
    align_modes: bool = True,
    seed_col: int | None = None,
    max_complex_step_rad: float = 1.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Force per-column unwrap cycle count to the high-coherence majority.

    SNAPHU can leave isolated multi-cycle *islands* whose wrapped phase is
    continuous with neighbors (``arg(z)`` smooth) but whose integer cycle
    ``N = round((ψ_unw - arg(z)) / 2π)`` differs from the column consensus.
    Consecutive-column step detectors miss gradual islands; column-wise
    majority voting on ``N`` restores them without inventing range structure.

    For each range column the modal ``N`` among samples with
    ``coherence >= coh_min`` is estimated.  When that mode covers at least
    ``min_frac`` of those samples, every finite pixel in the column is
    rewritten as ``arg(z) + 2π N_mode``.

    When ``align_modes`` is True (default), column modes are then walked from
    a seed column and forced continuous wherever the complex step is small,
    preventing vertical 2π walls between independently voted modes.

    Parameters
    ----------
    unwrapped_phase : numpy.ndarray
        2-D unwrapped phase (rad).
    complex_ifg : numpy.ndarray
        Matching complex interferogram (wrap source; not modified).
    coherence : numpy.ndarray, optional
        Coherence mask used only to select the majority population.
    coh_min : float, optional
        Coherence threshold for the majority vote. Default 0.2.
    min_frac : float, optional
        Minimum fraction of high-coherence samples that must agree on the
        modal cycle. Default 0.5.
    min_samples : int, optional
        Minimum high-coherence samples per column to attempt a vote.
    align_modes : bool, optional
        If True, remove range-direction mode jumps when the complex field
        is continuous. Default True.
    seed_col : int, optional
        Seed column for mode alignment (default: image centre).
    max_complex_step_rad : float, optional
        Maximum median ``|arg|`` step to treat the complex field as continuous
        when aligning modes. Default 1.5.

    Returns
    -------
    unwrapped_fixed : numpy.ndarray
        Copy with per-column majority cycle applied where warranted.
    report : dict
        ``applied``, ``n_pixels_changed``, ``n_columns_changed``.

    """
    unw = np.asarray(unwrapped_phase, dtype=np.float64)
    z = np.asarray(complex_ifg)
    if unw.ndim != 2 or z.shape != unw.shape:
        message = f"shape mismatch: {unw.shape} vs {z.shape}"
        logger.error(message)
        raise ValueError(message)

    height, width = unw.shape
    wrap = np.angle(z)
    if coherence is not None:
        coh = np.asarray(coherence, dtype=np.float64)
        if coh.shape != unw.shape:
            message = f"coherence shape {coh.shape} != phase shape {unw.shape}"
            logger.error(message)
            raise ValueError(message)
    else:
        coh = np.ones(unw.shape, dtype=np.float64)

    N = np.round((unw - wrap) / (2.0 * np.pi))
    N = np.where(np.isfinite(unw) & np.isfinite(wrap), N, np.nan)
    modes = np.full(width, np.nan, dtype=np.float64)
    for c in range(width):
        m = (coh[:, c] >= coh_min) & np.isfinite(N[:, c])
        if int(m.sum()) < min_samples:
            continue
        vals, counts = np.unique(N[m, c].astype(np.int32), return_counts=True)
        if float(counts.max()) / float(m.sum()) < min_frac:
            continue
        modes[c] = float(vals[int(np.argmax(counts))])

    n_mode_jumps_before = 0
    for c in range(1, width):
        if np.isfinite(modes[c]) and np.isfinite(modes[c - 1]) and modes[c] != modes[c - 1]:
            n_mode_jumps_before += 1

    if align_modes and np.any(np.isfinite(modes)):
        seed = int(seed_col) if seed_col is not None else width // 2
        seed = max(0, min(width - 1, seed))
        valid_idx = np.where(np.isfinite(modes))[0]
        if not np.isfinite(modes[seed]) and valid_idx.size:
            modes[seed] = modes[valid_idx[int(np.argmin(np.abs(valid_idx - seed)))]]
        for c in range(seed + 1, width):
            if not np.isfinite(modes[c]):
                modes[c] = modes[c - 1]
                continue
            dwr = np.angle(np.exp(1j * (wrap[:, c] - wrap[:, c - 1])))
            m = (
                (coh[:, c] >= coh_min)
                & (coh[:, c - 1] >= coh_min)
                & np.isfinite(dwr)
            )
            if int(m.sum()) < 20 or float(np.median(np.abs(dwr[m]))) < max_complex_step_rad:
                if modes[c] != modes[c - 1]:
                    modes[c] = modes[c - 1]
        for c in range(seed - 1, -1, -1):
            if not np.isfinite(modes[c]):
                modes[c] = modes[c + 1]
                continue
            dwr = np.angle(np.exp(1j * (wrap[:, c + 1] - wrap[:, c])))
            m = (
                (coh[:, c] >= coh_min)
                & (coh[:, c + 1] >= coh_min)
                & np.isfinite(dwr)
            )
            if int(m.sum()) < 20 or float(np.median(np.abs(dwr[m]))) < max_complex_step_rad:
                if modes[c] != modes[c + 1]:
                    modes[c] = modes[c + 1]

    out = unw.copy()
    n_pixels = 0
    n_cols = 0
    for c in range(width):
        if not np.isfinite(modes[c]):
            continue
        mode = int(modes[c])
        mall = np.isfinite(N[:, c]) & np.isfinite(wrap[:, c])
        tf = mall & (N[:, c] != mode)
        if not np.any(tf):
            # still apply if N is nan but wrap valid
            tf2 = (~np.isfinite(N[:, c])) & np.isfinite(wrap[:, c])
            if np.any(tf2):
                out[tf2, c] = wrap[tf2, c] + 2.0 * np.pi * mode
                n_pixels += int(tf2.sum())
                n_cols += 1
            continue
        out[tf, c] = wrap[tf, c] + 2.0 * np.pi * mode
        n_pixels += int(tf.sum())
        n_cols += 1

    n_mode_jumps_after = 0
    for c in range(1, width):
        if np.isfinite(modes[c]) and np.isfinite(modes[c - 1]) and modes[c] != modes[c - 1]:
            n_mode_jumps_after += 1

    report: dict[str, Any] = {
        "applied": bool(n_pixels > 0),
        "n_pixels_changed": int(n_pixels),
        "n_columns_changed": int(n_cols),
        "coh_min": float(coh_min),
        "min_frac": float(min_frac),
        "align_modes": bool(align_modes),
        "n_mode_jumps_before": int(n_mode_jumps_before),
        "n_mode_jumps_after": int(n_mode_jumps_after),
    }
    if report["applied"]:
        logger.info(
            "majority_cycle_align: n_cols=%d n_pixels=%d coh_min=%.2f min_frac=%.2f",
            n_cols,
            n_pixels,
            coh_min,
            min_frac,
        )
    return out, report


def reintegrate_highcoh_from_seed(
    unwrapped_phase: np.ndarray,
    complex_ifg: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    seed_col: int,
    col_lo: int,
    col_hi: int,
    coh_min: float = 0.3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reintegrate unwrapped phase along range through high-coherence paths.

    Starting from ``seed_col`` (left absolute level preserved), walks range
    left and right setting ``ψ[c] = ψ[c±1] + arg(z[c] conj(z[c±1]))`` only
    where both columns meet ``coh_min``.  This removes residual 2π steps that
    remain after majority cycle alignment while avoiding low-coherence paths
    that would accumulate unwrap errors.

    Parameters
    ----------
    unwrapped_phase, complex_ifg : numpy.ndarray
        Matching 2-D arrays.
    coherence : numpy.ndarray, optional
        Coherence weights.
    seed_col : int
        Seed column (absolute level retained).
    col_lo, col_hi : int
        Inclusive column range to reintegrate.
    coh_min : float, optional
        Minimum coherence on both sides of a step. Default 0.3.

    Returns
    -------
    unwrapped_fixed : numpy.ndarray
        Copy with high-coherence range reintegration applied.
    report : dict
        Diagnostics.

    """
    unw = np.asarray(unwrapped_phase, dtype=np.float64)
    z = np.asarray(complex_ifg)
    if unw.ndim != 2 or z.shape != unw.shape:
        message = f"shape mismatch: {unw.shape} vs {z.shape}"
        logger.error(message)
        raise ValueError(message)
    height, width = unw.shape
    if coherence is not None:
        coh = np.asarray(coherence, dtype=np.float64)
    else:
        coh = np.ones(unw.shape, dtype=np.float64)
    ph = np.angle(z)
    out = unw.copy()
    seed = int(np.clip(seed_col, 0, width - 1))
    lo = max(0, int(col_lo))
    hi = min(width - 1, int(col_hi))
    n_steps = 0
    for c in range(seed + 1, hi + 1):
        dwr = np.angle(np.exp(1j * (ph[:, c] - ph[:, c - 1])))
        m = (
            np.isfinite(out[:, c - 1])
            & np.isfinite(dwr)
            & (coh[:, c] >= coh_min)
            & (coh[:, c - 1] >= coh_min)
        )
        if np.any(m):
            out[m, c] = out[m, c - 1] + dwr[m]
            n_steps += int(m.sum())
    for c in range(seed - 1, lo - 1, -1):
        dwr = np.angle(np.exp(1j * (ph[:, c + 1] - ph[:, c])))
        m = (
            np.isfinite(out[:, c + 1])
            & np.isfinite(dwr)
            & (coh[:, c] >= coh_min)
            & (coh[:, c + 1] >= coh_min)
        )
        if np.any(m):
            out[m, c] = out[m, c + 1] - dwr[m]
            n_steps += int(m.sum())
    report: dict[str, Any] = {
        "applied": bool(n_steps > 0),
        "seed_col": seed,
        "col_lo": lo,
        "col_hi": hi,
        "coh_min": float(coh_min),
        "n_steps": int(n_steps),
    }
    if report["applied"]:
        logger.info(
            "reintegrate_highcoh_from_seed: seed=%d cols=[%d,%d] n_steps=%d coh_min=%.2f",
            seed,
            lo,
            hi,
            n_steps,
            coh_min,
        )
    return out, report


def reintegrate_unwrapped_along_range(
    unwrapped_phase: np.ndarray,
    complex_ifg: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    col_lo: int,
    col_hi: int,
    coh_min: float = 0.25,
    seed_width: int = 30,
    az_smooth: int = 21,
    n_iter: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove adjacent-column 2π unwrap steps (az-consistent, no striping).

    Iteratively detects columns where the unwrapped step differs from the
    complex step by ``≈ 2π k`` (``|k|∈{1,2}``) while the complex field is
    continuous, builds a per-column integer correction field, **median-filters
    it along azimuth** so neighboring rows agree (kills horizontal striping),
    then subtracts ``2π k`` from those pixels only.  Repeats until no more
    steps remain (or ``n_iter`` is reached).

    Parameters
    ----------
    unwrapped_phase, complex_ifg : numpy.ndarray
        Matching 2-D arrays.
    coherence : numpy.ndarray, optional
        Coherence mask.
    col_lo, col_hi : int
        Inclusive column range to scan.
    coh_min : float, optional
        Minimum coherence on both sides of a step. Default 0.25.
    seed_width : int, optional
        Unused (API compatibility).
    az_smooth : int, optional
        Odd median-filter width along azimuth for the k-field. Default 21.
    n_iter : int, optional
        Maximum outer iterations. Default 8.

    Returns
    -------
    unwrapped_fixed : numpy.ndarray
        Copy with 2π steps removed.
    report : dict
        Diagnostics.

    """
    del seed_width
    from scipy.ndimage import median_filter

    unw = np.asarray(unwrapped_phase, dtype=np.float64)
    z = np.asarray(complex_ifg)
    if unw.ndim != 2 or z.shape != unw.shape:
        message = f"shape mismatch: {unw.shape} vs {z.shape}"
        logger.error(message)
        raise ValueError(message)
    height, width = unw.shape
    report: dict[str, Any] = {
        "applied": False,
        "col_lo": int(col_lo),
        "col_hi": int(col_hi),
        "n_rows": 0,
        "n_steps_fixed": 0,
    }
    if col_lo < 1 or col_hi >= width or col_hi < col_lo:
        report["reason"] = "invalid column range"
        return unw.copy(), report

    if coherence is not None:
        coh = np.asarray(coherence, dtype=np.float64)
    else:
        coh = np.ones(unw.shape, dtype=np.float64)

    out = unw.copy()
    ph = np.angle(z)
    if az_smooth >= 3 and az_smooth % 2 == 0:
        az_smooth += 1

    total_fixed = 0
    rows_touched = np.zeros(height, dtype=bool)
    max_abs_corr = 0.0

    for _it in range(max(1, n_iter)):
        k_field = np.zeros(out.shape, dtype=np.float64)
        for c in range(col_lo, col_hi + 1):
            duw = out[:, c] - out[:, c - 1]
            dwr = np.angle(np.exp(1j * (ph[:, c] - ph[:, c - 1])))
            m = (
                (coh[:, c] >= coh_min)
                & (coh[:, c - 1] >= coh_min)
                & np.isfinite(duw)
                & np.isfinite(dwr)
            )
            extra = duw - dwr
            k = np.round(extra / (2.0 * np.pi))
            good = (
                m
                & (np.abs(dwr) < 1.5)
                & (np.abs(k) >= 1)
                & (np.abs(k) <= 2)
                & (np.abs(extra - 2.0 * np.pi * k) < 0.75)
            )
            k_field[good, c] = k[good]

        if az_smooth >= 3:
            for c in range(col_lo, col_hi + 1):
                if np.any(k_field[:, c] != 0):
                    k_field[:, c] = np.round(
                        median_filter(k_field[:, c], size=az_smooth, mode="nearest")
                    )

        n_fix = 0
        for c in range(col_lo, col_hi + 1):
            m = k_field[:, c] != 0
            if not np.any(m):
                continue
            out[m, c] = out[m, c] - 2.0 * np.pi * k_field[m, c]
            n_fix += int(m.sum())
            rows_touched[m] = True
            max_abs_corr = max(
                max_abs_corr, float(np.max(np.abs(2.0 * np.pi * k_field[m, c])))
            )
        total_fixed += n_fix
        if n_fix == 0:
            break

    report["n_rows"] = int(rows_touched.sum())
    report["n_steps_fixed"] = int(total_fixed)
    report["max_abs_correction"] = float(max_abs_corr)
    report["applied"] = bool(total_fixed > 0)
    if report["applied"]:
        logger.info(
            "reintegrate_unwrapped_along_range: cols=[%d,%d] n_rows=%d "
            "n_fixed=%d max|Δ|=%.3f rad az_smooth=%d",
            col_lo,
            col_hi,
            report["n_rows"],
            total_fixed,
            max_abs_corr,
            az_smooth,
        )
    return out, report


def correct_unwrapped_seam_cycles(
    unwrapped_phase: np.ndarray,
    complex_ifg: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    seam_col_lo: int,
    seam_col_hi: int,
    coh_min: float = 0.25,
    left_width: int = 40,
    right_width: int = 40,
    max_complex_step_rad: float = 1.0,
    min_rows: int = 50,
    per_row: bool = True,
    az_smooth: int = 31,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove integer 2π offsets east of a vertical seam.

    Estimates the high-coherence left/right unwrapped median difference and
    the complex circular-mean difference across
    ``[seam_col_lo - left_width, seam_col_lo)`` vs
    ``(seam_col_hi, seam_col_hi + right_width]``.  When the complex step is
    small (continuous wrapped field) but the unwrapped step is near
    ``2π k``, subtracts ``2π k`` from all columns strictly east of the seam
    midpoint.

    Parameters
    ----------
    unwrapped_phase : numpy.ndarray
        2-D unwrapped phase (rad).
    complex_ifg : numpy.ndarray
        Matching complex interferogram (used only for continuity check and
        complex-mean Δφ; not modified).
    coherence : numpy.ndarray, optional
        Coherence weights; samples below ``coh_min`` are ignored.
    seam_col_lo, seam_col_hi : int
        Inclusive column range of the seam / overlap band.
    coh_min : float, optional
        Coherence threshold. Default 0.25.
    left_width, right_width : int, optional
        Column windows west/east of the seam band used for L/R medians.
    max_complex_step_rad : float, optional
        Require ``|arg(E[z_R conj(z_L)])| < max_complex_step_rad`` so the
        complex field is continuous at the join. Default 1.0.
    min_rows : int, optional
        Minimum valid azimuth rows to attempt a correction.
    per_row : bool, optional
        If ``True`` (default), estimate integer ``k`` per azimuth row
        (smoothed); otherwise use a single global ``k``.
    az_smooth : int, optional
        Odd window length for smoothing per-row ``k`` along azimuth.

    Returns
    -------
    unwrapped_fixed : numpy.ndarray
        Copy of ``unwrapped_phase`` with east-side 2π cycles removed when
        warranted; otherwise a copy of the input.
    report : dict
        Diagnostic fields: ``applied``, ``k_global``, ``n_rows``,
        ``median_unw_jump_before``, ``median_unw_jump_after``,
        ``complex_step``, ``frac_rows_gt_pi_before``, etc.

    """
    unw = np.asarray(unwrapped_phase, dtype=np.float64)
    z = np.asarray(complex_ifg)
    if unw.ndim != 2 or z.shape != unw.shape:
        message = (
            f"unwrapped_phase and complex_ifg must share 2-D shape; "
            f"got {unw.shape} and {z.shape}"
        )
        logger.error(message)
        raise ValueError(message)

    height, width = unw.shape
    report: dict[str, Any] = {
        "applied": False,
        "seam_col_lo": int(seam_col_lo),
        "seam_col_hi": int(seam_col_hi),
        "k_global": 0,
        "n_rows": 0,
    }
    if seam_col_lo < left_width or seam_col_hi + right_width >= width:
        report["reason"] = "seam windows out of bounds"
        return unw.copy(), report
    if seam_col_hi < seam_col_lo:
        report["reason"] = "seam_col_hi < seam_col_lo"
        return unw.copy(), report

    if coherence is not None:
        coh = np.asarray(coherence, dtype=np.float64)
        if coh.shape != unw.shape:
            message = f"coherence shape {coh.shape} != phase shape {unw.shape}"
            logger.error(message)
            raise ValueError(message)
    else:
        coh = np.ones(unw.shape, dtype=np.float64)

    l0, l1 = seam_col_lo - left_width, seam_col_lo
    r0, r1 = seam_col_hi + 1, seam_col_hi + 1 + right_width

    row_unw_jump = np.full(height, np.nan, dtype=np.float64)
    row_c_re = np.zeros(height, dtype=np.float64)
    row_c_im = np.zeros(height, dtype=np.float64)
    row_ok = np.zeros(height, dtype=bool)
    # Per-row left median (for locating the column where the cycle jumps).
    row_left_med = np.full(height, np.nan, dtype=np.float64)

    for r in range(height):
        L = unw[r, l0:l1]
        R = unw[r, r0:r1]
        Lc = coh[r, l0:l1]
        Rc = coh[r, r0:r1]
        zL = z[r, l0:l1]
        zR = z[r, r0:r1]
        lm = np.isfinite(L) & np.isfinite(Lc) & (Lc >= coh_min)
        rm = np.isfinite(R) & np.isfinite(Rc) & (Rc >= coh_min)
        lm &= np.isfinite(zL.real) & (np.abs(zL) > 1e-9)
        rm &= np.isfinite(zR.real) & (np.abs(zR) > 1e-9)
        if int(lm.sum()) < 5 or int(rm.sum()) < 5:
            continue
        row_left_med[r] = float(np.median(L[lm]))
        row_unw_jump[r] = float(np.median(R[rm]) - row_left_med[r])
        mL = np.mean(zL[lm])
        mR = np.mean(zR[rm])
        prod = mR * np.conj(mL)
        row_c_re[r] = float(prod.real)
        row_c_im[r] = float(prod.imag)
        row_ok[r] = True

    n_rows = int(row_ok.sum())
    report["n_rows"] = n_rows
    if n_rows < min_rows:
        report["reason"] = f"too few rows ({n_rows} < {min_rows})"
        return unw.copy(), report

    jumps = row_unw_jump[row_ok]
    c_sum = complex(float(row_c_re[row_ok].sum()), float(row_c_im[row_ok].sum()))
    complex_step = float(np.angle(c_sum)) if abs(c_sum) > 0 else 0.0
    med_jump = float(np.median(jumps))
    report["complex_step"] = complex_step
    report["median_unw_jump_before"] = med_jump
    report["frac_rows_gt_pi_before"] = float(np.mean(np.abs(jumps) > np.pi))
    report["frac_rows_gt_2pi_before"] = float(np.mean(np.abs(jumps) > 2.0 * np.pi))

    if abs(complex_step) > max_complex_step_rad:
        report["reason"] = (
            f"complex step {complex_step:.3f} > {max_complex_step_rad:.3f} "
            "(wrapped field not continuous; refusing 2π align)"
        )
        return unw.copy(), report

    # Integer cycles such that unw_jump - 2πk ≈ complex_step.
    residual_for_k = med_jump - complex_step
    k_global = int(np.round(residual_for_k / (2.0 * np.pi)))
    # Robust: when only a minority of rows have multi-cycle jumps, the
    # overall median is diluted.  Estimate k from the large-jump subset.
    large = jumps[np.abs(jumps - complex_step) > np.pi]
    if large.size >= max(min_rows // 4, 10):
        k_large = int(np.round((float(np.median(large)) - complex_step) / (2.0 * np.pi)))
        if abs(k_large) > abs(k_global):
            k_global = k_large
    k_rows_raw = np.round((jumps - complex_step) / (2.0 * np.pi)).astype(np.int32)
    if k_rows_raw.size:
        vals, counts = np.unique(k_rows_raw[k_rows_raw != 0], return_counts=True)
        if vals.size:
            k_mode = int(vals[int(np.argmax(counts))])
            if abs(k_mode) > abs(k_global):
                k_global = k_mode
    report["k_global"] = k_global

    out = unw.copy()
    if per_row:
        k_row = np.zeros(height, dtype=np.int32)
        for r in np.where(row_ok)[0]:
            prod = complex(row_c_re[r], row_c_im[r])
            c_step_r = float(np.angle(prod)) if abs(prod) > 0 else complex_step
            kr = int(np.round((row_unw_jump[r] - c_step_r) / (2.0 * np.pi)))
            # Only correct rows with a clear multi-cycle residual.
            if abs(row_unw_jump[r] - c_step_r) > np.pi:
                k_row[r] = kr if kr != 0 else k_global
        # Mild azimuth smoothing of non-zero k (preserve zeros).
        if az_smooth >= 3 and az_smooth % 2 == 1 and np.any(k_row != 0):
            k_f = k_row.astype(np.float64)
            mask = (k_row != 0).astype(np.float64)
            ker = np.ones(az_smooth, dtype=np.float64)
            num = np.convolve(k_f * mask, ker, mode="same")
            den = np.convolve(mask, ker, mode="same")
            smooth = np.zeros(height, dtype=np.float64)
            okm = den > 0
            smooth[okm] = num[okm] / den[okm]
            k_row = np.where(den >= 3, np.round(smooth).astype(np.int32), k_row)

        n_fixed = 0
        cut_cols: list[int] = []
        for r in range(height):
            kr = int(k_row[r])
            if kr == 0:
                continue
            # Per-row cut: first column in the seam band that has already
            # jumped by ~2π relative to the left median.
            left_m = row_left_med[r]
            if not np.isfinite(left_m):
                left_m = 0.0
            cut_r = seam_col_hi + 1  # default: pure east of seam band
            for c in range(seam_col_lo, min(seam_col_hi + 1, width)):
                if not np.isfinite(unw[r, c]) or coh[r, c] < coh_min:
                    continue
                if abs(float(unw[r, c]) - left_m - complex_step) > np.pi:
                    cut_r = c
                    break
            out[r, cut_r:] = out[r, cut_r:] - 2.0 * np.pi * kr
            n_fixed += 1
            cut_cols.append(cut_r)
        report["k_row_abs_max"] = int(np.max(np.abs(k_row))) if height else 0
        report["k_row_nonzero_frac"] = float(np.mean(k_row != 0))
        report["n_rows_fixed"] = n_fixed
        report["cut_col_median"] = (
            int(np.median(cut_cols)) if cut_cols else (seam_col_lo + seam_col_hi) // 2
        )
        applied = n_fixed > 0
    else:
        if k_global == 0:
            report["reason"] = "k_global=0 (no integer cycle to remove)"
            return out, report
        # Global cut from first band column with multi-cycle median residual.
        cut_col = seam_col_hi + 1
        left_med_global = float(np.nanmedian(row_left_med[row_ok]))
        for c in range(seam_col_lo, min(seam_col_hi + 1, width)):
            col = unw[:, c]
            cc = coh[:, c]
            m = row_ok & np.isfinite(col) & (cc >= coh_min)
            if int(m.sum()) < max(min_rows // 4, 10):
                continue
            d = float(np.median(col[m]) - left_med_global)
            if abs(d - complex_step) > np.pi:
                cut_col = c
                break
        report["cut_col"] = int(cut_col)
        out[:, cut_col:] = out[:, cut_col:] - 2.0 * np.pi * k_global
        applied = True

    # After metrics on the same L/R windows.
    jumps_after: list[float] = []
    for r in np.where(row_ok)[0]:
        L = out[r, l0:l1]
        R = out[r, r0:r1]
        Lc = coh[r, l0:l1]
        Rc = coh[r, r0:r1]
        lm = np.isfinite(L) & (Lc >= coh_min)
        rm = np.isfinite(R) & (Rc >= coh_min)
        if int(lm.sum()) < 5 or int(rm.sum()) < 5:
            continue
        jumps_after.append(float(np.median(R[rm]) - np.median(L[lm])))
    ja = np.asarray(jumps_after, dtype=np.float64)
    if ja.size:
        report["median_unw_jump_after"] = float(np.median(ja))
        report["frac_rows_gt_pi_after"] = float(np.mean(np.abs(ja) > np.pi))
        report["frac_rows_gt_2pi_after"] = float(np.mean(np.abs(ja) > 2.0 * np.pi))
    report["applied"] = applied
    if applied:
        logger.info(
            "correct_unwrapped_seam_cycles: seam=[%d,%d] k_global=%d "
            "unw_jump %.3f->%s complex_step=%.3f n_rows=%d",
            seam_col_lo,
            seam_col_hi,
            k_global,
            med_jump,
            f"{report.get('median_unw_jump_after', float('nan')):.3f}",
            complex_step,
            n_rows,
        )
    return out, report
