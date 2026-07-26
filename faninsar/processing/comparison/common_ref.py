"""Common reference-pixel selection and residual subtraction for multi-stack compare."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)

# Pinned ref for 20161207-20161231 IW1 burst0 three-way UTM compare.
# Absolute path so scripts/tests work regardless of cwd. Do not auto-move.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PINNED_REF_JSON = (
    _REPO_ROOT
    / "reports"
    / "2026-07-16-three-way-unw-geocode-compare"
    / "common_ref_point.json"
)


@dataclass(frozen=True, slots=True)
class CommonRefPoint:
    """Shared geographic (or radar) reference pixel for multi-software residual.

    Attributes
    ----------
    row, col : int
        Indices on the shared grid.
    mean_coherence : float
        Mean of the three coherences at the selected pixel.
    coherences : dict[str, float]
        Per-stack coherence at the pixel (keys are stack labels).

    """

    row: int
    col: int
    mean_coherence: float
    coherences: dict[str, float]


def select_common_ref_point(
    coherences: dict[str, np.ndarray],
    *,
    valid: np.ndarray | None = None,
    min_coherence: float = 0.15,
    smooth_window: int = 5,
) -> CommonRefPoint:
    """Pick the pixel maximising the mean of multi-stack coherences.

    Parameters
    ----------
    coherences : dict of str to ndarray
        Coherence layers on a shared grid (same shape). At least one entry
        required; typically three keys (``fan``, ``isce``, ``insardev``).
    valid : ndarray of bool, optional
        Extra valid mask (e.g. finite unwrapped phase in all stacks). Combined
        with finite/coh thresholds.
    min_coherence : float, optional
        Each stack must exceed this at the candidate pixel.
    smooth_window : int, optional
        Odd window for median filter on the mean-coherence map before
        ``argmax``. Suppresses single-pixel speckles while still selecting by
        mean-of-three coherence (objective's definition). ``1`` disables smoothing.

    Returns
    -------
    CommonRefPoint
        Selected row/col and per-stack coherence.

    Raises
    ------
    InvalidProcessingStateError
        If no pixel is valid in all stacks.

    """
    if not coherences:
        reject_invalid_state("coherences dict must be non-empty")
    labels = list(coherences.keys())
    arrays = [np.asarray(coherences[k], dtype=np.float64) for k in labels]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        reject_invalid_state("all coherence arrays must share the same shape")

    stack = np.stack(arrays, axis=0)
    finite = np.all(np.isfinite(stack), axis=0)
    above = np.all(stack >= float(min_coherence), axis=0)
    mask = finite & above
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        if v.shape != shape:
            reject_invalid_state("valid mask must match coherence shape")
        mask &= v
    if not np.any(mask):
        reject_invalid_state(
            "no triple-valid high-coherence pixel for common reference"
        )

    mean_coh = np.mean(stack, axis=0)
    score = mean_coh.copy()
    score[~mask] = -np.inf
    win = int(smooth_window)
    if win > 1:
        if win % 2 == 0:
            win += 1
        # Median only over finite mean; non-mask stays -inf after restore
        med = median_filter(
            np.where(mask, mean_coh, 0.0), size=win, mode="nearest"
        )
        score = np.where(mask, med, -np.inf)

    flat_idx = int(np.argmax(score))
    row, col = np.unravel_index(flat_idx, shape)
    row, col = int(row), int(col)
    per = {lab: float(stack[i, row, col]) for i, lab in enumerate(labels)}
    mean_v = float(np.mean(list(per.values())))
    logger.info(
        "Common ref pixel row=%d col=%d mean_coh=%.4f (%s)",
        row,
        col,
        mean_v,
        ", ".join(f"{k}={v:.3f}" for k, v in per.items()),
    )
    return CommonRefPoint(
        row=row, col=col, mean_coherence=mean_v, coherences=per
    )


def load_pinned_ref_point(
    path: str | Path | None = None,
    *,
    coherences: dict[str, np.ndarray] | None = None,
) -> CommonRefPoint:
    """Load a frozen common-ref JSON; do not re-run argmax.

    Parameters
    ----------
    path : path, optional
        Defaults to the three-way campaign pin
        ``reports/2026-07-16-three-way-unw-geocode-compare/common_ref_point.json``
        (row=175, col=2276).
    coherences : dict, optional
        If given, fill ``coherences`` / ``mean_coherence`` from these layers
        at the pinned pixel (for bookkeeping only).

    Returns
    -------
    CommonRefPoint
        Fixed row/col from disk.

    """
    p = Path(path) if path is not None else DEFAULT_PINNED_REF_JSON
    if not p.is_file():
        reject_invalid_state(f"pinned common ref not found: {p}")
    data: dict[str, Any] = json.loads(p.read_text())
    row = int(data["row"])
    col = int(data["col"])
    per: dict[str, float] = {}
    if coherences:
        for k, arr in coherences.items():
            a = np.asarray(arr)
            if 0 <= row < a.shape[0] and 0 <= col < a.shape[1] and np.isfinite(
                a[row, col]
            ):
                per[k] = float(a[row, col])
            else:
                per[k] = float("nan")
        mean_v = float(np.nanmean(list(per.values()))) if per else float("nan")
    else:
        per = {
            k: float(v)
            for k, v in (data.get("coherences") or {}).items()
            if v is not None
        }
        raw_mean = data.get("mean_coherence", float("nan"))
        mean_v = float("nan") if raw_mean is None else float(raw_mean)
    logger.info(
        "Pinned common ref row=%d col=%d (from %s; not re-selected)",
        row,
        col,
        p,
    )
    return CommonRefPoint(
        row=row, col=col, mean_coherence=mean_v, coherences=per
    )


def resolve_common_ref_point(
    coherences: dict[str, np.ndarray],
    *,
    valid: np.ndarray | None = None,
    min_coherence: float = 0.15,
    smooth_window: int = 5,
    pinned_path: str | Path | None = DEFAULT_PINNED_REF_JSON,
    allow_auto: bool = False,
) -> CommonRefPoint:
    """Prefer pinned ref; optionally fall back to auto select.

    Default is **pinned** so residual figures stay comparable across runs.
    Set ``pinned_path=None`` and ``allow_auto=True`` only when intentionally
    re-selecting.
    """
    if pinned_path is not None:
        p = Path(pinned_path)
        if p.is_file():
            ref = load_pinned_ref_point(p, coherences=coherences)
            # Soft check: warn if pinned pixel is invalid but still return it
            if valid is not None:
                v = np.asarray(valid, dtype=bool)
                if (
                    0 <= ref.row < v.shape[0]
                    and 0 <= ref.col < v.shape[1]
                    and not v[ref.row, ref.col]
                ):
                    logger.warning(
                        "Pinned ref (%d,%d) is outside current valid mask; "
                        "keeping pin for consistency",
                        ref.row,
                        ref.col,
                    )
            return ref
        if not allow_auto:
            reject_invalid_state(
                f"pinned common ref missing at {p}; "
                "pass allow_auto=True to re-select"
            )
    if not allow_auto:
        reject_invalid_state(
            "no pinned ref and allow_auto=False; refuse to move the reference"
        )
    return select_common_ref_point(
        coherences,
        valid=valid,
        min_coherence=min_coherence,
        smooth_window=smooth_window,
    )


def dual_std(
    residual: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Circular residual std on finite pixels (ship-gate dual_std).

    Parameters
    ----------
    residual : ndarray
        Residual phase (typically after pin subtract).
    valid_mask : ndarray of bool, optional
        Extra mask; combined with finite residual.

    Returns
    -------
    float
        ``sqrt(mean(angle(exp(1j·φ))²))`` over selected pixels, or NaN.

    """
    arr = np.asarray(residual, dtype=np.float64)
    mask = np.isfinite(arr)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    z = np.exp(1j * arr[mask])
    return float(np.sqrt(np.mean(np.angle(z) ** 2)))


def subtract_ref_value(
    unwrapped: np.ndarray,
    ref: CommonRefPoint | tuple[int, int],
) -> np.ndarray:
    """Subtract unwrapped phase at the reference pixel (NaN-safe).

    Parameters
    ----------
    unwrapped : ndarray
        Unwrapped phase on the shared grid.
    ref : CommonRefPoint or (row, col)
        Reference location.

    Returns
    -------
    ndarray
        Residual phase with value ~0 at the reference when the ref is finite.

    """
    arr = np.asarray(unwrapped, dtype=np.float64)
    if isinstance(ref, CommonRefPoint):
        r, c = ref.row, ref.col
    else:
        r, c = int(ref[0]), int(ref[1])
    if not (0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]):
        reject_invalid_state("reference index out of bounds")
    ref_val = arr[r, c]
    if not np.isfinite(ref_val):
        reject_invalid_state("reference pixel unwrapped phase is not finite")
    out = arr - ref_val
    return out.astype(np.float32, copy=False)


def shared_residual_limits(
    residuals: dict[str, np.ndarray],
    *,
    percentile: float = 99.0,
    min_half_range: float = 0.5,
) -> tuple[float, float]:
    """Symmetric shared vmin/vmax from pooled residual percentiles.

    Parameters
    ----------
    residuals : dict of ndarray
        Residual maps after reference subtraction.
    percentile : float, optional
        Absolute-value percentile of pooled finite residuals.
    min_half_range : float, optional
        Floor on half-range so a near-zero residual still has a visible scale.

    Returns
    -------
    vmin, vmax : float
        Symmetric limits suitable for a shared Normalize.

    """
    chunks: list[np.ndarray] = []
    for arr in residuals.values():
        a = np.asarray(arr, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            chunks.append(a)
    if not chunks:
        return -min_half_range, min_half_range
    pooled = np.concatenate(chunks)
    half = float(np.percentile(np.abs(pooled), percentile))
    half = max(half, float(min_half_range))
    return -half, half
