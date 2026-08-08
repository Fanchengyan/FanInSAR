"""Network misregistration invert (NESD-class): pair arcs → per-date az/rg.

Mission-neutral linear algebra. Measurement quality and arc QC are the
caller's responsibility (PROPOSAL-0017).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class MisregArc:
    """One pair observation of residual misregistration."""

    primary: str
    """Earlier (or design-matrix primary) date id ``YYYYMMDD``."""
    secondary: str
    """Later (or design-matrix secondary) date id."""
    azimuth_shift_px: float
    range_shift_px: float
    azimuth_sigma_px: float = 1.0
    range_sigma_px: float = 1.0
    method: str = "unknown"
    n_valid: int = 0

    @property
    def pair_id(self) -> str:
        """Canonical ``primary_secondary`` id."""
        return f"{self.primary}_{self.secondary}"


@dataclass(frozen=True, slots=True)
class DateMisreg:
    """Per-date rigid misregistration relative to a master date."""

    master: str
    azimuth_px: Mapping[str, float]
    range_px: Mapping[str, float]
    metadata: Mapping[str, object]


def _design_matrix(
    arcs: list[MisregArc],
    dates: list[str],
    master: str,
) -> tuple[np.ndarray, list[str]]:
    """Build pair-difference design matrix with master column dropped."""
    free = [d for d in dates if d != master]
    col = {d: i for i, d in enumerate(free)}
    n_arc = len(arcs)
    n_free = len(free)
    g = np.zeros((n_arc, n_free), dtype=np.float64)
    for i, arc in enumerate(arcs):
        if arc.primary not in dates or arc.secondary not in dates:
            reject_invalid_state(
                f"arc {arc.pair_id} references unknown dates in network",
            )
        if arc.primary != master:
            g[i, col[arc.primary]] = -1.0
        if arc.secondary != master:
            g[i, col[arc.secondary]] = 1.0
    return g, free


def _weighted_lstsq(
    g: np.ndarray,
    d: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Solve weighted least squares; reject rank-deficient systems."""
    if g.size == 0:
        return np.zeros(0, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
        reject_invalid_state("arc sigma must be finite and positive")
    w = 1.0 / sigma
    gw = g * w[:, None]
    dw = d * w
    rank = int(np.linalg.matrix_rank(gw, tol=1e-8))
    if rank < g.shape[1]:
        reject_invalid_state(
            f"misreg network is rank-deficient (rank={rank}, n_param={g.shape[1]})",
        )
    x, residuals, rank_out, _ = np.linalg.lstsq(gw, dw, rcond=None)
    _ = residuals, rank_out
    return x


def invert_pair_misregistration(
    arcs: list[MisregArc] | tuple[MisregArc, ...],
    *,
    master: str,
    dates: list[str] | tuple[str, ...] | None = None,
    min_n_valid: int = 0,
    max_sigma_px: float = 1e3,
    on_empty: str = "error",
) -> DateMisreg:
    """Invert pair misreg arcs to per-date az/rg (master fixed at 0).

    Parameters
    ----------
    arcs : sequence of MisregArc
        Observed pair residuals. ``secondary − primary`` convention matches
        the design matrix used here.
    master : str
        Reference date id fixed at zero misregistration.
    dates : sequence of str, optional
        Full date list. Default: master plus all dates appearing in arcs.
    min_n_valid : int, optional
        Drop arcs with fewer valid samples (when reported).
    max_sigma_px : float, optional
        Drop arcs with larger az or rg sigma.
    on_empty : {"error", "zeros"}, optional
        Behavior when no arcs remain after QC.

    Returns
    -------
    DateMisreg
        Maps date → az/rg px with ``master`` at 0.

    """
    if not master:
        reject_invalid_state("master date id is required")
    kept: list[MisregArc] = []
    for arc in arcs:
        if arc.n_valid < min_n_valid:
            continue
        if arc.azimuth_sigma_px > max_sigma_px or arc.range_sigma_px > max_sigma_px:
            continue
        if not np.isfinite(arc.azimuth_shift_px) or not np.isfinite(arc.range_shift_px):
            continue
        kept.append(arc)

    if dates is None:
        date_set: set[str] = {master}
        for arc in kept:
            date_set.add(arc.primary)
            date_set.add(arc.secondary)
        date_list = sorted(date_set)
    else:
        date_list = list(dates)
        if master not in date_list:
            reject_invalid_state("master must be included in dates")

    if not kept:
        if on_empty == "zeros":
            zeros = dict.fromkeys(date_list, 0.0)
            return DateMisreg(
                master=master,
                azimuth_px=zeros,
                range_px=dict(zeros),
                metadata={"n_arcs": 0, "n_dates": len(date_list), "empty": True},
            )
        reject_invalid_state("no misreg arcs remain after QC")

    # Connectivity: every free date must appear in at least one arc.
    touched = {master}
    for arc in kept:
        touched.add(arc.primary)
        touched.add(arc.secondary)
    missing = [d for d in date_list if d not in touched]
    if missing:
        reject_invalid_state(
            f"misreg network disconnected; dates without arcs: {missing}",
        )

    g, free = _design_matrix(kept, date_list, master)
    d_az = np.array([a.azimuth_shift_px for a in kept], dtype=np.float64)
    d_rg = np.array([a.range_shift_px for a in kept], dtype=np.float64)
    s_az = np.array([a.azimuth_sigma_px for a in kept], dtype=np.float64)
    s_rg = np.array([a.range_sigma_px for a in kept], dtype=np.float64)
    x_az = _weighted_lstsq(g, d_az, s_az)
    x_rg = _weighted_lstsq(g, d_rg, s_rg)

    az_map = {master: 0.0}
    rg_map = {master: 0.0}
    for i, d in enumerate(free):
        az_map[d] = float(x_az[i])
        rg_map[d] = float(x_rg[i])
    for d in date_list:
        az_map.setdefault(d, 0.0)
        rg_map.setdefault(d, 0.0)

    residual_az = d_az - g @ x_az
    residual_rg = d_rg - g @ x_rg
    logger.info(
        "Inverted misreg network: %s arcs, %s dates, master=%s, "
        "rms_az=%.4f rms_rg=%.4f",
        len(kept),
        len(date_list),
        master,
        float(np.sqrt(np.mean(residual_az**2))) if kept else 0.0,
        float(np.sqrt(np.mean(residual_rg**2))) if kept else 0.0,
    )
    return DateMisreg(
        master=master,
        azimuth_px=az_map,
        range_px=rg_map,
        metadata={
            "n_arcs": len(kept),
            "n_dates": len(date_list),
            "n_free": len(free),
            "rms_az_px": float(np.sqrt(np.mean(residual_az**2))),
            "rms_rg_px": float(np.sqrt(np.mean(residual_rg**2))),
            "methods": sorted({a.method for a in kept}),
        },
    )
