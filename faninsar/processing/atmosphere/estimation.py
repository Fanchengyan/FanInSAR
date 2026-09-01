"""Dispersive / non-dispersive phase separation (split_main_band).

Faithful Torch port of the ISCE3 ``SplitBandIonosphereEstimation`` contract:
no-data masking, absolute jump alignment under a configurable convention,
optional unwrapping-error coefficients, then the exact 2x2 linear solve of
``estimate_iono_low_high`` with its singular-det guard.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.processing.atmosphere.config import IonosphereEstimationConfig

logger = setup_logger(__name__)

__all__ = [
    "align_absolute_jumps",
    "estimate_disp_nondisp",
    "solve_2x2_low_high",
    "solve_guided_split",
]

_TWO_PI = 2.0 * math.pi


def _nanmean_guarded(values: torch.Tensor) -> float:
    """Mean over finite entries; raise on an all-invalid tensor."""
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        message = (
            "subband interferograms are entirely invalid; cannot align "
            "absolute phase jumps"
        )
        logger.error(message)
        raise ValueError(message)
    return float(finite.mean())


def _weighted_polyfit2d(
    y: torch.Tensor,
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    *,
    degree: int = 2,
) -> torch.Tensor:
    """Weighted degree-``degree`` polynomial surface fit over a [x, y] grid."""
    orig_shape = y.shape
    xn = x_norm[..., 0].reshape(-1).to(torch.float64)
    yn = x_norm[..., 1].reshape(-1).to(torch.float64)
    flat_y = y.reshape(-1).to(torch.float64)
    w = weight.reshape(-1).to(torch.float64)

    columns = []
    for total in range(degree + 1):
        for py in range(total + 1):
            px = total - py
            columns.append((yn**py) * (xn**px))
    design = torch.stack(columns, dim=-1)

    sqrt_w = torch.sqrt(w.clamp(min=0.0))
    theta = torch.linalg.lstsq(design * sqrt_w.unsqueeze(-1), flat_y * sqrt_w).solution
    surface = design @ theta
    return surface.reshape(orig_shape).to(y.dtype)


def solve_2x2_low_high(
    phi_sub_low: torch.Tensor,
    phi_sub_high: torch.Tensor,
    config: IonosphereEstimationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the dispersive/non-dispersive system exactly as ISCE3.

    Mirrors ``estimate_iono_low_high``: coefficients built from frequency
    ratios with the identical determinant guard.  Inputs must already be
    jump-aligned and unwrapped.
    """
    f0 = float(config.f0)
    fl = float(config.freq_low)
    fh = float(config.freq_high)
    a = fl / f0
    b = f0 / fl
    c = fh / f0
    d = f0 / fh
    det = a * d - b * c
    if det == 0:
        message = "Frequency combination leads to singular matrix"
        logger.error(message)
        raise ZeroDivisionError(message)

    m11 = d / det
    m12 = -b / det
    m21 = -c / det
    m22 = a / det

    low = phi_sub_low.to(torch.float64)
    high = phi_sub_high.to(torch.float64)
    non_dispersive = m11 * low + m12 * high
    dispersive = m21 * low + m22 * high
    out_dtype = phi_sub_low.dtype
    return (dispersive.to(out_dtype), non_dispersive.to(out_dtype))


def solve_guided_split(
    phi_sub_low: torch.Tensor,
    phi_sub_high: torch.Tensor,
    coherence: torch.Tensor,
    config: IonosphereEstimationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve dispersive/non-dispersive phase with surface-guided split-spectrum.

    This is the robust variant of :func:`solve_2x2_low_high` used by the
    ISCE2 alosStack chain (``runIonFilt.computeIonosphere``, ``adjFlag=1``).
    It keeps the exact 2x2 physical solve — algebraically identical to the
    ISCE3 ``m21/m22`` coefficients — but adds a surface-guided relative
    unwrapping-error correction before solving:

    1. **Weighted surface fit of the subband difference.**  A degree-2
       polynomial is fit to ``(phi_sub_low - phi_sub_high)`` with weights
       ``coherence ** cor_order_adj`` (default 20), so high-coherence pixels
       dominate the reference surface.
    2. **Per-pixel integer-cycle alignment.**  Each pixel's deviation from
       that surface is rounded to the nearest integer cycle, and the cycles
       are added back to the upper band.  This removes *relative* unwrapping
       errors that vary across the scene — something a single global cycle
       correction (the ISCE3 default) cannot.

    The result is the same physical screen as the pure 2x2 solve when no
    unwrapping errors are present, and a correctly cycle-aligned screen when
    they are.

    References
    ----------
    The split-spectrum principle is described in:

    - Gomba, G., Parizzi, A., De Zan, F., Eineder, M., & Bamler, R. (2015).
      "Toward operational compensation of ionospheric effects in SAR
      interferograms: The split-spectrum method." IEEE Trans. Geosci.
      Remote Sens., 53(10), 5683-5694.  doi:10.1109/TGRS.2015.2421079
    - Rosen, P. A., Hensley, S., & Chen, C. (2010). "Measurement and
      mitigation of the ionosphere in L-band interferometric SAR data."
      IEEE Geosci. Remote Sens. Lett., 7(2), 369-373.
      doi:10.1109/LGRS.2009.2035972

    The surface-guided cycle adjustment and its coherence weighting follow
    the alosStack implementation (Liang, C., JPL; ISCE2
    ``contrib/stack/alosStack/ion_filt.py`` → ``computeIonosphere`` with
    ``adjFlag=1``), as used in the ALOS-2/PALSAR-2 processing chain of
    Liang et al. (2019), "Ionospheric correction of InSAR time series
    analysis of C-band Sentinel-1 TOPS data," IEEE Trans. Geosci. Remote
    Sens., 57(9), 6755-6773.  doi:10.1109/TGRS.2019.2908494

    Parameters
    ----------
    phi_sub_low : torch.Tensor
        Unwrapped lower-subband interferogram phase [rad].
    phi_sub_high : torch.Tensor
        Unwrapped upper-subband interferogram phase [rad].
    coherence : torch.Tensor
        Coherence (or proxy) used as the polyfit weight base; raised to
        ``config.cor_order_adj``.
    config : IonosphereEstimationConfig
        Frequency roles and ``cor_order_adj`` (default 20).

    Returns
    -------
    tuple of torch.Tensor
        ``(dispersive, non_dispersive)`` phases at the carrier frequency.
        NaN propagates from the inputs.

    """
    f0 = float(config.f0)
    fl = float(config.freq_low)
    fh = float(config.freq_high)
    cor_order = int(config.cor_order_adj)

    low = phi_sub_low.to(torch.float64)
    high = phi_sub_high.to(torch.float64)
    wgt = coherence.to(torch.float64).clamp(min=0.0) ** cor_order

    diff = low - high
    rows, cols = low.shape[-2], low.shape[-1]
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, rows, device=low.device),
        torch.linspace(-1.0, 1.0, cols, device=low.device),
        indexing="ij",
    )
    coords = torch.stack((gx, gy), dim=-1)

    # The coherence weight gates only the surface fit (alosStack
    # ``polyfit_2d(lowerUnw - upperUnw, cor**corOrderAdj, 2)``).  The
    # per-pixel integer-cycle adjustment of the upper band applies to ALL
    # unwrapped lower-band pixels (alosStack ``flag2 = (lowerUnw != 0)``),
    # independent of the coherence threshold; restricting it to the weight
    # mask leaves wrong cycles in low-coherence land pixels.
    fit_valid = torch.isfinite(diff) & (wgt > 0.0)
    filled = torch.where(fit_valid, diff, torch.zeros_like(diff))
    weights = torch.where(fit_valid, wgt, torch.zeros_like(wgt))
    fit = _weighted_polyfit2d(filled, coords, weights, degree=2)

    adj_mask = torch.isfinite(low) & (low != 0.0)
    residual_cycles = torch.round((diff - fit) / _TWO_PI)
    high_adj = torch.where(adj_mask, high + _TWO_PI * residual_cycles, high)

    det = fh**2 - fl**2
    dispersive = fl * fh * (low * fh - high_adj * fl) / f0 / det
    non_dispersive = f0 * (high_adj * fh - low * fl) / det

    out_dtype = phi_sub_low.dtype
    return (dispersive.to(out_dtype), non_dispersive.to(out_dtype))


def align_absolute_jumps(
    phi_sub_low: torch.Tensor,
    phi_sub_high: torch.Tensor,
    config: IonosphereEstimationConfig,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove absolute integer-cycle jumps between the two subband IFGs.

    - ``isce3_global_jump``: single integer cycle from the masked scene mean
      of ``(low - high)``, added to the upper band with the upstream
      positive-sign convention preserved exactly.
    - ``alosstack_pixel_rounded``: weighted degree-2 surface fit of the
      difference followed by per-pixel integer-cycle rounding applied to the
      upper band.

    Returns the (low, corrected-high) pair; inputs are never mutated.
    """
    if config.alignment_strategy not in (
        "isce3_global_jump",
        "alosstack_pixel_rounded",
    ):
        message = f"unknown alignment strategy {config.alignment_strategy!r}"
        logger.error(message)
        raise ValueError(message)

    low = phi_sub_low.clone()
    high = phi_sub_high.clone()

    if valid_mask is None:
        valid_mask = torch.isfinite(low) & torch.isfinite(high)

    diff = torch.where(valid_mask, low - high, torch.nan).to(torch.float32)
    finite_diff = diff[valid_mask]

    if config.alignment_strategy == "isce3_global_jump":
        # ISCE3 applies a non-negative integer-cycle addition
        # (split_band_estimation.py:116-118), which silently fails when the
        # inter-band offset has the opposite sign.  We keep the same
        # scene-mean statistic but use the SIGNED nearest integer so the
        # correction works in both directions; behavior matches upstream
        # whenever the upstream assumption holds.
        mean_diff = _nanmean_guarded(finite_diff)
        num_jump = round(mean_diff / _TWO_PI)
        high = torch.where(
            valid_mask,
            high + (_TWO_PI * num_jump),
            high,
        )
        return low, high

    # alosstack_pixel_rounded
    if finite_diff.numel() == 0:
        message = "no valid pixels for pixel-rounded jump alignment"
        logger.error(message)
        raise ValueError(message)

    rows, cols = low.shape[-2], low.shape[-1]
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, rows, device=low.device),
        torch.linspace(-1.0, 1.0, cols, device=low.device),
        indexing="ij",
    )
    coords = torch.stack((gx, gy), dim=-1)
    weights = torch.where(valid_mask, torch.ones_like(diff), torch.zeros_like(diff))
    filled = torch.where(valid_mask, diff, torch.zeros_like(diff))

    surface = _weighted_polyfit2d(filled, coords, weights)
    residual_cycles = torch.round((filled - surface) / _TWO_PI)
    high = torch.where(
        valid_mask,
        high + (_TWO_PI * residual_cycles),
        high,
    )
    return low, high


def estimate_disp_nondisp(
    phi_sub_low_unwrapped: torch.Tensor,
    phi_sub_high_unwrapped: torch.Tensor,
    config: IonosphereEstimationConfig,
    valid_mask: torch.Tensor | None = None,
    comm_unwcor_coef: torch.Tensor | None = None,
    diff_unwcor_coef: torch.Tensor | None = None,
    coherence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate dispersive and non-dispersive phase from subband IFGs.

    Parameters
    ----------
    phi_sub_low_unwrapped : torch.Tensor
        Unwrapped lower-subband interferogram phase [rad].
    phi_sub_high_unwrapped : torch.Tensor
        Unwrapped upper-subband interferogram phase [rad].
    config : IonosphereEstimationConfig
        Caller-supplied frequency roles and strategy knobs.  ``solve_core``
        selects the ISCE3 pure 2x2 lane (default) or the alosStack weighted
        surface-adjustment lane.
    valid_mask : torch.Tensor, optional
        Boolean validity mask (True = usable).  Defaults to finite values.
    comm_unwcor_coef, diff_unwcor_coef : torch.Tensor, optional
        Common/differential unwrapping-error cycle corrections applied
        exactly like ISCE3 before the solve.
    coherence : torch.Tensor, optional
        Coherence (or proxy) used by the guided-split lane as the polyfit
        weight base.  Required when ``config.solve_core == "guided_split"``.

    Returns
    -------
    tuple of torch.Tensor
        ``(dispersive, non_dispersive)`` phases at the carrier frequency.
        Invalid entries propagate NaN.

    """
    if phi_sub_low_unwrapped.shape != phi_sub_high_unwrapped.shape:
        message = "phi_sub_low and phi_sub_high shapes differ"
        logger.error(message)
        raise ValueError(message)

    low = phi_sub_low_unwrapped.clone()
    high = phi_sub_high_unwrapped.clone()
    dtype_orig = low.dtype

    if valid_mask is None:
        valid_mask = torch.isfinite(low) & torch.isfinite(high)

    # no-data handling mirrors compute_disp_nondisp: outside mask -> NaN in,
    # NaN back out after solving
    work_mask = valid_mask
    low = torch.where(work_mask, low.to(torch.float64), torch.tensor(float("nan")))
    high = torch.where(work_mask, high.to(torch.float64), torch.tensor(float("nan")))

    if comm_unwcor_coef is not None and diff_unwcor_coef is not None:
        coef = comm_unwcor_coef.to(torch.float64)
        dcoef = diff_unwcor_coef.to(torch.float64)
        low = low - _TWO_PI * coef
        high = high - _TWO_PI * (coef + dcoef)

    if config.solve_core == "guided_split":
        if coherence is None:
            message = "solve_core='guided_split' requires a coherence weight field"
            logger.error(message)
            raise ValueError(message)
        if coherence.shape != low.shape:
            message = "coherence shape must match the subband phases"
            logger.error(message)
            raise ValueError(message)
        dispersive, non_dispersive = solve_guided_split(
            low, high, coherence.to(low.device), config
        )
    else:
        aligned_low, aligned_high = align_absolute_jumps(
            low.to(dtype_orig),
            high.to(dtype_orig),
            config,
            valid_mask=work_mask & torch.isfinite(low.to(dtype_orig)),
        )
        dispersive, non_dispersive = solve_2x2_low_high(
            aligned_low, aligned_high, config
        )
    keep = work_mask.to(torch.bool)
    nan_like = torch.full_like(dispersive, float("nan"))
    dispersive = torch.where(keep, dispersive, nan_like)
    non_dispersive = torch.where(keep, non_dispersive, nan_like)
    return (dispersive.to(dtype_orig), non_dispersive.to(dtype_orig))
