"""Burst-overlap Enhanced Spectral Diversity (ESD) azimuth estimator.

Forms original-domain double-difference interferograms on adjacent TOPS
burst overlaps and converts the DD phase to an azimuth shift with the
secondary full-carrier response ``R_B = dphi_DD / ddelta_B`` evaluated at true
source coordinates (ISCE2 PR 28139).

A measurement is omitted (``None``) when no sample passes the coherence
gate; the estimator never stuffs a 0-shift observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.coregistration.tops.deramp import (
    TOPSCarrierModel,
    carrier_phase_at_points,
)
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = setup_logger(__name__)

_RESPONSE_EPS_PX = 0.5
_MIN_ABS_RESPONSE_RAD_PX = 1e-3

__all__ = [
    "OverlapESDAggregate",
    "OverlapESDInput",
    "OverlapESDRecord",
    "aggregate_overlap_esd",
    "estimate_overlap_esd",
]


@dataclass(frozen=True, slots=True)
class OverlapESDRecord:
    """Per-overlap azimuth residual and carrier-response diagnostics."""

    top_index: int
    """Burst index of the earlier (top) overlap member."""
    bottom_index: int
    """Burst index of the later (bottom) overlap member."""
    azimuth_shift_px: float
    """Residual azimuth shift in pixels (positive = secondary lags reference)."""
    coherence: float
    """Mean coherence of qualifying overlap cells after multilook (0..1)."""
    n_valid: int
    """Number of multilook cells that passed the coherence gate."""
    response_rb_rad_px: float
    """Median ``|R_B|`` over qualifying samples, in radians per pixel."""
    response_consistency: float
    """Median ``|R_A - R_B| / |R_B|`` (NaN when reference carriers are omitted)."""


@dataclass(frozen=True, slots=True)
class OverlapESDAggregate:
    """Pair-level overlap-ESD residual pooled from qualifying overlaps."""

    azimuth_shift_px: float
    """Coherence times sample-count weighted mean azimuth shift in pixels."""
    sigma_px: float
    """Weighted scatter of the per-overlap shifts, floored for invert legality."""
    n_overlaps: int
    """Number of overlap records that entered the aggregate."""
    n_valid: int
    """Sum of qualifying sample counts across those records."""
    method: str
    """Resolved method name (always ``overlap``)."""
    diagnostics: Mapping[str, object]
    """Per-aggregate QC fields (mean coherence, consistency, sigma floor)."""


@dataclass(frozen=True, slots=True)
class OverlapESDInput:
    """One adjacent-burst overlap; every array shares one crop shape.

    Secondary ``row0`` / ``col0`` / ``native_height`` are burst-absolute for
    **that secondary burst**. When ``ref_*_carrier`` is set, the matching
    ``ref_*_native_height`` (and row/col origins) are required so ``R_A``
    cannot fall back to a crop-local centre row.
    """

    ref_top: np.ndarray
    """Reference burst *i* overlap crop, original focused domain."""
    ref_bot: np.ndarray
    """Reference burst *i* + 1 overlap crop, original focused domain."""
    sec_top: np.ndarray
    """Secondary burst *i* on the reference crop, original focused domain."""
    sec_bot: np.ndarray
    """Secondary burst *i* + 1 on the reference crop, original focused domain."""
    sec_top_carrier: TOPSCarrierModel
    """Native-burst carrier of secondary burst *i*."""
    sec_bot_carrier: TOPSCarrierModel
    """Native-burst carrier of secondary burst *i* + 1."""
    top_row0: int
    """First overlap line inside secondary burst *i* (burst-local)."""
    bot_row0: int
    """First overlap line inside secondary burst *i* + 1 (burst-local)."""
    top_col0: int
    """First overlap range sample inside secondary burst *i*."""
    bot_col0: int
    """First overlap range sample inside secondary burst *i* + 1."""
    top_native_height: int
    """Full line count of secondary burst *i* (never the crop height)."""
    bot_native_height: int
    """Full line count of secondary burst *i* + 1."""
    top_offsets: tuple[np.ndarray, np.ndarray]
    """Geometry-only ``(azimuth, range)`` offset crops for the top resample."""
    bot_offsets: tuple[np.ndarray, np.ndarray]
    """Geometry-only ``(azimuth, range)`` offset crops for the bottom resample."""
    ref_top_carrier: TOPSCarrierModel | None = None
    """Optional reference burst-*i* carrier for the ``R_A`` diagnostic."""
    ref_bot_carrier: TOPSCarrierModel | None = None
    """Optional reference burst-*(i+1)* carrier for the ``R_A`` diagnostic."""
    ref_top_row0: int | None = None
    """Window start inside the reference top burst."""
    ref_bot_row0: int | None = None
    """Window start inside the reference bottom burst."""
    ref_top_col0: int | None = None
    """Window start in range inside the reference top burst."""
    ref_bot_col0: int | None = None
    """Window start in range inside the reference bottom burst."""
    ref_top_native_height: int | None = None
    """Full line count of the reference top burst (required with its carrier)."""
    ref_bot_native_height: int | None = None
    """Full line count of the reference bottom burst (required with its carrier)."""
    top_index: int = 0
    """Burst index of the top overlap member, copied onto the record."""
    bottom_index: int = 1
    """Burst index of the bottom overlap member, copied onto the record."""


def _boxcar_mean(arr: np.ndarray, looks: tuple[int, int]) -> np.ndarray:
    """Return the non-overlapping boxcar mean of a 2-D array."""
    az_looks, rg_looks = looks
    if az_looks < 1 or rg_looks < 1:
        reject_invalid_state("multilook factors must be >= 1")
    height, width = arr.shape
    if az_looks == 1 and rg_looks == 1:
        return np.asarray(arr)
    trimmed_h = (height // az_looks) * az_looks
    trimmed_w = (width // rg_looks) * rg_looks
    if trimmed_h == 0 or trimmed_w == 0:
        reject_invalid_state("overlap crop is smaller than the multilook window")
    cropped = arr[:trimmed_h, :trimmed_w]
    out_h = trimmed_h // az_looks
    out_w = trimmed_w // rg_looks
    reshaped = cropped.reshape(out_h, az_looks, out_w, rg_looks)
    return reshaped.mean(axis=(1, 3))


def _multilooked_ifg_coherence(
    reference: np.ndarray,
    secondary: np.ndarray,
    looks: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the multilooked complex IFG and its MLE coherence."""
    ifg = reference * np.conjugate(secondary)
    power_ref = np.abs(reference).astype(np.float64) ** 2
    power_sec = np.abs(secondary).astype(np.float64) ** 2
    ifg_ml = _boxcar_mean(ifg, looks)
    power_ref_ml = _boxcar_mean(power_ref, looks)
    power_sec_ml = _boxcar_mean(power_sec, looks)
    denom = np.sqrt(np.maximum(power_ref_ml * power_sec_ml, 1e-30))
    coherence = np.clip(np.abs(ifg_ml) / denom, 0.0, 1.0)
    return ifg_ml, coherence.astype(np.float64, copy=False)


def _source_coordinates(
    shape: tuple[int, int],
    row0: int,
    col0: int,
    offsets: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Burst-absolute source coordinates ``crop - offsets + (row0, col0)``."""
    n_lines, n_samples = shape
    az_off = np.asarray(offsets[0], dtype=np.float64)
    rg_off = np.asarray(offsets[1], dtype=np.float64)
    if az_off.shape != shape or rg_off.shape != shape:
        reject_invalid_state("offset crops must match the overlap array shape")
    src_row = np.arange(n_lines, dtype=np.float64)[:, None] + float(row0) - az_off
    src_col = np.arange(n_samples, dtype=np.float64)[None, :] + float(col0) - rg_off
    return np.broadcast_arrays(src_row, src_col)


def _carrier_response_rad_px(
    model: TOPSCarrierModel,
    src_row: np.ndarray,
    src_col: np.ndarray,
    *,
    native_height: int,
) -> np.ndarray:
    """Full-carrier azimuth response ``dphi/drow`` at source coordinates.

    The instantaneous frequency is the derivative of the TOPS carrier
    ``pi K_t (eta - eta_ref)^2 + 2 pi f_dc eta``, never the Doppler-centroid term
    alone. ``centre_row`` is the native burst centre.
    """
    if native_height <= 0:
        reject_invalid_state("native_height must be a positive burst line count")
    centre_row = float(native_height // 2)
    phi_plus = carrier_phase_at_points(
        model,
        src_row + _RESPONSE_EPS_PX,
        src_col,
        centre_row=centre_row,
        dtype=np.float64,
    )
    phi_minus = carrier_phase_at_points(
        model,
        src_row - _RESPONSE_EPS_PX,
        src_col,
        centre_row=centre_row,
        dtype=np.float64,
    )
    return (phi_plus - phi_minus) / (2.0 * _RESPONSE_EPS_PX)


def _require_matching_complex_crop(
    name: str,
    arr: np.ndarray,
    shape: tuple[int, int],
) -> None:
    """Reject a crop that is not a matching 2-D complex array."""
    if arr.shape != shape or arr.ndim != 2:
        reject_invalid_state(f"{name} must match overlap crop shape {shape}")
    if not np.iscomplexobj(arr):
        reject_invalid_state(f"{name} must be complex-valued")


def _validate_overlap_input(data: OverlapESDInput) -> tuple[int, int]:
    """Return the common crop shape after rejecting malformed inputs."""
    if data.ref_top.ndim != 2:
        reject_invalid_state("overlap ESD requires 2-D burst crops")
    shape = data.ref_top.shape
    for name in ("ref_top", "ref_bot", "sec_top", "sec_bot"):
        _require_matching_complex_crop(name, getattr(data, name), shape)
    if data.top_native_height <= 0 or data.bot_native_height <= 0:
        reject_invalid_state(
            "secondary native_height must be a positive burst line count"
        )
    if data.top_native_height < data.top_row0 + shape[0]:
        reject_invalid_state(
            "top_native_height must span the burst-absolute window "
            "(row0 + crop lines); crop-local native_height is invalid"
        )
    if data.bot_native_height < data.bot_row0 + shape[0]:
        reject_invalid_state(
            "bot_native_height must span the burst-absolute window "
            "(row0 + crop lines); crop-local native_height is invalid"
        )
    ref_top = data.ref_top_carrier
    ref_bot = data.ref_bot_carrier
    if (ref_top is None) ^ (ref_bot is None):
        reject_invalid_state(
            "ref_top_carrier and ref_bot_carrier must be provided together"
        )
    if ref_top is not None:
        if data.ref_top_native_height is None or data.ref_bot_native_height is None:
            reject_invalid_state(
                "ref_top_native_height and ref_bot_native_height are required "
                "when ref carriers are set"
            )
        if (
            data.ref_top_row0 is None
            or data.ref_bot_row0 is None
            or data.ref_top_col0 is None
            or data.ref_bot_col0 is None
        ):
            reject_invalid_state(
                "ref row0/col0 origins are required when ref carriers are set"
            )
        if int(data.ref_top_native_height) < int(data.ref_top_row0) + shape[0]:
            reject_invalid_state(
                "ref_top_native_height must span the burst-absolute window; "
                "crop-local native_height is invalid"
            )
        if int(data.ref_bot_native_height) < int(data.ref_bot_row0) + shape[0]:
            reject_invalid_state(
                "ref_bot_native_height must span the burst-absolute window; "
                "crop-local native_height is invalid"
            )
    return shape


def _secondary_response(
    data: OverlapESDInput,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return full-carrier ``R_B`` on the native (pre-multilook) crop."""
    top_row, top_col = _source_coordinates(
        shape, data.top_row0, data.top_col0, data.top_offsets
    )
    bot_row, bot_col = _source_coordinates(
        shape, data.bot_row0, data.bot_col0, data.bot_offsets
    )
    r_top = _carrier_response_rad_px(
        data.sec_top_carrier,
        top_row,
        top_col,
        native_height=data.top_native_height,
    )
    r_bot = _carrier_response_rad_px(
        data.sec_bot_carrier,
        bot_row,
        bot_col,
        native_height=data.bot_native_height,
    )
    return r_top - r_bot


def _reference_response(
    data: OverlapESDInput,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Return full-carrier ``R_A`` on the native crop, or ``None`` if omitted."""
    if data.ref_top_carrier is None or data.ref_bot_carrier is None:
        return None
    zeros = (
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
    )
    top_row, top_col = _source_coordinates(
        shape,
        int(data.ref_top_row0),
        int(data.ref_top_col0),
        zeros,
    )
    bot_row, bot_col = _source_coordinates(
        shape,
        int(data.ref_bot_row0),
        int(data.ref_bot_col0),
        zeros,
    )
    r_top = _carrier_response_rad_px(
        data.ref_top_carrier,
        top_row,
        top_col,
        native_height=int(data.ref_top_native_height),
    )
    r_bot = _carrier_response_rad_px(
        data.ref_bot_carrier,
        bot_row,
        bot_col,
        native_height=int(data.ref_bot_native_height),
    )
    return r_top - r_bot


def estimate_overlap_esd(
    data: OverlapESDInput,
    *,
    coherence_threshold: float = 0.85,
    multilook: tuple[int, int] = (5, 15),
) -> OverlapESDRecord | None:
    """Estimate one overlap's residual azimuth shift from original-domain DD.

    Builds ``DD = IFG_top * conj(IFG_bot)`` after multilook, gates cells
    by the per-IFG coherence, and converts DD phase with the secondary
    full-carrier response evaluated at ``src = crop - offsets + (row0, col0)``
    and ``centre_row = native_height // 2``.

    Parameters
    ----------
    data : OverlapESDInput
        Original-domain overlap crops, secondary carriers, burst-absolute
        coordinates, and geometry-only offset crops.
    coherence_threshold : float, optional
        Minimum per-cell coherence of **both** overlap IFGs. Default 0.85.
    multilook : tuple of int, optional
        ``(azimuth, range)`` looks. Default ``(5, 15)`` (ISCE2-aligned).

    Returns
    -------
    OverlapESDRecord or None
        Per-overlap residual when at least one cell passes the gate;
        ``None`` when no sample qualifies (never a stuffed 0-shift).

    Raises
    ------
    InvalidProcessingStateError
        If crops do not match, are not complex, or a reference carrier is
        set without its burst-absolute ``native_height`` triple.

    """
    if not 0.0 < coherence_threshold <= 1.0:
        reject_invalid_state("coherence_threshold must lie in (0, 1]")
    shape = _validate_overlap_input(data)
    ifg_top, coh_top = _multilooked_ifg_coherence(data.ref_top, data.sec_top, multilook)
    ifg_bot, coh_bot = _multilooked_ifg_coherence(data.ref_bot, data.sec_bot, multilook)
    dd = ifg_top * np.conjugate(ifg_bot)
    coherence = np.minimum(coh_top, coh_bot)

    r_b = _boxcar_mean(_secondary_response(data, shape), multilook)
    r_a_full = _reference_response(data, shape)
    r_a = None if r_a_full is None else _boxcar_mean(r_a_full, multilook)

    finite = np.isfinite(coherence) & np.isfinite(r_b) & np.isfinite(dd)
    valid = (
        finite
        & (coherence >= coherence_threshold)
        & (np.abs(r_b) >= _MIN_ABS_RESPONSE_RAD_PX)
        & (np.abs(dd) > 0.0)
    )
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        logger.warning(
            "Overlap ESD has no samples above coherence threshold %.2f; "
            "omitting azimuth observation",
            coherence_threshold,
        )
        return None

    delta = np.angle(dd) / r_b
    weights = coherence
    azimuth_shift_px = float(np.average(delta[valid], weights=weights[valid]))
    coherence_mean = float(np.mean(weights[valid]))
    response_rb = float(np.median(np.abs(r_b[valid])))
    if r_a is None:
        consistency = float("nan")
    else:
        denom = np.maximum(np.abs(r_b[valid]), _MIN_ABS_RESPONSE_RAD_PX)
        consistency = float(np.median(np.abs(r_a[valid] - r_b[valid]) / denom))

    logger.info(
        "Overlap ESD az_shift=%.4f px coherence=%.3f n_valid=%s R_B=%.3f rad/px",
        azimuth_shift_px,
        coherence_mean,
        n_valid,
        response_rb,
    )
    return OverlapESDRecord(
        top_index=data.top_index,
        bottom_index=data.bottom_index,
        azimuth_shift_px=azimuth_shift_px,
        coherence=coherence_mean,
        n_valid=n_valid,
        response_rb_rad_px=response_rb,
        response_consistency=consistency,
    )


def aggregate_overlap_esd(
    records: Sequence[OverlapESDRecord],
    *,
    sigma_floor_px: float = 0.01,
) -> OverlapESDAggregate | None:
    """Pool per-overlap residuals into one pair-level azimuth observation.

    Parameters
    ----------
    records : sequence of OverlapESDRecord
        Qualifying overlap measurements. An empty sequence omits the
        azimuth observation.
    sigma_floor_px : float, optional
        Minimum reported sigma so a single-overlap arc stays invert-legal.
        Default 0.01 px.

    Returns
    -------
    OverlapESDAggregate or None
        Weighted-mean residual when ``records`` is non-empty; ``None`` if
        there are no records (never a stuffed 0-shift).

    """
    if sigma_floor_px < 0.0:
        reject_invalid_state("sigma_floor_px must be >= 0")
    if len(records) == 0:
        logger.warning(
            "Overlap ESD aggregate received no records; omitting azimuth observation"
        )
        return None

    shifts = np.array([rec.azimuth_shift_px for rec in records], dtype=np.float64)
    weights = np.array(
        [rec.coherence * float(rec.n_valid) for rec in records],
        dtype=np.float64,
    )
    if not np.any(weights > 0.0):
        logger.warning(
            "Overlap ESD aggregate weights are all zero; omitting azimuth observation"
        )
        return None

    azimuth_shift_px = float(np.average(shifts, weights=weights))
    scatter = float(np.average((shifts - azimuth_shift_px) ** 2, weights=weights))
    sigma_px = max(float(np.sqrt(max(scatter, 0.0))), float(sigma_floor_px))
    n_valid = int(sum(rec.n_valid for rec in records))
    consistencies = np.array(
        [rec.response_consistency for rec in records],
        dtype=np.float64,
    )
    diagnostics: dict[str, object] = {
        "mean_coherence": float(
            np.average(
                np.array([rec.coherence for rec in records], dtype=np.float64),
                weights=weights,
            )
        ),
        "mean_response_consistency": float(np.nanmean(consistencies)),
        "sigma_floor_px": float(sigma_floor_px),
    }
    logger.info(
        "Overlap ESD aggregate az_shift=%.4f px sigma=%.4f n_overlaps=%s n_valid=%s",
        azimuth_shift_px,
        sigma_px,
        len(records),
        n_valid,
    )
    return OverlapESDAggregate(
        azimuth_shift_px=azimuth_shift_px,
        sigma_px=sigma_px,
        n_overlaps=len(records),
        n_valid=n_valid,
        method="overlap",
        diagnostics=diagnostics,
    )
