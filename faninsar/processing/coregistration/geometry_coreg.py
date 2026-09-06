"""Geometry-assisted coarse offsets from dual orbits."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.coregistration.offsets import (
    OffsetFieldResult,
    _torch_ampcor_admission_key,
    _validate_ampcor_inputs,
    _validate_torch_ampcor_runtime,
    estimate_patch_amplitude_shift,
    resolve_ampcor_policy,
)
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry.orbit import OrbitInterpolator

if TYPE_CHECKING:
    from faninsar.missions.s1.types import S1Burst, S1Swath

logger = setup_logger(__name__)


def _validate_ampcor_accelerator(device: str) -> None:
    """Validate explicit accelerator availability before input copies."""
    if device == "mps":
        message = "MPS is not a qualified Ampcor device; use CPU or CUDA"
        logger.error(message)
        reject_invalid_state(message)
    if device.startswith("cuda"):
        try:
            import torch
        except ImportError as error:
            message = "explicit CUDA Ampcor requires torch"
            logger.exception(message)
            raise ImportError(message) from error
        _validate_torch_ampcor_runtime(torch.device(device), torch)
        _torch_ampcor_admission_key(torch.device(device), torch)


def geometry_coarse_shift(
    reference: S1Swath,
    secondary: S1Swath,
    *,
    reference_burst: S1Burst,
    secondary_burst: S1Burst,
) -> tuple[float, float]:
    """Estimate global range/azimuth shift from orbit geometry."""
    ref_orbit = OrbitInterpolator.from_orbit(reference.orbit)
    sec_orbit = OrbitInterpolator.from_orbit(secondary.orbit)
    ref_time = reference_burst.azimuth_time + timedelta(
        seconds=0.5 * reference_burst.lines * reference.azimuth_time_interval_s
    )
    sec_time = secondary_burst.azimuth_time + timedelta(
        seconds=0.5 * secondary_burst.lines * secondary.azimuth_time_interval_s
    )
    try:
        ref_state = ref_orbit.evaluate(ref_time)
        sec_state = sec_orbit.evaluate(sec_time)
    except Exception as exc:
        message = f"orbit evaluation failed for geometry coarse shift: {exc}"
        logger.exception(message)
        raise ValueError(message) from exc

    baseline = np.asarray(sec_state.position_m, dtype=np.float64) - np.asarray(
        ref_state.position_m,
        dtype=np.float64,
    )
    vel = np.asarray(ref_state.velocity_m_s, dtype=np.float64)
    speed = float(np.linalg.norm(vel))
    if speed <= 0:
        reject_invalid_state("reference orbit velocity is zero")
    along_track = float(np.dot(baseline, vel / speed))
    cross_track = float(np.linalg.norm(baseline - along_track * (vel / speed)))

    az_shift = along_track / max(reference.azimuth_pixel_spacing_m, 1e-6)
    rg_shift = cross_track / max(reference.range_pixel_spacing_m, 1e-6)
    logger.info(
        "Geometry coarse shift estimate rg=%.3f px az=%.3f px (|B|=%.1f m)",
        rg_shift,
        az_shift,
        float(np.linalg.norm(baseline)),
    )
    return rg_shift, az_shift


def refine_shift_with_correlation(
    reference_samples: np.ndarray,
    secondary_samples: np.ndarray,
    *,
    prior_rg: float,
    prior_az: float,
    search_radius: int = 16,
    executor: str = "numpy",
    device: str = "auto",
) -> tuple[float, float]:
    """Refine a geometry prior with multi-window magnitude Ampcor.

    The secondary is integer-shifted by the rounded prior, then a robust
    residual is estimated with ISCE2-style patch Ampcor (64x32 windows,
    SNR cull, median over ~40x20 locations). Full-image FFT correlation is
    intentionally avoided: on TOPS bursts it locks onto the amplitude
    envelope and injects a false azimuth residual of ~0.4 px that destroys
    interferometric coherence.

    Parameters
    ----------
    reference_samples, secondary_samples : numpy.ndarray
        Complex 2-D arrays on the same grid.
    prior_rg, prior_az : float
        Geometry-predicted range/azimuth shifts in the
        :func:`~faninsar.processing.coregistration.offsets.resample_complex`
        convention (``source = output - offset``).
    search_radius : int, optional
        Correlation search half-width around the prior (default 16, matching
        ISCE2 topsApp Ampcor).
    executor : {"numpy", "torch"}, optional
        Requested Ampcor implementation. Automatic operation resolves to
        portable NumPy; explicit ``executor="torch", device="cpu"`` uses
        bounded Torch CPU batches and explicit CUDA uses Torch.
    device : {"auto", "cpu", "cuda"}, optional
        Requested execution device. Explicit CUDA is fail-closed when it is
        unavailable; automatic operation remains portable CPU NumPy. MPS is
        outside the qualified Ampcor contract and is rejected.

    Returns
    -------
    tuple[float, float]
        Refined ``(range_shift_px, azimuth_shift_px)`` in the same
        resample convention as ``prior_*``.

    """
    ampcor_executor, ampcor_device = resolve_ampcor_policy(executor, device)
    if ampcor_executor == "torch":
        _validate_ampcor_accelerator(ampcor_device)
    if (
        not isinstance(search_radius, (int, np.integer))
        or isinstance(search_radius, (bool, np.bool_))
        or int(search_radius) < 1
    ):
        reject_invalid_state("search_radius must be a positive integer")
    reference_samples, secondary_samples = _validate_ampcor_inputs(
        reference_samples,
        secondary_samples,
        torch_contract=ampcor_executor == "torch",
    )
    if not np.isfinite(prior_rg) or not np.isfinite(prior_az):
        logger.warning(
            "Correlation refinement skipped: non-finite prior offsets "
            "prior_rg=%s prior_az=%s",
            prior_rg,
            prior_az,
        )
        return float("nan"), float("nan")
    pre_rg = round(float(prior_rg))
    pre_az = round(float(prior_az))
    # Pre-align secondary under the resample_complex convention. Ampcor applies
    # the equivalent cyclic source shift per bounded tile, avoiding full-image
    # ``numpy.roll`` allocations on production bursts.
    patch = estimate_patch_amplitude_shift(
        reference_samples,
        secondary_samples,
        search_az=search_radius,
        search_rg=search_radius,
        subpixel=True,
        executor=ampcor_executor,
        device=ampcor_device,
        secondary_shift=(pre_az, pre_rg),
    )
    d_rg = float(patch.range_shift_px)
    d_az = float(patch.azimuth_shift_px)
    total_rg = prior_rg + d_rg
    total_az = prior_az + d_az
    logger.info(
        "Correlation refinement (patch Ampcor) d_rg=%.4f d_az=%.4f "
        "n_valid=%d snr_med=%.2f -> total rg=%.3f az=%.3f",
        d_rg,
        d_az,
        patch.n_valid,
        patch.snr_median,
        total_rg,
        total_az,
    )
    return total_rg, total_az


def combine_offset_fields(
    geometry_field: OffsetFieldResult,
    *,
    esd_azimuth_shift_px: float = 0.0,
    amplitude_residual_rg: float = 0.0,
    amplitude_residual_az: float = 0.0,
    misreg_az_px: float = 0.0,
    misreg_rg_px: float = 0.0,
) -> OffsetFieldResult:
    """Combine a dense geometry offset field with ESD and amplitude residuals.

    The ESD azimuth residual is added uniformly to the geometry azimuth
    offsets.  Optional amplitude-correlation residuals are added to both
    range and azimuth.  Stack network misreg constants (``misreg_*``) are
    added the same way (PROPOSAL-0017). Coverage and uncertainty are
    propagated conservatively.

    Parameters
    ----------
    geometry_field : OffsetFieldResult
        Dense geometry-driven offsets (e.g. from ``dense_geometry_offsets``).
    esd_azimuth_shift_px : float, optional
        Residual azimuth shift from spectral diversity (default 0).
    amplitude_residual_rg, amplitude_residual_az : float, optional
        Global residual shifts from amplitude cross-correlation (default 0).
    misreg_az_px, misreg_rg_px : float, optional
        Per-date (or relative) network misregistration constants (default 0).

    Returns
    -------
    OffsetFieldResult
        Combined dense offset field.

    """
    combined_rg = (
        geometry_field.range_offset_px.astype(np.float32, copy=False)
        + np.float32(amplitude_residual_rg)
        + np.float32(misreg_rg_px)
    )
    combined_az = (
        geometry_field.azimuth_offset_px.astype(np.float32, copy=False)
        + np.float32(esd_azimuth_shift_px)
        + np.float32(amplitude_residual_az)
        + np.float32(misreg_az_px)
    )
    # Coverage is unchanged (geometry already determined valid area)
    # Uncertainty: add amplitude residual uncertainty heuristically
    combined_uncertainty = geometry_field.uncertainty_px.astype(
        np.float32, copy=False
    ) + np.float32(
        0.1
        * (
            abs(amplitude_residual_rg)
            + abs(amplitude_residual_az)
            + abs(misreg_rg_px)
            + abs(misreg_az_px)
        )
    )
    logger.info(
        "Combined offsets ESD_az=%.4f amp_rg=%.4f amp_az=%.4f "
        "misreg_az=%.4f misreg_rg=%.4f",
        esd_azimuth_shift_px,
        amplitude_residual_rg,
        amplitude_residual_az,
        misreg_az_px,
        misreg_rg_px,
    )
    return OffsetFieldResult(
        range_offset_px=np.asarray(combined_rg, dtype=np.float32),
        azimuth_offset_px=np.asarray(combined_az, dtype=np.float32),
        coverage=geometry_field.coverage,
        uncertainty_px=np.asarray(combined_uncertainty, dtype=np.float32),
    )


def build_offset_field(
    shape: tuple[int, int],
    *,
    range_shift_px: float,
    azimuth_shift_px: float,
) -> OffsetFieldResult:
    """Materialize a dense constant offset field from a global shift."""
    from faninsar.processing.coregistration.offsets import geometry_shift_offsets

    return geometry_shift_offsets(
        shape,
        range_shift_px=range_shift_px,
        azimuth_shift_px=azimuth_shift_px,
    )
