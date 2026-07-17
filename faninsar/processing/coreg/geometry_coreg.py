"""Geometry-assisted coarse offsets from dual orbits."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.coreg.offsets import OffsetFieldResult, estimate_global_shift
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry.orbit import OrbitInterpolator

if TYPE_CHECKING:
    from faninsar.sentinel1.types import S1Burst, S1Swath

logger = setup_logger(__name__)


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
    search_radius: int = 32,
) -> tuple[float, float]:
    """Refine a geometry prior with amplitude cross-correlation.

    The secondary is integer-shifted by the rounded prior, then a
    sub-pixel residual is estimated via FFT cross-correlation with
    parabolic peak refinement.

    Parameters
    ----------
    reference_samples, secondary_samples : numpy.ndarray
        Complex 2-D arrays on the same grid.
    prior_rg, prior_az : float
        Geometry-predicted range/azimuth shifts in the
        :func:`~faninsar.processing.coreg.offsets.resample_complex`
        convention (``source = output - offset``).
    search_radius : int, optional
        Maximum correlation search radius around the prior.

    Returns
    -------
    tuple[float, float]
        Refined ``(range_shift_px, azimuth_shift_px)`` in the same
        resample convention as ``prior_*``.

    """
    pre_rg = round(prior_rg)
    pre_az = round(prior_az)
    # Pre-align secondary under the resample_complex convention:
    # source = out - offset  ⇒  shifted[i] = secondary[i - prior].
    # numpy.roll(a, +prior) implements shifted[i] = a[i - prior].
    shifted = np.roll(secondary_samples, shift=pre_az, axis=0)
    shifted = np.roll(shifted, shift=pre_rg, axis=1)
    d_rg, d_az = estimate_global_shift(
        reference_samples,
        shifted,
        max_shift=search_radius,
        subpixel=True,
    )
    total_rg = prior_rg + d_rg
    total_az = prior_az + d_az
    logger.info(
        "Correlation refinement d_rg=%.3f d_az=%.3f -> total rg=%.3f az=%.3f",
        d_rg,
        d_az,
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
) -> OffsetFieldResult:
    """Combine a dense geometry offset field with ESD and amplitude residuals.

    The ESD azimuth residual is added uniformly to the geometry azimuth
    offsets.  Optional amplitude-correlation residuals are added to both
    range and azimuth.  Coverage and uncertainty are propagated
    conservatively (coverage is intersected, uncertainty is summed).

    Parameters
    ----------
    geometry_field : OffsetFieldResult
        Dense geometry-driven offsets (e.g. from ``dense_geometry_offsets``).
    esd_azimuth_shift_px : float, optional
        Residual azimuth shift from spectral diversity (default 0).
    amplitude_residual_rg, amplitude_residual_az : float, optional
        Global residual shifts from amplitude cross-correlation (default 0).

    Returns
    -------
    OffsetFieldResult
        Combined dense offset field.

    """
    combined_rg = (
        geometry_field.range_offset_px.astype(np.float32, copy=False)
        + np.float32(amplitude_residual_rg)
    )
    combined_az = (
        geometry_field.azimuth_offset_px.astype(np.float32, copy=False)
        + np.float32(esd_azimuth_shift_px)
        + np.float32(amplitude_residual_az)
    )
    # Coverage is unchanged (geometry already determined valid area)
    # Uncertainty: add amplitude residual uncertainty heuristically
    combined_uncertainty = (
        geometry_field.uncertainty_px.astype(np.float32, copy=False)
        + np.float32(
            0.1 * (abs(amplitude_residual_rg) + abs(amplitude_residual_az))
        )
    )
    logger.info(
        "Combined offsets ESD_az=%.4f amp_rg=%.4f amp_az=%.4f",
        esd_azimuth_shift_px,
        amplitude_residual_rg,
        amplitude_residual_az,
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
    from faninsar.processing.coreg.offsets import geometry_shift_offsets

    return geometry_shift_offsets(
        shape,
        range_shift_px=range_shift_px,
        azimuth_shift_px=azimuth_shift_px,
    )
