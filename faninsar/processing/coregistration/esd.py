"""Enhanced Spectral Diversity (ESD) azimuth shift estimation.

Implements the dual-look spectral interferometry method for estimating
sub-pixel residual azimuth shifts between coregistered (or coarsely
shifted) TOPS complex pairs.

References
----------
- Prats-Iraola, P., et al. (2012). "SAR Signal Processing for
  Sentinel-1 TOPS Imaging Mode." IEEE TGARS.
- Standard TOPS coregistration literature: ESD uses the phase
  difference between upper/lower azimuth spectral looks to infer
  the residual timing/geometry shift.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.fft import fft, ifft

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class ESDResult:
    """Azimuth shift and quality from spectral-diversity estimation."""

    azimuth_shift_px: float
    """Residual azimuth shift in pixels (positive = secondary lags reference)."""
    coherence: float
    """Mean coherence of the differential interferogram (0..1)."""
    phase_rad: float
    """Estimated differential phase in radians."""


def _azimuth_bandpass_filter(
    n_az: int,
    look: Literal["lower", "upper"],
    *,
    bandwidth_fraction: float = 0.45,
    taper_alpha: float = 0.1,
) -> np.ndarray:
    """Build a 1-D azimuth bandpass filter for spectral look splitting.

    Parameters
    ----------
    n_az : int
        Number of azimuth samples.
    look : {"lower", "upper"}
        Which spectral half to retain.
    bandwidth_fraction : float, optional
        Fraction of the full Nyquist band to keep (default 0.45 leaves a
        small guard band to avoid wrap-around leakage).
    taper_alpha : float, optional
        Tukey taper fraction applied to the band edges (0 = rectangular,
        1 = Hann).  A mild taper reduces Gibbs sidelobes.

    Returns
    -------
    numpy.ndarray
        Real-valued 1-D filter of length ``n_az`` in FFT order
        (0..+freq, -freq..0).

    """
    if not 0.0 < bandwidth_fraction <= 0.5:
        reject_invalid_state("bandwidth_fraction must be in (0, 0.5]")
    freqs = np.fft.fftfreq(n_az, d=1.0)
    half_bw = bandwidth_fraction * 0.5
    center = -half_bw if look == "lower" else +half_bw
    # Distance from the band centre, wrapped to [-0.5, 0.5]
    dist = np.abs(freqs - center)
    dist = np.minimum(dist, 1.0 - dist)
    # Rectangular passband of width 2*half_bw
    inside = dist <= half_bw
    filter_arr = np.zeros(n_az, dtype=np.float64)
    filter_arr[inside] = 1.0
    # Tukey taper on the edges
    if taper_alpha > 0.0:
        edge = half_bw * taper_alpha
        transition = (dist - (half_bw - edge)) / (2.0 * edge)
        transition = np.clip(transition, 0.0, 1.0)
        taper = 0.5 * (1.0 + np.cos(np.pi * transition))
        filter_arr = np.where(inside, taper, 0.0)
    return filter_arr


def estimate_azimuth_shift_esd(
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    bandwidth_fraction: float = 0.45,
    taper_alpha: float = 0.1,
    min_coherence: float = 0.05,
) -> ESDResult:
    r"""Estimate residual azimuth shift via dual-look spectral diversity.

    The algorithm splits the azimuth spectrum into lower and upper looks,
    forms interferograms from each look, and measures the phase difference
    between them.  Because a residual azimuth shift introduces a frequency-
    dependent phase ramp, the differential phase is proportional to the
    shift:

    .. math::
        \\Delta\\phi = 2\\pi \\, \\Delta f_{az} \\, \\Delta t
                   = 2\\pi \\, \\Delta f_{az} \\,
                     \frac{\\Delta_{az}}{f_s}

    where :math:`\\Delta f_{az}` is the centre-frequency separation of the
    two looks and :math:`f_s` is the azimuth sampling frequency (1 px⁻¹).

    Parameters
    ----------
    reference, secondary : numpy.ndarray
        Complex 2-D arrays of shape ``(azimuth, range)``.  They must
        already be coarsely coregistered (e.g. via geometry shifts); ESD
        estimates the *residual* sub-pixel azimuth error.
    bandwidth_fraction : float, optional
        Fraction of the full azimuth bandwidth assigned to each look.
        Default ``0.45`` leaves a small guard band.
    taper_alpha : float, optional
        Tukey taper fraction on the look edges (``0.1`` = mild taper).
    min_coherence : float, optional
        Floor for the coherence magnitude used as a quality weight.
        Prevents division-by-near-zero when the scene is decorrelated.

    Returns
    -------
    ESDResult
        Estimated azimuth shift in pixels, mean coherence, and differential
        phase.

    Raises
    ------
    ValueError
        If inputs are not matching 2-D complex arrays.

    """
    if reference.shape != secondary.shape or reference.ndim != 2:
        reject_invalid_state("ESD requires matching 2-D arrays")
    if not (np.iscomplexobj(reference) and np.iscomplexobj(secondary)):
        reject_invalid_state("ESD requires complex-valued SLC arrays")

    n_az, _ = reference.shape

    # 1-D azimuth filters for lower / upper looks
    filt_lower = _azimuth_bandpass_filter(
        n_az, "lower", bandwidth_fraction=bandwidth_fraction, taper_alpha=taper_alpha
    )
    filt_upper = _azimuth_bandpass_filter(
        n_az, "upper", bandwidth_fraction=bandwidth_fraction, taper_alpha=taper_alpha
    )

    # Azimuth FFT (axis 0) for each array
    f_ref = fft(reference, axis=0)
    f_sec = fft(secondary, axis=0)

    # Form looks by bandpass filtering in azimuth frequency
    look_ref_lower = ifft(f_ref * filt_lower[:, None], axis=0)
    look_ref_upper = ifft(f_ref * filt_upper[:, None], axis=0)
    look_sec_lower = ifft(f_sec * filt_lower[:, None], axis=0)
    look_sec_upper = ifft(f_sec * filt_upper[:, None], axis=0)

    # Interferograms per look
    ifg_lower = look_ref_lower * np.conjugate(look_sec_lower)
    ifg_upper = look_ref_upper * np.conjugate(look_sec_upper)

    # Differential interferogram: phase difference between looks
    diff_ifg = ifg_lower * np.conjugate(ifg_upper)

    # Use unit phasors so coherence is in [0, 1]
    amp = np.abs(diff_ifg)
    valid = amp > 0
    if not np.any(valid):
        logger.warning("ESD has no valid differential samples; returning zero shift")
        return ESDResult(azimuth_shift_px=0.0, coherence=0.0, phase_rad=0.0)
    unit = np.zeros_like(diff_ifg)
    unit[valid] = diff_ifg[valid] / amp[valid]
    weight = amp
    weight_sum = float(np.sum(weight[valid]))
    if weight_sum <= 0.0:
        logger.warning("ESD weight sum is zero; returning zero shift")
        return ESDResult(azimuth_shift_px=0.0, coherence=0.0, phase_rad=0.0)

    # Circular mean of differential phase, weighted by amplitude
    weighted = np.sum(unit[valid] * weight[valid])
    phase_mean = np.angle(weighted)
    coherence = float(np.abs(weighted) / weight_sum)
    coherence = float(np.clip(coherence, min_coherence, 1.0))

    # Frequency separation between look centres
    # Lower look centred at -bandwidth_fraction/2, upper at +bandwidth_fraction/2
    # In cycles per pixel (fftfreq units), separation = bandwidth_fraction
    delta_f_cycles_per_px = bandwidth_fraction

    # Convert phase to shift:
    #   Δφ = 2π * Δf * Δaz   =>   Δaz = Δφ / (2π * Δf)
    # Here Δf is in cycles per pixel, so Δaz is directly in pixels.
    denom = 2.0 * np.pi * delta_f_cycles_per_px
    if abs(denom) < 1e-12:
        reject_invalid_state("ESD frequency separation is near zero")

    az_shift = -float(phase_mean) / denom

    logger.info(
        "ESD estimate az_shift=%.4f px coherence=%.3f phase=%.3f rad",
        az_shift,
        coherence,
        float(phase_mean),
    )

    return ESDResult(
        azimuth_shift_px=az_shift,
        coherence=coherence,
        phase_rad=float(phase_mean),
    )
