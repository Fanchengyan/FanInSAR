"""Sentinel-1 TOPS deramp and reramp phase operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)

SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True, slots=True)
class TOPSCarrierModel:
    """Range-dependent Doppler centroid and FM-rate model for one burst."""

    radar_frequency_hz: float
    slant_range_time0_s: float
    range_sampling_rate_hz: float
    azimuth_time_interval_s: float
    doppler_centroid_hz: tuple[float, ...]
    doppler_t0_s: float
    fm_rate_hz_s: tuple[float, ...]
    fm_t0_s: float
    burst_sensing_time_s: float
    burst_start_slant_range_time_s: float
    azimuth_steering_rate_hz_s: float

    def __post_init__(self) -> None:
        """Validate carrier model parameters."""
        if self.radar_frequency_hz <= 0 or self.range_sampling_rate_hz <= 0:
            reject_invalid_state("radar frequency and range sampling rate must be > 0")
        if self.azimuth_time_interval_s <= 0:
            reject_invalid_state("azimuth time interval must be > 0")
        if not self.doppler_centroid_hz or not self.fm_rate_hz_s:
            reject_invalid_state("Doppler and FM-rate polynomials are required")
        if self.azimuth_steering_rate_hz_s == 0.0:
            reject_invalid_state("TOPS azimuth steering rate must be non-zero")


def _poly_eval(coefficients: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial coefficients[0] + c1*x + c2*x^2 + ..."""
    result = np.zeros_like(x, dtype=np.float64)
    power = np.ones_like(x, dtype=np.float64)
    for coeff in coefficients:
        result = result + float(coeff) * power
        power = power * x
    return result


def range_time_axis(
    model: TOPSCarrierModel,
    n_samples: int,
) -> np.ndarray:
    """Return one-way slant-range time for each range sample index."""
    sample = np.arange(n_samples, dtype=np.float64)
    return model.slant_range_time0_s + sample / model.range_sampling_rate_hz


def azimuth_time_axis(
    model: TOPSCarrierModel,
    n_lines: int,
) -> np.ndarray:
    """Return azimuth time relative to burst centre for each line."""
    line = np.arange(n_lines, dtype=np.float64)
    centre = 0.5 * (n_lines - 1)
    return (line - centre) * model.azimuth_time_interval_s


def tops_carrier_phase(
    model: TOPSCarrierModel,
    n_lines: int,
    n_samples: int,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Compute the TOPS carrier phase (radians) on a burst window.

    The phase follows the Sentinel-1 TOPS steering model:

    ``phi = π K_t(tau) [eta - eta_ref(tau)]² + 2π f_dc(tau) eta``

    where ``K_t = K_s / (1 - K_s / K_a)`` and ``eta_ref`` accounts for
    range-dependent Doppler centroid.

    Parameters
    ----------
    model : TOPSCarrierModel
        Burst carrier parameters.
    n_lines, n_samples : int
        Burst window shape.
    dtype : numpy.dtype, optional
        Output phase dtype. ``float32`` halves peak memory for full bursts
        while remaining adequate for the subsequent complex multiply.
        Default ``float64``.

    Returns
    -------
    numpy.ndarray
        Real phase array of shape ``(n_lines, n_samples)``.

    """
    if n_lines <= 0 or n_samples <= 0:
        reject_invalid_state("burst window dimensions must be positive")
    cols = np.arange(n_samples, dtype=np.float64)[None, :]
    rows = np.arange(n_lines, dtype=np.float64)[:, None]
    phase = carrier_phase_at_points(
        model,
        np.broadcast_to(rows, (n_lines, n_samples)),
        np.broadcast_to(cols, (n_lines, n_samples)),
        centre_row=float(n_lines // 2),
        dtype=dtype,
    )
    return np.asarray(phase, dtype=dtype)


def _apply_carrier_phase(
    samples: np.ndarray,
    phase: np.ndarray,
    *,
    sign: float,
) -> np.ndarray:
    """Multiply complex samples by ``exp(sign * 1j * phase)`` without complex128.

    ``samples * np.exp(±1j * phase)`` materialises a complex128 carrier on a
    full burst (~0.5 GB for S1 IW). Cos/sin in float32 and an in-place-style
    real/imag product keep the temporary footprint close to one float32 phase
    plane plus the complex64 output.
    """
    # Evaluate cos/sin in float64 for large carrier phases (S1 FM * t² can
    # be many radians), then multiply in float32 to avoid complex128 temps.
    phase64 = np.asarray(phase, dtype=np.float64)
    cos_p = np.cos(phase64).astype(np.float32, copy=False)
    sin_p = np.sin(phase64).astype(np.float32, copy=False)
    # exp(sign * 1j * phi) = cos(phi) + sign * 1j * sin(phi)
    # (re + 1j*im) * (c + sign*1j*s) = (re*c - sign*im*s) + 1j*(im*c + sign*re*s)
    re = samples.real.astype(np.float32, copy=False)
    im = samples.imag.astype(np.float32, copy=False)
    s = float(sign)
    out_re = re * cos_p - s * im * sin_p
    out_im = im * cos_p + s * re * sin_p
    return (out_re + 1j * out_im).astype(samples.dtype, copy=False)


def _carrier_phase_rows(
    model: TOPSCarrierModel,
    row_indices: np.ndarray,
    n_samples: int,
    *,
    centre_row: float,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Carrier phase for a subset of absolute burst-local row indices."""
    rows = np.asarray(row_indices, dtype=np.float64)[:, None]
    cols = np.arange(n_samples, dtype=np.float64)[None, :]
    return carrier_phase_at_points(
        model,
        np.broadcast_to(rows, (rows.shape[0], n_samples)),
        np.broadcast_to(cols, (rows.shape[0], n_samples)),
        centre_row=centre_row,
        dtype=dtype,
    )


def _apply_carrier_tiled(
    samples: np.ndarray,
    model: TOPSCarrierModel,
    *,
    sign: float,
    phase_dtype: np.dtype | type,
    row_chunk: int | None,
) -> np.ndarray:
    """Shared tiled carrier multiply for deramp (sign=-1) and reramp (sign=+1)."""
    n_lines, n_samples = samples.shape
    if row_chunk is None or row_chunk <= 0 or n_lines <= row_chunk:
        phase = tops_carrier_phase(model, n_lines, n_samples, dtype=phase_dtype)
        return _apply_carrier_phase(samples, phase, sign=sign)

    out = np.empty_like(samples)
    for row0 in range(0, n_lines, row_chunk):
        row1 = min(row0 + row_chunk, n_lines)
        phase = _carrier_phase_rows(
            model,
            np.arange(row0, row1, dtype=np.float64),
            n_samples,
            centre_row=float(n_lines // 2),
            dtype=phase_dtype,
        )
        out[row0:row1] = _apply_carrier_phase(samples[row0:row1], phase, sign=sign)
    return out


def deramp(
    samples: np.ndarray,
    model: TOPSCarrierModel,
    *,
    phase_dtype: np.dtype | type = np.float64,
    row_chunk: int | None = 256,
) -> np.ndarray:
    """Remove the TOPS carrier phase from complex samples.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex burst window.
    model : TOPSCarrierModel
        Carrier model matching the window geometry.
    phase_dtype : numpy.dtype, optional
        Working dtype for the carrier phase plane. Default ``float64`` for
        polynomial accuracy; use ``float32`` only when peak memory is
        tighter than the accuracy budget.
    row_chunk : int or None, optional
        If set, apply the carrier in azimuth tiles of this many rows so the
        phase/cos/sin temporaries never cover the full burst. ``None``
        processes the whole array at once. Default 256.

    Returns
    -------
    numpy.ndarray
        Deramped complex array (same dtype/shape).

    """
    if samples.ndim != 2 or not np.iscomplexobj(samples):
        reject_invalid_state("deramp requires a 2-D complex array")
    return _apply_carrier_tiled(
        samples,
        model,
        sign=-1.0,
        phase_dtype=phase_dtype,
        row_chunk=row_chunk,
    )


def reramp(
    samples: np.ndarray,
    model: TOPSCarrierModel,
    *,
    phase_dtype: np.dtype | type = np.float64,
    row_chunk: int | None = 256,
) -> np.ndarray:
    """Restore the TOPS carrier phase to complex samples.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex deramped burst window.
    model : TOPSCarrierModel
        Carrier model matching the window geometry.
    phase_dtype : numpy.dtype, optional
        Working dtype for the carrier phase plane. Default ``float64``.
    row_chunk : int or None, optional
        Azimuth tile height for the carrier multiply. Default 256.

    Returns
    -------
    numpy.ndarray
        Reramped complex array (same dtype/shape).

    """
    if samples.ndim != 2 or not np.iscomplexobj(samples):
        reject_invalid_state("reramp requires a 2-D complex array")
    return _apply_carrier_tiled(
        samples,
        model,
        sign=1.0,
        phase_dtype=phase_dtype,
        row_chunk=row_chunk,
    )


def restore_original_domain_secondary(
    sec_resamp_deramped: np.ndarray,
    secondary_carrier: TOPSCarrierModel,
    range_offset_px: np.ndarray,
    azimuth_offset_px: np.ndarray,
) -> np.ndarray:
    """Restore original-domain secondary after deramped-domain resampling.

    Coregistration resamples the secondary in the TOPS-deramped domain.  The
    interferogram must be formed in the original focused-SLC phase domain, so
    the secondary carrier phase is reapplied at the **source** coordinates
    (``output_index - offset``), matching the samples that were interpolated.

    Reramping both scenes with the *reference* carrier is incorrect: it leaves
    a residual secondary-carrier phase and destroys geometric fringes.

    Parameters
    ----------
    sec_resamp_deramped : numpy.ndarray
        Secondary samples after deramp and resample onto the reference grid.
    secondary_carrier : TOPSCarrierModel
        Carrier model of the secondary burst (native grid geometry).
    range_offset_px, azimuth_offset_px : numpy.ndarray
        Offset fields used for the resample (same convention as
        ``resample_complex``: ``source = output - offset``).

    Returns
    -------
    numpy.ndarray
        Secondary on the reference grid with original-domain phase restored.

    """
    from scipy.ndimage import map_coordinates

    if sec_resamp_deramped.ndim != 2 or not np.iscomplexobj(sec_resamp_deramped):
        reject_invalid_state("restore requires a 2-D complex secondary array")
    height, width = sec_resamp_deramped.shape
    rg_off = np.asarray(range_offset_px, dtype=np.float64)
    az_off = np.asarray(azimuth_offset_px, dtype=np.float64)
    if rg_off.shape != (height, width) or az_off.shape != (height, width):
        reject_invalid_state("offset fields must match secondary resample shape")

    phi_sec = tops_carrier_phase(secondary_carrier, height, width, dtype=np.float64)
    az_idx = np.arange(height, dtype=np.float64)[:, None]
    rg_idx = np.arange(width, dtype=np.float64)[None, :]
    src_az = az_idx - az_off
    src_rg = rg_idx - rg_off
    phi_src = map_coordinates(
        phi_sec,
        [src_az.ravel(), src_rg.ravel()],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(height, width)
    return _apply_carrier_phase(sec_resamp_deramped, phi_src, sign=+1.0)


def carrier_phase_at_points(
    model: TOPSCarrierModel,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    centre_row: float,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Analytical TOPS carrier phase at arbitrary fractional pixel coordinates.

    Evaluates the Sentinel-1 steering carrier and range-dependent Doppler phase
    directly at fractional row and column coordinates.

    Parameters
    ----------
    model : TOPSCarrierModel
        Burst carrier parameters (built for the burst window being reramped).
    rows, cols : numpy.ndarray
        Fractional pixel coordinates on the burst-local grid (row 0 = first
        burst line, col 0 = first range sample). Same convention as
        :func:`tops_carrier_phase`; must broadcast to a common shape.
    centre_row : float
        Integer azimuth centre ``n_lines // 2`` of the burst window used by
        Sentinel-1 TOPS carrier evaluation.
    dtype : numpy.dtype, optional
        Output phase dtype. Default ``float32``.

    Returns
    -------
    numpy.ndarray
        Real carrier phase in radians, broadcast shape of ``rows``/``cols``.

    """
    if rows.shape != cols.shape:
        reject_invalid_state("rows and cols must have the same shape")
    rows64 = np.asarray(rows, dtype=np.float64)
    cols64 = np.asarray(cols, dtype=np.float64)
    tau = model.slant_range_time0_s + cols64 / model.range_sampling_rate_hz
    f_dc = _poly_eval(model.doppler_centroid_hz, tau - model.doppler_t0_s)
    k_a = _poly_eval(model.fm_rate_hz_s, tau - model.fm_t0_s)
    tau_start = model.burst_start_slant_range_time_s
    f_dc_start = _poly_eval(
        model.doppler_centroid_hz,
        np.asarray(tau_start - model.doppler_t0_s),
    )
    k_a_start = _poly_eval(
        model.fm_rate_hz_s,
        np.asarray(tau_start - model.fm_t0_s),
    )
    eta_ref = (f_dc_start / k_a_start) - (f_dc / k_a)
    k_s = model.azimuth_steering_rate_hz_s
    k_t = k_s / (1.0 - k_s / k_a)
    eta = (rows64 - float(centre_row)) * model.azimuth_time_interval_s
    steering_phase = np.pi * k_t * (eta - eta_ref) ** 2
    doppler_phase = 2.0 * np.pi * f_dc * eta
    phase = steering_phase + doppler_phase
    return np.asarray(phase, dtype=dtype)


def deramp_reramp_roundtrip_error(
    samples: np.ndarray,
    model: TOPSCarrierModel,
) -> float:
    """Return max relative complex error of deramp→reramp on ``samples``."""
    restored = reramp(deramp(samples, model), model)
    denom = np.maximum(np.abs(samples), 1e-12)
    return float(np.max(np.abs(restored - samples) / denom))
