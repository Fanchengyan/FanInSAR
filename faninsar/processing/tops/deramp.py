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

    def __post_init__(self) -> None:
        """Validate carrier model parameters."""
        if self.radar_frequency_hz <= 0 or self.range_sampling_rate_hz <= 0:
            reject_invalid_state("radar frequency and range sampling rate must be > 0")
        if self.azimuth_time_interval_s <= 0:
            reject_invalid_state("azimuth time interval must be > 0")
        if not self.doppler_centroid_hz or not self.fm_rate_hz_s:
            reject_invalid_state("Doppler and FM-rate polynomials are required")


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

    The phase model is:

    ``phi(t, tau) = 2π [ f_dc(tau) * t + 0.5 * K_a(tau) * t^2 ]``

    where ``t`` is azimuth time relative to burst centre and ``tau`` is
    range time. Polynomials use the annotation ``t0`` references.

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
    tau = range_time_axis(model, n_samples)
    t_az = azimuth_time_axis(model, n_lines)[:, None]
    f_dc = _poly_eval(model.doppler_centroid_hz, tau - model.doppler_t0_s)[None, :]
    k_a = _poly_eval(model.fm_rate_hz_s, tau - model.fm_t0_s)[None, :]
    phase = (2.0 * np.pi) * (f_dc * t_az + 0.5 * k_a * t_az**2)
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
    t_az_rows: np.ndarray,
    n_samples: int,
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Carrier phase for a subset of azimuth times (relative to burst centre)."""
    tau = range_time_axis(model, n_samples)
    t_az = np.asarray(t_az_rows, dtype=np.float64)[:, None]
    f_dc = _poly_eval(model.doppler_centroid_hz, tau - model.doppler_t0_s)[None, :]
    k_a = _poly_eval(model.fm_rate_hz_s, tau - model.fm_t0_s)[None, :]
    phase = (2.0 * np.pi) * (f_dc * t_az + 0.5 * k_a * t_az**2)
    return np.asarray(phase, dtype=dtype)


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
        phase = tops_carrier_phase(
            model, n_lines, n_samples, dtype=phase_dtype
        )
        return _apply_carrier_phase(samples, phase, sign=sign)

    out = np.empty_like(samples)
    t_az = azimuth_time_axis(model, n_lines)
    for row0 in range(0, n_lines, row_chunk):
        row1 = min(row0 + row_chunk, n_lines)
        phase = _carrier_phase_rows(
            model, t_az[row0:row1], n_samples, dtype=phase_dtype
        )
        out[row0:row1] = _apply_carrier_phase(
            samples[row0:row1], phase, sign=sign
        )
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


def deramp_reramp_roundtrip_error(
    samples: np.ndarray,
    model: TOPSCarrierModel,
) -> float:
    """Return max relative complex error of deramp→reramp on ``samples``."""
    restored = reramp(deramp(samples, model), model)
    denom = np.maximum(np.abs(samples), 1e-12)
    return float(np.max(np.abs(restored - samples) / denom))
