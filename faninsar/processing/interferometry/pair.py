"""Complex interferogram formation, coherence, multilook and Goldstein filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class InterferogramProduct:
    """Complex interferogram and derived quality layers."""

    complex_ifg: np.ndarray
    coherence: np.ndarray
    wrapped_phase: np.ndarray
    amplitude: np.ndarray


def _block_reduce(array: np.ndarray, az_looks: int, rg_looks: int) -> np.ndarray:
    """Average non-overlapping blocks (true multilook downsample)."""
    height, width = array.shape[:2]
    h = (height // az_looks) * az_looks
    w = (width // rg_looks) * rg_looks
    cropped = array[:h, :w]
    if array.ndim == 2:
        reshaped = cropped.reshape(h // az_looks, az_looks, w // rg_looks, rg_looks)
        return reshaped.mean(axis=(1, 3))
    raise ValueError("block reduce expects 2-D array")


def form_interferogram(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    multilook: tuple[int, int] = (1, 1),
) -> InterferogramProduct:
    """Form a complex interferogram and coherence from two complex SLCs.

    When ``multilook`` is greater than ``(1, 1)``, the product, primary
    power and secondary power are accumulated in non-overlapping blocks
    without materialising a full-resolution interferogram. On a full S1
    IW burst this avoids ~1 GB of temporary complex/float64 planes that
    would otherwise peak during ``primary * conj(secondary)``.

    Parameters
    ----------
    primary, secondary : numpy.ndarray
        Coregistered complex arrays on the same grid.
    multilook : tuple[int, int], optional
        ``(azimuth_looks, range_looks)`` non-overlapping boxcar looks. The
        output is downsampled by these factors (true multilook), not merely
        smoothed.

    Returns
    -------
    InterferogramProduct
        Complex interferogram, coherence, wrapped phase and amplitude.

    """
    if primary.shape != secondary.shape or primary.ndim != 2:
        reject_invalid_state("interferogram inputs must be matching 2-D arrays")
    if not np.iscomplexobj(primary) or not np.iscomplexobj(secondary):
        reject_invalid_state("interferogram inputs must remain complex")
    az_looks, rg_looks = multilook
    if az_looks < 1 or rg_looks < 1:
        reject_invalid_state("multilook factors must be >= 1")

    if az_looks == 1 and rg_looks == 1:
        ifg = primary * np.conjugate(secondary)
        power_pri = (primary.real**2 + primary.imag**2).astype(np.float64)
        power_sec = (secondary.real**2 + secondary.imag**2).astype(np.float64)
    else:
        height, width = primary.shape
        h = (height // az_looks) * az_looks
        w = (width // rg_looks) * rg_looks
        out_h = h // az_looks
        out_w = w // rg_looks
        # Accumulate look means per output row so peak temps stay O(az_looks * W)
        # rather than a full-resolution complex interferogram.
        ifg_real = np.empty((out_h, out_w), dtype=np.float64)
        ifg_imag = np.empty((out_h, out_w), dtype=np.float64)
        power_pri = np.empty((out_h, out_w), dtype=np.float64)
        power_sec = np.empty((out_h, out_w), dtype=np.float64)
        inv_looks = 1.0 / float(az_looks * rg_looks)
        for i in range(out_h):
            r0 = i * az_looks
            r1 = r0 + az_looks
            p = primary[r0:r1, :w]
            s = secondary[r0:r1, :w]
            pr = p.real.reshape(az_looks, out_w, rg_looks)
            pi = p.imag.reshape(az_looks, out_w, rg_looks)
            sr = s.real.reshape(az_looks, out_w, rg_looks)
            si = s.imag.reshape(az_looks, out_w, rg_looks)
            # ifg = p * conj(s)
            ifg_real[i] = (pr * sr + pi * si).sum(axis=(0, 2)) * inv_looks
            ifg_imag[i] = (pi * sr - pr * si).sum(axis=(0, 2)) * inv_looks
            power_pri[i] = (pr * pr + pi * pi).sum(axis=(0, 2)) * inv_looks
            power_sec[i] = (sr * sr + si * si).sum(axis=(0, 2)) * inv_looks
        ifg = ifg_real + 1j * ifg_imag
        logger.info(
            "Multilook %s -> output shape %s",
            multilook,
            ifg.shape,
        )

    denom = np.sqrt(np.maximum(power_pri * power_sec, 1e-30))
    coherence = np.abs(ifg) / denom
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    wrapped = np.angle(ifg).astype(np.float32)
    amplitude = np.abs(ifg).astype(np.float32)
    return InterferogramProduct(
        complex_ifg=ifg.astype(np.complex64, copy=False),
        coherence=coherence,
        wrapped_phase=wrapped,
        amplitude=amplitude,
    )


def goldstein_filter(
    complex_ifg: np.ndarray,
    *,
    alpha: float = 0.5,
    window: int = 32,
) -> np.ndarray:
    """Apply a simplified Goldstein adaptive spectral filter.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram.
    alpha : float, optional
        Filter exponent in ``[0, 1]``.
    window : int, optional
        Square FFT patch size (power of two recommended).

    Returns
    -------
    numpy.ndarray
        Filtered complex interferogram.

    """
    if complex_ifg.ndim != 2 or not np.iscomplexobj(complex_ifg):
        reject_invalid_state("Goldstein filter requires a 2-D complex array")
    if not 0.0 <= alpha <= 1.0:
        reject_invalid_state("Goldstein alpha must be in [0, 1]")
    if window < 8 or window % 2 != 0:
        reject_invalid_state("Goldstein window must be an even integer >= 8")

    height, width = complex_ifg.shape
    # Skip expensive filter on tiny arrays
    if height < window or width < window:
        return complex_ifg.astype(np.complex64, copy=False)

    out = np.zeros_like(complex_ifg, dtype=np.complex64)
    weight = np.zeros((height, width), dtype=np.float32)
    step = window // 2
    taper = np.hanning(window)
    window2d = np.outer(taper, taper).astype(np.float32)

    for row in range(0, max(height - window + 1, 1), step):
        for col in range(0, max(width - window + 1, 1), step):
            r1 = min(row + window, height)
            c1 = min(col + window, width)
            patch = np.zeros((window, window), dtype=np.complex64)
            pr = r1 - row
            pc = c1 - col
            patch[:pr, :pc] = complex_ifg[row:r1, col:c1]
            spectrum = np.fft.fft2(patch * window2d)
            magnitude = np.abs(spectrum)
            scale = magnitude**alpha
            filtered = np.fft.ifft2(spectrum * scale)
            out[row:r1, col:c1] += filtered[:pr, :pc] * window2d[:pr, :pc]
            weight[row:r1, col:c1] += window2d[:pr, :pc]

    mask = weight > 0
    out[mask] /= weight[mask]
    return out
