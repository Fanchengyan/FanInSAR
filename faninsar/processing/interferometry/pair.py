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
    message = "block reduce expects a two-dimensional array"
    logger.error(message)
    raise ValueError(message)


def mask_invalid_looks(
    complex_ifg: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    amp_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Replace zero-amplitude looks with NaN phase-safe complex values.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Multilooked complex interferogram.
    coherence : numpy.ndarray, optional
        Matching coherence layer.
    amp_eps : float, optional
        Amplitude threshold below which a look is invalid.

    Returns
    -------
    complex_ifg, coherence, wrapped_phase
        Copies with invalid looks set to NaN.

    """
    ifg = np.asarray(complex_ifg, dtype=np.complex64).copy()
    amp = np.abs(ifg)
    invalid = ~np.isfinite(amp) | (amp <= amp_eps)
    if np.any(invalid):
        ifg[invalid] = np.nan + 1j * np.nan
    phase = np.angle(ifg).astype(np.float32)
    phase = np.where(invalid, np.nan, phase)
    coh_out = None
    if coherence is not None:
        coh_out = np.asarray(coherence, dtype=np.float32).copy()
        coh_out[invalid] = np.nan
    return ifg, coh_out, phase


#: Default amplitude threshold for dead-pixel masking during multilook.
#: SLC pixels with amplitude below this value on either input are excluded
#: from the look-window average. The threshold is relative - typical S1 IW
#: SLC amplitudes range from ~5 to ~100, so 3.0 catches digitisation
#: artefacts (e.g. ``0+1j`` dead pixels) without affecting weak scatterers.
#: Set to 0.0 to disable masking (unweighted boxcar average, original behavior).
DEAD_PIXEL_AMP_THRESHOLD: float = 0.0


def form_interferogram(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    multilook: tuple[int, int] = (1, 1),
    dead_pixel_amp_threshold: float = DEAD_PIXEL_AMP_THRESHOLD,
) -> InterferogramProduct:
    """Form a complex interferogram and coherence from two complex SLCs.

    When ``multilook`` is greater than ``(1, 1)``, the product, primary
    power and secondary power are accumulated in non-overlapping blocks
    without materialising a full-resolution interferogram. On a full S1
    IW burst this avoids ~1 GB of temporary complex/float64 planes that
    would otherwise peak during ``primary * conj(secondary)``.

    Low-amplitude (dead) SLC pixels — e.g. digitisation artefacts where
    ``|slc| ~ 1`` while neighbours are ``|slc| ~ 10-50`` - are masked
    before the look-window average so their noisy phase does not
    contaminate the multilooked interferogram.  A pixel is masked when
    **either** input amplitude falls below *dead_pixel_amp_threshold*.
    The denominator becomes the count of valid (unmasked) pixels in
    each look window, which is the standard *valid-pixel-weighted*
    multilook used by most InSAR processors.

    Parameters
    ----------
    primary, secondary : numpy.ndarray
        Coregistered complex arrays on the same grid.
    multilook : tuple[int, int], optional
        ``(azimuth_looks, range_looks)`` non-overlapping boxcar looks. The
        output is downsampled by these factors (true multilook), not merely
        smoothed.
    dead_pixel_amp_threshold : float, optional
        SLC amplitude below which a pixel is excluded from the multilook
        average. Set to 0 to disable dead-pixel masking (revert to the
        unweighted boxcar average).

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

    use_dead_mask = dead_pixel_amp_threshold > 0.0 and (az_looks > 1 or rg_looks > 1)

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
            ir = pr * sr + pi * si
            ii = pi * sr - pr * si
            pp = pr * pr + pi * pi
            ss = sr * sr + si * si
            if use_dead_mask:
                # Mask pixels where either SLC amplitude is below threshold.
                # The dead-pixel weight is 0 (excluded); valid pixels get 1.
                amp_p = np.sqrt(np.maximum(pp, 0.0))
                amp_s = np.sqrt(np.maximum(ss, 0.0))
                valid = (amp_p >= dead_pixel_amp_threshold) & (
                    amp_s >= dead_pixel_amp_threshold
                )
                wgt = valid.astype(np.float64)
                w_sum = wgt.sum(axis=(0, 2))
                # Avoid division by zero: where no valid pixels, fall back
                # to unweighted average (will be NaN-flagged later via
                # power check).
                safe_w = np.where(w_sum > 0, w_sum, 1.0)
                ifg_real[i] = (ir * wgt).sum(axis=(0, 2)) / safe_w
                ifg_imag[i] = (ii * wgt).sum(axis=(0, 2)) / safe_w
                power_pri[i] = (pp * wgt).sum(axis=(0, 2)) / safe_w
                power_sec[i] = (ss * wgt).sum(axis=(0, 2)) / safe_w
            else:
                ifg_real[i] = ir.sum(axis=(0, 2)) * inv_looks
                ifg_imag[i] = ii.sum(axis=(0, 2)) * inv_looks
                power_pri[i] = pp.sum(axis=(0, 2)) * inv_looks
                power_sec[i] = ss.sum(axis=(0, 2)) * inv_looks
        ifg = ifg_real + 1j * ifg_imag
        logger.info(
            "Multilook %s -> output shape %s",
            multilook,
            ifg.shape,
        )

    denom = np.sqrt(np.maximum(power_pri * power_sec, 1e-30))
    coherence = np.abs(ifg) / denom
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    amplitude = np.abs(ifg).astype(np.float32)
    # Looks with zero power on either input are not data — mark NaN so
    # downstream plots/metrics do not paint phase=0 as a solid black edge.
    invalid = (power_pri <= 0.0) | (power_sec <= 0.0) | ~np.isfinite(power_pri)
    if np.any(invalid):
        ifg = np.asarray(ifg, dtype=np.complex64).copy()
        ifg[invalid] = np.nan + 1j * np.nan
        coherence = coherence.copy()
        coherence[invalid] = np.nan
        amplitude = amplitude.copy()
        amplitude[invalid] = np.nan
    wrapped = np.angle(ifg).astype(np.float32)
    wrapped = np.where(np.isfinite(ifg.real) & np.isfinite(ifg.imag), wrapped, np.nan)
    return InterferogramProduct(
        complex_ifg=ifg.astype(np.complex64, copy=False),
        coherence=coherence,
        wrapped_phase=wrapped.astype(np.float32, copy=False),
        amplitude=amplitude,
    )


def goldstein_filter(
    complex_ifg: np.ndarray,
    *,
    alpha: float = 0.5,
    window: int = 32,
) -> np.ndarray:
    """Apply the Goldstein-Werner adaptive spectral filter (ISCE2 psfilt).

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram.
    alpha : float, optional
        Filter exponent in ``[0, 1]``.
    window : int, optional
        Square FFT patch size. Default 32 (ISCE2 ``NFFT``).

    Returns
    -------
    numpy.ndarray
        Filtered complex interferogram with the original magnitude restored.

    Notes
    -----
    Faithful port of ISCE2's ``mroipac.filter`` ``psfilt``: a triangular
    window, ``|spectrum|**alpha`` spectral weighting, weighted overlap-add
    without renormalisation, and a final magnitude rescale to the input.
    This matches ISCE2's ``filt_topophase.flat`` so the filtered products
    are directly comparable.

    """
    if complex_ifg.ndim != 2 or not np.iscomplexobj(complex_ifg):
        reject_invalid_state("Goldstein filter requires a 2-D complex array")
    if not 0.0 <= alpha <= 1.0:
        reject_invalid_state("Goldstein alpha must be in [0, 1]")
    if window < 8 or window % 2 != 0:
        reject_invalid_state("Goldstein window must be an even integer >= 8")

    height, width = complex_ifg.shape
    if height < window or width < window:
        return complex_ifg.astype(np.complex64, copy=False)

    smoothed = np.zeros((height, width), dtype=np.complex64)
    step = window // 2
    half = window / 2
    axis = np.arange(window, dtype=np.float64)
    triangular = 1.0 - np.abs(2.0 * (axis - half) / (window + 1))
    window2d = np.outer(triangular, triangular) / (window * window)

    for row in range(0, height, step):
        r1 = min(row + window, height)
        for col in range(0, width, step):
            c1 = min(col + window, width)
            patch = np.zeros((window, window), dtype=np.complex64)
            patch[: r1 - row, : c1 - col] = complex_ifg[row:r1, col:c1]
            valid = patch != 0
            spectrum = np.fft.fft2(patch)
            power = spectrum.real**2 + spectrum.imag**2
            spectrum = spectrum * power ** (alpha / 2.0)
            filtered = np.fft.ifft2(spectrum) * (window * window)
            weight_block = window2d * filtered
            for i1 in range(window):
                row_out = row + i1
                if row_out >= height:
                    break
                for j1 in range(window):
                    col_out = col + j1
                    if col_out >= width:
                        break
                    if valid[i1, j1]:
                        smoothed[row_out, col_out] += weight_block[i1, j1]
                    else:
                        smoothed[row_out, col_out] = 0

    input_mag = np.abs(complex_ifg)
    smoothed_mag = np.abs(smoothed)
    mask = (smoothed_mag > 0) & (input_mag > 0)
    smoothed[mask] *= input_mag[mask] / smoothed_mag[mask]
    return smoothed
