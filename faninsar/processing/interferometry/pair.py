"""Complex interferogram formation, coherence, multilook and Goldstein filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faninsar.logging import reject, setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class InterferogramProduct:
    """Complex interferogram and derived quality layers."""

    complex: np.ndarray
    coherence: np.ndarray
    wrapped_phase: np.ndarray
    amplitude: np.ndarray
    valid_mask: np.ndarray | None = None


def validate_coherence_window(
    value: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Validate a centered coherence support before numerical work.

    Parameters
    ----------
    value : tuple of int or None
        ``(azimuth, range)`` support in pixels. ``None`` selects direct MLE
        within each output multilook block.

    Returns
    -------
    tuple of int or None
        The validated support.

    Raises
    ------
    ValueError
        If either axis is not an odd integer of at least three pixels.

    """
    if value is None:
        return None
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(axis) is not int for axis in value)
        or any(axis < 3 or axis % 2 == 0 for axis in value)
    ):
        message = "coherence_window axes must be odd integers >= 3"
        reject(message, ValueError)
    return value


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
    return reject(message, ValueError)


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


def _legacy_form_interferogram(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    multilook: tuple[int, int] = (1, 1),
    coherence_window: tuple[int, int] | None = (5, 5),
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
    coherence_window : tuple[int, int] or None, optional
        Centered stride-one ``(azimuth, range)`` support for the first MLE
        coherence stage. Axes must be odd integers at least 3. ``None``
        computes direct MLE in each output block.
    dead_pixel_amp_threshold : float, optional
        SLC amplitude below which a pixel is excluded from the multilook
        average. Set to 0 to disable dead-pixel masking (revert to the
        unweighted boxcar average).

    Returns
    -------
    InterferogramProduct
        Complex interferogram, coherence, wrapped phase and amplitude.

    """
    coherence_window = validate_coherence_window(coherence_window)
    if primary.shape != secondary.shape or primary.ndim != 2:
        reject_invalid_state("interferogram inputs must be matching 2-D arrays")
    if not np.iscomplexobj(primary) or not np.iscomplexobj(secondary):
        reject_invalid_state("interferogram inputs must remain complex")
    az_looks, rg_looks = multilook
    if (
        type(az_looks) is not int
        or type(rg_looks) is not int
        or az_looks < 1
        or rg_looks < 1
    ):
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
        complex=ifg.astype(np.complex64, copy=False),
        coherence=coherence,
        wrapped_phase=wrapped.astype(np.float32, copy=False),
        amplitude=amplitude,
    )


def _sliding_coherence(
    primary: np.ndarray,
    secondary: np.ndarray,
    window: tuple[int, int],
) -> np.ndarray:
    """Compute stride-one centered coherence MLE with clipped edges.

    Boxcar sums via :func:`scipy.ndimage.uniform_filter` reproduce the
    clipped window support and the exact MLE ratio while keeping the
    estimator O(N) instead of per-pixel Python loops.
    """
    from scipy.ndimage import uniform_filter

    size = (int(window[0]), int(window[1]))
    finite = np.isfinite(primary.real) & np.isfinite(primary.imag)
    finite &= np.isfinite(secondary.real) & np.isfinite(secondary.imag)
    p = np.where(finite, primary, 0).astype(np.complex128)
    s = np.where(finite, secondary, 0).astype(np.complex128)
    cross = p * np.conjugate(s)

    def box(values: np.ndarray) -> np.ndarray:
        return uniform_filter(values, size=size, mode="constant", cval=0.0)

    numerator = np.hypot(box(cross.real), box(cross.imag))
    denominator = np.sqrt(
        np.maximum(box(np.abs(p) ** 2), 0.0) * np.maximum(box(np.abs(s) ** 2), 0.0)
    )
    counts = box(finite.astype(np.float64)) * float(size[0] * size[1])
    valid = (counts >= 2.0) & (denominator > 0.0)
    output = np.full(primary.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=valid)
    output[~valid] = np.nan
    return output


def _block_finite_mean(array: np.ndarray, multilook: tuple[int, int]) -> np.ndarray:
    """Average finite values over non-overlapping blocks, dropping tails."""
    height, width = array.shape
    az_looks, rg_looks = multilook
    output = np.full(
        (height // az_looks, width // rg_looks),
        np.nan,
        dtype=np.float64,
    )
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            block = array[
                row * az_looks : min((row + 1) * az_looks, height),
                col * rg_looks : min((col + 1) * rg_looks, width),
            ]
            finite = block[np.isfinite(block)]
            if finite.size:
                output[row, col] = finite.mean()
    return output


def _sliding_coherence_from_fields(
    cross: np.ndarray,
    primary_power: np.ndarray,
    secondary_power: np.ndarray,
    valid_mask: np.ndarray,
    window: tuple[int, int],
) -> np.ndarray:
    """Native-window MLE ratios from selected cross-products and powers."""
    from scipy.ndimage import uniform_filter

    size = (int(window[0]), int(window[1]))
    support = np.asarray(valid_mask, dtype=bool)
    cross_z = np.where(support, cross, 0).astype(np.complex128)
    power_pri = np.where(support, primary_power, 0).astype(np.float64)
    power_sec = np.where(support, secondary_power, 0).astype(np.float64)

    def box(values: np.ndarray) -> np.ndarray:
        return uniform_filter(values, size=size, mode="constant", cval=0.0)

    numerator = np.hypot(box(cross_z.real), box(cross_z.imag))
    denominator = np.sqrt(
        np.maximum(box(power_pri), 0.0) * np.maximum(box(power_sec), 0.0)
    )
    counts = box(support.astype(np.float64)) * float(size[0] * size[1])
    valid = (counts >= 2.0) & (denominator > 0.0)
    output = np.full(cross.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=valid)
    output[~valid] = np.nan
    return output


def _complete_look_bounds(
    native_origin: tuple[int, int],
    input_shape: tuple[int, int],
    output_window: tuple[int, int, int, int],
    multilook: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return ``(first_i, last_i, first_j, last_j)`` for complete global looks."""
    az_looks, rg_looks = multilook
    row0, row1, col0, col1 = output_window
    first_i = (row0 + az_looks - 1) // az_looks
    last_i = row1 // az_looks
    first_j = (col0 + rg_looks - 1) // rg_looks
    last_j = col1 // rg_looks
    if last_i <= first_i or last_j <= first_j:
        return first_i, last_i, first_j, last_j
    native_row0 = first_i * az_looks
    native_row1 = last_i * az_looks
    native_col0 = first_j * rg_looks
    native_col1 = last_j * rg_looks
    input_row0, input_col0 = native_origin
    input_row1 = input_row0 + input_shape[0]
    input_col1 = input_col0 + input_shape[1]
    if (
        native_row0 < input_row0
        or native_row1 > input_row1
        or native_col0 < input_col0
        or native_col1 > input_col1
    ):
        reject_invalid_state(
            "selected-field interferogram input coverage does not contain the "
            "requested complete look blocks"
        )
    return first_i, last_i, first_j, last_j


def form_interferogram_from_fields(
    cross: np.ndarray,
    primary_power: np.ndarray,
    secondary_power: np.ndarray,
    valid_mask: np.ndarray,
    *,
    native_origin: tuple[int, int],
    output_window: tuple[int, int, int, int],
    multilook: tuple[int, int] = (1, 1),
    coherence_window: tuple[int, int] | None = None,
) -> InterferogramProduct:
    """Form an IFG from selected native cross-products and endpoint powers.

    Looks are aligned to global native origin ``(0, 0)``. Direct MLE
    (``coherence_window is None``) averages valid samples in each complete
    look and requires two jointly valid samples for coherence. Optional
    sliding MLE computes native-window ratios first, then finite-only look
    means. Supported zero-valued looks, including two-sample cancellation,
    remain valid.

    Parameters
    ----------
    cross, primary_power, secondary_power : numpy.ndarray
        Window-local selected native fields. ``cross`` is complex.
    valid_mask : numpy.ndarray
        Explicit scientific support; amplitude is not used as a proxy.
    native_origin : tuple of int
        Global native coordinates of ``cross[0, 0]``.
    output_window : tuple of int
        Half-open global native window ``(r0, r1, c0, c1)`` whose complete
        looks are returned.
    multilook : tuple of int, optional
        ``(azimuth, range)`` look factors.
    coherence_window : tuple of int or None, optional
        Odd native sliding support, or ``None`` for direct block MLE.

    Returns
    -------
    InterferogramProduct
        Look-grid complex IFG, coherence, wrapped phase, amplitude, and mask.

    Raises
    ------
    InvalidProcessingStateError
        If shapes disagree, looks are invalid, or the input crop does not
        cover a requested complete look block.

    """
    coherence_window = validate_coherence_window(coherence_window)
    cross_arr = np.asarray(cross)
    power_pri = np.asarray(primary_power)
    power_sec = np.asarray(secondary_power)
    support = np.asarray(valid_mask, dtype=bool)
    if cross_arr.ndim != 2 or not np.iscomplexobj(cross_arr):
        reject_invalid_state("selected-field cross must be a 2-D complex array")
    if power_pri.shape != cross_arr.shape or power_sec.shape != cross_arr.shape:
        reject_invalid_state("selected-field powers must match the cross shape")
    if support.shape != cross_arr.shape:
        reject_invalid_state("selected-field valid_mask must match the cross shape")
    if (
        not isinstance(native_origin, tuple)
        or len(native_origin) != 2
        or any(type(value) is not int for value in native_origin)
    ):
        reject_invalid_state("native_origin must be a tuple of two ints")
    if (
        not isinstance(output_window, tuple)
        or len(output_window) != 4
        or any(type(value) is not int for value in output_window)
        or output_window[1] <= output_window[0]
        or output_window[3] <= output_window[2]
    ):
        reject_invalid_state(
            "output_window must be a non-empty half-open (r0, r1, c0, c1)"
        )
    if (
        not isinstance(multilook, tuple)
        or len(multilook) != 2
        or any(type(axis) is not int or axis < 1 for axis in multilook)
    ):
        reject_invalid_state("multilook factors must be positive integers")
    az_looks, rg_looks = multilook
    first_i, last_i, first_j, last_j = _complete_look_bounds(
        native_origin, cross_arr.shape, output_window, multilook
    )
    n_i = last_i - first_i
    n_j = last_j - first_j
    if n_i <= 0 or n_j <= 0:
        empty = np.zeros((0, 0), dtype=np.complex64)
        empty_f = np.zeros((0, 0), dtype=np.float32)
        empty_m = np.zeros((0, 0), dtype=bool)
        return InterferogramProduct(
            complex=empty,
            coherence=empty_f,
            wrapped_phase=empty_f,
            amplitude=empty_f,
            valid_mask=empty_m,
        )
    native_row0 = first_i * az_looks
    native_col0 = first_j * rg_looks
    loc_r0 = native_row0 - native_origin[0]
    loc_c0 = native_col0 - native_origin[1]
    loc_r1 = loc_r0 + n_i * az_looks
    loc_c1 = loc_c0 + n_j * rg_looks
    region_support = support[loc_r0:loc_r1, loc_c0:loc_c1]
    region_cross = np.where(region_support, cross_arr[loc_r0:loc_r1, loc_c0:loc_c1], 0)
    region_p1 = np.where(
        region_support, power_pri[loc_r0:loc_r1, loc_c0:loc_c1], 0
    ).astype(np.float64)
    region_p2 = np.where(
        region_support, power_sec[loc_r0:loc_r1, loc_c0:loc_c1], 0
    ).astype(np.float64)
    shaped = (n_i, az_looks, n_j, rg_looks)
    counts = region_support.reshape(shaped).sum(axis=(1, 3))
    sum_cross = region_cross.reshape(shaped).sum(axis=(1, 3))
    sum_p1 = region_p1.reshape(shaped).sum(axis=(1, 3))
    sum_p2 = region_p2.reshape(shaped).sum(axis=(1, 3))
    safe = np.maximum(counts, 1)
    ifg = (sum_cross / safe).astype(np.complex64)
    mean_p1 = sum_p1 / safe
    mean_p2 = sum_p2 / safe
    valid_out = counts > 0
    ifg = np.where(valid_out, ifg, np.complex64(np.nan + 1j * np.nan))
    if coherence_window is None:
        denominator = np.sqrt(np.maximum(mean_p1, 0.0) * np.maximum(mean_p2, 0.0))
        direct = np.divide(
            np.abs(ifg.astype(np.complex128)),
            denominator,
            out=np.full(ifg.shape, np.nan, dtype=np.float64),
            where=(counts >= 2) & (denominator > 0.0),
        )
        coherence = np.where(counts >= 2, direct, np.nan)
    else:
        native_gamma = _sliding_coherence_from_fields(
            cross_arr, power_pri, power_sec, support, coherence_window
        )
        look_gamma = native_gamma[loc_r0:loc_r1, loc_c0:loc_c1]
        coherence = np.empty((n_i, n_j), dtype=np.float64)
        for row in range(n_i):
            for col in range(n_j):
                block = look_gamma[
                    row * az_looks : (row + 1) * az_looks,
                    col * rg_looks : (col + 1) * rg_looks,
                ]
                finite = block[np.isfinite(block)]
                coherence[row, col] = finite.mean() if finite.size else np.nan
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    coherence = np.where(valid_out, coherence, np.float32(np.nan))
    amplitude = np.abs(ifg).astype(np.float32)
    amplitude = np.where(valid_out, amplitude, np.float32(np.nan))
    wrapped = np.angle(ifg).astype(np.float32)
    wrapped = np.where(valid_out, wrapped, np.float32(np.nan))
    return InterferogramProduct(
        complex=ifg.astype(np.complex64, copy=False),
        coherence=coherence,
        wrapped_phase=wrapped,
        amplitude=amplitude,
        valid_mask=valid_out,
    )


def form_interferogram(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    multilook: tuple[int, int] = (1, 1),
    coherence_window: tuple[int, int] | None = None,
    dead_pixel_amp_threshold: float = DEAD_PIXEL_AMP_THRESHOLD,
) -> InterferogramProduct:
    """Form a valid-sample multilooked IFG with configurable coherence MLE.

    ``coherence_window`` is a centered, stride-one ``(azimuth, range)``
    support with odd axes at least 3. It is computed at full ``H x W`` and
    then reduced by finite-only blocks. ``None`` computes direct MLE per
    output block. Multilook blocks start at ``(0, 0)`` and use ``floor``
    dimensions: a trailing partial look block is dropped, matching ISCE2
    ``mroipac.looks.Looks`` and the geocode ``layer_frame_shape`` contract.
    """
    coherence_window = validate_coherence_window(coherence_window)
    if primary.shape != secondary.shape or primary.ndim != 2:
        reject_invalid_state("interferogram inputs must be matching 2-D arrays")
    if not np.iscomplexobj(primary) or not np.iscomplexobj(secondary):
        reject_invalid_state("interferogram inputs must remain complex")
    if (
        not isinstance(multilook, tuple)
        or len(multilook) != 2
        or any(type(axis) is not int or axis < 1 for axis in multilook)
    ):
        reject_invalid_state("multilook factors must be positive integers")
    az_looks, rg_looks = multilook
    use_dead_mask = dead_pixel_amp_threshold > 0.0
    height, width = primary.shape
    out_h = height // az_looks
    out_w = width // rg_looks
    finite_pair = np.isfinite(primary.real) & np.isfinite(primary.imag)
    finite_pair &= np.isfinite(secondary.real) & np.isfinite(secondary.imag)
    if use_dead_mask:
        finite_pair &= np.abs(primary) >= dead_pixel_amp_threshold
        finite_pair &= np.abs(secondary) >= dead_pixel_amp_threshold
    ifg = np.full((out_h, out_w), np.nan + 0j, dtype=np.complex64)
    power_pri = np.full((out_h, out_w), np.nan, dtype=np.float64)
    power_sec = np.full((out_h, out_w), np.nan, dtype=np.float64)
    valid_out = np.zeros((out_h, out_w), dtype=bool)
    sample_counts = np.zeros((out_h, out_w), dtype=np.intp)
    for row in range(out_h):
        r0, r1 = row * az_looks, min((row + 1) * az_looks, height)
        for col in range(out_w):
            c0, c1 = col * rg_looks, min((col + 1) * rg_looks, width)
            support = finite_pair[r0:r1, c0:c1]
            if not np.any(support):
                continue
            p = primary[r0:r1, c0:c1][support]
            s = secondary[r0:r1, c0:c1][support]
            sample_counts[row, col] = p.size
            ifg[row, col] = np.mean(p * np.conjugate(s))
            power_pri[row, col] = np.mean(np.abs(p) ** 2)
            power_sec[row, col] = np.mean(np.abs(s) ** 2)
            valid_out[row, col] = bool(
                np.isfinite(ifg[row, col].real)
                and np.isfinite(ifg[row, col].imag)
                and power_pri[row, col] > 0
                and power_sec[row, col] > 0
            )
    denominator = np.sqrt(power_pri * power_sec)
    direct = np.divide(
        np.abs(ifg),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    if coherence_window is None:
        coherence = np.where(sample_counts >= 2, direct, np.nan)
    else:
        coherence = _block_finite_mean(
            _sliding_coherence(primary, secondary, coherence_window), multilook
        )
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    invalid = ~valid_out | ~np.isfinite(ifg.real) | ~np.isfinite(ifg.imag)
    if np.any(invalid):
        ifg[invalid] = np.nan + 1j * np.nan
        coherence[invalid] = np.nan
    amplitude = np.abs(ifg).astype(np.float32)
    amplitude[invalid] = np.nan
    wrapped = np.angle(ifg).astype(np.float32)
    wrapped[invalid] = np.nan
    return InterferogramProduct(
        complex=ifg,
        coherence=coherence,
        wrapped_phase=wrapped,
        amplitude=amplitude,
        valid_mask=valid_out,
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
