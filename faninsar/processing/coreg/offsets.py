"""Geometry-driven coarse offset fields and fine correlation refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import map_coordinates

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from faninsar.processing.tops.deramp import TOPSCarrierModel

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class OffsetFieldResult:
    """Dense range/azimuth offsets with coverage and uncertainty."""

    range_offset_px: np.ndarray
    azimuth_offset_px: np.ndarray
    coverage: np.ndarray
    uncertainty_px: np.ndarray


def geometry_shift_offsets(
    shape: tuple[int, int],
    *,
    range_shift_px: float,
    azimuth_shift_px: float,
) -> OffsetFieldResult:
    """Build a constant geometry offset field for a declared global shift.

    Parameters
    ----------
    shape : tuple[int, int]
        Output field shape ``(azimuth, range)``.
    range_shift_px, azimuth_shift_px : float
        Constant offsets in pixels (secondary relative to reference).

    Returns
    -------
    OffsetFieldResult
        Dense constant offsets with full coverage and zero uncertainty.

    """
    height, width = shape
    if height <= 0 or width <= 0:
        reject_invalid_state("offset field shape must be positive")
    return OffsetFieldResult(
        range_offset_px=np.full(shape, range_shift_px, dtype=np.float32),
        azimuth_offset_px=np.full(shape, azimuth_shift_px, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )


def _refine_peak_subpixel_1d(
    values: np.ndarray,
) -> float:
    """Parabolic sub-pixel refinement of a 1-D correlation peak.

    Fits :math:`a x^2 + b x + c` to the three samples centred on the
    peak and returns the analytic extremum offset relative to the centre
    sample.

    Parameters
    ----------
    values : numpy.ndarray
        Three samples ``[left, centre, right]``.

    Returns
    -------
    float
        Sub-pixel offset of the peak (positive = toward ``right``).

    """
    left, centre, right = float(values[0]), float(values[1]), float(values[2])
    denom = left - 2.0 * centre + right
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (left - right) / denom


def refine_peak_subpixel(
    corr: np.ndarray,
    peak: tuple[int, int],
) -> tuple[float, float]:
    """Refine an integer correlation peak to sub-pixel precision.

    Uses independent parabolic fits along the azimuth and range axes
    through the 3x3 neighbourhood centred on the integer peak.

    Parameters
    ----------
    corr : numpy.ndarray
        2-D correlation surface.
    peak : tuple[int, int]
        Integer peak indices ``(az, rg)``.

    Returns
    -------
    tuple[float, float]
        ``(azimuth_subpx, range_subpx)`` offsets relative to the integer
        peak.

    """
    az_i, rg_i = peak
    h, w = corr.shape
    # Clamp neighbourhood to array bounds
    az0 = max(az_i - 1, 0)
    az1 = min(az_i + 1, h - 1)
    rg0 = max(rg_i - 1, 0)
    rg1 = min(rg_i + 1, w - 1)
    # Extract 3x3 window; if at the edge, the outer sample is duplicated
    # so the parabolic fit degrades gracefully to zero sub-pixel shift.
    window = corr[az0 : az1 + 1, rg0 : rg1 + 1]
    if window.shape != (3, 3):
        # Edge case: duplicate the nearest sample to pad to 3x3
        padded = np.full((3, 3), corr[az_i, rg_i], dtype=np.float64)
        pa0 = 1 - (az_i - az0)
        pa1 = pa0 + window.shape[0]
        pr0 = 1 - (rg_i - rg0)
        pr1 = pr0 + window.shape[1]
        padded[pa0:pa1, pr0:pr1] = window
        window = padded
    az_sub = _refine_peak_subpixel_1d(window[:, 1])
    rg_sub = _refine_peak_subpixel_1d(window[1, :])
    return az_sub, rg_sub


def estimate_global_shift(
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    max_shift: int = 32,
    subpixel: bool = True,
) -> tuple[float, float]:
    """Estimate a global shift via full-image amplitude cross-correlation.

    Prefer :func:`estimate_patch_amplitude_shift` for TOPS / production
    coregistration. A single full-burst FFT peak is biased by the TOPS
    amplitude envelope and invalid borders, and can inject a false azimuth
    residual of several tenths of a pixel.

    Parameters
    ----------
    reference, secondary : numpy.ndarray
        Complex or real 2-D arrays on the same grid.
    max_shift : int, optional
        Maximum absolute search radius in pixels.
    subpixel : bool, optional
        If ``True`` (default), refine the integer peak with a parabolic
        fit in a 3x3 neighbourhood.

    Returns
    -------
    tuple[float, float]
        ``(range_shift_px, azimuth_shift_px)`` of secondary relative to
        reference, where positive range shift means secondary is shifted to
        larger range indices.

    """
    if reference.shape != secondary.shape or reference.ndim != 2:
        reject_invalid_state("shift estimation requires matching 2-D arrays")
    if max_shift < 1:
        reject_invalid_state("max_shift must be >= 1")
    ref = np.abs(np.asarray(reference, dtype=np.complex64))
    sec = np.abs(np.asarray(secondary, dtype=np.complex64))
    ref = ref - float(np.mean(ref))
    sec = sec - float(np.mean(sec))
    # FFT cross-correlation
    f_ref = np.fft.fft2(ref)
    f_sec = np.fft.fft2(sec)
    corr = np.fft.ifft2(f_ref * np.conjugate(f_sec)).real
    corr = np.fft.fftshift(corr)
    cy, cx = (np.array(corr.shape) // 2).tolist()
    y0 = max(cy - max_shift, 0)
    y1 = min(cy + max_shift + 1, corr.shape[0])
    x0 = max(cx - max_shift, 0)
    x1 = min(cx + max_shift + 1, corr.shape[1])
    window = corr[y0:y1, x0:x1]
    peak = np.unravel_index(int(np.argmax(window)), window.shape)
    az_shift = float(peak[0] + y0 - cy)
    rg_shift = float(peak[1] + x0 - cx)
    if subpixel:
        # Map peak back to full correlation coordinates
        full_peak = (int(peak[0] + y0), int(peak[1] + x0))
        az_sub, rg_sub = refine_peak_subpixel(corr, full_peak)
        az_shift += az_sub
        rg_shift += rg_sub
    # Correlation peak location of ref*conj(sec) corresponds to the shift of
    # secondary relative to reference with opposite sign convention for map.
    return -rg_shift, -az_shift


@dataclass(frozen=True, slots=True)
class PatchAmplitudeShiftResult:
    """Robust multi-window amplitude-correlation residual."""

    range_shift_px: float
    azimuth_shift_px: float
    n_valid: int
    snr_median: float
    n_attempted: int


def _patch_ncc_shift(
    ref_win: np.ndarray,
    sec_search: np.ndarray,
    *,
    search_az: int,
    search_rg: int,
    subpixel: bool,
) -> tuple[float, float, float] | None:
    """Return the normalized cross-correlation peak for one Ampcor-style patch.

    Parameters
    ----------
    ref_win : numpy.ndarray
        Reference magnitude window ``(window_az, window_rg)``.
    sec_search : numpy.ndarray
        Secondary magnitude search chip of size
        ``(window_az + 2*search_az, window_rg + 2*search_rg)``.
    search_az, search_rg : int
        Half-width of the integer search range.
    subpixel : bool
        Parabolic peak refinement.

    Returns
    -------
    tuple[float, float, float] or None
        ``(d_rg, d_az, snr)`` in the resample convention
        (positive = secondary content at larger indices), or ``None`` if
        the surface is degenerate.

    """
    waz, wrg = ref_win.shape
    saz, srg = sec_search.shape
    if saz != waz + 2 * search_az or srg != wrg + 2 * search_rg:
        return None
    ref = ref_win.astype(np.float64, copy=False)
    sec = sec_search.astype(np.float64, copy=False)
    ref = ref - float(ref.mean())
    sec = sec - float(sec.mean())
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm < 1e-12:
        return None
    ref = ref / ref_norm

    # Full correlation of search chip against reference template via FFT.
    # Pad to avoid circular wrap; peak of template at shift (0,0) in
    # sec_search sits at index (search_az, search_rg) of the valid lag map.
    fft_shape = (
        int(2 ** int(np.ceil(np.log2(saz + waz - 1)))),
        int(2 ** int(np.ceil(np.log2(srg + wrg - 1)))),
    )
    f_sec = np.fft.rfft2(sec, s=fft_shape)
    f_ref = np.fft.rfft2(ref[::-1, ::-1], s=fft_shape)
    corr_full = np.fft.irfft2(f_sec * f_ref, s=fft_shape).real
    # Valid lags where template fully inside search chip: (2*search+1)^2
    lag_rows = slice(waz - 1, waz - 1 + 2 * search_az + 1)
    lag_columns = slice(wrg - 1, wrg - 1 + 2 * search_rg + 1)
    corr = corr_full[lag_rows, lag_columns]
    if corr.shape != (2 * search_az + 1, 2 * search_rg + 1):
        return None
    # Local energy of secondary under each lag for true NCC
    ones = np.ones((waz, wrg), dtype=np.float64)
    sec_sq = sec * sec
    f_ones = np.fft.rfft2(ones[::-1, ::-1], s=fft_shape)
    f_sec_sq = np.fft.rfft2(sec_sq, s=fft_shape)
    energy = np.fft.irfft2(f_sec_sq * f_ones, s=fft_shape).real
    energy = energy[lag_rows, lag_columns]
    energy = np.maximum(energy, 1e-12)
    ncc = corr / np.sqrt(energy)
    peak_flat = int(np.argmax(ncc))
    peak_az, peak_rg = np.unravel_index(peak_flat, ncc.shape)
    peak_val = float(ncc[peak_az, peak_rg])
    # Peak-to-sidelobe SNR: zero a 3x3 neighbourhood around the peak and take
    # peak / mean(|sidelobe|). Matches the practical cull used with ISCE Ampcor
    # better than (peak-mean)/std, which collapses for band-limited texture.
    sidelobe = ncc.copy()
    a0 = max(int(peak_az) - 1, 0)
    a1 = min(int(peak_az) + 2, ncc.shape[0])
    r0 = max(int(peak_rg) - 1, 0)
    r1 = min(int(peak_rg) + 2, ncc.shape[1])
    sidelobe[a0:a1, r0:r1] = np.nan
    side_mean = float(np.nanmean(np.abs(sidelobe)))
    if not np.isfinite(side_mean) or side_mean < 1e-12:
        return None
    snr = peak_val / side_mean
    az_shift = float(peak_az - search_az)
    rg_shift = float(peak_rg - search_rg)
    if subpixel and 0 < peak_az < ncc.shape[0] - 1 and 0 < peak_rg < ncc.shape[1] - 1:
        az_sub, rg_sub = refine_peak_subpixel(ncc, (int(peak_az), int(peak_rg)))
        az_shift += az_sub
        rg_shift += rg_sub
    # Peak lag: secondary feature is at ref + lag inside search chip that is
    # already centred on the reference window, so lag is secondary-reference.
    # Resample convention source = out - offset needs offset = that lag.
    return rg_shift, az_shift, snr


def estimate_patch_amplitude_shift(
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    window_az: int = 32,
    window_rg: int = 64,
    search_az: int = 16,
    search_rg: int = 16,
    n_az: int = 20,
    n_rg: int = 40,
    snr_threshold: float = 5.0,
    max_abs_residual: float = 1.2,
    margin_rg: int = 1000,
    margin_az: int | None = None,
    subpixel: bool = True,
) -> PatchAmplitudeShiftResult:
    """Estimate residual shift with multi-window magnitude Ampcor (ISCE2-style).

    Mirrors topsApp ``runRangeCoreg`` / ``runAmpcor`` defaults: magnitude-only
    patches (``window_rg x window_az = 64 x 32``), search half-width 16, ~40 x 20
    locations, SNR cull, and ``|residual| < 1.2`` px. Returns the **median**
    residual over surviving patches.

    Parameters
    ----------
    reference, secondary : numpy.ndarray
        Complex or real 2-D arrays on the same grid. Callers should
        integer-pre-align the secondary with the geometry prior so residuals
        are sub-pixel.
    window_az, window_rg : int, optional
        Reference chip size (azimuth, range). Defaults match ISCE2 TOPS.
    search_az, search_rg : int, optional
        Integer search half-width in each axis.
    n_az, n_rg : int, optional
        Number of patch centres in azimuth and range.
    snr_threshold : float, optional
        Minimum correlation SNR to keep a patch (default 5.0).
    max_abs_residual : float, optional
        Reject patches whose residual exceeds this magnitude (default 1.2,
        same cull as ISCE2 ``runRangeCoreg``).
    margin_rg : int, optional
        Range border excluded from patch centres (ISCE2 uses ~1000 samples).
        Reduced automatically when the burst is narrower.
    margin_az : int or None, optional
        Azimuth border; default is half the window height.
    subpixel : bool, optional
        Parabolic peak refinement (default True).

    Returns
    -------
    PatchAmplitudeShiftResult
        Median residual shifts and diagnostic counts. When no patch survives,
        shifts are 0.0 (keep geometry prior).

    """
    if reference.shape != secondary.shape or reference.ndim != 2:
        reject_invalid_state("patch amplitude shift requires matching 2-D arrays")
    if min(window_az, window_rg, search_az, search_rg, n_az, n_rg) < 1:
        reject_invalid_state("patch amplitude parameters must be positive")
    height, width = reference.shape
    ref_mag = np.abs(np.asarray(reference, dtype=np.complex64)).astype(np.float32)
    sec_mag = np.abs(np.asarray(secondary, dtype=np.complex64)).astype(np.float32)

    half_az = window_az // 2
    half_rg = window_rg // 2
    m_az = half_az + search_az + 1 if margin_az is None else int(margin_az)
    m_rg = int(margin_rg)
    # Shrink margins on small chips so unit tests / cropped bursts still work.
    if width < 2 * m_rg + window_rg + 2 * search_rg + 2:
        m_rg = half_rg + search_rg + 1
    if height < 2 * m_az + window_az + 2 * search_az + 2:
        m_az = half_az + search_az + 1
    az0, az1 = m_az, height - m_az
    rg0, rg1 = m_rg, width - m_rg
    if az1 <= az0 or rg1 <= rg0:
        logger.warning(
            "Patch Ampcor: image too small for margins (shape=%s); residual=0",
            reference.shape,
        )
        return PatchAmplitudeShiftResult(0.0, 0.0, 0, 0.0, 0)

    az_centres = np.linspace(az0, az1 - 1, num=n_az, dtype=np.int64)
    rg_centres = np.linspace(rg0, rg1 - 1, num=n_rg, dtype=np.int64)
    d_rg_list: list[float] = []
    d_az_list: list[float] = []
    snr_list: list[float] = []
    n_attempted = 0
    for az_c in az_centres:
        for rg_c in rg_centres:
            r0 = int(az_c) - half_az
            r1 = r0 + window_az
            c0 = int(rg_c) - half_rg
            c1 = c0 + window_rg
            sr0 = r0 - search_az
            sr1 = r1 + search_az
            sc0 = c0 - search_rg
            sc1 = c1 + search_rg
            if sr0 < 0 or sc0 < 0 or sr1 > height or sc1 > width:
                continue
            n_attempted += 1
            result = _patch_ncc_shift(
                ref_mag[r0:r1, c0:c1],
                sec_mag[sr0:sr1, sc0:sc1],
                search_az=search_az,
                search_rg=search_rg,
                subpixel=subpixel,
            )
            if result is None:
                continue
            d_rg, d_az, snr = result
            if snr < snr_threshold:
                continue
            if abs(d_rg) > max_abs_residual or abs(d_az) > max_abs_residual:
                continue
            d_rg_list.append(d_rg)
            d_az_list.append(d_az)
            snr_list.append(snr)

    if not d_rg_list:
        logger.warning(
            "Patch Ampcor: no patches survived cull "
            "(attempted=%d snr>=%.1f |res|<%.2f); residual=0",
            n_attempted,
            snr_threshold,
            max_abs_residual,
        )
        return PatchAmplitudeShiftResult(0.0, 0.0, 0, 0.0, n_attempted)

    rg_med = float(np.median(np.asarray(d_rg_list, dtype=np.float64)))
    az_med = float(np.median(np.asarray(d_az_list, dtype=np.float64)))
    snr_med = float(np.median(np.asarray(snr_list, dtype=np.float64)))
    logger.info(
        "Patch Ampcor residual rg=%.4f az=%.4f n_valid=%d/%d snr_med=%.2f",
        rg_med,
        az_med,
        len(d_rg_list),
        n_attempted,
        snr_med,
    )
    return PatchAmplitudeShiftResult(
        range_shift_px=rg_med,
        azimuth_shift_px=az_med,
        n_valid=len(d_rg_list),
        snr_median=snr_med,
        n_attempted=n_attempted,
    )


def resample_complex(
    samples: np.ndarray,
    *,
    range_offset_px: np.ndarray | float,
    azimuth_offset_px: np.ndarray | float,
    order: int | None = None,
    lanczos_a: int = 4,
    row_chunk: int = 64,
    executor: Literal["torch"] = "torch",
    device: str = "auto",
) -> np.ndarray:
    """Resample complex samples with a phase-preserving kernel.

    Coordinates are source indices for each output pixel:
    ``source = output_index - offset``.

    Complex SLC / interferogram data is a sampled bandlimited signal and
    must be reconstructed with a sinc-family kernel. Bilinear (``order=1``)
    or bicubic (``order=3``) kernels attenuate in-band signal, leak residual
    aliasing, and — for bicubic — introduce a non-flat group delay,
    producing a sub-pixel-offset-dependent *phase bias* that is invisible in
    amplitude but creates decorrelation and burst seams downstream. This
    function therefore defaults to a Lanczos (windowed-sinc) kernel and only
    falls back to spline interpolation of the given ``order`` when the caller
    explicitly opts in (e.g. for already-multilooked complex data with
    bandwidth well below the grid Nyquist, where bilinear is acceptable).

    Full-resolution offset fields are applied in azimuth row tiles so that
    ``np.indices`` / coordinate buffers never materialise for the whole
    burst at once (a full-burst float64 index grid alone is ~0.5 GB and the
    subsequent Lanczos gather would be tens of GB without tiling).

    Parameters
    ----------
    samples : numpy.ndarray
        Complex 2-D source array.
    range_offset_px, azimuth_offset_px : array or float
        Offsets of the secondary relative to the reference grid.
    order : int or None, optional
        If ``None`` (default), use the Lanczos windowed-sinc kernel
        (:func:`lanczos_resample` with half-width ``lanczos_a``), which is
        phase-preserving and the correct choice for full-bandwidth complex
        SAR data. If an integer is given, fall back to
        :func:`scipy.ndimage.map_coordinates` with that spline ``order``
        (1=bilinear, 3=bicubic). Only use a non-``None`` ``order`` for
        already-multilooked complex data where bilinear is acceptable.
    lanczos_a : int, optional
        Lanczos half-width when ``order is None``. ``a=4`` (8-tap) is the
        production default for SLC resampling; ``a=6`` (12-tap) for
        highest-precision demands. Default 4.
    row_chunk : int, optional
        Number of azimuth rows processed per tile when building source
        coordinates. Default 64 (~1.3 M samples on a full IW burst width).
    executor : {"torch"}, optional
        Lanczos compute path. Torch runs the same kernel on CPU, CUDA, or MPS.
        The source SLC is uploaded once per call and row tiles only move
        coordinates.
        Ignored when ``order`` is set (the spline path is always NumPy/SciPy).
        Default ``"torch"``.
    device : {"auto","cpu","cuda","mps"}, optional
        Torch device. ``"auto"`` selects CUDA, then MPS, then CPU.

    Returns
    -------
    numpy.ndarray
        Complex resampled array on the reference grid.

    """
    if samples.ndim != 2 or not np.iscomplexobj(samples):
        reject_invalid_state("complex resampling requires a 2-D complex array")
    if executor != "torch":
        reject_invalid_state(f"unsupported complex resampling executor: {executor}")
    if row_chunk < 1:
        reject_invalid_state("row_chunk must be >= 1")
    height, width = samples.shape
    az_off = np.asarray(azimuth_offset_px, dtype=np.float64)
    rg_off = np.asarray(range_offset_px, dtype=np.float64)
    scalar_az = az_off.ndim == 0
    scalar_rg = rg_off.ndim == 0
    if not scalar_az and az_off.shape != (height, width):
        reject_invalid_state("azimuth_offset_px must be scalar or match samples shape")
    if not scalar_rg and rg_off.shape != (height, width):
        reject_invalid_state("range_offset_px must be scalar or match samples shape")

    source_tensor: object | None = None
    resolved_device: object | None = None
    chunk_size = 0
    if order is None:
        from faninsar.processing.resampling_torch import (
            DEFAULT_LANCZOS_CHUNK,
            _cleanup_device,
            _lanczos_resample_device_persistent,
            _resolve_torch_device,
        )

        chunk_size = int(DEFAULT_LANCZOS_CHUNK)
        resolved_device = _resolve_torch_device(device)
        import torch

        if not isinstance(resolved_device, torch.device):
            reject_invalid_state("resolved torch device has an invalid type")
        source_tensor = torch.from_numpy(np.ascontiguousarray(samples)).to(
            resolved_device,
            non_blocking=True,
        )
        if resolved_device.type == "cuda":
            torch.cuda.synchronize()

    out = np.empty((height, width), dtype=samples.dtype)
    col_idx = np.arange(width, dtype=np.float64)

    try:
        for row0 in range(0, height, row_chunk):
            row1 = min(row0 + row_chunk, height)
            n_rows = row1 - row0
            row_idx = np.arange(row0, row1, dtype=np.float64)[:, None]
            cols = np.broadcast_to(col_idx[None, :], (n_rows, width))
            rows = np.broadcast_to(row_idx, (n_rows, width))

            az_tile = az_off if scalar_az else az_off[row0:row1]
            rg_tile = rg_off if scalar_rg else rg_off[row0:row1]
            src_row = rows - az_tile
            src_col = cols - rg_tile

            if order is None:
                coords = np.vstack([src_row.ravel(), src_col.ravel()])
                tile = _lanczos_resample_device_persistent(
                    samples,
                    coords[0],
                    coords[1],
                    a=lanczos_a,
                    mode="constant",
                    cval=0.0,
                    device=resolved_device,
                    chunk_size=chunk_size,
                    source_tensor=source_tensor,
                )
                out[row0:row1] = tile.reshape(n_rows, width)
            else:
                real = map_coordinates(
                    samples.real,
                    [src_row, src_col],
                    order=order,
                    mode="constant",
                    cval=0.0,
                )
                imag = map_coordinates(
                    samples.imag,
                    [src_row, src_col],
                    order=order,
                    mode="constant",
                    cval=0.0,
                )
                out[row0:row1] = (real + 1j * imag).astype(samples.dtype, copy=False)
    finally:
        if source_tensor is not None:
            del source_tensor
            _cleanup_device(resolved_device)

    return out


def resample_complex_deramped_reramp(
    sec_deramped: np.ndarray,
    *,
    secondary_carrier: TOPSCarrierModel,
    range_offset_px: np.ndarray | float,
    azimuth_offset_px: np.ndarray | float,
    lanczos_a: int = 4,
    row_chunk: int = 64,
    executor: Literal["torch"] = "torch",
    device: str = "auto",
    output_carrier: TOPSCarrierModel | None = None,
    row0: int = 0,
    col0: int = 0,
    native_height: int | None = None,
) -> np.ndarray:
    """Resample a deramped secondary onto the reference grid, then analytical reramp.

    1. Resample the **deramped** secondary (carrier removed, signal stationary)
       with the existing phase-preserving Lanczos kernel — no per-tap carrier
       work, the kernel sees a band-limited stationary signal.
    2. Apply the secondary carrier back at the **source** fractional
       coordinates ``output_index - offset`` via
       :func:`faninsar.processing.torch_kernels.carrier_phase_at_points_torch`
       (analytical polynomial), **not** by
       interpolating an integer-grid carrier plane. The latter is what
       collapsed to 65 rad in §6 because ``map_coordinates(order=1)``
       bilinearly interpolates the ~0.17 rad/pixel azimuth carrier.

    The output lives in the original focused-SLC phase domain, ready for
    interferogram formation against the reramped reference.

    Parameters
    ----------
    sec_deramped : numpy.ndarray
        Secondary complex samples with the TOPS carrier already removed
        (``deramp(sec, secondary_carrier)``).
    secondary_carrier : TOPSCarrierModel
        Carrier model of the secondary burst (native grid geometry).
    range_offset_px, azimuth_offset_px : array or float
        Offsets of the secondary relative to the reference grid
        (``source = output_index - offset``), same convention as
        :func:`resample_complex`.
    lanczos_a : int, optional
        Lanczos half-width. Default 4.
    row_chunk : int, optional
        Azimuth rows processed per tile. Default 64.
    executor : {"torch"}, optional
        Complex interpolation executor.
    device : {"auto", "cpu", "cuda"}, optional
        Torch compute device.
    output_carrier : TOPSCarrierModel, optional
        Carrier on the output reference grid. When omitted, the secondary
        carrier is evaluated at source coordinates.
    row0, col0 : int, optional
        Offset of the window inside the native burst for carrier coordinates.
    native_height : int, optional
        Native burst height used for the carrier centre row.

    Returns
    -------
    numpy.ndarray
        Complex secondary on the reference grid in the original phase domain.

    """
    if sec_deramped.ndim != 2 or not np.iscomplexobj(sec_deramped):
        reject_invalid_state("deramped resample requires a 2-D complex array")
    if row_chunk < 1:
        reject_invalid_state("row_chunk must be >= 1")
    height, width = sec_deramped.shape
    az_off = np.asarray(azimuth_offset_px, dtype=np.float64)
    rg_off = np.asarray(range_offset_px, dtype=np.float64)
    scalar_az = az_off.ndim == 0
    scalar_rg = rg_off.ndim == 0
    if not scalar_az and az_off.shape != (height, width):
        reject_invalid_state("azimuth_offset_px must be scalar or match samples shape")
    if not scalar_rg and rg_off.shape != (height, width):
        reject_invalid_state("range_offset_px must be scalar or match samples shape")

    centre_row = float(height // 2 if native_height is None else native_height // 2)
    out = np.empty((height, width), dtype=sec_deramped.dtype)
    col_idx = np.arange(width, dtype=np.float64)
    remapped_deramped = resample_complex(
        sec_deramped,
        range_offset_px=range_offset_px,
        azimuth_offset_px=azimuth_offset_px,
        lanczos_a=lanczos_a,
        row_chunk=row_chunk,
        executor=executor,
        device=device,
    )

    for row_start in range(0, height, row_chunk):
        row_stop = min(row_start + row_chunk, height)
        n_rows = row_stop - row_start
        row_idx = np.arange(row_start, row_stop, dtype=np.float64)[:, None]
        cols = np.broadcast_to(col_idx[None, :], (n_rows, width))
        rows = np.broadcast_to(row_idx, (n_rows, width))
        az_tile = az_off if scalar_az else az_off[row_start:row_stop]
        rg_tile = rg_off if scalar_rg else rg_off[row_start:row_stop]
        src_row = rows - az_tile
        src_col = cols - rg_tile
        tile = remapped_deramped[row_start:row_stop]
        carrier_model = secondary_carrier if output_carrier is None else output_carrier
        carrier_row = (src_row if output_carrier is None else rows) + float(row0)
        carrier_col = (src_col if output_carrier is None else cols) + float(col0)
        from faninsar.processing.torch_kernels import (
            carrier_multiply_torch,
            carrier_phase_at_points_torch,
        )

        phi_src = carrier_phase_at_points_torch(
            carrier_model,
            carrier_row,
            carrier_col,
            centre_row=centre_row,
            dtype=np.float64,
            device=device,
        )
        out[row_start:row_stop] = carrier_multiply_torch(
            tile,
            phi_src,
            sign=1.0,
            device=device,
        )

    return out
