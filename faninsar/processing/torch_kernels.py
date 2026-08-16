"""PyTorch GPU kernels for the production Stack pipeline.

Every kernel follows the dask-torch-numpy contract: NumPy arrays in, NumPy
arrays out, with PyTorch as an internal implementation detail.  The kernels
mirror the reference NumPy implementations in this package (deramp, multilook
interferogram formation, Goldstein filtering, topographic flattening, and
ESD) so accelerator results stay numerically comparable to the CPU truth.

Device policy matches :mod:`faninsar._core.device`: ``"auto"`` resolves from
hardware visible to the current process, preferring CUDA and otherwise using
CPU. Distributed callers pass ``"cuda"`` explicitly after selecting a trusted
GPU worker. On CPU the kernels compute in float64/complex128; accelerator
precision is selected per kernel from qualified parity evidence.
"""

from __future__ import annotations

import math
from functools import wraps
from typing import TYPE_CHECKING, Literal, ParamSpec, TypeVar

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from faninsar.processing.tops.deramp import TOPSCarrierModel

logger = setup_logger(__name__)

__all__ = [
    "CUDA_DTYPE_CHOICE",
    "CUDA_FLOAT32_QUALIFIED",
    "carrier_multiply_torch",
    "carrier_phase_at_points_torch",
    "cleanup_device",
    "esd_azimuth_shift_torch",
    "goldstein_filter_torch",
    "multilook_interferogram_torch",
    "multilook_real_torch",
    "remove_topographic_phase_torch",
    "resolve_torch_device",
    "tops_carrier_multiply_torch",
]

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
PrecisionName = Literal["auto", "float32", "float64"]
_P = ParamSpec("_P")
_R = TypeVar("_R")

CUDA_DTYPE_CHOICE: dict[str, Literal["float32", "float64"]] = {
    "carrier_phase": "float64",
    "carrier_multiply": "float64",
    "multilook_interferogram": "float64",
    "goldstein_filter": "float32",
    "remove_topographic_phase": "float64",
    "multilook_real": "float64",
    "esd_azimuth_shift": "float32",
}

# Enabling float32 is a qualification-time code change backed by recorded CUDA
# evidence. A mapping entry alone cannot enable an unqualified float32 path.
CUDA_FLOAT32_QUALIFIED: frozenset[str] = frozenset(
    {"goldstein_filter", "esd_azimuth_shift"}
)


def _import_torch() -> torch:
    """Import torch lazily with a descriptive error for optional installs."""
    try:
        import torch
    except ImportError as error:
        message = (
            "torch is required for accelerated kernels; install FanInSAR dependencies"
        )
        logger.exception(message)
        raise ImportError(message) from error
    return torch


def resolve_torch_device(
    device: DeviceName | torch.device | None = "auto",
) -> torch.device:
    """Resolve a Torch execution device following the Stack device policy.

    Thin wrapper around :func:`faninsar._core.device.parse_device`.
    ``"auto"`` inspects only hardware visible to the current process.
    CUDA is preferred among published devices, unpublished backends are
    never auto-selected, and an explicit request stays on that device.
    A missing backend or out-of-range ``cuda:N`` fails closed.

    Parameters
    ----------
    device : {"auto", "cpu", "cuda", "mps"} or torch.device, optional
        Requested execution device. Any value ``torch.device`` can
        construct is also admitted.

    Returns
    -------
    torch.device
        Admitted device identity.

    Raises
    ------
    RuntimeError
        If an explicit device is requested but unavailable or unusable.

    """
    from faninsar._core.device import parse_device

    return parse_device(device)


def cleanup_device(device: torch.device) -> None:
    """Release allocator caches after accelerator kernel execution.

    Required between Dask tasks so a persistent worker never accumulates GPU
    memory across scheduled chunks.

    Parameters
    ----------
    device : torch.device
        Device whose allocator cache is released.

    """
    torch = _import_torch()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _cleanup_after_kernel(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Guarantee accelerator cache cleanup on success and exception paths."""

    @wraps(func)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        device = kwargs.get("device", "auto")
        resolved = resolve_torch_device(device)
        try:
            return func(*args, **kwargs)
        finally:
            cleanup_device(resolved)

    return wrapped


def _precision_name(
    device: torch.device,
    kernel: str,
    precision: PrecisionName,
) -> Literal["float32", "float64"]:
    """Resolve a measured per-kernel precision choice."""
    if device.type == "cpu":
        return "float64"
    if precision == "float32" and kernel not in CUDA_FLOAT32_QUALIFIED:
        logger.warning(
            "%s requested unqualified float32 on %s; using float64 until "
            "recorded CUDA parity evidence qualifies this kernel",
            kernel,
            device.type,
        )
        return "float64"
    if precision != "auto":
        return precision
    selected = CUDA_DTYPE_CHOICE[kernel]
    if selected == "float32" and kernel not in CUDA_FLOAT32_QUALIFIED:
        logger.warning(
            "%s requested unqualified float32 on %s; using float64 until "
            "recorded CUDA parity evidence qualifies this kernel",
            kernel,
            device.type,
        )
        return "float64"
    if selected == "float64":
        logger.warning(
            "%s uses float64 on %s under the current qualification policy",
            kernel,
            device.type,
        )
    return selected


def _complex_dtype(
    device: torch.device,
    kernel: str,
    precision: PrecisionName = "auto",
) -> torch.dtype:
    """Return the qualified working complex dtype for a kernel."""
    torch = _import_torch()
    selected = _precision_name(device, kernel, precision)
    return torch.complex128 if selected == "float64" else torch.complex64


def _float_dtype(
    device: torch.device,
    kernel: str,
    precision: PrecisionName = "auto",
) -> torch.dtype:
    """Return the qualified working real dtype for a kernel."""
    torch = _import_torch()
    selected = _precision_name(device, kernel, precision)
    return torch.float64 if selected == "float64" else torch.float32


@_cleanup_after_kernel
def carrier_phase_at_points_torch(
    model: TOPSCarrierModel,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    centre_row: float,
    dtype: np.dtype | type = np.float32,
    device: DeviceName = "auto",
    precision: PrecisionName = "auto",
) -> np.ndarray:
    """Evaluate the analytical TOPS carrier phase with a Torch backend.

    Numerically mirrors :func:`faninsar.processing.tops.deramp.carrier_phase_at_points`
    but accepts broadcastable row/column arrays so tiled kernels never
    materialise a full meshgrid.

    Parameters
    ----------
    model : TOPSCarrierModel
        Burst carrier model.
    rows, cols : numpy.ndarray
        Fractional pixel coordinates; must broadcast to a common shape.
    centre_row : float
        Azimuth centre row of the burst window.
    dtype : numpy.dtype, optional
        Output phase dtype. Default ``float32``.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch execution device.
    precision : {"auto", "float32", "float64"}, optional
        Per-call precision override; ``"auto"`` uses the qualified mapping.

    Returns
    -------
    numpy.ndarray
        Carrier phase in radians of the broadcast shape.

    """
    rows64 = np.array(rows, dtype=np.float64, copy=True)
    cols64 = np.array(cols, dtype=np.float64, copy=True)
    try:
        np.broadcast_shapes(rows64.shape, cols64.shape)
    except ValueError as error:
        message = (
            f"rows shape {rows64.shape} and cols shape {cols64.shape} must broadcast"
        )
        logger.exception(message)
        raise ValueError(message) from error
    torch = _import_torch()
    resolved = resolve_torch_device(device)
    float_dtype = _float_dtype(resolved, "carrier_phase", precision)
    rows_t = torch.as_tensor(rows64, dtype=torch.float64, device=resolved)
    cols_t = torch.as_tensor(cols64, dtype=torch.float64, device=resolved)
    tau = model.slant_range_time0_s + cols_t / model.range_sampling_rate_hz
    f_dc = _poly_eval_torch(model.doppler_centroid_hz, tau - model.doppler_t0_s)
    k_a = _poly_eval_torch(model.fm_rate_hz_s, tau - model.fm_t0_s)
    tau_start = model.burst_start_slant_range_time_s
    f_dc_start = _poly_eval_torch(
        model.doppler_centroid_hz,
        torch.as_tensor(
            np.asarray(tau_start - model.doppler_t0_s),
            dtype=torch.float64,
            device=resolved,
        ),
    )
    k_a_start = _poly_eval_torch(
        model.fm_rate_hz_s,
        torch.as_tensor(
            np.asarray(tau_start - model.fm_t0_s),
            dtype=torch.float64,
            device=resolved,
        ),
    )
    eta_ref = (f_dc_start / k_a_start) - (f_dc / k_a)
    k_s = model.azimuth_steering_rate_hz_s
    k_t = k_s / (1.0 - k_s / k_a)
    eta = (rows_t - float(centre_row)) * model.azimuth_time_interval_s
    steering_phase = math.pi * k_t * (eta - eta_ref) ** 2
    doppler_phase = 2.0 * math.pi * f_dc * eta
    phase = steering_phase + doppler_phase
    output = phase.to(dtype=float_dtype)
    return np.asarray(output.cpu().numpy(), dtype=dtype)


def _poly_eval_torch(
    coefficients: tuple[float, ...],
    x: torch.Tensor,
) -> torch.Tensor:
    """Evaluate polynomial ``c0 + c1*x + c2*x^2 + ...`` in Torch."""
    torch = _import_torch()
    result = torch.zeros_like(x)
    power = torch.ones_like(x)
    for coeff in coefficients:
        result = result + float(coeff) * power
        power = power * x
    return result


@_cleanup_after_kernel
def carrier_multiply_torch(
    samples: np.ndarray,
    phase: np.ndarray,
    *,
    sign: float,
    device: DeviceName = "auto",
    precision: PrecisionName = "auto",
) -> np.ndarray:
    """Multiply complex samples by ``exp(sign * 1j * phase)`` on a device.

    Mirrors :func:`faninsar.processing.tops.deramp._apply_carrier_phase`:
    cos/sin are evaluated in float64 for large carrier phases and multiplied
    in float32, avoiding complex128 temporary planes on full bursts.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex 2-D samples.
    phase : numpy.ndarray
        Carrier phase in radians, broadcastable to ``samples``.
    sign : float
        ``-1`` for deramp, ``+1`` for reramp.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch execution device.
    precision : {"auto", "float32", "float64"}, optional
        Per-call precision override; ``"auto"`` uses the qualified mapping.

    Returns
    -------
    numpy.ndarray
        Phase-rotated complex array with the input dtype.

    """
    if samples.ndim != 2 or not np.iscomplexobj(samples):
        reject_invalid_state("carrier multiply requires a 2-D complex array")
    torch = _import_torch()
    resolved = resolve_torch_device(device)
    float_dtype = _float_dtype(resolved, "carrier_multiply", precision)
    complex_dtype = _complex_dtype(resolved, "carrier_multiply", precision)
    phase64 = torch.as_tensor(np.asarray(phase, dtype=np.float64), device=resolved)
    cos_p = torch.cos(phase64).to(dtype=float_dtype)
    sin_p = torch.sin(phase64).to(dtype=float_dtype)
    numpy_float_dtype = np.float64 if float_dtype == torch.float64 else np.float32
    re = torch.as_tensor(
        np.ascontiguousarray(samples.real.astype(numpy_float_dtype, copy=False)),
        dtype=float_dtype,
        device=resolved,
    )
    im = torch.as_tensor(
        np.ascontiguousarray(samples.imag.astype(numpy_float_dtype, copy=False)),
        dtype=float_dtype,
        device=resolved,
    )
    s = float(sign)
    out_re = re * cos_p - s * im * sin_p
    out_im = im * cos_p + s * re * sin_p
    out = torch.complex(out_re, out_im).to(dtype=complex_dtype)
    result = np.asarray(out.cpu().numpy())
    return result.astype(samples.dtype, copy=False)


@_cleanup_after_kernel
def tops_carrier_multiply_torch(
    samples: np.ndarray,
    model: TOPSCarrierModel,
    *,
    sign: float,
    row0: int = 0,
    col0: int = 0,
    native_height: int | None = None,
    row_chunk: int | None = 256,
    device: DeviceName = "auto",
    precision: PrecisionName = "auto",
) -> np.ndarray:
    """Deramp (``sign=-1``) or reramp (``sign=+1``) a burst window in Torch.

    Mirrors :func:`faninsar.processing.tops.deramp._apply_carrier_tiled`:
    the analytical carrier is evaluated in azimuth tiles so phase and
    cos/sin temporaries never cover the full burst.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex burst window.
    model : TOPSCarrierModel
        Carrier model matching the window geometry.
    sign : float
        ``-1`` for deramp, ``+1`` for reramp.
    row0, col0 : int, optional
        Offset of the window inside the native burst.
    native_height : int, optional
        Native burst height used for the carrier centre row.
    row_chunk : int or None, optional
        Azimuth tile height. Default 256.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch execution device.
    precision : {"auto", "float32", "float64"}, optional
        Per-call precision override; ``"auto"`` uses the qualified mapping.

    Returns
    -------
    numpy.ndarray
        Carrier-adjusted complex array.

    """
    if samples.ndim != 2 or not np.iscomplexobj(samples):
        reject_invalid_state("TOPS carrier multiply requires a 2-D complex array")
    n_lines, n_samples = samples.shape
    centre_row = float(n_lines // 2 if native_height is None else native_height // 2)
    if row_chunk is None or row_chunk <= 0 or n_lines <= row_chunk:
        rows = np.arange(n_lines, dtype=np.float64)[:, None] + float(row0)
        cols = np.arange(n_samples, dtype=np.float64)[None, :] + float(col0)
        phase = carrier_phase_at_points_torch(
            model,
            rows,
            cols,
            centre_row=centre_row,
            dtype=np.float64,
            device=device,
            precision=precision,
        )
        return carrier_multiply_torch(
            samples,
            phase,
            sign=sign,
            device=device,
            precision=precision,
        )

    out = np.empty_like(samples)
    for row_start in range(0, n_lines, row_chunk):
        row_stop = min(row_start + row_chunk, n_lines)
        rows = np.arange(row_start, row_stop, dtype=np.float64)[:, None] + float(row0)
        cols = np.arange(n_samples, dtype=np.float64)[None, :] + float(col0)
        phase = carrier_phase_at_points_torch(
            model,
            rows,
            cols,
            centre_row=centre_row,
            dtype=np.float64,
            device=device,
            precision=precision,
        )
        out[row_start:row_stop] = carrier_multiply_torch(
            samples[row_start:row_stop],
            phase,
            sign=sign,
            device=device,
            precision=precision,
        )
    return out


@_cleanup_after_kernel
def multilook_interferogram_torch(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    multilook: tuple[int, int] = (1, 1),
    dead_pixel_amp_threshold: float = 0.0,
    device: DeviceName = "auto",
    precision: PrecisionName = "auto",
) -> object:
    """Form a multilooked complex interferogram and coherence in Torch.

    Numerically mirrors
    :func:`faninsar.processing.interferometry.pair.form_interferogram`
    including the valid-pixel-weighted dead-pixel multilook and the invalid
    look masking convention (NaN where either input look has zero power).

    Parameters
    ----------
    primary, secondary : numpy.ndarray
        Coregistered complex arrays on the same grid.
    multilook : tuple[int, int], optional
        ``(azimuth_looks, range_looks)`` non-overlapping boxcar looks.
    dead_pixel_amp_threshold : float, optional
        SLC amplitude below which a pixel is excluded from the look average.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch execution device.
    precision : {"auto", "float32", "float64"}, optional
        Per-call precision override; ``"auto"`` uses the qualified mapping.

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
    torch = _import_torch()
    resolved = resolve_torch_device(device)
    complex_dtype = _complex_dtype(
        resolved,
        "multilook_interferogram",
        precision,
    )
    float_dtype = _float_dtype(resolved, "multilook_interferogram", precision)
    p = torch.as_tensor(
        np.ascontiguousarray(primary),
        dtype=complex_dtype,
        device=resolved,
    )
    s = torch.as_tensor(
        np.ascontiguousarray(secondary),
        dtype=complex_dtype,
        device=resolved,
    )
    use_dead_mask = dead_pixel_amp_threshold > 0.0 and (az_looks > 1 or rg_looks > 1)

    if az_looks == 1 and rg_looks == 1:
        ifg = p * s.conj()
        power_pri = (p.real**2 + p.imag**2).to(dtype=float_dtype)
        power_sec = (s.real**2 + s.imag**2).to(dtype=float_dtype)
    else:
        height, width = p.shape
        h = (height // az_looks) * az_looks
        w = (width // rg_looks) * rg_looks
        out_h = h // az_looks
        out_w = w // rg_looks
        pr = p[:h, :w].real.reshape(out_h, az_looks, out_w, rg_looks)
        pi = p[:h, :w].imag.reshape(out_h, az_looks, out_w, rg_looks)
        sr = s[:h, :w].real.reshape(out_h, az_looks, out_w, rg_looks)
        si = s[:h, :w].imag.reshape(out_h, az_looks, out_w, rg_looks)
        ir = pr * sr + pi * si
        ii = pi * sr - pr * si
        pp = pr * pr + pi * pi
        ss = sr * sr + si * si
        if use_dead_mask:
            amp_p = torch.sqrt(torch.clamp(pp, min=0.0))
            amp_s = torch.sqrt(torch.clamp(ss, min=0.0))
            valid = (amp_p >= dead_pixel_amp_threshold) & (
                amp_s >= dead_pixel_amp_threshold
            )
            wgt = valid.to(dtype=float_dtype)
            w_sum = wgt.sum(dim=(1, 3))
            safe_w = torch.where(
                w_sum > 0,
                w_sum,
                torch.ones_like(w_sum),
            )
            ifg_real = (ir * wgt).sum(dim=(1, 3)) / safe_w
            ifg_imag = (ii * wgt).sum(dim=(1, 3)) / safe_w
            power_pri = (pp * wgt).sum(dim=(1, 3)) / safe_w
            power_sec = (ss * wgt).sum(dim=(1, 3)) / safe_w
        else:
            inv_looks = 1.0 / float(az_looks * rg_looks)
            ifg_real = ir.sum(dim=(1, 3)) * inv_looks
            ifg_imag = ii.sum(dim=(1, 3)) * inv_looks
            power_pri = pp.sum(dim=(1, 3)) * inv_looks
            power_sec = ss.sum(dim=(1, 3)) * inv_looks
        ifg = torch.complex(ifg_real, ifg_imag)
        logger.info(
            "Torch multilook %s -> output shape %s",
            multilook,
            tuple(ifg.shape),
        )

    denom = torch.sqrt(torch.clamp(power_pri * power_sec, min=1e-30))
    output_complex_dtype = np.result_type(primary.dtype, secondary.dtype)
    output_float_dtype = (
        np.float64 if output_complex_dtype == np.dtype(np.complex128) else np.float32
    )
    torch_output_float_dtype = (
        torch.float64 if output_float_dtype == np.float64 else torch.float32
    )
    torch_output_complex_dtype = (
        torch.complex128
        if output_complex_dtype == np.dtype(np.complex128)
        else torch.complex64
    )
    coherence = torch.clamp(ifg.abs() / denom, 0.0, 1.0).to(
        dtype=torch_output_float_dtype
    )
    amplitude = ifg.abs().to(dtype=torch_output_float_dtype)
    invalid = (power_pri <= 0.0) | (power_sec <= 0.0) | ~torch.isfinite(power_pri)
    nan_complex = torch.tensor(
        complex(float("nan"), float("nan")),
        dtype=complex_dtype,
        device=resolved,
    )
    ifg_out = torch.where(invalid, nan_complex, ifg).to(
        dtype=torch_output_complex_dtype
    )
    nan_float = torch.tensor(
        float("nan"),
        dtype=torch_output_float_dtype,
        device=resolved,
    )
    coherence_out = torch.where(invalid, nan_float, coherence)
    amplitude_out = torch.where(invalid, nan_float, amplitude)
    wrapped = ifg_out.angle()
    wrapped = torch.where(
        torch.isfinite(ifg_out.real) & torch.isfinite(ifg_out.imag),
        wrapped,
        nan_float,
    )
    complex_ifg = np.asarray(ifg_out.cpu().numpy(), dtype=output_complex_dtype)
    coherence_np = np.asarray(coherence_out.cpu().numpy(), dtype=output_float_dtype)
    wrapped_np = np.asarray(wrapped.cpu().numpy(), dtype=output_float_dtype)
    amplitude_np = np.asarray(amplitude_out.cpu().numpy(), dtype=output_float_dtype)
    from faninsar.processing.interferometry.pair import InterferogramProduct

    return InterferogramProduct(
        complex_ifg=complex_ifg,
        coherence=coherence_np,
        wrapped_phase=wrapped_np,
        amplitude=amplitude_np,
    )


@_cleanup_after_kernel
def goldstein_filter_torch(
    complex_ifg: np.ndarray,
    *,
    alpha: float = 0.5,
    window: int = 32,
    device: DeviceName = "auto",
    output_row0: int = 0,
    output_rows: int | None = None,
    input_row0: int = 0,
    precision: PrecisionName = "auto",
) -> np.ndarray:
    """Apply the Goldstein-Werner adaptive spectral filter in Torch.

    Batches all column patches of one azimuth band into a single FFT so a
    full burst is filtered with ``H / step`` batched transforms instead of
    ``H * W / step**2`` scalar NumPy loops.  The weighted overlap-add and
    invalid-patch zeroing semantics match
    :func:`faninsar.processing.interferometry.pair.goldstein_filter`
    (ISCE2 ``psfilt`` parity).

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram.  May be a row slice when ``output_row0`` /
        ``output_rows`` are supplied for Dask chunked scheduling.
    alpha : float, optional
        Filter exponent in ``[0, 1]``.
    window : int, optional
        Square FFT patch size. Default 32.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch execution device.
    output_row0 : int, optional
        Global row of the first output row (for chunked scheduling).
    output_rows : int or None, optional
        Number of output rows; defaults to the full input height.
    input_row0 : int, optional
        Global row of the first input row of ``complex_ifg``.
    precision : {"auto", "float32", "float64"}, optional
        Per-call precision override; ``"auto"`` uses the qualified mapping.

    Returns
    -------
    numpy.ndarray
        Filtered complex interferogram with the original magnitude restored.

    """
    if complex_ifg.ndim != 2 or not np.iscomplexobj(complex_ifg):
        reject_invalid_state("Goldstein filter requires a 2-D complex array")
    if not 0.0 <= alpha <= 1.0:
        reject_invalid_state("Goldstein alpha must be in [0, 1]")
    if window < 8 or window % 2 != 0:
        reject_invalid_state("Goldstein window must be an even integer >= 8")
    height, width = complex_ifg.shape
    if height < window or width < window:
        return complex_ifg.copy()
    if output_rows is None:
        output_rows = height - output_row0
    if output_rows <= 0:
        reject_invalid_state("Goldstein output rows must be positive")

    torch = _import_torch()
    resolved = resolve_torch_device(device)
    complex_dtype = _complex_dtype(resolved, "goldstein_filter", precision)
    x = torch.as_tensor(
        np.ascontiguousarray(complex_ifg),
        dtype=complex_dtype,
        device=resolved,
    )
    step = window // 2
    half = window / 2.0
    axis = torch.arange(window, dtype=torch.float64, device=resolved)
    triangular = 1.0 - torch.abs(2.0 * (axis - half) / (window + 1))
    window2d = torch.outer(triangular, triangular) / (window * window)

    out = torch.zeros(
        (output_rows, width),
        dtype=complex_dtype,
        device=resolved,
    )
    p_min = max(0, math.ceil((output_row0 - (window - 1)) / step) * step)
    p_max = math.floor((output_row0 + output_rows - 1) / step) * step
    col_origins = list(range(0, width, step))
    n_cols = len(col_origins)
    for patch_row in range(p_min, p_max + 1, step):
        row_overlap = max(patch_row, input_row0)
        row_stop = min(patch_row + window, input_row0 + height)
        patch_batch = torch.zeros(
            (n_cols, window, window),
            dtype=complex_dtype,
            device=resolved,
        )
        for col_index, col in enumerate(col_origins):
            c1 = min(col + window, width)
            if row_stop > row_overlap:
                patch_batch[
                    col_index,
                    : row_stop - row_overlap,
                    : c1 - col,
                ] = x[
                    row_overlap - input_row0 : row_stop - input_row0,
                    col:c1,
                ]
        spectrum = torch.fft.fft2(patch_batch)
        power = spectrum.real**2 + spectrum.imag**2
        spectrum = spectrum * power ** (alpha / 2.0)
        filtered = torch.fft.ifft2(spectrum) * (window * window)
        weight_block = window2d[None, :, :] * filtered
        valid_batch = patch_batch != 0
        out_row0 = max(patch_row, output_row0) - output_row0
        out_row1 = min(patch_row + window, output_row0 + output_rows) - output_row0
        row_local0 = out_row0 + output_row0 - patch_row
        row_local1 = row_local0 + (out_row1 - out_row0)
        for col_index, col in enumerate(col_origins):
            c1 = min(col + window, width)
            local_slice = out[out_row0:out_row1, col:c1]
            valid_slice = valid_batch[col_index, row_local0:row_local1, : c1 - col]
            weight_slice = weight_block[col_index, row_local0:row_local1, : c1 - col]
            zero = torch.zeros_like(local_slice)
            out[out_row0:out_row1, col:c1] = torch.where(
                valid_slice,
                local_slice + weight_slice,
                zero,
            )

    row_start = output_row0 - input_row0
    input_mag = x[row_start : row_start + output_rows].abs()
    smoothed_mag = out.abs()
    mask = (smoothed_mag > 0) & (input_mag > 0)
    scale = input_mag / torch.clamp(smoothed_mag, min=1e-30)
    out = torch.where(mask, out * scale, out)
    return np.asarray(out.cpu().numpy(), dtype=complex_ifg.dtype)


@_cleanup_after_kernel
def remove_topographic_phase_torch(
    complex_ifg: np.ndarray,
    topo_phase: np.ndarray,
    device: DeviceName = "auto",
    precision: PrecisionName = "auto",
) -> np.ndarray:
    """Remove topographic phase from a complex interferogram in Torch.

    Mirrors :func:`faninsar.processing.interferometry.flatten.remove_topographic_phase`:
    multiplies the interferogram by ``exp(-1j * topo_phase)``.  On CPU the
    computation stays in complex128 (NumPy parity); accelerator devices use
    complex64.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram array.
    topo_phase : numpy.ndarray
        Topographic phase in radians, broadcastable to ``complex_ifg``.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch execution device.
    precision : {"auto", "float32", "float64"}, optional
        Per-call precision override; ``"auto"`` uses the qualified mapping.

    Returns
    -------
    numpy.ndarray
        Flattened complex interferogram.

    """
    ifg = np.asarray(complex_ifg)
    topo = np.asarray(topo_phase)
    if ifg.shape != topo.shape:
        try:
            np.broadcast_to(topo, ifg.shape)
        except ValueError as error:
            message = (
                f"topo_phase shape {topo.shape} cannot broadcast to "
                f"ifg shape {ifg.shape}"
            )
            logger.exception(message)
            raise ValueError(message) from error
    torch = _import_torch()
    resolved = resolve_torch_device(device)
    complex_dtype = _complex_dtype(
        resolved,
        "remove_topographic_phase",
        precision,
    )
    ifg_t = torch.as_tensor(
        np.ascontiguousarray(ifg),
        dtype=complex_dtype,
        device=resolved,
    )
    real_dtype = (
        torch.float64 if complex_dtype == torch.complex128 else torch.float32
    )
    topo_t = torch.as_tensor(
        np.ascontiguousarray(topo),
        dtype=real_dtype,
        device=resolved,
    )
    rotation = torch.exp(
        torch.complex(
            torch.zeros_like(topo_t),
            -topo_t,
        )
    )
    out = ifg_t * rotation
    return np.asarray(out.cpu().numpy(), dtype=ifg.dtype)


@_cleanup_after_kernel
def multilook_real_torch(
    array: np.ndarray,
    az_looks: int,
    rg_looks: int,
    device: DeviceName = "auto",
    precision: PrecisionName = "auto",
) -> np.ndarray:
    """Average non-overlapping blocks of a real field in Torch.

    Mirrors :func:`faninsar.processing.interferometry.pair._block_reduce`.

    Parameters
    ----------
    array : numpy.ndarray
        2-D real array.
    az_looks, rg_looks : int
        Non-overlapping azimuth and range look factors.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch execution device.
    precision : {"auto", "float32", "float64"}, optional
        Per-call precision override; ``"auto"`` uses the qualified mapping.

    Returns
    -------
    numpy.ndarray
        Multilooked real array.

    """
    if array.ndim != 2:
        reject_invalid_state("block reduce expects a two-dimensional array")
    if az_looks < 1 or rg_looks < 1:
        reject_invalid_state("multilook factors must be >= 1")
    torch = _import_torch()
    resolved = resolve_torch_device(device)
    height, width = array.shape[:2]
    h = (height // az_looks) * az_looks
    w = (width // rg_looks) * rg_looks
    cropped = np.ascontiguousarray(array[:h, :w])
    tensor = torch.as_tensor(
        cropped,
        dtype=_float_dtype(resolved, "multilook_real", precision),
        device=resolved,
    )
    reshaped = tensor.reshape(
        h // az_looks,
        az_looks,
        w // rg_looks,
        rg_looks,
    )
    out = reshaped.mean(dim=(1, 3))
    return np.asarray(out.cpu().numpy(), dtype=array.dtype)


@_cleanup_after_kernel
def esd_azimuth_shift_torch(
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    bandwidth_fraction: float = 0.45,
    taper_alpha: float = 0.1,
    min_coherence: float = 0.05,
    device: DeviceName = "auto",
    precision: PrecisionName = "auto",
    range_chunk_size: int = 512,
) -> object:
    """Estimate the ESD residual azimuth shift with a Torch backend.

    Mirrors :func:`faninsar.processing.coreg.esd.estimate_azimuth_shift_esd`:
    azimuth spectral looks are split with a Tukey-tapered bandpass, dual-look
    interferograms are formed, and the amplitude-weighted circular mean of
    the differential phase yields the residual shift in pixels.

    Parameters
    ----------
    reference, secondary : numpy.ndarray
        Coarsely coregistered complex 2-D arrays.
    bandwidth_fraction : float, optional
        Fraction of the full azimuth bandwidth assigned to each look.
    taper_alpha : float, optional
        Tukey taper fraction on the look edges.
    min_coherence : float, optional
        Floor for the reported coherence.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch execution device.
    precision : {"auto", "float32", "float64"}, optional
        Per-call precision override; ``"auto"`` uses the qualified mapping.
    range_chunk_size : int, optional
        Number of range columns processed per internal Torch chunk. Chunking
        bounds accelerator memory while preserving one Dask task per ESD call.

    Returns
    -------
    ESDResult
        Estimated azimuth shift, coherence, and differential phase.

    """
    if reference.shape != secondary.shape or reference.ndim != 2:
        reject_invalid_state("ESD requires matching 2-D arrays")
    if not (np.iscomplexobj(reference) and np.iscomplexobj(secondary)):
        reject_invalid_state("ESD requires complex-valued SLC arrays")
    if not 0.0 < bandwidth_fraction <= 0.5:
        reject_invalid_state("bandwidth_fraction must be in (0, 0.5]")
    if range_chunk_size < 1:
        reject_invalid_state("range_chunk_size must be >= 1")
    torch = _import_torch()
    resolved = resolve_torch_device(device)
    complex_dtype = _complex_dtype(resolved, "esd_azimuth_shift", precision)
    n_az, n_range = reference.shape
    real_dtype = (
        torch.float64 if complex_dtype == torch.complex128 else torch.float32
    )

    def bandpass(look: str) -> torch.Tensor:
        freqs = torch.fft.fftfreq(
            n_az,
            d=1.0,
            device=resolved,
            dtype=real_dtype,
        )
        half_bw = bandwidth_fraction * 0.5
        center = -half_bw if look == "lower" else +half_bw
        dist = torch.abs(freqs - center)
        dist = torch.minimum(dist, 1.0 - dist)
        inside = dist <= half_bw
        filter_arr = torch.zeros(n_az, dtype=real_dtype, device=resolved)
        filter_arr = torch.where(inside, torch.ones_like(filter_arr), filter_arr)
        if taper_alpha > 0.0:
            edge = half_bw * taper_alpha
            transition = (dist - (half_bw - edge)) / (2.0 * edge)
            transition = torch.clamp(transition, 0.0, 1.0)
            taper = 0.5 * (1.0 + torch.cos(math.pi * transition))
            filter_arr = torch.where(inside, taper, torch.zeros_like(filter_arr))
        return filter_arr

    filt_lower = bandpass("lower")
    filt_upper = bandpass("upper")
    weighted = torch.zeros((), dtype=complex_dtype, device=resolved)
    weight_sum_tensor = torch.zeros((), dtype=real_dtype, device=resolved)
    for col_start in range(0, n_range, range_chunk_size):
        col_stop = min(col_start + range_chunk_size, n_range)
        ref_t = torch.as_tensor(
            np.ascontiguousarray(reference[:, col_start:col_stop]),
            dtype=complex_dtype,
            device=resolved,
        )
        sec_t = torch.as_tensor(
            np.ascontiguousarray(secondary[:, col_start:col_stop]),
            dtype=complex_dtype,
            device=resolved,
        )
        f_ref = torch.fft.fft(ref_t, dim=0)
        f_sec = torch.fft.fft(sec_t, dim=0)
        look_ref_lower = torch.fft.ifft(
            f_ref * filt_lower[:, None].to(dtype=complex_dtype), dim=0
        )
        look_ref_upper = torch.fft.ifft(
            f_ref * filt_upper[:, None].to(dtype=complex_dtype), dim=0
        )
        look_sec_lower = torch.fft.ifft(
            f_sec * filt_lower[:, None].to(dtype=complex_dtype), dim=0
        )
        look_sec_upper = torch.fft.ifft(
            f_sec * filt_upper[:, None].to(dtype=complex_dtype), dim=0
        )
        diff_ifg = (
            look_ref_lower * look_sec_lower.conj()
        ) * (look_ref_upper * look_sec_upper.conj()).conj()
        amp = diff_ifg.abs()
        valid = amp > 0
        if bool(valid.any()):
            unit = torch.where(
                valid,
                diff_ifg / torch.clamp(amp, min=1e-30),
                torch.zeros_like(diff_ifg),
            )
            weighted = weighted + (unit * amp).sum()
            weight_sum_tensor = weight_sum_tensor + amp[valid].sum()

    weight_sum = float(weight_sum_tensor.item())
    if weight_sum <= 0.0:
        logger.warning(
            "Torch ESD has no valid differential samples; returning zero shift"
        )
        from faninsar.processing.coreg.esd import ESDResult

        return ESDResult(azimuth_shift_px=0.0, coherence=0.0, phase_rad=0.0)
    phase_mean = float(weighted.angle().item())
    coherence = float(weighted.abs().item() / weight_sum)
    coherence = float(np.clip(coherence, min_coherence, 1.0))
    denom = 2.0 * math.pi * bandwidth_fraction
    az_shift = -phase_mean / denom
    from faninsar.processing.coreg.esd import ESDResult

    logger.info(
        "Torch ESD az_shift=%.4f px coherence=%.3f phase=%.3f rad",
        az_shift,
        coherence,
        phase_mean,
    )
    return ESDResult(
        azimuth_shift_px=az_shift,
        coherence=coherence,
        phase_rad=phase_mean,
    )
