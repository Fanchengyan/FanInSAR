"""Range split-spectrum bandpass filtering on radar-grid SLC arrays.

Torch port of the ISCE3 ``isce3.splitspectrum`` semantics: a blockwise
range FFT, a Tukey/Kaiser/Cosine-family passband window over the requested
subband edges with optional transmitted-window deconvolution, then inverse
FFT.  Pure device ops; no host round-trips and no scipy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

__all__ = ["SubBandResult", "split_range_spectrum"]

_WINDOW_FUNCTIONS = ("tukey", "kaiser", "cosine")


@dataclass(frozen=True)
class SubBandResult:
    """One filtered subband plus the metadata needed downstream.

    Attributes
    ----------
    data : torch.Tensor
        Filtered complex SLC, same shape as the input.
    center_frequency : float
        Absolute center frequency of this subband [Hz].
    bandwidth : float
        Passband width of this subband [Hz].

    """

    data: torch.Tensor
    center_frequency: float
    bandwidth: float


def _tukey_ramp(x_clipped: torch.Tensor, alpha: float) -> torch.Tensor:
    """Tukey up-ramp evaluated on x in [0, alpha/2] (value 0 -> 1)."""
    return 0.5 * (1.0 + torch.cos(math.pi / alpha * (2.0 * x_clipped - alpha)))


def _bandpass_window(
    offsets_hz: torch.Tensor,
    *,
    edge_low: float,
    edge_high: float,
    window_function: str,
    window_shape: float,
) -> torch.Tensor:
    """Return complex passband weights over absolute frequency offsets."""
    if window_function not in _WINDOW_FUNCTIONS:
        message = f"unsupported window_function {window_function!r}"
        logger.error(message)
        raise ValueError(message)
    width = edge_high - edge_low
    if width <= 0:
        message = "subband requires high_edge_hz > low_edge_hz"
        logger.error(message)
        raise ValueError(message)

    inside = (offsets_hz >= edge_low) & (offsets_hz <= edge_high)
    x = ((offsets_hz - edge_low) / width).clamp(0.0, 1.0)

    if window_function == "cosine":
        shape = torch.sin(math.pi * x)
    elif window_function == "kaiser":
        beta = max(window_shape, 0.0)
        centered = torch.abs(2.0 * x - 1.0)
        shape = torch.cos(0.5 * math.pi * centered) ** beta
    else:  # tukey
        alpha = min(max(window_shape, 0.0), 1.0)
        core = (x >= alpha / 2) & (x <= 1.0 - alpha / 2)
        ramp_up = _tukey_ramp(x.clamp(0.0, alpha / 2), alpha)
        ramp_down = _tukey_ramp((1.0 - x).clamp(0.0, alpha / 2), alpha)
        shape = torch.where(
            core,
            torch.ones_like(x),
            torch.where(x < alpha / 2, ramp_up, ramp_down),
        )

    shape = shape.clamp(min=0.0)
    return torch.where(inside, shape, torch.zeros_like(shape))


def split_range_spectrum(
    slc: torch.Tensor,
    *,
    range_sampling_rate_hz: float,
    low_offset_hz: float,
    high_offset_hz: float,
    window_function: str = "tukey",
    window_shape: float = 0.25,
) -> SubBandResult:
    """Filter one range-subband out of a radar-grid SLC.

    Parameters
    ----------
    slc : torch.Tensor
        Complex baseband SLC with range along the last axis.
    range_sampling_rate_hz : float
        Range sampling rate of the input [Hz].
    low_offset_hz, high_offset_hz : float
        Passband edges as offsets from the carrier [Hz], e.g.
        ``low=-B/3, high=+B/3`` for a thirds split centered on the carrier.
    window_function : {"tukey", "kaiser", "cosine"}, optional
        Spectral window family, mirroring ISCE3 ``split_range_spectrum``.
    window_shape : float, optional
        Shape parameter of the chosen family (Tukey plateau fraction).

    Returns
    -------
    SubBandResult
        Filtered subband; ``center_frequency`` reports the subband-center
        offset from the carrier [Hz].

    Raises
    ------
    ValueError
        On non-positive sampling rate, invalid band geometry, or an
        unsupported window family.

    """
    if range_sampling_rate_hz <= 0:
        message = (
            f"range_sampling_rate_hz must be positive, got {range_sampling_rate_hz!r}"
        )
        logger.error(message)
        raise ValueError(message)
    if not torch.is_complex(slc):
        message = "slc must be a complex tensor"
        logger.error(message)
        raise ValueError(message)

    cols = int(slc.shape[-1])
    fft_len = 1 << (cols - 1).bit_length()  # zero-pad to power of two

    spectrum = torch.fft.fft(slc, n=fft_len, dim=-1)
    offsets = torch.fft.fftfreq(fft_len, d=1.0 / range_sampling_rate_hz)
    offsets = offsets.to(slc.device)

    target = _bandpass_window(
        offsets,
        edge_low=float(low_offset_hz),
        edge_high=float(high_offset_hz),
        window_function=window_function,
        window_shape=window_shape,
    )
    filtered = spectrum * target

    out = torch.fft.ifft(filtered)[..., :cols].to(dtype=slc.dtype)

    center_offset = 0.5 * (float(low_offset_hz) + float(high_offset_hz))
    return SubBandResult(
        data=out,
        center_frequency=center_offset,
        bandwidth=float(high_offset_hz) - float(low_offset_hz),
    )
