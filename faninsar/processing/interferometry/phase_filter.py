"""Torch-native runtime strategies for filtering wrapped interferograms."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.nn import functional

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class FilterProvenance:
    """Bounded, non-executable description of a phase-filter strategy."""

    name: str
    parameters: dict[str, int | float | bool | str | None]


@dataclass(frozen=True, slots=True)
class PhaseFilterResult:
    """Filtered complex IFG and the boolean mask of valid support."""

    interferogram: torch.Tensor
    valid_mask: torch.Tensor


def _validate_input(
    interferogram: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Validate shared tensor and support-mask preconditions."""
    if interferogram.ndim != 2 or not interferogram.is_complex():
        message = "phase filter input must be a 2-D complex tensor"
        logger.error(message)
        raise ValueError(message)
    finite = torch.isfinite(interferogram.real) & torch.isfinite(interferogram.imag)
    if valid_mask is None:
        return finite
    if valid_mask.shape != interferogram.shape:
        message = "phase filter valid_mask must match interferogram shape"
        logger.error(message)
        raise ValueError(message)
    if valid_mask.device != interferogram.device or valid_mask.dtype != torch.bool:
        message = "phase filter valid_mask must be bool on the input device"
        logger.error(message)
        raise ValueError(message)
    return valid_mask & finite


class PhaseFilter(ABC):
    """Runtime strategy for filtering one wrapped complex interferogram.

    Arrays use ``(azimuth, range)`` order and contain complex samples.  A
    strategy preserves shape, dtype family, and device; its returned boolean
    support is a subset of the supplied support.  Custom strategies are
    trusted runtime objects and are never serialized for later restoration.
    """

    @abstractmethod
    def apply(
        self,
        interferogram: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> PhaseFilterResult:
        """Filter a 2-D ``(azimuth, range)`` tensor without device transfer.

        Parameters
        ----------
        interferogram : torch.Tensor
            Complex wrapped-interferogram samples. Shape is
            ``(azimuth, range)``; amplitudes are dimensionless in the stored
            normalized product convention.
        valid_mask : torch.Tensor, optional
            Boolean input support mask. ``None`` derives support from finite
            real and imaginary samples. Invalid samples are not made valid by
            filtering.

        Returns
        -------
        PhaseFilterResult
            Filtered complex raster with the input shape, dtype family, and
            device, plus a mask that is a subset of the input support.

        Raises
        ------
        ValueError
            If the input is not a two-dimensional complex tensor or the mask
            does not match its shape, dtype, or device.

        Notes
        -----
        This is a runtime strategy object. Custom subclasses are trusted
        Python objects and are not serialized or reconstructed from product
        metadata. Built-ins document their own window units and boundary
        rules.

        """

    def describe(self) -> FilterProvenance:
        """Return inert provenance; custom strategies default to ``custom``."""
        return FilterProvenance("custom", {})


class GoldsteinWerner(PhaseFilter):
    """Adaptive Goldstein-Werner spectral filter implemented with Torch.

    Parameters
    ----------
    alpha : float, default=0.5
        Spectral exponent in ``[0, 1]``.
    patch_size : int, default=32
        Even square FFT patch size in pixels, bounded to ``[8, 1024]``.
        Patches use half-patch stride and zero-pad image tails.

    """

    def __init__(self, alpha: float = 0.5, patch_size: int = 32) -> None:
        """Initialize the spectral exponent and square patch size."""
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or not 0 <= alpha <= 1
        ):
            message = "Goldstein alpha must be finite and in [0, 1]"
            raise ValueError(message)
        if (
            type(patch_size) is not int
            or patch_size < 8
            or patch_size > 1024
            or patch_size % 2
        ):
            message = "Goldstein patch_size must be an even integer in [8, 1024]"
            raise ValueError(message)
        self.alpha = float(alpha)
        self.patch_size = patch_size

    def describe(self) -> FilterProvenance:
        """Return canonical inert filter parameters."""
        return FilterProvenance(
            "goldstein_werner", {"alpha": self.alpha, "patch_size": self.patch_size}
        )

    def apply(
        self,
        interferogram: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> PhaseFilterResult:
        """Apply Goldstein-Werner spectral weighting.

        Invalid input holes are zero-filled inside each zero-padded square
        FFT patch, while the returned mask keeps those holes invalid. Patches
        use half-patch stride and triangular overlap-add; image tails are
        retained through zero padding. The output remains on the input Torch
        device and has the same ``(azimuth, range)`` shape.
        """
        support = _validate_input(interferogram, valid_mask)
        height, width = interferogram.shape
        patch = self.patch_size
        if height < patch or width < patch:
            return PhaseFilterResult(interferogram.clone(), support)
        step = patch // 2
        taper_1d = 1.0 - torch.abs(
            2.0
            * (
                torch.arange(patch, device=interferogram.device, dtype=torch.float32)
                - patch / 2
            )
            / (patch + 1)
        )
        taper = torch.outer(taper_1d, taper_1d)
        source = torch.where(support, interferogram, torch.zeros_like(interferogram))
        output = torch.zeros_like(interferogram)
        for row in range(0, height, step):
            for col in range(0, width, step):
                rows, cols = min(patch, height - row), min(patch, width - col)
                block = torch.zeros(
                    (patch, patch),
                    dtype=interferogram.dtype,
                    device=interferogram.device,
                )
                block[:rows, :cols] = source[row : row + rows, col : col + cols]
                spectrum = torch.fft.fft2(block)
                magnitude = torch.abs(spectrum)
                # Goldstein-Werner uses the local spectral magnitude itself as
                # the adaptive weight.  Do not normalize each patch by its
                # own maximum: overlapping patches can have different maxima,
                # and that normalization changes their relative contribution
                # during overlap-add.  The final per-pixel magnitude restore
                # below supplies the amplitude convention without changing
                # the relative spatial weighting of patches.
                weight = torch.pow(magnitude, self.alpha)
                filtered = torch.fft.ifft2(spectrum * weight) * float(patch * patch)
                output[row : row + rows, col : col + cols] += (
                    filtered[:rows, :cols] * taper[:rows, :cols]
                )
        input_magnitude = torch.abs(interferogram)
        output_magnitude = torch.abs(output)
        scale = torch.where(
            (output_magnitude > 0) & (input_magnitude > 0),
            input_magnitude / output_magnitude.clamp_min(1e-12),
            torch.ones_like(output_magnitude),
        )
        output = output * scale
        finite = torch.isfinite(output.real) & torch.isfinite(output.imag)
        result_mask = support & finite
        return PhaseFilterResult(output, result_mask)


class _SpatialFilter(PhaseFilter):
    """Shared valid-weighted convolution implementation."""

    def _apply_kernel(
        self,
        interferogram: torch.Tensor,
        support: torch.Tensor,
        kernel: torch.Tensor,
    ) -> PhaseFilterResult:
        radius_y, radius_x = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        padded = functional.pad(
            torch.where(support, interferogram, torch.zeros_like(interferogram)),
            (radius_x, radius_x, radius_y, radius_y),
        )
        weights = functional.pad(
            support.to(interferogram.real.dtype),
            (radius_x, radius_x, radius_y, radius_y),
        )
        real = functional.conv2d(
            padded.real[None, None], kernel[None, None], padding=0
        )[0, 0]
        imag = functional.conv2d(
            padded.imag[None, None], kernel[None, None], padding=0
        )[0, 0]
        denominator = functional.conv2d(
            weights[None, None], kernel[None, None], padding=0
        )[0, 0]
        result = torch.complex(real, imag) / denominator.clamp_min(
            torch.finfo(kernel.dtype).eps
        )
        output_mask = (
            support
            & (denominator > 0)
            & torch.isfinite(result.real)
            & torch.isfinite(result.imag)
        )
        return PhaseFilterResult(result, output_mask)


class BoxcarFilter(_SpatialFilter):
    """Valid-weighted odd boxcar smoothing of a wrapped complex IFG.

    Parameters
    ----------
    window : tuple[int, int], default=(5, 5)
        Odd ``(azimuth, range)`` kernel widths, each in ``[3, 257]`` pixels.

    """

    def __init__(self, window: tuple[int, int] = (5, 5)) -> None:
        """Initialize odd ``(azimuth, range)`` boxcar dimensions."""
        if (
            not isinstance(window, tuple)
            or len(window) != 2
            or any(
                type(axis) is not int or axis < 3 or axis % 2 == 0 for axis in window
            )
            or any(axis > 257 for axis in window)
        ):
            message = "Boxcar window axes must be odd integers >= 3"
            raise ValueError(message)
        self.window = window

    def describe(self) -> FilterProvenance:
        """Return canonical inert filter parameters."""
        return FilterProvenance(
            "boxcar",
            {
                "window_azimuth": self.window[0],
                "window_range": self.window[1],
            },
        )

    def apply(
        self, interferogram: torch.Tensor, *, valid_mask: torch.Tensor | None = None
    ) -> PhaseFilterResult:
        """Apply valid-weighted clipped boxcar convolution.

        The window is an odd ``(azimuth, range)`` size in pixels. Edge windows
        are clipped by the valid-weight denominator; unsupported input pixels
        remain invalid in the returned mask.
        """
        support = _validate_input(interferogram, valid_mask)
        kernel = torch.ones(
            self.window, dtype=interferogram.real.dtype, device=interferogram.device
        )
        return self._apply_kernel(interferogram, support, kernel)


class GaussianFilter(_SpatialFilter):
    """Valid-weighted separable Gaussian smoothing of a wrapped IFG.

    Parameters
    ----------
    sigma : tuple[float, float]
        Positive finite ``(azimuth, range)`` standard deviations in pixels,
        each no greater than ``128``.
    truncate : float, default=4.0
        Positive finite kernel radius in standard deviations, no greater than
        ``16``.  The discrete radius is ``ceil(truncate * sigma)``.

    """

    def __init__(self, sigma: tuple[float, float], truncate: float = 4.0) -> None:
        """Initialize ``(azimuth, range)`` Gaussian widths in pixels."""
        if (
            not isinstance(sigma, tuple)
            or len(sigma) != 2
            or any(
                isinstance(axis, bool)
                or not isinstance(axis, (int, float))
                or not math.isfinite(axis)
                or axis <= 0
                or axis > 128.0
                for axis in sigma
            )
            or isinstance(truncate, bool)
            or not isinstance(truncate, (int, float))
            or not math.isfinite(truncate)
            or truncate <= 0
            or truncate > 16.0
        ):
            message = "Gaussian sigma and truncate must be finite and positive"
            raise ValueError(message)
        self.sigma = (float(sigma[0]), float(sigma[1]))
        self.truncate = float(truncate)

    def describe(self) -> FilterProvenance:
        """Return canonical inert filter parameters."""
        return FilterProvenance(
            "gaussian",
            {
                "sigma_azimuth": self.sigma[0],
                "sigma_range": self.sigma[1],
                "truncate": self.truncate,
            },
        )

    def apply(
        self, interferogram: torch.Tensor, *, valid_mask: torch.Tensor | None = None
    ) -> PhaseFilterResult:
        """Apply valid-weighted separable Gaussian convolution.

        ``sigma`` is measured in pixels as ``(azimuth, range)`` standard
        deviation. The radius is ``ceil(truncate * sigma)`` in each axis.
        Edges are handled by valid-weighted clipped convolution, and the
        returned support never includes an invalid input pixel.
        """
        support = _validate_input(interferogram, valid_mask)
        radii = tuple(math.ceil(self.truncate * value) for value in self.sigma)
        axes = [
            torch.arange(
                -radius,
                radius + 1,
                device=interferogram.device,
                dtype=interferogram.real.dtype,
            )
            for radius in radii
        ]
        kernels = [
            torch.exp(-((axis / sigma) ** 2) / 2)
            for axis, sigma in zip(axes, self.sigma, strict=True)
        ]
        kernel = torch.outer(kernels[0], kernels[1])
        return self._apply_kernel(interferogram, support, kernel)
