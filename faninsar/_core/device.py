"""Device utilities for PyTorch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from faninsar.typing import DeviceLike


def cuda_available() -> bool:
    """Check if CUDA (NVIDIA or ROCm) is available."""
    return torch.cuda.is_available()


def mps_available() -> bool:
    """Check if MPS (Mac) is available."""
    return torch.backends.mps.is_available()


def gpu_available() -> bool:
    """Check if GPU is available."""
    return cuda_available() or mps_available()


def parse_device(device: DeviceLike | None) -> torch.device:
    """Parse the device string and return a torch.device object.

    Parameters
    ----------
    device : str or torch.device or None
        The device string to be parsed. If None, it will default to "cuda" if
        CUDA is available, otherwise it will default to "cpu".

    """
    if isinstance(device, (str, type(None))):
        device = torch.device(_parse_device_str(device))
    elif isinstance(device, torch.device):
        pass
    else:
        msg = "device must be a string or torch.device"
        logger.error(msg, stacklevel=2)
        raise TypeError(msg)
    return device


def _parse_device_str(device: str | None) -> str:
    """Resolve a device string to a canonical torch device name.

    Policy (frozen by the GPU/CPU unwrap-stack architecture law):

    - ``None`` / ``"auto"`` / ``"gpu"`` → ``"cuda"`` if available, **else
      ``"cpu"``**. MPS is **never** auto-selected. This is deliberate: MPS
      lacks many ops used by this stack (sparse CG, full fft.dct, some
      linalg) and silently degrading onto it breaks the no-silent-fallback
      contract (R0.3).
    - ``"cuda"`` requested but unavailable → **hard error** (do not fall back
      to CPU silently). Callers that want graceful degradation must pass
      ``"auto"``.
    - ``"cpu"`` → ``"cpu"``.
    - ``"mps"`` → allowed only via **explicit** request (experimental; not a
      production-supported device for this stack). Logged as a warning.
    """
    if device is None or device.lower() in ("auto", "gpu"):
        if cuda_available():
            return "cuda"
        msg = (
            "No CUDA GPU detected. Resolving device to CPU. "
            "MPS is not auto-selected for this stack. "
            "If you want MPS, pass device='mps' explicitly (experimental)."
        )
        logger.warning(msg, stacklevel=2)
        return "cpu"
    device = device.lower()
    if device == "cuda" and not cuda_available():
        msg = (
            "device='cuda' requested but CUDA is not available. "
            "Pass device='auto' for graceful CPU fallback."
        )
        logger.error(msg, stacklevel=2)
        raise RuntimeError(msg)
    if device == "mps":
        logger.warning(
            "device='mps' is experimental and not production-supported "
            "for this stack (sparse CG / DCT / some linalg unsupported).",
            stacklevel=2,
        )
    return device
