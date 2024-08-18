"""Device utilities for PyTtorch."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

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
        raise TypeError(msg)
    return device


def _parse_device_str(device: str) -> str:
    if device is None or device.lower() == "gpu":
        if cuda_available():
            device = "cuda"
        elif mps_available():
            device = "mps"
        else:
            if isinstance(device, str):
                warnings.warn(
                    "No GPU detected. Falling back to CPU. "
                    "If you would like to use a GPU, please install PyTorch"
                    " with CUDA support.",
                    stacklevel=1,
                )
            device = "cpu"
    else:
        device = device.lower()

    return device
