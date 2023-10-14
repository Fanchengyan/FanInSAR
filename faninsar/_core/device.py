import warnings
from typing import Optional, Union

import torch


def cuda_available() -> bool:
    """Check if CUDA (NVIDIA or ROCm) is available."""
    return torch.cuda.is_available()


def mps_available() -> bool:
    """Check if MPS (Mac) is available."""
    return torch.backends.mps.is_available()


def gpu_available() -> bool:
    """Check if GPU is available."""
    return cuda_available() or mps_available()


def parse_device(device: Optional[Union[str, torch.device]]):
    if isinstance(device, (str, type(None))):
        device = torch.device(_parse_device_str(device))
    elif isinstance(device, torch.device):
        pass
    else:
        raise TypeError("device must be a string or torch.device")
    return device


def _parse_device_str(device: str):
    if device is None:
        if gpu_available():
            device = "cuda"
        else:
            device = "cpu"
    elif device.lower() == "gpu":
        if not gpu_available():
            warnings.warn("GPU is not available, using CPU instead.")
            device = "cpu"
    else:
        device = device.lower()

    return device
