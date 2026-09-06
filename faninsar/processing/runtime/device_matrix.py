"""Explicit device capability matrix for CPU/CUDA/MPS kernels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import torch

from faninsar.logging import setup_logger
from faninsar.processing.runtime.device import cuda_available, mps_available

logger = setup_logger(__name__)


class DeviceKind(StrEnum):
    """Supported execution devices."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass(frozen=True, slots=True)
class KernelCapability:
    """Capability row for one named kernel."""

    name: str
    cpu: bool
    cuda: bool
    mps: bool
    notes: str = ""


KERNEL_CAPABILITIES: Final[tuple[KernelCapability, ...]] = (
    KernelCapability("complex_multiply", cpu=True, cuda=True, mps=True),
    KernelCapability("phase_rotation", cpu=True, cuda=True, mps=True),
    KernelCapability("window_statistics", cpu=True, cuda=True, mps=True),
    KernelCapability(
        "fft_filter",
        cpu=True,
        cuda=True,
        mps=False,
        notes="MPS FFT limited",
    ),
    KernelCapability(
        "irls_pcg",
        cpu=True,
        cuda=True,
        mps=False,
        notes="sparse/global solver",
    ),
    KernelCapability("goldstein_filter", cpu=True, cuda=True, mps=False),
)


def capability_matrix() -> tuple[KernelCapability, ...]:
    """Return the static kernel capability matrix."""
    return KERNEL_CAPABILITIES


def probe_devices() -> dict[str, object]:
    """Probe runtime device availability without claiming untested kernels."""
    return {
        "cpu": True,
        "cuda": bool(cuda_available()),
        "mps": bool(mps_available()),
        "torch": True,
        "torch_version": torch.__version__,
    }


def select_device_for_kernel(
    kernel: str,
    preferred: DeviceKind | str = DeviceKind.CPU,
) -> DeviceKind:
    """Select a device that both advertises the kernel and is available."""
    preferred_kind = DeviceKind(str(preferred).lower())
    row = next((item for item in KERNEL_CAPABILITIES if item.name == kernel), None)
    if row is None:
        message = f"unknown kernel: {kernel}"
        logger.error(message)
        raise ValueError(message)

    available = probe_devices()
    order = [preferred_kind]
    for kind in (DeviceKind.CUDA, DeviceKind.MPS, DeviceKind.CPU):
        if kind not in order:
            order.append(kind)

    for kind in order:
        supported = {
            DeviceKind.CPU: row.cpu,
            DeviceKind.CUDA: row.cuda,
            DeviceKind.MPS: row.mps,
        }[kind]
        present = bool(available.get(kind.value, False))
        if supported and present:
            return kind
    message = f"no available device supports kernel {kernel!r}"
    logger.error(message)
    raise ValueError(message)


def complex_multiply(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Complex multiply with host/device tensors (CPU/CUDA/MPS)."""
    return left * right
