"""Device utilities for PyTorch."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from faninsar.typing import DeviceLike

PUBLISHED_DEVICE_TYPES: frozenset[str] = frozenset({"cpu", "cuda"})
_AUTO_ALIASES: frozenset[str] = frozenset({"auto", "gpu"})
_DEVICE_SPEC = re.compile(r"^([a-z][a-z0-9]*?)(?::(\d+))?$")


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
    """Admit a Torch device request and return its resolved identity.

    ``parse_device`` is the only device resolver. Any value Torch can
    construct is a valid *request*. This release *publishes and tests*
    CPU and CUDA, including ``cuda:N``. ``None``, ``"auto"``, and
    ``"gpu"`` select the first published device that is present (CUDA if
    available, otherwise CPU) and never an unpublished backend.

    An explicit request stays on that device. A missing or unusable
    backend, or a CUDA ordinal outside ``range(torch.cuda.device_count())``,
    fails before any array copy. ``torch.device("cuda:256")`` wrapping to
    ``cuda:0`` is not admission.

    Parameters
    ----------
    device : str or torch.device or None
        Requested device. Strings are stripped and compared
        case-insensitively for aliases and device type names.

    Returns
    -------
    torch.device
        Admitted device identity. Construction of a Torch device is not
        sufficient for admission.

    Raises
    ------
    TypeError
        If *device* is not a string, :class:`torch.device`, or ``None``.
    RuntimeError
        If Torch cannot construct the device, the backend is missing or
        unusable, or a CUDA ordinal is outside the visible device range.

    """
    if isinstance(device, torch.device):
        return _admit_constructed_device(device, requested=str(device))
    if device is not None and not isinstance(device, str):
        msg = "device must be a string or torch.device"
        logger.error(msg, stacklevel=2)
        raise TypeError(msg)
    if device is None or device.strip().lower() in _AUTO_ALIASES:
        return _select_published_device()
    requested = device.strip()
    match = _DEVICE_SPEC.fullmatch(requested.lower())
    if match is None:
        return _admit_torch_construction(requested)
    type_name, index_text = match.group(1), match.group(2)
    index = int(index_text) if index_text is not None else None
    if type_name == "cuda":
        return _admit_cuda(index, requested=requested)
    constructed = _construct_device(type_name, index, requested=requested)
    return _admit_constructed_device(constructed, requested=requested)


def _select_published_device() -> torch.device:
    """Return the first present published device (CUDA, else CPU)."""
    if cuda_available():
        return torch.device("cuda")
    msg = (
        "No CUDA GPU detected. Resolving device to CPU. "
        "Unpublished backends are not auto-selected."
    )
    logger.warning(msg, stacklevel=2)
    return torch.device("cpu")


def _admit_cuda(index: int | None, *, requested: str) -> torch.device:
    """Admit a CUDA identity only when the backend and ordinal are usable."""
    if not cuda_available():
        msg = (
            f"CUDA requested but is not available ({requested!r}). "
            "Pass device='auto' for graceful CPU fallback."
        )
        logger.error(msg, stacklevel=2)
        raise RuntimeError(msg)
    count = torch.cuda.device_count()
    if index is not None and index not in range(count):
        msg = (
            f"CUDA ordinal {index} is not in range({count}); "
            f"device={requested!r} is not admitted. "
            "Construction wrapping such as cuda:256 -> cuda:0 is forbidden."
        )
        logger.error(msg, stacklevel=2)
        raise RuntimeError(msg)
    if index is None:
        return torch.device("cuda")
    return torch.device("cuda", index)


def _construct_device(
    type_name: str,
    index: int | None,
    *,
    requested: str,
) -> torch.device:
    """Construct a non-CUDA Torch device, or fail if Torch rejects it."""
    try:
        if index is None:
            return torch.device(type_name)
        return torch.device(type_name, index)
    except (RuntimeError, ValueError) as error:
        msg = f"unsupported torch device: {requested!r}"
        logger.exception(msg)
        raise RuntimeError(msg) from error


def _admit_torch_construction(requested: str) -> torch.device:
    """Admit a string Torch can construct after backend checks."""
    try:
        constructed = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        msg = f"unsupported torch device: {requested!r}"
        logger.exception(msg)
        raise RuntimeError(msg) from error
    return _admit_constructed_device(constructed, requested=requested)


def _admit_constructed_device(
    device: torch.device,
    *,
    requested: str,
) -> torch.device:
    """Re-validate a constructed device; construction is not admission."""
    if device.type == "cpu":
        return device
    if device.type == "cuda":
        return _admit_cuda(device.index, requested=requested)
    if device.type == "mps" and not mps_available():
        msg = f"MPS requested but is not available ({requested!r})."
        logger.error(msg, stacklevel=2)
        raise RuntimeError(msg)
    if device.type not in PUBLISHED_DEVICE_TYPES:
        logger.warning(
            "device=%r is an unpublished Torch backend; this release "
            "publishes and tests CPU and CUDA only.",
            requested,
            stacklevel=2,
        )
    return device
