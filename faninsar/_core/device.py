"""Device utilities for PyTorch."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import torch

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from faninsar.typing import DeviceLike

GpuMemoryReclaim = Literal["lazy", "eager", "adaptive"]
ReclaimKind = Literal["persist", "stage", "oom", "explicit"]

_GIB = 1024**3
ADAPTIVE_LAZY_MIN_TOTAL_BYTES = 40 * _GIB
ADAPTIVE_EAGER_MAX_TOTAL_BYTES = 12 * _GIB
ADAPTIVE_RESERVED_RATIO = 0.85
_RECLAIM_KINDS: frozenset[str] = frozenset({"persist", "stage", "oom", "explicit"})
_RECLAIM_POLICIES: frozenset[str] = frozenset({"lazy", "eager", "adaptive"})

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


def probe_accelerator_memory(
    device: DeviceLike | torch.device | None,
) -> tuple[int | None, int | None]:
    """Return ``(total_bytes, reserved_bytes)`` for an accelerator.

    Tests monkeypatch this probe to inject capacity and pressure. CPU
    returns ``(None, None)``. Missing CUDA/MPS stats also return
    ``None`` rather than inventing a size.

    Parameters
    ----------
    device : str or torch.device or None
        Device whose allocator accounting is read.

    Returns
    -------
    tuple of int or None
        Total device memory and currently reserved caching-allocator
        bytes. Either element may be ``None`` when the backend does not
        expose the figure.

    """
    resolved = device if isinstance(device, torch.device) else parse_device(device)
    if resolved.type == "cuda":
        if not cuda_available():
            return None, None
        try:
            total = int(torch.cuda.get_device_properties(resolved).total_memory)
        except Exception:
            logger.exception("unable to read CUDA total memory")
            total = None
        try:
            reserved = int(torch.cuda.memory_reserved(resolved))
        except Exception:
            logger.exception("unable to read CUDA reserved memory")
            reserved = None
        return total, reserved
    if resolved.type == "mps":
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        current = getattr(torch.mps, "current_allocated_memory", None)
        try:
            total = int(driver()) if callable(driver) else None
        except Exception:
            logger.exception("unable to read MPS driver memory")
            total = None
        try:
            reserved = int(current()) if callable(current) else None
        except Exception:
            logger.exception("unable to read MPS allocated memory")
            reserved = None
        return total, reserved
    return None, None


def release_accelerator_cache(device: DeviceLike | torch.device | None) -> None:
    """Return unused caching-allocator slabs to the driver.

    This is the only product wrapper around ``torch.cuda.empty_cache`` /
    the MPS equivalent (PROPOSAL-0034). Live tensors are untouched.
    CPU is a no-op.

    Parameters
    ----------
    device : str or torch.device or None
        Accelerator whose unused cached slabs are released.

    """
    resolved = device if isinstance(device, torch.device) else parse_device(device)
    if resolved.type == "cuda":
        torch.cuda.empty_cache()
        return
    if resolved.type == "mps":
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()


def _should_reclaim(
    policy: GpuMemoryReclaim,
    kind: ReclaimKind,
    *,
    total_bytes: int | None,
    reserved_bytes: int | None,
) -> bool:
    """Evaluate the PROPOSAL-0034 reclaim table for one checkpoint."""
    if kind in {"oom", "explicit"}:
        return True
    if policy == "lazy":
        return False
    if policy == "eager":
        return True
    if total_bytes is None or total_bytes <= ADAPTIVE_EAGER_MAX_TOTAL_BYTES:
        return True
    if total_bytes >= ADAPTIVE_LAZY_MIN_TOTAL_BYTES or reserved_bytes is None:
        return False
    return (reserved_bytes / total_bytes) > ADAPTIVE_RESERVED_RATIO


def reclaim_checkpoint(
    device: DeviceLike | torch.device | None,
    policy: GpuMemoryReclaim,
    *,
    kind: ReclaimKind,
    total_bytes: int | None = None,
    reserved_bytes: int | None = None,
) -> bool:
    """Apply ``gpu_memory_reclaim`` at a Stack or worker checkpoint.

    Kernels never call this. ``lazy`` is a no-op except ``oom`` and
    ``explicit``. ``eager`` reclaims at ``persist`` and ``stage``.
    ``adaptive`` uses the 40 GiB / 12 GiB / 0.85 reserved-ratio table
    (PROPOSAL-0034; cutoffs are REFERENCE).

    Parameters
    ----------
    device : str or torch.device or None
        Process-local device that owns the caching allocator.
    policy : {"lazy", "eager", "adaptive"}
        Reclaim policy carried by the orchestrator.
    kind : {"persist", "stage", "oom", "explicit"}
        Checkpoint that triggered the evaluation.
    total_bytes, reserved_bytes : int, optional
        Injected capacity and reserved size for tests. When omitted the
        values come from :func:`probe_accelerator_memory`.

    Returns
    -------
    bool
        ``True`` when :func:`release_accelerator_cache` ran.

    Raises
    ------
    ValueError
        If *policy* or *kind* is not an admitted token.

    """
    if policy not in _RECLAIM_POLICIES:
        message = f"unsupported gpu_memory_reclaim policy: {policy!r}"
        logger.error(message)
        raise ValueError(message)
    if kind not in _RECLAIM_KINDS:
        message = f"unsupported reclaim checkpoint kind: {kind!r}"
        logger.error(message)
        raise ValueError(message)
    resolved = device if isinstance(device, torch.device) else parse_device(device)
    if resolved.type not in {"cuda", "mps"}:
        return False
    if total_bytes is None and reserved_bytes is None:
        total_bytes, reserved_bytes = probe_accelerator_memory(resolved)
    if not _should_reclaim(
        policy,
        kind,
        total_bytes=total_bytes,
        reserved_bytes=reserved_bytes,
    ):
        return False
    release_accelerator_cache(resolved)
    return True
