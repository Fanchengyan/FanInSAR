"""Geometry-driven coarse offset fields and fine correlation refinement."""

from __future__ import annotations

import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import map_coordinates

from faninsar.logging import setup_logger
from faninsar.processing.coreg.ampcor_backend import (
    AmpcorBackend,
    AmpcorCandidateError,
    AmpcorEnergyCandidate,
    eager_ampcor_candidate,
    native_workspace_bytes,
)
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Iterator

    from faninsar.processing.tops.deramp import TOPSCarrierModel

logger = setup_logger(__name__)

_TORCH_AMPCOR_WORKSPACE_CAP_BYTES = 256 * 1024**2
_TORCH_AMPCOR_MAX_PATCHES = 4096
_TORCH_AMPCOR_MAX_INPUT_DIM = 32768
_TORCH_AMPCOR_MAX_INPUT_BYTES = 512 * 1024**2
# Independent cap for compatibility copies, charged before any allocation.
_TORCH_AMPCOR_CONVERSION_CAP_BYTES = _TORCH_AMPCOR_MAX_INPUT_BYTES
_TORCH_AMPCOR_MAX_PATCH_DIM = 4096
_TORCH_AMPCOR_RESERVED_BYTES: dict[str, int] = {}
_TORCH_AMPCOR_QUALIFIED_TORCH = "2.8.0"
_TORCH_AMPCOR_QUALIFIED_CUDA = "12.8"
_TORCH_AMPCOR_QUALIFIED_DEVICE = "A100"
# The strict boundary-oracle drift window is qualified for these FFT shapes.
# CUDA requests outside this set fail closed before input-derived device work.
_TORCH_AMPCOR_QUALIFIED_SHAPES: frozenset[tuple[int, int, int, int]] = frozenset(
    {
        (32, 64, 8, 8),
        (32, 64, 16, 16),
    }
)
# Backend FFT paths differed by at most eight binary64 ULPs in the boundary
# qualification fixture; the next power-of-two bucket keeps that drift stable.
_AMPCOR_CULL_ULPS = 16
_TORCH_AMPCOR_ADMISSION_LOCK = Lock()
_TORCH_AMPCOR_LOCK_ROOT = Path(tempfile.gettempdir()) / (
    f"faninsar-ampcor-{getattr(os, 'getuid', lambda: 0)()}"
)


def _canonical_torch_device(device: object) -> object:
    """Return a stable Torch device, including a concrete CUDA index."""
    import torch as torch_module

    if isinstance(device, torch_module.device):
        resolved = device
    else:
        name = str(device).strip().lower()
        if name in {"gpu", "cuda", "cuda:default"}:
            name = "cuda"
        resolved = torch_module.device(name)
    if resolved.type == "cuda":
        index = resolved.index
        if index is None:
            index = (
                int(torch_module.cuda.current_device())
                if torch_module.cuda.is_available()
                else 0
            )
        return torch_module.device("cuda", index)
    return resolved


def _validate_torch_ampcor_runtime(device: object, torch_module: object) -> None:
    """Fail closed unless the runtime matches the qualified CUDA lane.

    The Torch executor has only been qualified for an A100 with Torch
    ``2.8.0+cu128`` and CUDA ``12.8``.  Capability checks are deliberately
    exact: allocator and FFT behavior on another driver, GPU family, or Torch
    build is not inferred from the A100 qualification record.

    Parameters
    ----------
    device : object
        Resolved Torch device.
    torch_module : object
        Imported Torch module, accepted as an object to keep the optional
        dependency lazy.

    Raises
    ------
    InvalidProcessingStateError
        If the requested CUDA runtime is outside the qualified lane.
    RuntimeError
        If CUDA is unavailable.

    """
    resolved = device
    if not isinstance(resolved, torch_module.device):
        resolved = torch_module.device(str(device))
    if resolved.type != "cuda":
        return
    if not torch_module.cuda.is_available():
        message = "CUDA requested for Torch Ampcor but is unavailable"
        logger.error(message)
        raise RuntimeError(message)
    torch_version = str(getattr(torch_module, "__version__", "")).split("+", 1)[0]
    cuda_version = str(getattr(getattr(torch_module, "version", None), "cuda", ""))
    try:
        properties = torch_module.cuda.get_device_properties(resolved)
        device_name = str(properties.name)
        capability = (int(properties.major), int(properties.minor))
    except Exception as error:
        message = "unable to inspect the CUDA runtime for Torch Ampcor"
        logger.exception(message)
        raise RuntimeError(message) from error
    if (
        torch_version != _TORCH_AMPCOR_QUALIFIED_TORCH
        or cuda_version != _TORCH_AMPCOR_QUALIFIED_CUDA
        or _TORCH_AMPCOR_QUALIFIED_DEVICE not in device_name
        or capability != (8, 0)
    ):
        message = (
            "Torch Ampcor CUDA runtime is outside the qualified lane: "
            f"requires {_TORCH_AMPCOR_QUALIFIED_DEVICE}/"
            f"compute-8.0, torch {_TORCH_AMPCOR_QUALIFIED_TORCH}+cu128, "
            f"CUDA {_TORCH_AMPCOR_QUALIFIED_CUDA}; got "
            f"{device_name!r}/compute-{capability[0]}.{capability[1]}, "
            f"torch {torch_version or '<unknown>'}, CUDA {cuda_version or '<unknown>'}"
        )
        logger.error(message)
        reject_invalid_state(message)


def _validate_torch_ampcor_shape(
    *,
    window_az: int,
    window_rg: int,
    search_az: int,
    search_rg: int,
    device_type: str,
) -> None:
    """Reject CUDA FFT shapes outside the measured boundary-oracle lane.

    Parameters
    ----------
    window_az, window_rg : int
        Ampcor reference-window dimensions.
    search_az, search_rg : int
        Ampcor search half-widths.
    device_type : str
        Resolved Torch device type.

    Raises
    ------
    InvalidProcessingStateError
        If CUDA is requested for an unqualified FFT shape.

    Notes
    -----
    CPU Torch remains a diagnostic path and is intentionally not restricted by
    the A100 boundary qualification. NumPy is unaffected.

    """
    if device_type != "cuda":
        return
    shape = (window_az, window_rg, search_az, search_rg)
    if shape not in _TORCH_AMPCOR_QUALIFIED_SHAPES:
        message = (
            "Torch Ampcor CUDA FFT shape is outside the qualified "
            f"boundary-oracle lane: got {shape}, allowed "
            f"{sorted(_TORCH_AMPCOR_QUALIFIED_SHAPES)}"
        )
        logger.error(message)
        reject_invalid_state(message)


def _synchronize_torch_device(device: object) -> None:
    """Synchronize an accelerator before releasing its admitted workspace."""
    import torch as torch_module

    resolved = device
    if not isinstance(resolved, torch_module.device):
        resolved = torch_module.device(str(resolved))
    if resolved.type == "cuda":
        torch_module.cuda.synchronize(resolved)
    elif resolved.type == "mps" and hasattr(torch_module, "mps"):
        torch_module.mps.synchronize()


def _release_torch_device_cache(device: object) -> None:
    """Request best-effort allocator cache release before a lease ends.

    Torch drivers may retain allocations after this request; callers must
    release Python references and rely on the admission ledger independently.
    """
    import torch as torch_module

    resolved = device
    if not isinstance(resolved, torch_module.device):
        resolved = torch_module.device(str(resolved))
    if resolved.type == "cuda":
        torch_module.cuda.empty_cache()
    elif resolved.type == "mps" and hasattr(torch_module, "mps"):
        empty_cache = getattr(torch_module.mps, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()


def _torch_ampcor_admission_key(device: object, torch_module: object) -> str:
    """Return a physical-device admission key for a Torch device."""
    resolved = device
    if not hasattr(torch_module, "device") or not isinstance(
        resolved, torch_module.device
    ):
        resolved = torch_module.device(str(device))
    if resolved.type != "cuda":
        return str(resolved)
    try:
        uuid = str(torch_module.cuda.get_device_properties(resolved).uuid)
    except Exception as error:
        message = "unable to identify the physical CUDA device for admission"
        logger.exception(message)
        raise RuntimeError(message) from error
    if not uuid or uuid == "None":
        message = "Torch did not expose a physical CUDA device UUID"
        logger.error(message)
        raise RuntimeError(message)
    return f"cuda-uuid:{uuid}"


@contextmanager
def _admit_torch_ampcor_process(device_key: str) -> Iterator[None]:
    """Serialize explicit accelerator work across processes on one device.

    ``threading.Lock`` cannot protect direct production calls made by separate
    workers.  A POSIX advisory lock gives each physical CUDA ordinal one
    process-wide admission domain and releases automatically if a worker dies.

    The admission domain is a cooperative contract for workers sharing the
    same UID and ``TMPDIR`` namespace.  A hostile process with the same UID
    can rename entries in an owner-writable temporary directory; isolation
    from that threat requires an OS- or service-managed lock outside this
    helper.
    """
    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - supported hosts are POSIX
        message = "cross-process CUDA admission requires POSIX file locking"
        logger.exception(message)
        raise RuntimeError(message) from error
    required_flags = ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC")
    if any(not hasattr(os, flag) for flag in required_flags):
        message = "secure CUDA admission requires POSIX no-follow directory opens"
        logger.error(message)
        raise RuntimeError(message)
    directory_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | os.O_CLOEXEC
    file_flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        _TORCH_AMPCOR_LOCK_ROOT.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError:
        logger.exception("Unable to create the Ampcor admission directory")
        raise
    try:
        root_descriptor = os.open(os.fspath(_TORCH_AMPCOR_LOCK_ROOT), directory_flags)
    except OSError:
        logger.exception("Unable to open the Ampcor admission directory securely")
        raise
    lock_descriptor: int | None = None
    safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", device_key)
    try:
        root_stat = os.fstat(root_descriptor)
        root_path_stat = _TORCH_AMPCOR_LOCK_ROOT.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != os.geteuid()
            or root_stat.st_mode & 0o077
            or (root_stat.st_dev, root_stat.st_ino)
            != (root_path_stat.st_dev, root_path_stat.st_ino)
        ):
            message = "Ampcor admission directory failed ownership or inode checks"
            logger.error(message)
            raise RuntimeError(message)
        lock_name = f"{safe_key}.lock"
        try:
            lock_descriptor = os.open(
                lock_name,
                file_flags,
                0o600,
                dir_fd=root_descriptor,
            )
        except OSError:
            logger.exception("Unable to open the Ampcor admission lock securely")
            raise

        def validate_lock_entry() -> None:
            """Verify the lock descriptor still names its private directory entry."""
            assert lock_descriptor is not None
            descriptor_stat = os.fstat(lock_descriptor)
            path_stat = os.stat(
                lock_name,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(descriptor_stat.st_mode)
                or descriptor_stat.st_uid != os.geteuid()
                or descriptor_stat.st_mode & 0o077
                or not stat.S_ISREG(path_stat.st_mode)
                or path_stat.st_uid != os.geteuid()
                or path_stat.st_mode & 0o077
                or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                message = "Ampcor admission lock failed ownership or inode checks"
                logger.error(message)
                raise RuntimeError(message)

        validate_lock_entry()
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = f"Ampcor device admission is busy for {device_key}"
            logger.exception(message)
            raise ValueError(message) from error
        validate_lock_entry()
        yield
    finally:
        try:
            if lock_descriptor is not None:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
            os.close(root_descriptor)


def _validate_real_scalar(name: str, value: object, *, nonnegative: bool) -> float:
    """Validate a finite real scalar used by Ampcor culling."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        reject_invalid_state(f"Ampcor {name} must be a finite real scalar")
    resolved = float(value)
    if not np.isfinite(resolved):
        reject_invalid_state(f"Ampcor {name} must be a finite real scalar")
    if nonnegative and resolved < 0:
        reject_invalid_state(f"Ampcor {name} must be non-negative")
    return resolved


def _is_cpu_ampcor_device(device: object) -> bool:
    """Return whether a device value is the portable CPU lane."""
    normalized = str(device).strip().lower()
    return normalized in {"auto", "cpu"} or normalized.startswith("cpu:")


def resolve_ampcor_policy(
    executor: object,
    device: object,
) -> tuple[Literal["torch"], str]:
    """Resolve the public Ampcor executor/device contract.

    ``numpy`` is retained as a compatibility spelling and is always routed
    to the bounded Torch implementation. ``auto`` and ``cpu`` select the
    portable Torch CPU lane. Explicit CUDA selects Torch and is canonicalized
    to a decimal device ordinal (so ``cuda:00`` and ``cuda:0`` have identical
    dispatch and admission identity).

    Parameters
    ----------
    executor : object
        Requested implementation, ``"auto"``, ``"numpy"`` or ``"torch"``.
    device : object
        Requested runtime device value.

    Raises
    ------
    InvalidProcessingStateError
        If the executor or device spelling is unsupported, or the device is
        outside the qualified Ampcor contract.

    """
    if not isinstance(executor, str) or executor not in {"auto", "numpy", "torch"}:
        message = f"unsupported Ampcor executor: {executor!r}"
        logger.error(message)
        reject_invalid_state(message)
    if executor == "numpy":
        logger.warning(
            "Ampcor executor='numpy' is deprecated; routing through Torch compatibility"
        )
    if not isinstance(device, str):
        message = f"unsupported Ampcor device: {device!r}"
        logger.error(message)
        reject_invalid_state(message)
    canonical = device.strip().lower()
    if canonical in {"auto", "cpu"}:
        return "torch", "cpu"
    if canonical in {"gpu", "cuda", "cuda:default"}:
        canonical = "cuda"
    elif canonical.startswith("cuda:"):
        suffix = canonical.removeprefix("cuda:")
        if not suffix.isdigit():
            message = f"unsupported Ampcor device: {device!r}"
            logger.error(message)
            reject_invalid_state(message)
        canonical = f"cuda:{int(suffix)}"
    elif canonical in {"mps", "metal"}:
        message = "MPS is not a qualified Ampcor device; use CPU or CUDA"
        logger.error(message)
        reject_invalid_state(message)
    else:
        message = f"unsupported Ampcor device: {device!r}"
        logger.error(message)
        reject_invalid_state(message)
    return "torch", canonical


def _validate_ampcor_device_contract(executor: str, device: object) -> None:
    """Validate executor/device requests through the canonical resolver."""
    resolve_ampcor_policy(executor, device)


def _validate_ampcor_accelerator(device: str) -> None:
    """Validate an explicit accelerator before reading caller-owned arrays."""
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError as error:
        message = "explicit CUDA Ampcor requires torch"
        logger.exception(message)
        raise ImportError(message) from error
    resolved = _canonical_torch_device(device)
    _validate_torch_ampcor_runtime(resolved, torch)
    _torch_ampcor_admission_key(resolved, torch)


def _validate_ampcor_inputs(
    reference: object,
    secondary: object,
    *,
    torch_contract: bool = False,
    conversion_limit_bytes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Admit only supported, directly indexable Ampcor input arrays.

    Ampcor materializes bounded tiles directly from the caller-owned arrays.
    Torch compatibility inputs are therefore checked for a bounded,
    non-overlapping layout and copied once when a safe contiguous copy is
    required. Float32/64 and complex64/128 inputs retain their dtype, while
    integer and boolean inputs are converted to float64; object arrays and
    malformed layouts fail closed.

    Parameters
    ----------
    reference, secondary : object
        Candidate two-dimensional NumPy arrays on the same grid.
    torch_contract : bool, optional
        Apply the Torch compatibility normalization. The default is retained
        for callers of the private validator and keeps the input arrays as-is.
    conversion_limit_bytes : int or None, optional
        Additional public workspace limit for compatibility copies. ``None``
        uses only the independent conversion cap.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The validated input arrays.

    Raises
    ------
    InvalidProcessingStateError
        If either input violates the shape contract, or the stricter Torch
        dtype/stride contract when requested.

    """
    if not isinstance(reference, np.ndarray) or not isinstance(secondary, np.ndarray):
        message = "patch amplitude shift requires NumPy arrays"
        logger.error(message)
        reject_invalid_state(message)
    if reference.ndim != 2 or secondary.ndim != 2 or reference.shape != secondary.shape:
        message = "patch amplitude shift requires matching 2-D arrays"
        logger.error(message)
        reject_invalid_state(message)
    if not torch_contract:
        return reference, secondary

    candidates: list[tuple[str, np.ndarray, np.dtype]] = []
    for name, samples in (("reference", reference), ("secondary", secondary)):
        if samples.dtype.kind not in "biufc":
            message = f"Ampcor {name} dtype {samples.dtype} is unsupported"
            logger.error(message)
            reject_invalid_state(message)
        if samples.nbytes > _TORCH_AMPCOR_MAX_INPUT_BYTES:
            message = f"Ampcor {name} input exceeds the admitted memory limit"
            logger.error(message)
            reject_invalid_state(message)
        if not _ampcor_layout_is_safe(samples):
            message = f"Ampcor {name} has an overlapping or out-of-bounds layout"
            logger.error(message)
            reject_invalid_state(message)
        if samples.dtype in {
            np.dtype(np.float32),
            np.dtype(np.float64),
            np.dtype(np.complex64),
            np.dtype(np.complex128),
        }:
            target_dtype = samples.dtype
        elif samples.dtype.kind in "biu":
            target_dtype = np.dtype(np.float64)
        else:
            message = f"Ampcor {name} dtype {samples.dtype} is unsupported"
            logger.error(message)
            reject_invalid_state(message)
        candidates.append((name, samples, target_dtype))

    conversion_bytes = sum(
        int(samples.size) * target_dtype.itemsize
        for _name, samples, target_dtype in candidates
        if not (
            samples.flags.c_contiguous
            and samples.flags.owndata
            and samples.dtype == target_dtype
        )
    )
    conversion_limit = _TORCH_AMPCOR_CONVERSION_CAP_BYTES
    if conversion_limit_bytes is not None:
        conversion_limit = min(conversion_limit, int(conversion_limit_bytes))
    if conversion_bytes > conversion_limit:
        message = (
            "Ampcor compatibility conversion exceeds the admitted memory limit "
            f"({conversion_limit} bytes)"
        )
        logger.error(message)
        reject_invalid_state(message)

    normalized: list[np.ndarray] = []
    for name, samples, target_dtype in candidates:
        if (
            samples.flags.c_contiguous
            and samples.flags.owndata
            and samples.dtype == target_dtype
        ):
            normalized.append(samples)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                copied = np.array(samples, dtype=target_dtype, order="C", copy=True)
            if not np.all(np.isfinite(copied)):
                message = f"Ampcor {name} contains values outside the Torch range"
                logger.error(message)
                reject_invalid_state(message)
            normalized.append(copied)
    return normalized[0], normalized[1]


def _ampcor_layout_is_safe(samples: np.ndarray) -> bool:
    """Return whether a two-dimensional array can be copied safely.

    Positive and negative strides are accepted when dimensions do not overlap
    and every addressed byte lies within the owning allocation. This rejects
    zero-stride ``as_strided`` views and fabricated out-of-bounds layouts
    before ``np.array(..., copy=True)`` can dereference them.
    """
    itemsize = int(samples.dtype.itemsize)
    if itemsize < 1 or samples.ndim != 2:
        return False
    dimensions = sorted(
        (
            (abs(int(stride)), int(size))
            for size, stride in zip(samples.shape, samples.strides, strict=True)
        ),
        key=lambda value: value[0],
    )
    footprint = itemsize
    for stride, size in dimensions:
        if size > 1 and stride < footprint:
            return False
        footprint *= max(size, 1)
    offsets = [
        int(size - 1) * int(stride)
        for size, stride in zip(samples.shape, samples.strides, strict=True)
    ]
    minimum = min(0, sum(offset for offset in offsets if offset < 0))
    maximum = max(0, sum(offset for offset in offsets if offset > 0)) + itemsize
    owner = samples
    while isinstance(getattr(owner, "base", None), np.ndarray):
        owner = owner.base
    try:
        sample_pointer = int(samples.__array_interface__["data"][0])
        owner_pointer = int(owner.__array_interface__["data"][0])
    except (KeyError, TypeError, ValueError):
        return False
    start = sample_pointer - owner_pointer + minimum
    end = sample_pointer - owner_pointer + maximum
    return 0 <= start <= end <= int(owner.nbytes)


@contextmanager
def _admit_torch_ampcor_workspace(
    device_key: str,
    planned_bytes: int,
    limit_bytes: int,
) -> Iterator[None]:
    """Reserve an estimated Ampcor workspace ledger entry.

    The estimate is intentionally conservative and process-local. It is a
    cooperative admission ledger, not a measurement or guarantee of total
    device memory consumed by a Torch driver or allocator.
    """
    with _TORCH_AMPCOR_ADMISSION_LOCK:
        reserved = _TORCH_AMPCOR_RESERVED_BYTES.get(device_key, 0)
        if reserved + planned_bytes > limit_bytes:
            message = (
                "Ampcor estimated workspace admission rejected for "
                f"{device_key}: reserved={reserved} planned={planned_bytes} "
                f"limit={limit_bytes}"
            )
            logger.error(message)
            raise ValueError(message)
        _TORCH_AMPCOR_RESERVED_BYTES[device_key] = reserved + planned_bytes
    try:
        if device_key.startswith(("cuda", "mps")):
            with _admit_torch_ampcor_process(device_key):
                yield
        else:
            yield
    finally:
        with _TORCH_AMPCOR_ADMISSION_LOCK:
            remaining = _TORCH_AMPCOR_RESERVED_BYTES.get(device_key, 0) - planned_bytes
            if remaining > 0:
                _TORCH_AMPCOR_RESERVED_BYTES[device_key] = remaining
            else:
                _TORCH_AMPCOR_RESERVED_BYTES.pop(device_key, None)


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


def _ampcor_magnitude_tile(
    samples: np.ndarray,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
    *,
    cyclic_shift: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Materialize one float64 magnitude tile from a complex input.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex or real two-dimensional input array.
    row_start, row_stop, column_start, column_stop : int
        Bounds of the tile to materialize.
    cyclic_shift : tuple[int, int], optional
        Integer ``(azimuth, range)`` shift applied lazily with cyclic source
        indexing. This avoids a full-image ``numpy.roll`` allocation during
        production pre-alignment.

    Returns
    -------
    numpy.ndarray
        Float64 magnitude tile. Complex64 inputs are magnituded at their
        native precision before promotion to float64; complex128 inputs retain
        full float64 magnitude precision.

    """
    if cyclic_shift == (0, 0):
        tile_source = samples[row_start:row_stop, column_start:column_stop]
    else:
        shift_az, shift_rg = cyclic_shift
        rows = (np.arange(row_start, row_stop) - shift_az) % samples.shape[0]
        columns = (np.arange(column_start, column_stop) - shift_rg) % samples.shape[1]
        tile_source = samples[np.ix_(rows, columns)]
    tile = np.asarray(tile_source)
    return np.abs(tile).astype(np.float64, copy=False)


def _ampcor_input_budget_bytes(
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    patch_elements: int,
    batch_capacity: int,
) -> int:
    """Estimate retained sources plus bounded magnitude tile memory.

    A materialized tile can transiently hold a complex128 source conversion,
    a float64 magnitude, and one float64 batch stack. Charging 32 bytes per
    sample covers the highest-precision compatibility path while avoiding a
    full-burst magnitude allocation.

    Parameters
    ----------
    reference, secondary : numpy.ndarray
        Source arrays retained for the duration of Ampcor.
    patch_elements : int
        Combined reference-window and secondary-search elements per patch.
    batch_capacity : int
        Maximum number of patches materialized at once.

    Returns
    -------
    int
        Conservative input-memory estimate in bytes.

    """
    source_bytes = int(reference.nbytes) + int(secondary.nbytes)
    tile_bytes = (
        int(patch_elements)
        * int(batch_capacity)
        * (np.dtype(np.complex128).itemsize + 2 * np.dtype(np.float64).itemsize)
    )
    return source_bytes + tile_bytes


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


def _ampcor_cull_mask_numpy(
    snr: object,
    d_rg: object,
    d_az: object,
    *,
    snr_threshold: float,
    max_abs_residual: float,
) -> np.ndarray:
    """Apply the strict inclusive Ampcor cull to NumPy values."""
    snr_values = np.asarray(snr, dtype=np.float64)
    range_values = np.asarray(d_rg, dtype=np.float64)
    azimuth_values = np.asarray(d_az, dtype=np.float64)
    threshold = np.float64(snr_threshold)
    limit = np.float64(max_abs_residual)
    return (
        np.isfinite(snr_values)
        & np.isfinite(range_values)
        & np.isfinite(azimuth_values)
        & (snr_values >= threshold)
        & (np.abs(range_values) <= limit)
        & (np.abs(azimuth_values) <= limit)
    )


def _ampcor_cull_mask_torch(
    snr: object,
    d_rg: object,
    d_az: object,
    *,
    snr_threshold: float,
    max_abs_residual: float,
    torch_module: object,
) -> object:
    """Apply the same strict inclusive Ampcor cull to Torch tensors."""
    snr_values = snr.to(dtype=torch_module.float64)
    range_values = d_rg.to(dtype=torch_module.float64)
    azimuth_values = d_az.to(dtype=torch_module.float64)
    threshold = torch_module.tensor(
        np.float64(snr_threshold),
        dtype=torch_module.float64,
        device=snr.device,
    )
    limit = torch_module.tensor(
        np.float64(max_abs_residual),
        dtype=torch_module.float64,
        device=snr.device,
    )
    return (
        torch_module.isfinite(snr_values)
        & torch_module.isfinite(range_values)
        & torch_module.isfinite(azimuth_values)
        & (snr_values >= threshold)
        & (range_values.abs() <= limit)
        & (azimuth_values.abs() <= limit)
    )


def _ampcor_boundary_mask_torch(
    snr: object,
    d_rg: object,
    d_az: object,
    *,
    snr_threshold: float,
    max_abs_residual: float,
    torch_module: object,
) -> object:
    """Find Torch results requiring the rare NumPy boundary oracle."""
    snr_values = snr.to(dtype=torch_module.float64)
    range_values = d_rg.to(dtype=torch_module.float64).abs()
    azimuth_values = d_az.to(dtype=torch_module.float64).abs()
    snr_quantum = _ampcor_boundary_quantum(snr_threshold)
    residual_quantum = _ampcor_boundary_quantum(max_abs_residual)
    snr_boundary = torch_module.tensor(
        np.float64(snr_threshold), dtype=torch_module.float64, device=snr.device
    )
    residual_boundary = torch_module.tensor(
        np.float64(max_abs_residual),
        dtype=torch_module.float64,
        device=snr.device,
    )
    return (
        (
            torch_module.isfinite(snr_values)
            & ((snr_values - snr_boundary).abs() <= snr_quantum)
        )
        | (
            torch_module.isfinite(range_values)
            & ((range_values - residual_boundary).abs() <= residual_quantum)
        )
        | (
            torch_module.isfinite(azimuth_values)
            & ((azimuth_values - residual_boundary).abs() <= residual_quantum)
        )
    )


def _ampcor_boundary_quantum(boundary: float) -> float:
    """Return the canonical ULP quantization quantum for a boundary."""
    boundary_value = np.float64(boundary)
    spacing = np.spacing(abs(boundary_value))
    if boundary_value == 0.0 or spacing == 0.0 or not np.isfinite(spacing):
        spacing = np.finfo(np.float64).eps
    return float(_AMPCOR_CULL_ULPS * spacing)


def _torch_integral_energy_is_safe(
    sec: object, torch_module: object
) -> tuple[object, object] | None:
    """Check whether float64 integral prefixes stay within numeric bounds.

    The check is performed independently for every bounded search chip. A
    failed or unavailable reduction deliberately selects the FFT fallback for
    the complete batch.  On success, the per-lane absolute error bound is
    returned with the centered-chip maximum for validation of the resulting
    local-energy surface.
    """
    try:
        _, height, width = sec.shape
        max_abs = torch_module.amax(torch_module.abs(sec), dim=(-2, -1))
        prefix_energy = max_abs.square() * (height * width)
        tile_error = (
            8.0
            * torch_module.finfo(sec.dtype).eps
            * (height + width + 2)
            * prefix_energy
        )
        tolerance = 1e-10 + 1e-12 * prefix_energy
        safe = (
            torch_module.isfinite(max_abs)
            & torch_module.isfinite(prefix_energy)
            & (prefix_energy <= 2**44)
            & torch_module.isfinite(tile_error)
            & (tile_error <= tolerance)
        )
        return (tile_error, max_abs) if bool(torch_module.all(safe).item()) else None
    except Exception:
        return None


def _torch_local_energy_fft(
    sec: object,
    ref: object,
    *,
    fft_height: int,
    fft_width: int,
    window_az: int,
    window_rg: int,
    search_az: int,
    search_rg: int,
    torch_module: object,
) -> object:
    """Compute local secondary energy with the legacy same-device FFT path."""
    ones = torch_module.ones_like(ref)
    f_ones = torch_module.fft.rfft2(
        torch_module.flip(ones, dims=(-2, -1)), s=(fft_height, fft_width)
    )
    f_sec_sq = torch_module.fft.rfft2(sec * sec, s=(fft_height, fft_width))
    energy_full = torch_module.fft.irfft2(f_sec_sq * f_ones, s=(fft_height, fft_width))
    row_start = window_az - 1
    col_start = window_rg - 1
    return energy_full[
        :,
        row_start : row_start + 2 * search_az + 1,
        col_start : col_start + 2 * search_rg + 1,
    ]


def _torch_ampcor_workspace_bytes(
    *,
    window_az: int,
    window_rg: int,
    search_az: int,
    search_rg: int,
    batch_size: int,
) -> int:
    """Estimate the conservative Torch NCC packet with a two-times margin.

    The packet includes float64 inputs, correlation FFT tensors, the
    integral-image temporaries, the legacy FFT-energy fallback packet, and
    retained energy/NCC surfaces.
    """
    search_height = window_az + 2 * search_az
    search_width = window_rg + 2 * search_rg
    fft_height = 2 ** int(np.ceil(np.log2(search_height + window_az - 1)))
    fft_width = 2 ** int(np.ceil(np.log2(search_width + window_rg - 1)))
    fft_pixels = fft_height * fft_width
    spectrum_bytes = fft_height * (fft_width // 2 + 1) * 16
    fft_real_bytes = fft_pixels * 8
    reference_bytes = window_az * window_rg * 8
    search_bytes = search_height * search_width * 8
    surface_bytes = (2 * search_az + 1) * (2 * search_rg + 1) * 8

    # Keep the major live allocations explicit.  The correlation transform
    # retains both input spectra, their product, and the inverse real surface.
    input_bytes = reference_bytes + search_bytes
    correlation_fft_bytes = 3 * spectrum_bytes + fft_real_bytes

    # Integral energy constructs sec_sq, both cumsum results, and both padded
    # cat results before the energy/NCC surfaces are reduced.
    sec_sq_bytes = search_bytes
    row_cumsum_bytes = search_bytes
    row_cat_bytes = search_height * (search_width + 1) * 8
    integral_cumsum_bytes = row_cat_bytes
    integral_cat_bytes = (search_height + 1) * (search_width + 1) * 8
    energy_ncc_bytes = 2 * surface_bytes
    integral_energy_bytes = (
        sec_sq_bytes
        + row_cumsum_bytes
        + row_cat_bytes
        + integral_cumsum_bytes
        + integral_cat_bytes
        + energy_ncc_bytes
    )

    # The boundary reference reruns the complete existing device batch with
    # FFT energy while the original batch inputs/outputs remain live. Admission
    # must cover the sequential integral packet and this full FFT packet.
    fft_reference_energy_bytes = (
        sec_sq_bytes + 3 * spectrum_bytes + fft_real_bytes + surface_bytes
    )
    energy_packet_bytes = max(integral_energy_bytes, fft_reference_energy_bytes)
    per_batch_bytes = input_bytes + correlation_fft_bytes + energy_packet_bytes
    return 2 * batch_size * per_batch_bytes


def _torch_patch_ncc_batch(
    ref_windows: object,
    sec_searches: object,
    *,
    search_az: int,
    search_rg: int,
    subpixel: bool,
    force_fft_energy: bool = False,
    energy_candidate: AmpcorEnergyCandidate | None = None,
    fallback_candidate: AmpcorEnergyCandidate | None = None,
) -> tuple[object, object, object]:
    """Evaluate a batch of Ampcor NCC surfaces with Torch float64 math."""
    import torch

    count, window_az, window_rg = ref_windows.shape
    search_height, search_width = sec_searches.shape[-2:]
    fft_height = 2 ** int(np.ceil(np.log2(search_height + window_az - 1)))
    fft_width = 2 ** int(np.ceil(np.log2(search_width + window_rg - 1)))
    ref = ref_windows.to(dtype=torch.float64)
    sec = sec_searches.to(dtype=torch.float64)
    ref = ref - ref.mean(dim=(-2, -1), keepdim=True)
    sec = sec - sec.mean(dim=(-2, -1), keepdim=True)
    ref_norm = torch.linalg.vector_norm(ref, dim=(-2, -1), keepdim=True)
    ref = ref / torch.where(ref_norm < 1e-12, torch.ones_like(ref_norm), ref_norm)

    f_sec = torch.fft.rfft2(sec, s=(fft_height, fft_width))
    f_ref = torch.fft.rfft2(torch.flip(ref, dims=(-2, -1)), s=(fft_height, fft_width))
    corr_full = torch.fft.irfft2(f_sec * f_ref, s=(fft_height, fft_width))
    row_start = window_az - 1
    col_start = window_rg - 1
    corr = corr_full[
        :,
        row_start : row_start + 2 * search_az + 1,
        col_start : col_start + 2 * search_rg + 1,
    ]
    precheck = None if force_fft_energy else _torch_integral_energy_is_safe(sec, torch)
    candidate_energy: object | None = None
    if (
        precheck is not None
        and energy_candidate is not None
        and energy_candidate.window_shape == (window_az, window_rg)
        and energy_candidate.device == str(sec.device)
    ):
        try:
            candidate_energy = energy_candidate.execute(sec)
        except AmpcorCandidateError:
            if fallback_candidate is None:
                raise
            candidate_energy = fallback_candidate.execute(sec)
    if candidate_energy is not None:
        energy = candidate_energy
    elif precheck is not None:
        tile_error, global_max_abs = precheck
        # Each valid lag selects one rectangular window from ``sec``.  A
        # padded float64 integral image gives all local energies directly.
        sec_sq = sec * sec
        row_cumulative = torch.cumsum(sec_sq, dim=-1)
        leading_column = torch.zeros(
            (count, search_height, 1), dtype=torch.float64, device=sec.device
        )
        row_cumulative = torch.cat((leading_column, row_cumulative), dim=-1)
        integral = torch.cumsum(row_cumulative, dim=-2)
        leading_row = torch.zeros(
            (count, 1, search_width + 1), dtype=torch.float64, device=sec.device
        )
        integral = torch.cat((leading_row, integral), dim=-2)
        bottom_right = integral[
            :,
            window_az : window_az + 2 * search_az + 1,
            window_rg : window_rg + 2 * search_rg + 1,
        ]
        top_right = integral[
            :,
            : 2 * search_az + 1,
            window_rg : window_rg + 2 * search_rg + 1,
        ]
        bottom_left = integral[
            :,
            window_az : window_az + 2 * search_az + 1,
            : 2 * search_rg + 1,
        ]
        top_left = integral[:, : 2 * search_az + 1, : 2 * search_rg + 1]
        energy = bottom_right - top_right - bottom_left + top_left
        try:
            lower_bound = torch.clamp(energy - tile_error[:, None, None], min=0.0)
            risk_upper = (
                window_az
                * window_rg
                * global_max_abs[:, None, None].square()
                / torch.clamp(lower_bound, min=torch.finfo(torch.float64).tiny)
            )
            output_safe = (
                torch.isfinite(energy)
                & torch.isfinite(lower_bound)
                & (energy >= 0.0)
                & (risk_upper <= 1e12)
            )
            integral_safe = bool(torch.all(output_safe).item())
        except Exception:
            integral_safe = False
        if not integral_safe:
            energy = _torch_local_energy_fft(
                sec,
                ref,
                fft_height=fft_height,
                fft_width=fft_width,
                window_az=window_az,
                window_rg=window_rg,
                search_az=search_az,
                search_rg=search_rg,
                torch_module=torch,
            )
    else:
        # A failed or unavailable lane reduction is batch-fatal for the
        # integral path; use the previous same-device FFT energy for all lanes.
        energy = _torch_local_energy_fft(
            sec,
            ref,
            fft_height=fft_height,
            fft_width=fft_width,
            window_az=window_az,
            window_rg=window_rg,
            search_az=search_az,
            search_rg=search_rg,
            torch_module=torch,
        )
    ncc = corr / torch.sqrt(torch.clamp(energy, min=1e-12))
    surface_width = 2 * search_rg + 1
    peak_flat = torch.argmax(ncc.reshape(count, -1), dim=1)
    peak_az = torch.div(peak_flat, surface_width, rounding_mode="floor")
    peak_rg = peak_flat.remainder(surface_width)
    peak_value = ncc.reshape(count, -1).gather(1, peak_flat[:, None]).squeeze(1)
    rows = torch.arange(2 * search_az + 1, device=ncc.device)[None, :, None]
    columns = torch.arange(2 * search_rg + 1, device=ncc.device)[None, None, :]
    near_peak = (
        (rows >= peak_az[:, None, None] - 1)
        & (rows <= peak_az[:, None, None] + 1)
        & (columns >= peak_rg[:, None, None] - 1)
        & (columns <= peak_rg[:, None, None] + 1)
    )
    sidelobe = torch.where(near_peak, torch.full_like(ncc, torch.nan), ncc.abs())
    sidelobe_mean = torch.nanmean(sidelobe, dim=(-2, -1))
    snr = torch.where(
        torch.isfinite(sidelobe_mean) & (sidelobe_mean > 0),
        peak_value / sidelobe_mean,
        torch.full_like(peak_value, torch.nan),
    )
    az_shift = peak_az.to(torch.float64) - search_az
    rg_shift = peak_rg.to(torch.float64) - search_rg
    if subpixel:
        interior = (
            (peak_az > 0)
            & (peak_az < 2 * search_az)
            & (peak_rg > 0)
            & (peak_rg < 2 * search_rg)
        )
        safe_az = peak_az.clamp(1, 2 * search_az - 1)
        safe_rg = peak_rg.clamp(1, 2 * search_rg - 1)
        indices = torch.arange(count, device=ncc.device)
        az_left = ncc[indices, safe_az - 1, safe_rg]
        az_center = ncc[indices, safe_az, safe_rg]
        az_right = ncc[indices, safe_az + 1, safe_rg]
        rg_left = ncc[indices, safe_az, safe_rg - 1]
        rg_right = ncc[indices, safe_az, safe_rg + 1]
        az_denom = az_left - 2 * az_center + az_right
        rg_denom = rg_left - 2 * az_center + rg_right
        az_sub = torch.where(
            az_denom.abs() < 1e-12,
            torch.zeros_like(az_denom),
            0.5 * (az_left - az_right) / az_denom,
        )
        rg_sub = torch.where(
            rg_denom.abs() < 1e-12,
            torch.zeros_like(rg_denom),
            0.5 * (rg_left - rg_right) / rg_denom,
        )
        az_shift += torch.where(interior, az_sub, torch.zeros_like(az_sub))
        rg_shift += torch.where(interior, rg_sub, torch.zeros_like(rg_sub))
    return rg_shift, az_shift, snr


def _torch_cpu_median(chunks: list[object], torch_module: object) -> float:
    """Return a NumPy-compatible median from CPU Torch tensor chunks.

    Parameters
    ----------
    chunks : list[object]
        Non-empty one-dimensional CPU Torch tensors containing finite values.
    torch_module : object
        Imported Torch module, accepted as an object to keep the dependency
        lazy at module import time.

    Returns
    -------
    float
        Scalar Python median. Even-sized inputs use the linear average of the
        two middle sorted values, matching :func:`numpy.median`.

    Raises
    ------
    ValueError
        If ``chunks`` is empty.

    """
    if not chunks:
        message = "Torch median requires at least one value"
        logger.error(message)
        raise ValueError(message)
    values = torch_module.cat(chunks).to(dtype=torch_module.float64)
    ordered = torch_module.sort(values).values
    count = int(ordered.numel())
    middle = count // 2
    if count % 2:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2.0
    return float(median.item())


def _ampcor_apply_boundary_oracle(
    ref_tensor: object,
    sec_tensor: object,
    d_rg: object,
    d_az: object,
    snr: object,
    *,
    search_az: int,
    search_rg: int,
    subpixel: bool,
    snr_threshold: float,
    max_abs_residual: float,
    torch_module: object,
) -> tuple[object, object, object]:
    """Recompute only boundary-near patches with the Torch FFT reference.

    The ordinary Torch batch remains the fast path. A patch close to a cull
    boundary is recomputed from the same materialized magnitude windows on
    the batch device with the reference FFT energy path. Strict inclusive
    comparisons then decide its membership and the reference values feed the
    final medians.
    """
    boundary = _ampcor_boundary_mask_torch(
        snr,
        d_rg,
        d_az,
        snr_threshold=snr_threshold,
        max_abs_residual=max_abs_residual,
        torch_module=torch_module,
    )
    if not bool(torch_module.any(boundary).item()):
        return d_rg, d_az, snr
    d_rg = d_rg.clone()
    d_az = d_az.clone()
    snr = snr.clone()
    oracle_rg, oracle_az, oracle_snr = _torch_patch_ncc_batch(
        ref_tensor,
        sec_tensor,
        search_az=search_az,
        search_rg=search_rg,
        subpixel=subpixel,
        force_fft_energy=True,
    )
    d_rg[boundary] = oracle_rg[boundary]
    d_az[boundary] = oracle_az[boundary]
    snr[boundary] = oracle_snr[boundary]
    return d_rg, d_az, snr


def _estimate_patch_amplitude_shift_torch(
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    window_az: int,
    window_rg: int,
    search_az: int,
    search_rg: int,
    n_az: int,
    n_rg: int,
    snr_threshold: float,
    max_abs_residual: float,
    margin_rg: int,
    margin_az: int | None,
    subpixel: bool,
    batch_size: int,
    max_workspace_bytes: int,
    device: Literal["auto", "cpu", "cuda"],
    secondary_shift: tuple[int, int],
    energy_candidate: AmpcorEnergyCandidate,
    fallback_candidate: AmpcorEnergyCandidate | None = None,
) -> PatchAmplitudeShiftResult:
    """Run bounded Torch batches for the opt-in Ampcor executor.

    Surviving batch outputs are copied to CPU Torch tensors before device
    cleanup. Final medians are reduced in Torch with NumPy-compatible
    odd/even semantics, and only scalar Python values are published.
    """
    try:
        import torch
    except ImportError as error:
        message = "Torch Ampcor execution requires torch"
        logger.exception(message)
        raise ImportError(message) from error
    resolved_device = _canonical_torch_device("cpu" if device == "auto" else device)
    _validate_torch_ampcor_runtime(resolved_device, torch)
    if resolved_device.type == "mps":
        message = "MPS is not a qualified Ampcor device; use CPU or CUDA"
        logger.error(message)
        reject_invalid_state(message)

    height, width = reference.shape
    if height > _TORCH_AMPCOR_MAX_INPUT_DIM or width > _TORCH_AMPCOR_MAX_INPUT_DIM:
        reject_invalid_state(
            "Ampcor input dimensions exceed the admitted limit "
            f"({_TORCH_AMPCOR_MAX_INPUT_DIM})"
        )
    search_height = window_az + 2 * search_az
    search_width = window_rg + 2 * search_rg
    _validate_torch_ampcor_shape(
        window_az=window_az,
        window_rg=window_rg,
        search_az=search_az,
        search_rg=search_rg,
        device_type=resolved_device.type,
    )
    if (
        window_az > height
        or window_rg > width
        or search_height > _TORCH_AMPCOR_MAX_PATCH_DIM
        or search_width > _TORCH_AMPCOR_MAX_PATCH_DIM
    ):
        reject_invalid_state(
            "Ampcor window/search dimensions exceed the admitted bounds"
        )
    planned_workspace = _torch_ampcor_workspace_bytes(
        window_az=window_az,
        window_rg=window_rg,
        search_az=search_az,
        search_rg=search_rg,
        batch_size=batch_size,
    )
    if energy_candidate.backend == "native":
        planned_workspace = max(
            planned_workspace,
            native_workspace_bytes(
                (batch_size, search_height, search_width),
                (window_az, window_rg),
            ),
        )
    planned_workspace = max(planned_workspace, energy_candidate.workspace_bytes)
    if planned_workspace > max_workspace_bytes:
        message = (
            f"Ampcor batch workspace {planned_workspace} bytes exceeds the "
            f"cooperative estimate limit {max_workspace_bytes} bytes"
        )
        logger.error(message)
        raise ValueError(message)
    half_az = window_az // 2
    half_rg = window_rg // 2
    margin_a = half_az + search_az + 1 if margin_az is None else int(margin_az)
    margin_r = int(margin_rg)
    if width < 2 * margin_r + window_rg + 2 * search_rg + 2:
        margin_r = half_rg + search_rg + 1
    if height < 2 * margin_a + window_az + 2 * search_az + 2:
        margin_a = half_az + search_az + 1
    az0, az1 = margin_a, height - margin_a
    rg0, rg1 = margin_r, width - margin_r
    if az1 <= az0 or rg1 <= rg0:
        return PatchAmplitudeShiftResult(0.0, 0.0, 0, 0.0, 0)

    az_centres = np.linspace(az0, az1 - 1, num=n_az, dtype=np.int64)
    rg_centres = np.linspace(rg0, rg1 - 1, num=n_rg, dtype=np.int64)
    total_patches = int(az_centres.size) * int(rg_centres.size)
    if energy_candidate.backend == "compile" and total_patches % batch_size:
        if fallback_candidate is None:
            message = (
                "Ampcor compile candidate requires a full final batch; "
                "choose a batch_size that divides the patch grid"
            )
            raise AmpcorCandidateError(message)
        # TorchInductor was prepared with a fixed shape.  Auto mode may use
        # the already-prepared same-device eager candidate for this workload;
        # it must not trigger a new compile from the dispatch path.
        energy_candidate = fallback_candidate
        fallback_candidate = None
    ref_windows: list[np.ndarray] = []
    sec_searches: list[np.ndarray] = []
    n_attempted = 0
    # Keep surviving values on CPU as Torch tensors so device allocations stay
    # bounded by each batch and the final reduction does not re-enter NumPy.
    range_shifts: list[object] = []
    azimuth_shifts: list[object] = []
    snr_values: list[object] = []
    n_valid = 0
    device_key = _torch_ampcor_admission_key(resolved_device, torch)

    def consume_batch() -> None:
        """Score and cull one materialized batch."""
        nonlocal n_valid, ref_windows, sec_searches
        if not ref_windows:
            return
        with _admit_torch_ampcor_workspace(
            device_key,
            planned_workspace,
            max_workspace_bytes,
        ):
            ref_tensor: object | None = None
            sec_tensor: object | None = None
            d_rg: object | None = None
            d_az: object | None = None
            snr: object | None = None
            valid: object | None = None
            primary_error: BaseException | None = None
            primary_traceback = None
            try:
                ref_tensor = torch.from_numpy(np.stack(ref_windows)).to(resolved_device)
                sec_tensor = torch.from_numpy(np.stack(sec_searches)).to(
                    resolved_device
                )
                d_rg, d_az, snr = _torch_patch_ncc_batch(
                    ref_tensor,
                    sec_tensor,
                    search_az=search_az,
                    search_rg=search_rg,
                    subpixel=subpixel,
                    energy_candidate=energy_candidate,
                    fallback_candidate=fallback_candidate,
                )
                d_rg, d_az, snr = _ampcor_apply_boundary_oracle(
                    ref_tensor,
                    sec_tensor,
                    d_rg,
                    d_az,
                    snr,
                    search_az=search_az,
                    search_rg=search_rg,
                    subpixel=subpixel,
                    snr_threshold=snr_threshold,
                    max_abs_residual=max_abs_residual,
                    torch_module=torch,
                )
                valid = _ampcor_cull_mask_torch(
                    snr,
                    d_rg,
                    d_az,
                    snr_threshold=snr_threshold,
                    max_abs_residual=max_abs_residual,
                    torch_module=torch,
                )
                valid_count = int(valid.sum().item())
                if valid_count:
                    n_valid += valid_count
                    range_shifts.append(d_rg[valid].detach().to(device="cpu"))
                    azimuth_shifts.append(d_az[valid].detach().to(device="cpu"))
                    snr_values.append(snr[valid].detach().to(device="cpu"))
            except BaseException as error:
                primary_error = error
                primary_traceback = error.__traceback__
            finally:
                # Drop all device tensor references before synchronization and
                # allocator cleanup, while the admission lease is still held.
                ref_tensor = None
                sec_tensor = None
                d_rg = None
                d_az = None
                snr = None
                valid = None
                # Keep admission held until asynchronous work has completed.
                # Driver synchronization/cache calls are best effort: neither
                # can guarantee allocator release across all Torch drivers.
                try:
                    _synchronize_torch_device(resolved_device)
                except BaseException as error:
                    if primary_error is None:
                        primary_error = error
                        primary_traceback = error.__traceback__
                    logger.exception("Ampcor device synchronization failed")
                try:
                    _release_torch_device_cache(resolved_device)
                except BaseException:
                    logger.exception(
                        "Ampcor allocator cache release failed; references were "
                        "still dropped"
                    )
                ref_windows = []
                sec_searches = []
            if primary_error is not None:
                raise primary_error.with_traceback(primary_traceback)

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
            ref_windows.append(_ampcor_magnitude_tile(reference, r0, r1, c0, c1))
            sec_searches.append(
                _ampcor_magnitude_tile(
                    secondary,
                    sr0,
                    sr1,
                    sc0,
                    sc1,
                    cyclic_shift=secondary_shift,
                )
            )
            if len(ref_windows) >= batch_size:
                consume_batch()
    consume_batch()
    if not range_shifts:
        return PatchAmplitudeShiftResult(0.0, 0.0, 0, 0.0, n_attempted)
    return PatchAmplitudeShiftResult(
        range_shift_px=_torch_cpu_median(range_shifts, torch),
        azimuth_shift_px=_torch_cpu_median(azimuth_shifts, torch),
        n_valid=n_valid,
        snr_median=_torch_cpu_median(snr_values, torch),
        n_attempted=n_attempted,
    )


def _ampcor_candidate_shape_matches(
    candidate: AmpcorEnergyCandidate,
    *,
    spatial_shape: tuple[int, int],
    batch_size: int,
) -> bool:
    """Check a prepared candidate's shape contract before dispatch."""
    if candidate.input_shape is None:
        return True
    expected_batch, expected_height, expected_width = candidate.input_shape
    if (expected_height, expected_width) != spatial_shape:
        return False
    if candidate.allow_partial_batch:
        return 0 < batch_size <= expected_batch
    return batch_size == expected_batch


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
    executor: Literal["auto", "numpy", "torch"] = "numpy",
    backend: AmpcorBackend | Literal["auto"] = "auto",
    ampcor_candidate: AmpcorEnergyCandidate | None = None,
    batch_size: int = 32,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    max_workspace_bytes: int = _TORCH_AMPCOR_WORKSPACE_CAP_BYTES,
    secondary_shift: tuple[int, int] = (0, 0),
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
    executor : {"auto", "numpy", "torch"}, optional
        Correlation implementation. ``"numpy"`` is a deprecated compatibility
        spelling; ``"auto"`` and all other values route through bounded Torch
        batches. Public float32/64 and complex64/128 input precision is
        retained through magnitude materialization; tiles and Torch batches
        are float64 for correlation math.
    backend : {"auto", "eager", "compile", "native"}, optional
        Prepared energy backend. In ``"auto"`` mode, missing or failed
        compile/native candidates use same-device Torch eager. Explicit
        compile/native mode fails closed unless its candidate is prepared.
    ampcor_candidate : AmpcorEnergyCandidate, optional
        Candidate prepared explicitly by the Ampcor backend preparation API.
    batch_size : int, optional
        Number of patches materialized in one Torch batch. Default 32.
    device : {"auto", "cpu", "cuda"}, optional
        Requested device. ``"auto"`` selects the bounded Torch CPU lane.
        Explicit CUDA requires availability and never silently falls back.
        MPS is outside the qualified Ampcor contract and is rejected.
    max_workspace_bytes : int, optional
        Cooperative estimated limit for one Torch batch workspace lease. This
        is not a physical device-memory guarantee. Default 256 MiB.
    secondary_shift : tuple[int, int], optional
        Integer production pre-alignment ``(azimuth, range)`` shift. The
        shift is applied while materializing bounded tiles, so no full-image
        secondary copy is allocated. Defaults to ``(0, 0)``.

    Returns
    -------
    PatchAmplitudeShiftResult
        Median residual shifts and diagnostic counts. When no patch survives,
        shifts are 0.0 (keep geometry prior).

    """
    integer_parameters = {
        "window_az": window_az,
        "window_rg": window_rg,
        "search_az": search_az,
        "search_rg": search_rg,
        "n_az": n_az,
        "n_rg": n_rg,
    }
    for name, value in integer_parameters.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            reject_invalid_state(f"Ampcor {name} must be a finite integer")
        if int(value) < 1:
            reject_invalid_state(f"Ampcor {name} must be positive")
    for name, value in {
        "snr_threshold": snr_threshold,
        "max_abs_residual": max_abs_residual,
    }.items():
        _validate_real_scalar(name, value, nonnegative=True)
    if isinstance(margin_rg, bool) or not isinstance(margin_rg, (int, np.integer)):
        reject_invalid_state("Ampcor margin_rg must be a finite integer")
    if margin_rg < 0:
        reject_invalid_state("Ampcor margin_rg must be non-negative")
    if margin_az is not None:
        if isinstance(margin_az, bool) or not isinstance(margin_az, (int, np.integer)):
            reject_invalid_state("Ampcor margin_az must be a finite integer")
        if margin_az < 0:
            reject_invalid_state("Ampcor margin_az must be non-negative")
    executor, device = resolve_ampcor_policy(executor, device)
    if backend not in ("auto", "eager", "compile", "native"):
        reject_invalid_state(f"unsupported Ampcor backend: {backend!r}")
    if executor == "torch" and device.startswith("cuda"):
        _validate_ampcor_accelerator(device)
    planned_patches = int(n_az) * int(n_rg)
    if planned_patches > _TORCH_AMPCOR_MAX_PATCHES:
        reject_invalid_state(
            "Ampcor patch grid exceeds the admitted total-work limit "
            f"({_TORCH_AMPCOR_MAX_PATCHES} patches)"
        )
    if executor == "torch":
        if isinstance(batch_size, bool) or not isinstance(
            batch_size, (int, np.integer)
        ):
            reject_invalid_state("Ampcor batch_size must be a finite integer")
        if batch_size < 1:
            reject_invalid_state("Ampcor batch_size must be positive")
        if isinstance(max_workspace_bytes, bool) or not isinstance(
            max_workspace_bytes, (int, np.integer)
        ):
            reject_invalid_state("Ampcor max_workspace_bytes must be a finite integer")
        if max_workspace_bytes < 1 or (
            max_workspace_bytes > _TORCH_AMPCOR_WORKSPACE_CAP_BYTES
        ):
            reject_invalid_state("Ampcor max_workspace_bytes must be in [1, 256 MiB]")
    if (
        not isinstance(secondary_shift, tuple)
        or len(secondary_shift) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in secondary_shift
        )
    ):
        reject_invalid_state("Ampcor secondary_shift must be a pair of integers")
    secondary_shift = (int(secondary_shift[0]), int(secondary_shift[1]))
    reference, secondary = _validate_ampcor_inputs(
        reference,
        secondary,
        torch_contract=executor == "torch",
        conversion_limit_bytes=(
            int(max_workspace_bytes) if executor == "torch" else None
        ),
    )
    height, width = reference.shape
    if height < 1 or width < 1:
        reject_invalid_state("Ampcor input dimensions must be positive")
    if executor == "torch":
        if height > _TORCH_AMPCOR_MAX_INPUT_DIM or width > _TORCH_AMPCOR_MAX_INPUT_DIM:
            reject_invalid_state(
                "Ampcor input dimensions exceed the admitted limit "
                f"({_TORCH_AMPCOR_MAX_INPUT_DIM})"
            )
        search_height = int(window_az) + 2 * int(search_az)
        search_width = int(window_rg) + 2 * int(search_rg)
        if (
            window_az > height
            or window_rg > width
            or search_height > _TORCH_AMPCOR_MAX_PATCH_DIM
            or search_width > _TORCH_AMPCOR_MAX_PATCH_DIM
        ):
            reject_invalid_state(
                "Ampcor window/search dimensions exceed the admitted bounds"
            )
        patch_elements = int(window_az) * int(window_rg) + search_height * search_width
        expected_input_bytes = _ampcor_input_budget_bytes(
            reference,
            secondary,
            patch_elements=patch_elements,
            batch_capacity=int(batch_size),
        )
        if expected_input_bytes > _TORCH_AMPCOR_MAX_INPUT_BYTES:
            reject_invalid_state(
                "Ampcor input and magnitude-tile workspace exceeds the admitted limit "
                f"({_TORCH_AMPCOR_MAX_INPUT_BYTES} bytes)"
            )
        resolved_torch_device = (
            "cpu" if device == "cpu" else str(_canonical_torch_device(device))
        )
        eager_candidate = eager_ampcor_candidate(
            device=resolved_torch_device,
            window_shape=(window_az, window_rg),
        )
        candidate_matches = (
            ampcor_candidate is not None
            and backend in ("auto", ampcor_candidate.backend)
            and ampcor_candidate.window_shape == (window_az, window_rg)
            and ampcor_candidate.device == resolved_torch_device
            and _ampcor_candidate_shape_matches(
                ampcor_candidate,
                spatial_shape=(search_height, search_width),
                batch_size=int(batch_size),
            )
        )
        if backend in ("compile", "native") and not candidate_matches:
            reject_invalid_state(
                f"Ampcor backend={backend!r} requires a prepared same-device candidate"
            )
        selected_candidate = ampcor_candidate if candidate_matches else eager_candidate
        return _estimate_patch_amplitude_shift_torch(
            reference,
            secondary,
            window_az=window_az,
            window_rg=window_rg,
            search_az=search_az,
            search_rg=search_rg,
            n_az=n_az,
            n_rg=n_rg,
            snr_threshold=snr_threshold,
            max_abs_residual=max_abs_residual,
            margin_rg=margin_rg,
            margin_az=margin_az,
            subpixel=subpixel,
            batch_size=int(batch_size),
            max_workspace_bytes=int(max_workspace_bytes),
            device=device,
            secondary_shift=secondary_shift,
            energy_candidate=selected_candidate,
            fallback_candidate=(
                eager_candidate
                if backend == "auto" and selected_candidate.backend != "eager"
                else None
            ),
        )
    height, width = reference.shape
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
                _ampcor_magnitude_tile(reference, r0, r1, c0, c1),
                _ampcor_magnitude_tile(
                    secondary,
                    sr0,
                    sr1,
                    sc0,
                    sc1,
                    cyclic_shift=secondary_shift,
                ),
                search_az=search_az,
                search_rg=search_rg,
                subpixel=subpixel,
            )
            if result is None:
                continue
            d_rg, d_az, snr = result
            if not bool(
                _ampcor_cull_mask_numpy(
                    snr,
                    d_rg,
                    d_az,
                    snr_threshold=snr_threshold,
                    max_abs_residual=max_abs_residual,
                )
            ):
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
