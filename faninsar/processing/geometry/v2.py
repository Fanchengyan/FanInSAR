# ruff: noqa: EM101, EM102, TRY003, SIM105

"""Public contracts for the v2 radar/geographic geometry backend.

This module contains the value objects shared by preparation, native ABI
adapters, and result publication.  It intentionally has no native or torch
dependencies: validation must finish before a native pointer is obtained.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

INT32_MAX = np.iinfo(np.int32).max


class GeometryValidationError(ValueError):
    """Raised when a v2 geometry contract is malformed."""


class Operation(StrEnum):
    """Supported geometry transform operations."""

    GEO2RDR = "geo2rdr"
    RDR2GEO = "rdr2geo"
    geo2rdr = "geo2rdr"
    rdr2geo = "rdr2geo"


def _operation(value: Operation | str) -> Operation:
    """Normalize an operation value and report malformed tags."""
    try:
        return value if isinstance(value, Operation) else Operation(value)
    except (TypeError, ValueError) as error:
        message = "operation must be 'geo2rdr' or 'rdr2geo'"
        logger.exception(message)
        raise GeometryValidationError(message) from error


def _finite_positive(value: float, name: str) -> float:
    """Validate a finite, strictly positive scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        message = f"{name} must be a finite positive number"
        logger.error(message)
        raise GeometryValidationError(message)
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        message = f"{name} must be a finite positive number"
        logger.error(message)
        raise GeometryValidationError(message)
    return result


def _nonnegative_int(value: int, name: str) -> int:
    """Validate a Python integer without accepting booleans or truncation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        message = f"{name} must be an integer"
        logger.error(message)
        raise GeometryValidationError(message)
    result = int(value)
    if result < 0:
        message = f"{name} must be non-negative"
        logger.error(message)
        raise GeometryValidationError(message)
    return result


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Operation solver limits and tolerances.

    Parameters
    ----------
    max_iter : int, optional
        Mandatory primary iteration budget.  It must be at least one.
    extra_iter : int, optional
        Additional attempts allowed after the primary budget.
    range_tolerance_m : float, optional
        Geo-to-radar range residual tolerance in metres.
    doppler_tolerance_hz : float, optional
        Doppler residual tolerance in hertz.
    slant_range_tolerance_m : float, optional
        Radar-to-geo signed slant-range tolerance in metres.

    """

    max_iter: int = 20
    extra_iter: int = 0
    range_tolerance_m: float = 0.01
    doppler_tolerance_hz: float = 0.1
    slant_range_tolerance_m: float = 0.01

    def __post_init__(self) -> None:
        """Validate limits before a solver or cache key can use them."""
        max_iter = _nonnegative_int(self.max_iter, "max_iter")
        if max_iter < 1:
            message = "max_iter must be at least 1"
            logger.error(message)
            raise GeometryValidationError(message)
        extra_iter = _nonnegative_int(self.extra_iter, "extra_iter")
        if max_iter + extra_iter > INT32_MAX:
            message = "max_iter + extra_iter exceeds int32 iteration capacity"
            logger.error(message)
            raise GeometryValidationError(message)
        object.__setattr__(self, "max_iter", max_iter)
        object.__setattr__(self, "extra_iter", extra_iter)
        for name in (
            "range_tolerance_m",
            "doppler_tolerance_hz",
            "slant_range_tolerance_m",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))

    @property
    def budget(self) -> int:
        """Return the checked total number of permitted attempts."""
        return self.max_iter + self.extra_iter

    def for_operation(self, operation: Operation | str) -> dict[str, float | int]:
        """Return the exact operation-specific solver settings."""
        op = _operation(operation)
        common: dict[str, float | int] = {
            "max_iter": self.max_iter,
            "extra_iter": self.extra_iter,
            "doppler_tolerance_hz": self.doppler_tolerance_hz,
        }
        if op is Operation.GEO2RDR:
            common["range_tolerance_m"] = self.range_tolerance_m
        else:
            common["slant_range_tolerance_m"] = self.slant_range_tolerance_m
        return common


@dataclass(frozen=True, slots=True)
class OperationSettings:
    """Immutable operation tag paired with solver settings."""

    operation: Operation
    solver: SolverSettings = field(default_factory=SolverSettings)

    def __post_init__(self) -> None:
        """Normalize operation and reject an invalid solver object."""
        object.__setattr__(self, "operation", _operation(self.operation))
        if not isinstance(self.solver, SolverSettings):
            message = "solver must be a SolverSettings instance"
            logger.error(message)
            raise GeometryValidationError(message)

    @property
    def max_iter(self) -> int:
        """Return the primary iteration limit."""
        return self.solver.max_iter

    @property
    def extra_iter(self) -> int:
        """Return the extra iteration limit."""
        return self.solver.extra_iter

    @property
    def budget(self) -> int:
        """Return the checked total iteration budget."""
        return self.solver.budget

    def canonical(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible scientific projection."""
        return {
            "operation": self.operation.value,
            "solver": self.solver.for_operation(self.operation),
        }

    @classmethod
    def geo2rdr(
        cls,
        *,
        max_iter: int = 20,
        extra_iter: int = 0,
        range_tolerance_m: float = 0.01,
        doppler_tolerance_hz: float = 0.1,
    ) -> OperationSettings:
        """Build operation settings with only geo2rdr tolerances."""
        solver = SolverSettings(
            max_iter=max_iter,
            extra_iter=extra_iter,
            range_tolerance_m=range_tolerance_m,
            doppler_tolerance_hz=doppler_tolerance_hz,
        )
        return cls(Operation.GEO2RDR, solver)

    @classmethod
    def rdr2geo(
        cls,
        *,
        max_iter: int = 20,
        extra_iter: int = 0,
        slant_range_tolerance_m: float = 0.01,
        doppler_tolerance_hz: float = 0.1,
    ) -> OperationSettings:
        """Build operation settings with rdr2geo tolerances."""
        solver = SolverSettings(
            max_iter=max_iter,
            extra_iter=extra_iter,
            slant_range_tolerance_m=slant_range_tolerance_m,
            doppler_tolerance_hz=doppler_tolerance_hz,
        )
        return cls(Operation.RDR2GEO, solver)


@dataclass(frozen=True, slots=True)
class DeviceKey:
    """Tagged physical device identity used in scientific cache keys."""

    kind: Literal["cpu", "cuda"]
    physical_uuid: str | None = None
    mig_uuid: str | None = None

    def __post_init__(self) -> None:
        """Enforce CPU/CUDA tags and their identity fields."""
        if self.kind not in ("cpu", "cuda"):
            message = "device kind must be 'cpu' or 'cuda'"
            logger.error(message)
            raise GeometryValidationError(message)
        if self.kind == "cpu" and (
            self.physical_uuid is not None or self.mig_uuid is not None
        ):
            message = "CPU device identity cannot contain CUDA UUIDs"
            logger.error(message)
            raise GeometryValidationError(message)
        if self.kind == "cuda" and not self.physical_uuid:
            message = "CUDA device identity requires a physical UUID"
            logger.error(message)
            raise GeometryValidationError(message)

    @classmethod
    def cpu(cls) -> DeviceKey:
        """Return the canonical host CPU tag."""
        return cls("cpu")

    @classmethod
    def cuda(cls, physical_uuid: str, mig_uuid: str | None = None) -> DeviceKey:
        """Build a CUDA tag bound to a physical and optional MIG UUID."""
        return cls("cuda", physical_uuid=physical_uuid, mig_uuid=mig_uuid)

    def canonical(self) -> dict[str, str | None]:
        """Return a stable tagged representation."""
        return {
            "kind": self.kind,
            "physical_uuid": self.physical_uuid,
            "mig_uuid": self.mig_uuid,
        }

    @property
    def physical_id(self) -> str | None:
        """Return the physical UUID under the generic identity name."""
        return self.physical_uuid

    @property
    def mig_id(self) -> str | None:
        """Return the optional MIG UUID under the generic identity name."""
        return self.mig_uuid


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Execution properties that can change numerical/performance behavior."""

    device: DeviceKey = field(default_factory=DeviceKey.cpu)
    openmp_runtime: str | None = None
    thread_count: int | None = None
    affinity: str | None = None
    launch_config: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate profile fields and keep them immutable."""
        if not isinstance(self.device, DeviceKey):
            message = "execution profile device must be a DeviceKey"
            logger.error(message)
            raise GeometryValidationError(message)
        if self.thread_count is not None:
            count = _nonnegative_int(self.thread_count, "thread_count")
            if count < 1:
                message = "thread_count must be at least 1"
                logger.error(message)
                raise GeometryValidationError(message)
            object.__setattr__(self, "thread_count", count)
        if self.launch_config is not None:
            config = tuple(
                _nonnegative_int(v, "launch_config value") for v in self.launch_config
            )
            if not config:
                message = "launch_config must not be empty"
                logger.error(message)
                raise GeometryValidationError(message)
            object.__setattr__(self, "launch_config", config)
        if self.device.kind == "cuda" and self.openmp_runtime is not None:
            message = "CUDA execution profiles cannot declare an OpenMP runtime"
            logger.error(message)
            raise GeometryValidationError(message)

    @classmethod
    def cpu(
        cls,
        *,
        openmp_runtime: str | None = None,
        thread_count: int | None = None,
        affinity: str | None = None,
    ) -> ExecutionProfile:
        """Build a CPU execution profile."""
        return cls(DeviceKey.cpu(), openmp_runtime, thread_count, affinity)

    @classmethod
    def cuda(
        cls,
        physical_uuid: str,
        *,
        mig_uuid: str | None = None,
        launch_config: Sequence[int] | None = None,
    ) -> ExecutionProfile:
        """Build a CUDA execution profile."""
        config = None if launch_config is None else tuple(launch_config)
        return cls(DeviceKey.cuda(physical_uuid, mig_uuid), launch_config=config)

    def canonical(self) -> dict[str, Any]:
        """Return a stable profile representation."""
        return {
            "device": self.device.canonical(),
            "openmp_runtime": self.openmp_runtime,
            "thread_count": self.thread_count,
            "affinity": self.affinity,
            "launch_config": self.launch_config,
        }

    @property
    def threads(self) -> int | None:
        """Return the configured CPU thread count."""
        return self.thread_count


def _canonical(value: Any) -> Any:
    """Convert key values into deterministic JSON-compatible primitives."""
    if isinstance(value, (Operation, StrEnum)):
        return value.value
    if isinstance(
        value, (DeviceKey, ExecutionProfile, OperationSettings, SolverSettings)
    ):
        return _canonical(
            value.canonical() if hasattr(value, "canonical") else value.__dict__
        )
    if isinstance(value, Mapping):
        return {
            str(k): _canonical(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class CandidateKey:
    """Complete immutable identity for a prepared geometry candidate."""

    operation: Operation
    backend: Literal["native", "compile"]
    device: DeviceKey
    dtype: str = "float64"
    shape: tuple[int, ...] = ()
    solver: SolverSettings = field(default_factory=SolverSettings)
    orbit_digest: str = ""
    dem_digest: str = ""
    model_digest: str = ""
    source_digest: str = ""
    toolchain_digest: str = ""
    runtime_digest: str = ""
    artifact_digest: str = ""
    abi_digest: str = ""
    support_contract_digest: str = ""
    profile: ExecutionProfile | None = None

    def __post_init__(self) -> None:
        """Validate tags and normalize shape/profile identity."""
        object.__setattr__(self, "operation", _operation(self.operation))
        if self.backend not in ("native", "compile"):
            message = "prepared backend must be 'native' or 'compile'"
            logger.error(message)
            raise GeometryValidationError(message)
        if not isinstance(self.device, DeviceKey) or not isinstance(
            self.solver, SolverSettings
        ):
            message = "candidate device and solver must use v2 value objects"
            logger.error(message)
            raise GeometryValidationError(message)
        normalized_shape = tuple(
            _nonnegative_int(v, "shape dimension") for v in self.shape
        )
        if any(v == 0 for v in normalized_shape):
            message = "candidate request shape dimensions must be positive"
            logger.error(message)
            raise GeometryValidationError(message)
        object.__setattr__(self, "shape", normalized_shape)
        if self.profile is None:
            object.__setattr__(self, "profile", ExecutionProfile(self.device))
        elif self.profile.device != self.device:
            message = "candidate device and execution profile device must match"
            logger.error(message)
            raise GeometryValidationError(message)

    def canonical(self) -> dict[str, Any]:
        """Return the full key as deterministic JSON-compatible data."""
        return {
            "operation": self.operation.value,
            "backend": self.backend,
            "device": self.device.canonical(),
            "dtype": self.dtype,
            "shape": self.shape,
            "solver": self.solver.for_operation(self.operation),
            "orbit_digest": self.orbit_digest,
            "dem_digest": self.dem_digest,
            "model_digest": self.model_digest,
            "source_digest": self.source_digest,
            "toolchain_digest": self.toolchain_digest,
            "runtime_digest": self.runtime_digest,
            "artifact_digest": self.artifact_digest,
            "abi_digest": self.abi_digest,
            "support_contract_digest": self.support_contract_digest,
            "profile": self.profile.canonical() if self.profile else None,
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 identity digest of the complete key."""
        payload = json.dumps(
            self.canonical(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def scientific_identity(self) -> tuple[Any, ...]:
        """Return the scientific/profile projection used for lookup slots."""
        return (
            self.operation,
            self.device,
            self.dtype,
            self.shape,
            self.solver,
            self.orbit_digest,
            self.dem_digest,
            self.model_digest,
            self.support_contract_digest,
            self.profile,
        )


_RESULT_FIELDS = (
    "latitude_deg",
    "longitude_deg",
    "height_m",
    "range_index",
    "azimuth_index",
    "converged",
    "iterations",
    "decision_residual",
    "final_residual",
    "tolerance",
    "max_iter_exhausted",
    "boundary_rechecked",
    "residual_range_m",
    "residual_doppler_hz",
)
_FLOAT_RESULT_FIELDS = {
    "latitude_deg",
    "longitude_deg",
    "height_m",
    "range_index",
    "azimuth_index",
    "decision_residual",
    "final_residual",
    "tolerance",
    "residual_range_m",
    "residual_doppler_hz",
}
_BOOL_RESULT_FIELDS = {"converged", "max_iter_exhausted", "boundary_rechecked"}


@dataclass(frozen=True, slots=True)
class TransformResultV2:
    """The exact fourteen-field result contract for both transforms."""

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    height_m: np.ndarray
    range_index: np.ndarray
    azimuth_index: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    decision_residual: np.ndarray
    final_residual: np.ndarray
    tolerance: np.ndarray
    max_iter_exhausted: np.ndarray
    boundary_rechecked: np.ndarray
    residual_range_m: np.ndarray
    residual_doppler_hz: np.ndarray

    def __post_init__(self) -> None:
        """Validate exact field names, dtypes, and common shape."""
        shape: tuple[int, ...] | None = None
        for name in _RESULT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                message = f"{name} must be a NumPy array"
                logger.error(message)
                raise GeometryValidationError(message)
            if name in _FLOAT_RESULT_FIELDS and value.dtype != np.dtype(np.float64):
                message = f"{name} must have dtype float64"
                logger.error(message)
                raise GeometryValidationError(message)
            if name == "iterations" and value.dtype != np.dtype(np.int32):
                message = "iterations must have dtype int32"
                logger.error(message)
                raise GeometryValidationError(message)
            if name in _BOOL_RESULT_FIELDS and value.dtype != np.dtype(bool):
                message = f"{name} must have dtype bool"
                logger.error(message)
                raise GeometryValidationError(message)
            if shape is None:
                shape = value.shape
            elif value.shape != shape:
                message = "all TransformResultV2 fields must have the same shape"
                logger.error(message)
                raise GeometryValidationError(message)

    @classmethod
    def invalid(
        cls,
        shape: int | Sequence[int],
        *,
        operation: Operation | str = Operation.GEO2RDR,
    ) -> TransformResultV2:
        """Create an all-invalid result with prescribed lane sentinels."""
        _operation(operation)
        normalized_shape = (
            (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        )
        if any(
            not isinstance(dim, (int, np.integer)) or int(dim) < 0
            for dim in normalized_shape
        ):
            message = "result shape must contain non-negative integer dimensions"
            logger.error(message)
            raise GeometryValidationError(message)
        shape_tuple = tuple(int(dim) for dim in normalized_shape)

        def nan() -> np.ndarray:
            """Allocate one floating-point NaN field."""
            return np.full(shape_tuple, np.nan, dtype=np.float64)

        def false() -> np.ndarray:
            """Allocate one false boolean field."""
            return np.zeros(shape_tuple, dtype=bool)

        tolerance = np.full(shape_tuple, np.nan, dtype=np.float64)
        return cls(
            nan(),
            nan(),
            nan(),
            nan(),
            nan(),
            false(),
            np.full(shape_tuple, -1, dtype=np.int32),
            nan(),
            nan(),
            tolerance,
            false(),
            false(),
            nan(),
            nan(),
        )

    @classmethod
    def from_arrays(
        cls,
        fields: Mapping[str, np.ndarray],
        *,
        operation: Operation | str = Operation.GEO2RDR,
        invalid_mask: np.ndarray | None = None,
    ) -> TransformResultV2:
        """Construct and validate a result mapping, applying invalid sentinels."""
        missing = [name for name in _RESULT_FIELDS if name not in fields]
        extra = [name for name in fields if name not in _RESULT_FIELDS]
        if missing or extra:
            message = f"result fields mismatch; missing={missing}, extra={extra}"
            logger.error(message)
            raise GeometryValidationError(message)
        values = {name: fields[name] for name in _RESULT_FIELDS}
        result = cls(**values)
        op = _operation(operation)
        if invalid_mask is None:
            valid_mask = np.ones(result.latitude_deg.shape, dtype=bool)
        else:
            mask = invalid_mask
            if (
                not isinstance(mask, np.ndarray)
                or mask.dtype != np.dtype(bool)
                or mask.shape != result.latitude_deg.shape
            ):
                message = "invalid_mask must be a bool array matching result shape"
                logger.error(message)
                raise GeometryValidationError(message)
            valid_mask = ~mask
        if op is Operation.GEO2RDR and np.any(valid_mask & (result.tolerance != 1.0)):
            message = "valid geo2rdr lanes must publish tolerance=1.0"
            logger.error(message)
            raise GeometryValidationError(message)
        if op is Operation.RDR2GEO and np.any(
            valid_mask & (~np.isfinite(result.tolerance) | (result.tolerance <= 0.0))
        ):
            message = "valid rdr2geo lanes must publish a positive finite tolerance"
            logger.error(message)
            raise GeometryValidationError(message)
        if invalid_mask is None:
            return result
        if not np.any(mask):
            return result
        mutable = {name: getattr(result, name).copy() for name in _RESULT_FIELDS}
        for name in _FLOAT_RESULT_FIELDS:
            mutable[name][mask] = np.nan
        mutable["iterations"][mask] = -1
        for name in _BOOL_RESULT_FIELDS:
            mutable[name][mask] = False
        mutable["tolerance"][mask] = np.nan
        return cls(**mutable)

    @classmethod
    def invalid_lane(
        cls,
        shape: int | Sequence[int] = (),
        *,
        operation: Operation | str = Operation.GEO2RDR,
    ) -> TransformResultV2:
        """Alias for creating one or more prescribed invalid lanes."""
        return cls.invalid(shape, operation=operation)

    invalid_lanes = invalid_lane

    @property
    def fields(self) -> tuple[str, ...]:
        """Return the exact ordered field names."""
        return _RESULT_FIELDS

    def validate(self) -> TransformResultV2:
        """Validate and return this immutable result."""
        self.__post_init__()
        return self


@dataclass(frozen=True, slots=True)
class RawSpan:
    """Owner-backed contiguous CPU array descriptor for a native call."""

    owner: np.ndarray
    array: np.ndarray
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    byte_length: int
    address: int
    device: DeviceKey


@dataclass(frozen=True, slots=True)
class RawTensorSpan:
    """Owner-backed Torch tensor descriptor for a native call."""

    owner: object
    tensor: object
    dtype: object
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    byte_length: int
    address: int
    device: DeviceKey


def validate_tensor_span(
    tensor: object,
    *,
    expected_dtype: object | None = None,
    expected_shape: Sequence[int] | None = None,
    expected_device: DeviceKey | None = None,
    require_finite: bool = False,
    name: str = "tensor",
) -> RawTensorSpan:
    """Validate an owner-backed Torch tensor before native pointer access."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency
        raise GeometryValidationError("Torch is required for tensor spans") from error
    if not isinstance(tensor, torch.Tensor):
        raise GeometryValidationError(f"{name} must be a Torch tensor")
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise GeometryValidationError(f"{name} must have dtype {expected_dtype}")
    if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
        raise GeometryValidationError(
            f"{name} has shape {tuple(tensor.shape)}, expected {tuple(expected_shape)}"
        )
    if not tensor.is_contiguous():
        raise GeometryValidationError(f"{name} must use exact contiguous strides")
    expected_strides = tuple(
        tensor.element_size() * int(np.prod(tensor.shape[index + 1 :]))
        for index in range(tensor.ndim)
    )
    actual_strides = tuple(
        int(value) * tensor.element_size() for value in tensor.stride()
    )
    if actual_strides != expected_strides:
        raise GeometryValidationError(f"{name} must use exact contiguous strides")
    if tensor.device.type == "cpu":
        actual_device = DeviceKey.cpu()
    elif tensor.device.type == "cuda":
        physical_uuid = str(tensor.device)
        try:
            physical_uuid = str(torch.cuda.get_device_properties(tensor.device).uuid)
        except (AttributeError, RuntimeError):
            pass
        actual_device = DeviceKey.cuda(physical_uuid)
    else:
        raise GeometryValidationError(f"{name} must use a CPU or CUDA device")
    if expected_device is not None:
        if actual_device.kind != expected_device.kind:
            raise GeometryValidationError(f"{name} is on the wrong device")
        if expected_device.physical_uuid and (
            actual_device.physical_uuid != expected_device.physical_uuid
        ):
            raise GeometryValidationError(f"{name} has the wrong physical CUDA device")
        if actual_device.mig_uuid != expected_device.mig_uuid:
            raise GeometryValidationError(f"{name} has the wrong MIG device")
    if require_finite and not bool(torch.isfinite(tensor).all().item()):
        raise GeometryValidationError(f"{name} must contain only finite values")
    storage = tensor.untyped_storage()
    address = int(tensor.data_ptr())
    byte_length = int(tensor.numel() * tensor.element_size())
    storage_address = int(storage.data_ptr())
    if address < storage_address or address + byte_length > storage_address + int(
        storage.nbytes()
    ):
        raise GeometryValidationError(f"{name} span is outside its owner storage")
    return RawTensorSpan(
        storage,
        tensor,
        tensor.dtype,
        tuple(tensor.shape),
        actual_strides,
        byte_length,
        address,
        actual_device,
    )


def validate_array_span(
    array: np.ndarray,
    *,
    dtype: np.dtype[Any] | str | None = None,
    expected_dtype: np.dtype[Any] | str | None = None,
    shape: Sequence[int] | None = None,
    expected_shape: Sequence[int] | None = None,
    strides: Sequence[int] | None = None,
    byte_length: int | None = None,
    device: DeviceKey | None = None,
    expected_device: DeviceKey | None = None,
    require_finite: bool = False,
    name: str = "array",
) -> RawSpan:
    """Validate one complete array span before exposing its pointer.

    Only host NumPy arrays are admitted by this first foundation slice.  CUDA
    providers will supply an owner-backed proof in the native adapter layer;
    a NumPy host owner can never be asserted to be a CUDA allocation here.
    """
    if dtype is None:
        dtype = expected_dtype
    elif expected_dtype is not None and np.dtype(dtype) != np.dtype(expected_dtype):
        message = f"{name} received conflicting dtype expectations"
        logger.error(message)
        raise GeometryValidationError(message)
    if dtype is None:
        message = f"{name} requires an expected dtype"
        logger.error(message)
        raise GeometryValidationError(message)
    expected_dtype = np.dtype(dtype)
    if shape is None:
        shape = expected_shape
    elif expected_shape is not None and tuple(shape) != tuple(expected_shape):
        message = f"{name} received conflicting shape expectations"
        logger.error(message)
        raise GeometryValidationError(message)
    if device is None:
        device = expected_device
    elif expected_device is not None and device != expected_device:
        message = f"{name} received conflicting device expectations"
        logger.error(message)
        raise GeometryValidationError(message)
    if not isinstance(array, np.ndarray):
        message = f"{name} must be a NumPy array"
        logger.error(message)
        raise GeometryValidationError(message)
    if array.dtype != expected_dtype:
        message = f"{name} must have dtype {expected_dtype}"
        logger.error(message)
        raise GeometryValidationError(message)
    if array.ndim != len(array.shape) or array.ndim != len(array.strides):
        message = f"{name} rank/shape/strides metadata is inconsistent"
        logger.error(message)
        raise GeometryValidationError(message)
    if any(d <= 0 for d in array.shape):
        message = f"{name} dimensions must be positive"
        logger.error(message)
        raise GeometryValidationError(message)
    if shape is not None and tuple(array.shape) != tuple(shape):
        message = f"{name} has shape {array.shape}, expected {tuple(shape)}"
        logger.error(message)
        raise GeometryValidationError(message)
    expected_strides = tuple(
        array.itemsize * int(np.prod(array.shape[index + 1 :]))
        for index in range(array.ndim)
    )
    if not array.flags.c_contiguous or tuple(array.strides) != expected_strides:
        message = f"{name} must use exact C-order strides"
        logger.error(message)
        raise GeometryValidationError(message)
    if strides is not None and tuple(strides) != tuple(array.strides):
        message = f"{name} has unexpected strides"
        logger.error(message)
        raise GeometryValidationError(message)
    if byte_length is not None:
        if isinstance(byte_length, (bool, np.bool_)) or int(byte_length) != byte_length:
            message = f"{name} byte_length must be an integer"
            logger.error(message)
            raise GeometryValidationError(message)
        if int(byte_length) != int(array.nbytes):
            message = f"{name} has an unexpected byte length"
            logger.error(message)
            raise GeometryValidationError(message)
    if require_finite and not np.all(np.isfinite(array)):
        message = f"{name} must contain only finite values"
        logger.error(message)
        raise GeometryValidationError(message)
    actual_device = device or DeviceKey.cpu()
    if actual_device.kind != "cpu":
        message = f"{name} NumPy owner cannot prove CUDA allocation"
        logger.error(message)
        raise GeometryValidationError(message)
    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    if not owner.flags.c_contiguous:
        message = f"{name} owner must be contiguous host memory"
        logger.error(message)
        raise GeometryValidationError(message)
    # Pointer and extent are deliberately read only after every structural check.
    address = int(array.__array_interface__["data"][0])
    owner_address = int(owner.__array_interface__["data"][0])
    end = address + int(array.nbytes)
    owner_end = owner_address + int(owner.nbytes)
    if address % expected_dtype.itemsize != 0:
        message = f"{name} address is not dtype aligned"
        logger.error(message)
        raise GeometryValidationError(message)
    if address <= 0 or end < address or address < owner_address or end > owner_end:
        message = f"{name} span is outside its owner allocation"
        logger.error(message)
        raise GeometryValidationError(message)
    return RawSpan(
        owner,
        array,
        expected_dtype,
        tuple(array.shape),
        tuple(array.strides),
        int(array.nbytes),
        address,
        actual_device,
    )


def validate_spans(
    spans: Mapping[str, np.ndarray] | Iterable[tuple[str, np.ndarray]],
    *,
    dtype: np.dtype[Any] | str | Mapping[str, np.dtype[Any] | str] = "float64",
    dtypes: Mapping[str, np.dtype[Any] | str] | None = None,
    shapes: Mapping[str, Sequence[int]] | None = None,
    device: DeviceKey | None = None,
    expected_device: DeviceKey | None = None,
) -> dict[str, RawSpan]:
    """Validate a collection of spans through the same ABI boundary."""
    items = spans.items() if isinstance(spans, Mapping) else spans
    validated: dict[str, RawSpan] = {}
    for name, array in items:
        if dtypes is not None and name in dtypes:
            item_dtype = dtypes[name]
        elif isinstance(dtype, Mapping):
            if name not in dtype:
                message = f"missing expected dtype for span {name}"
                logger.error(message)
                raise GeometryValidationError(message)
            item_dtype = dtype[name]
        else:
            item_dtype = dtype
        validated[name] = validate_array_span(
            array,
            dtype=item_dtype,
            shape=None if shapes is None else shapes.get(name),
            device=device or expected_device,
            name=name,
        )
    selected_device = device or expected_device
    if selected_device is not None and any(
        span.device != selected_device for span in validated.values()
    ):
        message = "all spans must share the expected device"
        logger.error(message)
        raise GeometryValidationError(message)
    return validated


# Explicit aliases keep the validator seam discoverable to native adapters.
validate_span = validate_array_span
validate_native_spans = validate_spans
ArraySpan = RawSpan
NativeSpan = RawSpan
validate_input_span = validate_array_span


__all__ = [
    "INT32_MAX",
    "ArraySpan",
    "CandidateKey",
    "DeviceKey",
    "ExecutionProfile",
    "GeometryValidationError",
    "NativeSpan",
    "Operation",
    "OperationSettings",
    "RawSpan",
    "RawTensorSpan",
    "SolverSettings",
    "TransformResultV2",
    "validate_array_span",
    "validate_input_span",
    "validate_native_spans",
    "validate_span",
    "validate_spans",
    "validate_tensor_span",
]
