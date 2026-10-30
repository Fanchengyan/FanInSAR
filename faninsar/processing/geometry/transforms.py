# ruff: noqa: EM101, EM102, TRY003, SIM105
"""Radar geometry value types and public contracts for rdr2geo and geo2rdr.

This module carries the unified geometry contract: value objects shared by
preparation, native ABI adapters, and result publication, plus the radar
geometry model used by both transforms.  It intentionally has no native or
torch dependencies: validation must finish before a native pointer is
obtained.  The NumPy CPU Newton solvers formerly in this module were deleted
under PROPOSAL-0031; production callers use
:mod:`faninsar.processing.geometry.prepare_production`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import reject
from faninsar.processing.geometry.orbit import OrbitInterpolator

if TYPE_CHECKING:
    from faninsar.core.orbit import OrbitMetadata
    from faninsar.processing.geometry.coordinates import RadarGrid

INT32_MAX = np.iinfo(np.int32).max


class GeometryValidationError(ValueError):
    """Raised when a geometry contract is malformed."""


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
        reject(message, GeometryValidationError, cause=error)


def _finite_positive(value: float, name: str) -> float:
    """Validate a finite, strictly positive scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        message = f"{name} must be a finite positive number"
        reject(message, GeometryValidationError)
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        message = f"{name} must be a finite positive number"
        reject(message, GeometryValidationError)
    return result


def _nonnegative_int(value: int, name: str) -> int:
    """Validate a Python integer without accepting booleans or truncation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        message = f"{name} must be an integer"
        reject(message, GeometryValidationError)
    result = int(value)
    if result < 0:
        message = f"{name} must be non-negative"
        reject(message, GeometryValidationError)
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
            reject(message, GeometryValidationError)
        extra_iter = _nonnegative_int(self.extra_iter, "extra_iter")
        if max_iter + extra_iter > INT32_MAX:
            message = "max_iter + extra_iter exceeds int32 iteration capacity"
            reject(message, GeometryValidationError)
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
            reject(message, GeometryValidationError)

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
            reject(message, GeometryValidationError)
        if self.kind == "cpu" and (
            self.physical_uuid is not None or self.mig_uuid is not None
        ):
            message = "CPU device identity cannot contain CUDA UUIDs"
            reject(message, GeometryValidationError)
        if self.kind == "cuda" and not self.physical_uuid:
            message = "CUDA device identity requires a physical UUID"
            reject(message, GeometryValidationError)

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
            reject(message, GeometryValidationError)
        if self.thread_count is not None:
            count = _nonnegative_int(self.thread_count, "thread_count")
            if count < 1:
                message = "thread_count must be at least 1"
                reject(message, GeometryValidationError)
            object.__setattr__(self, "thread_count", count)
        if self.launch_config is not None:
            config = tuple(
                _nonnegative_int(v, "launch_config value") for v in self.launch_config
            )
            if not config:
                message = "launch_config must not be empty"
                reject(message, GeometryValidationError)
            object.__setattr__(self, "launch_config", config)
        if self.device.kind == "cuda" and self.openmp_runtime is not None:
            message = "CUDA execution profiles cannot declare an OpenMP runtime"
            reject(message, GeometryValidationError)

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
            reject(message, GeometryValidationError)
        if not isinstance(self.device, DeviceKey) or not isinstance(
            self.solver, SolverSettings
        ):
            message = "candidate device and solver must use geometry value objects"
            reject(message, GeometryValidationError)
        normalized_shape = tuple(
            _nonnegative_int(v, "shape dimension") for v in self.shape
        )
        if any(v == 0 for v in normalized_shape):
            message = "candidate request shape dimensions must be positive"
            reject(message, GeometryValidationError)
        object.__setattr__(self, "shape", normalized_shape)
        if self.profile is None:
            object.__setattr__(self, "profile", ExecutionProfile(self.device))
        elif self.profile.device != self.device:
            message = "candidate device and execution profile device must match"
            reject(message, GeometryValidationError)

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
    "incidence_deg",
    "look_azimuth_deg",
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
    "geom_flags",
)
_FLOAT_RESULT_FIELDS = {
    "latitude_deg",
    "longitude_deg",
    "height_m",
    "incidence_deg",
    "look_azimuth_deg",
    "range_index",
    "azimuth_index",
    "decision_residual",
    "final_residual",
    "tolerance",
    "residual_range_m",
    "residual_doppler_hz",
}
_BOOL_RESULT_FIELDS = {"converged", "max_iter_exhausted", "boundary_rechecked"}
_UINT8_RESULT_FIELDS = {"geom_flags"}

#: ``geom_flags`` bit meanings (ISCE2-style layover/shadow mask).
GEOM_FLAG_LAYOVER = np.uint8(0b01)
GEOM_FLAG_SHADOW = np.uint8(0b10)


@dataclass(frozen=True, slots=True)
class TransformResult:
    """The exact seventeen-field result contract for both transforms.

    ``geom_flags`` is an rdr2geo-only output: a uint8 bitmask with bit 0 set
    on layover lanes and bit 1 on shadow lanes.  geo2rdr results carry a
    zero field so both operations share one contract.
    """

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    height_m: np.ndarray
    incidence_deg: np.ndarray
    look_azimuth_deg: np.ndarray
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
    geom_flags: np.ndarray

    def __post_init__(self) -> None:
        """Validate exact field names, dtypes, and common shape."""
        shape: tuple[int, ...] | None = None
        for name in _RESULT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                message = f"{name} must be a NumPy array"
                reject(message, GeometryValidationError)
            if name in _FLOAT_RESULT_FIELDS and value.dtype != np.dtype(np.float64):
                message = f"{name} must have dtype float64"
                reject(message, GeometryValidationError)
            if name == "iterations" and value.dtype != np.dtype(np.int32):
                message = "iterations must have dtype int32"
                reject(message, GeometryValidationError)
            if name in _BOOL_RESULT_FIELDS and value.dtype != np.dtype(bool):
                message = f"{name} must have dtype bool"
                reject(message, GeometryValidationError)
            if name in _UINT8_RESULT_FIELDS and value.dtype != np.dtype(np.uint8):
                message = f"{name} must have dtype uint8"
                reject(message, GeometryValidationError)
            if shape is None:
                shape = value.shape
            elif value.shape != shape:
                message = "all TransformResult fields must have the same shape"
                reject(message, GeometryValidationError)

    @classmethod
    def invalid(
        cls,
        shape: int | Sequence[int],
        *,
        operation: Operation | str = Operation.GEO2RDR,
    ) -> TransformResult:
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
            reject(message, GeometryValidationError)
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
            np.zeros(shape_tuple, dtype=np.uint8),
        )

    @classmethod
    def from_arrays(
        cls,
        fields: Mapping[str, np.ndarray],
        *,
        operation: Operation | str = Operation.GEO2RDR,
        invalid_mask: np.ndarray | None = None,
    ) -> TransformResult:
        """Construct and validate a result mapping, applying invalid sentinels."""
        missing = [name for name in _RESULT_FIELDS if name not in fields]
        extra = [name for name in fields if name not in _RESULT_FIELDS]
        if missing or extra:
            message = f"result fields mismatch; missing={missing}, extra={extra}"
            reject(message, GeometryValidationError)
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
                reject(message, GeometryValidationError)
            valid_mask = ~mask
        if op is Operation.GEO2RDR and np.any(valid_mask & (result.tolerance != 1.0)):
            message = "valid geo2rdr lanes must publish tolerance=1.0"
            reject(message, GeometryValidationError)
        if op is Operation.RDR2GEO and np.any(
            valid_mask & (~np.isfinite(result.tolerance) | (result.tolerance <= 0.0))
        ):
            message = "valid rdr2geo lanes must publish a positive finite tolerance"
            reject(message, GeometryValidationError)
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
        for name in _UINT8_RESULT_FIELDS:
            mutable[name][mask] = 0
        mutable["tolerance"][mask] = np.nan
        return cls(**mutable)

    @classmethod
    def invalid_lane(
        cls,
        shape: int | Sequence[int] = (),
        *,
        operation: Operation | str = Operation.GEO2RDR,
    ) -> TransformResult:
        """Alias for creating one or more prescribed invalid lanes."""
        return cls.invalid(shape, operation=operation)

    invalid_lanes = invalid_lane

    @property
    def fields(self) -> tuple[str, ...]:
        """Return the exact ordered field names."""
        return _RESULT_FIELDS

    def validate(self) -> TransformResult:
        """Validate and return this immutable result."""
        self.__post_init__()
        return self


def look_angles_deg(
    model: RadarGeometryModel,
    azimuth_index: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    height_m: np.ndarray,
    converged: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute incidence and look-azimuth angles for solved radar lanes.

    Parameters
    ----------
    model : RadarGeometryModel
        Timing/geometry model supplying the orbit interpolator.
    azimuth_index : numpy.ndarray
        Solved azimuth sample index per lane.
    latitude_deg, longitude_deg, height_m : numpy.ndarray
        Solved ground position per lane (WGS84 geodetic).
    converged : numpy.ndarray, optional
        Boolean lane mask; non-converged lanes receive NaN angles.

    Returns
    -------
    tuple of numpy.ndarray
        ``(incidence_deg, look_azimuth_deg)`` where incidence is measured
        from the local ellipsoid normal (zenith) at the ground point and
        look azimuth is the clockwise-from-north azimuth of the
        satellite-to-ground look direction projected onto the local
        tangent plane.  The conventions match
        ``NetworkGeometry.compute_los_unit_vectors``:
        ``los_east = -sin(inc) * sin(azim)``,
        ``los_north = -sin(inc) * cos(azim)``, ``los_up = cos(inc)`` with
        the LOS unit vector pointing from ground to satellite.

    """
    from faninsar.processing.geometry.ellipsoid import llh_to_ecef

    azimuth_index = np.asarray(azimuth_index, dtype=np.float64)
    shape = azimuth_index.shape
    flat_azimuth = azimuth_index.reshape(-1)
    flat_lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)
    flat_lon = np.asarray(longitude_deg, dtype=np.float64).reshape(-1)
    flat_hgt = np.asarray(height_m, dtype=np.float64).reshape(-1)
    times_s = model.azimuth_time_seconds(flat_azimuth)
    in_coverage = (
        np.isfinite(times_s)
        & (times_s >= model.orbit.t_min_s)
        & (times_s <= model.orbit.t_max_s)
    )
    positions = np.full((flat_azimuth.size, 3), np.nan, dtype=np.float64)
    if np.any(in_coverage):
        sat_xyz, _ = model.orbit.evaluate_array(times_s[in_coverage])
        positions[in_coverage] = sat_xyz
    ground_x, ground_y, ground_z = llh_to_ecef(flat_lat, flat_lon, flat_hgt)
    los = positions - np.stack([ground_x, ground_y, ground_z], axis=-1)
    norms = np.linalg.norm(los, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        los = los / norms[:, None]

    lat_rad = np.deg2rad(flat_lat)
    lon_rad = np.deg2rad(flat_lon)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    east = np.stack([-sin_lon, cos_lon, np.zeros_like(sin_lat)], axis=-1)
    north = np.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=-1)
    up = np.stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], axis=-1)
    los_east = np.einsum("ij,ij->i", los, east)
    los_north = np.einsum("ij,ij->i", los, north)
    los_up = np.einsum("ij,ij->i", los, up)
    incidence = np.rad2deg(np.arccos(np.clip(los_up, -1.0, 1.0)))
    look_azimuth = np.rad2deg(np.arctan2(-los_east, -los_north)) % 360.0
    if converged is not None:
        mask = ~np.asarray(converged, dtype=bool).reshape(-1)
        incidence = np.where(mask, np.nan, incidence)
        look_azimuth = np.where(mask, np.nan, look_azimuth)
    invalid = ~np.isfinite(flat_lat) | ~np.isfinite(flat_lon) | ~np.isfinite(flat_hgt)
    invalid |= ~in_coverage
    incidence = np.where(invalid, np.nan, incidence)
    look_azimuth = np.where(invalid, np.nan, look_azimuth)
    return incidence.reshape(shape), look_azimuth.reshape(shape)


def geom_flags(
    model: RadarGeometryModel,
    azimuth_index: np.ndarray,
    range_index: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    height_m: np.ndarray,
    converged: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the ISCE2-style layover/shadow bitmask for solved rdr2geo lanes.

    The test follows the ISCE3 ``Topo::setLayoverShadow`` criterion evaluated
    at the satellite.  For every lane the satellite position comes from the
    orbit interpolator and the ground position from the solved geodetic
    coordinates; both are converted to ECEF.  The geocentric elevation angle
    at the satellite is the angle between the sub-satellite (nadir) direction
    and the satellite-to-ground line of sight; the cross-track coordinate is
    the perpendicular distance from the satellite nadir line to the ground
    point in the slant-range plane.

    The criterion mirrors ISCE3 ``Topo::setLayoverShadow``.  For every lane
    the satellite position and velocity come from the orbit interpolator and
    the ground position from the solved geodetic coordinates; both are
    converted to ECEF.  A geocentric TCN basis is built per lane: the nadir
    unit ``n = -sat/|sat|`` and the cross-track unit ``c = (n x v)/|n x v|``.
    Each lane then has

    * a geocentric *elevation angle* — the angle at the satellite between the
      nadir direction and the satellite-to-ground line of sight,
      ``acos(sat · (sat - ground) / (|sat| |sat - ground|))``;
    * a *cross-track* coordinate ``satToGround · c`` (positive on the look
      side for a right-looking sensor); and
    * its *slant range* ``|sat - ground|``.

    Lanes in one azimuth row are sorted by cross-track and walked from near
    to far: a lane is in **shadow** (bit 1) when its elevation angle falls at
    or below the running maximum — a nearer lane occludes it.  The lanes are
    then re-sorted by slant range and walked in both directions: a lane is in
    **layover** (bit 0) when its cross-track coordinate falls inside the
    running cross-track envelope, i.e. the ground folds back over a nearer
    line of sight.  On flat terrain both walks are monotonic so the mask is
    identically zero.  Non-converged and non-finite lanes publish a zero
    bitmask and never occlude other lanes.

    Parameters
    ----------
    model : RadarGeometryModel
        Timing/geometry model supplying the orbit interpolator and the
        slant-range mapping.
    azimuth_index : numpy.ndarray
        Solved azimuth sample index per lane.
    range_index : numpy.ndarray
        Solved slant-range sample index per lane (float column index).
    latitude_deg, longitude_deg, height_m : numpy.ndarray
        Solved ground position per lane (WGS84 geodetic).
    converged : numpy.ndarray, optional
        Boolean lane mask; non-converged and non-finite lanes publish a
        zero bitmask and are never occluders.

    Returns
    -------
    numpy.ndarray
        ``uint8`` bitmask with bit 0 = layover (:data:`GEOM_FLAG_LAYOVER`)
        and bit 1 = shadow (:data:`GEOM_FLAG_SHADOW`).

    """
    from faninsar.processing.geometry.ellipsoid import llh_to_ecef

    azimuth_index = np.asarray(azimuth_index, dtype=np.float64)
    shape = azimuth_index.shape
    if azimuth_index.ndim == 0:
        return np.zeros(shape, dtype=np.uint8)
    flat_azimuth = azimuth_index.reshape(-1)
    flat_range = np.asarray(range_index, dtype=np.float64).reshape(-1)
    flat_lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)
    flat_lon = np.asarray(longitude_deg, dtype=np.float64).reshape(-1)
    flat_hgt = np.asarray(height_m, dtype=np.float64).reshape(-1)

    times_s = model.azimuth_time_seconds(flat_azimuth)
    in_coverage = (
        np.isfinite(times_s)
        & (times_s >= model.orbit.t_min_s)
        & (times_s <= model.orbit.t_max_s)
    )
    size = flat_azimuth.size
    sat_xyz = np.full((size, 3), np.nan, dtype=np.float64)
    sat_vel = np.full((size, 3), np.nan, dtype=np.float64)
    if np.any(in_coverage):
        positions, velocities = model.orbit.evaluate_array(times_s[in_coverage])
        sat_xyz[in_coverage] = positions
        sat_vel[in_coverage] = velocities
    ground_x, ground_y, ground_z = llh_to_ecef(flat_lat, flat_lon, flat_hgt)
    ground = np.stack([ground_x, ground_y, ground_z], axis=-1)

    valid = in_coverage & np.isfinite(flat_range)
    valid &= np.isfinite(flat_lat) & np.isfinite(flat_lon)
    valid &= np.isfinite(flat_hgt) & np.isfinite(sat_xyz).all(axis=-1)
    if converged is not None:
        valid &= np.asarray(converged, dtype=bool).reshape(-1)

    # Satellite-to-ground vector and geocentric elevation angle.
    sat_to_ground = ground - sat_xyz
    to_ground_norm = np.linalg.norm(sat_to_ground, axis=-1)
    sat_norm = np.linalg.norm(sat_xyz, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_elev = -np.einsum("ij,ij->i", sat_xyz, sat_to_ground) / (
            sat_norm * to_ground_norm
        )
        elevation = np.arccos(np.clip(cos_elev, -1.0, 1.0))

    # Geocentric TCN cross-track unit: n = -sat/|sat|, c = (n x v)/|n x v|.
    with np.errstate(invalid="ignore", divide="ignore"):
        nadir = -sat_xyz / sat_norm[:, None]
    cross_axis = np.cross(nadir, sat_vel)
    cross_norm = np.linalg.norm(cross_axis, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cross_unit = cross_axis / cross_norm[:, None]
    cross_track = np.einsum("ij,ij->i", sat_to_ground, cross_unit)
    if model.look_direction == "left":
        cross_track = -cross_track

    finite = valid & np.isfinite(elevation) & np.isfinite(cross_track)
    finite &= np.isfinite(to_ground_norm)
    elevation = np.where(finite, elevation, np.inf)
    cross_track = np.where(finite, cross_track, -np.inf)
    slant_range = np.where(finite, to_ground_norm, np.inf)

    flags = np.zeros(size, dtype=np.uint8)
    cols = shape[-1] if shape else size
    rows = size // cols if cols else 0
    for row in range(rows):
        lo, hi = row * cols, (row + 1) * cols
        ctrack = cross_track[lo:hi]
        elev = elevation[lo:hi]
        srange = slant_range[lo:hi]
        lane_valid = finite[lo:hi]
        order_c = np.argsort(ctrack, kind="stable")
        # Shadow: near-to-far in cross-track order, running-max elevation.
        max_elev = np.maximum.accumulate(elev[order_c])
        prev_max = np.concatenate(([-np.inf], max_elev[:-1]))
        shadow_sorted = lane_valid[order_c] & (elev[order_c] <= prev_max)
        shadow_row = np.zeros(cols, dtype=bool)
        shadow_row[order_c] = shadow_sorted
        # Layover: re-sort by slant range, two-sided cross-track reversal.
        order_r = np.argsort(srange, kind="stable")
        c_by_range = ctrack[order_r]
        r_valid = lane_valid[order_r]
        layover_sorted = np.zeros(cols, dtype=bool)
        run_min = np.minimum.accumulate(c_by_range)
        prev_min = np.concatenate(([np.inf], run_min[:-1]))
        layover_sorted[1:] |= r_valid[1:] & (c_by_range[1:] <= prev_min[1:])
        run_max = np.maximum.accumulate(c_by_range[::-1])[::-1]
        next_max = np.concatenate((run_max[1:], [-np.inf]))
        layover_sorted[:-1] |= r_valid[:-1] & (c_by_range[:-1] >= next_max[:-1])
        layover_row = np.zeros(cols, dtype=bool)
        layover_row[order_r] = layover_sorted
        flags[lo:hi] = layover_row.astype(np.uint8) | (shadow_row.astype(np.uint8) << 1)
    return flags.reshape(shape)


@dataclass(frozen=True, slots=True)
class RadarGeometryModel:
    """Radar timing/geometry model used by rdr2geo and geo2rdr."""

    orbit: OrbitInterpolator
    sensing_start: object
    azimuth_time_interval_s: float
    starting_slant_range_m: float
    range_spacing_m: float
    wavelength_m: float
    look_direction: str

    @classmethod
    def from_radar_grid(
        cls,
        grid: RadarGrid,
        orbit: OrbitMetadata,
    ) -> RadarGeometryModel:
        """Build a geometry model from a radar grid and orbit metadata."""
        return cls(
            orbit=OrbitInterpolator.from_orbit(orbit),
            sensing_start=grid.sensing_start,
            azimuth_time_interval_s=grid.azimuth_time_interval_s,
            starting_slant_range_m=grid.starting_slant_range_m,
            range_spacing_m=grid.range_spacing_m,
            wavelength_m=grid.wavelength_m,
            look_direction=grid.look_direction,
        )

    def azimuth_time(self, azimuth_index: float) -> datetime:
        """Return absolute sensing time for an azimuth sample index."""
        start = self.sensing_start
        if not isinstance(start, datetime):
            message = "sensing_start must be a datetime"
            reject(message, TypeError)
        return start + timedelta(
            seconds=float(azimuth_index) * self.azimuth_time_interval_s
        )

    def azimuth_time_seconds(self, azimuth_index: np.ndarray) -> np.ndarray:
        """Return seconds from orbit epoch for an array of azimuth indices."""
        start = self.sensing_start
        if not isinstance(start, datetime):
            message = "sensing_start must be a datetime"
            reject(message, TypeError)
        sensing_offset_s = (start - self.orbit.epoch).total_seconds()
        return (
            sensing_offset_s
            + np.asarray(azimuth_index, dtype=np.float64) * self.azimuth_time_interval_s
        )

    def slant_range_m(self, range_index: float) -> float:
        """Return two-way-consistent one-way slant range for a range index."""
        return self.starting_slant_range_m + float(range_index) * self.range_spacing_m


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
        reject(message, GeometryValidationError)
    if dtype is None:
        message = f"{name} requires an expected dtype"
        reject(message, GeometryValidationError)
    expected_dtype = np.dtype(dtype)
    if shape is None:
        shape = expected_shape
    elif expected_shape is not None and tuple(shape) != tuple(expected_shape):
        message = f"{name} received conflicting shape expectations"
        reject(message, GeometryValidationError)
    if device is None:
        device = expected_device
    elif expected_device is not None and device != expected_device:
        message = f"{name} received conflicting device expectations"
        reject(message, GeometryValidationError)
    if not isinstance(array, np.ndarray):
        message = f"{name} must be a NumPy array"
        reject(message, GeometryValidationError)
    if array.dtype != expected_dtype:
        message = f"{name} must have dtype {expected_dtype}"
        reject(message, GeometryValidationError)
    if array.ndim != len(array.shape) or array.ndim != len(array.strides):
        message = f"{name} rank/shape/strides metadata is inconsistent"
        reject(message, GeometryValidationError)
    if any(d <= 0 for d in array.shape):
        message = f"{name} dimensions must be positive"
        reject(message, GeometryValidationError)
    if shape is not None and tuple(array.shape) != tuple(shape):
        message = f"{name} has shape {array.shape}, expected {tuple(shape)}"
        reject(message, GeometryValidationError)
    expected_strides = tuple(
        array.itemsize * int(np.prod(array.shape[index + 1 :]))
        for index in range(array.ndim)
    )
    if not array.flags.c_contiguous or tuple(array.strides) != expected_strides:
        message = f"{name} must use exact C-order strides"
        reject(message, GeometryValidationError)
    if strides is not None and tuple(strides) != tuple(array.strides):
        message = f"{name} has unexpected strides"
        reject(message, GeometryValidationError)
    if byte_length is not None:
        if isinstance(byte_length, (bool, np.bool_)) or int(byte_length) != byte_length:
            message = f"{name} byte_length must be an integer"
            reject(message, GeometryValidationError)
        if int(byte_length) != int(array.nbytes):
            message = f"{name} has an unexpected byte length"
            reject(message, GeometryValidationError)
    if require_finite and not np.all(np.isfinite(array)):
        message = f"{name} must contain only finite values"
        reject(message, GeometryValidationError)
    actual_device = device or DeviceKey.cpu()
    if actual_device.kind != "cpu":
        message = f"{name} NumPy owner cannot prove CUDA allocation"
        reject(message, GeometryValidationError)
    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    if not owner.flags.c_contiguous:
        message = f"{name} owner must be contiguous host memory"
        reject(message, GeometryValidationError)
    # Pointer and extent are deliberately read only after every structural check.
    address = int(array.__array_interface__["data"][0])
    owner_address = int(owner.__array_interface__["data"][0])
    end = address + int(array.nbytes)
    owner_end = owner_address + int(owner.nbytes)
    if address % expected_dtype.itemsize != 0:
        message = f"{name} address is not dtype aligned"
        reject(message, GeometryValidationError)
    if address <= 0 or end < address or address < owner_address or end > owner_end:
        message = f"{name} span is outside its owner allocation"
        reject(message, GeometryValidationError)
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
                reject(message, GeometryValidationError)
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
        reject(message, GeometryValidationError)
    return validated


# Explicit aliases keep the validator seam discoverable to native adapters.
validate_span = validate_array_span
validate_native_spans = validate_spans
ArraySpan = RawSpan
NativeSpan = RawSpan
validate_input_span = validate_array_span


__all__ = [
    "GEOM_FLAG_LAYOVER",
    "GEOM_FLAG_SHADOW",
    "INT32_MAX",
    "ArraySpan",
    "CandidateKey",
    "DeviceKey",
    "ExecutionProfile",
    "GeometryValidationError",
    "NativeSpan",
    "Operation",
    "OperationSettings",
    "RadarGeometryModel",
    "RawSpan",
    "RawTensorSpan",
    "SolverSettings",
    "TransformResult",
    "geom_flags",
    "validate_array_span",
    "validate_input_span",
    "validate_native_spans",
    "validate_span",
    "validate_spans",
    "validate_tensor_span",
]
