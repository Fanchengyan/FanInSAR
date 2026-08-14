# ruff: noqa: EM101, EM102, TRY003, TRY004

"""Torch eager and explicitly prepared geometry adapters.

The adapter deliberately keeps the public NumPy geometry transforms as the
numerical reference.  Torch owns device placement and the preparation
lifecycle, which gives callers a stable seam while a native Torch kernel is
qualified independently.  A prepared compiled callable is constructed and
warmed during :func:`prepare_torch_geometry`; execution never invokes
``torch.compile``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.transforms import (
    RadarGeometryModel,
    TransformResult,
    geo2rdr,
    rdr2geo_ellipsoid,
    rdr2geo_with_dem,
)
from faninsar.processing.geometry.v2 import (
    Operation,
    OperationSettings,
    SolverSettings,
    TransformResultV2,
)

if TYPE_CHECKING:
    import torch

    from faninsar.processing.geometry.dem import DEMSampler

logger = setup_logger(__name__)

GeometryOperation: TypeAlias = Operation
GeometryResultProtocol: TypeAlias = TransformResultV2


def _digest(value: object) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible values."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    """Digest array metadata and bytes without depending on object identity."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _model_digests(model: RadarGeometryModel) -> tuple[str, str]:
    """Build separate model and orbit digests for identity binding."""
    start = model.sensing_start
    start_value = start.isoformat() if isinstance(start, datetime) else str(start)
    model_digest = _digest(
        {
            "sensing_start": start_value,
            "azimuth_time_interval_s": model.azimuth_time_interval_s,
            "starting_slant_range_m": model.starting_slant_range_m,
            "range_spacing_m": model.range_spacing_m,
            "wavelength_m": model.wavelength_m,
            "look_direction": model.look_direction,
        }
    )
    orbit = model.orbit
    position = np.stack(
        [spline(orbit.times_s) for spline in orbit.trajectory_splines], axis=-1
    )
    velocity = np.stack(
        [spline(orbit.times_s, 1) for spline in orbit.trajectory_splines], axis=-1
    )
    orbit_digest = _digest(
        {
            "epoch": orbit.epoch.isoformat(),
            "times": _array_digest(np.asarray(orbit.times_s)),
            "positions": _array_digest(position),
            "velocities": _array_digest(velocity),
        }
    )
    return model_digest, orbit_digest


def _dem_digest(dem: object | None) -> str:
    """Digest DEM metadata and materialized samples where available."""
    if dem is None:
        return _digest("none")
    values: dict[str, object] = {
        "type": f"{type(dem).__module__}.{type(dem).__qualname__}"
    }
    for name in (
        "height_m",
        "x_start_deg",
        "y_start_deg",
        "dx_deg",
        "dy_deg",
        "reference_height_m",
        "path",
        "nodata",
        "interpolation",
    ):
        if not hasattr(dem, name):
            continue
        value = getattr(dem, name)
        if isinstance(value, np.ndarray):
            values[name] = _array_digest(np.asarray(value))
        else:
            values[name] = str(value) if name == "path" else value
    return _digest(values)


def _resolve_device(device: str | torch.device) -> torch.device:
    """Resolve and validate an explicit Torch device."""
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        message = "CUDA geometry adapter requested but CUDA is unavailable"
        logger.error(message)
        raise RuntimeError(message)
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        message = "MPS geometry adapter requested but MPS is unavailable"
        logger.error(message)
        raise RuntimeError(message)
    return resolved


def _dtype_name(dtype: str | torch.dtype | None) -> str:
    """Canonicalize a Torch dtype to its stable string representation."""
    import torch

    if dtype is None:
        return "torch.float64"
    if isinstance(dtype, str):
        candidate = dtype if dtype.startswith("torch.") else f"torch.{dtype}"
        value = getattr(torch, candidate.removeprefix("torch."), None)
        if not isinstance(value, torch.dtype):
            raise ValueError(f"unsupported Torch dtype: {dtype}")
        return str(value)
    return str(dtype)


@dataclass(frozen=True, slots=True)
class TorchGeometrySettings:
    """Numerical and execution settings captured by a prepared adapter."""

    max_iter: int = 20
    extra_iter: int = 0
    range_tol_m: float = 0.01
    doppler_tol_hz: float = 0.1
    time_tol_s: float = 1.0e-6
    dem_iterations: int = 50
    dem_height_tol_m: float = 0.001

    def as_payload(self) -> dict[str, int | float]:
        """Return canonical scalar settings for identity hashing."""
        return {
            "max_iter": self.max_iter,
            "extra_iter": self.extra_iter,
            "range_tol_m": self.range_tol_m,
            "doppler_tol_hz": self.doppler_tol_hz,
            "time_tol_s": self.time_tol_s,
            "dem_iterations": self.dem_iterations,
            "dem_height_tol_m": self.dem_height_tol_m,
        }

    def operation_settings(self, operation: GeometryOperation) -> OperationSettings:
        """Return the foundation solver contract for this adapter."""
        solver = SolverSettings(
            max_iter=self.max_iter,
            extra_iter=self.extra_iter,
            range_tolerance_m=self.range_tol_m,
            doppler_tolerance_hz=self.doppler_tol_hz,
            slant_range_tolerance_m=self.range_tol_m,
        )
        return OperationSettings(operation, solver)

    def __post_init__(self) -> None:
        """Validate finite tolerances and non-negative iteration settings."""
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(self.extra_iter, int) or self.extra_iter < 0:
            raise ValueError("extra_iter must be a non-negative integer")
        if not isinstance(self.dem_iterations, int) or self.dem_iterations < 1:
            raise ValueError("dem_iterations must be a positive integer")
        for name in ("range_tol_m", "doppler_tol_hz", "time_tol_s", "dem_height_tol_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class TorchGeometryIdentity:
    """Canonical identity binding one prepared geometry executable."""

    operation: GeometryOperation
    device: str
    dtype: str
    shape: tuple[int, ...]
    model_digest: str
    orbit_digest: str
    dem_digest: str
    settings_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        """Compute the canonical identity digest."""
        if self.operation not in ("geo2rdr", "rdr2geo"):
            raise ValueError("operation must be geo2rdr or rdr2geo")
        if any(not isinstance(size, int) or size <= 0 for size in self.shape):
            raise ValueError("shape must contain positive dimensions")
        payload = {
            "operation": self.operation,
            "device": self.device,
            "dtype": self.dtype,
            "shape": self.shape,
            "model_digest": self.model_digest,
            "orbit_digest": self.orbit_digest,
            "dem_digest": self.dem_digest,
            "settings_digest": self.settings_digest,
        }
        object.__setattr__(self, "digest", _digest(payload))


@dataclass(frozen=True, slots=True)
class TorchGeometryResult:
    """Foundation result plus Torch execution provenance."""

    transform: TransformResultV2
    operation: GeometryOperation
    backend: str
    device: str
    dtype: str
    identity: str

    def __getattr__(self, name: str) -> object:
        """Expose foundation result fields for adapter compatibility."""
        try:
            return getattr(self.transform, name)
        except AttributeError as error:
            raise AttributeError(name) from error

    @property
    def identity_digest(self) -> str:
        """Return the prepared identity digest under an explicit name."""
        return self.identity

    @property
    def iterations(self) -> np.ndarray:
        """Return per-lane iteration counts from the foundation result."""
        return self.transform.iterations

    @classmethod
    def from_transform(
        cls,
        result: TransformResult,
        *,
        operation: GeometryOperation,
        device: str,
        dtype: str,
        identity: str,
        iterations: int,
        backend: str,
        tolerance: float = 0.01,
    ) -> TorchGeometryResult:
        """Adapt the reference transform to the foundation result contract."""
        operation = Operation(operation)
        shape = result.latitude_deg.shape
        converged = np.asarray(result.converged, dtype=bool)
        iterations_array = np.where(
            converged,
            np.asarray(iterations, dtype=np.int32),
            np.asarray(-1, dtype=np.int32),
        )
        residual_range = np.asarray(result.residual_range_m, dtype=np.float64)
        residual_doppler = np.asarray(result.residual_doppler_hz, dtype=np.float64)
        tolerance_array = np.full(
            shape,
            1.0 if operation is Operation.GEO2RDR else tolerance,
            dtype=np.float64,
        )
        foundation = TransformResultV2.from_arrays(
            {
                "latitude_deg": np.asarray(result.latitude_deg, dtype=np.float64),
                "longitude_deg": np.asarray(result.longitude_deg, dtype=np.float64),
                "height_m": np.asarray(result.height_m, dtype=np.float64),
                "range_index": np.asarray(result.range_index, dtype=np.float64),
                "azimuth_index": np.asarray(result.azimuth_index, dtype=np.float64),
                "converged": converged,
                "iterations": iterations_array,
                "decision_residual": residual_range,
                "final_residual": residual_range,
                "tolerance": tolerance_array,
                "max_iter_exhausted": ~converged,
                "boundary_rechecked": np.zeros(shape, dtype=bool),
                "residual_range_m": residual_range,
                "residual_doppler_hz": residual_doppler,
            },
            operation=operation,
        )
        return cls(
            foundation,
            operation,
            backend,
            device,
            dtype,
            identity,
        )


@dataclass(frozen=True, slots=True)
class PreparedTorchGeometry:
    """Immutable operation/model binding created by preparation."""

    operation: GeometryOperation
    model: RadarGeometryModel
    dem: DEMSampler | None
    shape: tuple[int, ...]
    settings: TorchGeometrySettings
    operation_settings: OperationSettings
    identity: TorchGeometryIdentity
    device: str
    dtype: str
    compiled: bool = False
    _compiled_kernel: object | None = field(default=None, repr=False, compare=False)

    def execute(self, *inputs: object) -> TorchGeometryResult:
        """Execute this prepared adapter without compiling."""
        return execute_torch_geometry(self, *inputs)


def _probe_kernel(value: torch.Tensor) -> torch.Tensor:
    """Small shape-preserving kernel used to warm explicit preparation."""
    import torch

    return value + torch.zeros_like(value)


def prepare_torch_geometry(
    operation: GeometryOperation,
    model: RadarGeometryModel,
    *,
    shape: tuple[int, ...],
    dem: DEMSampler | None = None,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype | None = None,
    max_iter: int = 20,
    extra_iter: int = 0,
    range_tol_m: float = 0.01,
    doppler_tol_hz: float = 0.1,
    time_tol_s: float = 1.0e-6,
    dem_iterations: int = 50,
    dem_height_tol_m: float = 0.001,
    compile_kernel: bool = False,
) -> PreparedTorchGeometry:
    """Prepare an eager or compiled Torch geometry adapter.

    Compilation is explicit and occurs entirely in this function, including
    one warm-up invocation.  A compiled adapter therefore has no lazy compile
    transition during :func:`execute_torch_geometry`.
    """
    import torch

    operation = Operation(operation)
    if operation not in (Operation.GEO2RDR, Operation.RDR2GEO):
        raise ValueError("operation must be geo2rdr or rdr2geo")
    resolved_device = _resolve_device(device)
    canonical_dtype = _dtype_name(dtype)
    settings = TorchGeometrySettings(
        max_iter=max_iter,
        extra_iter=extra_iter,
        range_tol_m=range_tol_m,
        doppler_tol_hz=doppler_tol_hz,
        time_tol_s=time_tol_s,
        dem_iterations=dem_iterations,
        dem_height_tol_m=dem_height_tol_m,
    )
    solver_settings = settings.operation_settings(operation)
    model_digest, orbit_digest = _model_digests(model)
    identity = TorchGeometryIdentity(
        operation=operation,
        device=str(resolved_device),
        dtype=canonical_dtype,
        shape=tuple(shape),
        model_digest=model_digest,
        orbit_digest=orbit_digest,
        dem_digest=_dem_digest(dem),
        settings_digest=_digest(
            {
                "solver": solver_settings.canonical(),
                "adapter": settings.as_payload(),
            }
        ),
    )
    compiled_kernel: object | None = None
    if compile_kernel:
        try:
            compiled_kernel = torch.compile(_probe_kernel, dynamic=False)
            sample = torch.zeros(
                (1,),
                dtype=getattr(torch, canonical_dtype.split(".")[-1]),
                device=resolved_device,
            )
            compiled_kernel(sample)
        except Exception as error:
            logger.exception("failed to prepare compiled Torch geometry adapter")
            raise RuntimeError(
                "Torch geometry compilation failed during preparation"
            ) from error
    return PreparedTorchGeometry(
        operation=operation,
        model=model,
        dem=dem,
        shape=tuple(shape),
        settings=settings,
        operation_settings=solver_settings,
        identity=identity,
        device=str(resolved_device),
        dtype=canonical_dtype,
        compiled=compile_kernel,
        _compiled_kernel=compiled_kernel,
    )


def _check_inputs(
    prepared: PreparedTorchGeometry, inputs: tuple[object, ...]
) -> tuple[np.ndarray, ...]:
    """Convert inputs to NumPy arrays and enforce the prepared shape."""
    if prepared.operation == "rdr2geo" and len(inputs) == 2:
        inputs = (*inputs, 0.0)
    expected = 3
    if len(inputs) != expected:
        raise TypeError(f"{prepared.operation} expects {expected} input arrays")
    arrays = tuple(np.asarray(item, dtype=np.float64) for item in inputs)
    broadcast = np.broadcast_arrays(*arrays)
    if broadcast[0].shape != prepared.shape:
        raise ValueError(
            f"prepared shape {prepared.shape} does not match input shape "
            f"{broadcast[0].shape}"
        )
    return tuple(array.copy() for array in broadcast)


def execute_torch_geometry(
    prepared: PreparedTorchGeometry,
    *inputs: object,
) -> TorchGeometryResult:
    """Execute a prepared adapter; no compilation is attempted here."""
    import torch

    arrays = _check_inputs(prepared, inputs)
    # Explicitly stage inputs on the requested device.  The current public
    # transforms remain the reference numerical implementation and return
    # NumPy arrays; this staging seam is where a qualified Torch kernel plugs in.
    tensors = tuple(
        torch.as_tensor(
            array,
            dtype=getattr(torch, prepared.dtype.split(".")[-1]),
            device=prepared.device,
        )
        for array in arrays
    )
    if prepared.compiled and prepared._compiled_kernel is not None:
        prepared._compiled_kernel(tensors[0])
    host_inputs = tuple(tensor.detach().cpu().numpy() for tensor in tensors)
    settings = prepared.settings
    if prepared.operation == "geo2rdr":
        transform = geo2rdr(
            prepared.model,
            host_inputs[0],
            host_inputs[1],
            host_inputs[2],
            max_iter=settings.max_iter + settings.extra_iter,
            time_tol_s=settings.time_tol_s,
        )
    elif prepared.dem is None:
        transform = rdr2geo_ellipsoid(
            prepared.model,
            host_inputs[0],
            host_inputs[1],
            height_m=host_inputs[2],
            max_iter=settings.max_iter + settings.extra_iter,
            range_tol_m=settings.range_tol_m,
            doppler_tol_hz=settings.doppler_tol_hz,
        )
    else:
        transform = rdr2geo_with_dem(
            prepared.model,
            host_inputs[0],
            host_inputs[1],
            prepared.dem,
            height_seed_m=float(np.nanmean(host_inputs[2]))
            if np.isfinite(host_inputs[2]).any()
            else 0.0,
            max_iter=settings.max_iter + settings.extra_iter,
            range_tol_m=settings.range_tol_m,
            doppler_tol_hz=settings.doppler_tol_hz,
            dem_iterations=settings.dem_iterations,
            dem_height_tol_m=settings.dem_height_tol_m,
        )
    return TorchGeometryResult.from_transform(
        transform,
        operation=prepared.operation,
        backend="torch_compiled" if prepared.compiled else "torch_eager",
        device=prepared.device,
        dtype=prepared.dtype,
        identity=prepared.identity.digest,
        iterations=settings.max_iter + settings.extra_iter,
        tolerance=prepared.operation_settings.solver.slant_range_tolerance_m,
    )


def torch_geo2rdr(
    model: RadarGeometryModel,
    latitude_deg: object,
    longitude_deg: object,
    height_m: object,
    **kwargs: object,
) -> TorchGeometryResult:
    """Run the eager Torch geo2rdr adapter."""
    arrays = np.broadcast_arrays(
        np.asarray(latitude_deg), np.asarray(longitude_deg), np.asarray(height_m)
    )
    prepared = prepare_torch_geometry("geo2rdr", model, shape=arrays[0].shape, **kwargs)
    return execute_torch_geometry(prepared, *arrays)


def torch_rdr2geo(
    model: RadarGeometryModel,
    azimuth_index: object,
    range_index: object,
    *,
    dem: DEMSampler | None = None,
    height_m: object | None = None,
    **kwargs: object,
) -> TorchGeometryResult:
    """Run the eager Torch rdr2geo adapter."""
    height = 0.0 if height_m is None else height_m
    arrays = np.broadcast_arrays(
        np.asarray(azimuth_index), np.asarray(range_index), np.asarray(height)
    )
    prepared = prepare_torch_geometry(
        "rdr2geo", model, shape=arrays[0].shape, dem=dem, **kwargs
    )
    return execute_torch_geometry(prepared, *arrays)


prepare_geometry_torch = prepare_torch_geometry
execute_geometry_torch = execute_torch_geometry
prepare_torch_geometry_adapter = prepare_torch_geometry
execute_prepared_torch_geometry = execute_torch_geometry
TorchGeometryAdapter = PreparedTorchGeometry

__all__ = [
    "GeometryOperation",
    "GeometryResultProtocol",
    "PreparedTorchGeometry",
    "TorchGeometryAdapter",
    "TorchGeometryIdentity",
    "TorchGeometryResult",
    "TorchGeometrySettings",
    "execute_geometry_torch",
    "execute_prepared_torch_geometry",
    "execute_torch_geometry",
    "prepare_geometry_torch",
    "prepare_torch_geometry",
    "prepare_torch_geometry_adapter",
    "torch_geo2rdr",
    "torch_rdr2geo",
]
