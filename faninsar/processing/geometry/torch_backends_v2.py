# ruff: noqa: EM101, EM102, TRY003, TRY004, TRY301

"""Torch eager and explicitly prepared geometry adapters.

The adapter keeps the public preparation lifecycle separate from execution.
Torch owns the device-resident numerical loop.  CPU GEO2RDR constructs its two
compiled callables during :func:`prepare_torch_geometry`, while execution
provides the first real tensors that trigger specialization; execution never
invokes ``torch.compile``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.torch_kernels import (
    _geo2rdr_active,
    _geo2rdr_finalize,
    _geo2rdr_initialize,
    _geo2rdr_step,
    geo2rdr_kernel,
    prepared_orbit_tensors,
    rdr2geo_kernel,
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
    from faninsar.processing.geometry.transforms import RadarGeometryModel

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


def _dem_payload(dem: object, seen: set[int]) -> dict[str, object]:
    """Return recursive DEM metadata and materialized values."""
    if id(dem) in seen:
        return {"cycle": f"{type(dem).__module__}.{type(dem).__qualname__}"}
    seen.add(id(dem))
    values: dict[str, object] = {
        "type": f"{type(dem).__module__}.{type(dem).__qualname__}"
    }
    try:
        names = {
            field.name
            for field in dataclass_fields(dem)
            if not field.name.startswith("_")
        }
    except TypeError:
        names = set()
    names.update(
        name
        for name in (
            "height_m",
            "values",
            "data",
            "path",
            "source",
            "nodata",
            "interpolation",
        )
        if hasattr(dem, name)
    )
    for name in sorted(names):
        value = getattr(dem, name, None)
        if value is dem:
            continue
        if name in {"orthometric_dem", "geoid", "dem", "sampler", "inner"}:
            if value is not None:
                values[name] = _dem_payload(value, seen)
        elif isinstance(value, (np.ndarray, list, tuple)) and name in {
            "height_m",
            "values",
            "data",
        }:
            values[name] = _array_digest(np.asarray(value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            values[name] = str(value) if name in {"path", "source"} else value
    materialized = getattr(dem, "_height_array", None)
    if materialized is None and hasattr(dem, "_open"):
        try:
            materialized = dem._open().read(1)  # type: ignore[attr-defined]
        except (ImportError, OSError, RuntimeError, AttributeError):
            materialized = None
    if materialized is not None:
        values["materialized_height"] = _array_digest(np.asarray(materialized))
    dataset = getattr(dem, "_dataset", None)
    transform = getattr(dataset, "transform", None)
    bounds = getattr(dataset, "bounds", None)
    if transform is not None:
        values["origin_spacing"] = tuple(float(value) for value in transform[:6])
    if bounds is not None:
        values["bounds"] = tuple(float(value) for value in bounds)
    return values


def _dem_digest(dem: object | None) -> str:
    """Digest nested DEM metadata and materialized samples."""
    if dem is None:
        return _digest("none")
    return _digest(_dem_payload(dem, set()))


def _resolve_device(device: str | torch.device) -> torch.device:
    """Resolve and validate an explicit Torch device."""
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            message = "CUDA geometry adapter requested but CUDA is unavailable"
            logger.error(message)
            raise RuntimeError(message)
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            message = (
                f"CUDA geometry adapter requested unsupported device ordinal "
                f"{resolved.index}"
            )
            logger.error(message)
            raise RuntimeError(message)
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        message = "MPS geometry adapter requested but MPS is unavailable"
        logger.error(message)
        raise RuntimeError(message)
    return resolved


def _canonical_torch_device(device: str | torch.device) -> torch.device:
    """Return a Torch device with implicit CUDA ordinals made explicit.

    Parameters
    ----------
    device : str or torch.device
        Device identifier to canonicalize.

    Returns
    -------
    torch.device
        Canonical device.  An unqualified CUDA device uses the current CUDA
        ordinal, so ``cuda`` and ``cuda:0`` compare equal when device zero is
        current while distinct explicit ordinals remain distinct.

    """
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
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
    ecef_x_m: np.ndarray | None = None
    ecef_y_m: np.ndarray | None = None
    ecef_z_m: np.ndarray | None = None

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
    def ecef_xyz_m(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return device-computed ECEF outputs when the kernel provided them."""
        if self.ecef_x_m is None or self.ecef_y_m is None or self.ecef_z_m is None:
            return None
        return self.ecef_x_m, self.ecef_y_m, self.ecef_z_m

    @property
    def iterations(self) -> np.ndarray:
        """Return per-lane iteration counts from the foundation result."""
        return self.transform.iterations

    @classmethod
    def from_transform(
        cls,
        result: dict[str, np.ndarray],
        *,
        operation: GeometryOperation,
        device: str,
        dtype: str,
        identity: str,
        backend: str,
        range_tolerance: float = 0.01,
        doppler_tolerance: float = 0.1,
        slant_range_tolerance: float = 0.01,
    ) -> TorchGeometryResult:
        """Adapt device-kernel arrays to the foundation result contract."""
        operation = Operation(operation)
        values = {name: np.asarray(value) for name, value in result.items()}
        shape = values["latitude_deg"].shape
        converged = np.asarray(values["converged"], dtype=bool)
        invalid_mask = np.asarray(
            values.get("invalid", ~np.isfinite(values["latitude_deg"])), dtype=bool
        )
        residual_range = np.asarray(values["residual_range_m"], dtype=np.float64)
        residual_doppler = np.asarray(values["residual_doppler_hz"], dtype=np.float64)
        max_iter_exhausted = np.asarray(
            values.get("max_iter_exhausted", (~converged) & ~invalid_mask),
            dtype=bool,
        )
        if operation is Operation.GEO2RDR:
            decision = np.maximum(
                np.abs(residual_range) / range_tolerance,
                np.abs(residual_doppler) / doppler_tolerance,
            )
            tolerance_array = np.ones(shape, dtype=np.float64)
        else:
            decision = residual_range
            tolerance_array = np.full(shape, slant_range_tolerance, dtype=np.float64)
        foundation = TransformResultV2.from_arrays(
            {
                "latitude_deg": np.asarray(values["latitude_deg"], dtype=np.float64),
                "longitude_deg": np.asarray(values["longitude_deg"], dtype=np.float64),
                "height_m": np.asarray(values["height_m"], dtype=np.float64),
                "range_index": np.asarray(values["range_index"], dtype=np.float64),
                "azimuth_index": np.asarray(values["azimuth_index"], dtype=np.float64),
                "converged": converged,
                "iterations": np.asarray(values["iterations"], dtype=np.int32),
                "decision_residual": decision,
                "final_residual": decision,
                "tolerance": tolerance_array,
                "max_iter_exhausted": max_iter_exhausted,
                "boundary_rechecked": np.zeros(shape, dtype=bool),
                "residual_range_m": residual_range,
                "residual_doppler_hz": residual_doppler,
            },
            operation=operation,
            invalid_mask=invalid_mask,
        )
        return cls(
            foundation,
            operation,
            backend,
            device,
            dtype,
            identity,
            ecef_x_m=(
                np.asarray(values["ecef_x_m"], dtype=np.float64)
                if "ecef_x_m" in values
                else None
            ),
            ecef_y_m=(
                np.asarray(values["ecef_y_m"], dtype=np.float64)
                if "ecef_y_m" in values
                else None
            ),
            ecef_z_m=(
                np.asarray(values["ecef_z_m"], dtype=np.float64)
                if "ecef_z_m" in values
                else None
            ),
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
    height_seed_m: float = 0.0
    compiled: bool = False
    _kernel: object | None = field(default=None, repr=False, compare=False)
    _compiled_kernel: object | None = field(default=None, repr=False, compare=False)
    _compiled_first_stage: object | None = field(
        default=None, repr=False, compare=False
    )
    _state_finalizer: object | None = field(default=None, repr=False, compare=False)

    def execute(self, *inputs: object) -> TorchGeometryResult:
        """Execute this prepared adapter without compiling."""
        return execute_torch_geometry(self, *inputs)

    def execute_ecef(
        self, *inputs: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute Rdr2Geo and transfer only device-computed ECEF arrays.

        Parameters
        ----------
        *inputs : object
            Prepared Rdr2Geo input arrays in azimuth, range, and height-seed
            order.

        Returns
        -------
        tuple of numpy.ndarray
            Host ``(x, y, z)`` ECEF arrays.  The LLH and telemetry fields stay
            device-resident and are not copied across the device boundary.

        Raises
        ------
        ValueError
            If this prepared operation is not Rdr2Geo.

        """
        import torch

        if self.operation is not Operation.RDR2GEO:
            logger.error("execute_ecef is only available for rdr2geo")
            raise ValueError("execute_ecef is only available for rdr2geo")
        arrays = _check_inputs(self, inputs)
        kernel = self._compiled_kernel if self.compiled else self._kernel
        if kernel is None:
            logger.error("prepared Torch geometry kernel is missing")
            raise RuntimeError("prepared Torch geometry kernel is missing")
        with torch.no_grad():
            output = kernel(*arrays)
        try:
            ecef_fields = tuple(
                output[name] for name in ("ecef_x_m", "ecef_y_m", "ecef_z_m")
            )
        except (AttributeError, KeyError) as error:
            logger.exception("prepared Rdr2Geo kernel did not publish ECEF outputs")
            raise RuntimeError(
                "prepared Rdr2Geo kernel did not publish ECEF outputs"
            ) from error
        # Keep the three published fields separate.  Stacking first creates a
        # transient device allocation and then copies it back as one larger
        # buffer, even though the public ECEF ABI is already three arrays.
        return tuple(field.detach().cpu().numpy() for field in ecef_fields)  # type: ignore[return-value]


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

    Compilation is explicit.  CPU GEO2RDR creates its two graph wrappers during
    preparation but defers their first invocation until execution, so the
    compiler sees the caller's real tensors and strides.
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
    orbit_tensors = prepared_orbit_tensors(model, str(resolved_device))
    orbit_times, orbit_positions, orbit_velocities, sensing_offset_s = orbit_tensors

    def raster_payload(value: object) -> tuple[object, ...]:
        """Materialize one RasterDEM and its affine metadata."""
        try:
            interpolation = value.interpolation  # type: ignore[attr-defined]
            if interpolation != "biquintic":
                raise ValueError(
                    "Torch RasterDEM supports only biquintic interpolation"
                )
            dataset = value._open()  # type: ignore[attr-defined]
            samples = value._height_array  # type: ignore[attr-defined]
            if samples is None:
                samples = dataset.read(1).astype(np.float32, copy=False)
                nodata = value.nodata  # type: ignore[attr-defined]
                if nodata is None:
                    nodata = dataset.nodata
                if nodata is not None:
                    samples = np.where(np.isclose(samples, nodata), np.nan, samples)
                value._height_array = samples  # type: ignore[attr-defined]
            transform = dataset.transform
            if abs(float(transform.b)) > 1.0e-12 or abs(float(transform.d)) > 1.0e-12:
                raise ValueError(
                    "Torch RasterDEM requires an unrotated geographic affine transform"
                )
            samples = torch.as_tensor(
                np.asarray(samples, dtype=np.float64),
                dtype=torch.float64,
                device=resolved_device,
            ).contiguous()
            if bool(torch.isinf(samples).any().item()):
                raise ValueError("RasterDEM contains infinite heights")
            if samples.ndim != 2 or min(samples.shape) < 6:
                raise ValueError("RasterDEM must provide at least a 6x6 grid")
            return (
                samples,
                float(transform.f),
                float(transform.c),
                float(transform.e),
                float(transform.a),
            )
        except ValueError:
            raise
        except Exception as error:
            logger.exception("failed to prepare RasterDEM for Torch geometry")
            raise TypeError(
                "RasterDEM could not be prepared for Torch geometry"
            ) from error

    def dem_payload(value: object) -> tuple[object, ...]:
        """Materialize supported DEM compositions into one device payload."""
        kind = type(value).__name__
        if kind == "ConstantHeightDEM":
            return ("constant", float(value.height_m))  # type: ignore[attr-defined]
        if kind == "RasterDEM":
            return ("raster", *raster_payload(value))
        if kind == "GeoidAdjustedDEM":
            left = dem_payload(value.orthometric_dem)  # type: ignore[attr-defined]
            right = dem_payload(value.geoid)  # type: ignore[attr-defined]
            if left[0] == right[0] == "constant":
                return ("constant", left[1] + right[1])
            if left[0] == "constant":
                left, right = right, left
            if right[0] == "constant":
                return ("raster", left[1] + right[1], *left[2:])
            if left[0] != right[0] or left[2:] != right[2:]:
                raise ValueError("composed RasterDEM grids must share affine metadata")
            return ("raster", left[1] + right[1], *left[2:])
        message = f"unsupported DEM type: {kind}"
        logger.error(message)
        raise TypeError(message)

    payload = dem_payload(dem) if dem is not None else ("none",)
    constant_dem_height = payload[1] if payload[0] == "constant" else None
    if payload[0] == "constant":
        height_seed_m = float(payload[1])
    elif payload[0] == "raster":
        raster_values = payload[1]
        finite = torch.isfinite(raster_values)
        if bool(finite.any().item()):
            height_seed_m = float(raster_values[finite].mean().item())
        else:
            logger.warning(
                "RasterDEM contains no finite heights; using 0 m height seed"
            )
            height_seed_m = 0.0
    else:
        height_seed_m = 0.0
    dem_samples = None
    dem_latitude_start = 0.0
    dem_longitude_start = 0.0
    dem_latitude_spacing = 1.0
    dem_longitude_spacing = 1.0
    if payload[0] == "raster":
        dem_samples = payload[1]
        dem_latitude_start, dem_longitude_start = payload[2], payload[3]
        dem_latitude_spacing, dem_longitude_spacing = payload[4], payload[5]

    def run_kernel(
        *values: object,
        geo2rdr_iterations: int,
        dynamic_iterations: bool,
    ) -> dict[str, object]:
        """Run the operation-specific device-resident solver."""
        latitude_or_azimuth, longitude_or_range, height = values
        if operation is Operation.GEO2RDR:
            return geo2rdr_kernel(
                latitude_or_azimuth,
                longitude_or_range,
                height,
                orbit_times,
                orbit_positions,
                orbit_velocities,
                sensing_offset_s=sensing_offset_s,
                azimuth_interval_s=model.azimuth_time_interval_s,
                starting_range_m=model.starting_slant_range_m,
                range_spacing_m=model.range_spacing_m,
                wavelength_m=model.wavelength_m,
                max_iter=geo2rdr_iterations,
                range_tol_m=settings.range_tol_m,
                doppler_tol_hz=settings.doppler_tol_hz,
                dynamic_iterations=dynamic_iterations,
                time_tol_s=settings.time_tol_s,
            )
        return rdr2geo_kernel(
            latitude_or_azimuth,
            longitude_or_range,
            height,
            orbit_times,
            orbit_positions,
            orbit_velocities,
            sensing_offset_s=sensing_offset_s,
            azimuth_interval_s=model.azimuth_time_interval_s,
            starting_range_m=model.starting_slant_range_m,
            range_spacing_m=model.range_spacing_m,
            wavelength_m=model.wavelength_m,
            look_sign=1.0 if model.look_direction == "right" else -1.0,
            max_iter=settings.max_iter,
            extra_iter=settings.extra_iter,
            range_tol_m=settings.range_tol_m,
            doppler_tol_hz=settings.doppler_tol_hz,
            dynamic_iterations=dynamic_iterations,
            dem_height_m=constant_dem_height,
            dem_samples=dem_samples,
            dem_latitude_start_deg=dem_latitude_start,
            dem_longitude_start_deg=dem_longitude_start,
            dem_latitude_spacing_deg=dem_latitude_spacing,
            dem_longitude_spacing_deg=dem_longitude_spacing,
            dem_iterations=settings.dem_iterations,
            dem_height_tol_m=settings.dem_height_tol_m,
        )

    total_geo2rdr_iterations = settings.max_iter + settings.extra_iter

    def kernel(*values: object) -> dict[str, object]:
        """Run the full-budget eager or fixed-shape solver."""
        # Dynamic masks are useful on CPU, where compacting lanes is cheap.
        # On CUDA they introduce host synchronizations and device indexing;
        # the fixed-shape loop keeps the entire prepared call asynchronous.
        staged_cpu_geo2rdr = (
            compile_kernel
            and operation is Operation.GEO2RDR
            and resolved_device.type == "cpu"
        )
        dynamic = resolved_device.type == "cpu" and (
            not compile_kernel or staged_cpu_geo2rdr
        )
        return run_kernel(
            *values,
            geo2rdr_iterations=total_geo2rdr_iterations,
            dynamic_iterations=dynamic,
        )

    def initialize_geo2rdr_state(*values: object) -> tuple[object, ...]:
        """Build the persistent CPU Geo2Rdr Newton state."""
        latitude, longitude, height = values
        return _geo2rdr_initialize(
            latitude,
            longitude,
            height,
            orbit_times,
            orbit_positions,
            orbit_velocities,
            sensing_offset_s=sensing_offset_s,
        )

    def finalize_geo2rdr_state(
        state: tuple[object, ...], *values: object
    ) -> dict[str, object]:
        """Publish the persistent CPU Geo2Rdr Newton state."""
        latitude, longitude, height = values
        return _geo2rdr_finalize(
            state,
            latitude,
            longitude,
            height,
            sensing_offset_s=sensing_offset_s,
            azimuth_interval_s=model.azimuth_time_interval_s,
            starting_range_m=model.starting_slant_range_m,
            range_spacing_m=model.range_spacing_m,
        )

    compiled_kernel: object | None = None
    compiled_first_stage: object | None = None
    state_finalizer: object | None = None
    if compile_kernel:
        if operation is Operation.GEO2RDR and resolved_device.type == "cpu":
            state_finalizer = finalize_geo2rdr_state

            def state_first_stage(*values: object) -> tuple[object, ...]:
                """Initialize and perform the first CPU Geo2Rdr transition."""
                return _geo2rdr_step(
                    initialize_geo2rdr_state(*values),
                    orbit_times,
                    orbit_positions,
                    orbit_velocities,
                    starting_range_m=model.starting_slant_range_m,
                    range_spacing_m=model.range_spacing_m,
                    wavelength_m=model.wavelength_m,
                    range_tol_m=settings.range_tol_m,
                    doppler_tol_hz=settings.doppler_tol_hz,
                )

            def state_step(*state_values: object) -> tuple[object, ...]:
                """Advance one persistent CPU Geo2Rdr Newton state."""
                return _geo2rdr_step(
                    tuple(state_values),
                    orbit_times,
                    orbit_positions,
                    orbit_velocities,
                    starting_range_m=model.starting_slant_range_m,
                    range_spacing_m=model.range_spacing_m,
                    wavelength_m=model.wavelength_m,
                    range_tol_m=settings.range_tol_m,
                    doppler_tol_hz=settings.doppler_tol_hz,
                )

            try:
                compile_kwargs = {
                    "mode": "reduce-overhead",
                    "fullgraph": True,
                    "dynamic": False,
                }
                compiled_first_stage = torch.compile(
                    state_first_stage, **compile_kwargs
                )
                compiled_kernel = torch.compile(state_step, **compile_kwargs)
            except Exception as error:
                logger.exception("failed to prepare compiled Torch geometry adapter")
                raise RuntimeError(
                    "Torch geometry compilation failed during preparation"
                ) from error

        else:
            try:
                compile_target = kernel
                compiled_kernel = torch.compile(
                    compile_target,
                    mode="reduce-overhead",
                    fullgraph=True,
                    dynamic=False,
                )
                sample = torch.zeros(
                    tuple(shape),
                    dtype=getattr(torch, canonical_dtype.split(".")[-1]),
                    device=resolved_device,
                )
                compiled_kernel(sample, sample, sample)
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
        height_seed_m=height_seed_m,
        compiled=compile_kernel,
        _kernel=kernel,
        _compiled_kernel=compiled_kernel,
        _compiled_first_stage=compiled_first_stage,
        _state_finalizer=state_finalizer,
    )


def _check_inputs(
    prepared: PreparedTorchGeometry, inputs: tuple[object, ...]
) -> tuple[torch.Tensor, ...]:
    """Validate input dtype/device and return broadcast device tensors."""
    import torch

    if prepared.operation == "rdr2geo" and len(inputs) == 2:
        inputs = (*inputs, prepared.height_seed_m)
    expected = 3
    if len(inputs) != expected:
        raise TypeError(f"{prepared.operation} expects {expected} input arrays")
    expected_dtype = getattr(torch, prepared.dtype.split(".")[-1])
    tensors: list[torch.Tensor] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            if item.dtype != expected_dtype:
                raise TypeError(f"Torch geometry inputs must use {prepared.dtype}")
            if _canonical_torch_device(item.device) != _canonical_torch_device(
                prepared.device
            ):
                raise TypeError(
                    f"Torch geometry inputs must use device {prepared.device}"
                )
            if not item.is_contiguous():
                raise ValueError("Torch geometry inputs must be contiguous")
            tensors.append(item)
        else:
            array = np.asarray(item)
            if array.dtype != np.dtype(np.float64):
                raise TypeError("NumPy geometry inputs must use float64")
            tensors.append(
                torch.as_tensor(array, dtype=expected_dtype, device=prepared.device)
            )
    broadcast = torch.broadcast_tensors(*tensors)
    if tuple(broadcast[0].shape) != prepared.shape:
        raise ValueError(
            f"prepared shape {prepared.shape} does not match input shape "
            f"{tuple(broadcast[0].shape)}"
        )
    # Expanded broadcast views have unstable strides for a shape-specialized
    # compiled graph.  Materialize one contiguous device buffer per argument.
    return tuple(value.contiguous() for value in broadcast)


def execute_torch_geometry(
    prepared: PreparedTorchGeometry,
    *inputs: object,
) -> TorchGeometryResult:
    """Execute a prepared adapter; no compilation is attempted here."""
    import torch

    arrays = _check_inputs(prepared, inputs)
    kernel = prepared._compiled_kernel if prepared.compiled else prepared._kernel
    total_iterations = prepared.settings.max_iter + prepared.settings.extra_iter
    uses_cpu_geo2rdr_state = (
        prepared.compiled
        and prepared.operation is Operation.GEO2RDR
        and _canonical_torch_device(prepared.device).type == "cpu"
    )
    if kernel is None:
        raise RuntimeError("prepared Torch geometry kernel is missing")
    with torch.no_grad():
        if uses_cpu_geo2rdr_state:
            finalizer = prepared._state_finalizer
            if finalizer is None:
                raise RuntimeError("prepared compiled Geo2Rdr finalizer is missing")

            compiled_first_stage = prepared._compiled_first_stage
            continuation = prepared._compiled_kernel
            if compiled_first_stage is None or continuation is None:
                raise RuntimeError("prepared compiled Geo2Rdr callables are missing")

            state = compiled_first_stage(*arrays)
            for _ in range(max(total_iterations - 1, 0)):
                if not bool(torch.any(_geo2rdr_active(state)).item()):
                    break
                state = continuation(*state)
            output = finalizer(state, *arrays)
        else:
            output = kernel(*arrays)
    host_output = {
        name: value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
        for name, value in output.items()
    }
    return TorchGeometryResult.from_transform(
        host_output,
        operation=prepared.operation,
        backend="torch_compiled" if prepared.compiled else "torch_eager",
        device=prepared.device,
        dtype=prepared.dtype,
        identity=prepared.identity.digest,
        range_tolerance=prepared.settings.range_tol_m,
        doppler_tolerance=prepared.settings.doppler_tol_hz,
        slant_range_tolerance=prepared.settings.range_tol_m,
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
    if height_m is None:
        arrays = np.broadcast_arrays(np.asarray(azimuth_index), np.asarray(range_index))
    else:
        arrays = np.broadcast_arrays(
            np.asarray(azimuth_index),
            np.asarray(range_index),
            np.asarray(height_m),
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
