# ruff: noqa: A002, EM101, EM102, TRY003

"""Public preparation and execution boundary for geometry v2.

The public boundary is intentionally small.  Preparation creates the eager
Torch adapter and, when requested, the compiled adapter or a native candidate.
Execution only selects an already prepared entry in the local dispatcher; it
never calls a compiler.  All results, including native results, are converted
through :class:`TransformResultV2`; callback-enabled results use the canonical
boundary normalizer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.backend_dispatch import (
    CandidateKey,
    Dispatcher,
    DispatchError,
    resolve_geometry_device,
)
from faninsar.processing.geometry.boundary import (
    BoundaryDecision,
    evaluate_canonical_boundary,
    normalize_result_boundary,
)
from faninsar.processing.geometry.native_v2.bindings import (
    ecef_from_native_outputs,
    result_from_native_outputs,
)
from faninsar.processing.geometry.torch_backends_v2 import (
    PreparedTorchGeometry,
    TorchGeometryResult,
    prepare_torch_geometry,
)
from faninsar.processing.geometry.v2 import (
    DeviceKey,
    ExecutionProfile,
    GeometryValidationError,
    Operation,
    SolverSettings,
    TransformResultV2,
    validate_array_span,
    validate_tensor_span,
)

if TYPE_CHECKING:
    from faninsar.processing.geometry import DEM
    from faninsar.processing.geometry.native_v2.builder import PreparedNativeCandidate
    from faninsar.processing.geometry.transforms import RadarGeometryModel
    from faninsar.processing.runtime.types import DeviceLike

logger = setup_logger(__name__)

BackendSelector: TypeAlias = Literal["native", "compile", "eager", "auto"]
NativeExecutor: TypeAlias = Callable[..., object] | ModuleType
NativeECEFExecutor: TypeAlias = Callable[..., object]
NativeContextInputs: TypeAlias = Mapping[str, object]
BoundaryCallback: TypeAlias = Callable[..., float | Sequence[float]]

_COMMON_NATIVE_CONTEXT_FIELDS = (
    "orbit_times",
    "orbit_positions",
    "orbit_velocities",
    "model_parameters",
    "look_right",
)
_RDR2GEO_NATIVE_CONTEXT_FIELDS = (
    *_COMMON_NATIVE_CONTEXT_FIELDS,
    "dem_values",
    "dem_metadata",
    "dem_height_bounds",
)
_CANONICAL_MODEL_PARAMETER_FIELDS = (
    "sensing_offset_s",
    "azimuth_interval_s",
    "starting_range_m",
    "range_spacing_m",
    "wavelength_m",
)


def _cuda_dem_metadata_abi_order(metadata: Sequence[float]) -> tuple[float, ...]:
    """Convert canonical latitude/longitude metadata to CUDA ABI order.

    Parameters
    ----------
    metadata : collections.abc.Sequence of float
        Canonical ``[latitude, longitude, latitude_spacing, longitude_spacing]``
        values.

    Returns
    -------
    tuple of float
        CUDA ABI order ``[longitude, latitude, longitude_spacing,
        latitude_spacing]``.

    """
    if len(metadata) != 4:
        message = "DEM metadata must contain exactly four values"
        logger.error(message)
        raise DispatchError(message)
    return (metadata[1], metadata[0], metadata[3], metadata[2])


def _resolve_native_entrypoint(
    executor: object,
    operation: Operation,
    device: DeviceKey,
    solver: SolverSettings,
    *,
    ecef: bool = False,
) -> Callable[..., object]:
    """Resolve a callable or loaded extension module to one native entrypoint.

    Parameters
    ----------
    executor : object
        A callable adapter, or a loaded native extension exposing operation
        entrypoints.
    operation : Operation
        Geometry operation being prepared.
    device : DeviceKey
        Target execution device.
    solver : SolverSettings
        Solver limits and tolerances used to complete the native scalar ABI.
    ecef : bool, optional
        Select the explicit CUDA ECEF entry point when true.

    Returns
    -------
    collections.abc.Callable
        The operation/device-specific native entrypoint.

    Raises
    ------
    DispatchError
        If an extension module does not expose the required entrypoint.

    Notes
    -----
    ``rdr2geo_cpu_dem`` is deliberately selected for every CPU radar-to-geo
    module call.  Its DEM context is part of the public native contract; using
    ``rdr2geo_cpu`` here silently changes the physical problem to an
    ellipsoid-only solve.

    """
    if callable(executor):
        if operation is Operation.RDR2GEO:
            message = (
                "rdr2geo native execution requires an extension module; "
                "pass the module so public dispatch can select rdr2geo_cpu_dem"
            )
            logger.error(message)
            raise DispatchError(message)
        return executor
    device_name = "cuda" if device.kind == "cuda" else "cpu"
    if operation is Operation.RDR2GEO:
        if ecef and device_name != "cuda":
            message = "native ECEF execution is currently CUDA-only"
            logger.error(message)
            raise DispatchError(message)
        symbol = (
            "rdr2geo_cuda_ecef"
            if ecef
            else ("rdr2geo_cuda" if device_name == "cuda" else "rdr2geo_cpu_dem")
        )
    else:
        symbol = "geo2rdr_cuda" if device_name == "cuda" else "geo2rdr_cpu"
    entry = getattr(executor, symbol, None)
    if not callable(entry):
        message = f"native geometry module is missing required entrypoint {symbol}"
        logger.error(message)
        raise DispatchError(message)
    if not isinstance(executor, ModuleType):
        return entry

    def invoke(*values: object) -> object:
        """Adapt public context fields to the extension's scalar ABI."""
        direct_inputs = values[:3]
        context = values[3:]
        orbit_times, orbit_positions, orbit_velocities, model_parameters, look_right = (
            context[:5]
        )

        def module_tensor(value: object) -> object:
            """Build an owner-backed contiguous float64 tensor for pybind."""
            import torch

            if isinstance(value, torch.Tensor):
                return value.to(dtype=torch.float64).contiguous()
            return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float64))

        if device.kind == "cpu":
            direct_inputs = tuple(module_tensor(value) for value in direct_inputs)
            orbit_times = module_tensor(orbit_times)
            orbit_positions = module_tensor(orbit_positions)
            orbit_velocities = module_tensor(orbit_velocities)
        if hasattr(model_parameters, "detach"):
            parameters = model_parameters.detach().cpu().tolist()
        else:
            parameters = np.asarray(model_parameters, dtype=np.float64).tolist()
        if len(parameters) != len(_CANONICAL_MODEL_PARAMETER_FIELDS):
            message = (
                "native model_parameters must follow the canonical five-field order"
            )
            logger.error(message)
            raise DispatchError(message)
        sensing, interval, starting_range, spacing, wavelength = map(float, parameters)
        if operation is Operation.RDR2GEO:
            dem_values, dem_metadata, dem_bounds = context[5:]
            if device.kind == "cpu":
                dem_values = module_tensor(dem_values)
            if hasattr(dem_metadata, "detach"):
                metadata = dem_metadata.detach().cpu().tolist()
            else:
                metadata = np.asarray(dem_metadata, dtype=np.float64).tolist()
            if hasattr(dem_bounds, "detach"):
                bounds = dem_bounds.detach().cpu().tolist()
            else:
                bounds = np.asarray(dem_bounds, dtype=np.float64).tolist()
            if device.kind == "cuda":
                metadata = list(_cuda_dem_metadata_abi_order(metadata))
                return entry(
                    *direct_inputs,
                    orbit_times,
                    orbit_positions,
                    orbit_velocities,
                    sensing,
                    interval,
                    starting_range,
                    spacing,
                    dem_values,
                    *metadata,
                    float(np.mean(direct_inputs[2].detach().cpu().numpy())),
                    *bounds,
                    wavelength,
                    solver.slant_range_tolerance_m,
                    solver.doppler_tolerance_hz,
                    solver.max_iter,
                    solver.extra_iter,
                    bool(look_right),
                )
            return entry(
                *direct_inputs,
                orbit_times,
                orbit_positions,
                orbit_velocities,
                sensing,
                interval,
                starting_range,
                spacing,
                wavelength,
                solver.max_iter,
                solver.extra_iter,
                solver.slant_range_tolerance_m,
                solver.doppler_tolerance_hz,
                bool(look_right),
                dem_values,
                *metadata,
                50,
                1.0e-3,
            )
        common = (
            *direct_inputs,
            orbit_times,
            orbit_positions,
            orbit_velocities,
            sensing,
            interval,
            starting_range,
            spacing,
            wavelength,
            solver.max_iter,
            solver.extra_iter,
            1.0e-6,
            solver.range_tolerance_m,
            solver.doppler_tolerance_hz,
        )
        if device.kind == "cuda":
            return entry(*common)
        return entry(*common, bool(look_right))

    return invoke


def _normalize_without_boundary_callback(
    result: TransformResultV2,
    *,
    invalid_mask: np.ndarray,
) -> TransformResultV2:
    """Publish a result without constructing scalar boundary decisions.

    Parameters
    ----------
    result : TransformResultV2
        Validated backend result.
    invalid_mask : numpy.ndarray
        Final coordinate-validity mask used by the public contract.

    Returns
    -------
    TransformResultV2
        The original result when no status needs changing, otherwise a result
        sharing untouched arrays and copying only fields that are normalized.

    Notes
    -----
    With no callback, a non-converged finite lane receives the same canonical
    status as the scalar path: ``decision_residual`` is NaN and
    ``boundary_rechecked`` is false.  Invalid lanes receive the usual public
    sentinels.  Both operations are expressed as array masks so large grids do
    not allocate one ``BoundaryDecision`` object per point.

    """
    flat_invalid = np.asarray(invalid_mask, dtype=bool).reshape(-1)
    flat_converged = np.asarray(result.converged, dtype=bool).reshape(-1)
    needs_status = (~flat_converged) & ~flat_invalid
    if not np.any(needs_status) and not np.any(flat_invalid):
        return result

    fields = {name: getattr(result, name) for name in result.fields}
    if np.any(needs_status):
        boundary_rechecked = np.array(result.boundary_rechecked, copy=True)
        boundary_rechecked.reshape(-1)[needs_status] = False
        decision_residual = np.array(result.decision_residual, copy=True)
        decision_residual.reshape(-1)[needs_status] = np.nan
        fields.update(
            {
                "boundary_rechecked": boundary_rechecked,
                "decision_residual": decision_residual,
            }
        )

    if np.any(flat_invalid):
        for name in (
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
        ):
            value = np.array(fields[name], copy=True)
            value.reshape(-1)[flat_invalid] = np.nan
            fields[name] = value
        value = np.array(fields["iterations"], copy=True)
        value.reshape(-1)[flat_invalid] = -1
        fields["iterations"] = value
        for name in ("converged", "max_iter_exhausted", "boundary_rechecked"):
            value = np.array(fields[name], copy=True)
            value.reshape(-1)[flat_invalid] = False
            fields[name] = value

    return TransformResultV2(**fields)


def _validate_native_context_inputs(
    operation: Operation,
    context: NativeContextInputs | None,
    expected_device: DeviceKey,
) -> tuple[object, ...]:
    """Validate every closed-over array and return ABI argument order."""
    if context is None:
        raise DispatchError(
            "native geometry requires explicit operation-specific context inputs"
        )
    required = (
        _RDR2GEO_NATIVE_CONTEXT_FIELDS
        if operation is Operation.RDR2GEO
        else _COMMON_NATIVE_CONTEXT_FIELDS
    )
    missing = tuple(name for name in required if name not in context)
    unknown = tuple(name for name in context if name not in required)
    if missing:
        raise DispatchError(
            "native geometry context is missing required fields: " + ", ".join(missing)
        )
    if unknown:
        raise DispatchError(
            "native geometry context contains unknown fields: " + ", ".join(unknown)
        )

    expected_shapes: dict[str, tuple[int, ...] | None] = {
        "orbit_times": None,
        "orbit_positions": None,
        "orbit_velocities": None,
        "model_parameters": (5,),
    }
    if operation is Operation.RDR2GEO:
        expected_shapes.update(
            {
                # The native sampler consumes a six-point stencil from any
                # contiguous 2-D raster.  Keeping the full raster resident
                # avoids forcing callers to manufacture one tile per call.
                "dem_values": None,
                "dem_metadata": (4,),
                "dem_height_bounds": (2,),
            }
        )
    validated: dict[str, object] = {}
    torch_type = None
    torch_module = None
    with suppress(ImportError):
        import torch

        torch_type = torch.Tensor
        torch_module = torch
    for name in required:
        value = context[name]
        if name == "look_right":
            if not isinstance(value, (bool, np.bool_)):
                raise GeometryValidationError("look_right must be a boolean")
            validated[name] = bool(value)
            continue
        shape = expected_shapes[name]
        finite_required = name != "dem_values"
        if torch_type is not None and isinstance(value, torch_type):
            validate_tensor_span(
                value,
                expected_dtype=torch_module.float64,
                expected_shape=shape,
                expected_device=expected_device,
                require_finite=finite_required,
                name=f"native context {name}",
            )
            if name == "dem_values" and (value.ndim != 2 or min(value.shape) < 6):
                raise GeometryValidationError(
                    "native context dem_values must be a 2-D raster with both "
                    "dimensions at least 6"
                )
            if name == "dem_values" and bool(torch_module.isinf(value).any().item()):
                raise GeometryValidationError(
                    "native context dem_values must not contain infinities"
                )
            validated[name] = value
            continue
        if not isinstance(value, np.ndarray):
            raise GeometryValidationError(
                f"native context {name} must be a NumPy array or Torch tensor"
            )
        validate_array_span(
            value,
            expected_dtype=np.dtype(np.float64),
            expected_shape=shape,
            expected_device=DeviceKey.cpu(),
            require_finite=finite_required,
            name=f"native context {name}",
        )
        if name == "dem_values" and (value.ndim != 2 or min(value.shape) < 6):
            raise GeometryValidationError(
                "native context dem_values must be a 2-D raster with both "
                "dimensions at least 6"
            )
        if name == "dem_values" and bool(np.isinf(value).any()):
            raise GeometryValidationError(
                "native context dem_values must not contain infinities"
            )
        if expected_device.kind != "cpu":
            raise GeometryValidationError(
                f"native context {name} NumPy owner cannot prove CUDA allocation"
            )
        validated[name] = value
    orbit_times = validated["orbit_times"]
    orbit_positions = validated["orbit_positions"]
    orbit_velocities = validated["orbit_velocities"]
    orbit_length = int(orbit_times.shape[0])  # type: ignore[union-attr]
    if orbit_length < 2:
        raise GeometryValidationError("native context orbit requires two samples")
    if (
        tuple(orbit_positions.shape) != (orbit_length, 3)  # type: ignore[union-attr]
        or tuple(orbit_velocities.shape) != (orbit_length, 3)  # type: ignore[union-attr]
    ):
        raise GeometryValidationError(
            "native context orbit positions and velocities must have shape (N, 3)"
        )
    return tuple(validated[name] for name in required)


def _device_key(
    device: DeviceLike | None,
    physical_uuid: str | None,
    mig_uuid: str | None = None,
) -> DeviceKey:
    """Build a foundation device identity from a public device request."""
    resolved = resolve_geometry_device(device)
    if resolved.type == "cuda":
        if not physical_uuid:
            raise DispatchError("CUDA geometry preparation requires a physical UUID")
        return DeviceKey.cuda(physical_uuid, mig_uuid=mig_uuid)
    if resolved.type != "cpu":
        raise DispatchError(f"geometry v2 only supports CPU and CUDA, got {device!r}")
    return DeviceKey.cpu()


def _validate_public_inputs(
    operation: Operation,
    shape: tuple[int, ...],
    inputs: tuple[object, ...],
    expected_device: DeviceKey,
    *,
    default_height_m: float = 0.0,
    target_device: DeviceLike | None = "cpu",
) -> tuple[object, ...]:
    """Validate host spans before handing arrays to a native callback."""
    if len(shape) > 2:
        raise GeometryValidationError(
            "native geometry supports only one- or two-dimensional inputs"
        )
    if operation is Operation.RDR2GEO and len(inputs) == 2:
        if expected_device.kind == "cuda":
            import torch

            inputs = (
                *inputs,
                torch.full(
                    shape,
                    default_height_m,
                    dtype=torch.float64,
                    device=target_device,
                ),
            )
        else:
            inputs = (*inputs, np.full(shape, default_height_m, dtype=np.float64))
    if len(inputs) != 3:
        raise TypeError(f"{operation} expects three input arrays")
    arrays: list[object] = []
    torch_type = None
    torch_module = None
    try:
        import torch

        torch_type = torch.Tensor
        torch_module = torch
    except ImportError:
        pass
    for index, value in enumerate(inputs):
        if torch_type is not None and isinstance(value, torch_type):
            validate_tensor_span(
                value,
                expected_dtype=torch_module.float64,
                expected_shape=shape,
                expected_device=expected_device,
                name=f"geometry input {index}",
            )
            arrays.append(value.reshape(-1) if expected_device.kind == "cpu" else value)
            continue
        if not isinstance(value, np.ndarray):
            raise TypeError("native geometry inputs must be NumPy arrays")
        if expected_device.kind == "cuda":
            import torch

            if value.dtype != np.dtype(np.float64):
                message = f"geometry input {index} must have dtype float64"
                logger.error(message)
                raise GeometryValidationError(message)
            try:
                normalized_value = torch.as_tensor(
                    value, dtype=torch.float64, device=target_device
                ).contiguous()
            except Exception as error:
                message = "native CUDA inputs could not be copied to the target device"
                logger.exception(message)
                raise GeometryValidationError(message) from error
            validate_tensor_span(
                normalized_value,
                expected_dtype=torch.float64,
                expected_shape=shape,
                expected_device=expected_device,
                name=f"geometry input {index}",
            )
            arrays.append(normalized_value)
            continue
        validate_array_span(
            value,
            expected_dtype=np.dtype(np.float64),
            expected_shape=shape,
            expected_device=DeviceKey.cpu(),
            name=f"geometry input {index}",
        )
        arrays.append(value.reshape(-1) if expected_device.kind == "cpu" else value)
    return tuple(arrays)


def _native_public_result(
    outputs: object,
    *,
    operation: Operation,
    boundary_callback: BoundaryCallback | None = None,
    range_tolerance_m: float = 0.01,
    doppler_tolerance_hz: float = 0.1,
    slant_range_tolerance_m: float = 0.01,
    iteration_budget: int | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> TransformResultV2:
    """Convert and canonically normalize one backend result."""
    if isinstance(outputs, TransformResultV2):
        result = TransformResultV2.from_arrays(
            {name: getattr(outputs, name) for name in outputs.fields},
            operation=operation,
            invalid_mask=~np.isfinite(
                np.asarray(outputs.latitude_deg, dtype=np.float64)
            ),
        )
    else:
        if not isinstance(outputs, Sequence):
            raise DispatchError("native geometry executor did not return a sequence")
        result = result_from_native_outputs(outputs, operation=operation)

    if output_shape is not None and result.latitude_deg.shape != output_shape:
        fields = {
            name: np.asarray(getattr(result, name)).reshape(output_shape)
            for name in result.fields
        }
        result = TransformResultV2.from_arrays(
            fields,
            operation=operation,
            invalid_mask=~np.isfinite(
                np.asarray(fields["latitude_deg"], dtype=np.float64)
            ),
        )

    invalid = ~np.isfinite(result.latitude_deg) | ~np.isfinite(result.longitude_deg)
    invalid |= ~np.isfinite(result.height_m)
    if boundary_callback is None:
        return _normalize_without_boundary_callback(result, invalid_mask=invalid)

    valid = np.asarray(result.converged, dtype=bool)
    flat_valid = valid.reshape(-1)
    decisions: list[BoundaryDecision] = []
    attempts: list[int] = []
    for index, is_valid in enumerate(flat_valid):
        if not is_valid:
            decisions.append(BoundaryDecision(False, False, float("nan"), float("nan")))
            attempts.append(-1)
            continue
        index_tuple = np.unravel_index(index, valid.shape)
        attempt = int(result.iterations[index_tuple])
        decision_residual = float(result.decision_residual[index_tuple])
        if operation is Operation.GEO2RDR:
            residual: float | tuple[float, float] = (
                float(result.residual_range_m[index_tuple]),
                float(result.residual_doppler_hz[index_tuple]),
            )
            if boundary_callback is None:
                decisions.append(
                    BoundaryDecision(
                        bool(result.converged[index_tuple]),
                        bool(result.boundary_rechecked[index_tuple]),
                        decision_residual,
                        max(
                            abs(float(result.residual_range_m[index_tuple]))
                            / range_tolerance_m,
                            abs(float(result.residual_doppler_hz[index_tuple]))
                            / doppler_tolerance_hz,
                        ),
                    )
                )
            else:
                coords = tuple(
                    np.asarray([getattr(result, name)[index_tuple]], dtype=np.float64)
                    for name in ("latitude_deg", "longitude_deg", "height_m")
                )
                decision = evaluate_canonical_boundary(
                    operation,
                    attempt,
                    coords,
                    boundary_callback,
                    decision_residual=residual,
                    range_tolerance_m=range_tolerance_m,
                    doppler_tolerance_hz=doppler_tolerance_hz,
                )
                while (
                    decision.boundary_rechecked
                    and not decision.converged
                    and attempt < (iteration_budget or attempt)
                ):
                    attempt += 1
                    decision = evaluate_canonical_boundary(
                        operation,
                        attempt,
                        coords,
                        boundary_callback,
                        decision_residual=residual,
                        range_tolerance_m=range_tolerance_m,
                        doppler_tolerance_hz=doppler_tolerance_hz,
                    )
                decisions.append(decision)
        elif boundary_callback is None:
            decisions.append(
                BoundaryDecision(
                    bool(result.converged[index_tuple]),
                    bool(result.boundary_rechecked[index_tuple]),
                    decision_residual,
                    abs(float(result.residual_range_m[index_tuple]))
                    / slant_range_tolerance_m,
                )
            )
        else:
            coords = tuple(
                np.asarray([getattr(result, name)[index_tuple]], dtype=np.float64)
                for name in ("latitude_deg", "longitude_deg", "height_m")
            )
            decision = evaluate_canonical_boundary(
                operation,
                attempt,
                coords,
                boundary_callback,
                decision_residual=decision_residual,
                slant_range_tolerance_m=slant_range_tolerance_m,
            )
            while (
                decision.boundary_rechecked
                and not decision.converged
                and attempt < (iteration_budget or attempt)
            ):
                attempt += 1
                decision = evaluate_canonical_boundary(
                    operation,
                    attempt,
                    coords,
                    boundary_callback,
                    decision_residual=decision_residual,
                    slant_range_tolerance_m=slant_range_tolerance_m,
                )
            decisions.append(decision)
        attempts.append(attempt)
    normalized = normalize_result_boundary(result, decisions, invalid_mask=invalid)
    if iteration_budget is None:
        return normalized
    fields = {name: getattr(normalized, name).copy() for name in normalized.fields}
    flat_iterations = fields["iterations"].reshape(-1)
    flat_exhausted = fields["max_iter_exhausted"].reshape(-1)
    for index, decision in enumerate(decisions):
        if attempts[index] > 0:
            flat_iterations[index] = attempts[index]
        if decision.boundary_rechecked and not decision.converged:
            flat_exhausted[index] = True
    return TransformResultV2(**dict(fields))


_normalize_public_result = _native_public_result


def _key(
    operation: Operation,
    backend: Literal["native", "compile"],
    prepared: PreparedTorchGeometry,
    profile: ExecutionProfile,
    native_candidate: PreparedNativeCandidate | None = None,
) -> CandidateKey:
    """Build a complete foundation candidate key for one prepared backend."""
    if native_candidate is not None:
        plan = native_candidate.plan
        expected_backend = "cuda" if profile.device.kind == "cuda" else "cpu"
        if plan.operation.value != operation.value:
            raise DispatchError("native candidate operation does not match preparation")
        if plan.backend.value != expected_backend:
            raise DispatchError("native candidate device does not match preparation")
        if native_candidate.artifact is None:
            raise DispatchError(
                "prepared native candidate requires an artifact manifest"
            )

        def digest_paths(paths: Sequence[object]) -> str:
            digest = hashlib.sha256()
            for value in paths:
                path = str(value)
                digest.update(path.encode())
                with suppress(FileNotFoundError, IsADirectoryError, OSError):
                    digest.update(Path(path).read_bytes())
            return digest.hexdigest()

        source_digest = digest_paths(plan.sources)
        artifact_digest = (
            digest_paths((native_candidate.artifact,))
            if native_candidate.artifact
            else ""
        )
        toolchain_digest = digest_paths((*plan.compile_flags, *plan.link_flags))
        runtime_digest = hashlib.sha256(str(plan.runtime_name).encode()).hexdigest()
        abi_digest = hashlib.sha256(
            b"faninsar.geometry.native_v2.14-field.v1"
        ).hexdigest()
    else:
        source_digest = toolchain_digest = runtime_digest = artifact_digest = (
            abi_digest
        ) = ""
    return CandidateKey(
        operation=operation,
        backend=backend,
        device=profile.device,
        dtype=prepared.dtype,
        shape=prepared.shape,
        solver=prepared.settings.operation_settings(operation).solver,
        orbit_digest=prepared.identity.orbit_digest,
        dem_digest=prepared.identity.dem_digest,
        model_digest=prepared.identity.model_digest,
        source_digest=source_digest,
        toolchain_digest=toolchain_digest,
        runtime_digest=runtime_digest,
        artifact_digest=artifact_digest,
        abi_digest=abi_digest,
        support_contract_digest=prepared.identity.settings_digest,
        profile=profile,
    )


def _validate_native_manifest(
    manifest: CandidateKey,
    expected: CandidateKey,
    *,
    exact_executable: bool,
) -> CandidateKey:
    """Validate an explicit native manifest against the prepared executable."""
    if manifest.backend != "native":
        raise DispatchError("native manifest must identify the native backend")
    if exact_executable and manifest != expected:
        raise DispatchError("native manifest does not match prepared executable")
    if (
        not exact_executable
        and manifest.scientific_identity != expected.scientific_identity
    ):
        raise DispatchError("native manifest does not match prepared geometry")
    for name in (
        "source_digest",
        "toolchain_digest",
        "runtime_digest",
        "artifact_digest",
        "abi_digest",
        "support_contract_digest",
    ):
        if not getattr(manifest, name):
            raise DispatchError(f"native manifest is missing {name}")
    return manifest


def _normalize_ecef_result(
    result: object, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a raw native or already-materialized ECEF result."""
    if isinstance(result, Sequence) and len(result) == 17:
        values = ecef_from_native_outputs(result)
    elif isinstance(result, Sequence) and len(result) == 3:
        values = tuple(np.asarray(value) for value in result)
        if any(value.dtype != np.dtype(np.float64) for value in values):
            message = "ECEF result arrays must have float64 dtype"
            logger.error(message)
            raise DispatchError(message)
    else:
        message = "ECEF executor returned neither a 17-field nor 3-array result"
        logger.error(message)
        raise DispatchError(message)
    if any(value.shape != shape for value in values):
        message = "ECEF result arrays have an incompatible shape"
        logger.error(message)
        raise DispatchError(message)
    return values


@dataclass(frozen=True, slots=True)
class PreparedGeometry:
    """Prepared public geometry operation with deterministic backend dispatch."""

    operation: Operation
    model: RadarGeometryModel
    shape: tuple[int, ...]
    eager: PreparedTorchGeometry
    dispatcher: Dispatcher
    device: str
    dtype: str
    native_key: CandidateKey | None = None
    compile_key: CandidateKey | None = None
    fallback_key: CandidateKey | None = None
    _native_ecef_execute: NativeECEFExecutor | None = None
    _compiled_torch: PreparedTorchGeometry | None = None
    profile: ExecutionProfile | None = None

    def execute(
        self,
        *inputs: object,
        selector: BackendSelector = "auto",
    ) -> TransformResultV2:
        """Execute the selected prepared backend without compiling."""
        result = self.dispatcher.dispatch(
            self._key_for_selector(selector), selector, *inputs
        )
        if isinstance(result, TorchGeometryResult):
            return _normalize_public_result(result.transform, operation=self.operation)
        if not isinstance(result, TransformResultV2):
            raise DispatchError("geometry backend returned an unknown result type")
        return result.validate()

    def _key_for_selector(self, selector: BackendSelector) -> CandidateKey:
        """Return the exact lookup key used by a selector."""
        if selector == "native" and self.native_key is not None:
            return self.native_key
        if selector == "compile" and self.compile_key is not None:
            return self.compile_key
        if selector in ("eager", "auto"):
            return (
                self.native_key
                or self.compile_key
                or self.fallback_key
                or _key(
                    self.operation,
                    "compile",
                    self.eager,
                    self.profile or ExecutionProfile.cpu(),
                )
            )
        raise DispatchError(f"no exact prepared {selector} candidate")

    def execute_ecef(
        self,
        *inputs: object,
        selector: BackendSelector = "auto",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute Rdr2Geo and transfer only typed ECEF outputs.

        Parameters
        ----------
        *inputs : object
            Operation inputs in the same order as :meth:`execute`.
        selector : {"native", "compile", "eager", "auto"}, optional
            Backend to use.  Native CUDA uses the direct seventeen-field ECEF
            ABI; Torch backends use their device-resident ECEF path.

        Returns
        -------
        tuple of numpy.ndarray
            Host ECEF ``(x, y, z)`` arrays.

        Raises
        ------
        DispatchError
            If the selected backend has no typed ECEF route.

        """
        if self.operation is not Operation.RDR2GEO:
            message = "execute_ecef is only available for rdr2geo"
            logger.error(message)
            raise DispatchError(message)
        if selector not in ("native", "compile", "eager", "auto"):
            message = f"unknown ECEF selector: {selector}"
            logger.error(message)
            raise DispatchError(message)
        alternate: dict[str, Callable[..., object]] = {}
        if self._native_ecef_execute is not None:
            alternate["native"] = self._native_ecef_execute
        if self._compiled_torch is not None:
            alternate["compile"] = self._compiled_torch.execute_ecef
        return self.dispatcher.dispatch_alternate(
            self._key_for_selector(selector),
            selector,
            alternate,
            self.eager.execute_ecef,
            lambda value: _normalize_ecef_result(value, self.shape),
            *inputs,
        )


def prepare_geometry(
    operation: Operation | str,
    model: RadarGeometryModel,
    *,
    shape: Sequence[int],
    device: DeviceLike | None = "cpu",
    dtype: str | None = None,
    dem: DEM | None = None,
    settings: SolverSettings | None = None,
    native_executor: NativeExecutor | None = None,
    native_candidate: PreparedNativeCandidate | None = None,
    native_context_inputs: NativeContextInputs | None = None,
    native_key: CandidateKey | None = None,
    native_correctness_qualified: bool = False,
    native_performance_eligible: bool = False,
    native_ecef_correctness_qualified: bool = False,
    compile: bool = False,
    compile_correctness_qualified: bool = True,
    compile_performance_eligible: bool = False,
    physical_uuid: str | None = None,
    mig_uuid: str | None = None,
    boundary_callback: BoundaryCallback | None = None,
) -> PreparedGeometry:
    """Prepare one public geometry operation.

    Preparation is the only place where Torch compilation is permitted.  A
    native executor must be supplied by an operation-aware builder or a caller
    that has already loaded the exact extension; this function never compiles
    native code implicitly.
    """
    op = Operation(operation)
    normalized_shape = tuple(int(value) for value in shape)
    if not normalized_shape or any(value <= 0 for value in normalized_shape):
        raise ValueError("shape must contain positive dimensions")
    solver = settings or SolverSettings()
    resolved_device = resolve_geometry_device(device)
    torch_range_tolerance = (
        solver.slant_range_tolerance_m
        if op is Operation.RDR2GEO
        else solver.range_tolerance_m
    )
    profile = ExecutionProfile(
        _device_key(resolved_device, physical_uuid, mig_uuid),
        thread_count=None,
    )
    eager = prepare_torch_geometry(
        op,
        model,
        shape=normalized_shape,
        dem=dem,
        device=resolved_device,
        dtype=dtype,
        max_iter=solver.max_iter,
        extra_iter=solver.extra_iter,
        range_tol_m=torch_range_tolerance,
        doppler_tol_hz=solver.doppler_tolerance_hz,
        compile_kernel=False,
    )
    compiled = (
        prepare_torch_geometry(
            op,
            model,
            shape=normalized_shape,
            dem=dem,
            device=resolved_device,
            dtype=dtype,
            max_iter=solver.max_iter,
            extra_iter=solver.extra_iter,
            range_tol_m=torch_range_tolerance,
            doppler_tol_hz=solver.doppler_tolerance_hz,
            compile_kernel=True,
        )
        if compile
        else None
    )

    def eager_execute(*args: object) -> TransformResultV2:
        """Execute eager reference geometry for the public dispatcher."""
        return _normalize_public_result(
            eager.execute(*args).transform,
            operation=op,
            boundary_callback=boundary_callback,
            range_tolerance_m=solver.range_tolerance_m,
            doppler_tolerance_hz=solver.doppler_tolerance_hz,
            slant_range_tolerance_m=solver.slant_range_tolerance_m,
            iteration_budget=solver.budget,
        )

    dispatcher = Dispatcher(eager_execute)
    registered_native_key: CandidateKey | None = None
    compile_key: CandidateKey | None = None
    native_ecef_execute: NativeECEFExecutor | None = None
    fallback_key = _key(op, "compile", eager, profile)
    if compiled is not None:
        compile_key = _key(op, "compile", compiled, profile)
        dispatcher.register(
            compile_key,
            lambda *args: _normalize_public_result(
                compiled.execute(*args).transform,
                operation=op,
                boundary_callback=boundary_callback,
                range_tolerance_m=solver.range_tolerance_m,
                doppler_tolerance_hz=solver.doppler_tolerance_hz,
                slant_range_tolerance_m=solver.slant_range_tolerance_m,
                iteration_budget=solver.budget,
            ),
            correctness_qualified=compile_correctness_qualified,
            performance_eligible=compile_performance_eligible,
            result_validator=lambda value: _native_public_result(
                value,
                operation=op,
                boundary_callback=boundary_callback,
                range_tolerance_m=solver.range_tolerance_m,
                doppler_tolerance_hz=solver.doppler_tolerance_hz,
                slant_range_tolerance_m=solver.slant_range_tolerance_m,
                iteration_budget=solver.budget,
            ),
        )
    if native_executor is not None or native_candidate is not None:
        if op is Operation.RDR2GEO and native_candidate is not None:
            message = (
                "rdr2geo native candidates are unsupported; pass a loaded "
                "extension module as native_executor so public dispatch selects "
                "rdr2geo_cpu_dem"
            )
            logger.error(message)
            raise DispatchError(message)
        derived_key = _key(op, "native", eager, profile, native_candidate)
        if (
            native_executor is not None
            and native_candidate is None
            and native_key is None
        ):
            raise DispatchError(
                "native_executor requires an explicit CandidateKey manifest"
            )
        registered_native_key = (
            _validate_native_manifest(
                native_key,
                derived_key,
                exact_executable=native_candidate is not None,
            )
            if native_key is not None
            else derived_key
        )
        candidate_context = (
            native_candidate.native_context_inputs
            if native_candidate is not None
            else None
        )
        if native_context_inputs is not None and candidate_context is not None:
            raise DispatchError(
                "native context must be supplied either by candidate or preparation"
            )
        context_inputs = native_context_inputs or candidate_context
        _validate_native_context_inputs(op, context_inputs, profile.device)
        native_entry = (
            _resolve_native_entrypoint(native_executor, op, profile.device, solver)
            if native_executor is not None
            else None
        )
        native_ecef_entry = None
        if (
            native_executor is not None
            and op is Operation.RDR2GEO
            and profile.device.kind == "cuda"
        ):
            native_ecef_entry = _resolve_native_entrypoint(
                native_executor, op, profile.device, solver, ecef=True
            )

        def execute_native(*args: object) -> TransformResultV2:
            """Execute and centrally validate the supplied native entry point."""
            native_inputs = _validate_public_inputs(
                op,
                normalized_shape,
                args,
                profile.device,
                default_height_m=eager.height_seed_m,
                target_device=resolved_device,
            )
            context_values = _validate_native_context_inputs(
                op, context_inputs, profile.device
            )
            entry = native_entry
            if entry is None and native_candidate is not None:
                return _native_public_result(
                    native_candidate.dispatch(*native_inputs, *context_values),
                    operation=op,
                    boundary_callback=boundary_callback,
                    range_tolerance_m=solver.range_tolerance_m,
                    doppler_tolerance_hz=solver.doppler_tolerance_hz,
                    slant_range_tolerance_m=solver.slant_range_tolerance_m,
                    iteration_budget=solver.budget,
                    output_shape=normalized_shape,
                )
            if entry is None:
                raise DispatchError("native geometry executor is missing")
            return _native_public_result(
                entry(*native_inputs, *context_values),
                operation=op,
                boundary_callback=boundary_callback,
                range_tolerance_m=solver.range_tolerance_m,
                doppler_tolerance_hz=solver.doppler_tolerance_hz,
                slant_range_tolerance_m=solver.slant_range_tolerance_m,
                iteration_budget=solver.budget,
                output_shape=normalized_shape,
            )

        def execute_native_ecef(*args: object) -> object:
            """Execute the typed seventeen-field native ECEF entry point."""
            if native_ecef_entry is None:
                logger.error("native CUDA ECEF executor is missing")
                raise DispatchError("native CUDA ECEF executor is missing")
            native_inputs = _validate_public_inputs(
                op,
                normalized_shape,
                args,
                profile.device,
                default_height_m=eager.height_seed_m,
                target_device=resolved_device,
            )
            context_values = _validate_native_context_inputs(
                op, context_inputs, profile.device
            )
            return native_ecef_entry(*native_inputs, *context_values)

        if native_ecef_entry is not None:
            native_ecef_execute = execute_native_ecef

        dispatcher.register(
            registered_native_key,
            execute_native,
            correctness_qualified=native_correctness_qualified,
            performance_eligible=native_performance_eligible,
            ecef_correctness_qualified=native_ecef_correctness_qualified,
        )
    return PreparedGeometry(
        op,
        model,
        normalized_shape,
        eager,
        dispatcher,
        eager.device,
        eager.dtype,
        registered_native_key,
        compile_key,
        fallback_key,
        native_ecef_execute,
        compiled,
        profile,
    )


def execute_geometry(
    prepared: PreparedGeometry,
    *inputs: object,
    selector: BackendSelector = "auto",
) -> TransformResultV2:
    """Execute a :func:`prepare_geometry` result without implicit compilation."""
    return prepared.execute(*inputs, selector=selector)


prepare_geometry_v2 = prepare_geometry
execute_geometry_v2 = execute_geometry

__all__ = [
    "BackendSelector",
    "NativeContextInputs",
    "PreparedGeometry",
    "execute_geometry",
    "execute_geometry_v2",
    "prepare_geometry",
    "prepare_geometry_v2",
]
