# ruff: noqa: A002, EM101, EM102, TRY003

"""Public preparation and execution boundary for geometry v2.

The public boundary is intentionally small.  Preparation creates the eager
Torch adapter and, when requested, the compiled adapter or a native candidate.
Execution only selects an already prepared entry in the local dispatcher; it
never calls a compiler.  All results, including native results, are converted
through :class:`TransformResultV2` and the canonical boundary normalizer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.backend_dispatch import (
    CandidateKey,
    Dispatcher,
    DispatchError,
)
from faninsar.processing.geometry.boundary import (
    BoundaryDecision,
    evaluate_canonical_boundary,
    normalize_result_boundary,
)
from faninsar.processing.geometry.native_v2.bindings import result_from_native_outputs
from faninsar.processing.geometry.torch_backends_v2 import (
    PreparedTorchGeometry,
    TorchGeometryResult,
    prepare_torch_geometry,
)
from faninsar.processing.geometry.v2 import (
    DeviceKey,
    ExecutionProfile,
    Operation,
    SolverSettings,
    TransformResultV2,
    validate_array_span,
    validate_tensor_span,
)

if TYPE_CHECKING:
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.geometry.native_v2.builder import PreparedNativeCandidate
    from faninsar.processing.geometry.transforms import RadarGeometryModel

logger = setup_logger(__name__)

BackendSelector: TypeAlias = Literal["native", "compile", "eager", "auto"]
NativeExecutor: TypeAlias = Callable[..., object]
BoundaryCallback: TypeAlias = Callable[..., float | Sequence[float]]


def _device_key(
    device: str, physical_uuid: str | None, mig_uuid: str | None = None
) -> DeviceKey:
    """Build a foundation device identity from a Torch device string."""
    if str(device).split(":", 1)[0] == "cuda":
        if not physical_uuid:
            raise DispatchError("CUDA geometry preparation requires a physical UUID")
        return DeviceKey.cuda(physical_uuid, mig_uuid=mig_uuid)
    if str(device).split(":", 1)[0] != "cpu":
        raise DispatchError(f"geometry v2 only supports CPU and CUDA, got {device!r}")
    return DeviceKey.cpu()


def _validate_public_inputs(
    operation: Operation,
    shape: tuple[int, ...],
    inputs: tuple[object, ...],
    expected_device: DeviceKey,
) -> tuple[object, ...]:
    """Validate host spans before handing arrays to a native callback."""
    if operation is Operation.RDR2GEO and len(inputs) == 2:
        inputs = (*inputs, np.zeros(shape, dtype=np.float64))
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
            arrays.append(value)
            continue
        if not isinstance(value, np.ndarray):
            raise TypeError("native geometry inputs must be NumPy arrays")
        validate_array_span(
            value,
            expected_dtype=np.dtype(np.float64),
            expected_shape=shape,
            expected_device=DeviceKey.cpu(),
            name=f"geometry input {index}",
        )
        arrays.append(value)
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
) -> TransformResultV2:
    """Convert and canonically normalize one backend result."""
    if isinstance(outputs, TransformResultV2):
        result = TransformResultV2.from_arrays(
            {name: getattr(outputs, name) for name in outputs.fields},
            operation=operation,
            invalid_mask=~np.asarray(outputs.converged, dtype=bool),
        )
    else:
        if not isinstance(outputs, Sequence):
            raise DispatchError("native geometry executor did not return a sequence")
        result = result_from_native_outputs(outputs, operation=operation)

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
    normalized = normalize_result_boundary(result, decisions, invalid_mask=~valid)
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
) -> CandidateKey:
    """Validate an explicit native manifest against the prepared executable."""
    if manifest.backend != "native":
        raise DispatchError("native manifest must identify the native backend")
    if (
        manifest != expected
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
                or _key(
                    self.operation,
                    "compile",
                    self.eager,
                    ExecutionProfile.cpu(),
                )
            )
        raise DispatchError(f"no exact prepared {selector} candidate")


def prepare_geometry(
    operation: Operation | str,
    model: RadarGeometryModel,
    *,
    shape: Sequence[int],
    device: str = "cpu",
    dtype: str | None = None,
    dem: DEMSampler | None = None,
    settings: SolverSettings | None = None,
    native_executor: NativeExecutor | None = None,
    native_candidate: PreparedNativeCandidate | None = None,
    native_key: CandidateKey | None = None,
    native_correctness_qualified: bool = False,
    native_performance_eligible: bool = False,
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
    profile = ExecutionProfile(
        _device_key(device, physical_uuid, mig_uuid),
        thread_count=None,
    )
    eager = prepare_torch_geometry(
        op,
        model,
        shape=normalized_shape,
        dem=dem,
        device=device,
        dtype=dtype,
        max_iter=solver.max_iter,
        extra_iter=solver.extra_iter,
        range_tol_m=solver.range_tolerance_m,
        doppler_tol_hz=solver.doppler_tolerance_hz,
        compile_kernel=False,
    )
    compiled = (
        prepare_torch_geometry(
            op,
            model,
            shape=normalized_shape,
            dem=dem,
            device=device,
            dtype=dtype,
            max_iter=solver.max_iter,
            extra_iter=solver.extra_iter,
            range_tol_m=solver.range_tolerance_m,
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
            _validate_native_manifest(native_key, derived_key)
            if native_key is not None
            else derived_key
        )

        def execute_native(*args: object) -> TransformResultV2:
            """Execute and centrally validate the supplied native entry point."""
            native_inputs = _validate_public_inputs(
                op, normalized_shape, args, profile.device
            )
            entry = native_executor
            if entry is None and native_candidate is not None:
                return _native_public_result(
                    native_candidate.dispatch(*native_inputs),
                    operation=op,
                    boundary_callback=boundary_callback,
                    range_tolerance_m=solver.range_tolerance_m,
                    doppler_tolerance_hz=solver.doppler_tolerance_hz,
                    slant_range_tolerance_m=solver.slant_range_tolerance_m,
                    iteration_budget=solver.budget,
                )
            if entry is None:
                raise DispatchError("native geometry executor is missing")
            return _native_public_result(
                entry(*native_inputs),
                operation=op,
                boundary_callback=boundary_callback,
                range_tolerance_m=solver.range_tolerance_m,
                doppler_tolerance_hz=solver.doppler_tolerance_hz,
                slant_range_tolerance_m=solver.slant_range_tolerance_m,
                iteration_budget=solver.budget,
            )

        dispatcher.register(
            registered_native_key,
            execute_native,
            correctness_qualified=native_correctness_qualified,
            performance_eligible=native_performance_eligible,
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
    "PreparedGeometry",
    "execute_geometry",
    "execute_geometry_v2",
    "prepare_geometry",
    "prepare_geometry_v2",
]
