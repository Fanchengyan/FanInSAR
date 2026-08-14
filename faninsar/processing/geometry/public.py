# ruff: noqa: A002, EM101, EM102, TRY003

"""Public preparation and execution boundary for geometry v2.

The public boundary is intentionally small.  Preparation creates the eager
Torch adapter and, when requested, the compiled adapter or a native candidate.
Execution only selects an already prepared entry in the local dispatcher; it
never calls a compiler.  All results, including native results, are converted
through :class:`TransformResultV2` and the canonical boundary normalizer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
)

if TYPE_CHECKING:
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.geometry.native_v2.builder import PreparedNativeCandidate
    from faninsar.processing.geometry.transforms import RadarGeometryModel

logger = setup_logger(__name__)

BackendSelector: TypeAlias = Literal["native", "compile", "eager", "auto"]
NativeExecutor: TypeAlias = Callable[..., object]
BoundaryCallback: TypeAlias = Callable[..., float | Sequence[float]]


def _device_key(device: str, physical_uuid: str | None) -> DeviceKey:
    """Build a foundation device identity from a Torch device string."""
    if str(device).split(":", 1)[0] == "cuda":
        return DeviceKey.cuda(physical_uuid or str(device))
    if str(device).split(":", 1)[0] != "cpu":
        raise DispatchError(f"geometry v2 only supports CPU and CUDA, got {device!r}")
    return DeviceKey.cpu()


def _native_public_result(
    outputs: object,
    *,
    operation: Operation,
    boundary_callback: BoundaryCallback | None = None,
) -> TransformResultV2:
    """Convert and normalize one native result at the public boundary."""
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
    for index, is_valid in enumerate(flat_valid):
        if not is_valid:
            decisions.append(BoundaryDecision(False, False, float("nan"), float("nan")))
            continue
        index_tuple = np.unravel_index(index, valid.shape)
        decision_residual = float(result.decision_residual[index_tuple])
        if operation is Operation.GEO2RDR:
            residual: float | tuple[float, float] = (
                float(result.residual_range_m[index_tuple]),
                float(result.residual_doppler_hz[index_tuple]),
            )
            tolerance = float(result.tolerance[index_tuple])
            # Native geo2rdr publishes normalized ``decision_residual``.  A
            # tolerance of one is the canonical v2 publication value.
            tolerance = tolerance if np.isfinite(tolerance) else 1.0
            if boundary_callback is None:
                decisions.append(
                    BoundaryDecision(
                        bool(result.converged[index_tuple]),
                        bool(result.boundary_rechecked[index_tuple]),
                        decision_residual,
                        abs(decision_residual),
                    )
                )
            else:
                coords = tuple(
                    np.asarray([getattr(result, name)[index_tuple]], dtype=np.float64)
                    for name in ("latitude_deg", "longitude_deg", "height_m")
                )
                decisions.append(
                    evaluate_canonical_boundary(
                        operation,
                        int(result.iterations[index_tuple]),
                        coords,
                        boundary_callback,
                        decision_residual=(residual[0] / tolerance, residual[1]),
                        range_tolerance_m=tolerance,
                        doppler_tolerance_hz=1.0,
                    )
                )
        else:
            tolerance = float(result.tolerance[index_tuple])
            if boundary_callback is None:
                decisions.append(
                    BoundaryDecision(
                        bool(result.converged[index_tuple]),
                        bool(result.boundary_rechecked[index_tuple]),
                        decision_residual,
                        abs(decision_residual) / tolerance
                        if tolerance > 0.0
                        else float("nan"),
                    )
                )
            else:
                coords = tuple(
                    np.asarray([getattr(result, name)[index_tuple]], dtype=np.float64)
                    for name in ("latitude_deg", "longitude_deg", "height_m")
                )
                decisions.append(
                    evaluate_canonical_boundary(
                        operation,
                        int(result.iterations[index_tuple]),
                        coords,
                        boundary_callback,
                        decision_residual=decision_residual,
                        slant_range_tolerance_m=tolerance,
                    )
                )
    return normalize_result_boundary(result, decisions, invalid_mask=~valid)


def _key(
    operation: Operation,
    backend: Literal["native", "compile"],
    prepared: PreparedTorchGeometry,
    profile: ExecutionProfile,
) -> CandidateKey:
    """Build a complete foundation candidate key for one prepared backend."""
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
        support_contract_digest=prepared.identity.settings_digest,
        profile=profile,
    )


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
            return result.transform.validate()
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
            return self.native_key or self.compile_key or _key(
                self.operation,
                "compile",
                self.eager,
                ExecutionProfile.cpu(),
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
    native_correctness_qualified: bool = False,
    native_performance_eligible: bool = False,
    compile: bool = False,
    compile_correctness_qualified: bool = True,
    compile_performance_eligible: bool = True,
    physical_uuid: str | None = None,
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
        _device_key(device, physical_uuid),
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
        return eager.execute(*args).transform

    dispatcher = Dispatcher(eager_execute)
    native_key: CandidateKey | None = None
    compile_key: CandidateKey | None = None
    if compiled is not None:
        compile_key = _key(op, "compile", compiled, profile)
        dispatcher.register(
            compile_key,
            lambda *args: compiled.execute(*args).transform,
            correctness_qualified=compile_correctness_qualified,
            performance_eligible=compile_performance_eligible,
            result_validator=lambda value: _native_public_result(
                value, operation=op, boundary_callback=boundary_callback
            ),
        )
    if native_executor is not None or native_candidate is not None:
        native_key = _key(op, "native", eager, profile)

        def execute_native(*args: object) -> TransformResultV2:
            """Execute and centrally validate the supplied native entry point."""
            entry = native_executor
            if entry is None and native_candidate is not None:
                return _native_public_result(
                    native_candidate.dispatch(*args),
                    operation=op,
                    boundary_callback=boundary_callback,
                )
            if entry is None:
                raise DispatchError("native geometry executor is missing")
            return _native_public_result(
                entry(*args), operation=op, boundary_callback=boundary_callback
            )

        dispatcher.register(
            native_key,
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
        native_key,
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
