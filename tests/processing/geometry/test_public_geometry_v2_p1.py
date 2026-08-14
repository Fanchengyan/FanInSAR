"""Focused regression tests for the public geometry v2 P1 contracts."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.geometry import Operation, execute_geometry, prepare_geometry
from faninsar.processing.geometry.backend_dispatch import (
    CudaExecutionError,
    Dispatcher,
)
from faninsar.processing.geometry.dem import ConstantHeightDEM
from faninsar.processing.geometry.native_v2.builder import (
    BuildPlan,
    GeometryOperation,
    NativeBackend,
    PreparationStatus,
    PreparedNativeCandidate,
)
from faninsar.processing.geometry.torch_backends_v2 import prepare_torch_geometry
from faninsar.processing.geometry.v2 import (
    CandidateKey,
    DeviceKey,
    GeometryValidationError,
    SolverSettings,
    validate_tensor_span,
)

from .test_public_geometry_v2 import _model, _native_key, _native_outputs


def test_torch_result_publishes_actual_iterations_and_invalid_tolerance() -> None:
    """Per-lane diagnostics come from the solver rather than a scalar budget."""
    model = _model()
    prepared = prepare_torch_geometry("geo2rdr", model, shape=(2,))
    result = prepared.execute(
        np.zeros(2, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.array([0.0, np.nan], dtype=np.float64),
    )
    assert result.iterations[0] == 1
    assert result.iterations[1] == -1
    assert bool(result.converged[0])
    assert not bool(result.converged[1])
    assert np.isnan(result.tolerance[1])


def test_native_callback_rejects_wrong_dtype_before_invocation() -> None:
    """The public native seam validates array ownership and dtype first."""
    calls: list[int] = []
    model = _model()
    prepared = prepare_geometry(
        Operation.GEO2RDR,
        model,
        shape=(1,),
        native_executor=lambda *values: (
            calls.append(1) or _native_outputs(model, values)
        ),
        native_key=_native_key(model, (1,), Operation.GEO2RDR),
        native_correctness_qualified=True,
    )
    with pytest.raises(GeometryValidationError, match="float64"):
        execute_geometry(
            prepared,
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            selector="native",
        )
    assert calls == []


def test_native_candidate_operation_is_bound_to_public_key() -> None:
    """A prepared candidate for the other operation cannot be reused."""
    plan = BuildPlan(
        GeometryOperation.RDR2GEO,
        NativeBackend.CPU,
        "wrong",
        "wrong",
        (),
        (),
        (),
    )
    candidate = PreparedNativeCandidate(plan, PreparationStatus.PREPARED)
    with pytest.raises(Exception, match="operation"):
        prepare_geometry(
            Operation.GEO2RDR,
            _model(),
            shape=(1,),
            native_candidate=candidate,
        )


def test_compile_is_not_performance_eligible_by_default() -> None:
    """Compilation remains correctness-only until benchmark evidence promotes it."""
    prepared = prepare_geometry(
        Operation.GEO2RDR,
        _model(),
        shape=(1,),
    )
    candidates = prepared.dispatcher.active_candidates()
    assert candidates == ()
    result = execute_geometry(
        prepared,
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        selector="auto",
    )
    assert result.fields
    assert prepared.dispatcher.records[-1].backend == "eager"


def test_cuda_unknown_failure_is_fatal_to_auto_dispatch() -> None:
    """Unknown CUDA failures are classified and never silently fall back."""
    key = CandidateKey(
        Operation.GEO2RDR,
        "native",
        DeviceKey.cuda("GPU-test"),
        shape=(1,),
        solver=SolverSettings(),
    )
    called = []
    dispatcher = Dispatcher(lambda: called.append("eager"))
    dispatcher.register(
        key,
        lambda: (_ for _ in ()).throw(RuntimeError("driver returned mystery error")),
        correctness_qualified=True,
        performance_eligible=True,
    )
    with pytest.raises(CudaExecutionError):
        dispatcher.dispatch(key, "auto")
    assert called == []


def test_typed_cuda_prelaunch_oom_can_fallback_with_healthy_context() -> None:
    """Only explicit pre-launch/context evidence permits automatic fallback."""
    key = CandidateKey(
        Operation.GEO2RDR,
        "native",
        DeviceKey.cuda("GPU-test"),
        shape=(1,),
        solver=SolverSettings(),
    )

    class PreflightOOMError(RuntimeError):
        """Typed pre-launch failure supplied by a native adapter."""

        cuda_phase = "pre_launch"
        context_healthy = True

    called: list[str] = []
    dispatcher = Dispatcher(lambda: called.append("eager") or object())
    dispatcher.register(
        key,
        lambda: (_ for _ in ()).throw(PreflightOOMError("out of memory")),
        correctness_qualified=True,
        performance_eligible=True,
    )
    dispatcher.dispatch(key, "auto")
    assert called == ["eager"]


def test_dem_identity_includes_material_value() -> None:
    """Changing the DEM sample value changes the prepared identity digest."""
    first = prepare_torch_geometry(
        "rdr2geo", _model(), shape=(1,), dem=ConstantHeightDEM(10.0)
    )
    second = prepare_torch_geometry(
        "rdr2geo", _model(), shape=(1,), dem=ConstantHeightDEM(11.0)
    )
    assert first.identity.dem_digest != second.identity.dem_digest


def test_native_executor_requires_complete_manifest() -> None:
    """A callback alone cannot publish an unbound native executable."""
    with pytest.raises(Exception, match=r"manifest|CandidateKey|key"):
        prepare_geometry(
            Operation.GEO2RDR,
            _model(),
            shape=(1,),
            native_executor=lambda *_: (),
        )


def test_boundary_rejection_uses_remaining_attempt_budget() -> None:
    """A rejected boundary result is retried through the remaining budget."""
    model = _model()
    calls = 0

    def callback(*_: np.ndarray) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return (2.0, 0.0) if calls == 1 else (0.0, 0.0)

    def native_boundary_output(*values: np.ndarray) -> list[np.ndarray]:
        output = _native_outputs(model, values)
        output[7] = np.ones(1, dtype=np.float64)
        output[8] = np.ones(1, dtype=np.float64)
        output[9] = np.ones(1, dtype=np.float64)
        output[12] = np.ones(1, dtype=np.float64)
        return output

    prepared = prepare_geometry(
        Operation.GEO2RDR,
        model,
        shape=(1,),
        settings=SolverSettings(
            max_iter=3, range_tolerance_m=1.0, doppler_tolerance_hz=1.0
        ),
        native_executor=native_boundary_output,
        native_key=_native_key(
            model,
            (1,),
            Operation.GEO2RDR,
            SolverSettings(max_iter=3, range_tolerance_m=1.0, doppler_tolerance_hz=1.0),
        ),
        boundary_callback=callback,
        native_correctness_qualified=True,
    )
    result = execute_geometry(
        prepared,
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        selector="native",
    )
    assert calls == 2
    assert result.converged[0]
    assert result.iterations[0] == 2


def test_tensor_span_validates_owner_and_device() -> None:
    """Torch tensor spans carry the same structural proof as host spans."""
    torch = pytest.importorskip("torch")
    tensor = torch.zeros((2,), dtype=torch.float64)
    span = validate_tensor_span(
        tensor, expected_dtype=torch.float64, expected_shape=(2,), name="tensor"
    )
    assert span.byte_length == tensor.numel() * tensor.element_size()
    with pytest.raises(Exception, match="dtype"):
        validate_tensor_span(
            tensor.float(),
            expected_dtype=torch.float64,
            expected_shape=(2,),
            name="tensor",
        )


def test_composed_dem_identity_changes_with_nested_sampler() -> None:
    """Nested DEM sampler values participate in the prepared digest."""
    from faninsar.processing.geometry.dem import GeoidAdjustedDEM

    first = prepare_torch_geometry(
        "rdr2geo",
        _model(),
        shape=(1,),
        dem=GeoidAdjustedDEM(ConstantHeightDEM(10.0), ConstantHeightDEM(1.0)),
    )
    second = prepare_torch_geometry(
        "rdr2geo",
        _model(),
        shape=(1,),
        dem=GeoidAdjustedDEM(ConstantHeightDEM(10.0), ConstantHeightDEM(2.0)),
    )
    assert first.identity.dem_digest != second.identity.dem_digest
