"""Focused regression tests for the public geometry v2 P1 contracts."""

from __future__ import annotations

import inspect
import traceback
from pathlib import Path

import numpy as np
import pytest

from faninsar.processing.geometry import (
    Operation,
    execute_geometry,
    prepare_geometry,
    torch_kernels,
)
from faninsar.processing.geometry import public as geometry_public
from faninsar.processing.geometry.backend_dispatch import (
    CudaExecutionError,
    Dispatcher,
    DispatchError,
)
from faninsar.processing.geometry.dem import ConstantHeightDEM
from faninsar.processing.geometry.native_v2.bindings import result_from_native_outputs
from faninsar.processing.geometry.native_v2.builder import (
    BuildPlan,
    GeometryOperation,
    NativeBackend,
    PreparationStatus,
    PreparedNativeCandidate,
)
from faninsar.processing.geometry.torch_backends_v2 import (
    TorchGeometryResult,
    prepare_torch_geometry,
)
from faninsar.processing.geometry.v2 import (
    CandidateKey,
    DeviceKey,
    GeometryValidationError,
    SolverSettings,
    TransformResultV2,
    validate_tensor_span,
)

from .test_public_geometry_v2 import (
    _model,
    _native_context,
    _native_key,
    _native_outputs,
)


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


def test_geo2rdr_convergence_uses_physical_metric_not_newton_step() -> None:
    """Keep last-step roundoff from changing the public convergence vote."""
    source = inspect.getsource(torch_kernels.geo2rdr_kernel)
    assert "step_small" not in source
    assert "newly = active & valid & in_bounds" in source
    assert "metric < 1.0" in source
    native_root = Path(torch_kernels.__file__).parent / "native_v2"
    assert (
        "fabs(step) < time_tolerance"
        not in (native_root / "cuda" / "geo2rdr_cuda.cu").read_text()
    )
    assert (
        "std::abs(step) <= time_tol_s" not in (native_root / "geo2rdr.cpp").read_text()
    )


def test_constant_dem_bypasses_fixed_point_and_commits_sampled_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constant DEM uses one solve and retains its height on convergence."""
    calls = 0
    original = torch_kernels._rdr2geo_once

    def count_solves(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_kernels, "_rdr2geo_once", count_solves)
    prepared = prepare_torch_geometry(
        "rdr2geo",
        _model(),
        shape=(2,),
        dem=ConstantHeightDEM(50.0),
        max_iter=4,
        range_tol_m=1.0,
        dem_iterations=50,
    )
    result = prepared.execute(
        np.array([0.0, 0.004], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        np.zeros(2, dtype=np.float64),
    ).transform

    assert calls == 1
    assert result.converged.tolist() == [True, True]
    assert result.iterations.tolist() == [2, 2]
    np.testing.assert_allclose(result.height_m, 50.0)


def test_constant_dem_eager_and_compiled_match_near_threshold_multilane() -> None:
    """Constant DEM completion is identical for eager and compiled Torch paths."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    model = _model()
    values = (
        np.array([0.0, 0.004], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        np.zeros(2, dtype=np.float64),
    )
    eager = (
        prepare_torch_geometry(
            "rdr2geo",
            model,
            shape=(2,),
            dem=ConstantHeightDEM(50.0),
            max_iter=4,
            range_tol_m=1.0,
        )
        .execute(*values)
        .transform
    )
    try:
        compiled = (
            prepare_torch_geometry(
                "rdr2geo",
                model,
                shape=(2,),
                dem=ConstantHeightDEM(50.0),
                max_iter=4,
                range_tol_m=1.0,
                compile_kernel=True,
            )
            .execute(*values)
            .transform
        )
    except RuntimeError as error:
        message = "".join(traceback.format_exception(error))
        if "libc++.1.dylib" in message:
            pytest.skip("the local Torch Inductor runtime cannot load libc++")
        raise

    for field in (
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "residual_range_m",
        "residual_doppler_hz",
    ):
        np.testing.assert_allclose(
            getattr(eager, field), getattr(compiled, field), atol=1.0e-8
        )
    assert eager.converged.tolist() == compiled.converged.tolist() == [True, True]
    assert eager.iterations.tolist() == compiled.iterations.tolist() == [2, 2]


def test_native_callback_rejects_wrong_dtype_before_invocation() -> None:
    """The public native seam validates array ownership and dtype first."""
    calls: list[int] = []
    model = _model()
    prepared = prepare_geometry(
        Operation.GEO2RDR,
        model,
        shape=(1,),
        native_executor=lambda *values: (
            calls.append(1) or _native_outputs(model, values[:3])
        ),
        native_context_inputs=_native_context(model, Operation.GEO2RDR),
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


def test_native_cpu_flattens_and_restores_two_dimensional_public_shape() -> None:
    """The CPU native seam adapts 2-D public arrays to the vector ABI."""
    model = _model()
    seen_shapes: list[tuple[tuple[int, ...], ...]] = []

    def native_executor(*values: np.ndarray) -> list[np.ndarray]:
        seen_shapes.append(tuple(tuple(value.shape) for value in values[:3]))
        return _native_outputs(model, values[:3])

    prepared = prepare_geometry(
        Operation.GEO2RDR,
        model,
        shape=(2, 2),
        native_executor=native_executor,
        native_context_inputs=_native_context(model, Operation.GEO2RDR),
        native_key=_native_key(model, (2, 2), Operation.GEO2RDR),
        native_correctness_qualified=True,
    )
    result = execute_geometry(
        prepared,
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        selector="native",
    )

    assert seen_shapes == [((4,), (4,), (4,))]
    assert result.latitude_deg.shape == (2, 2)
    assert result.iterations.shape == (2, 2)


def test_native_public_validation_rejects_three_dimensional_shape() -> None:
    """Native public execution has one consistent rank limit across devices."""
    calls: list[int] = []
    model = _model()
    prepared = prepare_geometry(
        Operation.GEO2RDR,
        model,
        shape=(1, 1, 1),
        native_executor=lambda *_: calls.append(1),
        native_context_inputs=_native_context(model, Operation.GEO2RDR),
        native_key=_native_key(model, (1, 1, 1), Operation.GEO2RDR),
        native_correctness_qualified=True,
    )

    with pytest.raises(GeometryValidationError, match="one- or two-dimensional"):
        execute_geometry(
            prepared,
            np.zeros((1, 1, 1), dtype=np.float64),
            np.zeros((1, 1, 1), dtype=np.float64),
            np.zeros((1, 1, 1), dtype=np.float64),
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
        output = _native_outputs(model, values[:3])
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
        native_context_inputs=_native_context(model, Operation.GEO2RDR),
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


def test_native_public_no_callback_uses_array_normalization_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-callback publication must not allocate one decision per lane."""
    shape = (2048,)
    fields = {
        "latitude_deg": np.ones(shape, dtype=np.float64),
        "longitude_deg": np.ones(shape, dtype=np.float64),
        "height_m": np.ones(shape, dtype=np.float64),
        "range_index": np.ones(shape, dtype=np.float64),
        "azimuth_index": np.ones(shape, dtype=np.float64),
        "converged": np.ones(shape, dtype=bool),
        "iterations": np.ones(shape, dtype=np.int32),
        "decision_residual": np.full(shape, 0.25, dtype=np.float64),
        "final_residual": np.full(shape, 0.25, dtype=np.float64),
        "tolerance": np.ones(shape, dtype=np.float64),
        "max_iter_exhausted": np.zeros(shape, dtype=bool),
        "boundary_rechecked": np.zeros(shape, dtype=bool),
        "residual_range_m": np.full(shape, 0.002, dtype=np.float64),
        "residual_doppler_hz": np.full(shape, 0.002, dtype=np.float64),
    }
    result = TransformResultV2.from_arrays(fields, operation=Operation.GEO2RDR)

    def unexpected_decision(*_: object, **__: object) -> None:
        raise AssertionError

    monkeypatch.setattr(geometry_public, "BoundaryDecision", unexpected_decision)
    monkeypatch.setattr(
        geometry_public,
        "normalize_result_boundary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no-callback publication used scalar normalization")
        ),
    )

    published = geometry_public._native_public_result(
        result,
        operation=Operation.GEO2RDR,
    )

    np.testing.assert_array_equal(published.latitude_deg, result.latitude_deg)
    np.testing.assert_array_equal(published.decision_residual, result.decision_residual)


def test_native_public_no_callback_preserves_scalar_status_semantics() -> None:
    """Array normalization keeps finite failures and invalid sentinels exact."""
    fields = {
        "latitude_deg": np.array([1.0, 2.0, np.nan], dtype=np.float64),
        "longitude_deg": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "height_m": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "range_index": np.ones(3, dtype=np.float64),
        "azimuth_index": np.ones(3, dtype=np.float64),
        "converged": np.array([True, False, False]),
        "iterations": np.array([1, 4, 4], dtype=np.int32),
        "decision_residual": np.array([0.1, 4.0, 4.0], dtype=np.float64),
        "final_residual": np.array([0.1, 4.0, 4.0], dtype=np.float64),
        "tolerance": np.ones(3, dtype=np.float64),
        "max_iter_exhausted": np.array([False, True, True]),
        "boundary_rechecked": np.array([False, True, True]),
        "residual_range_m": np.ones(3, dtype=np.float64),
        "residual_doppler_hz": np.ones(3, dtype=np.float64),
    }
    result = TransformResultV2.from_arrays(
        fields,
        operation=Operation.GEO2RDR,
        invalid_mask=np.array([False, False, True]),
    )

    published = geometry_public._native_public_result(
        result,
        operation=Operation.GEO2RDR,
    )

    assert published.converged.tolist() == [True, False, False]
    assert published.boundary_rechecked.tolist() == [False, False, False]
    assert np.isnan(published.decision_residual[1])
    assert published.iterations.tolist() == [1, 4, -1]
    assert np.isnan(published.latitude_deg[2])


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


def test_finite_exhausted_lane_keeps_diagnostics_and_coordinates() -> None:
    """Only explicitly invalid lanes receive NaN/-1 sentinels."""
    fields = {
        "latitude_deg": np.array([4.0]),
        "longitude_deg": np.array([5.0]),
        "height_m": np.array([6.0]),
        "range_index": np.array([7.0]),
        "azimuth_index": np.array([8.0]),
        "converged": np.array([False]),
        "iterations": np.array([3], dtype=np.int32),
        "residual_range_m": np.array([0.5]),
        "residual_doppler_hz": np.array([0.25]),
    }
    result = TorchGeometryResult.from_transform(
        fields,
        operation=Operation.RDR2GEO,
        device="cpu",
        dtype="torch.float64",
        identity="identity",
        backend="torch_eager",
    ).transform
    assert result.iterations[0] == 3
    assert result.max_iter_exhausted[0]
    assert np.isfinite(result.latitude_deg[0])
    assert np.isfinite(result.tolerance[0])


def test_native_binding_preserves_finite_exhaustion() -> None:
    """Native ABI normalization does not confuse exhaustion with invalidity."""
    values = [
        np.array([1.0]),
        np.array([2.0]),
        np.array([3.0]),
        np.array([4.0]),
        np.array([5.0]),
        np.array([False]),
        np.array([4], dtype=np.int32),
        np.array([0.5]),
        np.array([0.5]),
        np.array([0.01]),
        np.array([True]),
        np.array([False]),
        np.array([0.5]),
        np.array([0.25]),
    ]
    result = result_from_native_outputs(values, operation=Operation.RDR2GEO)
    assert result.iterations[0] == 4
    assert result.max_iter_exhausted[0]
    assert np.isfinite(result.latitude_deg[0])


def test_native_manifest_rejects_executable_identity_difference(tmp_path: Path) -> None:
    """A candidate manifest must match every executable identity field."""
    artifact = tmp_path / "native.so"
    artifact.write_bytes(b"native")
    plan = BuildPlan(
        GeometryOperation.GEO2RDR,
        NativeBackend.CPU,
        "native",
        "native",
        (),
        (),
        (),
    )
    candidate = PreparedNativeCandidate(
        plan,
        PreparationStatus.PREPARED,
        artifact=artifact,
        _entry_point=lambda *_: (),
    )
    with pytest.raises(Exception, match="executable"):
        prepare_geometry(
            Operation.GEO2RDR,
            _model(),
            shape=(1,),
            native_candidate=candidate,
            native_key=_native_key(_model(), (1,), Operation.GEO2RDR),
        )


@pytest.mark.parametrize("operation", [Operation.GEO2RDR, Operation.RDR2GEO])
def test_native_context_rejects_nonfinite_orbit_before_executor(
    operation: Operation,
) -> None:
    """A nonfinite closed-over orbit span never reaches the native callback."""
    model = _model()
    context = _native_context(model, operation)
    orbit_positions = np.asarray(context["orbit_positions"]).copy()
    orbit_positions[0, 0] = np.nan
    context["orbit_positions"] = orbit_positions
    calls: list[int] = []
    with pytest.raises(GeometryValidationError, match="finite"):
        prepare_geometry(
            operation,
            model,
            shape=(1,),
            native_executor=lambda *_: calls.append(1),
            native_context_inputs=context,
            native_key=_native_key(model, (1,), operation),
        )
    assert calls == []


def test_native_context_rejects_nonfinite_dem_before_executor() -> None:
    """A nonfinite rdr2geo DEM span is rejected before callback invocation."""
    model = _model()
    context = _native_context(model, Operation.RDR2GEO)
    dem_values = np.asarray(context["dem_values"]).copy()
    dem_values[0, 0] = np.inf
    context["dem_values"] = dem_values
    calls: list[int] = []
    with pytest.raises(GeometryValidationError, match="finite"):
        prepare_geometry(
            Operation.RDR2GEO,
            model,
            shape=(1,),
            native_executor=lambda *_: calls.append(1),
            native_context_inputs=context,
            native_key=_native_key(model, (1,), Operation.RDR2GEO),
        )
    assert calls == []


@pytest.mark.parametrize("operation", [Operation.GEO2RDR, Operation.RDR2GEO])
def test_native_context_rejects_noncontiguous_orbit_before_executor(
    operation: Operation,
) -> None:
    """A strided orbit span is rejected before exposing a native pointer."""
    model = _model()
    context = _native_context(model, operation)
    context["orbit_positions"] = np.asfortranarray(context["orbit_positions"])
    calls: list[int] = []
    with pytest.raises(GeometryValidationError, match="strides"):
        prepare_geometry(
            operation,
            model,
            shape=(1,),
            native_executor=lambda *_: calls.append(1),
            native_context_inputs=context,
            native_key=_native_key(model, (1,), operation),
        )
    assert calls == []


@pytest.mark.parametrize("operation", [Operation.GEO2RDR, Operation.RDR2GEO])
def test_native_context_rejects_wrong_device_before_executor(
    operation: Operation,
) -> None:
    """A context tensor on an unsupported device cannot reach native code."""
    torch = pytest.importorskip("torch")
    model = _model()
    context = _native_context(model, operation)
    context["orbit_times"] = torch.empty(
        len(context["orbit_times"]), dtype=torch.float64, device="meta"
    )
    calls: list[int] = []
    with pytest.raises(GeometryValidationError, match="device"):
        prepare_geometry(
            operation,
            model,
            shape=(1,),
            native_executor=lambda *_: calls.append(1),
            native_context_inputs=context,
            native_key=_native_key(model, (1,), operation),
        )
    assert calls == []


def test_native_executor_requires_explicit_context_descriptors() -> None:
    """Native callbacks cannot hide orbit/DEM ABI inputs in a closure."""
    model = _model()
    with pytest.raises(DispatchError, match="context"):
        prepare_geometry(
            Operation.GEO2RDR,
            model,
            shape=(1,),
            native_executor=lambda *_: (),
            native_key=_native_key(model, (1,), Operation.GEO2RDR),
        )
