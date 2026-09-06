"""Focused regression tests for the public geometry v2 P1 contracts."""

from __future__ import annotations

import inspect
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from faninsar.processing.geometry import ConstantDEM
from faninsar.processing.geometry import (
    Operation,
    execute_geometry,
    prepare_geometry,
    torch_backends_v2,
    torch_kernels,
)
from faninsar.processing.geometry import public as geometry_public
from faninsar.processing.geometry.backend_dispatch import (
    CudaExecutionError,
    Dispatcher,
    DispatchError,
    FatalExecutionError,
    RecoverableExecutionError,
)
from faninsar.processing.geometry.native_v2.bindings import (
    ecef_from_native_outputs,
    result_from_native_outputs,
)
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
    ExecutionProfile,
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


def _native_ecef_outputs(shape: tuple[int, ...] = (2,)) -> list[np.ndarray]:
    """Build a valid seventeen-field native ECEF result for boundary tests."""
    values: list[np.ndarray] = [
        np.full(shape, float(index), dtype=np.float64) for index in range(5)
    ]
    values.extend(
        [
            np.ones(shape, dtype=bool),
            np.zeros(shape, dtype=np.int32),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=bool),
            np.zeros(shape, dtype=bool),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
        ]
    )
    values.extend(
        np.full(shape, float(index), dtype=np.float64) for index in (100, 200, 300)
    )
    return values


def test_ecef_native_output_validator_accepts_only_typed_seventeen_fields() -> None:
    """The ECEF ABI validates the untouched canonical fields before transfer."""
    x, y, z = ecef_from_native_outputs(_native_ecef_outputs())
    np.testing.assert_array_equal(x, 100.0)
    np.testing.assert_array_equal(y, 200.0)
    np.testing.assert_array_equal(z, 300.0)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda values: values.pop(), id="field_count"),
        pytest.param(
            lambda values: values.__setitem__(0, np.zeros(3, dtype=np.float64)),
            id="canonical_shape",
        ),
        pytest.param(
            lambda values: values.__setitem__(0, np.zeros(2, dtype=np.float32)),
            id="canonical_dtype",
        ),
        pytest.param(
            lambda values: values.__setitem__(14, np.zeros(2, dtype=np.float32)),
            id="ecef_dtype",
        ),
        pytest.param(
            lambda values: values.__setitem__(16, np.zeros(3, dtype=np.float64)),
            id="ecef_shape",
        ),
    ],
)
def test_ecef_native_output_validator_rejects_malformed_fields(mutate: object) -> None:
    """Malformed ECEF fields cannot bypass the typed native boundary."""
    outputs = _native_ecef_outputs()
    mutate(outputs)  # type: ignore[operator]
    with pytest.raises(ValueError, match=r"native geometry|native field"):
        ecef_from_native_outputs(outputs)


def test_public_ecef_selectors_use_dispatch_gates_and_quarantine() -> None:
    """ECEF selectors share explicit qualification and automatic quarantine."""
    from types import SimpleNamespace

    native_key = CandidateKey(
        Operation.RDR2GEO,
        "native",
        DeviceKey.cpu(),
        shape=(1,),
        source_digest="native-source",
        toolchain_digest="native-toolchain",
        runtime_digest="native-runtime",
        artifact_digest="native-artifact",
        abi_digest="native-abi",
        support_contract_digest="native-support",
        profile=ExecutionProfile.cpu(),
    )
    compile_key = CandidateKey(
        Operation.RDR2GEO,
        "compile",
        DeviceKey.cpu(),
        shape=(1,),
        source_digest="compile-source",
        toolchain_digest="compile-toolchain",
        runtime_digest="compile-runtime",
        artifact_digest="compile-artifact",
        abi_digest="compile-abi",
        support_contract_digest="native-support",
        profile=ExecutionProfile.cpu(),
    )
    value = (np.ones(1, dtype=np.float64),) * 3
    eager = SimpleNamespace(execute_ecef=lambda *_: value)
    dispatcher = Dispatcher(eager.execute_ecef)
    dispatcher.register(native_key, lambda *_: value, correctness_qualified=False)
    dispatcher.register(compile_key, lambda *_: value, correctness_qualified=False)
    prepared = geometry_public.PreparedGeometry(
        Operation.RDR2GEO,
        None,  # type: ignore[arg-type]
        (1,),
        eager,  # type: ignore[arg-type]
        dispatcher,
        "cpu",
        "float64",
        native_key,
        compile_key,
        compile_key,
        lambda *_: value,
        SimpleNamespace(execute_ecef=lambda *_: value),
    )

    with pytest.raises(DispatchError, match="native candidate is unavailable"):
        prepared.execute_ecef(np.zeros(1), selector="native")

    with pytest.raises(DispatchError, match="compile candidate is unavailable"):
        prepared.execute_ecef(np.zeros(1), selector="compile")
    np.testing.assert_array_equal(
        prepared.execute_ecef(np.zeros(1), selector="eager")[0], 1.0
    )
    assert dispatcher.records[-1].backend == "eager"

    # A malformed alternate result must quarantine the native candidate just
    # like an execution failure, allowing the next qualified backend to run.
    dispatcher = Dispatcher(eager.execute_ecef)
    dispatcher.register(
        native_key,
        lambda *_: (np.ones(1),),
        correctness_qualified=True,
        performance_eligible=True,
        ecef_correctness_qualified=True,
    )
    dispatcher.register(compile_key, lambda *_: value)
    prepared = geometry_public.PreparedGeometry(
        Operation.RDR2GEO,
        None,  # type: ignore[arg-type]
        (1,),
        eager,  # type: ignore[arg-type]
        dispatcher,
        "cpu",
        "float64",
        native_key,
        compile_key,
        compile_key,
        lambda *_: (np.ones(1),),
        SimpleNamespace(execute_ecef=lambda *_: value),
    )
    np.testing.assert_array_equal(
        prepared.execute_ecef(np.zeros(1), selector="auto")[0], 1.0
    )
    assert dispatcher.records[-1].backend == "compile"
    with pytest.raises(DispatchError, match="native candidate is unavailable"):
        prepared.execute_ecef(np.zeros(1), selector="native")

    dispatcher = Dispatcher(eager.execute_ecef)
    dispatcher.register(
        native_key,
        lambda *_: value,
        correctness_qualified=True,
        performance_eligible=True,
        ecef_correctness_qualified=True,
    )
    dispatcher.register(compile_key, lambda *_: value)

    def failing_native(*_: object) -> object:
        message = "typed ECEF probe failed"
        raise RecoverableExecutionError(message)

    prepared = geometry_public.PreparedGeometry(
        Operation.RDR2GEO,
        None,  # type: ignore[arg-type]
        (1,),
        eager,  # type: ignore[arg-type]
        dispatcher,
        "cpu",
        "float64",
        native_key,
        compile_key,
        compile_key,
        failing_native,
        SimpleNamespace(execute_ecef=lambda *_: value),
    )
    np.testing.assert_array_equal(
        prepared.execute_ecef(np.zeros(1), selector="auto")[0], 1.0
    )
    assert dispatcher.records[-1].backend == "compile"
    with pytest.raises(DispatchError, match="native candidate is unavailable"):
        prepared.execute_ecef(np.zeros(1), selector="native")


def test_public_ecef_quarantines_seventeen_field_shape_mismatch() -> None:
    """A valid native ABI payload with the wrong public shape is quarantined."""
    native_key = CandidateKey(
        Operation.RDR2GEO,
        "native",
        DeviceKey.cpu(),
        shape=(1,),
        source_digest="native-source",
        toolchain_digest="native-toolchain",
        runtime_digest="native-runtime",
        artifact_digest="native-artifact",
        abi_digest="native-abi",
        support_contract_digest="native-support",
        profile=ExecutionProfile.cpu(),
    )
    compile_key = CandidateKey(
        Operation.RDR2GEO,
        "compile",
        DeviceKey.cpu(),
        shape=(1,),
        source_digest="compile-source",
        toolchain_digest="compile-toolchain",
        runtime_digest="compile-runtime",
        artifact_digest="compile-artifact",
        abi_digest="compile-abi",
        support_contract_digest="native-support",
        profile=ExecutionProfile.cpu(),
    )
    value = (np.ones(1, dtype=np.float64),) * 3
    malformed_native = _native_ecef_outputs((2,))
    dispatcher = Dispatcher(lambda *_: value)
    dispatcher.register(
        native_key,
        lambda *_: malformed_native,
        correctness_qualified=True,
        performance_eligible=True,
        ecef_correctness_qualified=True,
    )
    dispatcher.register(compile_key, lambda *_: value)
    prepared = geometry_public.PreparedGeometry(
        Operation.RDR2GEO,
        None,  # type: ignore[arg-type]
        (1,),
        SimpleNamespace(execute_ecef=lambda *_: value),  # type: ignore[arg-type]
        dispatcher,
        "cpu",
        "float64",
        native_key,
        compile_key,
        compile_key,
        lambda *_: malformed_native,
        SimpleNamespace(execute_ecef=lambda *_: value),
    )

    np.testing.assert_array_equal(
        prepared.execute_ecef(np.zeros(1), selector="auto")[0], 1.0
    )
    assert dispatcher.records[-1].backend == "compile"
    with pytest.raises(DispatchError, match="native candidate is unavailable"):
        prepared.execute_ecef(np.zeros(1), selector="native")


def test_native_ecef_requires_its_own_qualification_gate() -> None:
    """Ordinary native qualification cannot authorize the ECEF alternate."""
    native_key = CandidateKey(
        Operation.RDR2GEO,
        "native",
        DeviceKey.cpu(),
        shape=(1,),
        solver=SolverSettings(),
    )
    value = (np.ones(1, dtype=np.float64),) * 3
    eager_calls: list[str] = []
    dispatcher = Dispatcher(lambda *_: eager_calls.append("eager") or value)
    dispatcher.register(
        native_key,
        lambda *_: value,
        correctness_qualified=True,
        performance_eligible=True,
    )

    with pytest.raises(DispatchError, match="native ECEF candidate is unavailable"):
        dispatcher.dispatch_alternate(
            native_key,
            "native",
            {"native": lambda *_: value},
            lambda *_: eager_calls.append("eager") or value,
            lambda result: result,
        )
    assert eager_calls == []
    assert (
        dispatcher.dispatch_alternate(
            native_key,
            "auto",
            {"native": lambda *_: value},
            lambda *_: eager_calls.append("eager") or value,
            lambda result: result,
        )
        == value
    )
    assert eager_calls == ["eager"]


def test_alternate_dispatch_does_not_quarantine_fatal_execution_errors() -> None:
    """Fatal typed execution failures remain visible instead of falling back."""
    native_key = CandidateKey(
        Operation.RDR2GEO,
        "native",
        DeviceKey.cpu(),
        shape=(1,),
        solver=SolverSettings(),
    )
    eager_calls: list[str] = []
    dispatcher = Dispatcher(lambda *_: eager_calls.append("eager"))
    dispatcher.register(
        native_key,
        lambda *_: object(),
        correctness_qualified=True,
        performance_eligible=True,
        ecef_correctness_qualified=True,
    )

    def fatal_native(*_: object) -> object:
        message = "native ABI is incompatible"
        raise FatalExecutionError(message)

    with pytest.raises(FatalExecutionError, match="incompatible"):
        dispatcher.dispatch_alternate(
            native_key,
            "auto",
            {"native": fatal_native},
            lambda *_: object(),
            lambda result: result,
        )
    assert eager_calls == []


def _fake_raster_dem(values: np.ndarray) -> object:
    """Build a minimal RasterDEM-shaped test double for Torch preparation."""

    class Transform:
        """Minimal rasterio affine-compatible transform."""

        values = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        a, b, c, d, e, f = values

        def __getitem__(self, index: int) -> float:
            return self.values[index]

    class RasterDEM:
        interpolation = "biquintic"
        nodata = None

        def __init__(self) -> None:
            self._height_array = values
            self._dataset = SimpleNamespace(transform=Transform())

        def _open(self) -> object:
            return self._dataset

    return RasterDEM()


def _capture_rdr2geo_height_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> list[np.ndarray]:
    """Capture the third argument passed to the Torch rdr2geo kernel."""
    captured: list[np.ndarray] = []

    def fake_kernel(*args: object, **_: object) -> dict[str, object]:
        import torch

        captured.append(np.asarray(args[2].detach().cpu()))
        shape = args[0].shape
        zeros = torch.zeros(shape, dtype=torch.float64)
        return {
            "latitude_deg": zeros,
            "longitude_deg": zeros,
            "height_m": args[2],
            "range_index": zeros,
            "azimuth_index": zeros,
            "converged": torch.ones(shape, dtype=torch.bool),
            "iterations": torch.ones(shape, dtype=torch.int32),
            "residual_range_m": zeros,
            "residual_doppler_hz": zeros,
        }

    monkeypatch.setattr(torch_backends_v2, "rdr2geo_kernel", fake_kernel)
    return captured


def test_rdr2geo_default_height_seed_comes_from_prepared_dem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared Torch execution derives defaults and preserves explicit height."""
    captured = _capture_rdr2geo_height_seed(monkeypatch)
    prepared = prepare_torch_geometry(
        "rdr2geo",
        _model(),
        shape=(2,),
        dem=ConstantDEM(42.0),
    )

    prepared.execute(np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64))
    prepared.execute(
        np.zeros(2, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.full(2, 7.0, dtype=np.float64),
    )

    np.testing.assert_array_equal(captured[0], 42.0)
    np.testing.assert_array_equal(captured[1], 7.0)


def test_torch_rdr2geo_wrapper_reuses_prepared_default_height_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The convenience wrapper leaves omitted height for prepared validation."""
    captured = _capture_rdr2geo_height_seed(monkeypatch)

    torch_backends_v2.torch_rdr2geo(
        _model(),
        np.zeros(2, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        dem=ConstantDEM(42.0),
    )

    np.testing.assert_array_equal(captured[0], 42.0)


def test_rdr2geo_raster_seed_uses_finite_mean_and_composition_offset() -> None:
    """Raster and raster-plus-constant DEMs derive one finite scalar seed."""
    raster = _fake_raster_dem(
        np.tile(
            np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float64),
            (3, 3),
        )
    )
    prepared = prepare_torch_geometry("rdr2geo", _model(), shape=(1,), dem=raster)
    assert prepared.height_seed_m == 3.0

    class GeoidAdjustedDEM:
        def __init__(self) -> None:
            self.orthometric_dem = raster
            self.geoid = ConstantDEM(10.0)

    composed = prepare_torch_geometry(
        "rdr2geo", _model(), shape=(1,), dem=GeoidAdjustedDEM()
    )
    assert composed.height_seed_m == 13.0


def test_rdr2geo_all_nan_raster_seed_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raster with no finite samples uses zero and emits a warning."""
    warnings: list[str] = []
    monkeypatch.setattr(torch_backends_v2.logger, "warning", warnings.append)
    raster = _fake_raster_dem(np.full((6, 6), np.nan, dtype=np.float64))

    prepared = prepare_torch_geometry("rdr2geo", _model(), shape=(1,), dem=raster)

    assert prepared.height_seed_m == 0.0
    assert warnings
    assert "finite" in warnings[0]


def test_native_default_height_uses_prepared_eager_seed() -> None:
    """The public Native seam receives the same prepared DEM seed as Eager."""
    captured: list[np.ndarray] = []

    class NativeModule:
        """Extension-shaped test double for the DEM-aware CPU ABI."""

        def rdr2geo_cpu_dem(self, *values: object) -> list[np.ndarray]:
            captured.append(np.asarray(values[2]).copy())
            return _native_outputs(_model(), values[:3])  # type: ignore[arg-type]

    model = _model()
    prepared = prepare_geometry(
        Operation.RDR2GEO,
        model,
        shape=(1,),
        dem=ConstantDEM(37.0),
        native_executor=NativeModule(),
        native_context_inputs=_native_context(model, Operation.RDR2GEO),
        native_key=_native_key(
            model,
            (1,),
            Operation.RDR2GEO,
            dem=ConstantDEM(37.0),
        ),
        native_correctness_qualified=True,
    )

    execute_geometry(
        prepared,
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        selector="native",
    )

    np.testing.assert_array_equal(captured[0], 37.0)


@pytest.mark.parametrize(
    ("operation", "expected_tolerance"),
    [
        (Operation.RDR2GEO, 3.0),
        (Operation.GEO2RDR, 7.0),
    ],
)
def test_public_preparation_uses_operation_specific_range_tolerance(
    monkeypatch: pytest.MonkeyPatch,
    operation: Operation,
    expected_tolerance: float,
) -> None:
    """Eager and Compile receive the tolerance for their physical operation."""
    calls: list[float] = []
    original = geometry_public.prepare_torch_geometry

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(float(kwargs["range_tol_m"]))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(geometry_public, "prepare_torch_geometry", spy)
    settings = SolverSettings(
        range_tolerance_m=expected_tolerance if operation is Operation.GEO2RDR else 7.0,
        slant_range_tolerance_m=(
            expected_tolerance if operation is Operation.RDR2GEO else 3.0
        ),
    )
    prepare_geometry(
        operation,
        _model(),
        shape=(1,),
        dem=ConstantDEM(0.0),
        settings=settings,
        compile=True,
    )

    assert calls == [expected_tolerance, expected_tolerance]


def test_native_rdr2geo_rejects_callable_no_dem_abi() -> None:
    """A callable cannot bypass the public DEM-aware ABI selection."""
    model = _model()
    dem = ConstantDEM(37.0)
    with pytest.raises(DispatchError, match=r"extension module.*rdr2geo_cpu_dem"):
        prepare_geometry(
            Operation.RDR2GEO,
            model,
            shape=(1,),
            dem=dem,
            native_executor=lambda *_: (),
            native_context_inputs=_native_context(model, Operation.RDR2GEO),
            native_key=_native_key(model, (1,), Operation.RDR2GEO, dem=dem),
        )


def test_native_rdr2geo_raster_uses_dem_entrypoint_and_full_context() -> None:
    """Raster ``rdr2geo`` dispatch selects the DEM-aware native ABI."""
    model = _model()
    context = _native_context(model, Operation.RDR2GEO)
    dem = _fake_raster_dem(np.full((6, 6), 42.0, dtype=np.float64))
    calls: list[tuple[str, tuple[object, ...]]] = []

    class NativeModule:
        """Minimal extension-shaped object exposing both CPU ABIs."""

        def rdr2geo_cpu(self, *values: object) -> list[np.ndarray]:
            calls.append(("rdr2geo_cpu", values))
            return _native_outputs(model, values[:3])  # type: ignore[arg-type]

        def rdr2geo_cpu_dem(self, *values: object) -> list[np.ndarray]:
            calls.append(("rdr2geo_cpu_dem", values))
            return _native_outputs(model, values[:3])  # type: ignore[arg-type]

    prepared = prepare_geometry(
        Operation.RDR2GEO,
        model,
        shape=(1,),
        dem=dem,
        native_executor=NativeModule(),
        native_context_inputs=context,
        native_key=_native_key(model, (1,), Operation.RDR2GEO, dem=dem),
        native_correctness_qualified=True,
    )

    execute_geometry(
        prepared,
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        selector="native",
    )

    assert [name for name, _ in calls] == ["rdr2geo_cpu_dem"]
    passed_context = calls[0][1][3:]
    assert len(passed_context) == 8
    np.testing.assert_array_equal(passed_context[5], context["dem_values"])
    np.testing.assert_array_equal(passed_context[6], context["dem_metadata"])
    np.testing.assert_array_equal(passed_context[7], context["dem_height_bounds"])


def test_real_module_native_adapter_uses_contiguous_torch_and_canonical_scalars() -> (
    None
):
    """A pybind-shaped module receives tensors and canonical scalar arguments."""
    import torch

    model = _model()
    context = _native_context(model, Operation.GEO2RDR)
    captured: list[tuple[object, ...]] = []
    module = ModuleType("faninsar_native_test")

    def geo2rdr_cpu(*values: object) -> list[np.ndarray]:
        captured.append(values)
        direct = tuple(np.asarray(value.detach().cpu()) for value in values[:3])
        return _native_outputs(model, direct)  # type: ignore[arg-type]

    module.geo2rdr_cpu = geo2rdr_cpu  # type: ignore[attr-defined]
    prepared = prepare_geometry(
        Operation.GEO2RDR,
        model,
        shape=(2, 2),
        native_executor=module,
        native_context_inputs=context,
        native_key=_native_key(model, (2, 2), Operation.GEO2RDR),
        native_correctness_qualified=True,
    )

    execute_geometry(
        prepared,
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        selector="native",
    )

    assert len(captured) == 1
    values = captured[0]
    for value in values[:6]:
        assert isinstance(value, torch.Tensor)
        assert value.dtype is torch.float64
        assert value.device.type == "cpu"
        assert value.is_contiguous()
    assert tuple(float(value) for value in values[6:11]) == (
        (model.sensing_start - model.orbit.epoch).total_seconds(),
        model.azimuth_time_interval_s,
        model.starting_slant_range_m,
        model.range_spacing_m,
        model.wavelength_m,
    )


def test_real_module_rdr2geo_adapter_uses_contiguous_torch_and_dem_abi() -> None:
    """CPU RDR2GEO ModuleType adapters receive the complete native ABI order."""
    import torch

    model = _model()
    context = _native_context(model, Operation.RDR2GEO)
    context["dem_metadata"] = np.array([10.0, 20.0, 0.1, 0.2], dtype=np.float64)
    context["dem_height_bounds"] = np.array([-100.0, 100.0], dtype=np.float64)
    captured: list[tuple[object, ...]] = []
    module = ModuleType("faninsar_native_rdr2geo_test")

    def rdr2geo_cpu_dem(*values: object) -> list[np.ndarray]:
        captured.append(values)
        direct = tuple(np.asarray(value.detach().cpu()) for value in values[:3])
        return _native_outputs(model, direct)  # type: ignore[arg-type]

    module.rdr2geo_cpu_dem = rdr2geo_cpu_dem  # type: ignore[attr-defined]
    dem = ConstantDEM(37.0)
    prepared = prepare_geometry(
        Operation.RDR2GEO,
        model,
        shape=(2, 2),
        dem=dem,
        native_executor=module,
        native_context_inputs=context,
        native_key=_native_key(model, (2, 2), Operation.RDR2GEO, dem=dem),
        native_correctness_qualified=True,
    )

    execute_geometry(
        prepared,
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.full((2, 2), 37.0, dtype=np.float64),
        selector="native",
    )

    assert len(captured) == 1
    values = captured[0]
    for index in (*range(6), 16):
        value = values[index]
        assert isinstance(value, torch.Tensor)
        assert value.dtype is torch.float64
        assert value.device.type == "cpu"
        assert value.is_contiguous()
    assert tuple(float(value) for value in values[6:11]) == (
        (model.sensing_start - model.orbit.epoch).total_seconds(),
        model.azimuth_time_interval_s,
        model.starting_slant_range_m,
        model.range_spacing_m,
        model.wavelength_m,
    )
    assert values[11:16] == (20, 0, 0.01, 0.1, True)
    assert tuple(float(value) for value in values[17:21]) == (10.0, 20.0, 0.1, 0.2)
    assert values[21:] == (50, 1.0e-3)


def test_cuda_dem_metadata_helper_uses_cuda_abi_order() -> None:
    """The CUDA metadata reorder is independently testable on CPU."""
    assert geometry_public._cuda_dem_metadata_abi_order([10.0, 20.0, 0.1, 0.2]) == (
        20.0,
        10.0,
        0.2,
        0.1,
    )


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA is unavailable"
)
def test_cuda_omitted_height_is_created_on_target_device() -> None:
    """Omitted radar height uses a contiguous float64 tensor on CUDA."""
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_key = DeviceKey.cuda(str(properties.uuid))
    values = geometry_public._validate_public_inputs(
        Operation.RDR2GEO,
        (2,),
        (torch.zeros(2, device="cuda", dtype=torch.float64),) * 2,
        device_key,
        default_height_m=37.0,
        target_device="cuda",
    )

    height = values[2]
    assert isinstance(height, torch.Tensor)
    assert height.device.type == "cuda"
    assert height.dtype is torch.float64
    assert height.is_contiguous()
    # The native geometry ABI is float64, while torch.full defaults to the
    # process-wide default dtype (usually float32).  Compare against a tensor
    # derived from the actual public value so this test checks the value and
    # device contract without introducing a conflicting dtype assumption.
    torch.testing.assert_close(height, torch.full_like(height, 37.0))


@pytest.mark.parametrize("device_alias", ["auto", "gpu"])
def test_cuda_alias_omitted_height_uses_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
    device_alias: str,
) -> None:
    """CUDA auto aliases pass the resolved device to constant-height creation."""
    torch = pytest.importorskip("torch")
    resolved = torch.device("cuda")
    captured: list[object] = []

    def resolve(value: object) -> object:
        assert value == device_alias
        return resolved

    def fake_full(
        shape: tuple[int, ...],
        fill_value: float,
        *,
        dtype: object,
        device: object,
    ) -> torch.Tensor:
        del fill_value, dtype
        captured.append(device)
        return torch.zeros(shape)

    monkeypatch.setattr(geometry_public, "resolve_geometry_device", resolve)
    monkeypatch.setattr(torch, "full", fake_full)
    monkeypatch.setattr(
        geometry_public, "validate_tensor_span", lambda *_args, **_kwargs: None
    )

    target = geometry_public.resolve_geometry_device(device_alias)
    values = geometry_public._validate_public_inputs(
        Operation.RDR2GEO,
        (2,),
        (torch.zeros(2),) * 2,
        DeviceKey.cuda("GPU-test"),
        default_height_m=37.0,
        target_device=target,
    )

    assert captured == [resolved]
    assert isinstance(values[2], torch.Tensor)


def test_cuda_native_numpy_inputs_are_normalized_before_span_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NumPy native inputs are copied to the admitted CUDA device first."""
    torch = pytest.importorskip("torch")
    captured: list[tuple[object, object]] = []
    resolved = torch.device("cuda")

    def fake_as_tensor(
        value: np.ndarray,
        *,
        dtype: object,
        device: object,
    ) -> torch.Tensor:
        captured.append((dtype, device))
        return torch.from_numpy(value)

    monkeypatch.setattr(torch, "as_tensor", fake_as_tensor)
    monkeypatch.setattr(
        geometry_public, "validate_tensor_span", lambda *_args, **_kwargs: None
    )
    source = np.zeros(2, dtype=np.float64)
    values = geometry_public._validate_public_inputs(
        Operation.GEO2RDR,
        (2,),
        (source, source.copy(), source.copy()),
        DeviceKey.cuda("GPU-test"),
        target_device=resolved,
    )

    assert captured == [(torch.float64, resolved)] * 3
    assert all(isinstance(value, torch.Tensor) for value in values)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA is unavailable"
)
def test_cuda_module_adapter_reorders_canonical_dem_metadata() -> None:
    """CUDA pybind calls receive DEM metadata in x/y ABI order."""
    import torch

    captured: list[tuple[object, ...]] = []
    module = ModuleType("faninsar_native_cuda_test")

    def rdr2geo_cuda(*values: object) -> list[object]:
        captured.append(values)
        return []

    module.rdr2geo_cuda = rdr2geo_cuda  # type: ignore[attr-defined]
    entry = geometry_public._resolve_native_entrypoint(
        module,
        Operation.RDR2GEO,
        DeviceKey.cuda("test-device"),
        SolverSettings(),
    )

    def gpu(values: object) -> torch.Tensor:
        """Materialize one CUDA fixture tensor."""
        return torch.tensor(values, dtype=torch.float64, device="cuda")

    entry(
        gpu([0.0]),
        gpu([0.0]),
        gpu([11.0]),
        gpu([0.0, 1.0]),
        gpu([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        gpu([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        gpu([10.0, 0.002, 800000.0, 2.3, 0.0555]),
        True,
        gpu(np.zeros((6, 6))),
        gpu([10.0, 20.0, 0.1, 0.2]),
        gpu([-100.0, 100.0]),
    )

    assert len(captured) == 1
    assert [float(value) for value in captured[0][11:15]] == [20.0, 10.0, 0.2, 0.1]


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


def test_cuda_aliases_canonicalize_to_the_current_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unqualified CUDA and the current explicit ordinal compare equally."""
    import torch

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert torch_backends_v2._canonical_torch_device("cuda") == torch.device("cuda:0")
    assert torch_backends_v2._canonical_torch_device("cuda:0") == torch.device("cuda:0")


def test_cuda_unsupported_ordinal_fails_during_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit CUDA ordinal outside the visible device set is rejected."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="unsupported device ordinal 1"):
        torch_backends_v2._resolve_device("cuda:1")


def test_geo2rdr_convergence_uses_physical_metric_not_newton_step() -> None:
    """Keep last-step roundoff from changing the public convergence vote."""
    source = inspect.getsource(torch_kernels._geo2rdr_step)
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
        dem=ConstantDEM(50.0),
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
            dem=ConstantDEM(50.0),
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
                dem=ConstantDEM(50.0),
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


def test_native_rdr2geo_candidate_path_is_rejected() -> None:
    """RDR2GEO native dispatch requires the module adapter's DEM ABI."""
    model = _model()
    plan = BuildPlan(
        GeometryOperation.RDR2GEO,
        NativeBackend.CPU,
        "faninsar_rdr2geo_v2_cpu",
        "faninsar_rdr2geo_v2_cpu",
        (),
        (),
        (),
    )
    candidate = PreparedNativeCandidate(
        plan,
        PreparationStatus.PREPARED,
        _entry_point=lambda *_: (),
    )
    with pytest.raises(
        DispatchError, match="rdr2geo native candidates are unsupported"
    ):
        prepare_geometry(
            Operation.RDR2GEO,
            model,
            shape=(1,),
            native_candidate=candidate,
            native_key=_native_key(model, (1,), Operation.RDR2GEO),
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


def test_auto_eager_fallback_preserves_cuda_device_identity() -> None:
    """A CUDA preparation without candidates keeps its same-device key."""
    eager = prepare_torch_geometry("geo2rdr", _model(), shape=(1,))
    cuda_profile = ExecutionProfile.cuda("GPU-test")
    fallback_key = geometry_public._key(
        Operation.GEO2RDR,
        "compile",
        eager,
        cuda_profile,
    )
    prepared = geometry_public.PreparedGeometry(
        Operation.GEO2RDR,
        _model(),
        (1,),
        eager,
        Dispatcher(lambda: object()),
        eager.device,
        eager.dtype,
        fallback_key=fallback_key,
    )

    key = prepared._key_for_selector("auto")

    assert key.backend == "compile"
    assert key.device == DeviceKey.cuda("GPU-test")
    assert key.profile is not None
    assert key.profile.device == DeviceKey.cuda("GPU-test")


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
        "rdr2geo", _model(), shape=(1,), dem=ConstantDEM(10.0)
    )
    second = prepare_torch_geometry(
        "rdr2geo", _model(), shape=(1,), dem=ConstantDEM(11.0)
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
        dem=GeoidAdjustedDEM(ConstantDEM(10.0), ConstantDEM(1.0)),
    )
    second = prepare_torch_geometry(
        "rdr2geo",
        _model(),
        shape=(1,),
        dem=GeoidAdjustedDEM(ConstantDEM(10.0), ConstantDEM(2.0)),
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
    """Infinite rdr2geo DEM samples are rejected before callback invocation."""
    model = _model()
    context = _native_context(model, Operation.RDR2GEO)
    dem_values = np.asarray(context["dem_values"]).copy()
    dem_values[0, 0] = np.inf
    context["dem_values"] = dem_values
    calls: list[int] = []
    with pytest.raises(GeometryValidationError, match="infinities"):
        prepare_geometry(
            Operation.RDR2GEO,
            model,
            shape=(1,),
            native_executor=lambda *_: calls.append(1),
            native_context_inputs=context,
            native_key=_native_key(model, (1,), Operation.RDR2GEO),
        )
    assert calls == []


def test_native_context_allows_nan_dem_nodata() -> None:
    """NaN DEM nodata is part of the native raster contract."""
    from faninsar.processing.geometry.public import _validate_native_context_inputs
    from faninsar.processing.geometry.v2 import DeviceKey

    model = _model()
    context = _native_context(model, Operation.RDR2GEO)
    dem_values = np.asarray(context["dem_values"]).copy()
    dem_values[0, 0] = np.nan
    context["dem_values"] = dem_values
    values = _validate_native_context_inputs(
        Operation.RDR2GEO, context, DeviceKey.cpu()
    )
    assert np.isnan(np.asarray(values[5])[0, 0])


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
