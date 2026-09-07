"""Integration tests for the public geometry-v2 preparation seam."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry import (
    Operation,
    RadarGeometryModel,
    execute_geometry,
    prepare_geometry,
)
from faninsar.processing.geometry.prepare_production import run_geo2rdr
from faninsar.processing.geometry.torch_backends_v2 import prepare_torch_geometry
from faninsar.processing.geometry.v2 import (
    CandidateKey,
    DeviceKey,
    ExecutionProfile,
    SolverSettings,
)


def _model() -> RadarGeometryModel:
    """Return a small well-conditioned synthetic radar geometry model."""
    epoch = datetime(2020, 1, 1, tzinfo=UTC)
    vectors = tuple(
        OrbitStateVector(
            time=epoch + timedelta(seconds=10 * index),
            position_m=(7_071_000.0, 0.0, 0.0),
            velocity_m_s=(0.0, 7_500.0, 0.0),
        )
        for index in range(5)
    )
    orbit = OrbitMetadata("ITRF", "public-v2-test", vectors)
    grid = RadarGrid(
        shape=(8, 8),
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=epoch + timedelta(seconds=20),
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(grid, orbit)


def _native_context(
    model: RadarGeometryModel,
    operation: Operation,
) -> dict[str, object]:
    """Return explicit, owner-backed context for the native ABI fixture."""
    times = np.asarray(model.orbit.times_s, dtype=np.float64).copy()
    positions = np.ascontiguousarray(
        np.stack([spline(times) for spline in model.orbit.trajectory_splines], axis=-1),
        dtype=np.float64,
    )
    velocities = np.ascontiguousarray(
        np.stack(
            [spline(times, 1) for spline in model.orbit.trajectory_splines], axis=-1
        ),
        dtype=np.float64,
    )
    context: dict[str, object] = {
        "orbit_times": times,
        "orbit_positions": positions,
        "orbit_velocities": velocities,
        "model_parameters": np.array(
            [
                (model.sensing_start - model.orbit.epoch).total_seconds(),
                model.azimuth_time_interval_s,
                model.starting_slant_range_m,
                model.range_spacing_m,
                model.wavelength_m,
            ],
            dtype=np.float64,
        ),
        "look_right": model.look_direction == "right",
    }
    if operation is Operation.RDR2GEO:
        context.update(
            {
                "dem_values": np.zeros((6, 6), dtype=np.float64),
                "dem_metadata": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
                "dem_height_bounds": np.array([-1000.0, 10000.0], dtype=np.float64),
            }
        )
    return context


def _native_outputs(
    model: RadarGeometryModel, values: tuple[np.ndarray, ...]
) -> list[np.ndarray]:
    """Build a fourteen-field native-like output from the reference result."""
    result = run_geo2rdr(model, *values, device="cpu", max_iter=20)
    shape = result.latitude_deg.shape
    return [
        np.asarray(result.latitude_deg, dtype=np.float64),
        np.asarray(result.longitude_deg, dtype=np.float64),
        np.asarray(result.height_m, dtype=np.float64),
        np.asarray(result.range_index, dtype=np.float64),
        np.asarray(result.azimuth_index, dtype=np.float64),
        np.asarray(result.converged, dtype=bool),
        np.where(result.converged, 1, -1).astype(np.int32),
        np.asarray(result.residual_range_m, dtype=np.float64),
        np.asarray(result.residual_range_m, dtype=np.float64),
        np.ones(shape, dtype=np.float64),
        (~result.converged).astype(bool),
        np.zeros(shape, dtype=bool),
        np.asarray(result.residual_range_m, dtype=np.float64),
        np.asarray(result.residual_doppler_hz, dtype=np.float64),
    ]


def _native_key(
    model: RadarGeometryModel,
    shape: tuple[int, ...],
    operation: Operation,
    solver: SolverSettings | None = None,
    dem: object | None = None,
) -> CandidateKey:
    """Build a complete explicit native manifest for the test fixture."""
    solver = solver or SolverSettings()
    if (
        operation is Operation.GEO2RDR
        and solver.slant_range_tolerance_m != solver.range_tolerance_m
    ):
        solver = SolverSettings(
            max_iter=solver.max_iter,
            extra_iter=solver.extra_iter,
            range_tolerance_m=solver.range_tolerance_m,
            doppler_tolerance_hz=solver.doppler_tolerance_hz,
            slant_range_tolerance_m=solver.range_tolerance_m,
        )
    prepared = prepare_torch_geometry(
        operation,
        model,
        shape=shape,
        dem=dem,
        max_iter=solver.max_iter,
        extra_iter=solver.extra_iter,
        range_tol_m=solver.range_tolerance_m,
        doppler_tol_hz=solver.doppler_tolerance_hz,
    )
    return CandidateKey(
        operation=operation,
        backend="native",
        device=DeviceKey.cpu(),
        dtype=prepared.dtype,
        shape=shape,
        solver=solver,
        orbit_digest=prepared.identity.orbit_digest,
        dem_digest=prepared.identity.dem_digest,
        model_digest=prepared.identity.model_digest,
        source_digest="source-digest",
        toolchain_digest="toolchain-digest",
        runtime_digest="runtime-digest",
        artifact_digest="artifact-digest",
        abi_digest="abi-digest",
        support_contract_digest=prepared.identity.settings_digest,
        profile=ExecutionProfile.cpu(),
    )


def test_public_selectors_execute_prepared_eager_and_native() -> None:
    """Explicit selectors use prepared candidates and normalize native output."""
    model = _model()
    values = (
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
    )
    prepared = prepare_geometry(
        Operation.GEO2RDR,
        model,
        shape=(1,),
        native_executor=lambda *inputs: _native_outputs(model, inputs[:3]),
        native_context_inputs=_native_context(model, Operation.GEO2RDR),
        native_key=_native_key(model, (1,), Operation.GEO2RDR),
        native_correctness_qualified=True,
        native_performance_eligible=True,
    )

    eager = execute_geometry(prepared, *values, selector="eager")
    native = execute_geometry(prepared, *values, selector="native")
    assert eager.fields == native.fields
    assert native.tolerance.dtype == np.dtype(np.float64)
    assert prepared.dispatcher.records[-1].backend == "native"


def test_public_auto_requires_both_native_qualification_gates() -> None:
    """Auto uses eager when the native candidate is not performance eligible."""
    model = _model()
    values = tuple(np.zeros(1, dtype=np.float64) for _ in range(3))
    calls: list[str] = []
    prepared = prepare_geometry(
        Operation.GEO2RDR,
        model,
        shape=(1,),
        native_executor=lambda *inputs: (
            calls.append("native") or _native_outputs(model, inputs[:3])
        ),
        native_context_inputs=_native_context(model, Operation.GEO2RDR),
        native_key=_native_key(model, (1,), Operation.GEO2RDR),
        native_correctness_qualified=True,
        native_performance_eligible=False,
    )

    result = execute_geometry(prepared, *values, selector="auto")
    assert result.fields
    assert calls == []
    assert prepared.dispatcher.records[-1].backend == "eager"


def test_public_missing_explicit_backend_fails_without_compilation() -> None:
    """A missing explicit candidate raises instead of compiling on demand."""
    prepared = prepare_geometry(Operation.RDR2GEO, _model(), shape=(1,))
    with pytest.raises(Exception, match=r"prepared compile|exact prepared"):
        execute_geometry(
            prepared,
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            selector="compile",
        )
