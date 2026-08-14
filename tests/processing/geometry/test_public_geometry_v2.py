"""Integration tests for the public geometry-v2 preparation seam."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.geometry import (
    Operation,
    RadarGeometryModel,
    execute_geometry,
    prepare_geometry,
)
from faninsar.processing.geometry.transforms import geo2rdr


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


def _native_outputs(
    model: RadarGeometryModel, values: tuple[np.ndarray, ...]
) -> list[np.ndarray]:
    """Build a fourteen-field native-like output from the reference result."""
    result = geo2rdr(model, *values)
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
        native_executor=lambda *inputs: _native_outputs(model, inputs),
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
        native_executor=lambda *inputs: calls.append("native")
        or _native_outputs(model, inputs),
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
