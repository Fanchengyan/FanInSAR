"""Focused tests for the PROPOSAL-0026 foundation contracts."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.geometry import (
    INT32_MAX,
    CandidateKey,
    DeviceKey,
    ExecutionProfile,
    GeometryValidationError,
    Operation,
    OperationSettings,
    SolverSettings,
    TransformResultV2,
    validate_array_span,
)


def test_invalid_result_has_exact_fourteen_fields_and_invalid_tolerance() -> None:
    """Invalid lanes publish every required field and NaN tolerance."""
    result = TransformResultV2.invalid((2,), operation=Operation.GEO2RDR)

    assert result.fields == (
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "range_index",
        "azimuth_index",
        "converged",
        "iterations",
        "decision_residual",
        "final_residual",
        "tolerance",
        "max_iter_exhausted",
        "boundary_rechecked",
        "residual_range_m",
        "residual_doppler_hz",
    )
    assert len(result.fields) == 14
    assert result.iterations.dtype == np.int32
    assert np.array_equal(result.iterations, [-1, -1])
    assert np.all(np.isnan(result.tolerance))
    assert not np.any(result.converged)


def test_invalid_rdr2geo_lanes_use_nan_tolerance() -> None:
    """Radar-to-geo invalid lanes use NaN for all floating diagnostics."""
    result = TransformResultV2.invalid(3, operation="rdr2geo")
    assert np.all(np.isnan(result.tolerance))
    assert np.all(np.isnan(result.latitude_deg))
    assert np.array_equal(result.iterations, np.full(3, -1, dtype=np.int32))


def test_from_arrays_applies_per_lane_geo2rdr_tolerance_and_sentinels() -> None:
    """Valid geo lanes use one while invalid lanes are rewritten to NaN."""
    shape = (2,)
    fields = {
        "latitude_deg": np.zeros(shape, dtype=np.float64),
        "longitude_deg": np.zeros(shape, dtype=np.float64),
        "height_m": np.zeros(shape, dtype=np.float64),
        "range_index": np.zeros(shape, dtype=np.float64),
        "azimuth_index": np.zeros(shape, dtype=np.float64),
        "converged": np.array([True, False], dtype=bool),
        "iterations": np.array([1, 3], dtype=np.int32),
        "decision_residual": np.zeros(shape, dtype=np.float64),
        "final_residual": np.zeros(shape, dtype=np.float64),
        "tolerance": np.ones(shape, dtype=np.float64),
        "max_iter_exhausted": np.array([False, True], dtype=bool),
        "boundary_rechecked": np.zeros(shape, dtype=bool),
        "residual_range_m": np.zeros(shape, dtype=np.float64),
        "residual_doppler_hz": np.zeros(shape, dtype=np.float64),
    }
    result = TransformResultV2.from_arrays(
        fields, operation="geo2rdr", invalid_mask=np.array([False, True], dtype=bool)
    )
    assert result.tolerance[0] == 1.0
    assert np.isnan(result.tolerance[1])
    assert result.iterations[1] == -1


def test_result_rejects_wrong_dtypes() -> None:
    """Result publication does not silently wrap or coerce ABI fields."""
    fields = {
        name: np.zeros(1, dtype=np.float64)
        for name in TransformResultV2.invalid(1).fields
    }
    fields["converged"] = np.zeros(1, dtype=bool)
    fields["iterations"] = np.zeros(1, dtype=np.int32)
    fields["max_iter_exhausted"] = np.zeros(1, dtype=bool)
    fields["boundary_rechecked"] = np.zeros(1, dtype=bool)
    fields["iterations"] = np.zeros(1, dtype=np.int64)
    with pytest.raises(GeometryValidationError, match="iterations"):
        TransformResultV2.from_arrays(fields)


@pytest.mark.parametrize(
    ("max_iter", "extra_iter"),
    [(0, 0), (1, -1), (INT32_MAX, 1), (INT32_MAX + 1, 0)],
)
def test_solver_settings_check_iteration_budget(max_iter: int, extra_iter: int) -> None:
    """Iteration settings reject zero, negative, and int32-overflow budgets."""
    with pytest.raises(GeometryValidationError):
        SolverSettings(max_iter=max_iter, extra_iter=extra_iter)


def test_identity_is_tagged_and_scientific_projection_changes() -> None:
    """Device/profile tags and operation settings participate in candidate identity."""
    cpu = DeviceKey.cpu()
    cuda = DeviceKey.cuda("GPU-abc")
    assert cpu != cuda
    profile = ExecutionProfile.cpu(openmp_runtime="libomp", thread_count=4)
    first = CandidateKey(
        Operation.GEO2RDR,
        "native",
        cpu,
        shape=(4, 5),
        profile=profile,
    )
    changed = CandidateKey(
        Operation.RDR2GEO,
        "native",
        cpu,
        shape=(4, 5),
        profile=ExecutionProfile.cpu(openmp_runtime="libomp", thread_count=4),
    )
    assert first.digest != changed.digest
    assert first.scientific_identity != changed.scientific_identity


def test_array_span_validates_before_returning_pointer() -> None:
    """The span includes an owner and exact C-order byte extent."""
    array = np.ones((2, 3), dtype=np.float64)
    span = validate_array_span(array, dtype=np.float64)
    assert span.owner is array
    assert span.byte_length == array.nbytes
    assert span.address > 0
    with pytest.raises(GeometryValidationError, match="dtype"):
        validate_array_span(array.astype(np.float32), dtype=np.float64)


def test_operation_settings_exposes_checked_budget() -> None:
    """Operation settings preserve the operation-specific solver projection."""
    settings = OperationSettings("geo2rdr", SolverSettings(max_iter=2, extra_iter=3))
    assert settings.budget == 5
    assert settings.canonical()["operation"] == "geo2rdr"
    assert settings.canonical()["solver"]["range_tolerance_m"] == 0.01
