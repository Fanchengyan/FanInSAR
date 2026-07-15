"""Verify analytic generators, typed metrics, gates, schema, and evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.reference.manifest import load_manifest
from tests.reference.metrics import (
    MetricInputError,
    closure_metrics,
    coherence_metrics,
    displacement_metrics,
    geolocation_metrics,
    geometry_metrics,
    offset_metrics,
    performance_metrics,
    phase_metrics,
)
from tests.reference.reporting import (
    EvidenceContext,
    evaluate_gates,
    metric_gates_from_manifest,
    write_evidence_bundle,
)
from tests.reference.synthetic import (
    SyntheticInputError,
    closure_case,
    displacement_case,
    geolocation_case,
    offset_case,
    phase_case,
    ramp,
)


def test_ramp_is_deterministic_and_has_declared_gradient() -> None:
    """Verify repeated ramps preserve declared gradients exactly."""
    first = ramp((4, 5), range_slope=2.0, azimuth_slope=-0.5, intercept=3.0)
    second = ramp((4, 5), range_slope=2.0, azimuth_slope=-0.5, intercept=3.0)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.diff(first, axis=1), 2.0)
    np.testing.assert_allclose(np.diff(first, axis=0), -0.5)


def test_geometry_metrics_report_percentiles_and_round_trip_error() -> None:
    """Verify geometry metrics report known percentile residuals."""
    expected = np.zeros((2, 3, 2), dtype=np.float64)
    actual = expected.copy()
    actual[..., 0] = np.array([[0.0, 0.03, 0.04], [0.05, 0.08, 0.10]])

    result = geometry_metrics(expected, actual)

    assert result.valid_count == 6
    assert result.p95_sample_error == pytest.approx(0.095)
    assert result.p99_sample_error == pytest.approx(0.099)
    assert result.maximum_sample_error == pytest.approx(0.1)


def test_phase_and_coherence_metrics_match_known_case() -> None:
    """Verify phase and coherence metrics recover injected biases."""
    case = phase_case((6, 7), phase_bias=0.02, coherence_bias=-0.01)

    wrapped = phase_metrics(
        case.wrapped_reference, case.wrapped_candidate, wrapped=True
    )
    unwrapped = phase_metrics(
        case.unwrapped_reference,
        case.unwrapped_candidate,
        wrapped=False,
    )
    coherence = coherence_metrics(case.coherence_reference, case.coherence_candidate)

    assert wrapped.circular_rmse_rad == pytest.approx(0.02)
    assert unwrapped.rmse_rad == pytest.approx(0.02)
    assert coherence.mean_absolute_error == pytest.approx(0.01)
    assert coherence.median_loss == pytest.approx(0.01)


def test_offset_metrics_separate_range_and_azimuth_components() -> None:
    """Verify offset metrics preserve component errors and valid coverage."""
    case = offset_case((5, 6), range_shift=0.04, azimuth_shift=-0.003)
    candidate = case.candidate.copy()
    candidate[0, :3] = np.nan

    result = offset_metrics(case.reference, candidate)

    assert result.range_rmse_pixel == pytest.approx(0.04)
    assert result.azimuth_rmse_pixel == pytest.approx(0.003)
    assert result.coverage_fraction == pytest.approx(0.9)


def test_closure_metrics_detect_one_corrupted_pair() -> None:
    """Verify loop closure detects a known corrupted pair residual."""
    case = closure_case((4, 5), closure_bias=0.25)

    result = closure_metrics(case.pair_phase, case.loops)

    assert result.loop_count == 1
    assert result.circular_rmse_rad == pytest.approx(0.25)
    assert result.maximum_absolute_rad == pytest.approx(0.25)


def test_geolocation_and_displacement_metrics_match_known_errors() -> None:
    """Verify location, displacement, velocity, and uncertainty metrics."""
    geolocation = geolocation_case((3, 4), horizontal_error_m=2.0, vertical_error_m=1.0)
    displacement = displacement_case(
        acquisition_count=5,
        shape=(3, 4),
        velocity_m_per_year=0.01,
        candidate_bias_m=0.002,
    )

    geo_result = geolocation_metrics(geolocation.reference, geolocation.candidate)
    displacement_result = displacement_metrics(
        displacement.reference,
        displacement.candidate,
        displacement.elapsed_years,
        np.full(displacement.reference.shape, 0.003, dtype=np.float64),
    )

    assert geo_result.horizontal_rmse_m == pytest.approx(2.0)
    assert geo_result.vertical_rmse_m == pytest.approx(1.0)
    assert displacement_result.displacement_rmse_m == pytest.approx(0.002)
    assert displacement_result.velocity_bias_m_per_year == pytest.approx(0.0, abs=1e-12)
    assert displacement_result.uncertainty_95_coverage_fraction == pytest.approx(1.0)


def test_performance_metrics_are_typed_and_serializable() -> None:
    """Verify performance summaries remain scalar and JSON serializable."""
    result = performance_metrics(
        runtime_seconds=[1.0, 1.2, 0.8],
        peak_memory_bytes=[100, 120, 110],
        processed_pixels=1_000,
    )

    payload = result.to_json_dict()

    assert result.runtime_median_seconds == pytest.approx(1.0)
    assert result.runtime_p95_seconds == pytest.approx(1.18)
    assert result.peak_memory_bytes == 120
    assert payload["metric_family"] == "performance"
    json.dumps(payload)


def test_metrics_reject_malformed_or_non_finite_inputs() -> None:
    """Verify metric boundaries reject malformed and non-finite arrays."""
    with pytest.raises(MetricInputError, match="shape"):
        geometry_metrics(np.zeros((2, 2, 2)), np.zeros((2, 3, 2)))
    with pytest.raises(MetricInputError, match="finite"):
        phase_metrics(np.zeros((2, 2)), np.full((2, 2), np.nan))
    with pytest.raises(MetricInputError, match=r"\[0, 1\]"):
        coherence_metrics(np.zeros((2, 2)), np.full((2, 2), 1.1))
    with pytest.raises(MetricInputError, match="last axis"):
        offset_metrics(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)))
    with pytest.raises(MetricInputError, match="time axis"):
        displacement_metrics(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.array([0.0, 0.5, 1.0]),
        )
    with pytest.raises(MetricInputError, match="runtime"):
        performance_metrics([], [1], 1)
    with pytest.raises(MetricInputError, match="signed one-based"):
        closure_metrics(np.zeros((3, 2, 2)), np.array([[0, 1, 2]]))


def test_synthetic_arrays_are_immutable_and_invalid_requests_fail() -> None:
    """Verify generated references cannot be mutated or created invalidly."""
    generated = ramp((2, 3), range_slope=1.0, azimuth_slope=1.0)

    assert generated.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        generated[0, 0] = 2.0
    with pytest.raises(SyntheticInputError, match="positive dimensions"):
        ramp((0, 3), range_slope=1.0, azimuth_slope=1.0)


def test_metric_schema_declares_all_metric_families() -> None:
    """Verify the report schema enumerates every metric family."""
    schema_path = Path(__file__).with_name("metric.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    metric_schema = schema["$defs"]["metric"]
    families = set(metric_schema["properties"]["metric_family"]["enum"])

    assert families == {
        "geometry",
        "phase",
        "coherence",
        "offset",
        "closure",
        "geolocation",
        "displacement",
        "performance",
    }
    assert metric_schema["additionalProperties"] is False


def test_manifest_supplied_gates_drive_evaluation_and_evidence(tmp_path: Path) -> None:
    """Verify evidence gates use values loaded from the frozen manifest."""
    manifest = load_manifest(Path(__file__).with_name("manifest.yaml"))
    result = geometry_metrics(
        np.zeros((2, 2, 2), dtype=np.float64),
        np.full((2, 2, 2), 0.2, dtype=np.float64),
    )
    gates = metric_gates_from_manifest(manifest.metric_gates, "geometry")

    evaluations = evaluate_gates(result, gates)
    artifacts = write_evidence_bundle(
        output_directory=tmp_path,
        metric=result,
        gates=gates,
        context=EvidenceContext(
            versions={"numpy": np.__version__},
            input_hashes={"manifest.yaml": "sha256:fixture"},
            command_log=("uv run pytest tests/reference -m 'not slow'",),
        ),
    )

    assert gates[0].threshold == manifest.metric_gates["coordinate_residual_pixels_max"]
    assert [evaluation.passed for evaluation in evaluations] == [False]
    payload = json.loads(artifacts.metric_json.read_text(encoding="utf-8"))
    assert payload["gates"][0]["passed"] is False
    assert payload["input_hashes"]["manifest.yaml"] == "sha256:fixture"
    assert artifacts.quicklook_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_manifest_exposes_velocity_bias_and_uncertainty_coverage_gates() -> None:
    """Verify displacement gates include velocity bias and 95 percent coverage."""
    manifest = load_manifest(Path(__file__).with_name("manifest.yaml"))

    gates = metric_gates_from_manifest(manifest.metric_gates, "displacement")
    gates_by_field = {gate.field: gate for gate in gates}

    assert gates_by_field["velocity_bias_m_per_year"].threshold == (
        manifest.metric_gates["velocity_bias_millimeters_per_year_max"] / 1_000
    )
    assert (
        gates_by_field["uncertainty_95_coverage_fraction"].threshold
        == (manifest.metric_gates["uncertainty_95_coverage_fraction_min"])
    )
    assert gates_by_field["uncertainty_95_coverage_fraction"].mode == "minimum"


def test_persisted_metric_report_conforms_to_schema(tmp_path: Path) -> None:
    """Verify the exact persisted evidence report validates against its schema."""
    import jsonschema

    manifest = load_manifest(Path(__file__).with_name("manifest.yaml"))
    result = geometry_metrics(
        np.zeros((2, 2, 2), dtype=np.float64),
        np.full((2, 2, 2), 0.02, dtype=np.float64),
    )
    artifacts = write_evidence_bundle(
        output_directory=tmp_path,
        metric=result,
        gates=metric_gates_from_manifest(manifest.metric_gates, "geometry"),
        context=EvidenceContext(
            versions={"numpy": np.__version__},
            input_hashes={"manifest.yaml": "sha256:fixture"},
        ),
    )
    report = json.loads(artifacts.metric_json.read_text(encoding="utf-8"))
    schema = json.loads(
        Path(__file__).with_name("metric.schema.json").read_text(encoding="utf-8")
    )

    jsonschema.validate(report, schema)
