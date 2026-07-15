"""Manifest-driven gate evaluation and reproducible evidence artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, assert_never

import matplotlib.pyplot as plt

from faninsar.logging import setup_logger
from tests.reference.metric_types import (
    JsonScalar,
    MetricFamily,
    MetricInputError,
    MetricResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

GateMode = Literal["maximum", "minimum"]
logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class MetricGate:
    """A threshold supplied by the reference manifest."""

    field: str
    mode: GateMode
    threshold: float


@dataclass(frozen=True, slots=True)
class ManifestGateBinding:
    """Bind one manifest threshold key to a typed metric field and unit scale."""

    manifest_key: str
    field: str
    mode: GateMode
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    """The observed result of applying one manifest gate."""

    field: str
    mode: GateMode
    threshold: float
    observed: float
    passed: bool


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    """Reproducibility metadata attached to a metric evidence bundle."""

    versions: Mapping[str, str]
    input_hashes: Mapping[str, str]
    command_log: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class EvidenceArtifacts:
    """Paths written for one scientific metric evidence bundle."""

    metric_json: Path
    quicklook_png: Path


def _gate_bindings(family: MetricFamily) -> tuple[ManifestGateBinding, ...]:
    match family:
        case "geometry":
            bindings = (
                ManifestGateBinding(
                    "coordinate_residual_pixels_max", "p99_sample_error", "maximum"
                ),
            )
        case "phase":
            bindings = (
                ManifestGateBinding(
                    "wrapped_phase_circular_rmse_radians_max",
                    "circular_rmse_rad",
                    "maximum",
                ),
            )
        case "coherence":
            bindings = (
                ManifestGateBinding(
                    "coherence_absolute_error_max",
                    "mean_absolute_error",
                    "maximum",
                ),
            )
        case "offset":
            bindings = (
                ManifestGateBinding(
                    "range_offset_rmse_pixels_max", "range_rmse_pixel", "maximum"
                ),
                ManifestGateBinding(
                    "azimuth_offset_rmse_pixels_max",
                    "azimuth_rmse_pixel",
                    "maximum",
                ),
            )
        case "closure":
            bindings = (
                ManifestGateBinding(
                    "closure_phase_radians_max",
                    "maximum_absolute_rad",
                    "maximum",
                ),
            )
        case "geolocation":
            bindings = (
                ManifestGateBinding(
                    "geolocation_rmse_meters_max",
                    "horizontal_rmse_m",
                    "maximum",
                ),
            )
        case "displacement":
            bindings = (
                ManifestGateBinding(
                    "displacement_rmse_millimeters_max",
                    "displacement_rmse_m",
                    "maximum",
                    0.001,
                ),
                ManifestGateBinding(
                    "velocity_bias_millimeters_per_year_max",
                    "velocity_bias_m_per_year",
                    "maximum",
                    0.001,
                ),
                ManifestGateBinding(
                    "uncertainty_95_coverage_fraction_min",
                    "uncertainty_95_coverage_fraction",
                    "minimum",
                ),
            )
        case "performance":
            bindings = ()
        case unreachable:
            assert_never(unreachable)
    return bindings


def metric_gates_from_manifest(
    manifest_gates: Mapping[str, float],
    family: MetricFamily,
) -> tuple[MetricGate, ...]:
    """Resolve metric gates from loaded manifest thresholds and unit bindings."""
    resolved: list[MetricGate] = []
    for binding in _gate_bindings(family):
        threshold = manifest_gates.get(binding.manifest_key)
        if threshold is None:
            message = f"manifest metric gate is missing: {binding.manifest_key}"
            logger.error(message)
            raise MetricInputError(message)
        resolved.append(
            MetricGate(
                field=binding.field,
                mode=binding.mode,
                threshold=threshold * binding.scale,
            )
        )
    return tuple(resolved)


def _numeric_field(metric: MetricResult, field: str) -> float:
    value = metric.to_json_dict().get(field)
    match value:
        case str() | bool() | None:
            message = f"metric field {field!r} is absent or non-numeric"
            logger.error(message)
            raise MetricInputError(message)
        case float() | int():
            return float(value)
        case unreachable:
            assert_never(unreachable)


def evaluate_gates(
    metric: MetricResult,
    gates: Sequence[MetricGate],
) -> tuple[GateEvaluation, ...]:
    """Apply manifest-provided thresholds without defining local gate values."""
    evaluations: list[GateEvaluation] = []
    for gate in gates:
        observed = _numeric_field(metric, gate.field)
        match gate.mode:
            case "maximum":
                passed = observed <= gate.threshold
            case "minimum":
                passed = observed >= gate.threshold
            case unreachable:
                assert_never(unreachable)
        evaluations.append(
            GateEvaluation(
                field=gate.field,
                mode=gate.mode,
                threshold=gate.threshold,
                observed=observed,
                passed=passed,
            )
        )
    return tuple(evaluations)


def write_evidence_bundle(
    output_directory: Path,
    metric: MetricResult,
    gates: Sequence[MetricGate],
    context: EvidenceContext,
) -> EvidenceArtifacts:
    """Write deterministic JSON and PNG evidence for one metric result."""
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluations = evaluate_gates(metric, gates)
    metric_json = output_directory / f"{metric.metric_family}.metric.json"
    quicklook_png = output_directory / f"{metric.metric_family}.quicklook.png"
    payload: dict[
        str,
        JsonScalar
        | Mapping[str, JsonScalar]
        | Sequence[Mapping[str, JsonScalar]]
        | Sequence[str],
    ]
    payload = {
        "metric": metric.to_json_dict(),
        "gates": [asdict(evaluation) for evaluation in evaluations],
        "versions": context.versions,
        "input_hashes": context.input_hashes,
        "command_log": list(context.command_log),
    }
    metric_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(6.0, 3.0), layout="constrained")
    fields = [evaluation.field for evaluation in evaluations]
    observed = [evaluation.observed for evaluation in evaluations]
    thresholds = [evaluation.threshold for evaluation in evaluations]
    positions = list(range(len(evaluations)))
    axis.bar(positions, observed, label="observed", color="#2878B5")
    axis.scatter(positions, thresholds, label="manifest gate", color="#C82423")
    axis.set_xticks(positions, fields, rotation=20, ha="right")
    axis.set_ylabel("metric value")
    axis.set_title(metric.metric_family)
    axis.legend()
    figure.savefig(
        quicklook_png,
        dpi=120,
        metadata={"Software": "FanInSAR reference harness"},
    )
    plt.close(figure)
    return EvidenceArtifacts(metric_json=metric_json, quicklook_png=quicklook_png)
