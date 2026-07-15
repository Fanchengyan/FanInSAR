"""Immutable result types for scientific reference metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping

JsonScalar: TypeAlias = str | int | float | bool | None
MetricFamily: TypeAlias = Literal[
    "geometry",
    "phase",
    "coherence",
    "offset",
    "closure",
    "geolocation",
    "displacement",
    "performance",
]


class MetricInputError(ValueError):
    """Describe an invalid metric input at the comparison boundary."""

    def __init__(self, detail: str) -> None:
        """Initialize the error with actionable validation detail."""
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        """Return the actionable boundary-validation detail."""
        return self.detail


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Base type for JSON-serializable scientific metric results."""

    metric_family: MetricFamily

    def to_json_dict(self) -> Mapping[str, JsonScalar]:
        """Return scalar fields suitable for a metric JSON document."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GeometryMetrics(MetricResult):
    """Coordinate round-trip residual summary in sample units."""

    valid_count: int
    p95_sample_error: float
    p99_sample_error: float
    maximum_sample_error: float


@dataclass(frozen=True, slots=True)
class PhaseMetrics(MetricResult):
    """Wrapped and linear phase-error summary in radians."""

    valid_count: int
    wrapped: bool
    circular_rmse_rad: float
    rmse_rad: float
    mean_bias_rad: float


@dataclass(frozen=True, slots=True)
class CoherenceMetrics(MetricResult):
    """Coherence error and loss summary."""

    valid_count: int
    mean_absolute_error: float
    maximum_absolute_error: float
    median_loss: float


@dataclass(frozen=True, slots=True)
class OffsetMetrics(MetricResult):
    """Range and azimuth offset error summary in pixels."""

    valid_count: int
    range_rmse_pixel: float
    azimuth_rmse_pixel: float
    coverage_fraction: float


@dataclass(frozen=True, slots=True)
class ClosureMetrics(MetricResult):
    """Interferometric loop-closure residual summary."""

    loop_count: int
    valid_count: int
    circular_rmse_rad: float
    maximum_absolute_rad: float


@dataclass(frozen=True, slots=True)
class GeolocationMetrics(MetricResult):
    """Horizontal and vertical geolocation error summary."""

    valid_count: int
    horizontal_rmse_m: float
    vertical_rmse_m: float
    three_dimensional_p95_m: float


@dataclass(frozen=True, slots=True)
class DisplacementMetrics(MetricResult):
    """Displacement and fitted velocity error summary."""

    valid_count: int
    displacement_rmse_m: float
    displacement_bias_m: float
    velocity_bias_m_per_year: float
    uncertainty_95_coverage_fraction: float | None


@dataclass(frozen=True, slots=True)
class PerformanceMetrics(MetricResult):
    """Repeat-run runtime, memory, and throughput summary."""

    run_count: int
    runtime_median_seconds: float
    runtime_p95_seconds: float
    peak_memory_bytes: int
    throughput_pixels_per_second: float
