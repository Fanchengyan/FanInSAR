"""Typed comparison metrics for frozen scientific reference products."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from faninsar.logging import setup_logger
from tests.reference.metric_types import (
    ClosureMetrics,
    CoherenceMetrics,
    DisplacementMetrics,
    GeolocationMetrics,
    GeometryMetrics,
    MetricInputError,
    OffsetMetrics,
    PerformanceMetrics,
    PhaseMetrics,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = setup_logger(__name__)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _fail(detail: str) -> None:
    logger.error(detail)
    raise MetricInputError(detail)


def _paired_finite(
    reference: FloatArray,
    candidate: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if reference.shape != candidate.shape:
        _fail(
            f"reference and candidate shape mismatch: "
            f"{reference.shape} != {candidate.shape}"
        )
    if reference.size == 0:
        _fail("metric arrays must not be empty")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        _fail("metric arrays must contain only finite values")
    return reference, candidate


def _rmse(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def geometry_metrics(reference: FloatArray, candidate: FloatArray) -> GeometryMetrics:
    """Summarize coordinate-vector residual magnitudes."""
    reference, candidate = _paired_finite(reference, candidate)
    if reference.ndim < 2 or reference.shape[-1] != 2:
        _fail("geometry arrays must have a last axis of size 2")
    residual = np.linalg.norm(candidate - reference, axis=-1)
    return GeometryMetrics(
        metric_family="geometry",
        valid_count=residual.size,
        p95_sample_error=float(np.percentile(residual, 95)),
        p99_sample_error=float(np.percentile(residual, 99)),
        maximum_sample_error=float(np.max(residual)),
    )


def phase_metrics(
    reference: FloatArray,
    candidate: FloatArray,
    *,
    wrapped: bool = True,
) -> PhaseMetrics:
    """Summarize wrapped or unwrapped phase residuals."""
    reference, candidate = _paired_finite(reference, candidate)
    linear_residual = candidate - reference
    circular_residual = np.angle(np.exp(1j * linear_residual))
    selected_residual = circular_residual if wrapped else linear_residual
    return PhaseMetrics(
        metric_family="phase",
        valid_count=selected_residual.size,
        wrapped=wrapped,
        circular_rmse_rad=_rmse(circular_residual),
        rmse_rad=_rmse(selected_residual),
        mean_bias_rad=float(np.mean(selected_residual)),
    )


def coherence_metrics(
    reference: FloatArray,
    candidate: FloatArray,
) -> CoherenceMetrics:
    """Summarize bounded coherence errors and median loss."""
    reference, candidate = _paired_finite(reference, candidate)
    if np.any((reference < 0) | (reference > 1) | (candidate < 0) | (candidate > 1)):
        _fail("coherence values must lie in [0, 1]")
    residual = candidate - reference
    return CoherenceMetrics(
        metric_family="coherence",
        valid_count=residual.size,
        mean_absolute_error=float(np.mean(np.abs(residual))),
        maximum_absolute_error=float(np.max(np.abs(residual))),
        median_loss=float(np.median(reference - candidate)),
    )


def offset_metrics(reference: FloatArray, candidate: FloatArray) -> OffsetMetrics:
    """Summarize range and azimuth offset-vector residuals."""
    if reference.shape != candidate.shape:
        _fail(
            f"reference and candidate shape mismatch: "
            f"{reference.shape} != {candidate.shape}"
        )
    if reference.ndim < 2 or reference.shape[-1] != 2:
        _fail("offset arrays must have a last axis of size 2")
    if reference.size == 0 or not np.all(np.isfinite(reference)):
        _fail("offset reference must be non-empty and finite")
    finite_components = np.isfinite(candidate)
    if np.any(finite_components[..., 0] != finite_components[..., 1]):
        _fail("offset candidate components must share the same validity mask")
    valid = np.all(finite_components, axis=-1)
    if not np.any(valid):
        _fail("offset candidate must contain at least one valid vector")
    residual = candidate[valid] - reference[valid]
    return OffsetMetrics(
        metric_family="offset",
        valid_count=residual[:, 0].size,
        range_rmse_pixel=_rmse(residual[..., 0]),
        azimuth_rmse_pixel=_rmse(residual[..., 1]),
        coverage_fraction=float(np.mean(valid)),
    )


def closure_metrics(pair_phase: FloatArray, loops: IntArray) -> ClosureMetrics:
    """Summarize signed one-based pair loops as wrapped closure residuals."""
    if pair_phase.ndim < 2 or pair_phase.shape[0] == 0:
        _fail("pair phase must have a non-empty leading pair axis")
    if loops.ndim != 2 or loops.shape[1] < 3 or loops.size == 0:
        _fail("loops must be a non-empty two-dimensional signed index array")
    if not np.all(np.isfinite(pair_phase)):
        _fail("pair phase must contain only finite values")
    if np.any(loops == 0) or np.any(np.abs(loops) > pair_phase.shape[0]):
        _fail("loop pair indices must be signed one-based indices within pair phase")
    closure = np.zeros((loops.shape[0], *pair_phase.shape[1:]), dtype=np.float64)
    for column in range(loops.shape[1]):
        signed_index = loops[:, column]
        sign = np.sign(signed_index).reshape((-1,) + (1,) * (pair_phase.ndim - 1))
        closure += sign * pair_phase[np.abs(signed_index) - 1]
    wrapped_closure = np.angle(np.exp(1j * closure))
    return ClosureMetrics(
        metric_family="closure",
        loop_count=loops.shape[0],
        valid_count=wrapped_closure.size,
        circular_rmse_rad=_rmse(wrapped_closure),
        maximum_absolute_rad=float(np.max(np.abs(wrapped_closure))),
    )


def geolocation_metrics(
    reference: FloatArray,
    candidate: FloatArray,
) -> GeolocationMetrics:
    """Summarize Cartesian horizontal and vertical location residuals."""
    reference, candidate = _paired_finite(reference, candidate)
    if reference.ndim < 2 or reference.shape[-1] != 3:
        _fail("geolocation arrays must have a last axis of size 3")
    residual = candidate - reference
    horizontal = np.linalg.norm(residual[..., :2], axis=-1)
    vertical = residual[..., 2]
    distance = np.linalg.norm(residual, axis=-1)
    return GeolocationMetrics(
        metric_family="geolocation",
        valid_count=horizontal.size,
        horizontal_rmse_m=_rmse(horizontal),
        vertical_rmse_m=_rmse(vertical),
        three_dimensional_p95_m=float(np.percentile(distance, 95)),
    )


def displacement_metrics(
    reference: FloatArray,
    candidate: FloatArray,
    elapsed_years: FloatArray,
    uncertainty_95_half_width: FloatArray | None = None,
) -> DisplacementMetrics:
    """Summarize time-series displacement and linear velocity errors."""
    reference, candidate = _paired_finite(reference, candidate)
    if reference.ndim < 2 or reference.shape[0] != elapsed_years.size:
        _fail("displacement time axis must match elapsed years")
    if elapsed_years.size < 2 or not np.all(np.isfinite(elapsed_years)):
        _fail("elapsed years must contain at least two finite samples")
    centered_time = elapsed_years - np.mean(elapsed_years)
    denominator = float(np.dot(centered_time, centered_time))
    if denominator == 0:
        _fail("elapsed years must span more than one unique time")
    residual = candidate - reference
    spatial_axes = tuple(range(1, residual.ndim))
    mean_residual = np.mean(residual, axis=spatial_axes)
    velocity_bias = float(np.dot(centered_time, mean_residual) / denominator)
    coverage: float | None = None
    if uncertainty_95_half_width is not None:
        if uncertainty_95_half_width.shape != residual.shape:
            _fail("uncertainty shape must match displacement arrays")
        if np.any(~np.isfinite(uncertainty_95_half_width)) or np.any(
            uncertainty_95_half_width < 0
        ):
            _fail("uncertainty half widths must be finite and non-negative")
        coverage = float(np.mean(np.abs(residual) <= uncertainty_95_half_width))
    return DisplacementMetrics(
        metric_family="displacement",
        valid_count=residual.size,
        displacement_rmse_m=_rmse(residual),
        displacement_bias_m=float(np.mean(residual)),
        velocity_bias_m_per_year=velocity_bias,
        uncertainty_95_coverage_fraction=coverage,
    )


def performance_metrics(
    runtime_seconds: Sequence[float],
    peak_memory_bytes: Sequence[int],
    processed_pixels: int,
) -> PerformanceMetrics:
    """Summarize repeated runtime and peak-memory observations."""
    runtimes = np.asarray(runtime_seconds, dtype=np.float64)
    memories = np.asarray(peak_memory_bytes, dtype=np.int64)
    if runtimes.size == 0 or memories.size == 0:
        _fail("runtime and memory observations must not be empty")
    if runtimes.shape != memories.shape:
        _fail("runtime and memory observations must have matching shape")
    if not np.all(np.isfinite(runtimes)) or np.any(runtimes <= 0):
        _fail("runtime observations must be finite and positive")
    if np.any(memories < 0) or processed_pixels <= 0:
        _fail("memory observations must be non-negative and processed pixels positive")
    median_runtime = float(np.median(runtimes))
    return PerformanceMetrics(
        metric_family="performance",
        run_count=runtimes.size,
        runtime_median_seconds=median_runtime,
        runtime_p95_seconds=float(np.percentile(runtimes, 95)),
        peak_memory_bytes=int(np.max(memories)),
        throughput_pixels_per_second=processed_pixels / median_runtime,
    )
