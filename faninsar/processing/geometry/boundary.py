"""Canonical convergence-boundary evaluation for geometry v2."""

# Boundary errors are deliberately raised at the narrow public seam; each is
# validated before any callback or pointer-like array access.
# ruff: noqa: TRY003, EM101

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.v2 import (
    GeometryValidationError,
    Operation,
    TransformResultV2,
)

logger = setup_logger(__name__)
_BOUNDARY_ULPS = 32.0 * np.finfo(np.float64).eps


@dataclass(frozen=True, slots=True)
class BoundaryDecision:
    """Decision owned by the canonical evaluator for one current attempt."""

    converged: bool
    boundary_rechecked: bool
    decision_residual: float
    normalized_q: float


def _op(operation: Operation | str) -> Operation:
    """Normalize an operation tag."""
    try:
        return operation if isinstance(operation, Operation) else Operation(operation)
    except (TypeError, ValueError) as error:
        message = "operation must be 'geo2rdr' or 'rdr2geo'"
        logger.exception(message)
        raise GeometryValidationError(message) from error


def _snapshot(value: np.ndarray, name: str) -> np.ndarray:
    """Create a private, read-only float64 snapshot for the callback."""
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64):
        message = f"{name} must be a float64 NumPy array"
        logger.error(message)
        raise GeometryValidationError(message)
    snapshot = np.array(value, dtype=np.float64, copy=True, order="C")
    snapshot.setflags(write=False)
    return snapshot


def _original_metrics(
    operation: Operation,
    residuals: float | Sequence[float],
    *,
    range_tolerance_m: float | None,
    doppler_tolerance_hz: float | None,
    slant_range_tolerance_m: float | None,
) -> tuple[float, float]:
    """Return the original decision residual and normalized metric."""
    if operation is Operation.GEO2RDR:
        if range_tolerance_m is None or doppler_tolerance_hz is None:
            raise GeometryValidationError(
                "geo2rdr requires range and Doppler tolerances"
            )
        try:
            range_residual, doppler_residual = residuals  # type: ignore[misc]
        except (TypeError, ValueError) as error:
            raise GeometryValidationError(
                "geo2rdr residuals must be (range, Doppler)"
            ) from error
        decision = max(
            abs(float(range_residual)) / range_tolerance_m,
            abs(float(doppler_residual)) / doppler_tolerance_hz,
        )
        return decision, decision
    if slant_range_tolerance_m is None:
        raise GeometryValidationError("rdr2geo requires slant-range tolerance")
    decision = float(residuals)  # type: ignore[arg-type]
    return decision, abs(decision) / slant_range_tolerance_m


def evaluate_canonical_boundary(
    operation: Operation | str,
    current_attempt: int,
    final_coordinates: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    residual_callback: Callable[..., float | Sequence[float]],
    *,
    decision_residual: float | Sequence[float],
    range_tolerance_m: float | None = None,
    doppler_tolerance_hz: float | None = None,
    slant_range_tolerance_m: float | None = None,
) -> BoundaryDecision:
    """Apply the strict canonical boundary rule to the current attempt.

    The callback receives private read-only snapshots, never caller-owned
    arrays.  It is called only when the original normalized metric lies within
    the inclusive 32-binary64-ULP band around one.  Earlier attempts are not
    represented or rescanned by this API.
    """
    op = _op(operation)
    if (
        isinstance(current_attempt, (bool, np.bool_))
        or not isinstance(current_attempt, (int, np.integer))
        or current_attempt < 1
    ):
        raise GeometryValidationError("current_attempt must be at least one")
    original, normalized_q = _original_metrics(
        op,
        decision_residual,
        range_tolerance_m=range_tolerance_m,
        doppler_tolerance_hz=doppler_tolerance_hz,
        slant_range_tolerance_m=slant_range_tolerance_m,
    )
    if not np.isfinite(normalized_q):
        return BoundaryDecision(False, False, float("nan"), float("nan"))
    if abs(normalized_q - 1.0) > _BOUNDARY_ULPS:
        return BoundaryDecision(normalized_q < 1.0, False, original, normalized_q)
    if isinstance(final_coordinates, Mapping):
        values = tuple(
            _snapshot(value, name) for name, value in final_coordinates.items()
        )
    else:
        values = tuple(
            _snapshot(value, f"coordinate[{index}]")
            for index, value in enumerate(final_coordinates)
        )
    if any(not np.all(np.isfinite(value)) for value in values):
        return BoundaryDecision(False, False, float("nan"), float("nan"))
    recomputed = residual_callback(*values)
    if op is Operation.GEO2RDR:
        if range_tolerance_m is None or doppler_tolerance_hz is None:
            raise GeometryValidationError(
                "geo2rdr requires range and Doppler tolerances"
            )
        try:
            range_residual, doppler_residual = recomputed  # type: ignore[misc]
        except (TypeError, ValueError) as error:
            raise GeometryValidationError(
                "callback must return (range, Doppler)"
            ) from error
        final_decision = max(
            abs(float(range_residual)) / range_tolerance_m,
            abs(float(doppler_residual)) / doppler_tolerance_hz,
        )
        return BoundaryDecision(
            final_decision < 1.0, True, final_decision, final_decision
        )
    if slant_range_tolerance_m is None:
        raise GeometryValidationError("rdr2geo requires slant-range tolerance")
    final_decision = float(recomputed)
    return BoundaryDecision(
        abs(final_decision) < slant_range_tolerance_m,
        True,
        final_decision,
        abs(final_decision) / slant_range_tolerance_m,
    )


def normalize_result_boundary(
    result: TransformResultV2,
    decisions: BoundaryDecision | Sequence[BoundaryDecision],
    *,
    invalid_mask: np.ndarray | None = None,
) -> TransformResultV2:
    """Publish canonical status fields without disturbing invalid lanes."""
    values: list[BoundaryDecision] = (
        [decisions] if isinstance(decisions, BoundaryDecision) else list(decisions)
    )
    shape = result.latitude_deg.shape
    if len(values) != (
        1 if result.latitude_deg.ndim == 0 else result.latitude_deg.size
    ):
        raise GeometryValidationError("boundary decisions must match result lanes")
    flat = {name: getattr(result, name).copy().reshape(-1) for name in result.fields}
    mask = (
        np.zeros(flat["converged"].shape, dtype=bool)
        if invalid_mask is None
        else np.asarray(invalid_mask, dtype=bool).reshape(-1)
    )
    if mask.shape != flat["converged"].shape:
        raise GeometryValidationError("invalid_mask must match result shape")
    for index, decision in enumerate(values):
        if mask[index]:
            continue
        flat["converged"][index] = decision.converged
        flat["boundary_rechecked"][index] = decision.boundary_rechecked
        flat["decision_residual"][index] = decision.decision_residual
    if np.any(mask):
        float_fields = {
            "latitude_deg",
            "longitude_deg",
            "height_m",
            "range_index",
            "azimuth_index",
            "decision_residual",
            "final_residual",
            "tolerance",
            "residual_range_m",
            "residual_doppler_hz",
        }
        for name in float_fields:
            flat[name][mask] = np.nan
        flat["iterations"][mask] = -1
        for name in ("converged", "max_iter_exhausted", "boundary_rechecked"):
            flat[name][mask] = False
    return TransformResultV2(
        **{name: array.reshape(shape) for name, array in flat.items()}
    )


__all__ = [
    "BoundaryDecision",
    "evaluate_canonical_boundary",
    "normalize_result_boundary",
]
