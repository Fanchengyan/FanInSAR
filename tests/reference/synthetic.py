"""Deterministic analytic arrays for scientific boundary comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from faninsar.logging import setup_logger

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
logger = setup_logger(__name__)


class SyntheticInputError(ValueError):
    """Describe an invalid analytic-case request."""

    def __init__(self, detail: str) -> None:
        """Initialize the error with actionable validation detail."""
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        """Return the actionable generator-validation detail."""
        return self.detail


def _fail(detail: str) -> None:
    logger.error(detail)
    raise SyntheticInputError(detail)


@dataclass(frozen=True, slots=True)
class PhaseCase:
    """Wrapped phase, unwrapped phase, and coherence comparison arrays."""

    wrapped_reference: FloatArray
    wrapped_candidate: FloatArray
    unwrapped_reference: FloatArray
    unwrapped_candidate: FloatArray
    coherence_reference: FloatArray
    coherence_candidate: FloatArray


@dataclass(frozen=True, slots=True)
class OffsetCase:
    """Reference and shifted range-azimuth vector fields."""

    reference: FloatArray
    candidate: FloatArray


@dataclass(frozen=True, slots=True)
class ClosureCase:
    """Pair phases and signed one-based loop definitions."""

    pair_phase: FloatArray
    loops: IntArray


@dataclass(frozen=True, slots=True)
class GeolocationCase:
    """Reference and perturbed Cartesian geolocation fields."""

    reference: FloatArray
    candidate: FloatArray


@dataclass(frozen=True, slots=True)
class DisplacementCase:
    """Reference and biased displacement time series."""

    reference: FloatArray
    candidate: FloatArray
    elapsed_years: FloatArray


def _immutable(array: FloatArray) -> FloatArray:
    array.setflags(write=False)
    return array


def ramp(
    shape: tuple[int, int],
    *,
    range_slope: float,
    azimuth_slope: float,
    intercept: float = 0.0,
) -> FloatArray:
    """Create a deterministic two-dimensional linear ramp."""
    if len(shape) != 2 or min(shape) <= 0:
        _fail("ramp shape must contain two positive dimensions")
    azimuth, range_ = np.indices(shape, dtype=np.float64)
    return _immutable(intercept + range_slope * range_ + azimuth_slope * azimuth)


def phase_case(
    shape: tuple[int, int],
    *,
    phase_bias: float = 0.02,
    coherence_bias: float = -0.01,
) -> PhaseCase:
    """Create known wrapped, unwrapped, and coherence residuals."""
    unwrapped = ramp(shape, range_slope=0.4, azimuth_slope=-0.15)
    wrapped = np.angle(np.exp(1j * unwrapped))
    coherence = np.full(shape, 0.8, dtype=np.float64)
    return PhaseCase(
        wrapped_reference=_immutable(wrapped),
        wrapped_candidate=_immutable(np.angle(np.exp(1j * (unwrapped + phase_bias)))),
        unwrapped_reference=unwrapped,
        unwrapped_candidate=_immutable(unwrapped + phase_bias),
        coherence_reference=_immutable(coherence),
        coherence_candidate=_immutable(coherence + coherence_bias),
    )


def offset_case(
    shape: tuple[int, int],
    *,
    range_shift: float,
    azimuth_shift: float,
) -> OffsetCase:
    """Create a constant subpixel range-azimuth shift field."""
    reference = np.zeros((*shape, 2), dtype=np.float64)
    candidate = reference.copy()
    candidate[..., 0] = range_shift
    candidate[..., 1] = azimuth_shift
    return OffsetCase(_immutable(reference), _immutable(candidate))


def closure_case(
    shape: tuple[int, int],
    *,
    closure_bias: float,
) -> ClosureCase:
    """Create one three-pair loop with a known closure residual."""
    first = ramp(shape, range_slope=0.1, azimuth_slope=0.05)
    second = ramp(shape, range_slope=-0.03, azimuth_slope=0.02)
    third = first + second - closure_bias
    phases = np.stack((first, second, third))
    loops = np.array([[1, 2, -3]], dtype=np.int64)
    loops.setflags(write=False)
    return ClosureCase(_immutable(phases), loops)


def geolocation_case(
    shape: tuple[int, int],
    *,
    horizontal_error_m: float,
    vertical_error_m: float,
) -> GeolocationCase:
    """Create Cartesian points with known horizontal and vertical errors."""
    azimuth, range_ = np.indices(shape, dtype=np.float64)
    reference = np.stack((range_ * 10.0, azimuth * 10.0, azimuth + range_), axis=-1)
    candidate = reference.copy()
    candidate[..., 0] += horizontal_error_m
    candidate[..., 2] += vertical_error_m
    return GeolocationCase(_immutable(reference), _immutable(candidate))


def displacement_case(
    acquisition_count: int,
    shape: tuple[int, int],
    *,
    velocity_m_per_year: float,
    candidate_bias_m: float,
) -> DisplacementCase:
    """Create a linear displacement series with a constant candidate bias."""
    if acquisition_count < 2:
        _fail("a displacement case requires at least two acquisitions")
    elapsed_years = np.linspace(0.0, 1.0, acquisition_count, dtype=np.float64)
    reference = elapsed_years[:, None, None] * velocity_m_per_year
    reference = np.broadcast_to(reference, (acquisition_count, *shape)).copy()
    return DisplacementCase(
        reference=_immutable(reference),
        candidate=_immutable(reference + candidate_bias_m),
        elapsed_years=_immutable(elapsed_years),
    )
