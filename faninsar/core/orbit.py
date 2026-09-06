"""Orbit metadata value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from datetime import datetime

Vector3: TypeAlias = tuple[float, float, float]


def _reject_invalid_state(message: str) -> None:
    """Raise the shared processing error without a core import dependency."""
    from faninsar.processing.errors import reject_invalid_state

    reject_invalid_state(message)


@dataclass(frozen=True, slots=True)
class OrbitStateVector:
    """Time-tagged Cartesian orbit position and velocity."""

    time: datetime
    position_m: Vector3
    velocity_m_s: Vector3

    def __post_init__(self) -> None:
        """Validate that the orbit epoch is timezone-aware."""
        if self.time.tzinfo is None:
            _reject_invalid_state("orbit state-vector time must include a timezone")


@dataclass(frozen=True, slots=True)
class OrbitMetadata:
    """Ordered orbit state vectors and their reference frame."""

    reference_frame: str
    source: str
    vectors: tuple[OrbitStateVector, ...]

    def __post_init__(self) -> None:
        """Validate required identity and strictly ordered orbit epochs."""
        if not self.reference_frame or not self.source or not self.vectors:
            _reject_invalid_state("orbit frame, source, and vectors are required")
        times = tuple(vector.time for vector in self.vectors)
        if times != tuple(sorted(times)) or len(times) != len(set(times)):
            _reject_invalid_state("orbit state-vector times must be unique and ordered")


__all__ = ["OrbitMetadata", "OrbitStateVector"]
