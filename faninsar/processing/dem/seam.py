# ruff: noqa: E501, EM101, TRY003, TC003
"""Antimeridian-safe source query windows and logical sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = setup_logger(__name__)


class ExplicitAntimeridianError(ValueError):
    """Raised when caller-explicit source geometry crosses +/-180 degrees."""


def crosses_antimeridian(bounds: tuple[float, float, float, float]) -> bool:
    """Return whether longitude bounds use the wrapped (west > east) form."""
    west, _south, east, _north = bounds
    return west > east


def plan_query_windows(
    bounds: tuple[float, float, float, float], *, explicit: bool = False
) -> tuple[tuple[float, float, float, float], ...]:
    """Plan zero/one/two deterministic source windows.

    Automatic wrapped bounds become two windows.  Explicit wrapped source
    requests fail before provider discovery or network access.
    """
    west, south, east, north = (float(value) for value in bounds)
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValueError("longitude must be within [-180, 180]")
    if south > north:
        raise ValueError("latitude bounds must be ordered")
    if west > east:
        if explicit:
            raise ExplicitAntimeridianError(
                "explicit source bounds crossing the antimeridian are unsupported"
            )
        return ((west, south, 180.0, north), (-180.0, south, east, north))
    return ((west, south, east, north),)


def unwrap_longitude(longitude: float, center: float) -> float:
    """Map longitude to the continuous interval nearest a target centre."""
    value = float(longitude)
    midpoint = float(center)
    while value - midpoint > 180.0:
        value -= 360.0
    while value - midpoint < -180.0:
        value += 360.0
    return value


def canonical_item_ids(items: Sequence[object]) -> tuple[object, ...]:
    """Sort and deduplicate managed source items by stable identity."""
    unique: dict[str, object] = {}
    for item in items:
        identity = str(getattr(item, "identity", getattr(item, "id", item)))
        unique.setdefault(identity, item)
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True, slots=True)
class SeamAwareSourceSampler:
    """Logical source sampler that evaluates each target point exactly once."""

    source_sampler: Callable[[float, float], float]
    target_center_longitude: float

    def sample(self, longitudes: Sequence[float], latitudes: Sequence[float]) -> list[float]:
        """Sample points after target-centred longitude unwrapping."""
        if len(longitudes) != len(latitudes):
            raise ValueError("longitude and latitude sequences must have equal length")
        return [
            self.source_sampler(unwrap_longitude(lon, self.target_center_longitude), float(lat))
            for lon, lat in zip(longitudes, latitudes, strict=True)
        ]


__all__ = [
    "ExplicitAntimeridianError",
    "SeamAwareSourceSampler",
    "canonical_item_ids",
    "crosses_antimeridian",
    "plan_query_windows",
    "unwrap_longitude",
]
