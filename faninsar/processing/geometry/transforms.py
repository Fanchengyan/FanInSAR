"""Radar geometry value types for rdr2geo and geo2rdr.

The NumPy CPU Newton solvers formerly in this module were deleted under
PROPOSAL-0031. Production callers use
:mod:`faninsar.processing.geometry.prepare_production`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.orbit import OrbitInterpolator

if TYPE_CHECKING:
    from faninsar.core.orbit import OrbitMetadata
    from faninsar.processing.coordinates import RadarGrid

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Dense transform arrays plus convergence diagnostics."""

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    height_m: np.ndarray
    range_index: np.ndarray
    azimuth_index: np.ndarray
    converged: np.ndarray
    residual_range_m: np.ndarray
    residual_doppler_hz: np.ndarray


@dataclass(frozen=True, slots=True)
class RadarGeometryModel:
    """Radar timing/geometry model used by rdr2geo and geo2rdr."""

    orbit: OrbitInterpolator
    sensing_start: object
    azimuth_time_interval_s: float
    starting_slant_range_m: float
    range_spacing_m: float
    wavelength_m: float
    look_direction: str

    @classmethod
    def from_radar_grid(
        cls,
        grid: RadarGrid,
        orbit: OrbitMetadata,
    ) -> RadarGeometryModel:
        """Build a geometry model from a radar grid and orbit metadata."""
        return cls(
            orbit=OrbitInterpolator.from_orbit(orbit),
            sensing_start=grid.sensing_start,
            azimuth_time_interval_s=grid.azimuth_time_interval_s,
            starting_slant_range_m=grid.starting_slant_range_m,
            range_spacing_m=grid.range_spacing_m,
            wavelength_m=grid.wavelength_m,
            look_direction=grid.look_direction,
        )

    def azimuth_time(self, azimuth_index: float) -> datetime:
        """Return absolute sensing time for an azimuth sample index."""
        start = self.sensing_start
        if not isinstance(start, datetime):
            message = "sensing_start must be a datetime"
            logger.error(message)
            raise TypeError(message)
        return start + timedelta(
            seconds=float(azimuth_index) * self.azimuth_time_interval_s
        )

    def azimuth_time_seconds(self, azimuth_index: np.ndarray) -> np.ndarray:
        """Return seconds from orbit epoch for an array of azimuth indices."""
        start = self.sensing_start
        if not isinstance(start, datetime):
            message = "sensing_start must be a datetime"
            logger.error(message)
            raise TypeError(message)
        sensing_offset_s = (start - self.orbit.epoch).total_seconds()
        return (
            sensing_offset_s
            + np.asarray(azimuth_index, dtype=np.float64) * self.azimuth_time_interval_s
        )

    def slant_range_m(self, range_index: float) -> float:
        """Return two-way-consistent one-way slant range for a range index."""
        return self.starting_slant_range_m + float(range_index) * self.range_spacing_m
