"""Orbit state-vector interpolation and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from faninsar.logging import setup_logger
from faninsar.processing.errors import ProcessingContractError

if TYPE_CHECKING:
    from datetime import datetime

    from faninsar.core.orbit import OrbitMetadata

logger = setup_logger(__name__)


class OrbitInterpolationError(ProcessingContractError):
    """Raised when orbit interpolation cannot produce a valid state."""


@dataclass(frozen=True, slots=True)
class OrbitState:
    """Interpolated position and velocity at one epoch."""

    time: datetime
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class OrbitInterpolator:
    """Velocity-constrained interpolator over ECEF orbit state vectors."""

    times_s: np.ndarray
    trajectory_splines: tuple[
        CubicHermiteSpline,
        CubicHermiteSpline,
        CubicHermiteSpline,
    ]
    epoch: datetime
    t_min_s: float
    t_max_s: float

    @classmethod
    def from_orbit(cls, orbit: OrbitMetadata) -> OrbitInterpolator:
        """Build an interpolator from typed orbit metadata."""
        if len(orbit.vectors) < 2:
            message = "orbit interpolation requires at least two state vectors"
            logger.error(message)
            raise OrbitInterpolationError(message)
        epoch = orbit.vectors[0].time
        times = np.array(
            [(vector.time - epoch).total_seconds() for vector in orbit.vectors],
            dtype=np.float64,
        )
        if np.any(np.diff(times) <= 0):
            message = "orbit state-vector times must be strictly increasing"
            logger.error(message)
            raise OrbitInterpolationError(message)
        positions = np.array(
            [vector.position_m for vector in orbit.vectors],
            dtype=np.float64,
        )
        velocities = np.array(
            [vector.velocity_m_s for vector in orbit.vectors],
            dtype=np.float64,
        )
        trajectory_splines = tuple(
            CubicHermiteSpline(
                times,
                positions[:, axis],
                velocities[:, axis],
            )
            for axis in range(3)
        )
        return cls(
            times_s=times,
            trajectory_splines=trajectory_splines,  # type: ignore[arg-type]
            epoch=epoch,
            t_min_s=float(times[0]),
            t_max_s=float(times[-1]),
        )

    def evaluate(self, time: datetime) -> OrbitState:
        """Interpolate position and velocity at an absolute epoch.

        Parameters
        ----------
        time : datetime
            Timezone-aware evaluation epoch.

        Returns
        -------
        OrbitState
            Interpolated ECEF state.

        Raises
        ------
        OrbitInterpolationError
            If the epoch is outside the orbit coverage.

        """
        if time.tzinfo is None:
            message = "orbit evaluation time must be timezone-aware"
            logger.error(message)
            raise OrbitInterpolationError(message)
        t_s = (time - self.epoch).total_seconds()
        if t_s < self.t_min_s or t_s > self.t_max_s:
            message = (
                f"orbit time {time.isoformat()} outside coverage "
                f"[{self.t_min_s}, {self.t_max_s}] s from epoch"
            )
            logger.error(message)
            raise OrbitInterpolationError(message)
        position = tuple(float(spline(t_s)) for spline in self.trajectory_splines)
        velocity = tuple(float(spline(t_s, 1)) for spline in self.trajectory_splines)
        return OrbitState(time=time, position_m=position, velocity_m_s=velocity)  # type: ignore[arg-type]

    def evaluate_array(self, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate position and velocity arrays.

        Parameters
        ----------
        times_s : numpy.ndarray
            Seconds from the orbit epoch.

        Returns
        -------
        tuple of numpy.ndarray
            Positions with shape ``(..., 3)`` and velocities with shape
            ``(..., 3)`` in metres.

        Raises
        ------
        OrbitInterpolationError
            If any time is outside the orbit coverage.

        """
        times_s = np.asarray(times_s, dtype=np.float64)
        if np.any(times_s < self.t_min_s) or np.any(times_s > self.t_max_s):
            message = (
                f"orbit times outside coverage "
                f"[{self.t_min_s}, {self.t_max_s}] s from epoch"
            )
            logger.error(message)
            raise OrbitInterpolationError(message)
        positions = np.stack(
            [spline(times_s) for spline in self.trajectory_splines],
            axis=-1,
        )
        velocities = np.stack(
            [spline(times_s, 1) for spline in self.trajectory_splines],
            axis=-1,
        )
        return positions, velocities

    def evaluate_array_with_acceleration(
        self,
        times_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate position, velocity, and acceleration arrays.

        Parameters
        ----------
        times_s : numpy.ndarray
            Seconds from the orbit epoch.

        Returns
        -------
        tuple of numpy.ndarray
            Position, velocity, and acceleration arrays with shape
            ``(..., 3)`` in metres, metres per second, and metres per second
            squared.

        Raises
        ------
        OrbitInterpolationError
            If any time is outside the orbit coverage.

        """
        times_s = np.asarray(times_s, dtype=np.float64)
        if np.any(times_s < self.t_min_s) or np.any(times_s > self.t_max_s):
            message = (
                f"orbit times outside coverage "
                f"[{self.t_min_s}, {self.t_max_s}] s from epoch"
            )
            logger.error(message)
            raise OrbitInterpolationError(message)
        positions = np.stack(
            [spline(times_s) for spline in self.trajectory_splines],
            axis=-1,
        )
        velocities = np.stack(
            [spline(times_s, 1) for spline in self.trajectory_splines],
            axis=-1,
        )
        accelerations = np.stack(
            [spline(times_s, 2) for spline in self.trajectory_splines],
            axis=-1,
        )
        return positions, velocities, accelerations


def interpolate_orbit(orbit: OrbitMetadata, time: datetime) -> OrbitState:
    """Interpolate one orbit state at ``time``."""
    return OrbitInterpolator.from_orbit(orbit).evaluate(time)
