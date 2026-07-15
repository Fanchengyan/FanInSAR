"""Zero-Doppler residual and geometric baseline utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

from .orbit import OrbitInterpolator, OrbitState

if TYPE_CHECKING:
    from datetime import datetime

    from faninsar.processing.contracts import OrbitMetadata

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineComponents:
    """Parallel and perpendicular baseline components in metres."""

    parallel_m: float
    perpendicular_m: float
    magnitude_m: float


def zero_doppler_residual_hz(
    state: OrbitState,
    target_ecef_m: tuple[float, float, float],
    wavelength_m: float,
) -> float:
    """Return the Doppler residual for a platform state and ground target.

    Parameters
    ----------
    state : OrbitState
        Platform position and velocity in ECEF.
    target_ecef_m : tuple of float
        Target ECEF coordinates in metres.
    wavelength_m : float
        Radar wavelength in metres.

    Returns
    -------
    float
        Doppler residual in hertz.

    """
    if wavelength_m <= 0:
        message = "wavelength must be positive"
        logger.error(message)
        raise ValueError(message)
    position = np.asarray(state.position_m, dtype=np.float64)
    velocity = np.asarray(state.velocity_m_s, dtype=np.float64)
    target = np.asarray(target_ecef_m, dtype=np.float64)
    look = target - position
    range_m = float(np.linalg.norm(look))
    if range_m <= 0:
        message = "target coincides with platform position"
        logger.error(message)
        raise ValueError(message)
    unit_look = look / range_m
    radial_velocity = float(np.dot(velocity, unit_look))
    return 2.0 * radial_velocity / wavelength_m


def geometric_baseline(
    reference_orbit: OrbitMetadata,
    secondary_orbit: OrbitMetadata,
    *,
    time: datetime,
    look_unit_ecef: tuple[float, float, float],
) -> BaselineComponents:
    """Compute parallel/perpendicular baseline at a common epoch."""
    reference = OrbitInterpolator.from_orbit(reference_orbit).evaluate(time)
    secondary = OrbitInterpolator.from_orbit(secondary_orbit).evaluate(time)
    baseline = np.asarray(secondary.position_m, dtype=np.float64) - np.asarray(
        reference.position_m,
        dtype=np.float64,
    )
    look = np.asarray(look_unit_ecef, dtype=np.float64)
    look_norm = float(np.linalg.norm(look))
    if look_norm <= 0:
        message = "look unit vector must be non-zero"
        logger.error(message)
        raise ValueError(message)
    look = look / look_norm
    parallel = float(np.dot(baseline, look))
    perpendicular_vec = baseline - parallel * look
    perpendicular = float(np.linalg.norm(perpendicular_vec))
    magnitude = float(np.linalg.norm(baseline))
    return BaselineComponents(
        parallel_m=parallel,
        perpendicular_m=perpendicular,
        magnitude_m=magnitude,
    )
