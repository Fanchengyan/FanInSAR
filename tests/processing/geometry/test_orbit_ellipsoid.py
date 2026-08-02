"""Unit tests for orbit and ellipsoid reference geometry."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector
from faninsar.processing.geometry import (
    OrbitInterpolationError,
    OrbitInterpolator,
    ecef_to_llh,
    geometric_baseline,
    interpolate_orbit,
    llh_to_ecef,
    local_earth_radius_m,
    zero_doppler_residual_hz,
)


def _orbit() -> OrbitMetadata:
    epoch = datetime(2016, 12, 7, 11, 17, 45, tzinfo=UTC)
    vectors = []
    for index in range(5):
        time = epoch + timedelta(seconds=10 * index)
        # synthetic near-circular path
        angle = 0.001 * index
        vectors.append(
            OrbitStateVector(
                time=time,
                position_m=(
                    7000_000.0 * np.cos(angle),
                    7000_000.0 * np.sin(angle),
                    0.0,
                ),
                velocity_m_s=(
                    -7000_000.0 * 0.0001 * np.sin(angle),
                    7000_000.0 * 0.0001 * np.cos(angle),
                    0.0,
                ),
            )
        )
    return OrbitMetadata(
        reference_frame="ITRF", source="synthetic", vectors=tuple(vectors)
    )


def test_llh_ecef_round_trip_millimetre_level() -> None:
    """Round-trip geodetic coordinates through ECEF at millimetre level."""
    latitude = np.array([0.0, 37.4, -45.0, 89.0])
    longitude = np.array([0.0, 98.8, 120.0, 10.0])
    height = np.array([0.0, 4000.0, 100.0, 50.0])
    x, y, z = llh_to_ecef(latitude, longitude, height)
    lat2, lon2, h2 = ecef_to_llh(x, y, z)
    np.testing.assert_allclose(lat2, latitude, atol=1e-8)
    np.testing.assert_allclose(lon2, longitude, atol=1e-8)
    # height in metres: millimetre-level
    np.testing.assert_allclose(h2, height, atol=1e-3)


def test_orbit_interpolation_matches_nodes_and_rejects_out_of_range() -> None:
    """Reproduce node states and reject evaluation outside coverage."""
    orbit = _orbit()
    node = orbit.vectors[2]
    state = interpolate_orbit(orbit, node.time)
    np.testing.assert_allclose(state.position_m, node.position_m, rtol=0, atol=1e-6)
    np.testing.assert_allclose(state.velocity_m_s, node.velocity_m_s, rtol=0, atol=1e-6)

    with pytest.raises(OrbitInterpolationError, match="outside coverage"):
        interpolate_orbit(orbit, orbit.vectors[0].time - timedelta(seconds=1))


def test_orbit_position_derivative_matches_interpolated_velocity() -> None:
    """Keep interpolated position and velocity on one Hermite trajectory."""
    epoch = datetime(2020, 1, 1, tzinfo=UTC)
    positions = (0.0, 1.0, 0.0, -1.0)
    velocities = (1.0, 0.0, -1.0, 0.0)
    orbit = OrbitMetadata(
        reference_frame="ITRF",
        source="synthetic-hermite",
        vectors=tuple(
            OrbitStateVector(
                time=epoch + timedelta(seconds=index),
                position_m=(position, 0.0, 0.0),
                velocity_m_s=(velocity, 0.0, 0.0),
            )
            for index, (position, velocity) in enumerate(
                zip(positions, velocities, strict=True)
            )
        ),
    )
    interpolator = OrbitInterpolator.from_orbit(orbit)
    evaluation_time = 1.0
    step = 1e-5
    before, _ = interpolator.evaluate_array(np.array([evaluation_time - step]))
    after, velocity = interpolator.evaluate_array(
        np.array([evaluation_time + step, evaluation_time])
    )
    numerical_velocity = (after[0] - before[0]) / (2.0 * step)
    np.testing.assert_allclose(numerical_velocity, velocity[1], atol=1e-7)


def test_zero_doppler_residual_and_baseline_are_finite() -> None:
    """Compute finite Doppler residual and baseline for a synthetic look."""
    orbit = _orbit()
    state = interpolate_orbit(orbit, orbit.vectors[2].time)
    target = (
        state.position_m[0] - 600_000.0,
        state.position_m[1],
        state.position_m[2],
    )
    residual = zero_doppler_residual_hz(state, target, wavelength_m=0.0555)
    assert np.isfinite(residual)

    look = np.asarray(target) - np.asarray(state.position_m)
    look = look / np.linalg.norm(look)
    secondary = OrbitMetadata(
        reference_frame="ITRF",
        source="synthetic-secondary",
        vectors=tuple(
            OrbitStateVector(
                time=vector.time,
                position_m=(
                    vector.position_m[0] + 100.0,
                    vector.position_m[1],
                    vector.position_m[2],
                ),
                velocity_m_s=vector.velocity_m_s,
            )
            for vector in orbit.vectors
        ),
    )
    baseline = geometric_baseline(
        orbit,
        secondary,
        time=orbit.vectors[2].time,
        look_unit_ecef=(float(look[0]), float(look[1]), float(look[2])),
    )
    assert baseline.magnitude_m == pytest.approx(100.0, rel=1e-6)
    assert baseline.perpendicular_m >= 0.0
    assert local_earth_radius_m(37.0) > 6_300_000.0
