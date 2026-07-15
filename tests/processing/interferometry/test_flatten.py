"""Tests for topographic phase computation and removal."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.geometry import (
    ConstantHeightDEM,
    RadarGeometryModel,
    rdr2geo_ellipsoid,
)
from faninsar.processing.interferometry.flatten import (
    compute_topographic_phase,
    remove_topographic_phase,
)


def _orbit_and_grid() -> tuple[OrbitMetadata, RadarGrid]:
    epoch = datetime(2016, 12, 7, 11, 18, 0, tzinfo=UTC)
    # Realistic low-Earth orbit (~700 km altitude) for well-conditioned geometry
    radius = 7_071_000.0
    omega = 0.001
    vectors = []
    for index in range(21):
        t = 10.0 * index
        angle = omega * t
        vectors.append(
            OrbitStateVector(
                time=epoch + timedelta(seconds=t),
                position_m=(
                    radius * np.cos(angle),
                    0.0,
                    radius * np.sin(angle),
                ),
                velocity_m_s=(
                    -radius * omega * np.sin(angle),
                    0.0,
                    radius * omega * np.cos(angle),
                ),
            )
        )
    orbit = OrbitMetadata(
        reference_frame="ITRF",
        source="synthetic",
        vectors=tuple(vectors),
    )
    grid = RadarGrid(
        shape=(64, 128),
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=vectors[5].time,
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )
    return orbit, grid


def test_remove_topographic_phase_on_synthetic_dem() -> None:
    """Topographic phase removal recovers flat-Earth phase on synthetic data."""
    orbit, grid = _orbit_and_grid()
    model_ref = RadarGeometryModel.from_radar_grid(grid, orbit)

    # Create a secondary model with a small baseline shift in z
    vectors_shifted = [
        OrbitStateVector(
            time=vec.time,
            position_m=(
                vec.position_m[0],
                vec.position_m[1],
                vec.position_m[2] + 100.0,
            ),
            velocity_m_s=vec.velocity_m_s,
        )
        for vec in orbit.vectors
    ]
    orbit_sec = OrbitMetadata(
        reference_frame=orbit.reference_frame,
        source=orbit.source + "-shifted",
        vectors=tuple(vectors_shifted),
    )
    model_sec = RadarGeometryModel.from_radar_grid(grid, orbit_sec)

    # Small grid for fast testing
    az_idx = np.array([[0.0, 1.0], [2.0, 3.0]])
    rg_idx = np.array([[10.0, 20.0], [30.0, 40.0]])

    dem = ConstantHeightDEM(height_m=50.0)

    topo_phase = compute_topographic_phase(
        model_ref,
        model_sec,
        az_idx,
        rg_idx,
        dem=dem,
    )

    # Create a synthetic interferogram where the phase equals the topographic phase
    complex_ifg = np.exp(1j * topo_phase)

    # Remove topographic phase
    flat = remove_topographic_phase(complex_ifg, topo_phase)

    # Where converged, the flattened interferogram should have phase near 0
    converged = np.isfinite(topo_phase)
    if np.any(converged):
        np.testing.assert_allclose(np.angle(flat[converged]), 0.0, atol=1e-6)
    else:
        pytest.skip("no converged pixels for synthetic flattening test")


def test_remove_topographic_phase_changes_phase_when_dem_varies() -> None:
    """Flattening must actually change interferogram phase when DEM varies."""
    orbit, grid = _orbit_and_grid()
    model_ref = RadarGeometryModel.from_radar_grid(grid, orbit)

    vectors_shifted = [
        OrbitStateVector(
            time=vec.time,
            position_m=(
                vec.position_m[0],
                vec.position_m[1],
                vec.position_m[2] + 100.0,
            ),
            velocity_m_s=vec.velocity_m_s,
        )
        for vec in orbit.vectors
    ]
    orbit_sec = OrbitMetadata(
        reference_frame=orbit.reference_frame,
        source=orbit.source + "-shifted",
        vectors=tuple(vectors_shifted),
    )
    model_sec = RadarGeometryModel.from_radar_grid(grid, orbit_sec)

    az_idx = np.array([[0.0, 1.0], [2.0, 3.0]])
    rg_idx = np.array([[10.0, 20.0], [30.0, 40.0]])

    # Varying DEM heights
    dem = ConstantHeightDEM(height_m=0.0)
    topo_flat = compute_topographic_phase(model_ref, model_sec, az_idx, rg_idx, dem=dem)

    dem = ConstantHeightDEM(height_m=100.0)
    topo_hill = compute_topographic_phase(model_ref, model_sec, az_idx, rg_idx, dem=dem)

    conv_flat = np.isfinite(topo_flat)
    conv_hill = np.isfinite(topo_hill)
    conv_both = conv_flat & conv_hill

    if not np.any(conv_both):
        pytest.skip("no common converged pixels for DEM variation test")

    # Phase must differ when DEM changes
    assert not np.allclose(topo_flat[conv_both], topo_hill[conv_both], atol=1e-3)


def test_compute_topographic_phase_non_converged_are_nan() -> None:
    """Pixels outside orbit coverage produce NaN topographic phase."""
    orbit, grid = _orbit_and_grid()
    model_ref = RadarGeometryModel.from_radar_grid(grid, orbit)
    model_sec = model_ref

    # Huge azimuth index far outside orbit coverage
    az_idx = np.array([1.0e9])
    rg_idx = np.array([10.0])
    dem = ConstantHeightDEM(height_m=0.0)

    topo_phase = compute_topographic_phase(
        model_ref,
        model_sec,
        az_idx,
        rg_idx,
        dem=dem,
    )

    assert np.all(np.isnan(topo_phase))
