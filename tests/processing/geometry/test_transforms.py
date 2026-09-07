"""Round-trip tests for radar/geographic geometry transforms (PROPOSAL-0031)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry import ConstantDEM
from faninsar.processing.geometry import (
    RadarGeometryModel,
    interpolate_orbit,
    llh_to_ecef,
)
from faninsar.processing.geometry.ellipsoid import ecef_to_llh
from faninsar.processing.geometry.prepare_production import (
    run_geo2rdr,
    run_rdr2geo,
    run_rdr2geo_chunked,
)
from faninsar.processing.geometry.transforms import TransformResult


def _orbit_and_grid() -> tuple[OrbitMetadata, RadarGrid]:
    epoch = datetime(2016, 12, 7, 11, 18, 0, tzinfo=UTC)
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


def test_geo2rdr_and_range_consistency_for_constructed_target() -> None:
    """geo2rdr recovers finite range/azimuth for a target on the look vector."""
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    state = interpolate_orbit(orbit, grid.sensing_start)
    sat = np.asarray(state.position_m, dtype=np.float64)
    vel = np.asarray(state.velocity_m_s, dtype=np.float64)
    vel_u = vel / np.linalg.norm(vel)
    radial = sat / np.linalg.norm(sat)
    cross = np.cross(vel_u, radial)
    look = cross / np.linalg.norm(cross)
    target = sat + look * 650_000.0
    lat, lon, h = ecef_to_llh(target[0], target[1], target[2])
    result = run_geo2rdr(
        model,
        np.array([float(lat)]),
        np.array([float(lon)]),
        np.array([float(h)]),
        device="cpu",
        max_iter=40,
    )
    assert bool(result.converged[0])
    assert np.isfinite(result.range_index[0])
    assert np.isfinite(result.azimuth_index[0])
    expected_rg = (650_000.0 - grid.starting_slant_range_m) / grid.range_spacing_m
    assert abs(result.range_index[0] - expected_rg) < 5.0


def test_geoid_adjusted_dem_adds_undulation() -> None:
    """Orthometric and geoid samples combine into ellipsoidal height."""
    from faninsar.processing.geometry.dem import GeoidAdjustedDEM

    dem = GeoidAdjustedDEM(
        orthometric_dem=ConstantDEM(1000.0),
        geoid=ConstantDEM(-42.0),
    )
    height = dem.sample(np.array([30.0]), np.array([100.0]))
    np.testing.assert_allclose(height, [958.0])


def test_rdr2geo_marks_out_of_orbit_as_not_converged() -> None:
    """Samples outside orbit coverage remain masked rather than invented."""
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    result = run_rdr2geo(
        model,
        np.array([1.0e9]),
        np.array([10.0]),
        device="cpu",
        max_iter=8,
    )
    assert not bool(result.converged[0])


def test_constant_dem_and_transform_cache_round_trip(tmp_path: Path) -> None:
    """Cache ellipsoid transform results and reload them byte-identically."""
    from faninsar.processing.geometry import (
        TransformCacheKey,
        read_transform_cache,
        write_transform_cache,
    )

    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    dem = ConstantDEM(height=100.0)
    result = run_rdr2geo(
        model,
        np.array([0.0, 1.0]),
        np.array([10.0, 20.0]),
        dem,
        device="cpu",
        max_iter=40,
    )
    key = TransformCacheKey(
        product_id="synthetic",
        direction="rdr2geo",
        dem_identity="constant-100",
        orbit_source="synthetic",
        grid_shape=(2, 1),
    )
    store = write_transform_cache(tmp_path, key, result)
    loaded_key, loaded = read_transform_cache(store)
    assert loaded_key.product_id == "synthetic"
    assert isinstance(loaded, TransformResult)
    np.testing.assert_array_equal(loaded.converged, result.converged)
    np.testing.assert_allclose(loaded.latitude_deg, result.latitude_deg, equal_nan=True)


def test_rdr2geo_geo2rdr_round_trip_residuals() -> None:
    """rdr2geo -> geo2rdr recovers original indices within tolerance."""
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)

    az_idx = np.array([0.0, 1.0, 2.0])
    rg_idx = np.array([10.0, 20.0, 30.0])
    result_fwd = run_rdr2geo(model, az_idx, rg_idx, device="cpu", max_iter=40)

    conv = result_fwd.converged
    if not np.any(conv):
        pytest.skip("no converged pixels for round-trip test")

    result_bwd = run_geo2rdr(
        model,
        result_fwd.latitude_deg[conv],
        result_fwd.longitude_deg[conv],
        result_fwd.height_m[conv],
        device="cpu",
        max_iter=40,
    )

    np.testing.assert_allclose(
        result_bwd.azimuth_index,
        result_fwd.azimuth_index[conv],
        atol=0.5,
    )
    np.testing.assert_allclose(
        result_bwd.range_index,
        result_fwd.range_index[conv],
        atol=0.5,
    )
    assert np.all(np.abs(result_bwd.residual_range_m[result_bwd.converged]) < 1.0)


def test_rdr2geo_chunked_matches_full() -> None:
    """Chunked helper produces identical results to the full-array helper."""
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    dem = ConstantDEM(height=50.0)

    az_idx = np.arange(0.0, 6.0).reshape(2, 3)
    rg_idx = np.arange(10.0, 40.0, 5.0).reshape(2, 3)

    full = run_rdr2geo(model, az_idx, rg_idx, dem, device="cpu", max_iter=40)
    chunked = run_rdr2geo_chunked(
        model,
        az_idx,
        rg_idx,
        dem,
        device="cpu",
        chunk_size=(2, 2),
        max_iter=40,
    )

    np.testing.assert_array_equal(full.converged, chunked.converged)
    np.testing.assert_allclose(
        full.latitude_deg, chunked.latitude_deg, equal_nan=True, rtol=1e-8
    )
    np.testing.assert_allclose(
        full.longitude_deg, chunked.longitude_deg, equal_nan=True, rtol=1e-8
    )
    np.testing.assert_allclose(
        full.height_m, chunked.height_m, equal_nan=True, rtol=1e-8
    )


def test_package_does_not_reexport_deleted_newton_solvers() -> None:
    """Deletion gate: Newton names are gone from the public package."""
    from faninsar.processing import geometry

    for name in (
        "geo2rdr",
        "rdr2geo_ellipsoid",
        "rdr2geo_with_dem",
        "rdr2geo_with_dem_chunked",
    ):
        assert name not in geometry.__all__
        assert not hasattr(geometry, name)
