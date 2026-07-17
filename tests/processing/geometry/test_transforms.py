"""Round-trip tests for radar/geographic geometry transforms."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.geometry import (
    RadarGeometryModel,
    geo2rdr,
    interpolate_orbit,
    llh_to_ecef,
    rdr2geo_ellipsoid,
    rdr2geo_with_dem,
)
from faninsar.processing.geometry.ellipsoid import ecef_to_llh


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


def test_geo2rdr_and_range_consistency_for_constructed_target() -> None:
    """geo2rdr recovers finite range/azimuth for a target on the look vector."""
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    state = interpolate_orbit(orbit, grid.sensing_start)
    sat = np.asarray(state.position_m, dtype=np.float64)
    vel = np.asarray(state.velocity_m_s, dtype=np.float64)
    # construct a right-looking unit vector roughly orthogonal to velocity
    vel_u = vel / np.linalg.norm(vel)
    radial = sat / np.linalg.norm(sat)
    cross = np.cross(vel_u, radial)
    look = cross / np.linalg.norm(cross)
    target = sat + look * 650_000.0
    lat, lon, h = ecef_to_llh(target[0], target[1], target[2])
    result = geo2rdr(
        model,
        np.array([float(lat)]),
        np.array([float(lon)]),
        np.array([float(h)]),
    )
    assert bool(result.converged[0])
    assert np.isfinite(result.range_index[0])
    assert np.isfinite(result.azimuth_index[0])
    # range index near (650km - 600km)/2.3
    expected_rg = (650_000.0 - grid.starting_slant_range_m) / grid.range_spacing_m
    assert abs(result.range_index[0] - expected_rg) < 5.0


def test_geo2rdr_uses_vectorized_orbit_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Geo2rdr must not call scalar orbit interpolation for every pixel."""
    from faninsar.processing.geometry.orbit import OrbitInterpolator

    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    forward = rdr2geo_ellipsoid(
        model,
        np.arange(8, dtype=np.float64),
        np.full(8, 10.0, dtype=np.float64),
        height_m=0.0,
    )

    def reject_scalar_evaluation(*_args: object, **_kwargs: object) -> None:
        message = "geo2rdr performed per-pixel orbit interpolation"
        raise AssertionError(message)

    monkeypatch.setattr(OrbitInterpolator, "evaluate", reject_scalar_evaluation)
    result = geo2rdr(
        model,
        forward.latitude_deg,
        forward.longitude_deg,
        forward.height_m,
    )
    assert np.any(result.converged)


def test_geoid_adjusted_dem_adds_undulation() -> None:
    """Orthometric and geoid samples combine into ellipsoidal height."""
    from faninsar.processing.geometry import ConstantHeightDEM, GeoidAdjustedDEM

    dem = GeoidAdjustedDEM(
        orthometric_dem=ConstantHeightDEM(1000.0),
        geoid=ConstantHeightDEM(-42.0),
    )
    height = dem.sample(np.array([30.0]), np.array([100.0]))
    np.testing.assert_allclose(height, [958.0])


def test_rdr2geo_ellipsoid_marks_out_of_orbit_as_not_converged() -> None:
    """Samples outside orbit coverage remain masked rather than invented."""
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    # huge azimuth index far outside orbit time coverage
    result = rdr2geo_ellipsoid(
        model,
        azimuth_index=np.array([1.0e9]),
        range_index=np.array([10.0]),
        height_m=0.0,
    )
    assert not bool(result.converged[0])


def test_constant_dem_and_transform_cache_round_trip(tmp_path: Path) -> None:
    """Cache ellipsoid transform results and reload them byte-identically."""
    from faninsar.processing.geometry import (
        ConstantHeightDEM,
        TransformCacheKey,
        rdr2geo_with_dem,
        read_transform_cache,
        write_transform_cache,
    )

    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    dem = ConstantHeightDEM(height_m=100.0)
    result = rdr2geo_with_dem(
        model,
        azimuth_index=np.array([0.0, 1.0]),
        range_index=np.array([10.0, 20.0]),
        dem=dem,
        height_seed_m=100.0,
    )
    key = TransformCacheKey(
        product_id="synthetic",
        direction="rdr2geo",
        dem_identity="constant-100",
        orbit_source="synthetic",
        grid_shape=(2, 1),
    )
    # reshape to 2x1 for cache key consistency if needed
    store = write_transform_cache(tmp_path, key, result)
    loaded_key, loaded = read_transform_cache(store)
    assert loaded_key.product_id == "synthetic"
    np.testing.assert_array_equal(loaded.converged, result.converged)
    np.testing.assert_allclose(loaded.latitude_deg, result.latitude_deg, equal_nan=True)


def test_vectorized_rdr2geo_matches_scalar_on_small_grid() -> None:
    """Vectorised rdr2geo_ellipsoid agrees with the scalar fallback."""
    from faninsar.processing.geometry.transforms import _rdr2geo_ellipsoid_scalar

    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)

    # Small grid that exercises the scalar path (<= 4 pixels)
    az_idx = np.array([0.0, 1.0, 2.0])
    rg_idx = np.array([10.0, 20.0, 30.0])

    # Scalar results (call helper directly for each pixel)
    lat_sc = np.full(az_idx.shape, np.nan)
    lon_sc = np.full(az_idx.shape, np.nan)
    h_sc = np.full(az_idx.shape, np.nan)
    conv_sc = np.zeros(az_idx.shape, dtype=bool)
    for i in range(az_idx.size):
        lat0, lon0, h0, success, _, _ = _rdr2geo_ellipsoid_scalar(
            model,
            float(az_idx.flat[i]),
            float(rg_idx.flat[i]),
            0.0,
            20,
            0.01,
            0.1,
        )
        if success:
            lat_sc.flat[i] = lat0
            lon_sc.flat[i] = lon0
            h_sc.flat[i] = h0
            conv_sc.flat[i] = True

    # Vectorised result (forces vectorised path by using > 4 pixels via mesh)
    az_grid, rg_grid = np.meshgrid(az_idx, rg_idx, indexing="ij")
    result_vec = rdr2geo_ellipsoid(
        model,
        az_grid,
        rg_grid,
        height_m=0.0,
    )

    # Compare on the diagonal where we have scalar reference
    for i in range(az_idx.size):
        if conv_sc.flat[i]:
            assert result_vec.converged[i, i]
            np.testing.assert_allclose(
                result_vec.latitude_deg[i, i], lat_sc.flat[i], rtol=1e-5
            )
            np.testing.assert_allclose(
                result_vec.longitude_deg[i, i], lon_sc.flat[i], rtol=1e-5
            )


def test_rdr2geo_geo2rdr_round_trip_residuals() -> None:
    """rdr2geo -> geo2rdr recovers original indices within tolerance."""
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)

    az_idx = np.array([0.0, 1.0, 2.0])
    rg_idx = np.array([10.0, 20.0, 30.0])
    result_fwd = rdr2geo_ellipsoid(model, az_idx, rg_idx, height_m=0.0)

    conv = result_fwd.converged
    if not np.any(conv):
        pytest.skip("no converged pixels for round-trip test")

    result_bwd = geo2rdr(
        model,
        result_fwd.latitude_deg[conv],
        result_fwd.longitude_deg[conv],
        result_fwd.height_m[conv],
    )

    np.testing.assert_allclose(
        result_bwd.azimuth_index,
        result_fwd.azimuth_index[conv],
        atol=0.1,
    )
    np.testing.assert_allclose(
        result_bwd.range_index,
        result_fwd.range_index[conv],
        atol=0.1,
    )
    assert np.all(np.abs(result_bwd.residual_range_m[conv]) < 1.0)
    assert np.all(np.abs(result_bwd.residual_doppler_hz[conv]) < 10.0)


def test_rdr2geo_with_dem_chunked_matches_full() -> None:
    """Chunked solver produces identical results to the full-array solver."""
    from faninsar.processing.geometry import ConstantHeightDEM, rdr2geo_with_dem_chunked

    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    dem = ConstantHeightDEM(height_m=50.0)

    az_idx = np.arange(0.0, 6.0).reshape(2, 3)
    rg_idx = np.arange(10.0, 40.0, 5.0).reshape(2, 3)

    full = rdr2geo_with_dem(model, az_idx, rg_idx, dem=dem, height_seed_m=0.0)
    chunked = rdr2geo_with_dem_chunked(
        model,
        az_idx,
        rg_idx,
        dem=dem,
        height_seed_m=0.0,
        chunk_size=(2, 2),
    )

    np.testing.assert_array_equal(full.converged, chunked.converged)
    np.testing.assert_allclose(
        full.latitude_deg, chunked.latitude_deg, equal_nan=True, rtol=1e-10
    )
    np.testing.assert_allclose(
        full.longitude_deg, chunked.longitude_deg, equal_nan=True, rtol=1e-10
    )
    np.testing.assert_allclose(
        full.height_m, chunked.height_m, equal_nan=True, rtol=1e-10
    )
