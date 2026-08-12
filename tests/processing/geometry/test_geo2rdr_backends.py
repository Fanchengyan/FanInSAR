"""Backend consistency tests for the accelerated geo2rdr dispatch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.geometry import RadarGeometryModel, geo2rdr
from faninsar.processing.geometry.ellipsoid import ecef_to_llh
from faninsar.processing.geometry.geo2rdr_backends import (
    available_backends,
    cpp_library_path,
    resolve_backend,
)
from faninsar.processing.geometry.orbit import interpolate_orbit


def _orbit_and_grid() -> tuple[OrbitMetadata, RadarGrid]:
    """Build the shared synthetic low-Earth orbit and radar grid."""
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


def _target_llh(model: RadarGeometryModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one geodetic target on the right-looking look vector."""
    state = model.orbit.evaluate(model.sensing_start)
    sat = np.asarray(state.position_m, dtype=np.float64)
    vel = np.asarray(state.velocity_m_s, dtype=np.float64)
    vel_u = vel / np.linalg.norm(vel)
    radial = sat / np.linalg.norm(sat)
    cross = np.cross(vel_u, radial)
    look = cross / np.linalg.norm(cross)
    target = sat + look * 650_000.0
    lat, lon, h = ecef_to_llh(target[0], target[1], target[2])
    return (
        np.array([float(lat)], dtype=np.float64),
        np.array([float(lon)], dtype=np.float64),
        np.array([float(h)], dtype=np.float64),
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_geo2rdr_backend_converges_on_look_vector(backend: str) -> None:
    """Every available backend recovers the constructed look-vector target."""
    if backend == "torch" and "torch" not in available_backends():
        pytest.skip("torch backend unavailable")
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    lat, lon, h = _target_llh(model)
    result = geo2rdr(model, lat, lon, h, backend=backend)
    assert bool(result.converged[0])
    assert np.isfinite(result.range_index[0])
    assert np.isfinite(result.azimuth_index[0])
    expected_rg = (650_000.0 - grid.starting_slant_range_m) / grid.range_spacing_m
    assert abs(result.range_index[0] - expected_rg) < 5.0


def test_torch_backend_matches_numpy_reference() -> None:
    """Torch and numpy backends agree on indices and convergence."""
    if "torch" not in available_backends():
        pytest.skip("torch backend unavailable")
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    lat, lon, h = _target_llh(model)
    lat_b, lon_b, h_b = np.broadcast_arrays(lat, lon, h)
    numpy_result = geo2rdr(model, lat_b, lon_b, h_b, backend="numpy")
    torch_result = geo2rdr(model, lat_b, lon_b, h_b, backend="torch")
    assert np.array_equal(torch_result.converged, numpy_result.converged)
    assert np.allclose(
        torch_result.range_index,
        numpy_result.range_index,
        atol=1e-3,
        equal_nan=True,
    )
    assert np.allclose(
        torch_result.azimuth_index,
        numpy_result.azimuth_index,
        atol=1e-3,
        equal_nan=True,
    )


def test_cpp_backend_matches_numpy_reference() -> None:
    """C++/OpenMP backend agrees with the numpy reference when built."""
    if cpp_library_path() is None:
        pytest.skip("C++ geo2rdr library not built")
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    lat, lon, h = _target_llh(model)
    lat_b, lon_b, h_b = np.broadcast_arrays(lat, lon, h)
    numpy_result = geo2rdr(model, lat_b, lon_b, h_b, backend="numpy")
    cpp_result = geo2rdr(model, lat_b, lon_b, h_b, backend="cpp")
    assert np.array_equal(cpp_result.converged, numpy_result.converged)
    assert np.allclose(
        cpp_result.range_index,
        numpy_result.range_index,
        atol=1e-3,
        equal_nan=True,
    )
    assert np.allclose(
        cpp_result.azimuth_index,
        numpy_result.azimuth_index,
        atol=1e-3,
        equal_nan=True,
    )


def test_auto_backend_resolves_to_an_available_backend() -> None:
    """Auto resolution always returns a concrete usable backend."""
    resolved = resolve_backend("auto")
    assert resolved in available_backends()


def test_torch_backend_masks_nonfinite_inputs() -> None:
    """Non-finite coordinates are masked, not propagated as garbage."""
    if "torch" not in available_backends():
        pytest.skip("torch backend unavailable")
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    lat, lon, h = _target_llh(model)
    lat_input = np.array([lat[0], np.nan], dtype=np.float64)
    lon_input = np.array([lon[0], 0.0], dtype=np.float64)
    h_input = np.array([h[0], 0.0], dtype=np.float64)
    result = geo2rdr(model, lat_input, lon_input, h_input, backend="torch")
    assert not bool(result.converged[1])
    assert not np.isfinite(result.range_index[1])
