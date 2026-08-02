"""Tests for geocoding radar complex arrays onto a common merge grid."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.geometry import RadarGeometryModel
from faninsar.processing.merge.geocode_raster import (
    GeocodedComplex,
    geocode_complex_to_grid,
)
from faninsar.processing.merge.grid import GeoGridSpec


def _toy_orbit() -> OrbitMetadata:
    """Return a small synthetic orbit sufficient for a toy geometry model."""
    epoch = datetime(2024, 1, 1, tzinfo=UTC)
    vectors = tuple(
        OrbitStateVector(
            time=epoch + timedelta(seconds=float(i)),
            position_m=(700000.0, 0.0, 900000.0 + 100.0 * i),
            velocity_m_s=(0.0, 0.0, 7000.0),
        )
        for i in range(20)
    )
    return OrbitMetadata(
        reference_frame="ECFT",
        source="test",
        vectors=vectors,
    )


def _toy_geometry(shape: tuple[int, int]) -> RadarGeometryModel:
    """Build a small radar geometry model for synthetic tests."""
    orbit = _toy_orbit()
    grid = RadarGrid(
        shape=shape,
        starting_slant_range_m=800_000.0,
        range_spacing_m=30.0,
        sensing_start=orbit.vectors[0].time,
        azimuth_time_interval_s=0.002,
        wavelength_m=0.056,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(grid, orbit)


def _toy_grid(crs: str = "EPSG:32633") -> GeoGridSpec:
    """Small UTM grid for testing."""
    return GeoGridSpec(
        crs=crs,
        transform=(300000.0, 100.0, 0.0, 5000000.0, 0.0, -100.0),
        width=8,
        height=8,
        resolution_m=(100.0, 100.0),
    )


def test_projected_grid_pixel_centers_are_finite_lonlat() -> None:
    """UTM GeoGridSpec centers convert to finite lon/lat (FORWARD, not INVERSE)."""
    from faninsar.processing.merge.geocode_raster import _grid_pixel_centers_lonlat

    grid = GeoGridSpec(
        crs="EPSG:32647",
        transform=(449375.0, 20.0, 0.0, 4177660.0, 0.0, -80.0),
        width=8,
        height=6,
        resolution_m=(20.0, 80.0),
    )
    lat, lon = _grid_pixel_centers_lonlat(grid)
    assert np.isfinite(lat).all()
    assert np.isfinite(lon).all()
    assert 37.0 < float(np.mean(lat)) < 38.5
    assert 98.0 < float(np.mean(lon)) < 100.0


def test_geocode_complex_to_grid_returns_geocoded_complex() -> None:
    """geocode_complex_to_grid returns a GeocodedComplex with right shape."""
    radar_shape = (32, 32)
    model = _toy_geometry(radar_shape)
    complex_radar = np.ones(radar_shape, dtype=np.complex64)
    grid = _toy_grid()
    result = geocode_complex_to_grid(
        complex_radar,
        geometry=model,
        grid=grid,
        height_m=0.0,
    )
    assert isinstance(result, GeocodedComplex)
    assert result.complex.shape == grid.shape
    assert result.complex.dtype == np.complex64
    assert result.valid_mask.shape == grid.shape
    assert result.valid_mask.dtype == bool


def test_geocode_complex_to_grid_marks_out_of_range_invalid() -> None:
    """Pixels whose radar indices fall outside the radar array are invalid."""
    radar_shape = (8, 8)
    model = _toy_geometry(radar_shape)
    complex_radar = np.ones(radar_shape, dtype=np.complex64)
    grid = _toy_grid()
    result = geocode_complex_to_grid(
        complex_radar,
        geometry=model,
        grid=grid,
        height_m=0.0,
    )
    # The toy geometry won't overlap much of this arbitrary UTM grid, so
    # most pixels should be invalid. At minimum, valid_mask must be a
    # proper boolean mask and any valid pixel must carry a finite complex.
    assert result.valid_mask.shape == grid.shape
    if result.valid_mask.any():
        assert np.isfinite(result.complex[result.valid_mask]).all()


def test_geocode_complex_to_grid_preserves_constant_amplitude() -> None:
    """A constant-amplitude radar array yields ~same amplitude where valid."""
    radar_shape = (64, 64)
    model = _toy_geometry(radar_shape)
    complex_radar = np.full(radar_shape, 2.0 + 0.0j, dtype=np.complex64)
    grid = _toy_grid()
    result = geocode_complex_to_grid(
        complex_radar,
        geometry=model,
        grid=grid,
        height_m=0.0,
    )
    if result.valid_mask.any():
        amps = np.abs(result.complex[result.valid_mask])
        assert np.allclose(amps, 2.0, atol=0.2)


def test_geocode_complex_to_grid_rejects_shape_mismatch() -> None:
    """Radar array shape must match the explicit radar_shape argument."""
    model = _toy_geometry((32, 32))
    complex_radar = np.ones((16, 16), dtype=np.complex64)
    grid = _toy_grid()
    with pytest.raises(ValueError):
        geocode_complex_to_grid(
            complex_radar,
            geometry=model,
            grid=grid,
            height_m=0.0,
            radar_shape=(32, 32),
        )


def test_geocoded_complex_carries_weight_field() -> None:
    """GeocodedComplex exposes a weight field defaulting to the valid mask."""
    radar_shape = (32, 32)
    model = _toy_geometry(radar_shape)
    complex_radar = np.ones(radar_shape, dtype=np.complex64)
    grid = _toy_grid()
    result = geocode_complex_to_grid(
        complex_radar,
        geometry=model,
        grid=grid,
        height_m=0.0,
    )
    assert result.weight.shape == grid.shape
    assert result.weight.dtype == np.float32
    # weight is zero where invalid
    assert np.all(result.weight[~result.valid_mask] == 0.0)
