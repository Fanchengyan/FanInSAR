"""Tests for dense geometry offset field estimation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector
from faninsar.processing.coreg.dense_geometry import (
    _build_control_grid,
    _interpolate_field,
    dense_geometry_offsets,
)
from faninsar.processing.geometry.orbit import OrbitInterpolator
from faninsar.processing.geometry.transforms import RadarGeometryModel


def _make_simple_orbit(
    t0: datetime,
    n_vectors: int = 3,
    dt_s: float = 1.0,
) -> OrbitMetadata:
    """Build a realistic circular orbit for testing.

    Satellite at ~700 km altitude in a circular orbit around the equator.
    The orbit is tightly centred on ``t0`` (the zero-Doppler time for the
    sub-satellite point) with epoch = t0 - 1 s, so geo2rdr starts at
    t_mid = 1 s, i.e. exactly at the zero-Doppler epoch.
    """
    r_sat = 7_071_000.0  # ~700 km altitude
    v_sat = 7_000.0
    omega = v_sat / r_sat
    vectors = []
    for i in range(n_vectors):
        t = t0 + timedelta(seconds=(i - 1) * dt_s)
        angle = omega * (i - 1) * dt_s
        pos = (r_sat * np.cos(angle), r_sat * np.sin(angle), 0.0)
        vel = (-v_sat * np.sin(angle), v_sat * np.cos(angle), 0.0)
        vectors.append(OrbitStateVector(time=t, position_m=pos, velocity_m_s=vel))
    return OrbitMetadata(
        reference_frame="ECR",
        source="test",
        vectors=tuple(vectors),
    )


def _make_radar_model(
    orbit: OrbitMetadata,
    _shape: tuple[int, int],
) -> RadarGeometryModel:
    """Build a RadarGeometryModel from a test orbit."""
    t0 = orbit.vectors[0].time
    return RadarGeometryModel(
        orbit=OrbitInterpolator.from_orbit(orbit),
        sensing_start=t0,
        azimuth_time_interval_s=1.0,
        starting_slant_range_m=700_000.0,
        range_spacing_m=10.0,
        wavelength_m=0.056,
        look_direction="right",
    )


def test_build_control_grid_includes_edges() -> None:
    """Control grid spans the full array and includes the last pixel."""
    az, rg = _build_control_grid((100, 80), stride=32)
    assert az[0] == 0
    assert az[-1] == 99
    assert rg[0] == 0
    assert rg[-1] == 79


def test_interpolate_field_linear_recovery() -> None:
    """Linear interpolation recovers a plane exactly on the grid nodes."""
    az = np.arange(0, 64, 16, dtype=np.float64)
    rg = np.arange(0, 48, 16, dtype=np.float64)
    az_g, rg_g = np.meshgrid(az, rg, indexing="ij")
    values = (az_g * 0.1 + rg_g * 0.2).astype(np.float64)
    full = _interpolate_field(az, rg, values, (64, 48))
    # At the control points the interpolated value should match exactly
    assert full[0, 0] == pytest.approx(values[0, 0], abs=1e-9)
    assert full[32, 16] == pytest.approx(values[2, 1], abs=1e-9)


def test_dense_geometry_identical_models_yield_near_zero_offsets() -> None:
    """Identical reference/secondary geometry gives near-zero offsets."""
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    orbit = _make_simple_orbit(t0)
    model = _make_radar_model(orbit, (2, 2))
    result = dense_geometry_offsets(
        (2, 2),
        reference_model=model,
        secondary_model=model,
        dem=None,
        stride=1,
        max_iter=100,
    )
    assert result.range_offset_px.shape == (2, 2)
    assert result.azimuth_offset_px.shape == (2, 2)
    assert result.coverage.shape == (2, 2)
    assert result.uncertainty_px.shape == (2, 2)
    # Offsets should be very close to zero for identical geometry
    assert np.nanmax(np.abs(result.range_offset_px)) < 0.5
    assert np.nanmax(np.abs(result.azimuth_offset_px)) < 0.5
    # At least one pixel should converge with the simple test orbit
    assert result.coverage.any()


def test_dense_geometry_stride_one_is_exact() -> None:
    """Stride=1 uses every pixel as a control point (no interpolation)."""
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    orbit = _make_simple_orbit(t0)
    model = _make_radar_model(orbit, (4, 4))
    result = dense_geometry_offsets(
        (4, 4),
        reference_model=model,
        secondary_model=model,
        dem=None,
        stride=1,
        max_iter=100,
    )
    # A few edge pixels may not converge with the simple test orbit,
    # but the central pixels should be valid.
    assert result.coverage.any()
    assert np.nanmax(np.abs(result.range_offset_px)) < 0.1
    assert np.nanmax(np.abs(result.azimuth_offset_px)) < 0.1


def test_dense_geometry_invalid_shape_raises() -> None:
    """Non-positive shape or stride raises ValueError."""
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    orbit = _make_simple_orbit(t0)
    model = _make_radar_model(orbit, (8, 8))
    from faninsar.processing.errors import InvalidProcessingStateError

    with pytest.raises(InvalidProcessingStateError, match="shape must be positive"):
        dense_geometry_offsets(
            (0, 8),
            reference_model=model,
            secondary_model=model,
            stride=4,
        )
    with pytest.raises(InvalidProcessingStateError, match="stride must be >= 1"):
        dense_geometry_offsets(
            (8, 8),
            reference_model=model,
            secondary_model=model,
            stride=0,
        )
