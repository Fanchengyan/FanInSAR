"""Burst ground quads fall back to the ellipsoid outside the DEM footprint.

Stack DEMs are ROI-bounded, while the burst-frame quad corners are the
full-burst edges and therefore sit far outside that footprint; the
DEM-based rdr2geo lookup cannot bracket them and returns NaN.  The
documented contract is a per-corner ellipsoid fallback so ROI footprint
intersection keeps working with ROI-bounded DEMs (PROPOSAL-0061 seam).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from affine import Affine

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.processing.geocoding.geo_lut import burst_geo_quad_lonlat
from faninsar.processing.geometry import RadarGeometryModel
from faninsar.processing.geometry.api import RasterDEM
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.grids import GridSpec


def _orbit_and_grid() -> tuple[OrbitMetadata, RadarGrid]:
    """Return a synthetic circular orbit and matching radar grid."""
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


def _tiny_dem_away_from(lon: float, lat: float) -> RasterDEM:
    """Return an 8x8 ellipsoidal DEM box centred far from ``(lon, lat)``."""
    grid = GridSpec(
        "EPSG:4326",
        Affine(0.001, 0.0, lon - 0.004, 0.0, -0.001, lat + 0.004),
        shape=(8, 8),
    )
    values = np.full((8, 8), 100.0, dtype=np.float32)
    return RasterDEM(array=values, grid=grid, vertical_datum="ellipsoidal")


def test_quad_falls_back_to_ellipsoid_outside_dem_footprint() -> None:
    """Corners the ROI-bounded DEM cannot bracket still produce a quad."""
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    shape = (64, 128)
    reference = burst_geo_quad_lonlat(model, shape, None, device="cpu")
    assert reference is not None
    center_lon = float(np.mean(reference[:, 0]))
    center_lat = float(np.mean(reference[:, 1]))
    dem = _tiny_dem_away_from(center_lon + 5.0, center_lat + 5.0)
    quad = burst_geo_quad_lonlat(model, shape, dem, device="cpu")
    assert quad is not None
    assert quad.shape[0] >= 3
    assert np.isfinite(quad).all()
    np.testing.assert_allclose(quad, reference, atol=1e-6)


def test_quad_without_dem_is_unchanged() -> None:
    """The ellipsoid-only path keeps returning a finite quad."""
    orbit, grid = _orbit_and_grid()
    model = RadarGeometryModel.from_radar_grid(grid, orbit)
    quad = burst_geo_quad_lonlat(model, (64, 128), None, device="cpu")
    assert quad is not None
    assert np.isfinite(quad).all()
