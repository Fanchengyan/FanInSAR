"""Tests for dual-coordinate SLC API helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from faninsar.processing.contracts import (
    ArrayDescriptor,
    ArrayRepresentation,
    CalibrationState,
    CarrierState,
    CoregistrationState,
    OrbitMetadata,
    OrbitStateVector,
    SLCProduct,
)
from faninsar.processing.coordinates import GeoGrid, RadarGrid
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.slc import RadarSLC, choose_processing_grid
from faninsar.processing.slc.dual import _radar_to_geo_resample


def _radar_product() -> SLCProduct:
    grid = RadarGrid(
        shape=(8, 8),
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=datetime(2024, 1, 1, tzinfo=UTC),
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )
    orbit = OrbitMetadata(
        reference_frame="ITRF",
        source="synthetic",
        vectors=(
            OrbitStateVector(
                time=datetime(2024, 1, 1, tzinfo=UTC),
                position_m=(7_000_000.0, 0.0, 0.0),
                velocity_m_s=(0.0, 7500.0, 0.0),
            ),
            OrbitStateVector(
                time=datetime(2024, 1, 1, 0, 0, 10, tzinfo=UTC),
                position_m=(7_000_000.0, 75_000.0, 0.0),
                velocity_m_s=(0.0, 7500.0, 0.0),
            ),
        ),
    )
    return SLCProduct(
        acquisition_id="20240101",
        grid=grid,
        samples=ArrayDescriptor(
            uri="memory://20240101",
            shape=grid.shape,
            dtype="complex64",
            representation=ArrayRepresentation.COMPLEX,
        ),
        orbit=orbit,
        carrier=CarrierState.PRESENT,
        coregistration=CoregistrationState.NOT_REGISTERED,
        calibration=CalibrationState.RAW_DN,
    )


def test_choose_processing_grid_auto_prefers_radar() -> None:
    """Auto mode selects radar when a RadarSLC is available."""
    assert choose_processing_grid("auto", has_radar=True, has_geo=True) == "radar"
    assert choose_processing_grid("geo", has_radar=True, has_geo=True) == "geo"


def test_choose_processing_grid_rejects_missing_branch() -> None:
    """Requesting an unavailable branch raises a typed error."""
    with pytest.raises(InvalidProcessingStateError):
        choose_processing_grid("geo", has_radar=True, has_geo=False)


def test_radar_slc_rejects_non_complex_samples() -> None:
    """RadarSLC construction rejects real-valued samples."""
    product = _radar_product()
    with pytest.raises(InvalidProcessingStateError):
        RadarSLC(product=product, samples=np.ones((8, 8), dtype=np.float32))


def test_radar_slc_accepts_matching_complex_window() -> None:
    """RadarSLC stores a matching complex window on a radar grid."""
    product = _radar_product()
    samples = np.ones((8, 8), dtype=np.complex64)
    slc = RadarSLC(product=product, samples=samples)
    assert isinstance(slc.grid, RadarGrid)
    assert slc.samples.shape == (8, 8)


def test_radar_to_geo_resample_projects_wgs84_into_utm() -> None:
    """Projected Geo grids receive WGS84 geometry at the correct pixel."""
    from pyproj import Transformer

    longitude = np.array([[-147.0]], dtype=np.float64)
    latitude = np.array([[65.0]], dtype=np.float64)
    x_coordinate, y_coordinate = Transformer.from_crs(
        "EPSG:4326", "EPSG:32606", always_xy=True
    ).transform(longitude, latitude)
    target = GeoGrid(
        shape=(3, 3),
        crs="EPSG:32606",
        transform=(
            20.0,
            0.0,
            float(x_coordinate[0, 0]) - 20.0,
            0.0,
            -20.0,
            float(y_coordinate[0, 0]) + 20.0,
        ),
    )
    transform = SimpleNamespace(
        converged=np.array([[True]]),
        latitude_deg=latitude,
        longitude_deg=longitude,
        azimuth_index=np.array([[0.0]]),
        range_index=np.array([[0.0]]),
    )

    result = _radar_to_geo_resample(
        np.array([[2.0 + 3.0j]], dtype=np.complex64),
        transform,
        target,
    )

    assert result[1, 1] == np.complex64(2.0 + 3.0j)
    assert np.count_nonzero(result) == 1
