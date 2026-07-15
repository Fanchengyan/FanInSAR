"""Tests for dual-coordinate SLC API helpers."""

from __future__ import annotations

from datetime import UTC, datetime

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
