"""Unit tests for geographic-grid interferogram processing helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.geometry import RadarGeometryModel
from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.pipeline.geo_lut import (
    build_geo2rdr_lut,
    grid_lonlat,
)
from faninsar.processing.pipeline.geo_modes import (
    coregister_geocoded_slcs,
)
from faninsar.processing.pipeline.geo_resample import (
    apply_lut_complex,
    compose_secondary_coordinates,
)
from faninsar.processing.tops.deramp import TOPSCarrierModel

if TYPE_CHECKING:
    from pathlib import Path


def _toy_geometry(shape: tuple[int, int] = (64, 128)) -> RadarGeometryModel:
    epoch = datetime(2020, 1, 1, tzinfo=UTC)
    vectors = tuple(
        OrbitStateVector(
            time=epoch + timedelta(seconds=float(i)),
            position_m=(7000_000.0, 0.0, 0.0 + 100.0 * i),
            velocity_m_s=(0.0, 7500.0, 0.0),
        )
        for i in range(40)
    )
    orbit = OrbitMetadata(reference_frame="ECFT", source="test", vectors=vectors)
    rgrid = RadarGrid(
        shape=shape,
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=epoch,
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(rgrid, orbit)


def _projected_grid(pixel_size_m: float = 200.0) -> GeoGridSpec:
    return GeoGridSpec(
        crs="EPSG:32631",
        transform=(500_000.0, pixel_size_m, 0.0, 1_000.0, 0.0, -pixel_size_m),
        width=5,
        height=4,
        resolution_m=(pixel_size_m, pixel_size_m),
    )


def test_grid_lonlat_projected_is_finite() -> None:
    """UTM grid centers convert to finite lon/lat (FORWARD)."""
    grid = _projected_grid()
    lat, lon = grid_lonlat(grid)
    assert np.isfinite(lat).all()
    assert np.isfinite(lon).all()
    assert -1.0 < float(np.mean(lat)) < 1.0
    assert 2.0 < float(np.mean(lon)) < 4.0


def _carrier() -> TOPSCarrierModel:
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.005,
        range_sampling_rate_hz=32.0e6,
        azimuth_time_interval_s=0.002,
        doppler_centroid_hz=(20.0, 0.0, 0.0),
        doppler_t0_s=0.005,
        fm_rate_hz_s=(-2200.0, 0.0, 0.0),
        fm_t0_s=0.005,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.005,
        azimuth_steering_rate_hz_s=800.0,
    )


def test_compose_secondary_coordinates_uses_source_offset_convention() -> None:
    """Geo coordinates compose reference LUT and dense reference-minus-secondary."""
    from faninsar.processing.coreg.offsets import OffsetFieldResult
    from faninsar.processing.pipeline.geo_lut import Geo2RdrLUT

    lut = Geo2RdrLUT(
        az_full=np.array([[2.0, 3.0]], dtype=np.float64),
        rg_full=np.array([[4.0, 5.0]], dtype=np.float64),
        valid=np.ones((1, 2), dtype=bool),
        full_radar_shape=(8, 10),
        height_m=0.0,
    )
    offsets = OffsetFieldResult(
        range_offset_px=np.full((8, 10), -1.25, dtype=np.float32),
        azimuth_offset_px=np.full((8, 10), 0.5, dtype=np.float32),
        coverage=np.ones((8, 10), dtype=bool),
        uncertainty_px=np.zeros((8, 10), dtype=np.float32),
    )
    secondary_az, secondary_rg, valid = compose_secondary_coordinates(lut, offsets)
    assert np.allclose(secondary_az, [[1.5, 2.5]])
    assert np.allclose(secondary_rg, [[5.25, 6.25]])
    assert valid.all()


def test_coregister_geocoded_slcs_remaps_deramped_inputs_once() -> None:
    """Geo coregistration returns two reramped SLCs on the LUT grid."""
    from faninsar.processing.coreg.offsets import OffsetFieldResult
    from faninsar.processing.pipeline.geo_lut import Geo2RdrLUT

    shape = (8, 10)
    azimuth, range_index = np.meshgrid(
        np.arange(1.0, 7.0),
        np.arange(1.0, 9.0),
        indexing="ij",
    )
    lut = Geo2RdrLUT(
        az_full=azimuth,
        rg_full=range_index,
        valid=np.ones(azimuth.shape, dtype=bool),
        full_radar_shape=shape,
        height_m=0.0,
    )
    offsets = OffsetFieldResult(
        range_offset_px=np.zeros(shape, dtype=np.float32),
        azimuth_offset_px=np.zeros(shape, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )
    deramped = np.ones(shape, dtype=np.complex64)
    reference, secondary, valid = coregister_geocoded_slcs(
        deramped,
        deramped,
        reference_carrier=_carrier(),
        secondary_carrier=_carrier(),
        reference_lut=lut,
        offsets=offsets,
    )
    assert reference.shape == lut.shape
    assert secondary.shape == lut.shape
    assert valid.all()
    assert np.allclose(reference, secondary, atol=1e-6)


def test_shared_lut_apply_twice_same_shape() -> None:
    """Verify one LUT can resample independent complex fields."""
    geom = _toy_geometry((32, 64))
    grid = _projected_grid()
    lut = build_geo2rdr_lut(
        geometry=geom,
        grid=grid,
        full_radar_shape=(32, 64),
        height_m=0.0,
        chunk_size=2,
    )
    a = np.ones((32, 64), dtype=np.complex64)
    b = np.full((32, 64), 2 + 0j, dtype=np.complex64)
    ga, va = apply_lut_complex(a, lut)
    gb, vb = apply_lut_complex(b, lut)
    assert ga.shape == grid.shape
    assert gb.shape == grid.shape
    assert va.shape == vb.shape


def test_build_geo2rdr_lut_accepts_height_array() -> None:
    """Per-pixel height_m array is accepted (DEM-aware LUT path)."""
    from faninsar.processing.geometry import ConstantHeightDEM

    geom = _toy_geometry((32, 64))
    grid = _projected_grid()
    h = np.full(grid.shape, 1000.0, dtype=np.float64)
    h[0, 0] = 2000.0
    lut_arr = build_geo2rdr_lut(
        geometry=geom,
        grid=grid,
        full_radar_shape=(32, 64),
        height_m=h,
        chunk_size=2,
    )
    lut_dem = build_geo2rdr_lut(
        geometry=geom,
        grid=grid,
        full_radar_shape=(32, 64),
        height_m=1000.0,
        dem=ConstantHeightDEM(1000.0),
        chunk_size=2,
    )
    assert lut_arr.az_full.shape == grid.shape
    assert lut_dem.az_full.shape == grid.shape
    # DEM constant 1000 matches uniform height field
    m = lut_arr.valid & lut_dem.valid
    if m.any():
        assert float(np.nanmax(np.abs(lut_arr.az_full[m] - lut_dem.az_full[m]))) < 1e-6


def test_run_production_pair_requires_geo_grid_for_geo_mode(tmp_path: Path) -> None:
    """Geo coregistration rejects a missing grid before reading input data."""
    from faninsar.processing.errors import InvalidProcessingStateError
    from faninsar.processing.pipeline.production import run_production_pair

    with pytest.raises(InvalidProcessingStateError, match="geo_grid"):
        run_production_pair(
            "missing.zip",
            "missing2.zip",
            output_dir=tmp_path,
            coregistration_grid="geo",
            geo_grid=None,
        )


def test_run_production_pair_geo_mode_requires_burst_scope(tmp_path: Path) -> None:
    """Geo coregistration rejects a stitched swath before reading input data."""
    from faninsar.processing.errors import InvalidProcessingStateError
    from faninsar.processing.merge.grid import GeoGridSpec
    from faninsar.processing.pipeline.production import run_production_pair

    grid = GeoGridSpec(
        crs="EPSG:32647",
        transform=(0.0, 20.0, 0.0, 0.0, 0.0, -80.0),
        width=4,
        height=3,
        resolution_m=(20.0, 80.0),
    )
    with pytest.raises(InvalidProcessingStateError, match="scope='burst'"):
        run_production_pair(
            "missing.zip",
            "missing2.zip",
            output_dir=tmp_path,
            scope="swath",
            coregistration_grid="geo",
            geo_grid=grid,
        )
