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
from faninsar.processing.geocoding.geo_lut import (
    build_geo2rdr_lut,
    grid_lonlat,
    grid_lonlat_rows,
    roi_geo_bbox,
    roi_geo_mask,
)
from faninsar.processing.geocoding.geo_modes import (
    _apply_reramp,
    coregister_geocoded_slcs,
    coregister_geocoded_slcs_chunked,
)
from faninsar.processing.geocoding.geo_resample import (
    apply_lut_complex,
    compose_secondary_coordinates,
    resample_complex_at_coordinates,
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


def test_grid_lonlat_rows_matches_full_grid_slice() -> None:
    """Row-only coordinates match the corresponding full-grid rows."""
    grid = _projected_grid()
    full_latitude, full_longitude = grid_lonlat(grid)

    latitude, longitude = grid_lonlat_rows(grid, 1, 3)

    np.testing.assert_allclose(latitude, full_latitude[1:3])
    np.testing.assert_allclose(longitude, full_longitude[1:3])


def test_roi_geo_bbox_and_mask_cover_polygon_with_hole() -> None:
    """ROI helpers bound the polygon and exclude hole pixels."""
    from shapely.geometry import Polygon, box

    grid = _projected_grid()
    lat, lon = grid_lonlat(grid)
    outer = box(
        float(np.min(lon)),
        float(np.min(lat)),
        float(np.max(lon)),
        float(np.max(lat)),
    )
    hole = box(
        float(np.mean(lon)) - 0.002,
        float(np.mean(lat)) - 0.002,
        float(np.mean(lon)) + 0.002,
        float(np.mean(lat)) + 0.002,
    )
    polygon = Polygon(outer.exterior, holes=[hole.exterior])
    row0, row1, col0, col1 = roi_geo_bbox(polygon, grid, margin_px=0)
    assert (row0, row1, col0, col1) == (0, 4, 0, 5)
    mask = roi_geo_mask(polygon, grid, row0, row1, col0, col1)
    assert mask.shape == (4, 5)
    assert int(mask.sum()) < 20
    assert not mask[2, 2]
    dilated = roi_geo_mask(polygon, grid, row0, row1, col0, col1, dilate_px=1)
    assert dilated.sum() > mask.sum()


def test_build_geo2rdr_lut_with_roi_geometry_masks_outside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROI geometry prefilter keeps only converged pixels inside the ROI."""
    from shapely.geometry import box

    from faninsar.processing.geocoding import geo_lut

    def _fake_geo2rdr(
        _geometry: object,
        latitude: np.ndarray,
        _longitude: np.ndarray,
        _height: object,
        **_kwargs: object,
    ) -> object:
        class _FakeGeo2RdrResult:
            converged = np.ones(latitude.shape, dtype=bool)
            azimuth_index = np.full(latitude.shape, 10.0)
            range_index = np.full(latitude.shape, 20.0)

        return _FakeGeo2RdrResult()

    monkeypatch.setattr(geo_lut, "run_geo2rdr", _fake_geo2rdr)
    geom = _toy_geometry((32, 64))
    grid = _projected_grid()
    lat, lon = grid_lonlat(grid)
    roi = box(
        float(np.min(lon)),
        float(np.min(lat)),
        float(np.max(lon)),
        float(np.max(lat)),
    )
    expected = roi_geo_mask(roi, grid, 0, grid.height, 0, grid.width)
    lut = build_geo2rdr_lut(
        geometry=geom,
        grid=grid,
        full_radar_shape=(32, 64),
        height_m=0.0,
        device="cpu",
        chunk_size=2,
        roi_geometry=roi,
        polygon_dilate_px=0,
    )
    assert lut.valid.shape == grid.shape
    assert int(lut.valid.sum()) == int(expected.sum())
    half = box(
        float(np.min(lon)),
        float(np.min(lat)),
        float(np.max(lon)) - 0.002,
        float(np.max(lat)),
    )
    lut_half = build_geo2rdr_lut(
        geometry=geom,
        grid=grid,
        full_radar_shape=(32, 64),
        height_m=0.0,
        device="cpu",
        chunk_size=2,
        roi_geometry=half,
        polygon_dilate_px=0,
    )
    expected_half = roi_geo_mask(half, grid, 0, grid.height, 0, grid.width)
    assert int(lut_half.valid.sum()) == int(expected_half.sum())
    assert int(lut_half.valid.sum()) < int(expected.sum())


def test_geo2rdr_lut_retains_sampled_height() -> None:
    """Reuse the exact DEM samples needed by later geometric flattening."""
    grid = _projected_grid()
    lut = build_geo2rdr_lut(
        geometry=_toy_geometry(),
        grid=grid,
        full_radar_shape=(64, 128),
        height_m=123.5,
        device="cpu",
    )

    assert lut.height_full is not None
    np.testing.assert_array_equal(lut.height_full, np.full(grid.shape, 123.5))


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
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT

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


def test_compose_secondary_coordinates_torch_matches_numpy() -> None:
    """CUDA compose matches the SciPy bilinear oracle on interior samples."""
    pytest.importorskip("torch")
    from faninsar.processing.coreg.offsets import OffsetFieldResult
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
    from faninsar.processing.geocoding.geo_resample import (
        _compose_secondary_coordinates_torch,
    )

    rng = np.random.default_rng(0)
    radar_shape = (16, 20)
    az = rng.uniform(1.0, 14.0, size=(8, 10))
    rg = rng.uniform(1.0, 18.0, size=(8, 10))
    lut = Geo2RdrLUT(
        az_full=az,
        rg_full=rg,
        valid=np.ones((8, 10), dtype=bool),
        full_radar_shape=radar_shape,
        height_m=0.0,
    )
    offsets = OffsetFieldResult(
        range_offset_px=rng.normal(0.0, 0.4, size=radar_shape).astype(np.float32),
        azimuth_offset_px=rng.normal(0.0, 0.4, size=radar_shape).astype(np.float32),
        coverage=np.ones(radar_shape, dtype=bool),
        uncertainty_px=np.zeros(radar_shape, dtype=np.float32),
    )
    numpy_az, numpy_rg, numpy_valid = compose_secondary_coordinates(lut, offsets)
    torch_az, torch_rg, torch_valid = _compose_secondary_coordinates_torch(
        lut, offsets, "cpu"
    )
    assert numpy_valid.shape == torch_valid.shape
    common = numpy_valid & torch_valid
    assert common.any()
    assert np.allclose(numpy_az[common], torch_az[common], atol=1e-5, rtol=0.0)
    assert np.allclose(numpy_rg[common], torch_rg[common], atol=1e-5, rtol=0.0)


def test_coregister_geocoded_slcs_remaps_deramped_inputs_once() -> None:
    """Geo coregistration returns two reramped SLCs on the LUT grid."""
    from faninsar.processing.coreg.offsets import OffsetFieldResult
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT

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


def test_geo_reramp_keeps_nonqualified_stage_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Geographic reramp does not bypass the CUDA qualification registry."""
    from faninsar.backends import dask_gpu

    client = object()
    samples = np.ones((3, 4), dtype=np.complex64)

    def fail_remote(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError

    monkeypatch.setattr(dask_gpu, "should_accelerate", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(dask_gpu, "run_carrier_multiply_at_points", fail_remote)
    result = _apply_reramp(
        samples,
        _carrier(),
        np.zeros((3, 4)),
        np.zeros((3, 4)),
        native_height=8,
        device="auto",
        dask_client=client,
    )
    from faninsar.processing.tops.deramp import carrier_phase_at_points

    phase = carrier_phase_at_points(
        _carrier(),
        np.zeros((3, 4)),
        np.zeros((3, 4)),
        centre_row=4.0,
        dtype=np.float32,
    )
    expected = samples * np.exp(1j * np.asarray(phase, dtype=np.float64))
    np.testing.assert_allclose(result, expected.astype(np.complex64))


def test_chunked_coregistration_matches_full_result(tmp_path: Path) -> None:
    """Chunked Geo coregistration is numerically identical to the full path."""
    from faninsar.processing.coreg.offsets import OffsetFieldResult
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT

    shape = (12, 14)
    azimuth, range_index = np.meshgrid(
        np.arange(1.0, 11.0),
        np.arange(1.0, 13.0),
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
        range_offset_px=np.full(shape, -0.25, dtype=np.float32),
        azimuth_offset_px=np.full(shape, 0.125, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )
    rows, columns = np.indices(shape)
    deramped = np.exp(1j * (0.05 * rows + 0.02 * columns)).astype(np.complex64)
    expected_reference, expected_secondary, expected_valid = coregister_geocoded_slcs(
        deramped,
        deramped,
        reference_carrier=_carrier(),
        secondary_carrier=_carrier(),
        reference_lut=lut,
        offsets=offsets,
    )

    reference, secondary, valid = coregister_geocoded_slcs_chunked(
        deramped,
        deramped,
        reference_carrier=_carrier(),
        secondary_carrier=_carrier(),
        reference_lut=lut,
        offsets=offsets,
        output_dir=tmp_path,
        row_chunk=3,
    )

    assert isinstance(reference, np.memmap)
    assert isinstance(secondary, np.memmap)
    assert isinstance(valid, np.memmap)
    np.testing.assert_allclose(reference, expected_reference, atol=1e-6)
    np.testing.assert_allclose(secondary, expected_secondary, atol=1e-6)
    np.testing.assert_array_equal(valid, expected_valid)


def test_opencv_geo_resampling_preserves_smooth_complex_phase() -> None:
    """Keep the compiled Lanczos path phase-close on a bandlimited field."""
    pytest.importorskip("cv2")
    rows, columns = np.indices((64, 96), dtype=np.float64)
    source = np.exp(1j * (0.04 * rows + 0.015 * columns)).astype(np.complex64)
    azimuth = rows[4:-4, 4:-4] + 0.37
    range_index = columns[4:-4, 4:-4] + 0.61
    serial, serial_valid = resample_complex_at_coordinates(
        source,
        azimuth,
        range_index,
        executor="torch",
    )
    accelerated, accelerated_valid = resample_complex_at_coordinates(
        source,
        azimuth,
        range_index,
        executor="opencv",
    )

    np.testing.assert_array_equal(accelerated_valid, serial_valid)
    phase_delta = np.angle(accelerated * np.conj(serial))
    assert float(np.max(np.abs(phase_delta))) < 1e-3


def test_shared_lut_apply_twice_same_shape() -> None:
    """Verify one LUT can resample independent complex fields."""
    geom = _toy_geometry((32, 64))
    grid = _projected_grid()
    lut = build_geo2rdr_lut(
        geometry=geom,
        grid=grid,
        full_radar_shape=(32, 64),
        height_m=0.0,
        device="cpu",
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
    from faninsar.processing.dem import ConstantDEM

    geom = _toy_geometry((32, 64))
    grid = _projected_grid()
    h = np.full(grid.shape, 1000.0, dtype=np.float64)
    h[0, 0] = 2000.0
    lut_arr = build_geo2rdr_lut(
        geometry=geom,
        grid=grid,
        full_radar_shape=(32, 64),
        height_m=h,
        device="cpu",
        chunk_size=2,
    )
    lut_dem = build_geo2rdr_lut(
        geometry=geom,
        grid=grid,
        full_radar_shape=(32, 64),
        height_m=1000.0,
        dem=ConstantDEM(1000.0),
        device="cpu",
        chunk_size=2,
    )
    assert lut_arr.az_full.shape == grid.shape
    assert lut_dem.az_full.shape == grid.shape
    # DEM constant 1000 matches uniform height field
    m = lut_arr.valid & lut_dem.valid
    if m.any():
        assert float(np.nanmax(np.abs(lut_arr.az_full[m] - lut_dem.az_full[m]))) < 1e-6


def test_build_geo2rdr_lut_can_use_disk_backed_arrays(tmp_path: Path) -> None:
    """A disk-backed LUT keeps complete coordinates without resident arrays."""
    geom = _toy_geometry((32, 64))
    grid = _projected_grid()

    lut = build_geo2rdr_lut(
        geometry=geom,
        grid=grid,
        full_radar_shape=(32, 64),
        height_m=0.0,
        device="cpu",
        chunk_size=2,
        storage_dir=tmp_path,
    )

    assert isinstance(lut.az_full, np.memmap)
    assert isinstance(lut.rg_full, np.memmap)
    assert isinstance(lut.valid, np.memmap)
    assert lut.shape == grid.shape
