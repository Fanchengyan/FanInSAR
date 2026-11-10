"""Regression: geometry-mode geo secondary samples at direct geo2rdr coordinates.

The Reference geographic LUT already holds a ground solution (longitude,
latitude, height) for every destination pixel. Geometry-mode coregistration
solves the secondary radar coordinates at those same ground points. The dense
radar offset field is an rdr2geo inverse; where that inverse fails its
coverage is False, and it must not veto a pixel whose ground solution and
secondary samples are both valid.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
from pyproj import Transformer

import faninsar.missions.s1.processing as production_mod
from faninsar.missions.s1.processing import ProductionPairState, stage_coregister
from faninsar.processing.coregistration.offsets import OffsetFieldResult
from faninsar.processing.coregistration.tops.deramp import (
    TOPSCarrierModel,
    carrier_phase_at_points,
    deramp,
)
from faninsar.processing.geocoding.coordinates import RadarCoordinates
from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
from faninsar.processing.geocoding.geo_modes import (
    coregister_geocoded_slcs,
    coregister_geocoded_slcs_chunked,
    geocode_reference_slc,
    geocode_secondary_slc,
)
from faninsar.processing.geometry import ConstantDEM
from faninsar.processing.mosaicking.grid import GeoGridSpec

if TYPE_CHECKING:
    from pathlib import Path

RADAR_SHAPE = (24, 28)
GEO_SHAPE = (6, 8)
AZ_OFF = 0.35
RG_OFF = -0.42
AZ_AXIS = (6.3, 15.7)
RG_AXIS = (7.4, 19.6)
GRID_CRS = "EPSG:32631"
GRID_X0 = 500_000.0
GRID_Y0 = 1_000.0
GRID_STEP = 30.0
TOPO_PHASE = 0.3
# Radar cells where the rdr2geo-based dense field has no coverage, and the
# geographic pixels whose Reference coordinates fall inside them.
DENSE_HOLE = (slice(9, 13), slice(11, 16))
GEO_HOLE = (slice(2, 4), slice(2, 5))
# Native secondary samples that are genuinely invalid.
BAD_SOURCE = (slice(17, 19), slice(20, 23))
# Geographic pixels the synthetic secondary geo2rdr cannot place.
OUT_OF_SOURCE = (0, 0)
NOT_CONVERGED = (0, 7)


def _carrier(*, f_dc: float, k_a: float = -2200.0) -> TOPSCarrierModel:
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.005,
        range_sampling_rate_hz=32.0e6,
        azimuth_time_interval_s=0.002,
        doppler_centroid_hz=(f_dc, 0.0, 0.0),
        doppler_t0_s=0.005,
        fm_rate_hz_s=(k_a, 0.0, 0.0),
        fm_t0_s=0.005,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.005,
        azimuth_steering_rate_hz_s=800.0,
    )


def _geo_grid() -> GeoGridSpec:
    return GeoGridSpec(
        crs=GRID_CRS,
        transform=(GRID_X0, GRID_STEP, 0.0, GRID_Y0, 0.0, -GRID_STEP),
        width=GEO_SHAPE[1],
        height=GEO_SHAPE[0],
        resolution_m=(GRID_STEP, GRID_STEP),
    )


def _lut() -> Geo2RdrLUT:
    azimuth, range_index = np.meshgrid(
        np.linspace(*AZ_AXIS, GEO_SHAPE[0], dtype=np.float64),
        np.linspace(*RG_AXIS, GEO_SHAPE[1], dtype=np.float64),
        indexing="ij",
    )
    rows, cols = np.indices(GEO_SHAPE, dtype=np.float64)
    return Geo2RdrLUT(
        az_full=azimuth,
        rg_full=range_index,
        valid=np.ones(GEO_SHAPE, dtype=bool),
        full_radar_shape=RADAR_SHAPE,
        height_m=0.0,
        height_full=100.0 + 10.0 * rows + cols,
        row0=0,
        col0=0,
    )


def _offsets(
    *,
    azimuth: float = AZ_OFF,
    range_: float = RG_OFF,
    holed: bool = False,
) -> OffsetFieldResult:
    coverage = np.ones(RADAR_SHAPE, dtype=bool)
    if holed:
        coverage[DENSE_HOLE] = False
    return OffsetFieldResult(
        range_offset_px=np.full(RADAR_SHAPE, range_, dtype=np.float32),
        azimuth_offset_px=np.full(RADAR_SHAPE, azimuth, dtype=np.float32),
        coverage=coverage,
        uncertainty_px=np.zeros(RADAR_SHAPE, dtype=np.float32),
    )


def _native_slc(carrier: TOPSCarrierModel) -> np.ndarray:
    rows, cols = np.indices(RADAR_SHAPE, dtype=np.float64)
    baseband = np.exp(1j * (0.3 * rows + 0.2 * cols)).astype(np.complex64)
    phase = carrier_phase_at_points(
        carrier,
        rows,
        cols,
        centre_row=float(RADAR_SHAPE[0] // 2),
        dtype=np.float64,
    )
    return (baseband * np.exp(1j * phase)).astype(np.complex64)


def _scene(
    carrier: TOPSCarrierModel,
    samples: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> MagicMock:
    scene = MagicMock()
    scene.array.samples = samples
    scene.array.valid_mask = (
        np.ones(RADAR_SHAPE, dtype=bool) if valid_mask is None else valid_mask
    )
    scene.array.row0 = 0
    scene.array.col0 = 0
    scene.carrier = carrier
    scene.scene_id = f"scene_{carrier.doppler_centroid_hz[0]:+.0f}"
    scene.swath.swath = "IW1"
    scene.burst.index = 0
    scene.geometry.range_spacing_m = 2.3
    scene.geometry.wavelength_m = 0.0555
    return scene


class _GroundToSecondary:
    """Synthetic secondary geo2rdr over the test grid that records its inputs."""

    def __init__(self, *, faults: bool) -> None:
        self.faults = faults
        self.models: list[object] = []
        self.heights: list[np.ndarray] = []
        self._to_grid = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)

    def __call__(
        self,
        model: object,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
        height_m: np.ndarray,
        **_kwargs: object,
    ) -> SimpleNamespace:
        shape = np.shape(latitude_deg)
        x, y = self._to_grid.transform(
            np.ravel(longitude_deg),
            np.ravel(latitude_deg),
        )
        col = ((np.asarray(x) - GRID_X0) / GRID_STEP - 0.5).reshape(shape)
        row = ((GRID_Y0 - np.asarray(y)) / GRID_STEP - 0.5).reshape(shape)
        az_step = (AZ_AXIS[1] - AZ_AXIS[0]) / (GEO_SHAPE[0] - 1)
        rg_step = (RG_AXIS[1] - RG_AXIS[0]) / (GEO_SHAPE[1] - 1)
        azimuth = AZ_AXIS[0] + row * az_step - AZ_OFF
        range_index = RG_AXIS[0] + col * rg_step - RG_OFF
        converged = np.ones(shape, dtype=bool)
        if self.faults:
            pixel_row = np.rint(row).astype(int)
            pixel_col = np.rint(col).astype(int)
            azimuth[
                (pixel_row == OUT_OF_SOURCE[0]) & (pixel_col == OUT_OF_SOURCE[1])
            ] = -5.0
            converged[
                (pixel_row == NOT_CONVERGED[0]) & (pixel_col == NOT_CONVERGED[1])
            ] = False
        self.models.append(model)
        self.heights.append(np.array(height_m, dtype=np.float64))
        return SimpleNamespace(
            azimuth_index=azimuth,
            range_index=range_index,
            converged=converged,
        )


def _constant_phase(phase: float, azimuths: list[np.ndarray]) -> object:
    def geometric_phase(*args: object, **_kwargs: object) -> np.ndarray:
        azimuths.append(np.array(args[6], dtype=np.float64))
        return np.full(np.shape(args[2]), phase, dtype=np.float64)

    return geometric_phase


def test_stage_geo_geometry_mode_samples_direct_coordinates_through_dense_hole(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dense-coverage hole no longer invalidates solved ground pixels."""
    ref_carrier = _carrier(f_dc=12.0)
    sec_carrier = _carrier(f_dc=-18.0, k_a=-2100.0)
    ref_native = _native_slc(ref_carrier)
    sec_native = _native_slc(sec_carrier)
    ref_deramped = deramp(ref_native, ref_carrier)
    sec_deramped = deramp(sec_native, sec_carrier)
    source_valid = np.ones(RADAR_SHAPE, dtype=bool)
    source_valid[BAD_SOURCE] = False
    lut = _lut()
    solver = _GroundToSecondary(faults=True)
    dense_calls: list[object] = []
    azimuths: list[np.ndarray] = []

    def holed_dense(**kwargs: object) -> OffsetFieldResult:
        dense_calls.append(kwargs)
        return _offsets(holed=True)

    monkeypatch.setattr(production_mod, "dense_geometry_offsets", holed_dense)
    monkeypatch.setattr(production_mod, "run_geo2rdr", solver)
    monkeypatch.setattr(
        production_mod,
        "compute_geometric_phase_from_geo",
        _constant_phase(TOPO_PHASE, azimuths),
    )
    sec_scene = _scene(sec_carrier, sec_native, source_valid)
    state = ProductionPairState(
        pair_id="geo_direct_hole",
        primary=_scene(ref_carrier, ref_native),
        secondary=sec_scene,
        dem=ConstantDEM(0.0),
        coregistration_grid="geo",
    )
    state.primary_deramped = ref_deramped
    state.secondary_deramped = sec_deramped

    result = stage_coregister(
        state,
        device="cpu",
        coregistration_grid="geo",
        geo_grid=_geo_grid(),
        geo_work_dir=tmp_path / "geo_work",
        geo_chunk_size=2,
        prepared_geo_lut=lut,
    )

    geo_ones = np.ones(GEO_SHAPE, dtype=bool)
    # The same coordinates expressed as a hole-free constant offset field.
    oracle, oracle_valid = geocode_secondary_slc(
        sec_deramped,
        secondary_carrier=sec_carrier,
        reference_lut=lut,
        reference_valid=geo_ones,
        offsets=_offsets(),
        device="cpu",
        source_valid=source_valid,
    )
    _, vetoed_valid = geocode_secondary_slc(
        sec_deramped,
        secondary_carrier=sec_carrier,
        reference_lut=lut,
        reference_valid=geo_ones,
        offsets=_offsets(holed=True),
        device="cpu",
        source_valid=source_valid,
    )
    expected_ref, expected_ref_valid = geocode_reference_slc(
        ref_deramped,
        reference_carrier=ref_carrier,
        reference_lut=lut,
        device="cpu",
        source_valid=np.ones(RADAR_SHAPE, dtype=bool),
    )
    assert oracle_valid[GEO_HOLE].all()
    assert not vetoed_valid[GEO_HOLE].any()
    assert not oracle_valid[4:, 5:].any()
    expected_valid = oracle_valid & expected_ref_valid
    expected_valid[OUT_OF_SOURCE] = False
    expected_valid[NOT_CONVERGED] = False

    got = np.asarray(result.secondary_aligned)
    got_valid = np.asarray(result.geocoded_slc_valid, dtype=bool)
    assert got_valid[GEO_HOLE].all()
    np.testing.assert_array_equal(got_valid, expected_valid)
    np.testing.assert_allclose(
        got[expected_valid],
        (oracle * np.exp(1j * TOPO_PHASE))[expected_valid],
        atol=2e-5,
    )
    # Invalid native samples, out-of-source and unconverged pixels stay NaN.
    assert np.isnan(got[~expected_valid].real).all()
    assert int((~expected_valid).sum()) >= 8

    # The flattening reuses the exact secondary azimuth that was sampled.
    np.testing.assert_allclose(
        np.concatenate(azimuths)[expected_valid],
        (lut.az_full - AZ_OFF)[expected_valid],
        atol=1e-6,
    )
    # Ground points are the Reference LUT's own, solved on the secondary.
    assert solver.models
    assert all(model is sec_scene.geometry for model in solver.models)
    np.testing.assert_allclose(np.concatenate(solver.heights), lut.height_full)
    # The Reference payload is untouched.
    np.testing.assert_allclose(
        np.asarray(result.primary_geocoded_slc)[expected_ref_valid],
        expected_ref[expected_ref_valid],
        atol=2e-5,
    )
    assert dense_calls == []
    assert "secondary_geo2rdr" in result.coregistration_timings_s
    assert "dense_geometry_offsets" not in result.coregistration_timings_s


def test_stage_geo_direct_coordinates_subtract_constant_residuals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Residual constants keep ``source = output - offset`` on direct coordinates."""
    misreg_az, misreg_rg, esd_az, ampcor_rg = -1.0, 1.0, 0.25, -0.5
    total_az = AZ_OFF + misreg_az + esd_az
    total_rg = RG_OFF + misreg_rg + ampcor_rg
    ref_carrier = _carrier(f_dc=12.0)
    sec_carrier = _carrier(f_dc=-18.0, k_a=-2100.0)
    sec_native = _native_slc(sec_carrier)
    sec_deramped = deramp(sec_native, sec_carrier)
    lut = _lut()
    dense_calls: list[object] = []

    def holed_dense(**kwargs: object) -> OffsetFieldResult:
        dense_calls.append(kwargs)
        return _offsets(holed=True)

    monkeypatch.setattr(production_mod, "dense_geometry_offsets", holed_dense)
    monkeypatch.setattr(production_mod, "run_geo2rdr", _GroundToSecondary(faults=False))
    monkeypatch.setattr(
        production_mod,
        "compute_geometric_phase_from_geo",
        _constant_phase(0.0, []),
    )
    state = ProductionPairState(
        pair_id="geo_direct_signs",
        primary=_scene(ref_carrier, _native_slc(ref_carrier)),
        secondary=_scene(sec_carrier, sec_native),
        dem=ConstantDEM(0.0),
        coregistration_grid="geo",
    )
    state.secondary_deramped = sec_deramped

    result = stage_coregister(
        state,
        device="cpu",
        coregistration_grid="geo",
        geo_grid=_geo_grid(),
        geo_work_dir=tmp_path / "geo_work",
        geo_chunk_size=4,
        prepared_geo_lut=lut,
        secondary_only=True,
        reference_shape=RADAR_SHAPE,
        reference_geo_valid=np.ones(GEO_SHAPE, dtype=bool),
        misreg_az_px=misreg_az,
        misreg_rg_px=misreg_rg,
        force_esd_azimuth_shift_px=esd_az,
        force_amplitude_residual_rg=ampcor_rg,
    )

    def sampled(azimuth: float, range_: float) -> tuple[np.ndarray, np.ndarray]:
        return geocode_secondary_slc(
            sec_deramped,
            secondary_carrier=sec_carrier,
            reference_lut=lut,
            reference_valid=np.ones(GEO_SHAPE, dtype=bool),
            offsets=_offsets(azimuth=azimuth, range_=range_),
            device="cpu",
            source_valid=np.ones(RADAR_SHAPE, dtype=bool),
        )

    expected, expected_valid = sampled(total_az, total_rg)
    wrong, wrong_valid = sampled(
        AZ_OFF - misreg_az - esd_az,
        RG_OFF - misreg_rg - ampcor_rg,
    )
    assert expected_valid.all()
    assert wrong_valid.all()
    got = np.asarray(result.secondary_aligned)
    np.testing.assert_array_equal(
        np.asarray(result.geocoded_slc_valid, dtype=bool), expected_valid
    )
    np.testing.assert_allclose(got, expected, atol=2e-5)
    assert float(np.abs(got - wrong).max()) > 0.1
    # Reported shifts are the applied Reference-minus-secondary totals, once.
    assert result.azimuth_shift_px == pytest.approx(total_az, abs=1e-5)
    assert result.range_shift_px == pytest.approx(total_rg, abs=1e-5)
    assert dense_calls == []


def test_injected_secondary_coordinates_agree_between_full_and_chunked(
    tmp_path: Path,
) -> None:
    """Full and row-tiled entry points sample one injected coordinate field."""
    ref_carrier = _carrier(f_dc=12.0)
    sec_carrier = _carrier(f_dc=-18.0, k_a=-2100.0)
    ref_deramped = deramp(_native_slc(ref_carrier), ref_carrier)
    sec_deramped = deramp(_native_slc(sec_carrier), sec_carrier)
    lut = _lut()
    coordinates = RadarCoordinates(
        lut.az_full - AZ_OFF,
        lut.rg_full - RG_OFF,
        np.ones(GEO_SHAPE, dtype=bool),
    )

    def full(secondary_coordinates: RadarCoordinates | None) -> tuple[np.ndarray, ...]:
        return coregister_geocoded_slcs(
            ref_deramped,
            sec_deramped,
            reference_carrier=ref_carrier,
            secondary_carrier=sec_carrier,
            reference_lut=lut,
            offsets=_offsets(holed=True),
            device="cpu",
            secondary_coordinates=secondary_coordinates,
        )

    direct = full(coordinates)
    composed = full(None)
    tiled = coregister_geocoded_slcs_chunked(
        ref_deramped,
        sec_deramped,
        reference_carrier=ref_carrier,
        secondary_carrier=sec_carrier,
        reference_lut=lut,
        offsets=_offsets(holed=True),
        output_dir=tmp_path / "chunked",
        row_chunk=2,
        device="cpu",
        secondary_coordinates=coordinates,
    )

    assert direct[2].all()
    # ``None`` keeps the composed-offset default, including its coverage veto.
    assert not composed[2][GEO_HOLE].any()
    np.testing.assert_array_equal(np.asarray(tiled[2], dtype=bool), direct[2])
    for whole, part in zip(direct[:2], tiled[:2], strict=True):
        np.testing.assert_allclose(np.asarray(part), whole, atol=1e-6, equal_nan=True)
