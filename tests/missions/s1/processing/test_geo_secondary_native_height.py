"""Regression: geo secondary reramp must use pre-conform native height.

``stage_coregister`` trailing-edge conforms an unequal secondary onto the
Reference shape. The TOPS carrier centre is ``native_height // 2`` of the
array that was deramped, not the conformed shape and not a geo-chunk height.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
from pyproj import Transformer

from faninsar.missions.s1.processing import ProductionPairState, stage_coregister
from faninsar.processing.coregistration.offsets import OffsetFieldResult
from faninsar.processing.coregistration.tops.deramp import (
    TOPSCarrierModel,
    carrier_phase_at_points,
    deramp,
)
from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
from faninsar.processing.geocoding.geo_modes import (
    geocode_reference_slc,
    geocode_secondary_slc,
    geocode_secondary_slc_chunked,
)
from faninsar.processing.geometry import ConstantDEM
from faninsar.processing.mosaicking.grid import GeoGridSpec

if TYPE_CHECKING:
    import pytest

REF_SHAPE = (16, 20)
SEC_SHAPE = (12, 18)
GEO_SHAPE = (5, 7)
AZ_OFF = 0.35
RG_OFF = -0.42


def _carrier(*, f_dc: float = 20.0, k_a: float = -2200.0) -> TOPSCarrierModel:
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
        crs="EPSG:32631",
        transform=(500_000.0, 30.0, 0.0, 1_000.0, 0.0, -30.0),
        width=GEO_SHAPE[1],
        height=GEO_SHAPE[0],
        resolution_m=(30.0, 30.0),
    )


def _lut() -> Geo2RdrLUT:
    rows = np.linspace(2.3, 8.7, GEO_SHAPE[0], dtype=np.float64)
    cols = np.linspace(3.4, 12.6, GEO_SHAPE[1], dtype=np.float64)
    azimuth, range_index = np.meshgrid(rows, cols, indexing="ij")
    return Geo2RdrLUT(
        az_full=azimuth,
        rg_full=range_index,
        valid=np.ones(GEO_SHAPE, dtype=bool),
        full_radar_shape=REF_SHAPE,
        height_m=0.0,
        height_full=np.zeros(GEO_SHAPE, dtype=np.float32),
        row0=0,
        col0=0,
    )


def _offsets(shape: tuple[int, int] = REF_SHAPE) -> OffsetFieldResult:
    return OffsetFieldResult(
        range_offset_px=np.full(shape, RG_OFF, dtype=np.float32),
        azimuth_offset_px=np.full(shape, AZ_OFF, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )


def _native_slc(shape: tuple[int, int], carrier: TOPSCarrierModel) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float64)
    baseband = np.exp(1j * (0.04 * rows + 0.015 * cols)).astype(np.complex64)
    phase = carrier_phase_at_points(
        carrier,
        rows,
        cols,
        centre_row=float(shape[0] // 2),
        dtype=np.float64,
    )
    return (baseband * np.exp(1j * phase)).astype(np.complex64)


def _mock_scene(
    shape: tuple[int, int],
    carrier: TOPSCarrierModel,
    samples: np.ndarray,
) -> MagicMock:
    scene = MagicMock()
    scene.array.samples = samples
    scene.array.valid_mask = np.ones(shape, dtype=bool)
    scene.array.row0 = 0
    scene.array.col0 = 0
    scene.carrier = carrier
    scene.scene_id = f"scene_{shape[0]}x{shape[1]}"
    scene.swath.swath = "IW1"
    scene.burst.index = 0
    scene.geometry.range_spacing_m = 2.3
    scene.geometry.wavelength_m = 0.0555
    return scene


def _direct_geo2rdr(
    _model: object,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    _height_m: np.ndarray,
    **_kwargs: object,
) -> SimpleNamespace:
    """Synthetic secondary geo2rdr: the LUT coordinates minus the offsets.

    Geometry-mode geo coregistration solves the secondary at the LUT ground
    points instead of composing the dense offset field, so the stage tests
    supply the equivalent direct solution.
    """
    shape = np.shape(latitude_deg)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform(
        np.ravel(longitude_deg),
        np.ravel(latitude_deg),
    )
    col = ((np.asarray(x) - 500_000.0) / 30.0 - 0.5).reshape(shape)
    row = ((1_000.0 - np.asarray(y)) / 30.0 - 0.5).reshape(shape)
    return SimpleNamespace(
        azimuth_index=2.3 + row * (8.7 - 2.3) / (GEO_SHAPE[0] - 1) - AZ_OFF,
        range_index=3.4 + col * (12.6 - 3.4) / (GEO_SHAPE[1] - 1) - RG_OFF,
        converged=np.ones(shape, dtype=bool),
    )


def _no_topo(
    _state: object,
    lut: Geo2RdrLUT,
    *_args: object,
    output_dir: object,
    **_kwargs: object,
) -> tuple[np.memmap, np.ndarray]:
    from pathlib import Path

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    phase = np.memmap(
        directory / "topo_phase.float32",
        mode="w+",
        dtype=np.float32,
        shape=lut.shape,
    )
    phase[:] = 0.0
    phase.flush()
    return phase, np.zeros(lut.shape, dtype=np.float32)


def test_geocode_secondary_uses_explicit_pre_conform_native_height() -> None:
    """Conformed secondary still reramps about the deramped native centre."""
    sec_carrier = _carrier(f_dc=-18.0, k_a=-2100.0)
    lut = _lut()
    offsets = _offsets()
    sec_native = deramp(_native_slc(SEC_SHAPE, sec_carrier), sec_carrier)
    conformed = np.zeros(REF_SHAPE, dtype=np.complex64)
    conformed[: SEC_SHAPE[0], : SEC_SHAPE[1]] = sec_native
    source_valid = np.zeros(REF_SHAPE, dtype=bool)
    source_valid[: SEC_SHAPE[0], : SEC_SHAPE[1]] = True

    own, own_valid = geocode_secondary_slc(
        sec_native,
        secondary_carrier=sec_carrier,
        reference_lut=lut,
        reference_valid=np.ones(GEO_SHAPE, dtype=bool),
        offsets=offsets,
        device="cpu",
        source_valid=np.ones(SEC_SHAPE, dtype=bool),
    )
    threaded, threaded_valid = geocode_secondary_slc(
        conformed,
        secondary_carrier=sec_carrier,
        reference_lut=lut,
        reference_valid=np.ones(GEO_SHAPE, dtype=bool),
        offsets=offsets,
        device="cpu",
        source_valid=source_valid,
        secondary_native_height=SEC_SHAPE[0],
    )
    wrong, _ = geocode_secondary_slc(
        conformed,
        secondary_carrier=sec_carrier,
        reference_lut=lut,
        reference_valid=np.ones(GEO_SHAPE, dtype=bool),
        offsets=offsets,
        device="cpu",
        source_valid=source_valid,
    )
    mask = own_valid & threaded_valid
    assert int(mask.sum()) > 8
    np.testing.assert_allclose(threaded[mask], own[mask], atol=2e-5)
    delta = np.angle(wrong[mask] * np.conjugate(own[mask]))
    assert float(np.sqrt(np.mean(delta**2))) > 0.05


def test_chunked_geocode_secondary_threads_native_height(tmp_path: object) -> None:
    """Chunked geo secondary must not take the carrier centre from the tile."""
    sec_carrier = _carrier(f_dc=-18.0, k_a=-2100.0)
    lut = _lut()
    offsets = _offsets()
    sec_native = deramp(_native_slc(SEC_SHAPE, sec_carrier), sec_carrier)
    conformed = np.zeros(REF_SHAPE, dtype=np.complex64)
    conformed[: SEC_SHAPE[0], : SEC_SHAPE[1]] = sec_native
    source_valid = np.zeros(REF_SHAPE, dtype=bool)
    source_valid[: SEC_SHAPE[0], : SEC_SHAPE[1]] = True
    expected, expected_valid = geocode_secondary_slc(
        conformed,
        secondary_carrier=sec_carrier,
        reference_lut=lut,
        reference_valid=np.ones(GEO_SHAPE, dtype=bool),
        offsets=offsets,
        device="cpu",
        source_valid=source_valid,
        secondary_native_height=SEC_SHAPE[0],
    )
    chunked, chunked_valid = geocode_secondary_slc_chunked(
        conformed,
        secondary_carrier=sec_carrier,
        reference_lut=lut,
        reference_valid=np.ones(GEO_SHAPE, dtype=bool),
        offsets=offsets,
        output_dir=tmp_path / "sec_chunked",
        row_chunk=2,
        device="cpu",
        source_valid=source_valid,
        secondary_native_height=SEC_SHAPE[0],
    )
    mask = expected_valid & np.asarray(chunked_valid, dtype=bool)
    np.testing.assert_allclose(np.asarray(chunked)[mask], expected[mask], atol=1e-6)


def test_stage_coregister_geo_reramps_secondary_at_own_pre_conform_centre(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual ``stage_coregister`` geo path after unequal-height conform."""
    import faninsar.missions.s1.processing as production_mod
    from faninsar.processing.geocoding import geo_modes

    ref_carrier = _carrier(f_dc=12.0)
    sec_carrier = _carrier(f_dc=-18.0, k_a=-2100.0)
    ref_native = _native_slc(REF_SHAPE, ref_carrier)
    sec_native = _native_slc(SEC_SHAPE, sec_carrier)
    ref_deramped = deramp(ref_native, ref_carrier)
    sec_deramped = deramp(sec_native, sec_carrier)
    lut = _lut()
    offsets = _offsets()
    captured: dict[str, object] = {}
    real_secondary = geo_modes.geocode_secondary_slc_chunked

    def wrapped_secondary(samples: np.ndarray, **kwargs: object) -> object:
        captured["shape"] = tuple(samples.shape)
        captured["native_height"] = kwargs.get("secondary_native_height")
        captured["source_valid_shape"] = tuple(np.shape(kwargs["source_valid"]))
        return real_secondary(samples, **kwargs)

    monkeypatch.setattr(geo_modes, "geocode_secondary_slc_chunked", wrapped_secondary)
    monkeypatch.setattr(production_mod, "run_geo2rdr", _direct_geo2rdr)
    monkeypatch.setattr(
        production_mod, "dense_geometry_offsets", lambda **_kwargs: offsets
    )
    monkeypatch.setattr(
        production_mod, "combine_offset_fields", lambda field, **_kwargs: field
    )
    monkeypatch.setattr(
        production_mod, "_apply_geo_topographic_phase_chunked", _no_topo
    )

    ref_scene = _mock_scene(REF_SHAPE, ref_carrier, ref_native)
    sec_scene = _mock_scene(SEC_SHAPE, sec_carrier, sec_native)
    state = ProductionPairState(
        pair_id="unequal_geo_centre",
        primary=ref_scene,
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
        secondary_only=True,
        reference_shape=REF_SHAPE,
        reference_geo_valid=np.ones(GEO_SHAPE, dtype=bool),
    )

    assert captured["shape"] == REF_SHAPE
    assert captured["native_height"] == SEC_SHAPE[0]
    assert captured["source_valid_shape"] == REF_SHAPE
    assert result.secondary_aligned is not None

    expected, expected_valid = geocode_secondary_slc(
        sec_deramped,
        secondary_carrier=sec_carrier,
        reference_lut=lut,
        reference_valid=np.ones(GEO_SHAPE, dtype=bool),
        offsets=offsets,
        device="cpu",
        source_valid=np.ones(SEC_SHAPE, dtype=bool),
    )
    wrong, _ = geocode_secondary_slc(
        np.pad(
            sec_deramped,
            ((0, REF_SHAPE[0] - SEC_SHAPE[0]), (0, REF_SHAPE[1] - SEC_SHAPE[1])),
        ),
        secondary_carrier=sec_carrier,
        reference_lut=lut,
        reference_valid=np.ones(GEO_SHAPE, dtype=bool),
        offsets=offsets,
        device="cpu",
        source_valid=np.pad(
            np.ones(SEC_SHAPE, dtype=bool),
            ((0, REF_SHAPE[0] - SEC_SHAPE[0]), (0, REF_SHAPE[1] - SEC_SHAPE[1])),
        ),
    )
    got = np.asarray(result.secondary_aligned)
    mask = expected_valid & np.isfinite(got.real)
    assert int(mask.sum()) > 8
    np.testing.assert_allclose(got[mask], expected[mask], atol=2e-5)
    delta_wrong = np.angle(wrong[mask] * np.conjugate(expected[mask]))
    assert float(np.sqrt(np.mean(delta_wrong**2))) > 0.05


def test_stage_coregister_geo_pair_keeps_reference_own_centre(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference geo reramp stays on the Reference native height."""
    import faninsar.missions.s1.processing as production_mod

    ref_carrier = _carrier(f_dc=12.0)
    sec_carrier = _carrier(f_dc=-18.0, k_a=-2100.0)
    ref_native = _native_slc(REF_SHAPE, ref_carrier)
    sec_native = _native_slc(SEC_SHAPE, sec_carrier)
    lut = _lut()
    offsets = _offsets()
    monkeypatch.setattr(production_mod, "run_geo2rdr", _direct_geo2rdr)
    monkeypatch.setattr(
        production_mod, "dense_geometry_offsets", lambda **_kwargs: offsets
    )
    monkeypatch.setattr(
        production_mod, "combine_offset_fields", lambda field, **_kwargs: field
    )
    monkeypatch.setattr(
        production_mod, "_apply_geo_topographic_phase_chunked", _no_topo
    )
    state = ProductionPairState(
        pair_id="unequal_geo_pair",
        primary=_mock_scene(REF_SHAPE, ref_carrier, ref_native),
        secondary=_mock_scene(SEC_SHAPE, sec_carrier, sec_native),
        dem=ConstantDEM(0.0),
        coregistration_grid="geo",
    )
    ref_deramped = deramp(ref_native, ref_carrier)
    state.primary_deramped = ref_deramped
    state.secondary_deramped = deramp(sec_native, sec_carrier)
    result = stage_coregister(
        state,
        device="cpu",
        coregistration_grid="geo",
        geo_grid=_geo_grid(),
        geo_work_dir=tmp_path / "geo_pair",
        geo_chunk_size=2,
        prepared_geo_lut=lut,
    )
    expected_ref, _ = geocode_reference_slc(
        ref_deramped,
        reference_carrier=ref_carrier,
        reference_lut=lut,
        device="cpu",
        source_valid=np.ones(REF_SHAPE, dtype=bool),
    )
    got = np.asarray(result.primary_geocoded_slc)
    mask = np.isfinite(expected_ref.real) & np.isfinite(got.real)
    np.testing.assert_allclose(got[mask], expected_ref[mask], atol=2e-5)
