"""Precise Polygon scientific support at the coregister publication seam (P0074)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.data.query import BoundingBox
from faninsar.data.query.polygons import Polygons
from faninsar.io.storage.scene_store import CoregisteredSceneStore
from faninsar.missions.protocols import DopplerCentroidPolynomial
from faninsar.missions.s1 import coregister as coreg_module
from faninsar.missions.s1.coregister import (
    S1ReferenceContext,
    coregister_s1_secondary,
    prepare_s1_reference,
)
from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
from faninsar.processing.geometry import ConstantDEM, RadarGeometryModel
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.mosaicking.grid import GeoGridSpec

REFERENCE = "S1A_IW_SLC__1SDV_20200101T000000.SAFE"
SECONDARY = "S1A_IW_SLC__1SDV_20200113T000000.SAFE"
_ORIGIN = datetime(2020, 1, 1, tzinfo=UTC)
_LINES = 8
_SAMPLES = 8
# Linear rdr2geo: lon = 10 + 0.01 * range, lat = 45 + 0.01 * azimuth.
# Outer polygon covers integer pixels (r, c) in [2, 6]; hole covers [4, 5].
_HOLE_VALID = frozenset(
    (row, col)
    for row in range(2, 7)
    for col in range(2, 7)
    if not (row in {4, 5} and col in {4, 5})
)
_INTERIOR_ZERO = (3, 3)
_CANVAS_INVALID = (3, 2)
_HOLE_PIXEL = (4, 4)
_BBOX_CORNER = (0, 0)
_BURST_FOOTPRINT = (
    (10.02, 45.02),
    (10.06, 45.02),
    (10.08, 45.02),
    (10.07, 45.06),
    (10.05, 45.06),
    (10.01, 45.06),
)


def _burst() -> S1Burst:
    """Return one IW1 burst covering the 8x8 test lattice."""
    last = _SAMPLES - 1
    return S1Burst(
        index=0,
        azimuth_time=_ORIGIN,
        sensing_time=_ORIGIN,
        azimuth_anx_time_s=0.0,
        byte_offset=0,
        lines=_LINES,
        samples=_SAMPLES,
        first_valid_sample=tuple([0] * _LINES),
        last_valid_sample=tuple([last] * _LINES),
    )


def _product(path: Path) -> S1Product:
    """Return a one-burst IW1 product whose date id is taken from ``path``."""
    burst = _burst()
    stop = burst.azimuth_time + timedelta(seconds=_LINES - 1)
    swath = S1Swath(
        swath="IW1",
        polarization="VV",
        annotation_path="memory://annotation.xml",
        measurement_path="memory://measurement.tiff",
        lines=_LINES,
        samples=_SAMPLES,
        lines_per_burst=_LINES,
        samples_per_burst=_SAMPLES,
        range_pixel_spacing_m=2.3,
        azimuth_pixel_spacing_m=14.0,
        azimuth_time_interval_s=1.0,
        slant_range_time_s=0.005,
        range_sampling_rate_hz=64_000_000.0,
        radar_frequency_hz=5.405e9,
        sensing_start=burst.azimuth_time,
        sensing_stop=stop,
        bursts=(burst,),
        orbit=OrbitMetadata(
            reference_frame="ITRF",
            source="test",
            vectors=tuple(
                OrbitStateVector(
                    time=burst.azimuth_time + timedelta(seconds=10.0 * step),
                    position_m=(7_000_000.0, 75_000.0 * step, 0.0),
                    velocity_m_s=(0.0, 7_500.0, 0.0),
                )
                for step in range(-3, 4)
            ),
        ),
        doppler_centroid=(
            DopplerCentroidPolynomial(
                coefficients_hz=(100.0, -0.01),
                reference_range_m=800_000.0,
                reference_time_s=0.0,
            ),
        ),
        azimuth_fm_rate=(),
        azimuth_steering_rate_rad_s=0.0,
    )
    return S1Product(
        product_id=path.stem,
        source_path=path.parent / "tracks" / "1A" / "acquisitions" / path.name,
        mission_id="S1A",
        product_type="SLC",
        mode="IW",
        absolute_orbit=1,
        polarizations=("VV",),
        swaths=(swath,),
    )


def _geometry() -> RadarGeometryModel:
    """Return a real radar geometry model so exact support is not skipped."""
    product = _product(Path(REFERENCE))
    swath = product.swaths[0]
    grid = RadarGrid(
        shape=(_LINES, _SAMPLES),
        starting_slant_range_m=float(swath.slant_range_time_s * 299_792_458.0 / 2.0),
        range_spacing_m=float(swath.range_pixel_spacing_m),
        sensing_start=_ORIGIN,
        azimuth_time_interval_s=float(swath.azimuth_time_interval_s),
        wavelength_m=0.05546576,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(grid, swath.orbit)


def _hole_roi() -> Polygons:
    """Return a lon/lat polygon with an interior hole on the test lattice."""
    outer = (
        (10.015, 45.015),
        (10.065, 45.015),
        (10.065, 45.065),
        (10.015, 45.065),
    )
    hole = (
        (10.035, 45.035),
        (10.055, 45.035),
        (10.055, 45.055),
        (10.035, 45.055),
    )
    frame = gpd.GeoDataFrame(
        geometry=[Polygon(outer, [hole])],
        crs="EPSG:4326",
    )
    return Polygons(frame, types="desired")


def _linear_rdr2geo(
    _model: object,
    azimuth_index: np.ndarray,
    range_index: np.ndarray,
    dem: object | None = None,
    **_kwargs: object,
) -> SimpleNamespace:
    """Map every radar sample to lon/lat; never a four-corner affine."""
    del dem
    azimuth = np.asarray(azimuth_index, dtype=np.float64)
    slant = np.asarray(range_index, dtype=np.float64)
    assert azimuth.size > 4
    assert azimuth.shape == slant.shape
    return SimpleNamespace(
        latitude_deg=45.0 + 0.01 * azimuth,
        longitude_deg=10.0 + 0.01 * slant,
        converged=np.ones(azimuth.shape, dtype=bool),
    )


def _fake_scene(path: Path, scene_id: str) -> MagicMock:
    """Return a production scene with native validity distinct from amplitude."""
    scene = MagicMock()
    scene.scene_id = scene_id
    scene.path = path
    scene.burst.index = 0
    scene.burst.azimuth_time = _ORIGIN
    scene.burst.azimuth_anx_time_s = 0.0
    scene.burst.footprint = _BURST_FOOTPRINT
    scene.burst.first_valid_sample = tuple([0] * _LINES)
    scene.burst.last_valid_sample = tuple([_SAMPLES - 1] * _LINES)
    scene.swath = _product(path).swaths[0]
    scene.geometry = _geometry()
    samples = np.ones((_LINES, _SAMPLES), dtype=np.complex64)
    samples[_INTERIOR_ZERO] = np.complex64(0.0)
    native_valid = np.ones((_LINES, _SAMPLES), dtype=bool)
    native_valid[_CANVAS_INVALID] = False
    samples[_CANVAS_INVALID] = np.complex64(0.0)
    scene.array.samples = samples
    scene.array.valid_mask = native_valid
    scene.array.row0 = 0
    scene.array.col0 = 0
    return scene


@pytest.fixture
def fake_s1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch SAFE access and numerical stages; keep geometry and publication."""
    names = (REFERENCE, SECONDARY)
    products = {tmp_path / name: _product(tmp_path / name) for name in names}
    for path in products:
        path.mkdir(parents=True)
        (path / "manifest.safe").write_text("acquired-support-canvas\n")

    def fake_load(path: Path, **_kwargs: object) -> MagicMock:
        scene_id = Path(path).name.split("_")[5][:8]
        return _fake_scene(Path(path), scene_id)

    monkeypatch.setattr(
        f"{coreg_module.__name__}.open_acquisition_products",
        lambda paths, **_k: [products[Path(item)] for item in paths],
    )
    monkeypatch.setattr(f"{coreg_module.__name__}.load_production_scene", fake_load)
    monkeypatch.setattr(
        f"{coreg_module.__name__}.deramp",
        lambda samples, _carrier, **_k: np.asarray(samples),
    )
    monkeypatch.setattr(
        f"{coreg_module.__name__}.reramp",
        lambda samples, _carrier, **_k: np.asarray(samples) * np.complex64(2.0),
    )

    def fake_stage_deramp(state: object, **_kwargs: object) -> object:
        state.secondary_deramped = np.asarray(state.secondary.array.samples)
        return state

    monkeypatch.setattr(f"{coreg_module.__name__}.stage_deramp", fake_stage_deramp)

    def fake_coregister(state: object, **_kwargs: object) -> object:
        aligned = np.full((_LINES, _SAMPLES), 1.0 + 1.0j, dtype=np.complex64)
        aligned[_INTERIOR_ZERO] = np.complex64(0.0)
        aligned[_CANVAS_INVALID] = np.complex64(0.0)
        native = np.asarray(state.secondary.array.valid_mask, dtype=bool)
        aligned[~native] = np.complex64(0.0)
        state.secondary_aligned = aligned
        if getattr(state, "coregistration_grid", "radar") == "geo":
            state.secondary_aligned_valid = np.ones(aligned.shape, dtype=bool)
        else:
            state.secondary_aligned_valid = native
        state.radar_roi_origin = (0, 0)
        state.scientific_lineage = [{"stage": "fake"}]
        return state

    monkeypatch.setattr(f"{coreg_module.__name__}.stage_coregister", fake_coregister)
    monkeypatch.setattr(f"{coreg_module.__name__}._without_pixels", lambda scene: scene)
    monkeypatch.setattr(
        f"{coreg_module.__name__}._phase_state_manifest", lambda _state: None
    )
    monkeypatch.setattr(
        f"{coreg_module.__name__}._roi_burst_window",
        lambda *_a, **_k: (0, _LINES, 0, _SAMPLES),
    )
    monkeypatch.setattr(
        f"{coreg_module.__name__}.reference_window_crop",
        lambda window, _shape: window,
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_rdr2geo",
        _linear_rdr2geo,
    )
    return tmp_path


def _expected_support() -> np.ndarray:
    """Return the worked scientific support, independent of publication code."""
    support = np.zeros((_LINES, _SAMPLES), dtype=bool)
    for row, col in _HOLE_VALID:
        support[row, col] = True
    support[_CANVAS_INVALID] = False
    return support


def _assert_valid_mask(product: object, tag: str, expected: np.ndarray) -> None:
    """Assert the persisted unit valid_mask matches ``expected``."""
    mask = product.read_valid_mask(tag)
    assert mask is not None
    np.testing.assert_array_equal(np.asarray(mask, dtype=bool), expected)


def test_radar_reference_publication_excludes_hole_and_canvas_fill(
    fake_s1: Path,
) -> None:
    """Public Reference write keeps the crop window but not hole or canvas zeros."""
    tmp_path = fake_s1
    (tmp_path / REFERENCE).mkdir(parents=True, exist_ok=True)
    context = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101",
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_hole_roi(),
    )
    unit = context.units["f0_IW1_b0"]
    assert unit.crop == (0, _LINES, 0, _SAMPLES)
    assert unit.row_origin == 0
    assert unit.col_origin == 0
    payload, _ = context.product.read("f0_IW1_b0")
    assert payload.shape == (_LINES, _SAMPLES)
    expected = _expected_support()
    outside = ~expected
    assert not np.isfinite(payload.real[outside]).any()
    assert not np.isfinite(payload.imag[outside]).any()
    interior = expected.copy()
    interior[_INTERIOR_ZERO] = False
    np.testing.assert_array_equal(
        payload[interior],
        np.full(int(interior.sum()), 2.0, dtype=np.complex64),
    )
    assert payload[_INTERIOR_ZERO] == np.complex64(0.0)
    assert not np.isfinite(payload[_HOLE_PIXEL].real)
    assert not np.isfinite(payload[_BBOX_CORNER].real)
    assert not np.isfinite(payload[_CANVAS_INVALID].real)
    _assert_valid_mask(context.product, "f0_IW1_b0", expected)
    mask = context.product.read_valid_mask("f0_IW1_b0")
    assert bool(mask[_INTERIOR_ZERO])
    assert not bool(mask[_CANVAS_INVALID])
    assert (tmp_path / REFERENCE / "manifest.safe").is_file()
    gdf = context.product.footprint
    assert (context.product.root / "footprint.geojson").is_file()
    assert len(gdf) == 1
    assert gdf.iloc[0]["tag"] == "f0_IW1_b0"
    assert gdf.iloc[0]["geometry"].bounds == (10.01, 45.02, 10.08, 45.06)


def _prepare_hole_reference(tmp_path: Path) -> S1ReferenceContext:
    """Prepare the Reference scene for the worked hole ROI."""
    return prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101",
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_hole_roi(),
    )


def _assert_payload_support(payload: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Assert NaN outside support, preserved interior zeros, and kept samples."""
    outside = ~expected
    assert not np.isfinite(payload.real[outside]).any()
    assert not np.isfinite(payload.imag[outside]).any()
    interior = expected.copy()
    interior[_INTERIOR_ZERO] = False
    assert payload[_INTERIOR_ZERO] == np.complex64(0.0)
    assert not np.isfinite(payload[_HOLE_PIXEL].real)
    assert not np.isfinite(payload[_BBOX_CORNER].real)
    assert not np.isfinite(payload[_CANVAS_INVALID].real)
    return interior


def test_radar_secondary_publication_matches_reference_roi_support(
    fake_s1: Path,
) -> None:
    """Secondary publication uses the same exact support after resampling."""
    tmp_path = fake_s1
    context = _prepare_hole_reference(tmp_path)
    product = coregister_s1_secondary(
        context,
        tmp_path / SECONDARY,
        output_scene=tmp_path / "coreg" / "20200113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    payload, placed = product.read("f0_IW1_b0")
    assert (placed.row_origin, placed.col_origin) == (0, 0)
    assert payload.shape == (_LINES, _SAMPLES)
    expected = _expected_support()
    interior = _assert_payload_support(payload, expected)
    np.testing.assert_array_equal(
        payload[interior],
        np.full(int(interior.sum()), 1.0 + 1.0j, dtype=np.complex64),
    )
    _assert_valid_mask(product, "f0_IW1_b0", expected)
    assert (tmp_path / SECONDARY / "manifest.safe").is_file()
    assert (tmp_path / REFERENCE / "manifest.safe").is_file()


def _l_shaped_roi() -> Polygons:
    """Return a concave L whose bounding-box corner is not scientific support."""
    ring = (
        (10.015, 45.015),
        (10.065, 45.015),
        (10.065, 45.045),
        (10.045, 45.045),
        (10.045, 45.065),
        (10.015, 45.065),
    )
    frame = gpd.GeoDataFrame(geometry=[Polygon(ring)], crs="EPSG:4326")
    return Polygons(frame, types="desired")


def test_radar_publication_excludes_concave_bbox_corner(fake_s1: Path) -> None:
    """A bbox corner that is not in the L is invalid; the crop origin is kept."""
    tmp_path = fake_s1
    context = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101",
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_l_shaped_roi(),
    )
    payload, _ = context.product.read("f0_IW1_b0")
    unit = context.units["f0_IW1_b0"]
    assert unit.crop == (0, _LINES, 0, _SAMPLES)
    assert payload.shape == (_LINES, _SAMPLES)
    missing_corner = (6, 6)
    foot = (2, 6)
    stem = (6, 2)
    assert not np.isfinite(payload[missing_corner].real)
    assert payload[foot] == np.complex64(2.0)
    assert payload[stem] == np.complex64(2.0)
    assert payload[_INTERIOR_ZERO] == np.complex64(0.0)
    assert not np.isfinite(payload[_CANVAS_INVALID].real)


def test_geo_publication_uses_pixel_centers_and_excludes_holes(
    fake_s1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Geo publication tests destination pixel centres, not a buffered rectangle."""
    tmp_path = fake_s1
    grid = GeoGridSpec(
        crs="EPSG:4326",
        transform=(10.0, 0.01, 0.0, 45.08, 0.0, -0.01),
        height=_LINES,
        width=_SAMPLES,
        resolution_m=(0.01, 0.01),
    )
    geo_zero = (2, 2)
    geo_canvas = (4, 2)
    geo_hole = (3, 4)
    native = np.ones((_LINES, _SAMPLES), dtype=bool)
    native[geo_canvas] = False
    lut = Geo2RdrLUT(
        az_full=np.zeros((_LINES, _SAMPLES), dtype=np.float64),
        rg_full=np.zeros((_LINES, _SAMPLES), dtype=np.float64),
        valid=native,
        full_radar_shape=(_LINES, _SAMPLES),
        height_m=0.0,
        row0=0,
        col0=0,
    )

    def fake_geocode(*_args: object, **_kwargs: object) -> tuple[np.ndarray, None]:
        samples = np.full((_LINES, _SAMPLES), 3.0 + 0.0j, dtype=np.complex64)
        samples[geo_zero] = np.complex64(0.0)
        samples[geo_canvas] = np.complex64(0.0)
        return samples, None

    monkeypatch.setattr(coreg_module, "build_reference_geo_lut", lambda *_a, **_k: lut)
    monkeypatch.setattr(
        "faninsar.processing.geocoding.geo_modes.geocode_reference_slc_chunked",
        fake_geocode,
    )
    context = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101-geo",
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_hole_roi(),
        coregistration_grid="geo",
        geo_grid=grid,
    )
    payload, placed = context.product.read("f0_IW1_b0")
    assert (placed.row_origin, placed.col_origin) == (0, 0)
    assert payload.shape == (_LINES, _SAMPLES)
    expected = np.zeros((_LINES, _SAMPLES), dtype=bool)
    xs = 10.0 + 0.01 * (np.arange(_SAMPLES) + 0.5)
    ys = 45.08 - 0.01 * (np.arange(_LINES) + 0.5)
    polygon = _hole_roi().geometry.iloc[0]
    for row, lat in enumerate(ys):
        for col, lon in enumerate(xs):
            expected[row, col] = bool(polygon.contains(Point(lon, lat)))
    expected[geo_canvas] = False
    outside = ~expected
    assert not np.isfinite(payload.real[outside]).any()
    assert bool(expected[geo_zero])
    assert not bool(expected[geo_hole])
    assert payload[geo_zero] == np.complex64(0.0)
    assert not np.isfinite(payload[geo_hole].real)
    assert not np.isfinite(payload[geo_canvas].real)
    interior = expected.copy()
    interior[geo_zero] = False
    np.testing.assert_array_equal(
        payload[interior],
        np.full(int(interior.sum()), 3.0, dtype=np.complex64),
    )
    _assert_valid_mask(context.product, "f0_IW1_b0", expected)
    secondary = coregister_s1_secondary(
        context,
        tmp_path / SECONDARY,
        output_scene=tmp_path / "coreg" / "20200113-geo",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    sec_payload, _ = secondary.read("f0_IW1_b0")
    assert not np.isfinite(sec_payload.real[outside]).any()
    assert not np.isfinite(sec_payload[geo_hole].real)
    _assert_valid_mask(secondary, "f0_IW1_b0", expected)
    assert (tmp_path / SECONDARY / "manifest.safe").is_file()


def test_radar_scientific_support_rejects_unknown_geometry() -> None:
    """Unknown geometry must not publish the full crop as ROI support."""
    from faninsar.missions.s1.roi_support import radar_scientific_support

    crop = (0, _LINES, 0, _SAMPLES)
    native = np.ones((_LINES, _SAMPLES), dtype=bool)
    roi = BoundingBox(10.0, 45.0, 10.08, 45.08)
    with pytest.raises(InvalidProcessingStateError, match="RadarGeometryModel"):
        radar_scientific_support(
            MagicMock(),
            ConstantDEM(0.0),
            crop=crop,
            native_valid=native,
            roi=roi,
        )
    support = radar_scientific_support(
        MagicMock(),
        ConstantDEM(0.0),
        crop=crop,
        native_valid=native,
        roi=None,
    )
    np.testing.assert_array_equal(support, native)


SECONDARY_C = "S1A_IW_SLC__1SDV_20200125T000000.SAFE"
_B_HOLE = (3, 5)
_C_HOLE = (6, 3)


def test_reused_reference_keeps_write_id_and_persisted_mask(fake_s1: Path) -> None:
    """Reopening an existing Reference must not rewrite its generation identity."""
    tmp_path = fake_s1
    output = tmp_path / "coreg" / "20200101"
    context = _prepare_hole_reference(tmp_path)
    write_id = context.product.write_id
    mask = np.asarray(context.product.read_valid_mask("f0_IW1_b0"), dtype=bool)
    reused = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=output,
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_hole_roi(),
        overwrite=False,
    )
    assert reused.product.write_id == write_id
    np.testing.assert_array_equal(
        np.asarray(reused.product.read_valid_mask("f0_IW1_b0"), dtype=bool),
        mask,
    )
    payload, _ = reused.product.read("f0_IW1_b0")
    assert np.isnan(payload.real[~mask]).all()
    assert bool(reused.units["f0_IW1_b0"].scientific_support[_CANVAS_INVALID])


def test_legacy_reference_without_validity_revision_is_recomputed(
    fake_s1: Path,
) -> None:
    """A committed Reference missing the validity token is recomputed, not stamped."""
    from faninsar.missions.s1.coregister import S1_SLC_VALIDITY_REVISION

    tmp_path = fake_s1
    output = tmp_path / "coreg" / "20200101"
    context = _prepare_hole_reference(tmp_path)
    first_id = context.product.write_id
    assert (
        context.product.properties.get("faninsar:slc_validity_revision")
        == S1_SLC_VALIDITY_REVISION
    )

    item_path = output / "item.json"
    item = json.loads(item_path.read_text())
    removed = item["properties"].pop("faninsar:slc_validity_revision")
    assert removed == S1_SLC_VALIDITY_REVISION
    item_path.write_text(json.dumps(item))

    opened = CoregisteredSceneStore.open(output)
    assert "faninsar:slc_validity_revision" not in opened.properties
    assert opened.write_id == first_id

    recomputed = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=output,
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_hole_roi(),
        overwrite=False,
    )
    assert recomputed.product.write_id != first_id
    assert (
        recomputed.product.properties.get("faninsar:slc_validity_revision")
        == S1_SLC_VALIDITY_REVISION
    )
    matching = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=output,
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_hole_roi(),
        overwrite=False,
    )
    assert matching.product.write_id == recomputed.product.write_id


def test_radar_secondaries_keep_independent_echo_holes(
    fake_s1: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B and C publication masks follow their own echo, not Reference holes."""
    tmp_path = fake_s1
    extra = tmp_path / SECONDARY_C
    extra.mkdir(parents=True, exist_ok=True)
    (extra / "manifest.safe").write_text("acquired-support-canvas\n")
    products = {
        tmp_path / REFERENCE: _product(tmp_path / REFERENCE),
        tmp_path / SECONDARY: _product(tmp_path / SECONDARY),
        extra: _product(extra),
    }
    holes = {
        "20200101": _CANVAS_INVALID,
        "20200113": _B_HOLE,
        "20200125": _C_HOLE,
    }

    def fake_load(path: Path, **_kwargs: object) -> MagicMock:
        scene = _fake_scene(Path(path), Path(path).name.split("_")[5][:8])
        native = np.ones((_LINES, _SAMPLES), dtype=bool)
        native[holes[scene.scene_id]] = False
        samples = np.ones((_LINES, _SAMPLES), dtype=np.complex64)
        samples[_INTERIOR_ZERO] = np.complex64(0.0)
        samples[~native] = np.complex64(0.0)
        scene.array.valid_mask = native
        scene.array.samples = samples
        return scene

    monkeypatch.setattr(
        f"{coreg_module.__name__}.open_acquisition_products",
        lambda paths, **_k: [products[Path(item)] for item in paths],
    )
    monkeypatch.setattr(f"{coreg_module.__name__}.load_production_scene", fake_load)

    context = _prepare_hole_reference(tmp_path)
    geometry = np.asarray(context.units["f0_IW1_b0"].scientific_support, dtype=bool)
    assert bool(geometry[_CANVAS_INVALID])
    assert bool(geometry[_B_HOLE])
    assert bool(geometry[_C_HOLE])
    ref_mask = np.asarray(context.product.read_valid_mask("f0_IW1_b0"), dtype=bool)
    assert not bool(ref_mask[_CANVAS_INVALID])
    assert bool(ref_mask[_B_HOLE])
    assert bool(ref_mask[_C_HOLE])

    secondary_b = coregister_s1_secondary(
        context,
        tmp_path / SECONDARY,
        output_scene=tmp_path / "coreg" / "20200113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    mask_b = np.asarray(secondary_b.read_valid_mask("f0_IW1_b0"), dtype=bool)
    assert bool(mask_b[_CANVAS_INVALID])
    assert not bool(mask_b[_B_HOLE])
    assert bool(mask_b[_C_HOLE])
    payload_b, _ = secondary_b.read("f0_IW1_b0")
    assert np.isnan(payload_b.real[~mask_b]).all()
    assert bool(context.units["f0_IW1_b0"].scientific_support[_CANVAS_INVALID])

    secondary_c = coregister_s1_secondary(
        context,
        extra,
        output_scene=tmp_path / "coreg" / "20200125",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    mask_c = np.asarray(secondary_c.read_valid_mask("f0_IW1_b0"), dtype=bool)
    assert bool(mask_c[_CANVAS_INVALID])
    assert bool(mask_c[_B_HOLE])
    assert not bool(mask_c[_C_HOLE])
    payload_c, _ = secondary_c.read("f0_IW1_b0")
    assert np.isnan(payload_c.real[~mask_c]).all()


def test_warns_once_for_unsupported_pixels_inside_roi(
    fake_s1: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One warning is emitted for actual unsupported pixels inside requested support."""
    caplog.set_level(logging.WARNING)
    _prepare_hole_reference(fake_s1)
    messages = [
        record.getMessage()
        for record in caplog.records
        if "lack interpolation support" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "f0_IW1_b0" in messages[0]
