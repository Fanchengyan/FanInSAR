"""Sentinel-1 Stack coregistration with the Reference computed once (P0059)."""

from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.data.query import BoundingBox
from faninsar.io.storage.scene_store import (
    CoregisteredSceneStore,
    validate_reference_binding,
)
from faninsar.missions.protocols import DopplerCentroidPolynomial
from faninsar.missions.s1 import coregister as coreg_module
from faninsar.missions.s1.coregister import (
    S1ReferenceContext,
    coregister_s1_scene,
    coregister_s1_secondary,
    prepare_s1_reference,
)
from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import ConstantDEM
from faninsar.processing.geometry.record import radar_geometry_from_record

REFERENCE = "S1A_IW_SLC__1SDV_20200101T000000.SAFE"
SECONDARIES = (
    "S1A_IW_SLC__1SDV_20200113T000000.SAFE",
    "S1A_IW_SLC__1SDV_20200125T000000.SAFE",
)


def test_coregister_s1_scene_is_exported() -> None:
    """The apply stage is part of the public Sentinel-1 mission surface."""
    from faninsar.missions.s1 import coregister_s1_scene as exported

    assert exported is coregister_s1_scene


def test_coregister_s1_scene_public_signature() -> None:
    """The convenience wrapper keeps one product directory per date."""
    parameters = inspect.signature(coregister_s1_scene).parameters
    assert list(parameters) == [
        "reference_path",
        "secondary_path",
        "output_scene",
        "reference_scene",
        "swaths",
        "bursts",
        "roi",
        "dem",
        "misreg_az_px",
        "misreg_rg_px",
        "coregistration_grid",
        "geo_grid",
        "geo_lut_cache_dir",
        "primary_orbit_path",
        "secondary_orbit_path",
        "device",
    ]
    for name in ("output_scene", "reference_scene", "dem", "misreg_az_px"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty
    assert parameters["coregistration_grid"].default == "radar"
    assert parameters["device"].default == "auto"


@pytest.mark.parametrize(
    ("misreg_az_px", "misreg_rg_px"),
    [
        (np.nan, 0.0),
        (0.0, np.inf),
        (-np.inf, 0.1),
    ],
)
def test_coregister_s1_scene_rejects_non_finite_misreg(
    tmp_path: Path,
    misreg_az_px: float,
    misreg_rg_px: float,
) -> None:
    """Non-finite date constants are rejected before any apply work."""
    with pytest.raises(InvalidProcessingStateError, match="finite"):
        coregister_s1_scene(
            tmp_path / "ref.SAFE",
            tmp_path / "sec.SAFE",
            output_scene=tmp_path / "coreg" / "sec",
            reference_scene=tmp_path / "coreg" / "ref",
            dem=ConstantDEM(0.0),
            misreg_az_px=misreg_az_px,
            misreg_rg_px=misreg_rg_px,
        )


_ORIGIN = datetime(2020, 1, 1, tzinfo=UTC)
_LINES = 8
_SAMPLES = 8


def _burst() -> S1Burst:
    """Build a single IW1 burst for apply-stage tests."""
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
    """Build a one-burst IW1 product whose date id is taken from ``path``."""
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
        source_path=path,
        mission_id="S1A",
        product_type="SLC",
        mode="IW",
        absolute_orbit=1,
        polarizations=("VV",),
        swaths=(swath,),
    )


class _Recorder:
    """Count Reference SAFE reads and Reference deramps across dates."""

    def __init__(self) -> None:
        self.reference_reads = 0
        self.reference_loads = 0
        self.deramps = 0
        self.coregister_kwargs: list[dict[str, object]] = []


def _fake_scene(path: Path, scene_id: str) -> MagicMock:
    """Return a production-scene stand-in with the fields apply needs."""
    scene = MagicMock()
    scene.scene_id = scene_id
    scene.path = path
    scene.burst.index = 0
    scene.burst.azimuth_time = _ORIGIN
    scene.swath = _product(path).swaths[0]
    scene.geometry.wavelength_m = 0.05546576
    scene.array.samples = np.ones((_LINES, _SAMPLES), dtype=np.complex64)
    scene.array.valid_mask = np.ones((_LINES, _SAMPLES), dtype=bool)
    scene.array.row0 = 0
    scene.array.col0 = 0
    return scene


@pytest.fixture
def fake_s1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Patch SAFE access and the numerical stages with deterministic fakes."""
    recorder = _Recorder()
    names = (REFERENCE, *SECONDARIES)
    products = {tmp_path / name: _product(tmp_path / name) for name in names}

    def fake_load(path: Path, **kwargs: object) -> MagicMock:
        scene_id = Path(path).name.split("_")[5][:8]
        if Path(path).name == REFERENCE:
            recorder.reference_loads += 1
            if kwargs.get("read_samples", True):
                recorder.reference_reads += 1
        return _fake_scene(Path(path), scene_id)

    def fake_deramp(
        samples: np.ndarray, _carrier: object, **_kwargs: object
    ) -> np.ndarray:
        recorder.deramps += 1
        return np.asarray(samples)

    def fake_reramp(
        samples: np.ndarray, _carrier: object, **_kwargs: object
    ) -> np.ndarray:
        return np.asarray(samples) * np.complex64(2.0)

    def fake_stage_deramp(state: object, **kwargs: object) -> object:
        assert kwargs.get("secondary_only") is True
        state.secondary_deramped = np.asarray(state.secondary.array.samples)
        return state

    def fake_coregister(state: object, **kwargs: object) -> object:
        recorder.coregister_kwargs.append(kwargs)
        state.secondary_aligned = np.full(
            (_LINES, _SAMPLES), 1.0 + 1.0j, dtype=np.complex64
        )
        state.radar_roi_origin = None
        state.scientific_lineage = [{"stage": "fake"}]
        return state

    module = "faninsar.missions.s1.coregister"
    monkeypatch.setattr(
        f"{module}.open_acquisition_products",
        lambda paths, **_k: [products[Path(path)] for path in paths],
    )
    monkeypatch.setattr(f"{module}.load_production_scene", fake_load)
    monkeypatch.setattr(f"{module}.deramp", fake_deramp)
    monkeypatch.setattr(f"{module}.reramp", fake_reramp)
    monkeypatch.setattr(f"{module}.stage_deramp", fake_stage_deramp)
    monkeypatch.setattr(f"{module}.stage_coregister", fake_coregister)
    monkeypatch.setattr(f"{module}._without_pixels", lambda scene: scene)
    monkeypatch.setattr(f"{module}._phase_state_manifest", lambda _state: None)
    return recorder


def _prepare(tmp_path: Path, **kwargs: object) -> S1ReferenceContext:
    return prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101",
        dem=ConstantDEM(0.0),
        **kwargs,
    )


def test_reference_is_read_and_deramped_once_across_dates(
    tmp_path: Path, fake_s1: _Recorder
) -> None:
    """Every secondary reuses one Reference product; its pixels are read once."""
    context = _prepare(tmp_path)
    assert fake_s1.reference_reads == 1
    assert fake_s1.deramps == 1
    products = [
        coregister_s1_secondary(
            context,
            tmp_path / name,
            output_scene=tmp_path / "coreg" / name.split("_")[5][:8],
            misreg_az_px=0.05,
            misreg_rg_px=-0.02,
        )
        for name in SECONDARIES
    ]
    assert fake_s1.reference_reads == 1
    assert fake_s1.deramps == 1
    reference = context.product
    assert reference.is_reference
    assert reference.role == "reference"
    np.testing.assert_array_equal(
        reference.read("f0_IW1_b0")[0],
        np.full((_LINES, _SAMPLES), 2.0, dtype=np.complex64),
    )
    for product in products:
        assert product.inputs["reference"] == reference.write_id
        validate_reference_binding(reference, product)
        np.testing.assert_array_equal(
            product.read("f0_IW1_b0")[0],
            np.full((_LINES, _SAMPLES), 1.0 + 1.0j, dtype=np.complex64),
        )
    for kwargs in fake_s1.coregister_kwargs:
        assert kwargs["secondary_only"] is True
        assert kwargs["reference_shape"] == (_LINES, _SAMPLES)
        assert "esd_enabled" not in kwargs
        assert "amplitude_refinement_enabled" not in kwargs
        assert kwargs["misreg_az_px"] == pytest.approx(0.05)
        assert kwargs["misreg_rg_px"] == pytest.approx(-0.02)


def test_reopened_reference_is_reused_without_reading_pixels(
    tmp_path: Path, fake_s1: _Recorder
) -> None:
    """A second prepare loads metadata only and keeps the Reference write id."""
    first = _prepare(tmp_path)
    second = _prepare(tmp_path)
    assert fake_s1.reference_reads == 1
    assert fake_s1.deramps == 1
    assert second.write_id == first.write_id
    rebuilt = _prepare(tmp_path, overwrite=True)
    assert fake_s1.reference_reads == 2
    assert rebuilt.write_id != first.write_id


def test_reference_persists_frame_radar_geometry(
    tmp_path: Path, fake_s1: _Recorder
) -> None:
    """The Reference Item carries the merged-grid geometry for reopen."""
    del fake_s1
    context = _prepare(tmp_path)
    record = context.product.radar_geometry
    assert record is not None
    item = json.loads((context.product.root / "item.json").read_text())
    assert item["properties"]["faninsar:radar_geometry"] == record
    swath = _product(tmp_path / REFERENCE).swaths[0]
    assert record["shape"] == [_LINES, _SAMPLES]
    assert record["sensing_start"] == _ORIGIN.isoformat()
    assert record["azimuth_time_interval_s"] == swath.azimuth_time_interval_s
    assert record["range_spacing_m"] == swath.range_pixel_spacing_m
    assert record["swath_column_offsets"] == {"IW1": 0}
    geometry = radar_geometry_from_record(record)
    assert geometry.sensing_start == _ORIGIN
    assert geometry.starting_slant_range_m == pytest.approx(
        swath.slant_range_time_s * 299_792_458.0 / 2.0
    )


def test_secondary_rejects_the_reference_date(
    tmp_path: Path, fake_s1: _Recorder
) -> None:
    """The Reference is never coregistered onto itself."""
    del fake_s1
    context = _prepare(tmp_path)
    with pytest.raises(InvalidProcessingStateError, match="non-Reference"):
        coregister_s1_secondary(
            context,
            tmp_path / REFERENCE,
            output_scene=tmp_path / "coreg" / "x",
            misreg_az_px=0.0,
            misreg_rg_px=0.0,
        )


def test_reused_reference_rejects_a_changed_selection(
    tmp_path: Path, fake_s1: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted Reference placed differently fails closed on reuse."""
    del fake_s1
    _prepare(tmp_path)
    monkeypatch.setattr(
        coreg_module, "_roi_burst_window", lambda *_a, **_k: (2, 6, 2, 6)
    )
    monkeypatch.setattr(
        coreg_module, "reference_window_crop", lambda window, _shape: window
    )
    with pytest.raises(InvalidProcessingStateError, match="overwrite=True"):
        _prepare(
            tmp_path,
            swaths=("IW1",),
            bursts={"IW1": [0]},
            roi=BoundingBox(0.0, 0.0, 1.0, 1.0),
        )


def test_wrapper_keeps_explicit_bursts_when_roi_is_set(
    tmp_path: Path, fake_s1: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ROI crops the apply window; it must not re-select resolved bursts."""
    del fake_s1

    def fake_select(*_args: object, **_kwargs: object) -> dict[object, object]:
        pytest.fail("explicit bursts must not re-run ROI selection")

    monkeypatch.setattr(coreg_module, "_select_bursts_by_roi", fake_select)
    monkeypatch.setattr(coreg_module, "_roi_burst_window", lambda *_a, **_k: None)
    product = coregister_s1_scene(
        tmp_path / REFERENCE,
        tmp_path / SECONDARIES[0],
        output_scene=tmp_path / "coreg" / "20200113",
        reference_scene=tmp_path / "coreg" / "20200101",
        dem=ConstantDEM(0.0),
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=BoundingBox(0.0, 0.0, 1.0, 1.0),
    )
    assert product.date_id == "20200113"
    assert isinstance(product, CoregisteredSceneStore)


def _multi_swath_product(path: Path) -> S1Product:
    """Build an IW1-IW3 product with two bursts per swath and offset timing."""
    from dataclasses import replace

    base = _product(path).swaths[0]
    swaths = []
    for index, name in enumerate(("IW1", "IW2", "IW3")):
        start = _ORIGIN + timedelta(seconds=0.3 * index)
        bursts = tuple(
            replace(
                _burst(),
                index=burst,
                azimuth_time=start + timedelta(seconds=_LINES * burst * 0.9),
                sensing_time=start + timedelta(seconds=_LINES * burst * 0.9),
            )
            for burst in range(2)
        )
        swaths.append(
            replace(
                base,
                swath=name,
                lines=2 * _LINES,
                slant_range_time_s=base.slant_range_time_s + 1e-5 * index,
                range_pixel_spacing_m=base.range_pixel_spacing_m + 0.01 * index,
                azimuth_time_interval_s=1.0 + 0.001 * index,
                bursts=bursts,
            )
        )
    return replace(_product(path), swaths=tuple(swaths))


def test_frame_geometry_record_matches_unit_placement_for_iw1_to_iw3(
    tmp_path: Path,
) -> None:
    """Row/column 0 of the record is exactly where _unit_placement puts units."""
    from faninsar.missions.s1.processing import (
        SPEED_OF_LIGHT_M_S,
        _swath_range_offsets,
    )

    product = _multi_swath_product(tmp_path / REFERENCE)
    swaths = ("IW1", "IW2", "IW3")
    resolved = {(0, name): [0, 1] for name in swaths}
    offsets = _swath_range_offsets(swaths, [product])
    units, rows, cols = coreg_module._unit_placement(
        [product], resolved, swaths, offsets
    )
    record = coreg_module.frame_geometry_record(
        [product],
        resolved,
        swaths,
        offsets,
        (rows, cols),
        orbit=product.swaths[0].orbit,
    )
    first = product.swath("IW1")
    earliest = min(
        product.swath(name).bursts[b].azimuth_time for name in swaths for b in (0, 1)
    )
    assert record["shape"] == [rows, cols]
    assert record["sensing_start"] == earliest.isoformat()
    assert record["azimuth_time_interval_s"] == first.azimuth_time_interval_s
    assert record["range_spacing_m"] == first.range_pixel_spacing_m
    assert record["starting_slant_range_m"] == pytest.approx(
        first.slant_range_time_s * SPEED_OF_LIGHT_M_S / 2.0
    )
    assert record["swath_column_offsets"] == offsets
    geometry = radar_geometry_from_record(record)
    for name in swaths:
        for _frame, burst, row in units[name]:
            sensing = product.swath(name).bursts[burst].azimuth_time
            expected = round(
                (sensing - earliest).total_seconds() / first.azimuth_time_interval_s
            )
            assert row == expected
            assert geometry.azimuth_time(row) - sensing == pytest.approx(
                timedelta(0), abs=timedelta(seconds=first.azimuth_time_interval_s)
            )
