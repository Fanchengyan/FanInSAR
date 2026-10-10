"""Developer-stage Sentinel-1 scene coregistration apply."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.io.storage.scene_store import CoregisteredSceneStore
from faninsar.missions.protocols import DopplerCentroidPolynomial
from faninsar.missions.s1.coregister import coregister_s1_scene
from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import ConstantDEM


def test_coregister_s1_scene_is_exported() -> None:
    """The apply stage is part of the public Sentinel-1 mission surface."""
    from faninsar.missions.s1 import coregister_s1_scene as exported

    assert exported is coregister_s1_scene


def test_coregister_s1_scene_public_signature() -> None:
    """The apply stage exposes the PROPOSAL-0056 developer-stage signature."""
    parameters = inspect.signature(coregister_s1_scene).parameters
    assert list(parameters) == [
        "reference_path",
        "secondary_path",
        "output_scenes",
        "swaths",
        "bursts",
        "roi",
        "dem",
        "misreg_az_px",
        "misreg_rg_px",
        "coregistration_grid",
        "geo_grid",
        "primary_orbit_path",
        "secondary_orbit_path",
        "device",
    ]
    assert parameters["output_scenes"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["swaths"].default is None
    assert parameters["bursts"].default is None
    assert parameters["roi"].default is None
    assert parameters["coregistration_grid"].default == "radar"
    assert parameters["geo_grid"].default is None
    assert parameters["primary_orbit_path"].default is None
    assert parameters["secondary_orbit_path"].default is None
    assert parameters["device"].default == "auto"
    assert parameters["output_scenes"].default is inspect.Parameter.empty
    assert parameters["dem"].default is inspect.Parameter.empty
    assert parameters["misreg_az_px"].default is inspect.Parameter.empty
    assert parameters["misreg_rg_px"].default is inspect.Parameter.empty


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
            output_scenes=tmp_path / "scenes",
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
            vectors=(
                OrbitStateVector(
                    time=burst.azimuth_time,
                    position_m=(7_000_000.0, 0.0, 0.0),
                    velocity_m_s=(0.0, 7_500.0, 0.0),
                ),
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


def _fake_scene(path: Path, scene_id: str) -> MagicMock:
    """Return a production-scene stand-in with the fields apply writes need."""
    scene = MagicMock()
    scene.scene_id = scene_id
    scene.path = path
    scene.burst.index = 0
    scene.burst.azimuth_time = _ORIGIN
    scene.swath.swath = "IW1"
    scene.swath.lines_per_burst = _LINES
    scene.swath.samples_per_burst = _SAMPLES
    scene.swath.azimuth_time_interval_s = 1.0
    scene.geometry.wavelength_m = 0.05546576
    samples = np.ones((_LINES, _SAMPLES), dtype=np.complex64)
    scene.array.samples = samples
    scene.array.valid_mask = np.ones((_LINES, _SAMPLES), dtype=bool)
    scene.array.row0 = 0
    scene.array.col0 = 0
    return scene


def test_coregister_s1_scene_writes_store_without_measuring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply writes a scene store using date constants and never enables ESD/Ampcor."""
    reference = tmp_path / "S1A_IW_SLC__1SDV_20200101T000000.SAFE"
    secondary = tmp_path / "S1A_IW_SLC__1SDV_20200113T000000.SAFE"
    output = tmp_path / "scenes"
    products = {reference: _product(reference), secondary: _product(secondary)}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.open_safe_product",
        lambda path, **_kwargs: products[Path(path)],
        raising=False,
    )

    def fake_load(path: Path, **_kwargs: object) -> MagicMock:
        scene_id = "20200101" if "20200101" in Path(path).name else "20200113"
        return _fake_scene(Path(path), scene_id)

    def fake_deramp(state: object, **_kwargs: object) -> object:
        aligned = np.ones((_LINES, _SAMPLES), dtype=np.complex64)
        state.primary_deramped = aligned
        state.secondary_deramped = aligned
        return state

    def fake_coregister(state: object, **kwargs: object) -> object:
        captured.update(kwargs)
        aligned = np.ones((_LINES, _SAMPLES), dtype=np.complex64)
        state.primary_deramped = aligned
        state.secondary_aligned = aligned * (1.0 + 1.0j)
        state.radar_roi_origin = None
        state.scientific_lineage = []
        state.phase_state = None
        return state

    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.load_production_scene",
        fake_load,
        raising=False,
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.stage_deramp",
        fake_deramp,
        raising=False,
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.stage_coregister",
        fake_coregister,
        raising=False,
    )

    result = coregister_s1_scene(
        reference,
        secondary,
        output_scenes=output,
        dem=ConstantDEM(0.0),
        misreg_az_px=0.05,
        misreg_rg_px=-0.02,
        roi=None,
    )

    assert result == output
    assert captured["esd_enabled"] is False
    assert captured["amplitude_refinement_enabled"] is False
    assert captured["misreg_az_px"] == pytest.approx(0.05)
    assert captured["misreg_rg_px"] == pytest.approx(-0.02)
    store = CoregisteredSceneStore.open(output)
    assert store.date_id == "20200113"
    assert store.reference_id == "20200101"
    assert store.domain == "radar"
    assert store.flatten_stage == "coregistration"
    assert len(store.units) == 1
    primary, secondary_arr, _unit = store.read(store.units[0].tag)
    np.testing.assert_array_equal(
        primary,
        np.ones((_LINES, _SAMPLES), dtype=np.complex64),
    )
    np.testing.assert_array_equal(
        secondary_arr,
        np.ones((_LINES, _SAMPLES), dtype=np.complex64) * (1.0 + 1.0j),
    )
