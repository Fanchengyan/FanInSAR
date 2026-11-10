"""Reference crop prepass reports the union DEM demand before pixel reads (P0079)."""

# The repository intentionally keeps tests as a namespace package.
# ruff: noqa: INP001

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from affine import Affine

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.data.query import BoundingBox
from faninsar.missions.protocols import DopplerCentroidPolynomial
from faninsar.missions.s1 import coregister as coreg_module
from faninsar.missions.s1.coregister import S1ReferenceContext, prepare_s1_reference
from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath
from faninsar.processing.geometry import ConstantDEM, RadarGeometryModel, RasterDEM
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.grids import GridSpec
from faninsar.processing.geometry.managed_dem import (
    DEMCoverageInsufficientError,
    expand_bounds,
    interpolation_margin_deg,
    union_bounds,
)

REFERENCE = "S1A_IW_SLC__1SDV_20200101T000000.SAFE"
_ORIGIN = datetime(2020, 1, 1, tzinfo=UTC)
_LINES = 8
_SAMPLES = 8
_COVERING_ROI = BoundingBox(9.99, 44.99, 10.08, 45.08)
NORTHERN_CROP = (6, 8, 3, 5)
WESTERN_CROP = (3, 5, 0, 2)
_SPACING = 0.001


def _orbit() -> OrbitMetadata:
    return OrbitMetadata(
        reference_frame="ITRF",
        source="test",
        vectors=tuple(
            OrbitStateVector(
                time=_ORIGIN + timedelta(seconds=10.0 * step),
                position_m=(7_000_000.0, 75_000.0 * step, 0.0),
                velocity_m_s=(0.0, 7_500.0, 0.0),
            )
            for step in range(-3, 4)
        ),
    )


def _burst(index: int, *, azimuth_time: datetime) -> S1Burst:
    last = _SAMPLES - 1
    return S1Burst(
        index=index,
        azimuth_time=azimuth_time,
        sensing_time=azimuth_time,
        azimuth_anx_time_s=float(index),
        byte_offset=0,
        lines=_LINES,
        samples=_SAMPLES,
        first_valid_sample=tuple([0] * _LINES),
        last_valid_sample=tuple([last] * _LINES),
    )


def _swath(*, name: str, bursts: tuple[S1Burst, ...]) -> S1Swath:
    stop = bursts[-1].azimuth_time + timedelta(seconds=_LINES - 1)
    return S1Swath(
        swath=name,  # type: ignore[arg-type]
        polarization="VV",
        annotation_path="memory://annotation.xml",
        measurement_path=f"memory://s1a-{name.lower()}-slc-vv.tiff",
        lines=_LINES * len(bursts),
        samples=_SAMPLES,
        lines_per_burst=_LINES,
        samples_per_burst=_SAMPLES,
        range_pixel_spacing_m=2.3,
        azimuth_pixel_spacing_m=14.0,
        azimuth_time_interval_s=1.0,
        slant_range_time_s=0.005,
        range_sampling_rate_hz=64_000_000.0,
        radar_frequency_hz=5.405e9,
        sensing_start=bursts[0].azimuth_time,
        sensing_stop=stop,
        bursts=bursts,
        orbit=_orbit(),
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


def _product(path: Path) -> S1Product:
    iw1 = _swath(name="IW1", bursts=(_burst(0, azimuth_time=_ORIGIN),))
    iw2 = _swath(
        name="IW2",
        bursts=tuple(
            _burst(index, azimuth_time=_ORIGIN + timedelta(seconds=index))
            for index in range(3)
        ),
    )
    return S1Product(
        product_id=path.stem,
        source_path=path.parent / "tracks" / "1A" / "acquisitions" / path.name,
        mission_id="S1A",
        product_type="SLC",
        mode="IW",
        absolute_orbit=1,
        polarizations=("VV",),
        swaths=(iw1, iw2),
    )


def _geometry(swath: S1Swath, burst: S1Burst) -> RadarGeometryModel:
    grid = RadarGrid(
        shape=(_LINES, _SAMPLES),
        starting_slant_range_m=float(swath.slant_range_time_s * 299_792_458.0 / 2.0),
        range_spacing_m=float(swath.range_pixel_spacing_m),
        sensing_start=burst.azimuth_time,
        azimuth_time_interval_s=float(swath.azimuth_time_interval_s),
        wavelength_m=0.05546576,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(grid, swath.orbit)


def _linear_rdr2geo(
    _model: object,
    azimuth_index: np.ndarray,
    range_index: np.ndarray,
    dem: object | None = None,
    **_kwargs: object,
) -> SimpleNamespace:
    del dem
    azimuth = np.asarray(azimuth_index, dtype=np.float64)
    slant = np.asarray(range_index, dtype=np.float64)
    return SimpleNamespace(
        latitude_deg=45.0 + 0.01 * azimuth,
        longitude_deg=10.0 + 0.01 * slant,
        converged=np.ones(azimuth.shape, dtype=bool),
    )


def _linear_crop_demand(
    crop: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    row0, row1, col0, col1 = crop
    envelope = (
        10.0 + 0.01 * col0,
        45.0 + 0.01 * row0,
        10.0 + 0.01 * (col1 - 1),
        45.0 + 0.01 * (row1 - 1),
    )
    return expand_bounds(envelope, interpolation_margin_deg(_SPACING))


def _union_demand() -> tuple[float, float, float, float]:
    return union_bounds(
        _linear_crop_demand(NORTHERN_CROP), _linear_crop_demand(WESTERN_CROP)
    )


def _geographic_dem(
    west: float,
    south: float,
    east: float,
    north: float,
) -> RasterDEM:
    width = max(8, int(np.ceil((east - west) / _SPACING)))
    height = max(8, int(np.ceil((north - south) / _SPACING)))
    grid = GridSpec(
        crs="EPSG:4326",
        transform=Affine(_SPACING, 0.0, west, 0.0, -_SPACING, north),
        height=height,
        width=width,
    )
    return RasterDEM(
        array=np.full((height, width), 100.0, dtype=np.float32),
        grid=grid,
        vertical_datum="ellipsoidal",
    )


def _partial_dem() -> RasterDEM:
    """Cover the ROI middle, missing the northern and western crop demands."""
    return _geographic_dem(10.02, 45.035, 10.05, 45.055)


def _covering_dem() -> RasterDEM:
    west, south, east, north = _union_demand()
    return _geographic_dem(west - 0.01, south - 0.01, east + 0.01, north + 0.01)


class _PrepareProbe:
    def __init__(self) -> None:
        self.pixel_reads = 0
        self.writers = 0


@pytest.fixture
def two_unit_s1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _PrepareProbe:
    """Two selected radar units with disjoint northern and western crops."""
    path = tmp_path / REFERENCE
    path.mkdir(parents=True)
    (path / "manifest.safe").write_text("acquired-support-canvas\n")
    product = _product(path)
    probe = _PrepareProbe()

    def fake_load(load_path: Path, **kwargs: object) -> MagicMock:
        if kwargs.get("read_samples", True):
            probe.pixel_reads += 1
        burst_index = int(kwargs.get("burst_index", 0))
        swath_name = str(kwargs.get("swath", "IW1"))
        swath = product.swath(swath_name)
        burst = swath.bursts[burst_index]
        scene = MagicMock()
        scene.scene_id = "20200101"
        scene.path = Path(load_path)
        scene.burst = burst
        scene.swath = swath
        scene.array.samples = np.ones((_LINES, _SAMPLES), dtype=np.complex64)
        scene.array.valid_mask = np.ones((_LINES, _SAMPLES), dtype=bool)
        scene.array.row0 = 0
        scene.array.col0 = 0
        scene.geometry = _geometry(swath, burst)
        return scene

    def fake_window(
        _roi: object, geometry: object, *_args: object, **_kwargs: object
    ) -> tuple[int, int, int, int]:
        start = getattr(geometry, "sensing_start", None)
        if start == _ORIGIN:
            return NORTHERN_CROP
        return WESTERN_CROP

    real_writer = coreg_module.SceneWriter

    class _CountingWriter(real_writer):  # type: ignore[valid-type,misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            probe.writers += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        f"{coreg_module.__name__}.open_acquisition_products",
        lambda paths, **_k: [product for _ in paths],
    )
    monkeypatch.setattr(f"{coreg_module.__name__}.load_production_scene", fake_load)
    monkeypatch.setattr(f"{coreg_module.__name__}._roi_burst_window", fake_window)
    monkeypatch.setattr(
        f"{coreg_module.__name__}.reference_window_crop",
        lambda window, _shape, **_k: window,
    )
    monkeypatch.setattr(
        f"{coreg_module.__name__}.deramp",
        lambda samples, _carrier, **_k: np.asarray(samples),
    )
    monkeypatch.setattr(
        f"{coreg_module.__name__}.reramp",
        lambda samples, _carrier, **_k: np.asarray(samples),
    )
    monkeypatch.setattr(f"{coreg_module.__name__}._without_pixels", lambda scene: scene)
    monkeypatch.setattr(f"{coreg_module.__name__}.SceneWriter", _CountingWriter)
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_rdr2geo",
        _linear_rdr2geo,
    )
    return probe


def _prepare(
    tmp_path: Path, dem: object, *, swaths: tuple[str, ...] = ("IW1", "IW2")
) -> S1ReferenceContext:
    return prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101",
        swaths=swaths,
        bursts={"IW1": [0], "IW2": [2]},
        roi=_COVERING_ROI,
        dem=dem,
    )


def test_two_unit_crops_raise_union_demand_before_pixel_reads_or_writer(
    tmp_path: Path,
    two_unit_s1: _PrepareProbe,
) -> None:
    """IW1_b0 north plus IW2_b2 west aggregate before any burst pixel read."""
    with pytest.raises(DEMCoverageInsufficientError) as info:
        _prepare(tmp_path, _partial_dem())
    assert info.value.required_bounds == pytest.approx(_union_demand())
    assert two_unit_s1.pixel_reads == 0
    assert two_unit_s1.writers == 0


def test_union_demand_is_invariant_to_unit_order(
    tmp_path: Path,
    two_unit_s1: _PrepareProbe,
) -> None:
    """Swapping IW1/IW2 processing order does not change the union demand."""
    first_error: DEMCoverageInsufficientError | None = None
    with pytest.raises(DEMCoverageInsufficientError) as first:
        _prepare(tmp_path, _partial_dem(), swaths=("IW1", "IW2"))
    first_error = first.value
    with pytest.raises(DEMCoverageInsufficientError) as second:
        _prepare(
            tmp_path,
            _partial_dem(),
            swaths=("IW2", "IW1"),
        )
    assert first_error.required_bounds == pytest.approx(second.value.required_bounds)
    assert first_error.required_bounds == pytest.approx(_union_demand())
    assert two_unit_s1.pixel_reads == 0
    assert two_unit_s1.writers == 0


def test_covering_union_dem_prepares_selected_crops(
    tmp_path: Path,
    two_unit_s1: _PrepareProbe,
) -> None:
    """A raster that already covers the union proceeds through Reference prepare."""
    context = _prepare(tmp_path, _covering_dem())
    assert set(context.units) == {"f0_IW1_b0", "f0_IW2_b2"}
    assert two_unit_s1.pixel_reads >= 1
    assert two_unit_s1.writers == 1


def test_constant_dem_bypasses_crop_union_preflight(
    tmp_path: Path,
    two_unit_s1: _PrepareProbe,
) -> None:
    """Non-raster DEMs do not raise a coverage shortage from the crop prepass."""
    context = _prepare(tmp_path, ConstantDEM(0.0))
    assert set(context.units) == {"f0_IW1_b0", "f0_IW2_b2"}
    assert two_unit_s1.writers == 1


@pytest.mark.usefixtures("two_unit_s1")
def test_call_wrapper_expands_once_to_union_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing single-retry wrapper grows once to the union and succeeds."""
    from faninsar.processing.geometry import DEM, SourceDEM
    from tests.stack.test_dem_handoff import _dummy_safes, _fake_source_to_raster
    from tests.support.local_stack import LocalStack

    cache = tmp_path / "dem-cache"
    cache.mkdir()
    source = DEM.from_source("auto", cache_dir=cache)
    grids: list[GridSpec] = []
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_source_to_raster(grids))
    work = tmp_path / "work-union"
    stack = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=work,
        dem=source,
        roi=BoundingBox(10.02, 45.035, 10.05, 45.055, crs="EPSG:4326"),
        dem_cache_dir=cache,
    )
    stack.prepare_data()
    n_materialize = len(grids)
    stack.config.dem = _partial_dem()
    calls = 0

    def _run() -> object:
        nonlocal calls
        calls += 1
        return _prepare(tmp_path, stack._geometry_dem())

    result = stack._call_with_dem_coverage(None, _run)
    assert set(result.units) == {"f0_IW1_b0", "f0_IW2_b2"}
    assert calls == 2
    assert len(grids) == n_materialize + 1
    west, south, east, north = stack.dem.grid.bounds
    required = _union_demand()
    assert west <= required[0]
    assert south <= required[1]
    assert east >= required[2]
    assert north >= required[3]
