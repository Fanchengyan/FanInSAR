"""Reference merge-layout publication at the coregister seam (P0074)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.data.query import BoundingBox
from faninsar.io.storage.merge_layout import open_merge_layout
from faninsar.missions.protocols import DopplerCentroidPolynomial
from faninsar.missions.s1 import coregister as coreg_module
from faninsar.missions.s1.coregister import (
    S1ReferenceContext,
    S1ReferenceUnit,
    prepare_s1_reference,
)
from faninsar.missions.s1.merge_layout import (
    layout_product_dir,
    polygon_identity,
    track_identity,
)
from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
from faninsar.processing.geometry import ConstantDEM, RadarGeometryModel
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.mosaicking.grid import GeoGridSpec, build_geo_grid

REFERENCE = "S1A_IW_SLC__1SDV_20200101T000000.SAFE"
_ORIGIN = datetime(2020, 1, 1, tzinfo=UTC)
_LINES = 8
_SAMPLES = 8
_BURST_AZIMUTH_OFFSET = 4
_PARENT_MIDPOINT = _BURST_AZIMUTH_OFFSET + (_LINES - _BURST_AZIMUTH_OFFSET) // 2
_REAL_AZIMUTH_INTERVAL_S = 2.055556299999998e-3
# Measured IW1 burst-1 residual on annotation 2016-12-07 (~3.25e-4 px).
_REAL_BURST1_DT_S = 2.762667
_PHYSICAL_IDS = ("IW1-anx0", "IW1-anx1000000")
_TRACK_KEY = "1A"
# Linear rdr2geo: lon = 10 + 0.01 * range, lat = 45 + 0.01 * azimuth.
_COVERING_ROI = BoundingBox(9.99, 44.99, 10.08, 45.08)
_SECOND_POLYGON = BoundingBox(10.02, 45.02, 10.08, 45.08)


def _esa_measurement_path(swath: str, polarization: str) -> str:
    """Return an ESA-like measurement name (``vh`` sorts before ``vv``)."""
    pol = polarization.lower()
    return (
        f"memory://measurement/s1a-{swath.lower()}-slc-{pol}-"
        "20200101t000000-20200101t000026-000001-xxxxxx-001.tiff"
    )


def _orbit() -> OrbitMetadata:
    """Return a short linear orbit covering the fake bursts."""
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
    """Return one burst on the 8x8 test lattice."""
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


def _swath(
    *,
    name: str = "IW1",
    bursts: tuple[S1Burst, ...],
    polarization: str = "VV",
    azimuth_time_interval_s: float = 1.0,
    slant_range_time_s: float = 0.005,
    range_sampling_rate_hz: float = 64_000_000.0,
) -> S1Swath:
    """Return a sub-swath carrying ``bursts``."""
    stop = bursts[-1].azimuth_time + timedelta(seconds=_LINES - 1)
    return S1Swath(
        swath=name,  # type: ignore[arg-type]
        polarization=polarization,
        annotation_path="memory://annotation.xml",
        measurement_path=_esa_measurement_path(name, polarization),
        lines=_LINES * len(bursts),
        samples=_SAMPLES,
        lines_per_burst=_LINES,
        samples_per_burst=_SAMPLES,
        range_pixel_spacing_m=2.3,
        azimuth_pixel_spacing_m=14.0,
        azimuth_time_interval_s=azimuth_time_interval_s,
        slant_range_time_s=slant_range_time_s,
        range_sampling_rate_hz=range_sampling_rate_hz,
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


def _source_path(path: Path, *, track: str | None = _TRACK_KEY) -> Path:
    """Return a staged path that encodes the selection track key."""
    resolved = Path(path)
    if track is None or "tracks" in resolved.parts:
        return resolved
    return resolved.parent / "tracks" / track / "acquisitions" / resolved.name


def _product(
    path: Path,
    *,
    swaths: tuple[S1Swath, ...],
    track: str | None = _TRACK_KEY,
) -> S1Product:
    """Return a product whose date id is taken from ``path``."""
    return S1Product(
        product_id=path.stem,
        source_path=_source_path(path, track=track),
        mission_id="S1A",
        product_type="SLC",
        mode="IW",
        absolute_orbit=1,
        polarizations=tuple(dict.fromkeys(item.polarization for item in swaths)),
        swaths=swaths,
    )


def _two_burst_swath(**kwargs: object) -> S1Swath:
    """Return IW1 with two overlapping bursts (parent overlap [4, 8))."""
    bursts = (
        _burst(0, azimuth_time=_ORIGIN),
        _burst(1, azimuth_time=_ORIGIN + timedelta(seconds=_BURST_AZIMUTH_OFFSET)),
    )
    return _swath(bursts=bursts, **kwargs)


def _geometry(swath: S1Swath, burst: S1Burst) -> RadarGeometryModel:
    """Return a real radar geometry model so ROI support is not fail-closed."""
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
    """Map every radar sample to lon/lat; never a four-corner affine."""
    del dem
    azimuth = np.asarray(azimuth_index, dtype=np.float64)
    slant = np.asarray(range_index, dtype=np.float64)
    return SimpleNamespace(
        latitude_deg=45.0 + 0.01 * azimuth,
        longitude_deg=10.0 + 0.01 * slant,
        converged=np.ones(azimuth.shape, dtype=bool),
    )


def _fake_scene(
    path: Path, burst_index: int, *, swath: S1Swath | None = None
) -> MagicMock:
    """Return a production scene for one overlapping test burst."""
    scene = MagicMock()
    scene.scene_id = "20200101"
    scene.path = path
    resolved = swath if swath is not None else _two_burst_swath()
    burst = resolved.bursts[min(burst_index, len(resolved.bursts) - 1)]
    scene.burst = burst
    scene.swath = resolved
    samples = np.ones((_LINES, _SAMPLES), dtype=np.complex64)
    scene.array.samples = samples
    scene.array.valid_mask = np.ones((_LINES, _SAMPLES), dtype=bool)
    scene.array.row0 = 0
    scene.array.col0 = 0
    scene.geometry = _geometry(resolved, burst)
    return scene


@pytest.fixture
def two_burst_s1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch SAFE access for a two-burst Reference publication."""
    path = tmp_path / REFERENCE
    path.mkdir(parents=True)
    (path / "manifest.safe").write_text("acquired-support-canvas\n")
    product = _product(path, swaths=(_two_burst_swath(),))

    def fake_load(load_path: Path, **kwargs: object) -> MagicMock:
        return _fake_scene(Path(load_path), int(kwargs.get("burst_index", 0)))

    monkeypatch.setattr(
        f"{coreg_module.__name__}.open_acquisition_products",
        lambda paths, **_k: [product for _ in paths],
    )
    monkeypatch.setattr(f"{coreg_module.__name__}.load_production_scene", fake_load)
    monkeypatch.setattr(
        f"{coreg_module.__name__}.deramp",
        lambda samples, _carrier, **_k: np.asarray(samples),
    )
    monkeypatch.setattr(
        f"{coreg_module.__name__}.reramp",
        lambda samples, _carrier, **_k: np.asarray(samples),
    )
    monkeypatch.setattr(f"{coreg_module.__name__}._without_pixels", lambda scene: scene)
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_rdr2geo",
        _linear_rdr2geo,
    )
    return tmp_path


def _prepare_two_burst(tmp_path: Path, **kwargs: object) -> S1ReferenceContext:
    """Publish the two-burst Reference through the public coregister path."""
    options = {
        "swaths": ("IW1",),
        "bursts": {"IW1": [0, 1]},
        "dem": ConstantDEM(0.0),
    }
    options.update(kwargs)
    return prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101",
        **options,
    )


def _parent_midpoint(unit_a: S1ReferenceUnit, unit_b: S1ReferenceUnit) -> int:
    """Return the independent parent azimuth midpoint of two overlapping units."""
    lo = max(int(unit_a.parent_row0), int(unit_b.parent_row0))
    hi = min(
        int(unit_a.parent_row0) + int(unit_a.native_shape[0]),
        int(unit_b.parent_row0) + int(unit_b.native_shape[0]),
    )
    return lo + (hi - lo) // 2


def _layout_path(
    tmp_path: Path,
    *,
    reference_id: str = "20200101",
    polygon: int | None = None,
) -> Path:
    """Return the committed layout product directory for this Reference."""
    base = tmp_path / "layout" / reference_id
    if polygon is None:
        return base
    return base / f"p{polygon}"


def _read_item(path: Path) -> dict[str, object]:
    """Return the STAC item properties of a product directory."""
    return json.loads((path / "item.json").read_text(encoding="utf-8"))


def test_radar_reference_publishes_parent_azimuth_midpoint_layout(
    two_burst_s1: Path,
) -> None:
    """Public Reference publication persists the parent-native azimuth cut."""
    tmp_path = two_burst_s1
    context = _prepare_two_burst(tmp_path)
    earlier = context.units["f0_IW1_b0"]
    later = context.units["f0_IW1_b1"]
    assert earlier.parent_row0 == 0
    assert later.parent_row0 == _BURST_AZIMUTH_OFFSET
    expected_parent = _parent_midpoint(earlier, later)
    assert expected_parent == _PARENT_MIDPOINT

    layout_dir = _layout_path(tmp_path)
    assert layout_dir == tmp_path / "layout" / "20200101"
    assert "full-frame" not in layout_dir.parts
    assert not (layout_dir / "full-frame").exists()
    assert (layout_dir / "item.json").is_file()
    item = _read_item(context.product.root)
    properties = item["properties"]
    layout_id = properties["faninsar:layout_id"]
    assert layout_id
    assert properties["faninsar:inputs"]["layout"] == layout_id

    layout = open_merge_layout(layout_dir)
    assert layout.scope.domain == "radar"
    assert layout.scope.track_id == _TRACK_KEY
    assert layout.scope.channel == "vv"
    assert layout.scope.reference_id == "20200101"
    assert layout.scope.polygon_identity == "full-frame"
    assert layout.source_ids == _PHYSICAL_IDS
    assert item["assets"]["f0_IW1_b0"]["faninsar:burst_id"] == _PHYSICAL_IDS[0]
    assert item["assets"]["f0_IW1_b1"]["faninsar:burst_id"] == _PHYSICAL_IDS[1]
    assert len(layout.seams) == 1
    seam = layout.seams[0]
    assert seam.axis == "azimuth"
    assert seam.earlier_id == _PHYSICAL_IDS[0]
    assert seam.later_id == _PHYSICAL_IDS[1]
    crop0 = (earlier.crop or (0, 0, 0, 0))[0]
    scene_row0 = earlier.parent_row0 + crop0 - earlier.row_origin
    assert seam.boundary == expected_parent - scene_row0
    below = expected_parent - scene_row0 - 1
    at_cut = expected_parent - scene_row0
    assert layout.owner[below, 0] == 1
    assert layout.owner[at_cut, 0] == 2
    mask = context.product.read_valid_mask("f0_IW1_b0")
    assert mask is not None
    assert mask.shape == (_LINES, _SAMPLES)


def test_radar_layout_reopen_and_reuse_keep_parent_midpoint(
    two_burst_s1: Path,
) -> None:
    """Reopening and reusing the Reference does not move the parent cut."""
    tmp_path = two_burst_s1
    first = _prepare_two_burst(tmp_path)
    first_layout = open_merge_layout(_layout_path(tmp_path))
    reused = _prepare_two_burst(tmp_path)
    assert reused.write_id == first.write_id
    reopened = open_merge_layout(_layout_path(tmp_path))
    assert reopened.seams == first_layout.seams
    np.testing.assert_array_equal(reopened.owner, first_layout.owner)
    item = _read_item(reused.product.root)
    props = item["properties"]
    assert props["faninsar:layout_id"] == props["faninsar:inputs"]["layout"]


def test_radar_layout_parent_midpoint_is_invariant_to_roi_crop(
    two_burst_s1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ROI clipping must not rewrite the parent-native azimuth midpoint."""
    tmp_path = two_burst_s1

    def _prepare_with_crop(
        crop: tuple[int, int, int, int], output_name: str
    ) -> S1ReferenceContext:
        monkeypatch.setattr(coreg_module, "_roi_burst_window", lambda *_a, **_k: crop)
        monkeypatch.setattr(
            coreg_module, "reference_window_crop", lambda window, _shape: window
        )
        output_scene = tmp_path / output_name / "coreg" / "20200101"
        return prepare_s1_reference(
            tmp_path / REFERENCE,
            output_scene=output_scene,
            dem=ConstantDEM(0.0),
            swaths=("IW1",),
            bursts={"IW1": [0, 1]},
            roi=_COVERING_ROI,
        )

    def _seam_parent(context: S1ReferenceContext) -> int:
        item = _read_item(context.product.root)
        layout = open_merge_layout(_layout_path(context.product.root.parent.parent))
        assert item["properties"]["faninsar:layout_id"]
        assert layout.source_ids == _PHYSICAL_IDS
        unit = context.units["f0_IW1_b0"]
        crop = unit.crop or (0, 0, 0, 0)
        scene_row0 = unit.parent_row0 + crop[0] - unit.row_origin
        return layout.seams[0].boundary + scene_row0

    full = _prepare_with_crop((0, _LINES, 0, _SAMPLES), "20200101-full")
    parent_full = _parent_midpoint(full.units["f0_IW1_b0"], full.units["f0_IW1_b1"])
    seam_full = _seam_parent(full)
    cropped = _prepare_with_crop((1, _LINES, 0, _SAMPLES), "20200101-crop")
    parent_crop = _parent_midpoint(
        cropped.units["f0_IW1_b0"], cropped.units["f0_IW1_b1"]
    )
    seam_crop = _seam_parent(cropped)
    assert parent_full == parent_crop == _PARENT_MIDPOINT
    assert seam_full == seam_crop == _PARENT_MIDPOINT


def test_geo_layout_cuts_on_lut_native_coords_not_geo_bbox(
    two_burst_s1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Geo owner follows parent-native LUT coordinates, not the Geo bbox midpoint."""
    tmp_path = two_burst_s1
    height, width = 8, 4
    grid = GeoGridSpec(
        crs="EPSG:4326",
        transform=(10.0, 0.01, 0.0, 45.08, 0.0, -0.01),
        height=height,
        width=width,
        resolution_m=(0.01, 0.01),
    )

    def _lut(burst_index: int) -> Geo2RdrLUT:
        if burst_index == 0:
            row0, rows = 0, 6
            az = np.arange(rows, dtype=np.float64)[:, None] * np.ones(
                (1, width), dtype=np.float64
            )
        else:
            row0, rows = 2, 6
            geo_rows = row0 + np.arange(rows, dtype=np.float64)
            az = (geo_rows - _BURST_AZIMUTH_OFFSET)[:, None] * np.ones(
                (1, width), dtype=np.float64
            )
        valid = (az >= 0.0) & (az < _LINES)
        rg = np.zeros((rows, width), dtype=np.float64)
        return Geo2RdrLUT(
            az_full=az,
            rg_full=rg,
            valid=valid,
            full_radar_shape=(_LINES, _SAMPLES),
            height_m=0.0,
            row0=row0,
            col0=0,
        )

    def fake_lut(_scene: MagicMock, **_kwargs: object) -> Geo2RdrLUT:
        return _lut(int(_scene.burst.index))

    def fake_geocode(
        *_args: object, reference_lut: Geo2RdrLUT, **_kwargs: object
    ) -> tuple[np.ndarray, None]:
        return np.ones(reference_lut.valid.shape, dtype=np.complex64), None

    monkeypatch.setattr(coreg_module, "build_reference_geo_lut", fake_lut)
    monkeypatch.setattr(
        "faninsar.processing.geocoding.geo_modes.geocode_reference_slc_chunked",
        fake_geocode,
    )
    context = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101-geo",
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0, 1]},
        coregistration_grid="geo",
        geo_grid=grid,
    )
    layout = open_merge_layout(_layout_path(tmp_path))
    assert layout.scope.domain == "geo"
    assert layout.scope.track_id == _TRACK_KEY
    # Geo bbox midpoint of windows (0,6) and (2,8) would cut at row 4.
    # Parent overlap [4, 8) midpoints at 6, so geo row 4 stays earlier.
    assert layout.owner[4, 0] == 1
    assert layout.owner[5, 0] == 1
    assert layout.owner[_PARENT_MIDPOINT, 0] == 2
    assert (tmp_path / "coreg" / "20200101-geo" / "item.json").is_file()
    item = _read_item(context.product.root)
    assert item["properties"]["faninsar:layout_id"]


def test_geo_prepare_twice_reuses_projected_lut_window_not_native_shape(
    two_burst_s1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second geo prepare reuses the LUT window, not the native radar burst."""
    tmp_path = two_burst_s1
    roi = _COVERING_ROI
    grid = build_geo_grid(
        [((roi.left, roi.bottom), (roi.right, roi.top))],
        resolution_m=(30.0, 30.0),
    )
    window_rows, window_cols = 6, 9
    row0, col0 = 3, 2
    assert tuple(grid.shape) != (_LINES, _SAMPLES)
    assert (row0 + window_rows) <= int(grid.shape[0])
    assert (col0 + window_cols) <= int(grid.shape[1])
    lut = Geo2RdrLUT(
        az_full=np.zeros((window_rows, window_cols), dtype=np.float64),
        rg_full=np.zeros((window_rows, window_cols), dtype=np.float64),
        valid=np.ones((window_rows, window_cols), dtype=bool),
        full_radar_shape=(_LINES, _SAMPLES),
        height_m=0.0,
        row0=row0,
        col0=col0,
    )
    geocode_calls = {"n": 0}

    def fake_lut(_scene: MagicMock, **_kwargs: object) -> Geo2RdrLUT:
        return lut

    def fake_geocode(
        *_args: object, reference_lut: Geo2RdrLUT, **_kwargs: object
    ) -> tuple[np.ndarray, None]:
        geocode_calls["n"] += 1
        return np.ones(reference_lut.valid.shape, dtype=np.complex64), None

    monkeypatch.setattr(coreg_module, "build_reference_geo_lut", fake_lut)
    monkeypatch.setattr(
        "faninsar.processing.geocoding.geo_modes.geocode_reference_slc_chunked",
        fake_geocode,
    )
    options = {
        "output_scene": tmp_path / "coreg" / "20200101",
        "dem": ConstantDEM(0.0),
        "swaths": ("IW1",),
        "bursts": {"IW1": [0]},
        "roi": roi,
        "coregistration_grid": "geo",
        "geo_grid": grid,
    }
    first = prepare_s1_reference(tmp_path / REFERENCE, **options)
    assert geocode_calls["n"] == 1
    unit = first.units["f0_IW1_b0"]
    placed = first.product.unit_map()["f0_IW1_b0"]
    assert unit.lut is not None
    assert unit.native_shape == (_LINES, _SAMPLES)
    assert tuple(unit.lut.shape) == (window_rows, window_cols)
    assert tuple(placed.shape) == (window_rows, window_cols)
    assert tuple(placed.shape) != tuple(unit.native_shape)
    assert (placed.row_origin, placed.col_origin) == (row0, col0)
    assert tuple(first.product.grid_shape) == tuple(grid.shape)
    mask = first.product.read_valid_mask("f0_IW1_b0")
    assert mask is not None
    assert mask.shape == (window_rows, window_cols)
    layout = open_merge_layout(_layout_path(tmp_path))
    assert layout.scope.domain == "geo"

    second = prepare_s1_reference(tmp_path / REFERENCE, **options)
    assert geocode_calls["n"] == 1
    assert second.write_id == first.write_id
    reused = second.units["f0_IW1_b0"]
    stored = second.product.unit_map()["f0_IW1_b0"]
    assert reused.lut is not None
    assert reused.native_shape == (_LINES, _SAMPLES)
    assert tuple(reused.lut.shape) == (window_rows, window_cols)
    assert tuple(stored.shape) == (window_rows, window_cols)
    assert (reused.row_origin, reused.col_origin) == (row0, col0)
    assert (stored.row_origin, stored.col_origin) == (row0, col0)
    assert second.product.grid_identity == first.product.grid_identity
    reopened = open_merge_layout(_layout_path(tmp_path))
    assert reopened.scope.domain == "geo"
    assert reopened.scope.reference_id == layout.scope.reference_id


def test_incompatible_range_sampling_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-swath placement must not silently round mismatched range lattices."""
    path = tmp_path / REFERENCE
    path.mkdir(parents=True)
    (path / "manifest.safe").write_text("acquired-support-canvas\n")
    iw1 = _swath(
        name="IW1",
        bursts=(_burst(0, azimuth_time=_ORIGIN),),
        range_sampling_rate_hz=64_000_000.0,
        slant_range_time_s=0.005,
    )
    iw2 = _swath(
        name="IW2",
        bursts=(_burst(0, azimuth_time=_ORIGIN),),
        range_sampling_rate_hz=50_000_000.0,
        slant_range_time_s=0.0053,
    )
    product = _product(path, swaths=(iw1, iw2))
    monkeypatch.setattr(
        f"{coreg_module.__name__}.open_acquisition_products",
        lambda paths, **_k: [product for _ in paths],
    )
    monkeypatch.setattr(
        f"{coreg_module.__name__}.load_production_scene",
        lambda *_a, **_k: _fake_scene(path, 0),
    )
    monkeypatch.setattr(f"{coreg_module.__name__}._without_pixels", lambda scene: scene)
    with pytest.raises(InvalidProcessingStateError, match="range sampling"):
        prepare_s1_reference(
            path,
            output_scene=tmp_path / "coreg" / "20200101",
            dem=ConstantDEM(0.0),
            swaths=("IW1", "IW2"),
            bursts={"IW1": [0], "IW2": [0]},
        )


def test_annotation_scale_azimuth_fraction_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real annotation-scale azimuth residuals are a common lattice, not a reject."""
    path = tmp_path / REFERENCE
    path.mkdir(parents=True)
    (path / "manifest.safe").write_text("acquired-support-canvas\n")
    bursts = (
        _burst(0, azimuth_time=_ORIGIN),
        _burst(
            1,
            azimuth_time=_ORIGIN + timedelta(seconds=_REAL_BURST1_DT_S),
        ),
    )
    swath = _swath(
        bursts=bursts,
        azimuth_time_interval_s=_REAL_AZIMUTH_INTERVAL_S,
    )
    product = _product(path, swaths=(swath,))

    def fake_load(load_path: Path, **kwargs: object) -> MagicMock:
        return _fake_scene(
            Path(load_path),
            int(kwargs.get("burst_index", 0)),
            swath=swath,
        )

    monkeypatch.setattr(
        f"{coreg_module.__name__}.open_acquisition_products",
        lambda paths, **_k: [product for _ in paths],
    )
    monkeypatch.setattr(f"{coreg_module.__name__}.load_production_scene", fake_load)
    monkeypatch.setattr(
        f"{coreg_module.__name__}.deramp",
        lambda samples, _carrier, **_k: np.asarray(samples),
    )
    monkeypatch.setattr(
        f"{coreg_module.__name__}.reramp",
        lambda samples, _carrier, **_k: np.asarray(samples),
    )
    monkeypatch.setattr(f"{coreg_module.__name__}._without_pixels", lambda scene: scene)
    context = prepare_s1_reference(
        path,
        output_scene=tmp_path / "coreg" / "20200101",
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0, 1]},
    )
    later = context.units["f0_IW1_b1"]
    raw = _REAL_BURST1_DT_S / _REAL_AZIMUTH_INTERVAL_S
    assert later.parent_row0 == round(raw)
    assert abs(raw - later.parent_row0) < 1.0e-3


def test_absolute_orbit_is_not_a_track_key(tmp_path: Path) -> None:
    """MergeScope track_id must be the selection group key, not absolute orbit."""
    product = replace(
        _product(tmp_path / REFERENCE, swaths=(_two_burst_swath(),), track=None),
        absolute_orbit=0,
    )
    assert product.absolute_orbit == 0
    with pytest.raises(InvalidProcessingStateError, match="relative orbit"):
        track_identity(product)
    staged = _product(tmp_path / REFERENCE, swaths=(_two_burst_swath(),))
    assert track_identity(staged) == _TRACK_KEY
    assert track_identity(staged) != f"{staged.mission_id}-{staged.absolute_orbit}"


def test_layout_product_dir_uses_pn_not_polygon_hash(tmp_path: Path) -> None:
    """Formation/views spelling is layout/ref[/pN], never a WKT hash suffix."""
    reference_id = "20200101"
    unsuffixed = layout_product_dir(tmp_path / "coreg" / reference_id, reference_id)
    assert unsuffixed == tmp_path / "layout" / reference_id
    assert "full-frame" not in unsuffixed.parts
    hashed = polygon_identity(_COVERING_ROI)
    p0 = layout_product_dir(tmp_path / "coreg" / reference_id / "p0", reference_id)
    p1 = layout_product_dir(tmp_path / "coreg" / reference_id / "p1", reference_id)
    assert p0 == tmp_path / "layout" / reference_id / "p0"
    assert p1 == tmp_path / "layout" / reference_id / "p1"
    assert p0.name != hashed
    assert hashed != "p0"


def test_two_polygons_publish_independent_layouts(
    two_burst_s1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same Reference date must not share one layout directory across polygons."""
    tmp_path = two_burst_s1
    monkeypatch.setattr(
        coreg_module, "_roi_burst_window", lambda *_a, **_k: (0, _LINES, 0, _SAMPLES)
    )
    monkeypatch.setattr(
        coreg_module, "reference_window_crop", lambda window, _shape: window
    )
    first = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101" / "p0",
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0, 1]},
        roi=_COVERING_ROI,
    )
    second = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101" / "p1",
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0, 1]},
        roi=_SECOND_POLYGON,
    )
    path_a = _layout_path(tmp_path, polygon=0)
    path_b = _layout_path(tmp_path, polygon=1)
    assert path_a != path_b
    assert path_a.name == "p0"
    assert path_b.name == "p1"
    assert path_a.parent == tmp_path / "layout" / "20200101"
    assert (path_a / "item.json").is_file()
    assert (path_b / "item.json").is_file()
    layout_a = open_merge_layout(path_a)
    layout_b = open_merge_layout(path_b)
    assert layout_a.scope.track_id == layout_b.scope.track_id == _TRACK_KEY
    assert layout_a.scope.polygon_identity != layout_b.scope.polygon_identity
    assert layout_a.scope.polygon_identity == polygon_identity(_COVERING_ROI)
    assert path_a.name != layout_a.scope.polygon_identity
    assert first.product.inputs["layout"] != second.product.inputs["layout"]


def test_explicit_all_invalid_burst_is_excluded_from_layout(
    two_burst_s1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An annotation mask of all -1 is not a full-burst owner."""
    tmp_path = two_burst_s1
    invalid = tuple([-1] * _LINES)

    def fake_load(load_path: Path, **kwargs: object) -> MagicMock:
        scene = _fake_scene(Path(load_path), int(kwargs.get("burst_index", 0)))
        if int(kwargs.get("burst_index", 0)) == 1:
            scene.burst = replace(
                scene.burst,
                first_valid_sample=invalid,
                last_valid_sample=invalid,
            )
        return scene

    monkeypatch.setattr(f"{coreg_module.__name__}.load_production_scene", fake_load)
    context = _prepare_two_burst(tmp_path)
    layout = open_merge_layout(_layout_path(tmp_path))
    assert layout.source_ids == (_PHYSICAL_IDS[0],)
    assert layout.seams == ()
    assert not np.any(layout.owner == 2)
    assert context.units["f0_IW1_b1"].scene.burst.first_valid_sample == invalid


def test_all_invalid_reference_bursts_reject_layout(
    two_burst_s1: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Reference with only explicit empty masks cannot publish a layout."""
    tmp_path = two_burst_s1
    invalid = tuple([-1] * _LINES)

    def fake_load(load_path: Path, **kwargs: object) -> MagicMock:
        scene = _fake_scene(Path(load_path), int(kwargs.get("burst_index", 0)))
        scene.burst = replace(
            scene.burst,
            first_valid_sample=invalid,
            last_valid_sample=invalid,
        )
        return scene

    monkeypatch.setattr(f"{coreg_module.__name__}.load_production_scene", fake_load)
    with pytest.raises(InvalidProcessingStateError, match="native validity"):
        _prepare_two_burst(tmp_path)


def _dual_pol_product(path: Path) -> S1Product:
    """Return a 1SDV-like product whose unfiltered IW1 swath is VH."""
    product = _product(
        path,
        swaths=(
            _two_burst_swath(polarization="VH"),
            _two_burst_swath(polarization="VV"),
        ),
    )
    first = product.swath("IW1")
    assert first.polarization == "VH"
    assert _measurement_channel(first.measurement_path) == "vh"
    assert product.polarizations == ("VH", "VV")
    return product


def _measurement_channel(path: object) -> str:
    """Return the ESA ``-slc-<pol>-`` token from a measurement path."""
    text = str(path).lower()
    assert "-slc-vv-" in text or "-slc-vh-" in text, (
        f"measurement path has no vv/vh token: {path}"
    )
    return "vv" if "-slc-vv-" in text else "vh"


def _patch_dual_pol_open(monkeypatch: pytest.MonkeyPatch, product: S1Product) -> None:
    """Open/load the requested polarizations, not the VH-first annotation."""

    def fake_open(paths: object, **kwargs: object) -> list[S1Product]:
        selected = kwargs.get("polarizations")
        if not selected or selected == "all":
            opened = product
        else:
            wanted = {str(value).upper() for value in selected}  # type: ignore[union-attr]
            filtered = tuple(
                item for item in product.swaths if item.polarization in wanted
            )
            opened = replace(
                product,
                swaths=filtered,
                polarizations=tuple(
                    pol for pol in product.polarizations if pol in wanted
                ),
            )
        return [opened for _ in paths]  # type: ignore[union-attr]

    def fake_load(load_path: Path, **kwargs: object) -> MagicMock:
        prod = kwargs.get("product")
        swath_obj = (
            prod.swath(str(kwargs.get("swath", "IW1")))  # type: ignore[union-attr]
            if prod is not None
            else product.swath("IW1")
        )
        return _fake_scene(
            Path(load_path),
            int(kwargs.get("burst_index", 0)),
            swath=swath_obj,
        )

    monkeypatch.setattr(f"{coreg_module.__name__}.open_acquisition_products", fake_open)
    monkeypatch.setattr(f"{coreg_module.__name__}.load_production_scene", fake_load)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (("VV",), "vv"),
        (("VH",), "vh"),
        ("VV", "vv"),
        ("VH", "vh"),
    ],
)
def test_requested_polarization_uses_that_measurement_path(
    two_burst_s1: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: tuple[str, ...] | str,
    expected: str,
) -> None:
    """Requested VV/VH must select that measurement TIFF, not the first annotation."""
    tmp_path = two_burst_s1
    path = tmp_path / REFERENCE
    _patch_dual_pol_open(monkeypatch, _dual_pol_product(path))
    context = prepare_s1_reference(
        path,
        output_scene=tmp_path / "coreg" / "20200101",
        dem=ConstantDEM(0.0),
        swaths=("IW1",),
        bursts={"IW1": [0, 1]},
        polarizations=requested,
    )
    layout = open_merge_layout(_layout_path(tmp_path))
    unit = context.units["f0_IW1_b0"]
    assert layout.scope.channel == expected
    assert context.polarizations == (expected.upper(),)
    assert unit.scene.swath.polarization == expected.upper()
    assert _measurement_channel(unit.scene.swath.measurement_path) == expected


@pytest.mark.parametrize("requested", [None, "all", ("VV", "VH"), "VV+VH"])
def test_dual_pol_coregistration_rejects_ambiguous_channels(
    two_burst_s1: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str | tuple[str, ...] | None,
) -> None:
    """One layout cannot silently process the first of several dual-pol channels."""
    tmp_path = two_burst_s1
    path = tmp_path / REFERENCE
    _patch_dual_pol_open(monkeypatch, _dual_pol_product(path))
    with pytest.raises(InvalidProcessingStateError, match="ambiguous"):
        prepare_s1_reference(
            path,
            output_scene=tmp_path / "coreg" / "20200101",
            dem=ConstantDEM(0.0),
            swaths=("IW1",),
            bursts={"IW1": [0, 1]},
            polarizations=requested,
        )
