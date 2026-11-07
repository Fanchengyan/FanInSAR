"""Reference merge-layout publication at the coregister seam (P0074)."""

from __future__ import annotations

import json
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
from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
from faninsar.processing.geometry import ConstantDEM, RadarGeometryModel
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.mosaicking.grid import GeoGridSpec

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
# Linear rdr2geo: lon = 10 + 0.01 * range, lat = 45 + 0.01 * azimuth.
_COVERING_ROI = BoundingBox(9.99, 44.99, 10.08, 45.08)


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
    azimuth_time_interval_s: float = 1.0,
    slant_range_time_s: float = 0.005,
    range_sampling_rate_hz: float = 64_000_000.0,
) -> S1Swath:
    """Return a sub-swath carrying ``bursts``."""
    stop = bursts[-1].azimuth_time + timedelta(seconds=_LINES - 1)
    return S1Swath(
        swath=name,  # type: ignore[arg-type]
        polarization="VV",
        annotation_path="memory://annotation.xml",
        measurement_path="memory://measurement.tiff",
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


def _product(
    path: Path,
    *,
    swaths: tuple[S1Swath, ...],
) -> S1Product:
    """Return a product whose date id is taken from ``path``."""
    return S1Product(
        product_id=path.stem,
        source_path=path,
        mission_id="S1A",
        product_type="SLC",
        mode="IW",
        absolute_orbit=1,
        polarizations=("VV",),
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


def _layout_path(tmp_path: Path) -> Path:
    """Return the committed layout product directory for this Reference."""
    return tmp_path / "layout" / "20200101"


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
    assert (layout_dir / "item.json").is_file()
    item = _read_item(context.product.root)
    properties = item["properties"]
    layout_id = properties["faninsar:layout_id"]
    assert layout_id
    assert properties["faninsar:inputs"]["layout"] == layout_id

    layout = open_merge_layout(layout_dir)
    assert layout.scope.domain == "radar"
    assert layout.scope.reference_id == "20200101"
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
        layout = open_merge_layout(
            context.product.root.parent.parent / "layout" / "20200101"
        )
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
    layout = open_merge_layout(tmp_path / "layout" / "20200101")
    assert layout.scope.domain == "geo"
    # Geo bbox midpoint of windows (0,6) and (2,8) would cut at row 4.
    # Parent overlap [4, 8) midpoints at 6, so geo row 4 stays earlier.
    assert layout.owner[4, 0] == 1
    assert layout.owner[5, 0] == 1
    assert layout.owner[_PARENT_MIDPOINT, 0] == 2
    assert (tmp_path / "coreg" / "20200101-geo" / "item.json").is_file()
    item = _read_item(context.product.root)
    assert item["properties"]["faninsar:layout_id"]


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
