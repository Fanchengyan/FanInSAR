"""Stack reference-point persistence, inversion, and radar mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tests.support.local_stack import LocalStack

from faninsar import Pairs
from faninsar.data.query.points import Points
from faninsar.io.storage.ifg_store import (
    InterferogramArtifactStore,
    write_ifg_artifact,
    write_unwrapped_artifact,
)
from faninsar.io.storage.product import open_product
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import ConstantDEM
from faninsar.stack.identity import PREPARATION_EXCLUDED_SCIENCE_FIELDS
from faninsar.stack.session import Stack
from faninsar.timeseries.results import invert_unwrapped_pairs

GEO_TRANSFORM = (1.0, 0.0, 10.0, 0.0, -1.0, 50.0)
GRID_IDENTITY = "a" * 64
REF_LONLAT = Points([[10.5, 49.5], [11.5, 49.5]], crs="EPSG:4326")


def _stack(
    tmp_path: Path, dates: tuple[str, ...] = ("20240101", "20240113", "20240125")
) -> Stack:
    paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        path.mkdir()
        paths.append(path)
    names = [f"{dates[index]}_{dates[index + 1]}" for index in range(len(dates) - 1)]
    if len(dates) == 3:
        names.append(f"{dates[0]}_{dates[2]}")
    stack = LocalStack.from_paths(
        paths,
        work_dir=tmp_path / "out",
        pairs=Pairs.from_names(names),
        multilook=(1, 1),
        coreg_mode="geometry",
        coregistration_grid="geo",
        dem=ConstantDEM(0.0),
        device="cpu",
    )
    stack.prepare_data()
    return stack


def _write_geo_unwrapped(
    stack: Stack,
    phases: dict[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
) -> None:
    looks = stack.config.multilook
    for pair_id, phase in phases.items():
        extras = {} if mask is None else {"valid_mask": mask}
        store = write_ifg_artifact(
            stack.config.work_dir / "ifg" / f"ml_{looks.range}x{looks.azimuth}" / pair_id,
            pair=tuple(pair_id.split("_")),  # type: ignore[arg-type]
            looks=looks,
            domain="geo",
            wavelength_m=0.05546576,
            grid_identity=GRID_IDENTITY,
            crs="EPSG:4326",
            transform=GEO_TRANSFORM,
            filter_name="none",
            filter_parameters={},
            source_manifest_digests={"primary": "a" * 64, "secondary": "b" * 64},
            complex_ifg=np.exp(1j * np.nan_to_num(phase)).astype(np.complex64),
            coherence=np.ones(phase.shape, dtype=np.float32),
            valid_mask=mask,
        )
        write_unwrapped_artifact(
            store.root,
            unwrapped_phase=phase,
            connected_components=np.zeros(phase.shape, dtype=np.int32),
            method="test",
            method_parameters={},
            ifg_manifest_digest=store.write_id,
            extra_layers=extras or None,
        )


def test_set_ref_points_validates_atomically_and_reopens(tmp_path: Path) -> None:
    """Setter copies, resets common, persists, and rolls back a failed save."""
    stack = _stack(tmp_path, dates=("20240101", "20240113"))
    points = Points([[100.12, 31.45], [100.18, 31.50]], dtype=np.float64)
    stack.set_ref_points(points, policy="per_image")
    assert stack.ref_policy == "per_image"
    returned = stack.ref_points
    assert returned is not None
    assert returned.dtype == np.dtype(np.float64)
    returned.values[0, 0] = 0.0
    assert stack.ref_points.values[0, 0] != 0.0
    stack.set_ref_points(points)
    assert stack.ref_policy == "common"
    with pytest.raises(ValueError, match="2 columns"):
        stack.set_ref_points(Points([]))
    with pytest.raises(ValueError, match="finite"):
        stack.set_ref_points(Points([[np.nan, 1.0]]))
    assert stack.ref_points is not None
    reopened = Stack.open(stack.config.work_dir, device="cpu")
    np.testing.assert_allclose(reopened.ref_points.values, points.values)
    assert reopened.ref_policy == "common"
    stack.set_ref_points(None)
    assert stack.ref_points is None
    assert stack.ref_policy == "common"


def test_set_ref_points_rolls_back_failed_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed identity write must not leave half-updated in-memory settings."""
    stack = _stack(tmp_path, dates=("20240101", "20240113"))
    original = Points([[12.0, 48.0]])
    stack.set_ref_points(original, policy="per_image")

    def _fail(_stack: Stack) -> Path:
        message = "disk full"
        raise OSError(message)

    monkeypatch.setattr("faninsar.stack.identity.save_stack_identity", _fail)
    with pytest.raises(OSError, match="disk full"):
        stack.set_ref_points(Points([[13.0, 49.0]]), policy="common")
    np.testing.assert_allclose(stack.ref_points.values, [[12.0, 48.0]])
    assert stack.ref_policy == "per_image"


def test_reference_settings_are_excluded_from_prepare_and_coreg(tmp_path: Path) -> None:
    """Changing refs must not reject prepare or alter coreg identity."""
    stack = _stack(tmp_path, dates=("20240101", "20240113"))
    assert "ref_points" in PREPARATION_EXCLUDED_SCIENCE_FIELDS
    before = stack._coreg_science_config()
    stack.set_ref_points(Points([[10.5, 49.5]]))
    after = stack._coreg_science_config()
    assert before == after
    stack.prepare_data()
    document = json.loads(
        (stack.config.work_dir / "inputs" / "request.json").read_text()
    )
    assert "ref_points" in document.get("science", {})
    stack.set_ref_points(None)
    stack.prepare_data()


def test_invert_subtracts_before_sbas_on_both_branches(tmp_path: Path) -> None:
    """Constant pair offsets are removed before SBAS; stored unwraps stay put."""
    stack = _stack(tmp_path)
    shape = (3, 4)
    phases = {
        "20240101_20240113": np.full(shape, 10.0, dtype=np.float32),
        "20240113_20240125": np.full(shape, 20.0, dtype=np.float32),
        "20240101_20240125": np.full(shape, 30.0, dtype=np.float32),
    }
    originals = {key: np.array(value, copy=True) for key, value in phases.items()}
    _write_geo_unwrapped(stack, phases)
    stack.set_ref_points(Points([[10.5, 49.5]], crs="EPSG:4326"))
    memory = stack.invert_timeseries(pair_phases=phases, ref_points="inherit")
    np.testing.assert_allclose(memory.phase_increments_rad, 0.0, atol=1e-5)
    for key, value in phases.items():
        np.testing.assert_array_equal(value, originals[key])
    assert memory.metadata["reference"]["offsets"][
        "20240101_20240113"
    ] == pytest.approx(10.0)

    persisted = stack.invert_timeseries(overwrite=True)
    np.testing.assert_allclose(persisted.phase_increments_rad, 0.0, atol=1e-5)
    store = InterferogramArtifactStore.open(
        stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113"
    )
    np.testing.assert_allclose(store.read_unwrapped().unwrapped_phase, 10.0)
    assert persisted.metadata["reference"]["policy"] == "common"


def test_common_versus_per_image_and_all_invalid(tmp_path: Path) -> None:
    """Common uses the intersection; per-image uses each pair; all-invalid errors."""
    stack = _stack(tmp_path)
    pair_a = np.array([[1.0, 3.0], [0.0, 0.0]], dtype=np.float32)
    pair_b = np.array([[1.0, np.nan], [0.0, 0.0]], dtype=np.float32)
    pair_c = np.array([[2.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    phases = {
        "20240101_20240113": pair_a,
        "20240113_20240125": pair_b,
        "20240101_20240125": pair_c,
    }
    _write_geo_unwrapped(stack, phases)
    stack.set_ref_points(REF_LONLAT, policy="common")
    common = stack.invert_timeseries(overwrite=True)
    assert common.metadata["reference"]["offsets"][
        "20240101_20240113"
    ] == pytest.approx(1.0)
    stack.set_ref_points(REF_LONLAT, policy="per_image")
    per_image = stack.invert_timeseries(overwrite=True)
    assert per_image.metadata["reference"]["offsets"][
        "20240101_20240113"
    ] == pytest.approx(2.0)
    assert per_image.metadata["reference"]["offsets"][
        "20240113_20240125"
    ] == pytest.approx(1.0)
    dead = Points([[10.5, 46.5]], crs="EPSG:4326")
    stack.set_ref_points(dead)
    with pytest.raises(ValueError, match="valid"):
        stack.invert_timeseries(overwrite=True)


def test_reuse_includes_reference_definition(tmp_path: Path) -> None:
    """Unchanged refs reuse; changed/cleared/old-unreferenced results do not."""
    stack = _stack(tmp_path)
    phases = {
        "20240101_20240113": np.full((2, 2), 0.2, dtype=np.float32),
        "20240113_20240125": np.full((2, 2), 0.3, dtype=np.float32),
        "20240101_20240125": np.full((2, 2), 0.5, dtype=np.float32),
    }
    _write_geo_unwrapped(stack, phases)
    first = stack.invert_timeseries()
    assert "reference" not in first.metadata
    reused = stack.invert_timeseries()
    assert reused.canonical_path == first.canonical_path
    stack.set_ref_points(Points([[10.5, 49.5]], crs="EPSG:4326"))
    with pytest.raises(InvalidProcessingStateError, match="reference-point"):
        stack.invert_timeseries()
    referenced = stack.invert_timeseries(overwrite=True)
    assert referenced.metadata["reference"]["policy"] == "common"
    again = stack.invert_timeseries()
    assert again.canonical_path == referenced.canonical_path
    stack.set_ref_points(None)
    with pytest.raises(InvalidProcessingStateError, match="reference-point"):
        stack.invert_timeseries()
    disabled = stack.invert_timeseries(overwrite=True)
    assert "reference" not in disabled.metadata


def _roundtrip_geometry() -> object:
    """Return a synthetic geometry that round-trips rdr2geo/geo2rdr."""
    from datetime import UTC, datetime, timedelta

    from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
    from faninsar.processing.geometry import RadarGeometryModel
    from faninsar.processing.geometry.coordinates import RadarGrid

    epoch = datetime(2016, 12, 7, 11, 18, 0, tzinfo=UTC)
    radius = 7_071_000.0
    omega = 0.001
    vectors = []
    for index in range(21):
        t = 10.0 * index
        angle = omega * t
        vectors.append(
            OrbitStateVector(
                time=epoch + timedelta(seconds=t),
                position_m=(
                    radius * np.cos(angle),
                    0.0,
                    radius * np.sin(angle),
                ),
                velocity_m_s=(
                    -radius * omega * np.sin(angle),
                    0.0,
                    radius * omega * np.cos(angle),
                ),
            )
        )
    orbit = OrbitMetadata(
        reference_frame="ITRF",
        source="synthetic",
        vectors=tuple(vectors),
    )
    grid = RadarGrid(
        shape=(64, 128),
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=vectors[5].time,
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(grid, orbit)


def test_radar_mapping_uses_dem_and_rejects_missing_height() -> None:
    """Radar mapping samples DEM heights and does not substitute zero."""
    from faninsar.processing.geometry.prepare_production import run_rdr2geo
    from faninsar.stack.reference import map_radar_points

    geometry = _roundtrip_geometry()
    az, rg = 1.0, 20.0
    geo = run_rdr2geo(
        geometry,
        np.array([az], dtype=np.float64),
        np.array([rg], dtype=np.float64),
        device="cpu",
        max_iter=40,
    )
    if not bool(geo.converged[0]):
        pytest.skip("synthetic geometry did not converge for rdr2geo")
    points = Points(
        [[float(geo.longitude_deg[0]), float(geo.latitude_deg[0])]],
        crs="EPSG:4326",
        dtype=np.float64,
    )
    mapping = map_radar_points(
        points,
        geometry=geometry,
        dem=ConstantDEM(float(geo.height_m[0])),
        origin=(0, 0),
        looks=(1, 1),
        shape=(64, 128),
        device="cpu",
    )
    assert mapping.mapped[0]
    assert mapping.row_frac[0] == pytest.approx(az, abs=0.6)
    assert mapping.col_frac[0] == pytest.approx(rg, abs=0.6)
    cropped = map_radar_points(
        points,
        geometry=geometry,
        dem=ConstantDEM(float(geo.height_m[0])),
        origin=(0, 4),
        looks=(2, 1),
        shape=(64, 64),
        device="cpu",
    )
    assert cropped.mapped[0]
    assert cropped.col_frac[0] == pytest.approx((rg - 4.0 - 0.5) / 2.0, abs=0.6)
    with pytest.raises(ValueError, match="DEM"):
        map_radar_points(
            points,
            geometry=geometry,
            dem=None,
            origin=(0, 0),
            looks=(1, 1),
            shape=(64, 128),
        )


def test_map_radar_points_skips_nonfinite_heights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NaN DEM heights are invalid locally and never enter geo2rdr."""
    from types import SimpleNamespace

    captured: dict[str, np.ndarray] = {}

    def _geo2rdr(
        _geometry: object,
        latitude: np.ndarray,
        _longitude: np.ndarray,
        height: np.ndarray,
        device: str = "cpu",
        **_kwargs: object,
    ) -> SimpleNamespace:
        del _geometry, device, _kwargs
        captured["height"] = np.asarray(height, dtype=np.float64)
        captured["n"] = int(np.asarray(latitude).size)
        return SimpleNamespace(
            azimuth_index=np.full(captured["n"], 10.0),
            range_index=np.full(captured["n"], 20.0),
            converged=np.ones(captured["n"], dtype=bool),
        )

    class _PartialDEM:
        def sample(
            self, latitude_deg: np.ndarray, longitude_deg: np.ndarray
        ) -> np.ndarray:
            del longitude_deg
            values = np.zeros(np.shape(latitude_deg), dtype=np.float64)
            values.reshape(-1)[0] = np.nan
            return values

    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_geo2rdr",
        _geo2rdr,
    )
    from faninsar.stack.reference import map_radar_points

    points = Points([[10.0, 20.0], [11.0, 21.0]], crs="EPSG:4326", dtype=np.float64)
    mapping = map_radar_points(
        points,
        geometry=object(),
        dem=_PartialDEM(),
        origin=(0, 0),
        looks=(1, 1),
        shape=(64, 128),
        device="cpu",
    )
    assert captured["n"] == 1
    assert np.all(np.isfinite(captured["height"]))
    assert not mapping.mapped[0]
    assert mapping.mapped[1]


def test_native_origin_combines_row_origin_and_look_origin(tmp_path: Path) -> None:
    """row_origin and look_origin are independent terms of one native origin."""
    from faninsar.io.storage.ifg_store import ifg_sample_placement
    from faninsar.stack.reference import native_layer_origin, origin_from_store

    looks = (3, 2)
    store = write_ifg_artifact(
        tmp_path / "ifg" / "ml_3x2" / "20240101_20240113",
        pair=("20240101", "20240113"),
        looks=looks,
        row_origin=10,
        col_origin=20,
        look_origin=(1, 2),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": "a" * 64, "secondary": "b" * 64},
        complex_ifg=np.ones((4, 6), dtype=np.complex64),
        coherence=np.ones((4, 6), dtype=np.float32),
    )
    expected = native_layer_origin(
        row_origin=10, col_origin=20, look_origin=(1, 2), looks=looks
    )
    assert expected == (12, 26)
    assert origin_from_store(store) == expected
    center, spacing = ifg_sample_placement(store)
    assert spacing == (2.0, 3.0)
    assert center == pytest.approx((12.5, 27.0))


def test_unreferenced_sbas_still_matches_low_level_solver(tmp_path: Path) -> None:
    """Disabled refs leave invert_timeseries identical to invert_unwrapped_pairs."""
    stack = _stack(tmp_path)
    phases = {
        "20240101_20240113": np.full((2, 2), 0.1, dtype=np.float32),
        "20240113_20240125": np.full((2, 2), -0.05, dtype=np.float32),
        "20240101_20240125": np.full((2, 2), 0.05, dtype=np.float32),
    }
    _write_geo_unwrapped(stack, phases)
    result = stack.invert_timeseries()
    expected = invert_unwrapped_pairs(phases, device="cpu", wavelength_m=0.05546576)
    np.testing.assert_allclose(
        result.phase_increments_rad, expected.phase_increments_rad
    )
    product = open_product(stack.config.work_dir / "timeseries")
    assert product.properties["faninsar:phase_source"] == "unwrapped"
