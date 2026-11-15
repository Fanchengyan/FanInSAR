"""Stack reopen, ``stack.json`` identity, and status/verify tests (P0059)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tests.support.local_stack import LocalStack

from faninsar import Pairs
from faninsar.core.sar_tools import Looks
from faninsar.missions.s1 import S1Stack
from faninsar.processing.errors import (
    InvalidProcessingStateError,
)
from faninsar.processing.unwrapping.irls import SpatialIRLS
from faninsar.stack.identity import (
    STACK_JSON_NAME,
    encode_value,
    save_stack_identity,
)
from faninsar.stack.session import Stack
from tests.support.scene_stores import write_test_scene


def _stack(
    tmp_path: Path,
    *,
    name: str = "out",
    dates: tuple[str, ...] = ("20240101", "20240113"),
) -> S1Stack:
    """Build a prepared Stack with synthetic scene products."""
    paths = []
    for date_id in dates:
        path = tmp_path / name / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        path.mkdir(parents=True)
        paths.append(path)
    stack = LocalStack.from_paths(
        paths,
        work_dir=tmp_path / name / "work",
        pairs=Pairs.from_names(
            [f"{first}_{second}" for first, second in zip(dates, dates[1:])]
        ),
        multilook=(1, 1),
        coreg_mode="geometry",
    )
    stack._request_config.extra["orbit_paths"] = _orbit_fixtures(stack)
    stack.config.extra["orbit_paths"] = dict(stack._request_config.extra["orbit_paths"])
    data = np.ones((2, 2), dtype=np.complex64)
    for date_id in dates:
        # The committed identity matches a geometry-mode resume so
        # coregister_scenes reuses every scene without touching the SAFEs.
        properties: dict[str, object] = {}
        if date_id != dates[0]:
            properties["faninsar:coreg_identity"] = stack._coreg_resume_identity(
                date_id, misreg_az_px=0.0, misreg_rg_px=0.0
            )
        write_test_scene(
            stack.config.work_dir,
            date_id=date_id,
            reference_id=dates[0],
            payload=data,
            reference_payload=data,
            tag="f0_IW1_b0",
            row_origin=0,
            col_origin=0,
            grid_shape=(2, 2),
            properties=properties,
        )
    # The fixture SAFEs are empty directories; mission preparation is
    # replaced so prepare() exercises only the shared identity flow.
    from faninsar.stack.search import MissionPreparedInputs
    from faninsar.stack.preparation import PreparedInputs

    stack._ensure_precise_orbits = lambda: None  # type: ignore[method-assign]
    # Track resolution would write ``track_metadata`` into ``extra`` during
    # prepare(), changing the resume digest of the pre-committed scenes;
    # these tests exercise the identity flow, not the track map.
    stack._track_metadata = lambda: None  # type: ignore[method-assign]
    # Every scene product is pre-committed, so Reference preparation would
    # only open the fixture SAFEs to resolve a selection nothing consumes.
    stack._prepare_reference_scene = (  # type: ignore[method-assign]
        lambda *, overwrite, polygon=None: None
    )
    stack._prepare_inputs = (  # type: ignore[method-assign]
        lambda context: PreparedInputs(
            sources=stack.catalog.paths,
            mission=MissionPreparedInputs(
                orbit_paths_by_acquisition={
                    date_id: path for date_id, path in _orbit_fixtures(stack).items()
                }
            ),
        )
    )
    return stack


def _orbit_fixtures(stack: S1Stack) -> dict[str, Path]:
    """Write one empty orbit file per acquisition, outside the work directory."""
    orbits: dict[str, Path] = {}
    inputs = stack.config.work_dir.parent / "inputs"
    for date_id in stack.catalog.dates:
        path = inputs / "orbits" / f"{date_id}.EOF"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        orbits[date_id] = path
    return orbits


def test_prepare_saves_identity_and_open_restores_the_stack(
    tmp_path: Path,
) -> None:
    """A fresh-process reopen restores identity and discovers scene products."""
    stack = _stack(tmp_path)
    stack.prepare_data()
    document = json.loads(
        (stack.config.work_dir / STACK_JSON_NAME).read_text(encoding="utf-8")
    )
    assert document["schema"] == "faninsar.stack.v1"
    assert document["mission"] == "LocalStack"
    assert document["reference"] == "20240101"
    assert document["pairs"] == ["20240101_20240113"]
    assert "work_dir" not in document["science"]

    reopened = Stack.open(stack.config.work_dir, device="cpu")

    assert type(reopened) is LocalStack
    assert reopened.reference == "20240101"
    assert reopened.config.multilook == Looks(1, 1)
    assert reopened.config.coreg_mode == "geometry"
    assert reopened.config.device == "cpu"
    assert set(reopened.scenes) == {"20240101", "20240113"}
    assert reopened.misreg is None
    assert reopened.status()["scenes"]["missing"] == []


def test_reopen_after_stages_reuses_committed_work(tmp_path: Path) -> None:
    """A reopened Stack resumes: committed scenes, IFGs, and unwraps are kept."""
    stack = _stack(tmp_path)
    stack.prepare_data()
    stack.coregister_scenes()
    stack.form_interferograms()
    stack.unwrap(SpatialIRLS())
    write_ids = {product.canonical for product in stack.network_product_index.products}

    reopened = Stack.open(stack.config.work_dir)

    assert reopened.analysis_ready
    assert {
        product.canonical for product in reopened.network_product_index.products
    } == write_ids
    # Nothing recomputes: forming again and unwrapping again reuse products.
    reopened.form_interferograms()
    reopened.unwrap(SpatialIRLS())
    np.testing.assert_allclose(
        reopened.unwrapped_phase.values,
        stack.unwrapped_phase.values,
    )


def test_reopen_reuses_layout_bound_ifg_without_false_source_mismatch(
    tmp_path: Path,
) -> None:
    """A layout input is structural lineage, not a scene-source mismatch."""
    stack = _stack(tmp_path)
    stack.prepare_data()
    stack.coregister_scenes()
    for date_id in ("20240101", "20240113"):
        path = stack.config.work_dir / "coreg" / date_id / "item.json"
        item = json.loads(path.read_text(encoding="utf-8"))
        item["properties"]["faninsar:layout_id"] = "layout-test"
        item["properties"].setdefault("faninsar:inputs", {})["layout"] = "layout-test"
        path.write_text(json.dumps(item), encoding="utf-8")

    stack.form_interferograms()
    ifg_item = json.loads(
        (
            stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113" / "item.json"
        ).read_text(encoding="utf-8")
    )
    assert ifg_item["properties"]["faninsar:inputs"]["layout"] == "layout-test"
    first = {
        store.pair: store.write_id
        for store in stack._pair_artifact_stores(looks=(1, 1))
    }
    reopened = Stack.open(stack.config.work_dir)
    reopened.form_interferograms()
    assert {
        store.pair: store.write_id
        for store in reopened._pair_artifact_stores(looks=(1, 1))
    } == first


def test_reopen_rejects_changed_layout_binding(tmp_path: Path) -> None:
    """Layout is still compared strictly through the separate layout_id field."""
    stack = _stack(tmp_path)
    stack.prepare_data()
    stack.coregister_scenes()
    for date_id in ("20240101", "20240113"):
        path = stack.config.work_dir / "coreg" / date_id / "item.json"
        item = json.loads(path.read_text(encoding="utf-8"))
        item["properties"]["faninsar:layout_id"] = "layout-test"
        item["properties"].setdefault("faninsar:inputs", {})["layout"] = "layout-test"
        path.write_text(json.dumps(item), encoding="utf-8")
    stack.form_interferograms()
    path = stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113" / "item.json"
    item = json.loads(path.read_text(encoding="utf-8"))
    item["properties"]["faninsar:layout_id"] = "other-layout"
    item["properties"]["faninsar:inputs"]["layout"] = "other-layout"
    path.write_text(json.dumps(item), encoding="utf-8")

    reopened = Stack.open(stack.config.work_dir)
    with pytest.raises(InvalidProcessingStateError, match="layout_id"):
        reopened.form_interferograms()


def test_reopen_rejects_unknown_runtime_option(tmp_path: Path) -> None:
    """Science options cannot be changed through ``Stack.open``."""
    stack = _stack(tmp_path)
    stack.prepare_data()
    with pytest.raises(TypeError, match="runtime values only"):
        Stack.open(stack.config.work_dir, multilook=(2, 2))


def test_reopen_missing_identity_asks_for_rebuild(tmp_path: Path) -> None:
    """A work directory without stack.json fails with one rebuild message."""
    work = tmp_path / "empty"
    work.mkdir()
    with pytest.raises(InvalidProcessingStateError, match="rebuild"):
        Stack.open(work)
    legacy = tmp_path / "legacy"
    (legacy / "ifg" / "ml_1x1" / "20240101_20240113").mkdir(parents=True)
    with pytest.raises(InvalidProcessingStateError, match="rebuild"):
        Stack.open(legacy)


def test_reopen_rejects_changed_or_missing_input(tmp_path: Path) -> None:
    """A missing or truncated SAFE input fails at open by stat alone."""
    import shutil

    stack = _stack(tmp_path)
    stack.prepare_data()
    work = stack.config.work_dir
    safe = next(iter((tmp_path / "out").glob("*.SAFE")))
    shutil.rmtree(safe)
    reopened = Stack.open(work)
    assert not reopened.data_ready
    assert set(reopened.scenes) == {"20240101", "20240113"}
    safe.write_bytes(b"x")
    changed = Stack.open(work)
    assert not changed.data_ready
    assert set(changed.scenes) == {"20240101", "20240113"}
    with pytest.raises(InvalidProcessingStateError, match="call prepare_data"):
        changed.coregister_scenes()


def test_status_reports_missing_and_stale_products(tmp_path: Path) -> None:
    """status() compares the planned network with committed products."""
    stack = _stack(tmp_path)
    stack.prepare_data()
    status = stack.status()
    assert status["scenes"]["missing"] == []
    assert status["interferograms"]["missing"] == ["20240101_20240113"]
    assert status["analysis_ready"] is False

    stack.coregister_scenes()
    stack.form_interferograms()
    status = stack.status()
    assert status["interferograms"]["committed"] == ["20240101_20240113"]
    assert status["unwrap"]["missing"] == ["20240101_20240113"]
    stack.unwrap(SpatialIRLS())
    assert stack.status()["analysis_ready"] is True

    # Replacing the secondary scene makes the IFG and unwrap stale.
    data = np.full((2, 2), 2.0, dtype=np.complex64)
    write_test_scene(
        stack.config.work_dir,
        date_id="20240113",
        reference_id="20240101",
        payload=data,
        reference_payload=np.ones((2, 2), dtype=np.complex64),
        grid_shape=(2, 2),
    )
    status = stack.status()
    # The recomputed scene itself is current; its downstream products are
    # bound to the replaced scene write id and are the stale ones.
    assert status["scenes"]["stale"] == []
    assert status["interferograms"]["stale"] == ["20240101_20240113"]
    assert status["unwrap"]["stale"] == ["20240101_20240113"]


def test_verify_detects_a_damaged_payload(tmp_path: Path) -> None:
    """verify() names missing assets and ignores same-size payload edits."""
    import shutil

    import zarr

    stack = _stack(tmp_path)
    stack.prepare_data()
    assert stack.verify() == []
    stack.coregister_scenes()
    assert stack.verify() == []

    scene = stack.config.work_dir / "coreg" / "20240113"
    array = zarr.open_array(store=str(scene / "f0_IW1_b0.slc.zarr"), mode="r+")
    array[0, 0] = 9.0 + 9.0j
    assert stack.verify() == []
    shutil.rmtree(scene / "f0_IW1_b0.slc.zarr")
    problems = stack.verify()
    assert problems == [f"{scene}: f0_IW1_b0"]


def test_form_interferograms_pairs_outside_the_plan_extend_it(
    tmp_path: Path,
) -> None:
    """``pairs=`` with a Pair outside the plan extends it and rewrites stack.json."""
    from faninsar.processing.unwrapping.irls import SpatialIRLS

    stack = _stack(tmp_path, dates=("20240101", "20240113", "20240125"), name="extend")
    stack.prepare_data()
    document = json.loads(
        (stack.config.work_dir / STACK_JSON_NAME).read_text(encoding="utf-8")
    )
    assert document["pairs"] == ["20240101_20240113", "20240113_20240125"]

    # The requested Pair is outside the plan: the plan extends and the IFG
    # product for the new Pair alone is committed.
    stack.form_interferograms(pairs=Pairs.from_names(["20240101_20240125"]))

    document = json.loads(
        (stack.config.work_dir / STACK_JSON_NAME).read_text(encoding="utf-8")
    )
    assert sorted(document["pairs"]) == [
        "20240101_20240113",
        "20240101_20240125",
        "20240113_20240125",
    ]
    committed = sorted(
        path.parent.name
        for path in (stack.config.work_dir / "ifg" / "ml_1x1").glob("*/item.json")
    )
    assert committed == ["20240101_20240125"]
    reopened = Stack.open(stack.config.work_dir)
    assert "20240113_20240125" in reopened.status()["interferograms"]["planned"]


def test_form_interferograms_subset_forms_only_the_requested_pairs(
    tmp_path: Path,
) -> None:
    """A subset of the plan forms only those Pairs and never shrinks the plan."""
    from faninsar.processing.unwrapping.irls import SpatialIRLS

    stack = _stack(tmp_path, dates=("20240101", "20240113", "20240125"), name="subset")
    stack.prepare_data()
    stack.form_interferograms(pairs=Pairs.from_names(["20240101_20240113"]))

    committed = sorted(
        path.parent.name
        for path in (stack.config.work_dir / "ifg" / "ml_1x1").glob("*/item.json")
    )
    assert committed == ["20240101_20240113"]
    document = json.loads(
        (stack.config.work_dir / STACK_JSON_NAME).read_text(encoding="utf-8")
    )
    assert document["pairs"] == ["20240101_20240113", "20240113_20240125"]


def test_finished_work_directory_holds_only_products(tmp_path: Path) -> None:
    """A finished work directory holds only stack.json, products, and cache/."""
    from faninsar.processing.unwrapping.irls import SpatialIRLS

    stack = _stack(tmp_path, name="walk")
    stack.prepare_data()
    stack.coregister_scenes()
    stack.form_interferograms()
    stack.unwrap(SpatialIRLS())
    stack.invert_timeseries()

    work = stack.config.work_dir

    def under(root: Path, path: Path) -> bool:
        return root == path or root in path.parents

    # Allowed: the identity, cache/, product directories (their payloads,
    # including opaque Zarr subtrees), and ancestors of any of them.
    allowed: list[Path] = [work, work / "cache"]
    for item in work.rglob("item.json"):
        allowed.append(item.parent)
    zarr_roots = [path for path in work.rglob("*") if path.name.endswith(".zarr")]
    allowed.extend(zarr_roots)

    stray: list[str] = []
    for path in sorted(work.rglob("*")):
        if any(under(root, path) for root in allowed):
            continue
        stray.append(str(path.relative_to(work)))
    assert stray == [], stray


def test_identity_save_rejects_unserializable_science(tmp_path: Path) -> None:
    """An unserializable science value fails at save, never at reopen."""
    stack = _stack(tmp_path)
    stack.config.roi = object()
    with pytest.raises(InvalidProcessingStateError, match="cannot represent"):
        save_stack_identity(stack)


def test_science_configuration_round_trips(tmp_path: Path) -> None:
    """Tuples, grids, ROI, DEM recipe, and mask plans round-trip exactly."""
    from faninsar.data.query import BoundingBox
    from faninsar.processing.geometry import DEM
    from faninsar.processing.mosaicking.grid import GeoGridSpec

    stack = _stack(tmp_path)
    stack.config.roi = BoundingBox(
        458000.0, 4162000.0, 472000.0, 4174000.0, crs="EPSG:32633"
    )
    stack.config.geo_grid = GeoGridSpec(
        crs="EPSG:32633",
        transform=(454000.0, 40.0, 0.0, 4178000.0, 0.0, -40.0),
        height=450,
        width=500,
        resolution_m=(40.0, 40.0),
    )
    stack.config.dem = DEM.from_constant(12.0)
    stack.config.extra["polarizations"] = ("VV", "VH")
    from dataclasses import replace

    stack._request_config = replace(stack.config, extra=dict(stack.config.extra))
    stack.prepare_data()
    work = stack.config.work_dir

    reopened = Stack.open(work)
    assert tuple(reopened.config.roi)[:4] == tuple(stack.config.roi)[:4]
    assert str(reopened.config.roi.crs) == "EPSG:32633"
    assert reopened.config.geo_grid == stack.config.geo_grid
    assert type(reopened.config.dem).__name__ == "ConstantDEM"
    assert float(reopened.config.dem.height) == 12.0
    assert reopened.config.extra["polarizations"] == ("VV", "VH")
    save_stack_identity(reopened)


def test_new_config_field_is_saved_as_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A StackConfig field not in UNSAVED_FIELDS is saved as a science value."""
    from dataclasses import dataclass
    from dataclasses import field as dc_field
    from typing import Any as AnyType

    from faninsar.stack.config import StackConfig

    stack = _stack(tmp_path)

    @dataclass
    class Extended(StackConfig):
        """Config carrying an additional field."""

        future_option: AnyType | None = dc_field(default=None)

    config = Extended(
        **{
            f.name: getattr(stack.config, f.name)
            for f in __import__("dataclasses").fields(StackConfig)
        }
    )
    config.future_option = 3
    config.work_dir = stack.config.work_dir
    monkeypatch.setattr(type(stack), "config", config, raising=False)
    object.__setattr__(stack, "config", config)
    path = save_stack_identity(stack)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["science"]["future_option"] == 3


def test_persisted_inputs_carry_download_verification_status(tmp_path: Path) -> None:
    """Each persisted input record carries its download-time verification.

    PROPOSAL-0060 goal: ``verification.status`` is a *consumed* provenance
    fact. The shared input manifest (``stack.json`` ``inputs`` block) records
    each input's download-time verification so a reopened Stack reports which
    inputs were verified, failed, or unverifiable. An input with no record is
    surfaced as ``unverifiable``.
    """
    stack = _stack(tmp_path)
    # Two orbit files registered as downloaded inputs.
    orbits: dict[str, Path] = {}
    inputs_dir = stack.config.work_dir.parent / "inputs"
    for date_id in stack.catalog.dates:
        path = inputs_dir / f"{date_id}.EOF"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"orbit")
        orbits[date_id] = path
    stack.config.extra["orbit_paths"] = dict(orbits)
    stack._prepare_inputs = lambda context: __import__(
        "faninsar.stack.preparation", fromlist=["PreparedInputs"]
    ).PreparedInputs(
        sources=stack.catalog.paths,
        mission=__import__(
            "faninsar.stack.search", fromlist=["MissionPreparedInputs"]
        ).MissionPreparedInputs(orbit_paths_by_acquisition=orbits),
    )
    # Mark the first orbit as verified at download time (the P0055 record).
    verified_orbit = orbits[stack.catalog.dates[0]]
    manifest = verified_orbit.with_name(
        f"{verified_orbit.name}.faninsar.remote.abc.json"
    )
    manifest.write_text(
        json.dumps(
            {
                "identity": "abc",
                "verification": {
                    "algorithm": "md5",
                    "checksum": "d41d8cd98f00b204e9800998ecf8427e",
                    "status": "verified",
                    "size": 5,
                },
            }
        )
    )
    stack.prepare_data()
    document = json.loads(
        (stack.config.work_dir / STACK_JSON_NAME).read_text(encoding="utf-8")
    )
    orbit_records = document["inputs"]["orbits"]
    verified_record = orbit_records[stack.catalog.dates[0]]
    assert verified_record["verification"]["status"] == "verified"
    assert verified_record["verification"]["algorithm"] == "md5"
    # The other orbit has no download record: surfaced as unverifiable.
    other = orbit_records[stack.catalog.dates[1]]
    assert other["verification"]["status"] == "unverifiable"


def test_aux_cal_bindings_round_trip_through_stack_json(tmp_path: Path) -> None:
    """Per-date AUX_CAL bindings persist under ``inputs.aux_cal`` and reopen."""
    stack = _stack(tmp_path)
    aux = (
        stack.config.work_dir
        / "aux_cal"
        / "S1A_AUX_CAL_V20140908T000000_G20240327T101157.SAFE"
        / "data"
        / "s1a-aux-cal.xml"
    )
    aux.parent.mkdir(parents=True, exist_ok=True)
    aux.write_bytes(b"<auxiliaryCalibration/>")
    bound = dict.fromkeys(stack.catalog.dates, aux)
    stack.config.extra["aux_cal_paths"] = dict(bound)
    stack._request_config.extra["aux_cal_paths"] = dict(bound)
    stack.prepare_data()
    document = json.loads(
        (stack.config.work_dir / STACK_JSON_NAME).read_text(encoding="utf-8")
    )
    assert set(document["inputs"]["aux_cal"]) == set(bound)
    assert "calibrations" not in document["inputs"]

    reopened = Stack.open(stack.config.work_dir, device="cpu")

    restored = reopened.config.extra["aux_cal_paths"]
    assert {date: str(path) for date, path in restored.items()} == {
        date: str(path) for date, path in bound.items()
    }
    assert reopened._aux_cal_path_for(stack.catalog.dates[1]) == aux


def _write_untagged_raster(path: Path) -> Path:
    """Write a tiny GeoTIFF that has no FanInSAR vertical-datum tag."""
    import rasterio
    from affine import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=Affine(0.01, 0.0, 10.0, 0.0, -0.01, 50.0),
    ) as dataset:
        dataset.write(np.full((8, 8), 12.0, dtype=np.float32), 1)
    return path


def _dem_recipe(document: dict[str, object]) -> dict[str, object]:
    """Return the ``faninsar:dem`` payload from one identity or request document."""
    science = document["science"]
    assert isinstance(science, dict)
    encoded = science["dem"]
    assert isinstance(encoded, dict)
    payload = encoded["faninsar:dem"]
    assert isinstance(payload, dict)
    return payload


def _prepare_dem_stack(tmp_path: Path, dem: object, *, name: str) -> LocalStack:
    """Prepare a local Stack whose request and live config share one DEM."""
    from dataclasses import replace

    stack = _stack(tmp_path, name=name)
    stack.config.dem = dem
    stack._request_config = replace(stack.config, extra=dict(stack.config.extra))
    stack.prepare_data()
    return stack


@pytest.mark.parametrize("vertical_datum", ["ellipsoidal", "egm96"])
def test_explicit_raster_vertical_datum_survives_prepare_and_open(
    tmp_path: Path,
    vertical_datum: str,
) -> None:
    """A supplied raster is ingested once into canonical work_dir/dem.tif."""
    from faninsar.processing.geometry import DEM

    path = _write_untagged_raster(tmp_path / f"dem-{vertical_datum}.tif")
    dem = DEM.from_raster(path, vertical_datum=vertical_datum)
    stack = _prepare_dem_stack(tmp_path, dem, name=f"raster-{vertical_datum}")
    work = stack.config.work_dir
    managed = work / "dem.tif"
    identity = json.loads((work / STACK_JSON_NAME).read_text(encoding="utf-8"))
    request = json.loads((work / "inputs" / "request.json").read_text(encoding="utf-8"))
    assert managed.is_file()
    assert path.is_file()
    assert stack.dem.vertical_datum == "ellipsoidal"
    assert Path(stack.dem.path).resolve() == managed.resolve()
    assert _dem_recipe(identity)["vertical_datum"] == "ellipsoidal"
    assert _dem_recipe(request)["vertical_datum"] == vertical_datum

    reopened = Stack.open(work)

    assert type(reopened.config.dem).__name__ == "RasterDEM"
    assert reopened.config.dem.vertical_datum == "ellipsoidal"
    assert Path(reopened.config.dem.path).resolve() == managed.resolve()
    assert reopened._request_config.dem.vertical_datum == vertical_datum
    assert Path(reopened._request_config.dem.path).resolve() == path.resolve()


def test_constant_dem_height_and_vertical_datum_survive_prepare_and_open(
    tmp_path: Path,
) -> None:
    """Constant height recipes persist the declared datum with the height."""
    from faninsar.processing.geometry import DEM

    dem = DEM.from_constant(12.0, vertical_datum="egm96")
    stack = _prepare_dem_stack(tmp_path, dem, name="constant-egm96")
    work = stack.config.work_dir
    identity = json.loads((work / STACK_JSON_NAME).read_text(encoding="utf-8"))
    assert _dem_recipe(identity)["height"] == 12.0
    assert _dem_recipe(identity)["vertical_datum"] == "egm96"

    reopened = Stack.open(work)

    assert type(reopened.config.dem).__name__ == "ConstantDEM"
    assert float(reopened.config.dem.height) == 12.0
    assert reopened.config.dem.vertical_datum == "egm96"
    assert reopened._request_config.dem.vertical_datum == "egm96"


def test_persisted_raster_dem_without_vertical_datum_still_rejects(
    tmp_path: Path,
) -> None:
    """Legacy raster recipes with no datum stay fail-closed; nothing is guessed."""
    from faninsar.processing.geometry import DEM

    path = _write_untagged_raster(tmp_path / "dem-legacy.tif")
    stack = _prepare_dem_stack(
        tmp_path,
        DEM.from_raster(path, vertical_datum="ellipsoidal"),
        name="raster-legacy",
    )
    work = stack.config.work_dir
    for relative in (STACK_JSON_NAME, Path("inputs") / "request.json"):
        target = work / relative
        document = json.loads(target.read_text(encoding="utf-8"))
        _dem_recipe(document).pop("vertical_datum", None)
        target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous"):
        Stack.open(work)


def test_raster_vertical_datum_change_changes_canonical_identity(
    tmp_path: Path,
) -> None:
    """The same raster path with a different datum is a different source identity."""
    from faninsar.processing.geometry import DEM

    path = _write_untagged_raster(tmp_path / "dem-identity.tif")
    ellipsoidal = DEM.from_raster(path, vertical_datum="ellipsoidal")
    orthometric = DEM.from_raster(path, vertical_datum="egm96")
    first = encode_value(ellipsoidal)
    second = encode_value(orthometric)
    assert first != second
    assert first["faninsar:dem"]["vertical_datum"] == "ellipsoidal"
    assert second["faninsar:dem"]["vertical_datum"] == "egm96"
    assert ellipsoidal.identity != orthometric.identity


def test_verify_reports_a_foreign_item_and_keeps_scanning(
    tmp_path: Path,
) -> None:
    """An item.json that is not a FanInSAR product is a problem, not a crash."""
    import json
    import shutil

    stack = _stack(tmp_path)
    stack.prepare_data()
    stack.coregister_scenes()
    assert stack.verify() == []

    # An export directory, a vendor layout, or a hand-edited file all look like
    # this: a valid STAC item that is not a FanInSAR product envelope.
    foreign = stack.config.work_dir / "exports" / "ts"
    foreign.mkdir(parents=True)
    (foreign / "item.json").write_text(
        json.dumps(
            {
                "stac_version": "1.0.0",
                "type": "Feature",
                "id": "ts",
                "properties": {"datetime": None},
                "assets": {},
            }
        ),
        encoding="utf-8",
    )

    problems = stack.verify()
    assert len(problems) == 1
    assert problems[0].startswith(f"{foreign}: invalid-item")

    # A real damaged product is still reported in the same scan.
    scene = stack.config.work_dir / "coreg" / "20240113"
    shutil.rmtree(scene / "f0_IW1_b0.slc.zarr")
    problems = stack.verify()
    assert f"{scene}: f0_IW1_b0" in problems
    assert any(line.startswith(f"{foreign}: invalid-item") for line in problems)
