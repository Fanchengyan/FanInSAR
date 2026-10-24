"""Stack reopen, ``stack.json`` identity, and status/verify tests (P0059)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar import Pairs
from faninsar.missions.s1 import S1Stack
from faninsar.processing.errors import (
    InvalidProcessingStateError,
)
from faninsar.processing.unwrapping.irls import SpatialIRLS
from faninsar.stack.identity import STACK_JSON_NAME, save_stack_identity
from faninsar.stack.session import Stack
from tests.support.scene_stores import write_test_scene

if TYPE_CHECKING:
    from pathlib import Path


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
    stack = S1Stack.from_safes(
        paths,
        work_dir=tmp_path / name / "work",
        activation_mode="reference",
        pairs=Pairs.from_names(
            [f"{first}_{second}" for first, second in zip(dates, dates[1:])]
        ),
        multilook=(1, 1),
        coreg_mode="geometry",
    )
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

    stack._ensure_precise_orbits = lambda: None  # type: ignore[method-assign]
    # Every scene product is pre-committed, so Reference preparation would
    # only open the fixture SAFEs to resolve a selection nothing consumes.
    stack._prepare_reference_scene = (  # type: ignore[method-assign]
        lambda *, overwrite: None
    )
    stack._prepare_mission = (  # type: ignore[method-assign]
        lambda: MissionPreparedInputs(
            orbit_paths_by_acquisition={
                date_id: path for date_id, path in _orbit_fixtures(stack).items()
            }
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
    stack.prepare()
    document = json.loads(
        (stack.config.work_dir / STACK_JSON_NAME).read_text(encoding="utf-8")
    )
    assert document["schema"] == "faninsar.stack.v1"
    assert document["mission"] == "S1Stack"
    assert document["reference"] == "20240101"
    assert document["pairs"] == ["20240101_20240113"]
    assert "work_dir" not in document["science"]

    reopened = Stack.open(stack.config.work_dir, device="cpu")

    assert type(reopened) is S1Stack
    assert reopened.reference == "20240101"
    assert reopened.config.multilook == (1, 1)
    assert reopened.config.coreg_mode == "geometry"
    assert reopened.config.device == "cpu"
    assert set(reopened.scenes) == {"20240101", "20240113"}
    assert reopened.misreg is None
    assert reopened.status()["scenes"]["missing"] == []


def test_reopen_after_stages_reuses_committed_work(tmp_path: Path) -> None:
    """A reopened Stack resumes: committed scenes, IFGs, and unwraps are kept."""
    stack = _stack(tmp_path)
    stack.prepare()
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
        reopened.unwrapped.values,
        stack.unwrapped.values,
    )


def test_reopen_rejects_unknown_runtime_option(tmp_path: Path) -> None:
    """Science options cannot be changed through ``Stack.open``."""
    stack = _stack(tmp_path)
    stack.prepare()
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
    stack.prepare()
    work = stack.config.work_dir
    safe = next(iter((tmp_path / "out").glob("*.SAFE")))
    shutil.rmtree(safe)
    with pytest.raises(InvalidProcessingStateError, match="missing"):
        Stack.open(work)
    safe.write_bytes(b"x")
    with pytest.raises(InvalidProcessingStateError, match="changed on disk"):
        Stack.open(work)


def test_status_reports_missing_and_stale_products(tmp_path: Path) -> None:
    """status() compares the planned network with committed products."""
    stack = _stack(tmp_path)
    stack.prepare()
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
    """verify() re-hashes payloads and names the damaged asset."""
    import zarr

    stack = _stack(tmp_path)
    stack.prepare()
    assert stack.verify() == []
    stack.coregister_scenes()
    assert stack.verify() == []

    scene = stack.config.work_dir / "coreg" / "20240113"
    array = zarr.open_array(store=str(scene / "f0_IW1_b0.slc.zarr"), mode="r+")
    array[0, 0] = 9.0 + 9.0j
    problems = stack.verify()
    assert problems == [f"{scene}: f0_IW1_b0"]


def test_form_interferograms_pairs_outside_the_plan_extend_it(
    tmp_path: Path,
) -> None:
    """``pairs=`` with a Pair outside the plan extends it and rewrites stack.json."""
    from faninsar.processing.unwrapping.irls import SpatialIRLS

    stack = _stack(tmp_path, dates=("20240101", "20240113", "20240125"), name="extend")
    stack.prepare()
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
    stack.prepare()
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
    stack.prepare()
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
    stack.prepare()
    work = stack.config.work_dir

    reopened = Stack.open(work)
    assert tuple(reopened.config.roi)[:4] == tuple(stack.config.roi)[:4]
    assert str(reopened.config.roi.crs) == "EPSG:32633"
    assert reopened.config.geo_grid == stack.config.geo_grid
    assert type(reopened.config.dem).__name__ == "ConstantDEM"
    assert float(reopened.config.dem.height) == 12.0
    assert reopened.config.extra["polarizations"] == ("VV", "VH")
    save_stack_identity(reopened)


def test_unclassified_config_field_fails_at_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A StackConfig field added without classification fails at save."""
    from dataclasses import dataclass, field as dc_field
    from typing import Any as AnyType

    from faninsar.stack.config import StackConfig

    stack = _stack(tmp_path)

    @dataclass
    class Extended(StackConfig):
        """Config carrying an unclassified field."""

        future_option: AnyType | None = dc_field(default=None)

    config = Extended(
        **{
            f.name: getattr(stack.config, f.name)
            for f in __import__("dataclasses").fields(StackConfig)
        }
    )
    config.work_dir = stack.config.work_dir
    monkeypatch.setattr(type(stack), "config", config, raising=False)
    object.__setattr__(stack, "config", config)
    with pytest.raises(InvalidProcessingStateError, match="not classified"):
        save_stack_identity(stack)


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
    stack.prepare()
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
