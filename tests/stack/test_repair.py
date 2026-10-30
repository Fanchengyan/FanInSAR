"""Stack identity repair and rescan for cross-version recovery (P0069)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar import Pairs
from faninsar.missions.s1 import S1Stack
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.unwrapping.irls import SpatialIRLS
from faninsar.stack.identity import (
    STACK_IDENTITY_SCHEMA,
    STACK_JSON_NAME,
    read_stack_identity_lenient,
)
from faninsar.stack.repair import RepairReport, rescan
from faninsar.stack.session import Stack
from tests.support.scene_stores import write_test_scene

if TYPE_CHECKING:
    from pathlib import Path

from tests.stack.test_reopen import _orbit_fixtures


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
        pairs=Pairs.from_names(
            [f"{first}_{second}" for first, second in zip(dates, dates[1:])]
        ),
        multilook=(1, 1),
        coreg_mode="geometry",
    )
    data = np.ones((2, 2), dtype=np.complex64)
    for date_id in dates:
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
    from faninsar.stack.search import MissionPreparedInputs

    stack._ensure_precise_orbits = lambda: None  # type: ignore[method-assign]
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


def _drift_schema(work_dir: Path, schema: str = "faninsar.stack.v99") -> None:
    """Corrupt the recorded schema tag to simulate a pre-upgrade identity."""
    path = work_dir / STACK_JSON_NAME
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema"] = schema
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def test_missing_stack_json_still_rejects_on_open(tmp_path: Path) -> None:
    """Tolerant open keeps the missing-file rebuild message (nothing to recover)."""
    work = tmp_path / "empty"
    work.mkdir()
    with pytest.raises(InvalidProcessingStateError, match="rebuild"):
        Stack.open(work)


def test_rescan_reports_recovered_dropped_and_defaulted(tmp_path: Path) -> None:
    """rescan audits a drifted stack.json without writing."""
    stack = _stack(tmp_path)
    stack.prepare()
    work = stack.config.work_dir

    _drift_schema(work)
    # Corrupt the mask_plan payload so decode fails outright.
    document = json.loads((work / STACK_JSON_NAME).read_text())
    document["science"]["mask_plan"] = {"faninsar:mask_plan": 12345}
    (work / STACK_JSON_NAME).write_text(json.dumps(document))

    report = rescan(work)

    assert isinstance(report, RepairReport)
    assert report.schema_found == "faninsar.stack.v99"
    assert report.science_recovered["coreg_mode"] == "geometry"
    # mask_plan's typed payload is not a mapping -> dropped.
    assert "mask_plan" in report.science_dropped
    # A dropped field falls back to its StackConfig default -> defaulted.
    assert "mask_plan" in report.science_defaulted
    # Decodable fields are recovered, not defaulted.
    assert "roi" in report.science_recovered
    assert "grid" in report.science_recovered
    # The committed scene products are discovered.
    assert "scene" in report.products_found
    # rescan never rewrites stack.json.
    still = json.loads((work / STACK_JSON_NAME).read_text())
    assert still["schema"] == "faninsar.stack.v99"


def test_open_is_tolerant_and_attaches_repair_report(tmp_path: Path) -> None:
    """A drifted stack.json still opens a degraded-but-usable Stack."""
    stack = _stack(tmp_path)
    stack.prepare()
    stack.coregister_scenes()
    work = stack.config.work_dir
    _drift_schema(work)

    reopened = Stack.open(work)

    assert reopened.repair_report is not None
    assert reopened.repair_report.schema_found == "faninsar.stack.v99"
    # Committed scenes are still reachable through the degraded Stack.
    assert "20240101" in reopened.scenes
    # stack.json is not rewritten on open.
    still = json.loads((work / STACK_JSON_NAME).read_text())
    assert still["schema"] == "faninsar.stack.v99"
    # status() surfaces the identity section.
    identity = reopened.status()["identity"]
    assert identity["schema"] == "faninsar.stack.v99"
    assert identity["repair"]["schema_found"] == "faninsar.stack.v99"


def test_repair_rewrites_identity_and_recovers_stack(tmp_path: Path) -> None:
    """repair rewrites stack.json at v1 and returns a working Stack."""
    stack = _stack(tmp_path)
    stack.prepare()
    stack.coregister_scenes()
    stack.form_interferograms()
    work = stack.config.work_dir
    _drift_schema(work)

    repaired = Stack.repair(work)

    # stack.json is rewritten at the current schema.
    rewritten = json.loads((work / STACK_JSON_NAME).read_text())
    assert rewritten["schema"] == STACK_IDENTITY_SCHEMA
    assert repaired.repair_report is not None
    assert repaired.repair_report.schema_found == "faninsar.stack.v99"
    # The repaired Stack discovers the committed scene and IFG products.
    assert "20240101" in repaired.scenes
    status = repaired.status()
    assert status["interferograms"]["committed"] == ["20240101_20240113"]
    # A subsequent strict open now succeeds (the file is valid v1 again).
    assert Stack.open(work).reference == "20240101"


def test_repair_infers_pairs_and_reference_from_products(tmp_path: Path) -> None:
    """Committed ifg/ml_* products reconstruct the planned network."""
    stack = _stack(tmp_path)
    stack.prepare()
    stack.coregister_scenes()
    stack.form_interferograms()
    stack.unwrap(SpatialIRLS())
    work = stack.config.work_dir

    # Blank out pairs + reference; repair must infer them from products.
    document = json.loads((work / STACK_JSON_NAME).read_text())
    document["pairs"] = []
    document["misreg_pairs"] = []
    document["reference"] = ""
    (work / STACK_JSON_NAME).write_text(json.dumps(document))

    report = rescan(work)
    assert report.inferred_reference == "20240101"
    assert "20240101_20240113" in report.inferred_pairs

    repaired = Stack.repair(work)
    assert repaired.reference == "20240101"
    assert len(repaired.pairs.values) == 1  # inferred network is non-empty


def test_repair_reports_inputs_changed_without_blocking(tmp_path: Path) -> None:
    """A size-changed input is reported in the RepairReport but never blocks."""
    stack = _stack(tmp_path)
    stack.prepare()
    work = stack.config.work_dir

    # Corrupt a recorded SAFE size so _check_input fails.
    document = json.loads((work / STACK_JSON_NAME).read_text())
    entry = document["inputs"]["sources"]["20240101"][0]
    entry["size"] = 999_999_999
    (work / STACK_JSON_NAME).write_text(json.dumps(document))

    report = rescan(work)
    assert any("SLC" in changed for changed in report.inputs_changed)

    repaired = Stack.repair(work)
    assert any("SLC" in c for c in repaired.repair_report.inputs_changed)
    # Repair still returns a usable Stack; the input record is re-stated
    # honestly (the rewritten stack.json records the real size).
    rewritten = json.loads((work / STACK_JSON_NAME).read_text())
    recorded = rewritten["inputs"]["sources"]["20240101"][0]["size"]
    assert recorded != 999_999_999


def test_lenient_reader_returns_none_for_missing_and_bad_json(tmp_path: Path) -> None:
    """read_stack_identity_lenient signals 'nothing to recover' as None."""
    assert read_stack_identity_lenient(tmp_path) is None
    bad = tmp_path / STACK_JSON_NAME
    bad.write_text("{ not json", encoding="utf-8")
    assert read_stack_identity_lenient(tmp_path) is None
