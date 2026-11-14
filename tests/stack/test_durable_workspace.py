"""Durable factory, open, migration and extension contracts (PROPOSAL-0088)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.support.local_stack import LocalStack

from faninsar.data.query import BoundingBox
from faninsar.missions.s1 import S1Stack
from faninsar.missions.s1.s1search import S1Search
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import ConstantDEM
from faninsar.remote.records import AcquisitionMetadata, CatalogItem
from faninsar.stack.identity import STACK_JSON_NAME
from faninsar.stack.session import Stack
from faninsar.stack.workspace import (
    BACKUP_SUFFIX,
    WORKSPACE_SCHEMA_VERSION,
    REQUEST_RECORD,
    WorkspaceConflictError,
    WorkspaceIntegrityError,
)


def _safe(tmp_path: Path, date_id: str = "20240101") -> Path:
    """Create one SAFE-like directory without opening a product."""
    path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
    path.mkdir(exist_ok=True)
    return path


def _configured(
    tmp_path: Path,
    *,
    name: str = "work",
    roi: object | None = None,
) -> S1Stack:
    """Build one durable configured S1 workspace without source work."""
    return S1Stack.from_safes(
        [_safe(tmp_path, "20240101")],
        work_dir=tmp_path / name,
        roi=roi,
        dem=ConstantDEM(0),
    )


def _document(work: Path) -> dict[str, Any]:
    """Return the parsed authoritative intent."""
    return json.loads((work / STACK_JSON_NAME).read_text(encoding="utf-8"))


def _search() -> S1Search:
    """Return one minimal provider-neutral search result."""
    item = CatalogItem(
        provider="cdse",
        catalog="sentinel1-cdse",
        collection="sentinel-1-slc",
        item_id="item-20240101",
        acquisition=AcquisitionMetadata(
            acquisition_id="item-20240101",
            start=None,
            end=None,
        ),
        assets={},
        raw_metadata={},
    )
    return S1Search(
        items=(item,),
        catalog="sentinel1-cdse",
        spatial=BoundingBox(0.0, 0.0, 1.0, 1.0, crs="EPSG:4326"),
        query={},
    )


_VV_EXTRA = {"polarizations": {"faninsar:tuple": ["VV"]}}
"""The encoded interpretation of ``S1Stack.from_safes(polarizations=("VV",))``."""


def _vv_stack(tmp_path: Path, *, name: str = "work") -> S1Stack:
    """Build one configured VV-only S1 workspace without source work."""
    return S1Stack.from_safes(
        [_safe(tmp_path)],
        work_dir=tmp_path / name,
        polarizations=("VV",),
        dem=ConstantDEM(0),
    )


def _legacy_request(work: Path, extra: dict[str, Any]) -> dict[str, Any]:
    """Return the legacy preparation request for the current intent."""
    document = _document(work)
    return {
        "schema": "faninsar.preparation.v1",
        "mission": document["mission"],
        "source": document["request"],
        "science": document["requested_science"],
        "extra": extra,
    }


def _write_legacy_request(work: Path, extra: dict[str, Any], *, backup: bool) -> None:
    """Publish one legacy preparation request, optionally as a preserved copy."""
    path = work / REQUEST_RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_legacy_request(work, extra))
    if backup:
        path.with_name(path.name + BACKUP_SUFFIX).write_text(payload, encoding="utf-8")
    else:
        path.write_text(payload, encoding="utf-8")


def test_factory_return_publishes_reopenable_configured_intent_without_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A factory return writes the full intent but performs no source work."""
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append("called")
        pytest.fail("factory or configured open performed source I/O")

    monkeypatch.setattr("faninsar.missions.s1.safe.open_safe_product", forbidden)
    monkeypatch.setattr(S1Stack, "_prepare_inputs", forbidden)
    monkeypatch.setattr(S1Stack, "_ensure_precise_orbits", forbidden)

    stack = _configured(tmp_path)

    assert calls == []
    assert not stack.data_ready
    assert stack.intent_revision == 1
    document = _document(stack.config.work_dir)
    assert document["schema_version"] == WORKSPACE_SCHEMA_VERSION
    assert document["identity_status"] == "configured"
    assert document["intent_revision"] == 1
    assert document["request"]["kind"] == "safes"
    assert document["inputs"]["sources"] == {}
    assert document["pairs"] == []

    opened = Stack.open(stack.config.work_dir)

    assert calls == []
    assert type(opened) is S1Stack
    assert not opened.data_ready
    assert opened.intent_revision == 1
    assert tuple(Path(path) for path in opened.request["paths"]) == (
        Path(document["request"]["paths"][0]),
    )
    assert opened.config.dem is not None


def test_repeated_factory_is_idempotent_and_conflict_never_overwrites(
    tmp_path: Path,
) -> None:
    """Identical construction is idempotent; changed intent is rejected."""
    stack = _configured(tmp_path)
    before = (stack.config.work_dir / STACK_JSON_NAME).read_bytes()

    repeated = S1Stack.from_safes(
        [Path(_document(stack.config.work_dir)["request"]["paths"][0])],
        work_dir=stack.config.work_dir,
        dem=ConstantDEM(0),
    )
    assert repeated.intent_revision == 1
    assert (stack.config.work_dir / STACK_JSON_NAME).read_bytes() == before

    with pytest.raises(WorkspaceConflictError, match="different Stack intent"):
        S1Stack.from_safes(
            [Path(_document(stack.config.work_dir)["request"]["paths"][0])],
            work_dir=stack.config.work_dir,
            roi=BoundingBox(10.0, 10.0, 11.0, 11.0, crs="EPSG:4326"),
            dem=ConstantDEM(0),
        )
    assert (stack.config.work_dir / STACK_JSON_NAME).read_bytes() == before


def test_open_adopts_request_only_legacy_workspace_with_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supported request-only workspace becomes one durable intent."""
    work = tmp_path / "work"
    stack = _configured(tmp_path)
    work = stack.config.work_dir
    request_path = work / REQUEST_RECORD
    document = _document(work)
    request_document = {
        "schema": "faninsar.preparation.v1",
        "mission": "S1Stack",
        "source": document["request"],
        "science": document["requested_science"],
        "extra": {},
    }
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request_document), encoding="utf-8")
    (work / STACK_JSON_NAME).unlink()

    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append("called")
        pytest.fail("legacy adoption performed source I/O")

    monkeypatch.setattr("faninsar.missions.s1.safe.open_safe_product", forbidden)

    opened = Stack.open(work)

    assert calls == []
    assert (work / STACK_JSON_NAME).is_file()
    assert (work / "inputs" / ("request.json" + BACKUP_SUFFIX)).is_file()
    adopted = _document(work)
    assert adopted["schema_version"] == WORKSPACE_SCHEMA_VERSION
    assert adopted["request"]["kind"] == "safes"
    assert adopted["identity_status"] == "configured"
    assert opened.request["kind"] == "safes"
    assert not opened.data_ready


def test_open_rejects_unsupported_workspace_schema(tmp_path: Path) -> None:
    """A version other than the single v1 layout is rejected, not adopted."""
    safe = _safe(tmp_path)
    work = tmp_path / "work"
    stack = LocalStack.from_paths(
        [safe],
        work_dir=work,
        dem=ConstantDEM(0),
        multilook=(1, 1),
    )
    assert stack.prepare_data().complete
    path = work / STACK_JSON_NAME
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = WORKSPACE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(WorkspaceIntegrityError, match="supports only version"):
        Stack.open(work)

    assert not (work / ("stack.json" + BACKUP_SUFFIX)).is_file()


def test_open_adopts_request_only_legacy_workspace_with_request_extras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request-only adoption keeps the caller's mission request extras."""
    stack = _vv_stack(tmp_path)
    work = stack.config.work_dir
    document = _document(work)
    _write_legacy_request(work, _VV_EXTRA, backup=False)
    (work / STACK_JSON_NAME).unlink()

    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append("called")
        pytest.fail("legacy adoption performed source I/O")

    monkeypatch.setattr("faninsar.missions.s1.safe.open_safe_product", forbidden)
    monkeypatch.setattr(S1Stack, "_prepare_inputs", forbidden)

    opened = Stack.open(work)

    assert calls == []
    adopted = _document(work)
    assert adopted["requested_science_extra"] == _VV_EXTRA
    assert adopted["science_extra"] == _VV_EXTRA
    assert adopted["request"] == document["request"]
    assert adopted["targets"] == document["targets"]
    assert opened.config.extra["polarizations"] == ("VV",)
    assert opened._request_config.extra["polarizations"] == ("VV",)

    before = (work / STACK_JSON_NAME).read_bytes()
    reopened = Stack.open(work)
    assert reopened.intent_revision == 1
    assert reopened._request_config.extra["polarizations"] == ("VV",)
    assert (work / STACK_JSON_NAME).read_bytes() == before


def test_open_accepts_absent_schema_version_as_v1(
    tmp_path: Path,
) -> None:
    """An absent schema_version is the single v1 layout."""
    stack = _vv_stack(tmp_path)
    work = stack.config.work_dir
    path = work / STACK_JSON_NAME
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("schema_version", None)
    path.write_text(json.dumps(document), encoding="utf-8")

    opened = Stack.open(work)

    assert opened.config.extra["polarizations"] == ("VV",)
    assert opened._request_config.extra["polarizations"] == ("VV",)

    before = path.read_bytes()
    assert Stack.open(work).intent_revision == 1
    assert path.read_bytes() == before


def test_open_repairs_adopted_intent_extras_from_science_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adopted intent with no requested extras recovers from its science extra."""
    stack = _vv_stack(tmp_path)
    work = stack.config.work_dir
    path = work / STACK_JSON_NAME
    document = _document(work)
    document["requested_science_extra"] = {}
    path.write_text(json.dumps(document), encoding="utf-8")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("metadata repair performed source I/O")

    monkeypatch.setattr("faninsar.missions.s1.safe.open_safe_product", forbidden)
    monkeypatch.setattr(S1Stack, "_prepare_inputs", forbidden)

    opened = Stack.open(work)

    repaired = _document(work)
    assert repaired["requested_science_extra"] == _VV_EXTRA
    assert repaired["science_extra"] == document["science_extra"]
    assert repaired["intent_revision"] == 2
    assert (work / ("stack.json" + BACKUP_SUFFIX)).is_file()
    backup = json.loads((work / ("stack.json" + BACKUP_SUFFIX)).read_text())
    assert backup["intent_revision"] == 1
    unchanged = (
        "request",
        "targets",
        "requested_science",
        "science",
        "inputs",
        "pairs",
    )
    for key in unchanged:
        assert repaired[key] == document[key]
    assert opened.config.extra["polarizations"] == ("VV",)
    assert opened._request_config.extra["polarizations"] == ("VV",)

    before = path.read_bytes()
    assert Stack.open(work).intent_revision == 2
    assert path.read_bytes() == before


def test_open_repairs_adopted_intent_extras_from_preserved_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An emptied adopted intent recovers from its preserved request snapshot."""
    stack = _vv_stack(tmp_path)
    work = stack.config.work_dir
    path = work / STACK_JSON_NAME
    document = _document(work)
    document["requested_science_extra"] = {}
    document["science_extra"] = {}
    path.write_text(json.dumps(document), encoding="utf-8")
    _write_legacy_request(work, {}, backup=False)
    _write_legacy_request(work, _VV_EXTRA, backup=True)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("metadata repair performed source I/O")

    monkeypatch.setattr("faninsar.missions.s1.safe.open_safe_product", forbidden)
    monkeypatch.setattr(S1Stack, "_prepare_inputs", forbidden)

    opened = Stack.open(work)

    repaired = _document(work)
    assert repaired["requested_science_extra"] == _VV_EXTRA
    assert repaired["science_extra"] == _VV_EXTRA
    assert repaired["intent_revision"] == 2
    assert opened.config.extra["polarizations"] == ("VV",)
    assert opened._request_config.extra["polarizations"] == ("VV",)

    before = path.read_bytes()
    assert Stack.open(work).intent_revision == 2
    assert path.read_bytes() == before


def test_open_does_not_guess_ambiguous_request_extras(tmp_path: Path) -> None:
    """Disagreeing preserved records keep the recorded value and add nothing."""
    stack = _vv_stack(tmp_path)
    work = stack.config.work_dir
    path = work / STACK_JSON_NAME
    document = _document(work)
    document["requested_science_extra"] = {}
    path.write_text(json.dumps(document), encoding="utf-8")
    _write_legacy_request(
        work, {"polarizations": {"faninsar:tuple": ["HH"]}}, backup=True
    )

    opened = Stack.open(work)

    retained = _document(work)
    assert retained["requested_science_extra"] == {}
    assert retained["intent_revision"] == 1
    assert opened.config.extra["polarizations"] == ("VV",)
    assert opened._request_config.extra["polarizations"] == ("VV",)
    with pytest.raises(WorkspaceIntegrityError, match="disagree"):
        Stack.open(work, on_error="raise")


def test_extend_roi_union_is_idempotent_and_preserves_invariants(
    tmp_path: Path,
) -> None:
    """ROI extension unions, never replaces, and keeps the Reference pin."""
    first = BoundingBox(0.0, 0.0, 1.0, 1.0, crs="EPSG:4326")
    second = BoundingBox(1.0, 0.0, 2.0, 1.0, crs="EPSG:4326")
    stack = _configured(tmp_path, roi=first)
    reference = stack.reference if stack._inputs_state is not None else None
    grid = stack._request_config.grid

    report = stack.extend(roi=second)

    assert report.changed
    assert report.revision == 2
    assert (stack.config.roi.left, stack.config.roi.right) == (0.0, 2.0)
    assert (stack._request_config.roi.left, stack._request_config.roi.right) == (
        0.0,
        2.0,
    )
    assert stack._request_config.grid == grid
    assert (stack.reference if stack._inputs_state is not None else None) == reference

    repeated = stack.extend(roi=second)
    assert not repeated.changed
    assert repeated.revision == 2

    opened = Stack.open(stack.config.work_dir)
    assert (opened.config.roi.left, opened.config.roi.right) == (0.0, 2.0)
    assert opened.intent_revision == 2


def test_extend_burst_and_swath_union_is_additive(tmp_path: Path) -> None:
    """Explicit burst/swath selections union; an all-selection cannot shrink."""
    explicit = S1Stack.from_safes(
        [_safe(tmp_path, "20240101")],
        work_dir=tmp_path / "explicit",
        bursts={"IW1": [1]},
        swaths=("IW1",),
        dem=ConstantDEM(0),
    )
    report = explicit.extend(
        bursts={"IW1": [2], "IW2": [1]},
        swaths=("IW2",),
    )
    assert report.changed
    assert report.added_bursts and report.added_swaths
    assert explicit._request_config.bursts == {"IW1": [1, 2], "IW2": [1]}
    assert explicit._request_config.swaths == ("IW1", "IW2")

    all_bursts = _configured(tmp_path, name="all-bursts")
    all_bursts.extend(bursts={"IW1": [1]}, swaths=("IW1",))
    assert all_bursts._request_config.bursts is None
    assert all_bursts._request_config.swaths is None


def test_extend_rejects_contradictory_modes_and_unsupported_schema(
    tmp_path: Path,
) -> None:
    """Source modes are exclusive and unsupported intent schemas are rejected."""
    stack = _configured(tmp_path)
    with pytest.raises(ValueError, match="search or paths"):
        stack.extend(search=_search(), paths=[tmp_path / "other"])
    with pytest.raises(ValueError, match="at least one"):
        stack.extend()

    document = _document(stack.config.work_dir)
    document["schema_version"] = WORKSPACE_SCHEMA_VERSION + 1
    (stack.config.work_dir / STACK_JSON_NAME).write_text(
        json.dumps(document), encoding="utf-8"
    )
    with pytest.raises(WorkspaceIntegrityError, match="supports only version"):
        Stack.open(stack.config.work_dir)


def test_coregister_scenes_never_acquires_from_unprepared_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Processing-only execution does not acquire missing prerequisites."""
    stack = _configured(tmp_path)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("coregister_scenes acquired an input")

    monkeypatch.setattr(S1Stack, "_prepare_inputs", forbidden)
    opened = Stack.open(stack.config.work_dir)
    with pytest.raises(InvalidProcessingStateError, match="not prepared"):
        opened.coregister_scenes()


def test_verification_policy_is_runtime_only_and_not_persisted(
    tmp_path: Path,
) -> None:
    """Fast/strict is an execution policy, never stack.json science."""
    stack = _configured(tmp_path)
    before = (stack.config.work_dir / STACK_JSON_NAME).read_bytes()

    strict = Stack.open(stack.config.work_dir, verification="strict", on_error="raise")

    assert strict.verification == "strict"
    assert (stack.config.work_dir / STACK_JSON_NAME).read_bytes() == before
    with pytest.raises(ValueError, match="verification"):
        Stack.open(stack.config.work_dir, verification="paranoid")


def test_coregister_persists_and_reloads_the_latest_report(tmp_path: Path) -> None:
    """Real coregistration writes JSON authority and a reloadable view."""
    from tests.stack.test_reopen import _stack

    from faninsar.stack.reporting import ReportWriter

    stack = _stack(tmp_path, name="report")
    stack.prepare_data()
    stack.coregister_scenes()

    report = stack.coregistration_report
    assert report is not None
    assert report.mode == "geometry"
    assert report.intent_revision == stack.intent_revision
    assert any(record.stage == "coregistration" for record in report.records)
    assert any(
        record.status in {"completed", "reused"} for record in report.records
    )
    latest = ReportWriter.load_latest(stack.config.work_dir)
    assert latest is not None
    assert latest.attempt_id == report.attempt_id
    attempt = (
        stack.config.work_dir
        / "reports"
        / "coregistration"
        / report.attempt_id
    )
    assert (attempt / "report.json").is_file()
    assert (attempt / "report.md").is_file()


def test_prepare_and_coregister_returns_one_combined_report(tmp_path: Path) -> None:
    """The combined entry authorizes preparation and processing in one report."""
    from tests.stack.test_reopen import _stack

    stack = _stack(tmp_path, name="combined")
    report = stack.prepare_and_coregister()

    assert report is not None
    assert report.outcome in {"complete", "partial"}
    assert any(record.stage == "preparation" for record in report.records)
    assert any(record.stage == "coregistration" for record in report.records)
    assert report is stack.coregistration_report or (
        stack.coregistration_report is not None
        and report.attempt_id == stack.coregistration_report.attempt_id
    )


def test_factory_from_search_publishes_request_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_search`` persists the enriched request and downloads nothing."""
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append("called")
        pytest.fail("from_search downloaded or prepared at construction")

    monkeypatch.setattr("faninsar.missions.s1.acquisition.materialize_s1_acquisitions", forbidden)
    monkeypatch.setattr("faninsar.remote.download", forbidden)

    stack = S1Stack.from_search(_search(), work_dir=tmp_path / "work")

    assert calls == []
    document = _document(stack.config.work_dir)
    assert document["request"]["kind"] == "search"
    assert len(document["request"]["search"]["items"]) == 1
    assert document["inputs"]["sources"] == {}
    opened = Stack.open(stack.config.work_dir)
    assert opened.request["kind"] == "search"
    assert opened.request["provider"] == "cdse"
