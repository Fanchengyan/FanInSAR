"""Durable intent authority over legacy preparation records (PROPOSAL-0088)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faninsar.data.query import BoundingBox
from faninsar.missions.s1 import S1Stack
from faninsar.processing.geometry import ConstantDEM
from faninsar.stack.identity import STACK_JSON_NAME, save_stack_identity
from faninsar.stack.session import Stack
from faninsar.stack.workspace import WorkspaceConflictError


def _safe(tmp_path: Path, date_id: str) -> Path:
    path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
    path.mkdir(exist_ok=True)
    return path


def _orbit(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"EOF\n")
    return path


def _document(work: Path) -> dict[str, object]:
    return json.loads((work / STACK_JSON_NAME).read_text(encoding="utf-8"))


def _prepared_workspace(
    tmp_path: Path,
    work: Path,
    *,
    pins: dict[str, str] | None,
    resolved: dict[str, str],
    dem: ConstantDEM | None = None,
) -> S1Stack:
    """Build a prepared-like workspace whose resolved orbit is in its extra.

    Preparation records the date bindings it actually used in the live
    ``science_extra``; only a caller pin also appears in the authoritative
    request.
    """
    stack = S1Stack.from_safes(
        [_safe(tmp_path, "20240101")],
        work_dir=work,
        polarizations=("VV",),
        orbit_paths=pins,
        dem=dem or ConstantDEM(0),
    )
    stack.config.extra["orbit_paths"] = dict(resolved)
    save_stack_identity(stack)
    return stack


def test_open_prefers_authoritative_intent_over_stale_legacy_request(
    tmp_path: Path,
) -> None:
    """An extension survives reopen even when a legacy request view is stale."""
    original = _safe(tmp_path, "20240101")
    stale = _safe(tmp_path, "20240202")
    work = tmp_path / "work"
    stack = S1Stack.from_safes([original], work_dir=work, dem=ConstantDEM(0))
    second = BoundingBox(1.0, 0.0, 2.0, 1.0, crs="EPSG:4326")
    stack.extend(roi=second)

    request_path = work / "inputs" / "request.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps(
            {
                "schema": "faninsar.preparation.v1",
                "mission": "S1Stack",
                "source": {"kind": "safes", "paths": [str(stale)]},
                "science": {},
                "extra": {},
            }
        ),
        encoding="utf-8",
    )

    opened = Stack.open(work)

    assert tuple(Path(path) for path in opened.request["paths"]) == (original,)
    assert opened._request_config.roi is not None
    assert opened._request_config.roi.right == second.right


def test_pinned_orbit_survives_prepare_and_repeated_factory_call(
    tmp_path: Path,
) -> None:
    """A caller orbit pin compares equal after preparation and reopen."""
    orbit = _orbit(tmp_path, "S1A_OPER_AUX_POEORB.EOF")
    pins = {"20240101": str(orbit)}
    work = tmp_path / "work"
    _prepared_workspace(tmp_path, work, pins=pins, resolved=pins)

    opened = Stack.open(work)

    assert dict(opened.request["orbit_paths"]) == pins
    document = _document(work)
    assert document["requested_science_extra"]["orbit_paths"] == pins
    revision = opened.intent_revision
    before = (work / STACK_JSON_NAME).read_bytes()

    repeated = S1Stack.from_safes(
        [_safe(tmp_path, "20240101")],
        work_dir=work,
        polarizations=("VV",),
        orbit_paths=pins,
        dem=ConstantDEM(0),
    )

    assert repeated.intent_revision == revision
    assert (work / STACK_JSON_NAME).read_bytes() == before


def test_derived_orbit_binding_is_neither_promoted_nor_a_conflict(
    tmp_path: Path,
) -> None:
    """An auto-resolved orbit stays execution state across reopen and repeat."""
    orbit = _orbit(tmp_path, "S1A_OPER_AUX_POEORB.EOF")
    resolved = {"20240101": str(orbit)}
    work = tmp_path / "work"
    _prepared_workspace(tmp_path, work, pins=None, resolved=resolved)

    opened = Stack.open(work)

    document = _document(work)
    assert "orbit_paths" not in document["requested_science_extra"]
    assert document["science_extra"]["orbit_paths"] == resolved
    assert opened.intent_revision == 1
    before = (work / STACK_JSON_NAME).read_bytes()

    repeated = S1Stack.from_safes(
        [_safe(tmp_path, "20240101")],
        work_dir=work,
        polarizations=("VV",),
        dem=ConstantDEM(0),
    )

    assert repeated.intent_revision == 1
    assert (work / STACK_JSON_NAME).read_bytes() == before


def test_changed_orbit_resolution_dem_or_source_still_conflicts(
    tmp_path: Path,
) -> None:
    """Only the caller-pinned contract compares; real science changes reject."""
    orbit = _orbit(tmp_path, "S1A_OPER_AUX_POEORB.EOF")
    changed_orbit = _orbit(tmp_path, "S1A_OPER_AUX_POEORB_CHANGED.EOF")
    pins = {"20240101": str(orbit)}
    work = tmp_path / "work"
    _prepared_workspace(tmp_path, work, pins=pins, resolved=pins)
    Stack.open(work)
    before = (work / STACK_JSON_NAME).read_bytes()

    variants = (
        {"orbit_paths": {"20240101": str(changed_orbit)}},
        {"resolution_m": 60.0},
        {"dem": ConstantDEM(5)},
        {"paths": [_safe(tmp_path, "20240202")]},
    )
    for variant in variants:
        options = dict(variant)
        paths = options.pop("paths", [_safe(tmp_path, "20240101")])
        kwargs = {"dem": ConstantDEM(0), **options}
        with pytest.raises(WorkspaceConflictError):
            S1Stack.from_safes(paths, work_dir=work, **kwargs)
        assert (work / STACK_JSON_NAME).read_bytes() == before
