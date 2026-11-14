"""Regression tests for prepared footprint identity recovery."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from shapely.geometry import box

from faninsar.missions.s1 import S1Stack
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.identity import save_stack_identity
from faninsar.stack.session import Stack
from tests.stack.test_reopen import _stack


def test_selected_footprints_roundtrip_preserves_resume_identity(tmp_path: Path) -> None:
    stack = _stack(tmp_path)
    stack.prepare_data()
    stack.config.extra["selected_footprints"] = (box(0, 0, 1, 1),)
    save_stack_identity(stack)
    before = stack._coreg_resume_identity(
        "20240113", misreg_az_px=0.0, misreg_rg_px=0.0
    )
    reopened = Stack.open(stack.config.work_dir)
    assert reopened.config.extra["selected_footprints"] == (
        stack.config.extra["selected_footprints"]
    )
    assert reopened._coreg_resume_identity(
        "20240113", misreg_az_px=0.0, misreg_rg_px=0.0
    ) == before


def _legacy_s1(tmp_path: Path):
    stack = _stack(tmp_path)
    stack.prepare_data()
    path = stack.config.work_dir / "stack.json"
    document = json.loads(path.read_text())
    document["mission"] = "S1Stack"
    document["science_extra"].pop("selected_footprints", None)
    path.write_text(json.dumps(document))
    return stack, path


def test_legacy_s1_open_restores_only_local_footprints(tmp_path: Path, monkeypatch) -> None:
    stack, path = _legacy_s1(tmp_path)
    footprints = (box(0, 0, 1, 1), box(1, 0, 2, 1))
    calls = []

    def open_safe(source, *, polarizations):
        calls.append((source, polarizations))
        return SimpleNamespace(swaths=[SimpleNamespace(bursts=[
            SimpleNamespace(footprint=footprints[len(calls) - 1])
        ])])

    def forbidden(*args, **kwargs):
        pytest.fail("open must not prepare, acquire auxiliaries or materialize DEMs")

    monkeypatch.setattr("faninsar.missions.s1.safe.open_safe_product", open_safe)
    for method in ("prepare_data", "_ensure_aux_cal", "_ensure_precise_orbits", "materialize_dem"):
        monkeypatch.setattr(S1Stack, method, forbidden)
    original = path.read_bytes()
    reopened = Stack.open(stack.config.work_dir)
    assert reopened.config.extra["selected_footprints"] == footprints
    assert len(calls) == 2
    assert reopened.data_ready
    assert path.read_bytes() == original


def test_unreadable_legacy_footprints_allow_inspection_not_processing(tmp_path: Path, monkeypatch) -> None:
    stack, path = _legacy_s1(tmp_path)

    def unavailable(*args, **kwargs):
        raise OSError("annotation unavailable")

    monkeypatch.setattr("faninsar.missions.s1.safe.open_safe_product", unavailable)
    original = path.read_bytes()
    reopened = Stack.open(stack.config.work_dir)
    assert set(reopened.scenes) == {"20240101", "20240113"}
    assert not reopened.data_ready
    with pytest.raises(InvalidProcessingStateError, match="not prepared"):
        reopened.coregister_scenes()
    assert path.read_bytes() == original
