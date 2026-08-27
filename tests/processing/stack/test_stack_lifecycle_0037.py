"""Focused lifecycle/provider regressions for PROPOSAL-0037."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from faninsar.processing.stack import (
    S1Stack,
    Stack,
    StackSceneProvider,
    UnsupportedStackCapabilityError,
)

if TYPE_CHECKING:
    from pathlib import Path


def _safe(path: Path, date_id: str, suffix: str = "") -> Path:
    """Create one small SAFE-like source directory."""
    result = path / f"S1A_IW_SLC__1SDV_{date_id}T000000{suffix}.SAFE"
    result.mkdir()
    return result


def test_s1_stack_installs_provider_and_groups_source_segments(tmp_path: Path) -> None:
    """Raw S1 initialization retains all segments under one acquisition."""
    first = _safe(tmp_path, "20240101", "_frame-a")
    second = _safe(tmp_path, "20240101", "_frame-b")
    later = _safe(tmp_path, "20240113")

    stack = S1Stack.from_safes(
        [first, second, later],
        work_dir=tmp_path / "work",
        activation_mode="reference",
    )

    assert stack.reference == "20240101"
    assert stack.catalog.paths_for("20240101") == (first, second)
    assert isinstance(stack.scene_provider, StackSceneProvider)


def test_base_stack_provider_boundary_fails_closed(tmp_path: Path) -> None:
    """A mission-neutral Stack cannot guess a production engine."""
    source = [_safe(tmp_path, "20240101"), _safe(tmp_path, "20240113")]
    stack = Stack._from_safes(
        source,
        work_dir=tmp_path / "work",
        activation_mode="reference",
    )

    with pytest.raises(UnsupportedStackCapabilityError, match="scene-production"):
        stack._produce_pair(
            source[0],
            source[1],
            output_dir=tmp_path / "pair",
        )


def test_s1_provider_owns_pair_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1 production dispatch is reachable only through its provider."""
    source = [_safe(tmp_path, "20240101"), _safe(tmp_path, "20240113")]
    stack = S1Stack.from_safes(
        source,
        work_dir=tmp_path / "work",
        activation_mode="reference",
    )
    calls: list[tuple[object, object]] = []

    def fake_run_pair(reference: object, secondary: object, **kwargs: object) -> object:
        calls.append((reference, secondary))
        assert kwargs["output_dir"] == tmp_path / "pair"
        return SimpleNamespace()

    monkeypatch.setattr(
        "faninsar.processing.pipeline.production.run_pair", fake_run_pair
    )
    stack._produce_pair(source[0], source[1], output_dir=tmp_path / "pair")
    assert calls == [(source[0], source[1])]
