"""Focused lifecycle/provider regressions for PROPOSAL-0037."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from faninsar.processing.errors import PairConfigurationMigrationError
from faninsar.missions.s1 import S1Stack
from faninsar.stack import (
    SourceHandle,
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

    def fake_producer(
        reference: SourceHandle, secondary: SourceHandle, **kwargs: object
    ) -> object:
        calls.append((reference, secondary))
        assert kwargs["output_dir"] == tmp_path / "pair"
        return SimpleNamespace()

    monkeypatch.setattr(
        "faninsar.missions.s1.processing.produce_interferogram_pair",
        fake_producer,
    )
    stack._produce_pair(source[0], source[1], output_dir=tmp_path / "pair")
    assert len(calls) == 1
    assert calls == [(source[0], source[1])]


def test_stack_provider_receives_opaque_handles_without_path_leak(
    tmp_path: Path,
) -> None:
    """Provider callbacks receive handles whose repr and string are opaque."""
    source = [_safe(tmp_path, "20240101"), _safe(tmp_path, "20240113")]
    stack = Stack._from_safes(
        source,
        work_dir=tmp_path / "work",
        activation_mode="reference",
    )
    received: list[tuple[SourceHandle, SourceHandle]] = []

    def produce_pair(
        reference: SourceHandle,
        secondary: SourceHandle,
        *,
        output_dir: Path,
        options: dict[str, object],
    ) -> SimpleNamespace:
        del output_dir, options
        received.append((reference, secondary))
        return SimpleNamespace()

    stack.scene_provider = StackSceneProvider(
        name="opaque-test",
        produce_pair=produce_pair,
    )
    stack._produce_pair(source[0], source[1], output_dir=tmp_path / "pair")

    reference, secondary = received[0]
    assert str(source[0]) not in repr(reference)
    assert str(source[1]) not in repr(secondary)
    assert str(source[0]) not in str(reference)
    assert str(source[1]) not in str(secondary)


def test_stack_reference_is_the_only_public_common_acquisition_name(
    tmp_path: Path,
) -> None:
    """Stack constructors expose Reference and reject the removed spelling."""
    source = [_safe(tmp_path, "20240101"), _safe(tmp_path, "20240113")]
    stack = S1Stack.from_safes(
        source,
        work_dir=tmp_path / "work",
        reference="20240113",
        activation_mode="reference",
    )

    assert stack.reference == "20240113"
    assert not hasattr(stack, "master")
    with pytest.raises(PairConfigurationMigrationError, match="reference"):
        S1Stack.from_safes(
            source,
            work_dir=tmp_path / "old-work",
            master="20240113",
        )
