"""Focused tests for the simple Stack input manifest."""

# The repository intentionally keeps tests as a namespace package.
# ruff: noqa: INP001

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from faninsar.stack import MissionPreparedInputs, Stack
from faninsar.stack.input_manifest import (
    build_manifest,
    read_manifest,
    write_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


class _BrokenPreparationStack(Stack):
    """Stack whose mission preparation deliberately fails."""

    def _prepare_mission(self) -> MissionPreparedInputs:
        """Raise after the ordinary local Stack has been constructed."""
        message = "mission preparation failed"
        raise RuntimeError(message)


def test_manifest_round_trip_rehashes_nested_source_bindings(tmp_path: Path) -> None:
    """A source file mutation invalidates a previously valid manifest."""
    source = tmp_path / "source.safe"
    source.write_bytes(b"source")
    manifest_path = tmp_path / "input_manifest.json"
    document = build_manifest(
        mission="test",
        catalog_paths={"20240101": (source,)},
        prepared=MissionPreparedInputs(provenance={"asset": "https://x/a?sig=secret"}),
    )
    write_manifest(manifest_path, document)

    assert read_manifest(manifest_path)["schema"] == "faninsar.stack.input_manifest.v1"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_manifest(manifest_path)


def test_manifest_rehashes_safe_directory_contents(tmp_path: Path) -> None:
    """Changing a file inside a directory SAFE invalidates its manifest."""
    source = tmp_path / "S1_TEST.SAFE"
    annotation = source / "annotation" / "test.xml"
    annotation.parent.mkdir(parents=True)
    annotation.write_bytes(b"first")
    manifest_path = tmp_path / "input_manifest.json"
    document = build_manifest(
        mission="test",
        catalog_paths={"20240101": (source,)},
        prepared=MissionPreparedInputs(),
    )
    write_manifest(manifest_path, document)

    annotation.write_bytes(b"second")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_manifest(manifest_path)


def test_manifest_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    """Arbitrary provider metadata cannot enter the allowlisted manifest."""
    path = tmp_path / "input_manifest.json"
    path.write_text('{"schema":"faninsar.stack.input_manifest.v1","secret":"x"}')
    with pytest.raises(ValueError, match="unknown fields"):
        read_manifest(path)


def test_stack_manifest_reuse_requires_processing_identity(tmp_path: Path) -> None:
    """Manifest reuse rejects a changed coregistration option."""
    first = tmp_path / "S1_20240101.SAFE"
    second = tmp_path / "S1_20240113.SAFE"
    first.mkdir()
    second.mkdir()
    stack = Stack._from_safes(
        [first, second],
        work_dir=tmp_path / "work",
        activation_mode="reference",
    )
    stack.prepare()
    stack.config.multilook = (3, 3)

    with pytest.raises(ValueError, match="processing options"):
        stack.prepare()


def test_failed_mission_preparation_leaves_stack_unprepared(tmp_path: Path) -> None:
    """A mission failure after scene setup publishes no prepared state."""
    first = tmp_path / "S1_20240101.SAFE"
    second = tmp_path / "S1_20240113.SAFE"
    first.mkdir()
    second.mkdir()
    stack = _BrokenPreparationStack._from_safes(
        [first, second],
        work_dir=tmp_path / "work",
        activation_mode="reference",
    )

    with pytest.raises(RuntimeError, match="mission preparation failed"):
        stack.prepare()

    assert stack._prepared is False
    assert stack._mission_prepared is None
    assert not (tmp_path / "work" / "input_manifest.json").exists()
