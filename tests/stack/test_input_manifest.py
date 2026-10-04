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


def test_manifest_truncated_source_fails_the_size_tripwire(tmp_path: Path) -> None:
    """A source truncation invalidates a previously valid manifest."""
    source = tmp_path / "source.safe"
    source.write_bytes(b"source")
    manifest_path = tmp_path / "input_manifest.json"
    document = build_manifest(
        mission="test",
        catalog_paths={"20240101": (source,)},
        prepared=MissionPreparedInputs(provenance={"asset": "https://x/a?sig=secret"}),
    )
    write_manifest(manifest_path, document)

    assert document["schema"] == "faninsar.stack.input_manifest.v2"
    source.write_bytes(b"src")
    with pytest.raises(ValueError, match="size mismatch"):
        read_manifest(manifest_path)


def test_manifest_same_size_edit_reuses_without_hashing(tmp_path: Path) -> None:
    """Same-size edits are not detected: no hashing happens by design."""
    source = tmp_path / "source.safe"
    source.write_bytes(b"source")
    manifest_path = tmp_path / "input_manifest.json"
    document = build_manifest(
        mission="test",
        catalog_paths={"20240101": (source,)},
        prepared=MissionPreparedInputs(),
    )
    write_manifest(manifest_path, document)

    source.write_bytes(b"surce!")
    assert read_manifest(manifest_path)["schema"] == "faninsar.stack.input_manifest.v2"


def test_manifest_rejects_truncated_directory_source(tmp_path: Path) -> None:
    """Removing a file inside a directory SAFE fails the size tripwire."""
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

    annotation.write_bytes(b"")
    with pytest.raises(ValueError, match="size mismatch"):
        read_manifest(manifest_path)


def test_v1_manifest_with_sha256_loads_with_inert_fields(tmp_path: Path) -> None:
    """A legacy v1 manifest loads; its sha256 fields are never compared."""
    source = tmp_path / "source.safe"
    source.write_bytes(b"source")
    manifest_path = tmp_path / "input_manifest.json"
    legacy = {
        "schema": "faninsar.stack.input_manifest.v1",
        "mission": "test",
        "sources": {
            "20240101": [
                {
                    "path": str(source),
                    "kind": "file",
                    "size": source.stat().st_size,
                    "sha256": "00" * 32,
                }
            ]
        },
        "orbits": {},
        "calibration": [],
        "selected_footprints": [],
        "provenance": {},
    }
    write_manifest(manifest_path, legacy)

    source.write_bytes(b"changd")
    loaded = read_manifest(manifest_path)
    assert loaded["schema"] == "faninsar.stack.input_manifest.v1"
    assert loaded["sources"]["20240101"][0]["sha256"] == "00" * 32


def test_download_verification_record_is_anchored(tmp_path: Path) -> None:
    """A transfer-layer verification record anchors its input source entry."""
    source = tmp_path / "source.safe"
    source.write_bytes(b"source")
    document = build_manifest(
        mission="test",
        catalog_paths={"20240101": (source,)},
        prepared=MissionPreparedInputs(),
        verification={
            str(source): {
                "algorithm": "md5",
                "checksum": "d41d8cd98f00b204e9800998ecf8427e",
                "status": "verified",
            }
        },
    )

    entry = document["sources"]["20240101"][0]
    assert entry["verification"]["status"] == "verified"
    assert entry["verification"]["algorithm"] == "md5"


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
