"""Tests for immutable local source snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from faninsar.processing import snapshot_local_source
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from pathlib import Path


def test_file_snapshot_is_stable_after_source_mutation(tmp_path: Path) -> None:
    """The snapshot keeps the bytes that were hashed before source mutation."""
    source = tmp_path / "orbit.EOF"
    source.write_bytes(b"orbit-v1")

    snapshot = snapshot_local_source(source, tmp_path / "snapshots")
    source.write_bytes(b"orbit-v2-with-a-different-size")

    assert snapshot.read_bytes() == b"orbit-v1"
    assert snapshot.total_bytes == len(b"orbit-v1")
    assert snapshot.path != source


def test_directory_snapshot_preserves_relative_entries(tmp_path: Path) -> None:
    """SAFE-like directory trees are copied and read by relative name."""
    source = tmp_path / "product.SAFE"
    (source / "annotation").mkdir(parents=True)
    (source / "measurement").mkdir()
    (source / "manifest.safe").write_text("manifest", encoding="utf-8")
    (source / "annotation" / "iw1.xml").write_text("annotation", encoding="utf-8")
    (source / "measurement" / "iw1.tiff").write_bytes(b"pixels")

    snapshot = snapshot_local_source(source, tmp_path / "snapshots")

    assert snapshot.kind == "directory"
    assert snapshot.path.name.endswith(".SAFE")
    assert snapshot.read_bytes("annotation/iw1.xml") == b"annotation"
    assert snapshot.read_bytes("measurement/iw1.tiff") == b"pixels"
    assert "../manifest.safe" not in snapshot.entry_paths


def test_snapshot_detects_payload_tampering(tmp_path: Path) -> None:
    """A changed snapshot payload fails hash validation before consumption."""
    source = tmp_path / "dem.tif"
    source.write_bytes(b"dem-bytes")
    snapshot = snapshot_local_source(source, tmp_path / "snapshots")
    snapshot.path.chmod(0o600)
    snapshot.path.write_bytes(b"tampered")

    with pytest.raises(
        InvalidProcessingStateError,
        match=r"(size|digest) changed",
    ):
        snapshot.read_bytes()


def test_snapshot_rejects_symlink_and_remote_sources(tmp_path: Path) -> None:
    """Symlink and remote path bypasses are rejected at admission."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    symlink = tmp_path / "source-link.bin"
    symlink.symlink_to(source)

    with pytest.raises(InvalidProcessingStateError, match="symlink"):
        snapshot_local_source(symlink, tmp_path / "snapshots")
    with pytest.raises(InvalidProcessingStateError, match="remote"):
        snapshot_local_source(
            "https://example.invalid/source.zip",
            tmp_path / "snapshots",
        )


def test_snapshot_enforces_file_limit(tmp_path: Path) -> None:
    """A directory with too many files is rejected before publication."""
    source = tmp_path / "product.SAFE"
    source.mkdir()
    (source / "one").write_bytes(b"1")
    (source / "two").write_bytes(b"2")

    with pytest.raises(InvalidProcessingStateError, match="file limit"):
        snapshot_local_source(source, tmp_path / "snapshots", max_files=1)
