"""Corpus downloader and offline cache boundary tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from faninsar.validation.provenance import validate_pipeline_rebuild_manifest
from tests.reference.corpus import (
    Artifact,
    ChecksumMismatchError,
    DownloadError,
    OfflineCacheMissError,
    resolve_artifact,
)


def _artifact(payload: bytes) -> Artifact:
    """Build a tiny artifact identity for a payload."""
    return Artifact(
        identifier="wire-fixture",
        url="https://example.invalid/fixture.bin",
        filename="fixture.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def test_valid_cached_artifact_is_reused_without_network(tmp_path: Path) -> None:
    """Reuse a valid cache entry without invoking the network fetch."""
    payload = b"frozen-reference"
    artifact = _artifact(payload)
    cached_path = tmp_path / artifact.filename
    cached_path.write_bytes(payload)

    def fail_fetch(_url: str, _destination: Path) -> None:
        pytest.fail("a valid cache entry must not trigger a download")

    assert resolve_artifact(artifact, tmp_path, fetch=fail_fetch) == cached_path


def test_corrupt_cached_bytes_are_rejected_before_network(tmp_path: Path) -> None:
    """Reject corrupt cache bytes before any network fetch is attempted."""
    artifact = _artifact(b"expected")
    (tmp_path / artifact.filename).write_bytes(b"corrupt")

    def fail_fetch(_url: str, _destination: Path) -> None:
        pytest.fail("corrupt cached bytes must be rejected before network access")

    with pytest.raises(ChecksumMismatchError, match=r"fixture\.bin"):
        resolve_artifact(artifact, tmp_path, fetch=fail_fetch)


def test_offline_cache_miss_has_no_network_side_effect(tmp_path: Path) -> None:
    """Raise a typed offline miss without contacting the network."""
    artifact = _artifact(b"expected")

    def fail_fetch(_url: str, _destination: Path) -> None:
        pytest.fail("offline mode must not access the network")

    with pytest.raises(OfflineCacheMissError, match="wire-fixture"):
        resolve_artifact(artifact, tmp_path, offline=True, fetch=fail_fetch)


def test_download_is_verified_then_promoted_atomically(tmp_path: Path) -> None:
    """Promote a verified download only after checksum validation succeeds."""
    payload = b"downloaded-reference"
    artifact = _artifact(payload)

    def fetch(_url: str, destination: Path) -> None:
        destination.write_bytes(payload)

    resolved = resolve_artifact(artifact, tmp_path, fetch=fetch)

    assert resolved.read_bytes() == payload
    assert not resolved.with_suffix(".bin.part").exists()


def test_bad_download_is_removed_and_never_promoted(tmp_path: Path) -> None:
    """Discard a checksum-failing download without promoting the target path."""
    artifact = _artifact(b"expected")

    def fetch(_url: str, destination: Path) -> None:
        destination.write_bytes(b"wrong")

    with pytest.raises(ChecksumMismatchError):
        resolve_artifact(artifact, tmp_path, fetch=fetch)

    assert not (tmp_path / artifact.filename).exists()
    assert not (tmp_path / f"{artifact.filename}.part").exists()


def test_stale_partial_download_is_replaced(tmp_path: Path) -> None:
    """Replace a stale partial download before writing a fresh payload."""
    payload = b"fresh"
    artifact = _artifact(payload)
    partial_path = tmp_path / f"{artifact.filename}.part"
    tmp_path.mkdir(exist_ok=True)
    partial_path.write_bytes(b"stale")

    def fetch(_url: str, destination: Path) -> None:
        assert not destination.exists()
        destination.write_bytes(payload)

    assert resolve_artifact(artifact, tmp_path, fetch=fetch).read_bytes() == payload


def test_fetch_success_without_output_is_not_misreported(tmp_path: Path) -> None:
    """Treat a no-output fetch as a download failure rather than success."""
    artifact = _artifact(b"expected")

    def misleading_fetch(_url: str, _destination: Path) -> None:
        return

    with pytest.raises(DownloadError, match="did not create"):
        resolve_artifact(artifact, tmp_path, fetch=misleading_fetch)


def test_interrupted_fetch_removes_partial_bytes(tmp_path: Path) -> None:
    """Remove partial bytes when a fetch is interrupted mid-write."""
    artifact = _artifact(b"expected")

    def interrupted_fetch(_url: str, destination: Path) -> None:
        destination.write_bytes(b"partial")
        message = "connection reset"
        raise OSError(message)

    with pytest.raises(OSError, match="connection reset"):
        resolve_artifact(artifact, tmp_path, fetch=interrupted_fetch)

    assert not (tmp_path / f"{artifact.filename}.part").exists()


def test_pipeline_rebuild_manifest_verifies_fixed_corpus() -> None:
    """Verify the frozen ISCE2 and InSAR.dev oracle corpus metadata."""
    summary = validate_pipeline_rebuild_manifest(
        Path("tests/reference/pipeline_rebuild_manifest.yaml")
    )

    assert summary.scene_count == 3
    assert summary.pair_count == 3
    assert summary.orbit_count == 3
    assert summary.dem_tile_count == 12
    assert summary.primary_processors == ("isce2", "insardev")
    assert summary.out_of_scope_hashes_match
