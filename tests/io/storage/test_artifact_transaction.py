"""Security tests for immutable artifact transactions."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest

from faninsar.io.storage.artifact_transaction import (
    _PinnedDescriptorState,
    _PinnedGenerationPath,
    commit_generation,
    open_current_generation,
    stage_generation,
)
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from pathlib import Path


def test_pinned_generation_path_constructs(tmp_path: Path) -> None:
    """Pinned paths must construct when pathlib.Path.__init__ is object.__init__."""
    descriptor = os.open(tmp_path, os.O_RDONLY)
    state = _PinnedDescriptorState(
        generation_descriptor=descriptor,
        display_path=tmp_path,
        payload_descriptors=[],
    )
    pinned = _PinnedGenerationPath(state, tmp_path)
    assert pinned._display_path == tmp_path
    child = pinned / "payload.npy"
    assert child._relative_parts == ("payload.npy",)


def test_open_current_generation_reopens_with_pinned_path(tmp_path: Path) -> None:
    """Opening a published generation must work on Linux pathlib runtimes."""
    root = tmp_path / "artifacts"
    _publish_minimal_generation(root)

    opened = open_current_generation(root, "ifg")
    try:
        assert (opened.path / "payload.bin").read_bytes() == b"payload"
    finally:
        opened.lease.close()


def test_network_commit_failure_restores_previous_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Network publication leaves the previous snapshot readable."""
    import faninsar.io.storage.artifact_transaction as transaction

    root = tmp_path / "network"

    def publish(manifest: dict[str, str]) -> str:
        with stage_generation(
            root,
            "network",
            final_bytes=7,
            temporary_bytes=7,
            file_count=1,
        ) as (generation_id, staging):
            (staging / "payload.bin").write_bytes(b"payload")
            commit_generation(
                root,
                "network",
                generation_id,
                staging,
                manifest_digest="a" * 64,
                compatibility_manifest=manifest,
            )
        return generation_id

    first_manifest = {"generation": "first"}
    first_generation = publish(first_manifest)
    previous_manifest = (root / "manifest.json").read_bytes()
    previous_current = (root / "CURRENT").read_bytes()
    current = json.loads(previous_current)
    assert current["schema"] == "faninsar_artifact_current_v1"
    assert current["namespace"] == "network"
    assert current["schema_version"] == "network_current_v1"

    original_atomic_control = transaction._atomic_control_at

    def fail_current(
        directory_descriptor: int,
        name: str,
        value: object,
        **kwargs: object,
    ) -> None:
        if name == "CURRENT":
            message = "injected CURRENT publication failure"
            raise OSError(message)
        original_atomic_control(
            directory_descriptor,
            name,
            value,
            **kwargs,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(transaction, "_atomic_control_at", fail_current)
    with (  # noqa: PT012
        pytest.raises(OSError, match="injected"),
        stage_generation(
            root,
            "network",
            final_bytes=7,
            temporary_bytes=7,
            file_count=1,
        ) as (generation_id, staging),
    ):
        (staging / "payload.bin").write_bytes(b"new")
        commit_generation(
            root,
            "network",
            generation_id,
            staging,
            manifest_digest="b" * 64,
            compatibility_manifest={"generation": "second"},
        )

    assert (root / "manifest.json").read_bytes() == previous_manifest
    assert (root / "CURRENT").read_bytes() == previous_current
    opened = open_current_generation(root, "network")
    try:
        assert opened.generation_id == first_generation
    finally:
        opened.lease.close()


def test_staging_rejects_symlinked_namespace_directory(tmp_path: Path) -> None:
    """A namespace component cannot redirect staging outside the root."""
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / ".ifg_generations").symlink_to(outside)

    with (
        pytest.raises(InvalidProcessingStateError, match="unsafe"),
        stage_generation(
            root,
            "ifg",
            final_bytes=0,
            temporary_bytes=0,
            file_count=0,
        ),
    ):
        pass

    assert list(outside.iterdir()) == []


def test_staging_accepts_group_writable_mount_ancestor(tmp_path: Path) -> None:
    """Only the managed root, not an ordinary mount ancestor, must be private."""
    ancestor = tmp_path / "mount"
    ancestor.mkdir(mode=0o770)
    ancestor.chmod(0o770)
    root = ancestor / "artifacts"

    with stage_generation(
        root,
        "ifg",
        final_bytes=0,
        temporary_bytes=0,
        file_count=0,
    ):
        pass

    assert root.stat().st_mode & 0o777 == 0o700


def test_staging_forces_private_payload_permissions_under_group_umask(
    tmp_path: Path,
) -> None:
    """Staged payloads remain private when the process umask is permissive."""
    root = tmp_path / "artifacts"
    previous_umask = os.umask(0o002)
    try:
        with stage_generation(
            root,
            "ifg",
            final_bytes=7,
            temporary_bytes=7,
            file_count=1,
        ) as (generation_id, staging):
            payload = staging / "payload.bin"
            payload.write_bytes(b"payload")
            assert payload.stat().st_mode & 0o077 == 0
            commit_generation(
                root,
                "ifg",
                generation_id,
                staging,
                manifest_digest="a" * 64,
            )
    finally:
        os.umask(previous_umask)


def test_commit_rejects_hardlinked_payload(tmp_path: Path) -> None:
    """A staged payload with another name cannot enter a generation."""
    root = tmp_path / "artifacts"

    with stage_generation(
        root,
        "ifg",
        final_bytes=7,
        temporary_bytes=7,
        file_count=1,
    ) as (generation_id, staging):
        payload = staging / "payload.bin"
        payload.write_bytes(b"payload")
        os.link(payload, tmp_path / "alias.bin")

        with pytest.raises(InvalidProcessingStateError, match="hardlink"):
            commit_generation(
                root,
                "ifg",
                generation_id,
                staging,
                manifest_digest="a" * 64,
            )

    assert not (root / "CURRENT").exists()


def _publish_minimal_generation(root: Path) -> None:
    """Publish one regular payload for hostile-reader tests."""
    with stage_generation(
        root,
        "ifg",
        final_bytes=7,
        temporary_bytes=7,
        file_count=1,
    ) as (generation_id, staging):
        (staging / "payload.bin").write_bytes(b"payload")
        commit_generation(
            root,
            "ifg",
            generation_id,
            staging,
            manifest_digest="a" * 64,
        )


def test_open_rejects_hardlinked_current_control(tmp_path: Path) -> None:
    """A multiply linked CURRENT record cannot select a generation."""
    root = tmp_path / "artifacts"
    _publish_minimal_generation(root)
    os.link(root / "CURRENT", tmp_path / "current-alias")

    with pytest.raises(InvalidProcessingStateError, match="control file is unsafe"):
        open_current_generation(root, "ifg")


def test_commit_remains_bound_to_root_descriptor_after_ancestor_replacement(
    tmp_path: Path,
) -> None:
    """Replacing the root pathname cannot redirect an in-flight commit."""
    root = tmp_path / "artifacts"
    relocated = tmp_path / "relocated"

    with stage_generation(
        root,
        "ifg",
        final_bytes=7,
        temporary_bytes=7,
        file_count=1,
    ) as (generation_id, staging):
        (staging / "payload.bin").write_bytes(b"trusted")
        root.rename(relocated)
        (root / ".ifg_staging" / generation_id).mkdir(parents=True)
        (root / ".ifg_generations").mkdir()
        (root / ".ifg_leases").mkdir()
        (root / ".ifg_staging" / generation_id / "payload.bin").write_bytes(b"attacker")

        commit_generation(
            root,
            "ifg",
            generation_id,
            staging,
            manifest_digest="a" * 64,
        )

    assert (relocated / "CURRENT").is_file()
    assert not (root / "CURRENT").exists()
    assert (
        root / ".ifg_staging" / generation_id / "payload.bin"
    ).read_bytes() == b"attacker"
    assert (
        relocated / ".ifg_generations" / generation_id / "payload.bin"
    ).read_bytes() == b"trusted"


def test_open_generation_payload_remains_pinned_after_root_replacement(
    tmp_path: Path,
) -> None:
    """An opened generation never re-resolves its payload through root paths."""
    root = tmp_path / "artifacts"
    relocated = tmp_path / "relocated"
    _publish_minimal_generation(root)
    opened = open_current_generation(root, "ifg")

    root.rename(relocated)
    attacker_generation = root / ".ifg_generations" / opened.generation_id
    attacker_generation.mkdir(parents=True)
    (attacker_generation / "payload.bin").write_bytes(b"attacker")

    try:
        assert (opened.path / "payload.bin").read_bytes() == b"payload"
    finally:
        opened.lease.close()
