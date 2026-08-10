"""Tests for the crash-safe prepared generation store."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from faninsar.processing.contracts import PreparedIdentity
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.prepared_store import PreparedGenerationStore

if TYPE_CHECKING:
    from pathlib import Path


def _digest(value: object) -> str:
    """Return a deterministic test digest."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _identity(generation_id: str = "generation-1") -> PreparedIdentity:
    """Build a valid prepared identity for store tests."""
    return PreparedIdentity(
        provider_schema="prepared_geometry_provider.v1",
        parent_generation_id=generation_id,
        domain="radar",
        ordered_source_ids=("reference", "secondary"),
        common_domain_digest=_digest("common"),
        policy_digest=_digest("policy"),
        schema_code_backend_device_digest=_digest("runtime"),
        payload_manifest_digest=_digest("payload"),
    )


def test_generation_is_published_and_read_only_payloads_are_hash_checked(
    tmp_path: Path,
) -> None:
    """CURRENT and payload bytes survive a normal publish/read lifecycle."""
    store = PreparedGenerationStore(tmp_path / "prepared", max_payload_bytes=1024)
    record = store.stage(
        "generation-1",
        identity=_identity(),
        metadata={"units": ["burst-0"]},
        payloads={"controls.bin": b"controls", "mask.bin": b"mask"},
    )
    assert record.payload_names == ("controls.bin", "mask.bin")
    store.publish("generation-1")

    lease = store.pin()
    reader = store.open(lease)
    assert reader.generation_id == "generation-1"
    assert reader.read_payload("controls.bin") == b"controls"
    assert reader.manifest["metadata"] == {"units": ["burst-0"]}
    store.unpin(lease)
    store.close("generation-1")


def test_generation_reader_rejects_payload_tampering(tmp_path: Path) -> None:
    """A payload changed after publication is rejected before consumption."""
    store = PreparedGenerationStore(tmp_path / "prepared", max_payload_bytes=1024)
    store.stage(
        "generation-1",
        identity=_identity(),
        metadata={},
        payloads={"controls.bin": b"controls"},
    )
    store.publish("generation-1")
    lease = store.pin()
    payload_path = (
        tmp_path / "prepared" / "generations" / "generation-1" / "controls.bin"
    )
    payload_path.write_bytes(b"tampered")
    with pytest.raises(InvalidProcessingStateError):
        store.open(lease).read_payload("controls.bin")


def test_generation_pin_must_be_released_before_close_or_abort(tmp_path: Path) -> None:
    """The lifecycle cannot close or quarantine a live generation pin."""
    store = PreparedGenerationStore(tmp_path / "prepared", max_payload_bytes=1024)
    store.stage(
        "generation-1",
        identity=_identity(),
        metadata={},
        payloads={"controls.bin": b"controls"},
    )
    store.publish("generation-1")
    lease = store.pin()
    with pytest.raises(InvalidProcessingStateError):
        store.close("generation-1")
    with pytest.raises(InvalidProcessingStateError):
        store.abort("generation-1")
    store.unpin(lease)
    quarantined = store.abort("generation-1")
    assert quarantined.name.startswith(".generation-1.aborted")


def test_generation_rejects_symlinked_root_and_payload_name_traversal(
    tmp_path: Path,
) -> None:
    """The store refuses path aliases and traversal-shaped payload names."""
    target = tmp_path / "real"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(InvalidProcessingStateError):
        PreparedGenerationStore(alias)

    store = PreparedGenerationStore(tmp_path / "prepared", max_payload_bytes=1024)
    with pytest.raises(InvalidProcessingStateError):
        store.stage(
            "generation-1",
            identity=_identity(),
            metadata={},
            payloads={"../escape": b"bad"},
        )
