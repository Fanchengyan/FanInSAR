"""Cross-process generation lease heartbeat and reclamation tests."""

from __future__ import annotations

import json
import multiprocessing
import time
from contextlib import suppress
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING

import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.stack.artifact_transaction import (
    collect_generations,
    commit_generation,
    open_current_generation,
    stage_generation,
)

if TYPE_CHECKING:
    from multiprocessing.queues import Queue


def _publish(root: Path) -> str:
    """Publish one minimal immutable generation."""
    with stage_generation(
        root,
        "ifg",
        final_bytes=7,
        temporary_bytes=7,
        file_count=1,
    ) as (generation_id, staging):
        (staging / "payload.bin").write_bytes(b"payload")
        commit_generation(root, "ifg", generation_id, staging, manifest_digest="a" * 64)
    return generation_id


def _hold_current(root_name: str, ready: Queue[str], release: Queue[bool]) -> None:
    """Open the current generation in a child process until released."""
    opened = open_current_generation(Path(root_name), "ifg")
    ready.put(opened.generation_id)
    release.get(timeout=20)
    opened.lease.close()


def test_lease_heartbeat_updates_owner_nonce_metadata(tmp_path: Path) -> None:
    """Heartbeat updates one nonce-bound durable lease record atomically."""
    root = tmp_path / "artifacts"
    _publish(root)
    opened = open_current_generation(root, "ifg", lease_ttl_s=10, lease_grace_s=10)
    try:
        before = json.loads(opened.lease.path.read_text(encoding="utf-8"))
        time.sleep(0.001)
        opened.lease.heartbeat()
        after = json.loads(opened.lease.path.read_text(encoding="utf-8"))
        assert after["nonce"] == before["nonce"]
        assert after["heartbeat_at_ns"] > before["heartbeat_at_ns"]
        assert after["expires_at_ns"] > before["expires_at_ns"]
    finally:
        opened.lease.close()


def test_lease_renewal_failure_is_explicit(tmp_path: Path) -> None:
    """A missing lease record aborts renewal instead of continuing unpinned."""
    root = tmp_path / "artifacts"
    _publish(root)
    opened = open_current_generation(root, "ifg")
    opened.lease.path.unlink()
    with pytest.raises(InvalidProcessingStateError, match="control file"):
        opened.lease.renew()
    opened.lease.close()


def test_expired_lease_is_retained_during_grace_then_collected(tmp_path: Path) -> None:
    """Expired readers retain their generation until the grace deadline."""
    root = tmp_path / "artifacts"
    first_id = _publish(root)
    opened = open_current_generation(root, "ifg", lease_ttl_s=1, lease_grace_s=1)
    second_id = _publish(root)
    try:
        now_ns = time.time_ns()
        metadata = json.loads(opened.lease.path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "issued_at_ns": now_ns - 3_000_000_000,
                "heartbeat_at_ns": now_ns - 2_000_000_000,
                "expires_at_ns": now_ns - 1_000_000_000,
                "grace_until_ns": now_ns + 1_000_000_000,
            }
        )
        opened.lease.path.write_text(json.dumps(metadata), encoding="utf-8")
        assert collect_generations(root, "ifg", now_ns=now_ns) == ()
        metadata["grace_until_ns"] = now_ns - 1
        opened.lease.path.write_text(json.dumps(metadata), encoding="utf-8")
        assert collect_generations(root, "ifg", now_ns=now_ns) == (first_id,)
        assert (root / ".ifg_generations" / second_id).is_dir()
    finally:
        opened.lease.close()


def test_cross_process_lease_blocks_collection_until_close(tmp_path: Path) -> None:
    """A child-process lease protects a non-current generation from GC."""
    root = tmp_path / "artifacts"
    first_id = _publish(root)
    context = multiprocessing.get_context("spawn")
    ready: Queue[str] = context.Queue()
    release: Queue[bool] = context.Queue()
    child = context.Process(target=_hold_current, args=(str(root), ready, release))
    child.start()
    try:
        assert ready.get(timeout=20) == first_id
        _publish(root)
        assert collect_generations(root, "ifg") == ()
        release.put(True)
        child.join(timeout=20)
        assert child.exitcode == 0
        assert collect_generations(root, "ifg") == (first_id,)
    finally:
        if child.is_alive():
            release.put(True)
            child.join(timeout=5)
        with suppress(Exception):
            ready.close()
            release.close()
