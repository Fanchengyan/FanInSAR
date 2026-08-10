"""Crash-safe local storage for prepared geometry generations.

The numerical provider is deliberately separate from this module.  This
module owns the filesystem boundary used by a provider: immutable generation
directories, a manifest-last ``CURRENT`` pointer, payload hashes, and an
in-process pin/close lifecycle.  A solver can therefore publish controls and
LUTs without exposing partially written files to Pair, geo, or Stack readers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faninsar.logging import setup_logger
from faninsar.processing.contracts.prepared_geometry import (
    PROVIDER_SCHEMA,
    PreparedIdentity,
)
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)

_GENERATION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_PAYLOAD_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_STORE_SCHEMA = "prepared_generation_store.v1"


def _canonical_json(value: object) -> bytes:
    """Serialize a JSON-compatible value deterministically."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"generation metadata is not canonicalizable: {error}")


def _digest(value: object) -> str:
    """Return the SHA-256 digest of canonical JSON."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    """Reject symlinked path components before opening a store path."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            reject_invalid_state(f"prepared store path contains a symlink: {current}")


def _require_generation_id(generation_id: str) -> None:
    """Validate a bounded content-addressed generation identifier."""
    if not isinstance(generation_id, str) or (
        _GENERATION_RE.fullmatch(generation_id) is None
    ):
        reject_invalid_state("generation_id must match the bounded lowercase grammar")


def _require_payload_name(payload_name: str) -> None:
    """Validate a payload basename and reject path traversal."""
    if (
        not isinstance(payload_name, str)
        or Path(payload_name).name != payload_name
        or _PAYLOAD_RE.fullmatch(payload_name) is None
    ):
        reject_invalid_state("payload names must be bounded basenames")


def _fsync_directory(directory: Path) -> None:
    """Flush directory metadata after an atomic rename."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write bytes with same-directory fsync and atomic replacement."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except OSError:
        logger.exception("atomic generation write failed for %s", path)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.exception("failed to clean temporary generation file %s", temporary)
        raise


@dataclass(frozen=True, slots=True)
class PreparedGenerationRecord:
    """Published generation metadata returned by the local store."""

    generation_id: str
    manifest_digest: str
    payload_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedGenerationLease:
    """In-process pin capability for one immutable generation."""

    generation_id: str
    nonce: str


class PreparedGenerationReader:
    """Read-only, hash-validating view of one pinned generation."""

    def __init__(
        self, store: PreparedGenerationStore, lease: PreparedGenerationLease
    ) -> None:
        """Create a reader for a pinned generation lease."""
        self._store = store
        self.lease = lease

    @property
    def generation_id(self) -> str:
        """Return the pinned generation identifier."""
        return self.lease.generation_id

    @property
    def manifest(self) -> Mapping[str, Any]:
        """Return a freshly validated immutable-generation manifest."""
        return self._store._read_manifest(self.lease)

    def read_payload(self, payload_name: str) -> bytes:
        """Read and hash-check one payload from the pinned generation."""
        return self._store._read_payload(self.lease, payload_name)


class PreparedGenerationStore:
    """Crash-safe local generation store for prepared geometry artifacts.

    Parameters
    ----------
    root : pathlib.Path or str
        Caller-owned ``prepared`` directory.  The directory is created with
        mode ``0700`` and all existing symlink components are rejected.
    max_payload_bytes : int, optional
        Hard per-payload limit used before publication and read-back.

    Notes
    -----
    The store is intentionally a local, in-process implementation for the
    first provider increment.  It does not claim cross-process lease recovery;
    callers must keep the owning process alive and close pins before exit.
    Persistent cross-process leases and GC remain a separate activation gate.

    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_payload_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        """Create and validate a caller-owned prepared generation root."""
        if max_payload_bytes <= 0:
            reject_invalid_state("max_payload_bytes must be positive")
        self.root = Path(root)
        _reject_symlink_components(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError:
            logger.exception("unable to secure prepared store root %s", self.root)
            raise
        self._validate_directory(self.root, "prepared store root")
        self.generations = self.root / "generations"
        self.generations.mkdir(mode=0o700, exist_ok=True)
        try:
            self.generations.chmod(0o700)
        except OSError:
            logger.exception("unable to secure prepared generations directory")
            raise
        self._validate_directory(self.generations, "prepared generations directory")
        self.current_path = self.root / "CURRENT"
        self.lock_path = self.root / ".lock"
        self.max_payload_bytes = max_payload_bytes
        self._lock = threading.RLock()
        self._pins: dict[str, dict[str, PreparedGenerationLease]] = {}
        self._closed: set[str] = set()

    @staticmethod
    def _validate_directory(path: Path, label: str) -> None:
        """Validate that ``path`` is a real directory owned by the caller."""
        if path.is_symlink() or not path.is_dir():
            reject_invalid_state(f"{label} must be a real directory")
        stat_result = path.stat()
        if stat_result.st_uid != os.getuid():
            reject_invalid_state(f"{label} is not owned by the current user")
        if stat_result.st_mode & 0o077:
            reject_invalid_state(f"{label} must not be group/world accessible")

    @staticmethod
    def _generation_path(generations: Path, generation_id: str) -> Path:
        """Return a validated generation directory path."""
        _require_generation_id(generation_id)
        path = generations / generation_id
        if path.is_symlink():
            reject_invalid_state(f"generation directory is a symlink: {path}")
        return path

    def stage(
        self,
        generation_id: str,
        *,
        identity: PreparedIdentity,
        metadata: Mapping[str, Any],
        payloads: Mapping[str, bytes],
    ) -> PreparedGenerationRecord:
        """Stage an immutable generation without changing ``CURRENT``.

        Parameters
        ----------
        generation_id : str
            Bounded lowercase generation identifier.
        identity : PreparedIdentity
            Canonical parent identity bound into the manifest.
        metadata : mapping
            JSON-compatible provenance metadata.
        payloads : mapping[str, bytes]
            Primitive byte payloads keyed by safe basenames.

        """
        _require_generation_id(generation_id)
        if not isinstance(payloads, Mapping) or not payloads:
            reject_invalid_state("a prepared generation must contain payloads")
        with self._lock:
            target = self._generation_path(self.generations, generation_id)
            if target.exists():
                reject_invalid_state(f"generation already exists: {generation_id}")
            staging = self.generations / (
                f".{generation_id}.{secrets.token_hex(8)}.staging"
            )
            staging.mkdir(mode=0o700)
            try:
                entries: dict[str, dict[str, int | str]] = {}
                for name, payload in payloads.items():
                    _require_payload_name(name)
                    if not isinstance(payload, bytes):
                        reject_invalid_state(f"payload {name} must be bytes")
                    if len(payload) > self.max_payload_bytes:
                        reject_invalid_state(
                            f"payload {name} exceeds the configured bound"
                        )
                    path = staging / name
                    _atomic_write(path, payload)
                    entries[name] = {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                manifest = {
                    "schema": _STORE_SCHEMA,
                    "provider_schema": PROVIDER_SCHEMA,
                    "generation_id": generation_id,
                    "identity": _identity_json(identity),
                    "metadata": dict(metadata),
                    "payloads": entries,
                }
                manifest_bytes = _canonical_json(manifest)
                if len(manifest_bytes) > self.max_payload_bytes:
                    reject_invalid_state(
                        "generation manifest exceeds the configured bound"
                    )
                _atomic_write(staging / "manifest.json", manifest_bytes)
                _fsync_directory(staging)
                staging.replace(target)
                _fsync_directory(self.generations)
            except BaseException:
                _quarantine_staging(staging)
                raise
        return PreparedGenerationRecord(
            generation_id=generation_id,
            manifest_digest=hashlib.sha256(manifest_bytes).hexdigest(),
            payload_names=tuple(sorted(entries)),
        )

    def publish(self, generation_id: str) -> PreparedGenerationRecord:
        """Atomically publish a staged generation through ``CURRENT``."""
        with self._lock:
            target = self._generation_path(self.generations, generation_id)
            if self.current_path.is_symlink():
                reject_invalid_state("prepared store CURRENT pointer is a symlink")
            lease = self._pins.get(generation_id)
            if lease:
                reject_invalid_state("a pinned generation cannot be republished")
            manifest = self._read_manifest_from_path(target)
            _atomic_write(self.current_path, f"{generation_id}\n".encode("ascii"))
            return PreparedGenerationRecord(
                generation_id=generation_id,
                manifest_digest=hashlib.sha256(
                    _canonical_json(manifest)
                ).hexdigest(),
                payload_names=tuple(sorted(manifest["payloads"])),
            )

    def pin(self, generation_id: str | None = None) -> PreparedGenerationLease:
        """Pin ``generation_id`` or the current generation before opening data."""
        with self._lock:
            chosen = generation_id or self._read_current_id()
            if chosen in self._closed:
                reject_invalid_state(f"generation is closed: {chosen}")
            target = self._generation_path(self.generations, chosen)
            self._read_manifest_from_path(target)
            lease = PreparedGenerationLease(chosen, secrets.token_hex(16))
            self._pins.setdefault(chosen, {})[lease.nonce] = lease
            return lease

    def attach(self, lease: PreparedGenerationLease) -> None:
        """Attach a serialized lease in a worker process.

        The owning process keeps the original pin alive while the worker
        reads the immutable generation.  Each process has its own in-memory
        reader registry, so attaching never republishes or mutates the
        generation; the provider validates the lease capability and root
        identity before calling this method.
        """
        with self._lock:
            _require_generation_id(lease.generation_id)
            if lease.generation_id in self._closed:
                reject_invalid_state(f"generation is closed: {lease.generation_id}")
            target = self._generation_path(self.generations, lease.generation_id)
            self._read_manifest_from_path(target)
            leases = self._pins.setdefault(lease.generation_id, {})
            if lease.nonce in leases:
                reject_invalid_state("generation lease nonce is already attached")
            leases[lease.nonce] = lease

    def open(self, lease: PreparedGenerationLease) -> PreparedGenerationReader:
        """Open a read-only reader only for a currently pinned lease."""
        with self._lock:
            if not self._is_pinned(lease):
                reject_invalid_state("generation lease is not pinned")
            self._read_manifest(lease)
            return PreparedGenerationReader(self, lease)

    def unpin(self, lease: PreparedGenerationLease) -> None:
        """Release one pin capability."""
        with self._lock:
            leases = self._pins.get(lease.generation_id)
            if leases is None or leases.pop(lease.nonce, None) is None:
                reject_invalid_state("unknown or already released generation lease")
            if not leases:
                self._pins.pop(lease.generation_id, None)

    def close(self, generation_id: str) -> None:
        """Close a generation after all pins have been released."""
        with self._lock:
            if self._pins.get(generation_id):
                reject_invalid_state("cannot close a generation while it is pinned")
            self._generation_path(self.generations, generation_id)
            self._closed.add(generation_id)

    def abort(self, generation_id: str) -> Path:
        """Quarantine an incomplete generation instead of consuming it."""
        with self._lock:
            if self._pins.get(generation_id):
                reject_invalid_state("cannot abort a pinned generation")
            target = self._generation_path(self.generations, generation_id)
            if not target.exists():
                reject_invalid_state(f"generation does not exist: {generation_id}")
            quarantined = self.generations / f".{generation_id}.aborted"
            if quarantined.exists():
                quarantined = self.generations / (
                    f".{generation_id}.aborted.{secrets.token_hex(4)}"
                )
            target.replace(quarantined)
            _fsync_directory(self.generations)
            return quarantined

    def _read_current_id(self) -> str:
        """Read and validate the bounded ``CURRENT`` pointer."""
        if self.current_path.is_symlink() or not self.current_path.is_file():
            reject_invalid_state("prepared store CURRENT pointer is missing or unsafe")
        value = self.current_path.read_text(encoding="ascii").strip()
        _require_generation_id(value)
        target = self._generation_path(self.generations, value)
        if not target.is_dir():
            reject_invalid_state("CURRENT points to a missing generation")
        return value

    def _is_pinned(self, lease: PreparedGenerationLease) -> bool:
        """Return whether a lease is known and currently active."""
        return lease.nonce in self._pins.get(lease.generation_id, {})

    def _read_manifest(self, lease: PreparedGenerationLease) -> Mapping[str, Any]:
        """Read and validate one manifest for a pinned lease."""
        if not self._is_pinned(lease):
            reject_invalid_state("generation lease is not pinned")
        return self._read_manifest_from_path(
            self._generation_path(self.generations, lease.generation_id)
        )

    def _read_manifest_from_path(self, generation: Path) -> Mapping[str, Any]:
        """Read and validate a generation manifest before any payload read."""
        if generation.is_symlink() or not generation.is_dir():
            reject_invalid_state("generation path is not a real directory")
        manifest_path = generation / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            reject_invalid_state("generation manifest is missing or unsafe")
        raw = manifest_path.read_bytes()
        if len(raw) > self.max_payload_bytes:
            reject_invalid_state("generation manifest exceeds the configured bound")
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            reject_invalid_state(f"generation manifest is invalid: {error}")
        if not isinstance(manifest, dict):
            reject_invalid_state("generation manifest must be an object")
        if manifest.get("schema") != _STORE_SCHEMA:
            reject_invalid_state("unsupported prepared generation store schema")
        if manifest.get("provider_schema") != PROVIDER_SCHEMA:
            reject_invalid_state("prepared generation provider schema mismatch")
        generation_id = manifest.get("generation_id")
        _require_generation_id(generation_id)
        if generation_id != generation.name:
            reject_invalid_state("generation manifest ID does not match its directory")
        payloads = manifest.get("payloads")
        if not isinstance(payloads, dict) or not payloads:
            reject_invalid_state("generation manifest has no payload table")
        for name, entry in payloads.items():
            _require_payload_name(name)
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("sha256"), str)
                or not isinstance(entry.get("size"), int)
                or entry["size"] < 0
            ):
                reject_invalid_state(f"payload manifest entry is invalid: {name}")
            if len(entry["sha256"]) != 64 or any(
                char not in "0123456789abcdef" for char in entry["sha256"]
            ):
                reject_invalid_state(f"payload digest is invalid: {name}")
        return manifest

    def _read_payload(self, lease: PreparedGenerationLease, payload_name: str) -> bytes:
        """Read one payload after manifest, size, and digest validation."""
        _require_payload_name(payload_name)
        manifest = self._read_manifest(lease)
        entry = manifest["payloads"].get(payload_name)
        if entry is None:
            reject_invalid_state(
                f"payload is not listed in the generation: {payload_name}"
            )
        expected_size = entry["size"]
        if expected_size > self.max_payload_bytes:
            reject_invalid_state(f"payload exceeds configured bound: {payload_name}")
        path = (
            self._generation_path(self.generations, lease.generation_id) / payload_name
        )
        if path.is_symlink() or not path.is_file():
            reject_invalid_state(f"payload is missing or unsafe: {payload_name}")
        payload = path.read_bytes()
        if len(payload) != expected_size:
            reject_invalid_state(f"payload size changed: {payload_name}")
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            reject_invalid_state(f"payload digest changed: {payload_name}")
        return payload


def _identity_json(identity: PreparedIdentity) -> dict[str, Any]:
    """Convert a prepared identity into canonical JSON-compatible fields."""
    return {
        "provider_schema": identity.provider_schema,
        "parent_generation_id": identity.parent_generation_id,
        "domain": identity.domain,
        "ordered_source_ids": identity.ordered_source_ids,
        "common_domain_digest": identity.common_domain_digest,
        "policy_digest": identity.policy_digest,
        "schema_code_backend_device_digest": identity.schema_code_backend_device_digest,
        "payload_manifest_digest": identity.payload_manifest_digest,
    }


def _quarantine_staging(staging: Path) -> None:
    """Move a failed staging directory out of the consumable namespace."""
    if not staging.exists():
        return
    quarantined = staging.with_name(f"{staging.name}.aborted")
    try:
        staging.replace(quarantined)
    except OSError:
        logger.exception("failed to quarantine incomplete generation %s", staging)


__all__ = [
    "PreparedGenerationLease",
    "PreparedGenerationReader",
    "PreparedGenerationRecord",
    "PreparedGenerationStore",
]
