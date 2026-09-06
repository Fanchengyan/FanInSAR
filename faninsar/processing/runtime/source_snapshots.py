"""Immutable local source snapshots for provider-bound processing inputs.

The snapshot boundary is intentionally independent from Sentinel-1 parsing.
Callers copy a local SAFE, orbit, or DEM source once, hash the bytes that were
actually copied, and pass the returned snapshot path to later readers.  A
later consumer therefore never hashes one path and reopens a potentially
different source path.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator

_CHUNK_BYTES = 1024 * 1024
_MANIFEST_SCHEMA = "source_snapshot.v1"


def _fail(message: str) -> None:
    """Log and reject an unsafe source snapshot operation."""
    logger.error("source snapshot rejected: %s", message)
    reject_invalid_state(message)


def _canonical_json(value: object) -> bytes:
    """Serialize snapshot metadata using one deterministic encoding."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail(f"snapshot metadata is not canonicalizable: {error}")


def _digest_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of a byte payload."""
    return hashlib.sha256(payload).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    """Reject symlinked ancestors and the source leaf."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            _fail(f"source path contains a symlink component: {current}")


def _require_safe_root(root: Path) -> None:
    """Create and validate a caller-owned private snapshot root."""
    _reject_symlink_components(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        root.chmod(0o700)
    except OSError as error:
        _fail(f"cannot secure snapshot root {root}: {error}")
    result = root.stat()
    if result.st_uid != os.getuid() or result.st_mode & 0o077:
        _fail("snapshot root must be owned by the caller and mode 0700")


def _open_regular(path: Path) -> int:
    """Open one regular source file without following the leaf symlink."""
    _reject_symlink_components(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        result = os.fstat(descriptor)
    except OSError as error:
        _fail(f"cannot open source file {path}: {error}")
    if not stat.S_ISREG(result.st_mode):
        os.close(descriptor)
        _fail(f"source is not a regular file: {path}")
    return descriptor


def _iter_files(root: Path) -> Iterator[tuple[Path, str]]:
    """Yield regular files and POSIX relative names from a source directory."""
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as error:
        _fail(f"cannot enumerate source directory {root}: {error}")
    for entry in entries:
        if entry.is_symlink():
            _fail(f"source directory contains a symlink: {entry}")
        relative = entry.relative_to(root).as_posix()
        if entry.is_dir():
            yield from (
                (path, f"{relative}/{name}") for path, name in _iter_files(entry)
            )
        elif entry.is_file():
            yield entry, relative
        else:
            _fail(f"source entry is not a regular file or directory: {entry}")


def _copy_file(
    source: Path,
    destination: Path,
    *,
    remaining_bytes: int,
) -> tuple[str, int]:
    """Copy one source file through an FD while hashing the copied bytes."""
    descriptor = _open_regular(source)
    digest = hashlib.sha256()
    total = 0
    try:
        with (
            os.fdopen(descriptor, "rb", closefd=True) as source_stream,
            destination.open("xb") as target,
        ):
            while True:
                chunk = source_stream.read(_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > remaining_bytes:
                    _fail("source snapshot exceeds the configured byte limit")
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    except OSError as error:
        _fail(f"source snapshot copy failed for {source}: {error}")
    destination.chmod(0o400)
    return digest.hexdigest(), total


@dataclass(frozen=True, slots=True)
class SourceSnapshotEntry:
    """One immutable source file entry."""

    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ImmutableSourceSnapshot:
    """Content-addressed local source snapshot and validated read handle."""

    source_id: str
    path: Path
    kind: Literal["file", "directory"]
    manifest_digest: str
    snapshot_device: int
    snapshot_inode: int
    entries: tuple[SourceSnapshotEntry, ...]
    total_bytes: int

    def __post_init__(self) -> None:
        """Validate the immutable snapshot metadata."""
        if len(self.manifest_digest) != 64 or any(
            char not in "0123456789abcdef" for char in self.manifest_digest
        ):
            _fail("snapshot manifest digest must be a lowercase SHA-256 digest")
        if self.kind not in ("file", "directory"):
            _fail("snapshot kind must be file or directory")
        if self.total_bytes < 0 or not self.entries:
            _fail("snapshot must contain at least one non-empty entry table")

    @property
    def entry_paths(self) -> tuple[str, ...]:
        """Return deterministic relative source entry names."""
        return tuple(entry.relative_path for entry in self.entries)

    def read_bytes(self, relative_path: str | None = None) -> bytes:
        """Read one snapshot entry and verify its digest before returning it.

        Parameters
        ----------
        relative_path : str, optional
            Relative entry name for a directory snapshot.  File snapshots
            accept ``None`` or their sole entry name.

        Returns
        -------
        bytes
            Exact bytes from the immutable snapshot.

        """
        if self.kind == "file":
            if relative_path not in (None, self.entries[0].relative_path):
                _fail("file snapshot accepts only its sole entry")
            target = self.path
            expected = self.entries[0]
        else:
            if not isinstance(relative_path, str) or not relative_path:
                _fail("directory snapshot reads require a relative entry name")
            candidate = Path(relative_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                _fail("snapshot entry must remain within the snapshot root")
            target = self.path / candidate
            expected = next(
                (
                    entry
                    for entry in self.entries
                    if entry.relative_path == relative_path
                ),
                None,
            )
            if expected is None:
                _fail(f"snapshot entry is not listed in the manifest: {relative_path}")
        descriptor = _open_regular(target)
        try:
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                payload = stream.read(expected.size)
                if len(payload) != expected.size or stream.read(1):
                    _fail(f"snapshot entry size changed: {expected.relative_path}")
        except OSError as error:
            _fail(f"snapshot entry read failed: {error}")
        if _digest_bytes(payload) != expected.sha256:
            _fail(f"snapshot entry digest changed: {expected.relative_path}")
        return payload


def snapshot_local_source(
    source: str | Path,
    snapshot_root: str | Path,
    *,
    max_files: int = 100_000,
    max_bytes: int = 8 * 1024**4,
) -> ImmutableSourceSnapshot:
    """Create one private, content-addressed snapshot of a local source.

    Parameters
    ----------
    source : str or pathlib.Path
        Existing local regular file or directory.  Symlinks and non-regular
        entries are rejected; remote URLs are not accepted by this V1 seam.
    snapshot_root : str or pathlib.Path
        Caller-owned private directory for immutable snapshots.
    max_files : int, optional
        Maximum number of files copied from a directory source.
    max_bytes : int, optional
        Maximum decoded bytes copied across the complete source.

    Returns
    -------
    ImmutableSourceSnapshot
        Snapshot path, source manifest digest, and validated entry table.

    """
    source_path = Path(source)
    if not source_path.is_absolute() and "://" in str(source):
        _fail("remote source URLs are not accepted by local snapshot V1")
    _reject_symlink_components(source_path)
    if not source_path.exists():
        _fail(f"source does not exist: {source_path}")
    if not isinstance(max_files, int) or max_files <= 0:
        _fail("max_files must be positive")
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        _fail("max_bytes must be positive")
    _require_safe_root(Path(snapshot_root))
    root = Path(snapshot_root)
    kind: Literal["file", "directory"]
    if source_path.is_dir():
        kind = "directory"
        source_entries = list(_iter_files(source_path))
        if not source_entries:
            _fail("source directory contains no regular files")
        if len(source_entries) > max_files:
            _fail("source directory exceeds the configured file limit")
        staging = root / f".{secrets.token_hex(16)}.staging"
        staging.mkdir(mode=0o700)
        staging_source = staging / source_path.name
        staging_source.mkdir(mode=0o700)
        entries: list[SourceSnapshotEntry] = []
        total_bytes = 0
        try:
            for source_file, relative in source_entries:
                destination = staging_source / relative
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                digest, size = _copy_file(
                    source_file,
                    destination,
                    remaining_bytes=max_bytes - total_bytes,
                )
                total_bytes += size
                entries.append(SourceSnapshotEntry(relative, size, digest))
        except BaseException:
            _remove_staging(staging)
            raise
    else:
        kind = "file"
        if not source_path.is_file():
            _fail("source must be a regular file or directory")
        staging = root / f".{secrets.token_hex(16)}.staging"
        staging.mkdir(mode=0o700)
        staging_source = staging / source_path.name
        entries = []
        total_bytes = 0
        try:
            digest, size = _copy_file(
                source_path,
                staging_source,
                remaining_bytes=max_bytes,
            )
            total_bytes = size
            entries.append(SourceSnapshotEntry(source_path.name, size, digest))
        except BaseException:
            _remove_staging(staging)
            raise

    manifest_value = {
        "schema": _MANIFEST_SCHEMA,
        "kind": kind,
        "source_name": source_path.name,
        "entries": [
            {
                "relative_path": entry.relative_path,
                "size": entry.size,
                "sha256": entry.sha256,
            }
            for entry in entries
        ],
        "total_bytes": total_bytes,
    }
    manifest_bytes = _canonical_json(manifest_value)
    manifest_digest = _digest_bytes(manifest_bytes)
    final_path = root / manifest_digest
    manifest_path = final_path / "manifest.json"
    if final_path.exists():
        if not final_path.is_dir() or not manifest_path.exists():
            _remove_staging(staging)
            _fail("incomplete source snapshot already occupies its content ID")
        _remove_staging(staging)
    else:
        staging.replace(final_path)
        _write_manifest(manifest_path, manifest_bytes)
    _make_tree_read_only(final_path)
    snapshot_source = final_path / source_path.name
    snapshot_stat = snapshot_source.stat()
    return ImmutableSourceSnapshot(
        source_id=manifest_digest,
        path=snapshot_source,
        kind=kind,
        manifest_digest=manifest_digest,
        snapshot_device=snapshot_stat.st_dev,
        snapshot_inode=snapshot_stat.st_ino,
        entries=tuple(entries),
        total_bytes=total_bytes,
    )


def _write_manifest(path: Path, payload: bytes) -> None:
    """Write one private manifest atomically."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        path.chmod(0o400)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        _remove_staging(temporary)
        _fail(f"snapshot manifest publication failed: {error}")


def _make_tree_read_only(path: Path) -> None:
    """Remove write permissions from a copied directory tree."""
    for current, directories, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        current_path.chmod(0o500)
        for directory in directories:
            (current_path / directory).chmod(0o500)
        for file_name in files:
            (current_path / file_name).chmod(0o400)


def _remove_staging(path: Path) -> None:
    """Remove a private incomplete snapshot staging path."""
    if not path.exists():
        return
    if path.is_dir():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        path.rmdir()
    else:
        path.unlink()


__all__ = [
    "ImmutableSourceSnapshot",
    "SourceSnapshotEntry",
    "snapshot_local_source",
]
