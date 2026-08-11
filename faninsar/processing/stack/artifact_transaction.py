"""Local immutable-generation transactions for derived Stack artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

_CONTROL_LIMIT_BYTES = 4096
_DEFAULT_MAX_BYTES = 64 * 1024**3
_DEFAULT_RESERVE_BYTES = 256 * 1024**2
_MAX_FILES = 4096
_MAX_DIMENSION = 2**31 - 1


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Encode a canonical JSON control frame."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"artifact control frame is not canonical JSON: {error}")


def sha256_bytes(payload: bytes) -> str:
    """Return the canonical SHA-256 digest of *payload*."""
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_control(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(value) + b"\n"
    if len(payload) > _CONTROL_LIMIT_BYTES:
        reject_invalid_state(f"artifact control frame exceeds limit: {path.name}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except OSError as error:
        reject_invalid_state(f"artifact control publication failed: {error}")
    finally:
        temporary.unlink(missing_ok=True)


def _read_control(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > _CONTROL_LIMIT_BYTES
            or path.is_symlink()
        ):
            reject_invalid_state(f"artifact control file is unsafe: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        reject_invalid_state(f"artifact control file cannot be read: {error}")
    if not isinstance(value, dict):
        reject_invalid_state("artifact control frame must be a JSON object")
    return value


def _validate_component(value: str, field: str) -> str:
    if (
        not value
        or len(value) > 128
        or value in {".", ".."}
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in value
        )
    ):
        reject_invalid_state(f"artifact {field} is unsafe")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactResourceLimits:
    """Hard preflight limits for one derived-artifact publication.

    Parameters
    ----------
    max_final_bytes : int
        Maximum payload bytes in one immutable generation.
    max_temporary_bytes : int
        Maximum simultaneous staging bytes.
    min_free_bytes : int
        Disk space that must remain free after reservation.
    max_files : int
        Maximum files in the generation.

    """

    max_final_bytes: int = _DEFAULT_MAX_BYTES
    max_temporary_bytes: int = _DEFAULT_MAX_BYTES
    min_free_bytes: int = _DEFAULT_RESERVE_BYTES
    max_files: int = _MAX_FILES


def preflight_resources(
    root: Path,
    *,
    final_bytes: int,
    temporary_bytes: int,
    file_count: int,
    dimensions: tuple[int, ...] = (),
    limits: ArtifactResourceLimits | None = None,
) -> None:
    """Reject unsafe or unreservable publication resource declarations."""
    resolved = limits or ArtifactResourceLimits()
    values = (final_bytes, temporary_bytes, file_count, *dimensions)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        reject_invalid_state("artifact resource declaration is invalid")
    if any(value > _MAX_DIMENSION for value in dimensions):
        reject_invalid_state("artifact dimension exceeds the hard limit")
    if (
        final_bytes > resolved.max_final_bytes
        or temporary_bytes > resolved.max_temporary_bytes
        or file_count > min(resolved.max_files, _MAX_FILES)
    ):
        reject_invalid_state("artifact resource quota exceeded")
    try:
        free_bytes = shutil.disk_usage(root).free
    except OSError as error:
        reject_invalid_state(f"artifact free space cannot be determined: {error}")
    required = temporary_bytes + resolved.min_free_bytes
    if required > free_bytes:
        reject_invalid_state("artifact disk reservation cannot be satisfied")


@contextmanager
def root_lock(root: Path) -> Iterator[None]:
    """Serialize control-plane mutations and reader pin creation."""
    import fcntl

    lock_path = root / ".root.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(slots=True)
class GenerationLease:
    """Durable reader pin preventing reclamation of one generation."""

    path: Path
    generation_id: str
    _closed: bool = False

    def close(self) -> None:
        """Release this reader pin."""
        if self._closed:
            return
        try:
            with root_lock(self.path.parents[1]):
                self.path.unlink(missing_ok=True)
                _fsync_directory(self.path.parent)
        finally:
            self._closed = True

    def __enter__(self) -> Self:
        """Return this lease for use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the pin on context-manager exit."""
        self.close()

    def __del__(self) -> None:
        """Best-effort release for compatibility callers without ``close``."""
        with suppress(Exception):
            self.close()


@dataclass(frozen=True, slots=True)
class OpenGeneration:
    """One explicitly selected immutable generation and its reader pin."""

    root: Path
    namespace: str
    generation_id: str
    path: Path
    manifest_digest: str
    lease: GenerationLease


def initialize_root(root: Path) -> Path:
    """Create or validate a local transaction root without following links."""
    absolute = root.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.exists() and current.is_symlink():
            reject_invalid_state(f"artifact root contains a symbolic link: {current}")
    try:
        absolute.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as error:
        reject_invalid_state(f"artifact transaction root cannot be created: {error}")
    if not absolute.is_dir() or absolute.is_symlink():
        reject_invalid_state("artifact transaction root is unsafe")
    if absolute.stat().st_mode & stat.S_IWOTH:
        reject_invalid_state("artifact transaction root is world-writable")
    return absolute


def _namespace_paths(root: Path, namespace: str) -> tuple[Path, Path, Path, Path]:
    safe_namespace = _validate_component(namespace, "namespace")
    generations = root / f".{safe_namespace}_generations"
    staging = root / f".{safe_namespace}_staging"
    leases = root / f".{safe_namespace}_leases"
    current_name = "CURRENT" if namespace == "ifg" else f"{namespace.upper()}_CURRENT"
    current = root / current_name
    return generations, staging, leases, current


@contextmanager
def stage_generation(
    root: Path,
    namespace: str,
    *,
    final_bytes: int,
    temporary_bytes: int,
    file_count: int,
    dimensions: tuple[int, ...] = (),
    limits: ArtifactResourceLimits | None = None,
) -> Iterator[tuple[str, Path]]:
    """Reserve and stage one unique generation while holding the root lock."""
    root = initialize_root(root)
    with root_lock(root):
        generations, staging_root, leases, _ = _namespace_paths(root, namespace)
        for path in (generations, staging_root, leases):
            path.mkdir(mode=0o700, exist_ok=True)
        preflight_resources(
            root,
            final_bytes=final_bytes,
            temporary_bytes=temporary_bytes,
            file_count=file_count,
            dimensions=dimensions,
            limits=limits,
        )
        generation_id = uuid.uuid4().hex
        staging_path = staging_root / generation_id
        staging_path.mkdir(mode=0o700)
        try:
            yield generation_id, staging_path
        finally:
            if staging_path.exists():
                shutil.rmtree(staging_path)


def commit_generation(
    root: Path,
    namespace: str,
    generation_id: str,
    staging_path: Path,
    *,
    manifest_digest: str,
    compatibility_manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Commit a staged directory and atomically select it as ``CURRENT``.

    The caller must hold :func:`root_lock` via :func:`stage_generation`.
    """
    generations, _, _, current = _namespace_paths(root, namespace)
    generation_id = _validate_component(generation_id, "generation id")
    final_path = generations / generation_id
    if final_path.exists():
        reject_invalid_state("artifact generation id already exists")
    _fsync_directory(staging_path)
    staging_path.replace(final_path)
    _fsync_directory(generations)
    if compatibility_manifest is not None:
        name = "manifest.json" if namespace == "ifg" else f"{namespace}_manifest.json"
        _atomic_control(root / name, compatibility_manifest)
    unsigned = {
        "schema": "faninsar_artifact_current_v1",
        "namespace": namespace,
        "generation_id": generation_id,
        "manifest_digest": manifest_digest,
    }
    pointer = {**unsigned, "control_digest": sha256_bytes(canonical_json(unsigned))}
    _atomic_control(current, pointer)
    return final_path


def _read_current(root: Path, namespace: str) -> tuple[str, str]:
    _, _, _, current = _namespace_paths(root, namespace)
    pointer = _read_control(current)
    expected = pointer.get("control_digest")
    unsigned = dict(pointer)
    unsigned.pop("control_digest", None)
    if (
        pointer.get("schema") != "faninsar_artifact_current_v1"
        or pointer.get("namespace") != namespace
        or expected != sha256_bytes(canonical_json(unsigned))
    ):
        reject_invalid_state("artifact CURRENT pointer is invalid or tampered")
    generation_id = pointer.get("generation_id")
    manifest_digest = pointer.get("manifest_digest")
    if not isinstance(generation_id, str) or not isinstance(manifest_digest, str):
        reject_invalid_state("artifact CURRENT pointer fields are invalid")
    return _validate_component(generation_id, "generation id"), manifest_digest


def open_current_generation(root: Path, namespace: str) -> OpenGeneration:
    """Select and durably pin exactly one current immutable generation."""
    root = initialize_root(root)
    with root_lock(root):
        generations, _, leases, _ = _namespace_paths(root, namespace)
        generation_id, manifest_digest = _read_current(root, namespace)
        generation = generations / generation_id
        if not generation.is_dir() or generation.is_symlink():
            reject_invalid_state(
                "artifact CURRENT names a missing or unsafe generation"
            )
        lease_id = uuid.uuid4().hex
        lease_path = leases / f"{generation_id}-{lease_id}.json"
        _atomic_control(
            lease_path,
            {
                "schema": "faninsar_reader_lease_v1",
                "generation_id": generation_id,
                "pid": os.getpid(),
                "nonce": lease_id,
            },
        )
    return OpenGeneration(
        root=root,
        namespace=namespace,
        generation_id=generation_id,
        path=generation,
        manifest_digest=manifest_digest,
        lease=GenerationLease(lease_path, generation_id),
    )


def collect_generations(
    root: str | Path,
    namespace: Literal["ifg", "unwrap", "timeseries"],
) -> tuple[str, ...]:
    """Remove non-current generations not protected by a reader lease."""
    root_path = initialize_root(Path(root))
    removed: list[str] = []
    with root_lock(root_path):
        generations, _, leases, _ = _namespace_paths(root_path, namespace)
        current_id, _ = _read_current(root_path, namespace)
        leased_ids = {
            path.name.split("-", 1)[0]
            for path in leases.glob("*.json")
            if path.is_file() and not path.is_symlink()
        }
        for generation in generations.iterdir():
            if (
                generation.is_dir()
                and not generation.is_symlink()
                and generation.name != current_id
                and generation.name not in leased_ids
            ):
                shutil.rmtree(generation)
                removed.append(generation.name)
        _fsync_directory(generations)
    return tuple(sorted(removed))


__all__ = [
    "ArtifactResourceLimits",
    "GenerationLease",
    "OpenGeneration",
    "canonical_json",
    "collect_generations",
    "commit_generation",
    "initialize_root",
    "open_current_generation",
    "preflight_resources",
    "root_lock",
    "sha256_bytes",
    "stage_generation",
]
