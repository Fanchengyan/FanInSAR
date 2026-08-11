"""Local immutable-generation transactions for derived Stack artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager, suppress
from contextvars import ContextVar
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
_ACTIVE_ROOTS: ContextVar[tuple[tuple[str, int], ...]] = ContextVar(
    "artifact_transaction_active_roots",
    default=(),
)
_ConcretePath = type(Path())


@dataclass(slots=True)
class _PinnedDescriptorState:
    """Descriptors shared by one caller-facing pinned path tree."""

    generation_descriptor: int
    display_path: Path
    payload_descriptors: list[int]

    def __del__(self) -> None:
        """Release descriptors when the last pinned path is gone."""
        with suppress(OSError):
            os.close(self.generation_descriptor)
        for descriptor in self.payload_descriptors:
            with suppress(OSError):
                os.close(descriptor)


class _PinnedGenerationPath(_ConcretePath):
    """Filesystem path backed by a pinned generation directory descriptor."""

    __slots__ = (
        "_display_path",
        "_relative_parts",
        "_state",
    )

    def __new__(
        cls,
        _state: _PinnedDescriptorState,
        _display_path: Path,
        _relative_parts: tuple[str, ...] = (),
    ) -> Self:
        """Create a descriptor-backed path with stable display metadata."""
        return super().__new__(cls)

    def __init__(
        self,
        state: _PinnedDescriptorState,
        display_path: Path,
        relative_parts: tuple[str, ...] = (),
    ) -> None:
        """Initialize descriptor and caller-facing path representations."""
        super().__init__(display_path)
        self._display_path = display_path
        self._relative_parts = relative_parts
        self._state = state

    @classmethod
    def _create_child(
        cls,
        state: _PinnedDescriptorState,
        display_path: Path,
        relative_parts: tuple[str, ...],
    ) -> Self:
        """Create a child path without changing the pinned descriptor prefix."""
        return cls(
            state,
            display_path,
            relative_parts,
        )

    def __truediv__(self, key: str | os.PathLike[str]) -> Self:
        """Append the same component to descriptor and display paths."""
        component_path = Path(key)
        if component_path.is_absolute() or any(
            component in {"", ".", ".."} for component in component_path.parts
        ):
            reject_invalid_state("artifact generation relative path is unsafe")
        relative_parts = (*self._relative_parts, *component_path.parts)
        return type(self)._create_child(
            self._state,
            self._display_path / component_path,
            relative_parts,
        )

    def with_segments(self, *pathsegments: str | os.PathLike[str]) -> Self:
        """Propagate descriptor state through pathlib-generated child paths."""
        display_path = Path(*pathsegments)
        try:
            relative_parts = display_path.relative_to(
                self._state.display_path
            ).parts
        except ValueError:
            relative_parts = self._relative_parts
        return type(self)._create_child(
            self._state,
            display_path,
            relative_parts,
        )

    def __fspath__(self) -> str:
        """Return a descriptor path for a securely opened regular payload."""
        if not self._relative_parts:
            return os.fspath(self._display_path)
        parent_descriptor = os.dup(self._state.generation_descriptor)
        try:
            for component in self._relative_parts[:-1]:
                child = _open_directory_at(
                    parent_descriptor,
                    component,
                    label=str(self._display_path),
                )
                os.close(parent_descriptor)
                parent_descriptor = child
            name = self._relative_parts[-1]
            metadata = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            descriptor = os.open(
                name,
                os.O_RDONLY
                | os.O_NONBLOCK
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or metadata.st_nlink != 1
                or opened.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
            ):
                os.close(descriptor)
                reject_invalid_state(
                    f"artifact generation payload is unsafe: {self._display_path.name}"
                )
            self._state.payload_descriptors.append(descriptor)
            descriptor_path = f"/dev/fd/{descriptor}"
        except OSError as error:
            reject_invalid_state(
                f"artifact generation payload cannot be opened: {error}"
            )
        finally:
            os.close(parent_descriptor)
        return descriptor_path

    @property
    def parent(self) -> Path:
        """Return the stable caller-facing parent path."""
        return self._display_path.parent

    @property
    def name(self) -> str:
        """Return the stable caller-facing generation name."""
        return self._display_path.name

    def open(
        self,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> Any:
        """Open reads through a descriptor and explicit mutations by display path."""
        if any(flag in mode for flag in "wax+"):
            return self._display_path.open(
                mode,
                buffering,
                encoding,
                errors,
                newline,
            )
        return super().open(mode, buffering, encoding, errors, newline)

    def unlink(self, missing_ok: bool = False) -> None:
        """Permit explicit hostile-state test mutation through the display path."""
        self._display_path.unlink(missing_ok=missing_ok)


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


def _validate_directory_descriptor(
    descriptor: int,
    *,
    label: str,
    require_private: bool = True,
) -> os.stat_result:
    """Validate one already-open directory descriptor."""
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        reject_invalid_state(f"artifact directory is unsafe: {label}")
    if require_private and (
        metadata.st_uid != os.getuid()
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        reject_invalid_state(f"artifact directory is unsafe: {label}")
    return metadata


def _open_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    create: bool = False,
    label: str,
) -> int:
    """Open one directory component relative to a pinned parent."""
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            pass
        except OSError as error:
            reject_invalid_state(f"artifact directory cannot be created: {error}")
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        reject_invalid_state(f"artifact directory is unsafe: {label}: {error}")
    try:
        _validate_directory_descriptor(descriptor, label=label)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_root_descriptor(root: Path, *, create: bool) -> int:
    """Open a root through descriptor-relative, no-follow traversal."""
    absolute = root.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute.anchor, flags)
        _validate_directory_descriptor(
            descriptor,
            label=absolute.anchor,
            require_private=False,
        )
        for index, component in enumerate(absolute.parts[1:], start=1):
            is_root = index == len(absolute.parts) - 1
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            _validate_directory_descriptor(
                child,
                label=str(Path(*absolute.parts[: index + 1])),
                require_private=is_root,
            )
            os.close(descriptor)
            descriptor = child
        os.fchmod(descriptor, 0o700)
    except OSError as error:
        with suppress(UnboundLocalError):
            os.close(descriptor)
        reject_invalid_state(f"artifact transaction root is unsafe: {error}")
    return descriptor


def _open_transaction_root(root: Path) -> int:
    """Duplicate an active pinned root, or securely open an inactive root."""
    key = os.fspath(root.absolute())
    for active_key, active_descriptor in reversed(_ACTIVE_ROOTS.get()):
        if active_key == key:
            descriptor = os.dup(active_descriptor)
            _validate_directory_descriptor(descriptor, label=key)
            return descriptor
    return _open_root_descriptor(root, create=False)


def _atomic_control(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = _open_root_descriptor(path.parent, create=False)
    try:
        _atomic_control_at(descriptor, path.name, value)
    finally:
        os.close(descriptor)


def _atomic_control_at(
    directory_descriptor: int,
    name: str,
    value: Mapping[str, Any],
) -> None:
    """Atomically write one control frame relative to a pinned directory."""
    payload = canonical_json(value) + b"\n"
    if len(payload) > _CONTROL_LIMIT_BYTES:
        reject_invalid_state(f"artifact control frame exceeds limit: {name}")
    try:
        existing = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    except OSError as error:
        reject_invalid_state(f"artifact control destination is unsafe: {error}")
    else:
        if (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_nlink != 1
            or existing.st_uid != os.getuid()
            or existing.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            reject_invalid_state(f"artifact control destination is unsafe: {name}")
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory_descriptor)
        os.rename(
            temporary,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    except OSError as error:
        reject_invalid_state(f"artifact control publication failed: {error}")
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory_descriptor)


def _validate_generation_tree(directory_descriptor: int) -> None:
    """Fsync and validate a staged generation without following names."""
    try:
        names = os.listdir(directory_descriptor)
    except OSError as error:
        reject_invalid_state(f"artifact generation cannot be listed: {error}")
    for name in names:
        try:
            metadata = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            reject_invalid_state(
                f"artifact generation entry cannot be inspected: {error}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            child = _open_directory_at(
                directory_descriptor,
                name,
                label=f"generation/{name}",
            )
            try:
                _validate_generation_tree(child)
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            reject_invalid_state(
                f"artifact generation contains a non-regular entry: {name}"
            )
        if metadata.st_nlink != 1:
            reject_invalid_state(f"artifact generation contains a hardlink: {name}")
        if (
            metadata.st_uid != os.getuid()
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            reject_invalid_state(f"artifact generation entry is unsafe: {name}")
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | os.O_NONBLOCK
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
        except OSError as error:
            reject_invalid_state(f"artifact generation entry cannot be opened: {error}")
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
                or opened.st_nlink != 1
            ):
                reject_invalid_state(
                    f"artifact generation entry changed while opening: {name}"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.fsync(directory_descriptor)


def _remove_tree_at(parent_descriptor: int, name: str) -> None:
    """Remove one staging tree without following any directory entry."""
    try:
        metadata = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(metadata.st_mode):
        os.unlink(name, dir_fd=parent_descriptor)
        return
    directory_descriptor = _open_directory_at(
        parent_descriptor,
        name,
        label=f"staging/{name}",
    )
    try:
        for child_name in os.listdir(directory_descriptor):  # noqa: PTH208
            _remove_tree_at(directory_descriptor, child_name)
    finally:
        os.close(directory_descriptor)
    os.rmdir(name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


def _read_control(path: Path) -> dict[str, Any]:
    descriptor = _open_root_descriptor(path.parent, create=False)
    try:
        return _read_control_at(descriptor, path.name)
    finally:
        os.close(descriptor)


def _read_control_at(directory_descriptor: int, name: str) -> dict[str, Any]:
    """Read one control frame relative to a pinned directory."""
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or metadata.st_size > _CONTROL_LIMIT_BYTES
        ):
            reject_invalid_state(f"artifact control file is unsafe: {name}")
        payload = os.read(descriptor, _CONTROL_LIMIT_BYTES + 1)
        if len(payload) > _CONTROL_LIMIT_BYTES:
            reject_invalid_state(f"artifact control file is unsafe: {name}")
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        reject_invalid_state(f"artifact control file cannot be read: {error}")
    finally:
        with suppress(UnboundLocalError):
            os.close(descriptor)
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
def root_lock(root: Path) -> Iterator[int]:
    """Serialize control-plane mutations and reader pin creation."""
    import fcntl

    root_descriptor = _open_root_descriptor(root, create=False)
    try:
        descriptor = os.open(
            ".root.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_descriptor,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            reject_invalid_state("artifact root lock is unsafe")
    except Exception:
        os.close(root_descriptor)
        raise
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        token = _ACTIVE_ROOTS.set(
            (*_ACTIVE_ROOTS.get(), (os.fspath(root.absolute()), root_descriptor))
        )
        try:
            yield root_descriptor
        finally:
            _ACTIVE_ROOTS.reset(token)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        os.close(root_descriptor)


@dataclass(slots=True)
class GenerationLease:
    """Durable reader pin preventing reclamation of one generation."""

    path: Path
    generation_id: str
    _root_descriptor: int | None = None
    _leases_descriptor: int | None = None
    _generation_descriptor: int | None = None
    _closed: bool = False

    def close(self) -> None:
        """Release this reader pin."""
        if self._closed:
            return
        try:
            if (
                self._root_descriptor is not None
                and self._leases_descriptor is not None
            ):
                import fcntl

                lock_descriptor = os.open(
                    ".root.lock",
                    os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=self._root_descriptor,
                )
                try:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
                    with suppress(FileNotFoundError):
                        os.unlink(self.path.name, dir_fd=self._leases_descriptor)
                    os.fsync(self._leases_descriptor)
                finally:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                    os.close(lock_descriptor)
            else:
                with root_lock(self.path.parents[1]):
                    self.path.unlink(missing_ok=True)
                    _fsync_directory(self.path.parent)
        finally:
            for descriptor in (
                self._generation_descriptor,
                self._leases_descriptor,
                self._root_descriptor,
            ):
                if descriptor is not None:
                    os.close(descriptor)
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
    descriptor = _open_root_descriptor(absolute, create=True)
    os.close(descriptor)
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
    with root_lock(root) as root_descriptor:
        generations, staging_root, leases, _ = _namespace_paths(root, namespace)
        namespace_descriptors = [
            _open_directory_at(
                root_descriptor,
                path.name,
                create=True,
                label=str(path),
            )
            for path in (generations, staging_root, leases)
        ]
        generations_descriptor, staging_root_descriptor, leases_descriptor = (
            namespace_descriptors
        )
        os.close(generations_descriptor)
        os.close(leases_descriptor)
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
        try:
            os.mkdir(generation_id, mode=0o700, dir_fd=staging_root_descriptor)
        except OSError as error:
            os.close(staging_root_descriptor)
            reject_invalid_state(
                f"artifact staging directory cannot be created: {error}"
            )
        try:
            yield generation_id, staging_path
        finally:
            try:
                _remove_tree_at(staging_root_descriptor, generation_id)
            finally:
                os.close(staging_root_descriptor)


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
    generations, staging_root, _, current = _namespace_paths(root, namespace)
    generation_id = _validate_component(generation_id, "generation id")
    final_path = generations / generation_id
    if (
        staging_path.name != generation_id
        or staging_path.parent.name != staging_root.name
    ):
        reject_invalid_state("artifact staging path does not match its generation")
    root_descriptor = _open_transaction_root(root)
    descriptors: list[int] = [root_descriptor]
    try:
        generations_descriptor = _open_directory_at(
            root_descriptor,
            generations.name,
            label=str(generations),
        )
        descriptors.append(generations_descriptor)
        staging_root_descriptor = _open_directory_at(
            root_descriptor,
            staging_root.name,
            label=str(staging_root),
        )
        descriptors.append(staging_root_descriptor)
        try:
            os.stat(
                generation_id,
                dir_fd=generations_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            reject_invalid_state("artifact generation id already exists")
        staging_descriptor = _open_directory_at(
            staging_root_descriptor,
            generation_id,
            label=str(staging_path),
        )
        descriptors.append(staging_descriptor)
        _validate_generation_tree(staging_descriptor)
        os.rename(
            generation_id,
            generation_id,
            src_dir_fd=staging_root_descriptor,
            dst_dir_fd=generations_descriptor,
        )
        os.fsync(generations_descriptor)
        if compatibility_manifest is not None:
            name = (
                "manifest.json"
                if namespace == "ifg"
                else f"{namespace}_manifest.json"
            )
            _atomic_control_at(root_descriptor, name, compatibility_manifest)
        unsigned = {
            "schema": "faninsar_artifact_current_v1",
            "namespace": namespace,
            "generation_id": generation_id,
            "manifest_digest": manifest_digest,
        }
        pointer = {
            **unsigned,
            "control_digest": sha256_bytes(canonical_json(unsigned)),
        }
        _atomic_control_at(root_descriptor, current.name, pointer)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return final_path


def _read_current_at(
    root_descriptor: int,
    root: Path,
    namespace: str,
) -> tuple[str, str]:
    """Read CURRENT relative to a pinned transaction root."""
    _, _, _, current = _namespace_paths(root, namespace)
    pointer = _read_control_at(root_descriptor, current.name)
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


def _read_current(root: Path, namespace: str) -> tuple[str, str]:
    """Read CURRENT after securely opening its transaction root."""
    descriptor = _open_root_descriptor(root, create=False)
    try:
        return _read_current_at(descriptor, root, namespace)
    finally:
        os.close(descriptor)


def open_current_generation(root: Path, namespace: str) -> OpenGeneration:
    """Select and durably pin exactly one current immutable generation."""
    root = initialize_root(root)
    generation_descriptor: int | None = None
    leases_descriptor: int | None = None
    lease_root_descriptor: int | None = None
    with root_lock(root) as root_descriptor:
        generations, _, leases, _ = _namespace_paths(root, namespace)
        generation_id, manifest_digest = _read_current_at(
            root_descriptor,
            root,
            namespace,
        )
        generation = generations / generation_id
        generations_descriptor = _open_directory_at(
            root_descriptor,
            generations.name,
            label=str(generations),
        )
        leases_descriptor = _open_directory_at(
            root_descriptor,
            leases.name,
            label=str(leases),
        )
        try:
            generation_descriptor = _open_directory_at(
                generations_descriptor,
                generation_id,
                label=str(generation),
            )
            _validate_generation_tree(generation_descriptor)
        finally:
            os.close(generations_descriptor)
        lease_id = uuid.uuid4().hex
        lease_path = leases / f"{generation_id}-{lease_id}.json"
        _atomic_control_at(
            leases_descriptor,
            lease_path.name,
            {
                "schema": "faninsar_reader_lease_v1",
                "generation_id": generation_id,
                "pid": os.getpid(),
                "nonce": lease_id,
            },
        )
        lease_root_descriptor = os.dup(root_descriptor)
    if (
        generation_descriptor is None
        or leases_descriptor is None
        or lease_root_descriptor is None
    ):
        reject_invalid_state("artifact generation descriptors were not pinned")
    pinned_state = _PinnedDescriptorState(
        os.dup(generation_descriptor),
        generation,
        [],
    )
    pinned_path = _PinnedGenerationPath(
        pinned_state,
        generation,
    )
    return OpenGeneration(
        root=root,
        namespace=namespace,
        generation_id=generation_id,
        path=pinned_path,
        manifest_digest=manifest_digest,
        lease=GenerationLease(
            lease_path,
            generation_id,
            _root_descriptor=lease_root_descriptor,
            _leases_descriptor=leases_descriptor,
            _generation_descriptor=generation_descriptor,
        ),
    )


def collect_generations(
    root: str | Path,
    namespace: Literal["ifg", "unwrap", "timeseries"],
) -> tuple[str, ...]:
    """Remove non-current generations not protected by a reader lease."""
    root_path = initialize_root(Path(root))
    removed: list[str] = []
    with root_lock(root_path) as root_descriptor:
        generations, _, leases, _ = _namespace_paths(root_path, namespace)
        current_id, _ = _read_current_at(root_descriptor, root_path, namespace)
        generations_descriptor = _open_directory_at(
            root_descriptor,
            generations.name,
            label=str(generations),
        )
        leases_descriptor = _open_directory_at(
            root_descriptor,
            leases.name,
            label=str(leases),
        )
        try:
            leased_ids: set[str] = set()
            for lease_name in os.listdir(leases_descriptor):  # noqa: PTH208
                metadata = os.stat(
                    lease_name,
                    dir_fd=leases_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not lease_name.endswith(".json")
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_uid != os.getuid()
                ):
                    reject_invalid_state("artifact reader lease is unsafe")
                leased_ids.add(lease_name.split("-", 1)[0])
            for generation_name in os.listdir(  # noqa: PTH208
                generations_descriptor
            ):
                _validate_component(generation_name, "generation id")
                if generation_name in {current_id, *leased_ids}:
                    continue
                generation_descriptor = _open_directory_at(
                    generations_descriptor,
                    generation_name,
                    label=f"{generations}/{generation_name}",
                )
                try:
                    _validate_generation_tree(generation_descriptor)
                finally:
                    os.close(generation_descriptor)
                _remove_tree_at(generations_descriptor, generation_name)
                removed.append(generation_name)
        finally:
            os.close(leases_descriptor)
            os.close(generations_descriptor)
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
