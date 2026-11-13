"""Strict original-source integrity for Stack inputs (PROPOSAL-0088 section 8).

Fast Stack execution trusts stable local inputs and never hashes payloads
(PROPOSAL-0082). Strict execution is the narrow opt-in exception: a source can
be admitted only when a trusted baseline was established for that exact source
incarnation, and verification then rereads the original SLC/RSLC/SAFE bytes.

This module does not introduce a second integrity framework. Single-file
sources use the existing :func:`faninsar.stack.integrity.establish_integrity`
and :func:`faninsar.stack.integrity.verify_integrity` contracts directly. SAFE
directories use the same baseline record, atomic publication helpers and
verdict type, but hash only the declared scientific tree (``manifest.safe``,
``annotation/``, ``measurement/``, ``calibration/``) so transfer manifests,
reports and caches can never be certified as science inputs.

A baseline records bytes, not scientific truth, and is keyed by the cheap
source identity (resolved path, kind, size and ``mtime_ns``). It is stored
workspace-locally under ``<work_dir>/integrity/sources``. Without a baseline
strict verification reports ``integrity_unverifiable``; current bytes are never
hashed into a synthetic baseline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.stack.integrity import (
    BASELINE_SCHEMA,
    IntegrityBaseline,
    IntegrityStatus,
    IntegrityVerdict,
    _baseline_path,
    _fingerprint,
    _is_safe_relative,
    _load_baseline,
    _resolve_root,
    _write_baseline_atomic,
    establish_integrity,
    verify_integrity,
)

logger = setup_logger(__name__)

SOURCE_BASELINE_SCHEMA = BASELINE_SCHEMA
"""Source baselines use the existing strict integrity record schema."""

SourceIntegrityStatus = IntegrityStatus
"""Outcome of one :func:`verify_source_integrity` call."""

_SAFE_SCIENTIFIC_DIRECTORIES = frozenset({"annotation", "measurement", "calibration"})
_SIDECAR_MARKERS = ("faninsar.remote.", ".part-state")
"""Transport manifests and partial-transfer state are never scientific inputs."""

__all__ = [
    "SOURCE_BASELINE_SCHEMA",
    "SourceIntegrityStatus",
    "establish_source_integrity",
    "source_identity",
    "verify_source_integrity",
]


def _resolve_source(source: str | Path) -> Path:
    """Return one existing source file/directory without following a root link."""
    path = Path(source)
    if path.is_symlink():
        reject_invalid_state("strict source must not be a symbolic link")
    if not path.exists():
        reject_invalid_state(f"strict source is missing: {path}")
    if not (path.is_dir() or path.is_file()):
        reject_invalid_state(f"strict source is not a file or directory: {path}")
    return path.resolve()


def source_identity(source: str | Path) -> str:
    """Return the cheap identity of one source incarnation.

    Only the resolved path, file/directory kind, size and modification time
    are read. No directory is walked and no payload is hashed, so this is safe
    for the default fast path.
    """
    path = _resolve_source(source)
    stat = path.stat()
    document = {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_sidecar(relative: PurePosixPath) -> bool:
    """Return whether a relative member is transport/report state."""
    text = relative.as_posix()
    return any(marker in text for marker in _SIDECAR_MARKERS)


def _is_scientific_member(relative: PurePosixPath) -> bool:
    """Return whether a SAFE member belongs to the declared science tree."""
    if relative.name == "manifest.safe":
        return True
    return any(part in _SAFE_SCIENTIFIC_DIRECTORIES for part in relative.parts[:-1])


def _scientific_files(source: Path) -> dict[str, Path]:
    """Return declared SAFE scientific members, rejecting unsafe links.

    A directory with no declared scientific member yields an empty map so the
    caller can reject it instead of producing a baseline that verifies
    nothing. Sidecar transfer manifests, reports and caches are excluded by the
    allowlist.
    """
    root = source.resolve()
    files: dict[str, Path] = {}
    for child in sorted(source.rglob("*")):
        if child.is_symlink():
            reject_invalid_state(
                f"strict source tree must not contain symbolic links: {child}"
            )
        if not child.is_file():
            continue
        relative = PurePosixPath(child.relative_to(source).as_posix())
        if not _is_safe_relative(relative.as_posix()):
            reject_invalid_state(f"strict source member is unsafe: {relative}")
        if not _is_scientific_member(relative) or _is_sidecar(relative):
            continue
        if not child.resolve().is_relative_to(root):
            reject_invalid_state(f"strict source member escaped its root: {relative}")
        files[relative.as_posix()] = child
    return files


def establish_source_integrity(
    root: str | Path,
    source: str | Path,
    *,
    identity: str | None = None,
    provider_digest: str | None = None,
) -> IntegrityBaseline:
    """Record one trusted baseline for a newly admitted source.

    Call this only at an explicit trusted admission boundary: a local input the
    caller has accepted, or a fresh provider-verified download. It rereads the
    declared scientific payload and is therefore outside the fast path.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace-local source baseline directory, such as
        ``<work_dir>/integrity/sources``.
    source : str or pathlib.Path
        Original SLC/RSLC/HDF5 file or a materialized SAFE directory.
    identity : str, optional
        Caller-supplied cheap source identity. When supplied it must equal
        :func:`source_identity`, so a baseline can never be recorded for a
        different source incarnation.
    provider_digest : str, optional
        Independent ``sha256:<hex>`` evidence for one single-file source. A
        SAFE directory has no canonical single-file digest, so passing one is
        rejected rather than pretending a container digest equals the tree.

    Returns
    -------
    IntegrityBaseline
        The written baseline, keyed by source identity, and its fingerprints.

    Raises
    ------
    InvalidProcessingStateError
        If the source tree is unsafe, empty, changed during the reread, or a
        baseline already exists for the same identity.

    """
    baseline_root = _resolve_root(root)
    source_path = _resolve_source(source)
    current = source_identity(source_path)
    if identity is not None and identity != current:
        reject_invalid_state(
            "strict source identity changed before baseline establishment"
        )

    if source_path.is_file():
        baseline = establish_integrity(
            baseline_root,
            source_path,
            write_id=current,
            provider_digest=provider_digest,
        )
        if source_identity(source_path) != current:
            reject_invalid_state("strict source changed while its baseline was read")
        logger.info(
            "established strict source baseline for %s (identity=%s)",
            source_path,
            current,
        )
        return baseline

    if provider_digest is not None:
        message = "provider_digest requires a single-file source"
        raise ValueError(message)
    files = _scientific_files(source_path)
    if not files:
        reject_invalid_state(
            f"strict source has no declared scientific inputs: {source_path}"
        )
    baseline_path = _baseline_path(baseline_root, source_path)
    existing, error = _load_baseline(baseline_path, source_path)
    if error:
        logger.warning(
            "ignoring invalid prior source baseline %s: %s", baseline_path, error
        )
    if existing is not None and existing.write_id == current:
        reject_invalid_state(
            "a strict source baseline already exists for this source identity; "
            "re-admit a changed source before baselining"
        )

    fingerprints = _fingerprint(files)
    if source_identity(source_path) != current:
        reject_invalid_state("strict source changed while its baseline was read")
    created_at = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "schema": SOURCE_BASELINE_SCHEMA,
        "product": str(source_path),
        "write_id": current,
        "algorithm": "sha256",
        "files": {
            relative: {"size": value.size, "sha256": value.sha256}
            for relative, value in sorted(fingerprints.items())
        },
        "created_at": created_at,
    }
    _write_baseline_atomic(baseline_path, payload)
    logger.info(
        "established strict source baseline for %s (identity=%s, files=%d)",
        source_path,
        current,
        len(fingerprints),
    )
    return IntegrityBaseline(
        product=source_path,
        write_id=current,
        files=fingerprints,
        baseline_path=baseline_path,
        created_at=created_at,
    )


def _verify_directory(  # noqa: PLR0911
    root: Path,
    source: Path,
    identity: str,
    *,
    strict: bool,
    provider_digest: str | None,
) -> IntegrityVerdict:
    """Verify the declared SAFE scientific tree against its baseline."""
    if provider_digest is not None:
        message = "provider_digest requires a single-file source"
        raise ValueError(message)
    baseline_path = _baseline_path(root, source)
    baseline, error = _load_baseline(baseline_path, source)
    if error:
        return IntegrityVerdict(
            status="invalid_baseline",
            product=source,
            write_id=identity,
            strict=strict,
            detail=error,
        )
    if baseline is None:
        return IntegrityVerdict(
            status="integrity_unverifiable",
            product=source,
            write_id=identity,
            strict=strict,
            detail="no strict source baseline exists for this source",
        )
    if baseline.write_id != identity:
        return IntegrityVerdict(
            status="stale",
            product=source,
            write_id=identity,
            baseline_write_id=baseline.write_id,
            strict=strict,
            detail="source identity changed since its baseline was established",
        )
    if not strict:
        return IntegrityVerdict(
            status="metadata_verified",
            product=source,
            write_id=identity,
            baseline_write_id=baseline.write_id,
            strict=False,
            detail="source identity matches its baseline; payload not reread",
        )

    files = _scientific_files(source)
    if not files:
        return IntegrityVerdict(
            status="integrity_unverifiable",
            product=source,
            write_id=identity,
            baseline_write_id=baseline.write_id,
            strict=True,
            detail="source has no declared scientific inputs to verify",
        )
    fingerprints = _fingerprint(files)
    if source_identity(source) != identity:
        return IntegrityVerdict(
            status="stale",
            product=source,
            write_id=identity,
            baseline_write_id=baseline.write_id,
            strict=True,
            detail="source changed while its payload was being read",
        )
    mismatched = {
        relative
        for relative, expected in baseline.files.items()
        if fingerprints.get(relative) != expected
    }
    mismatched.update(set(fingerprints) - set(baseline.files))
    if mismatched:
        return IntegrityVerdict(
            status="mismatch",
            product=source,
            write_id=identity,
            baseline_write_id=baseline.write_id,
            strict=True,
            checked_files=len(fingerprints),
            mismatched=tuple(sorted(mismatched)),
            basis="baseline",
            detail="source payload bytes differ from the strict baseline",
        )
    return IntegrityVerdict(
        status="verified",
        product=source,
        write_id=identity,
        baseline_write_id=baseline.write_id,
        strict=True,
        checked_files=len(fingerprints),
        basis="baseline",
    )


def verify_source_integrity(
    root: str | Path,
    source: str | Path,
    *,
    strict: bool = False,
    identity: str | None = None,
    provider_digest: str | None = None,
) -> IntegrityVerdict:
    """Verify one original source against its trusted baseline.

    Fast mode (``strict=False``) reads only the cheap source identity and the
    baseline record; it never walks a SAFE tree or hashes payload bytes.
    Strict mode rereads every declared scientific member. A source without a
    baseline is ``integrity_unverifiable`` unless an independent provider
    digest verifies a single-file source; verification never fabricates a
    baseline from current bytes.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace-local source baseline directory.
    source : str or pathlib.Path
        Original SLC/RSLC/HDF5 file or materialized SAFE directory.
    strict : bool, optional
        Reread declared source payloads. Default ``False`` stays metadata-only.
    identity : str, optional
        Caller-supplied identity. When supplied it must equal
        :func:`source_identity`; a mismatch reports ``stale``.
    provider_digest : str, optional
        Independent ``sha256:<hex>`` evidence for one single-file source.

    Returns
    -------
    IntegrityVerdict
        Status, checked member count, and mismatched relative paths.

    """
    baseline_root = _resolve_root(root)
    source_path = _resolve_source(source)
    current = source_identity(source_path)
    if identity is not None and identity != current:
        return IntegrityVerdict(
            status="stale",
            product=source_path,
            write_id=current,
            baseline_write_id=identity,
            strict=strict,
            detail="caller source identity differs from the current source identity",
        )
    if source_path.is_file():
        verdict = verify_integrity(
            baseline_root,
            source_path,
            strict=strict,
            write_id=current,
            provider_digest=provider_digest,
        )
        if verdict.status == "verified":
            after = source_identity(source_path)
            if after != current:
                return IntegrityVerdict(
                    status="stale",
                    product=source_path,
                    write_id=after,
                    baseline_write_id=current,
                    strict=True,
                    detail="source changed while its payload was being read",
                )
        return verdict
    return _verify_directory(
        baseline_root,
        source_path,
        current,
        strict=strict,
        provider_digest=provider_digest,
    )
