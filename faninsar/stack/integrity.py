"""Optional strict byte-integrity baselines for Stack products (PROPOSAL-0088).

Fast Stack reuse never hashes product payloads (PROPOSAL-0082). This module is
the narrow opt-in exception: a caller that publishes a new strict result may
establish one baseline of that product's declared payload files. Verification
then rereads those bytes only when ``strict=True``.

A baseline records bytes, not scientific truth. It is keyed by the committed
product ``write_id`` so a replacement product can never be accepted against an
older baseline. Products published without a baseline report
``integrity_unverifiable``; current bytes are never hashed into a synthetic
historical baseline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = setup_logger(__name__)

BASELINE_SCHEMA = "faninsar.integrity.baseline.v1"
"""Versioned JSON schema written for one strict product baseline."""

ITEM_NAME = "item.json"
"""Committed product metadata name, also written by :mod:`product`."""

_HASH_CHUNK_BYTES = 1024 * 1024
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

IntegrityStatus: TypeAlias = Literal[
    "metadata_verified",
    "verified",
    "integrity_unverifiable",
    "stale",
    "mismatch",
    "invalid_baseline",
]
"""Outcome of one :func:`verify_integrity` call."""

__all__ = [
    "BASELINE_SCHEMA",
    "FileFingerprint",
    "IntegrityBaseline",
    "IntegrityStatus",
    "IntegrityVerdict",
    "establish_integrity",
    "verify_integrity",
]


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    """Recorded size and SHA-256 for one payload file."""

    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class IntegrityBaseline:
    """One parsed baseline for a committed product write id."""

    product: Path
    write_id: str
    files: Mapping[str, FileFingerprint]
    baseline_path: Path
    created_at: str


@dataclass(frozen=True, slots=True)
class IntegrityVerdict:
    """Outcome of one integrity verification.

    ``metadata_verified`` means only the product metadata and baseline write id
    matched; no payload bytes were read. ``verified`` means the strict payload
    reread (or an independent provider digest) matched. ``mismatched`` lists
    the relative payload paths that differ, or is empty for a provider-only
    digest failure described by ``detail``.
    """

    status: IntegrityStatus
    product: Path
    write_id: str | None
    baseline_write_id: str | None = None
    strict: bool = False
    checked_files: int = 0
    mismatched: tuple[str, ...] = ()
    basis: str | None = None
    detail: str = ""

    @property
    def verified(self) -> bool:
        """Return whether payload bytes were actually verified."""
        return self.status == "verified"


def _valid_write_id(value: object) -> bool:
    """Return whether ``value`` is a non-empty, short identity string."""
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 128
        and not any(character.isspace() or ord(character) < 32 for character in value)
    )


def _resolve_root(root: str | Path) -> Path:
    """Return an existing baseline directory without following unsafe links."""
    path = Path(root)
    if path.is_symlink():
        reject_invalid_state("integrity baseline root must not be a symbolic link")
    if path.exists() and not path.is_dir():
        reject_invalid_state("integrity baseline root must be a directory")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _resolve_product(product: str | Path) -> Path:
    """Return one existing product directory or payload file."""
    path = Path(product)
    if path.is_symlink():
        reject_invalid_state("integrity product must not be a symbolic link")
    if not path.exists():
        reject_invalid_state(f"integrity product is missing: {path}")
    if not (path.is_dir() or path.is_file()):
        reject_invalid_state(f"integrity product is not a file or directory: {path}")
    return path.resolve()


def _baseline_path(root: Path, product: Path) -> Path:
    """Return the deterministic baseline path for ``product`` under ``root``."""
    key = hashlib.sha256(str(product).encode("utf-8")).hexdigest()[:32]
    path = root / f"baseline-{key}.json"
    if path.resolve().parent != root:
        reject_invalid_state("integrity baseline path escaped its root")
    return path


def _fsync_directory(path: Path) -> None:
    """Flush one directory entry so an atomic rename survives a crash."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _replace_file(source: Path, target: Path) -> None:
    """Atomically replace ``target`` with ``source``."""
    source.replace(target)


def _write_baseline_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one strict-JSON baseline to a sibling temp file, then rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_file(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _is_safe_relative(value: object) -> bool:
    """Return whether ``value`` is a safe POSIX relative file key."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts:
        return False
    return all(part not in {"", ".", ".."} for part in candidate.parts)


def _asset_href(value: object) -> str:
    """Return one plain asset basename, else reject a path escape."""
    if not isinstance(value, str):
        reject_invalid_state("integrity product asset href must be a string")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != value
        or value in {"", ".", ".."}
        or "\\" in value
        or "\x00" in value
    ):
        reject_invalid_state(
            f"integrity product asset href must be a basename: {value!r}"
        )
    return value


def _read_item_metadata(
    product: Path,
) -> tuple[str, Mapping[str, Any]] | None:
    """Read the committed write id and asset map without inspecting payloads.

    A directory without ``item.json`` is treated as a raw payload tree whose
    write id must be supplied by the caller. A present but malformed item is
    rejected rather than silently treated as raw payload.
    """
    if not product.is_dir():
        return None
    item_path = product / ITEM_NAME
    if not item_path.exists() and not item_path.is_symlink():
        return None
    if item_path.is_symlink() or not item_path.is_file():
        reject_invalid_state(f"integrity product item is unsafe: {item_path}")
    try:
        item = json.loads(item_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        reject_invalid_state(f"integrity product item cannot be read: {error}")
    properties = item.get("properties") if isinstance(item, dict) else None
    assets = item.get("assets") if isinstance(item, dict) else None
    if not isinstance(properties, dict) or not isinstance(assets, dict):
        reject_invalid_state(
            f"integrity product item is not a FanInSAR product: {item_path}"
        )
    write_id = properties.get("faninsar:write_id")
    if not _valid_write_id(write_id):
        reject_invalid_state(f"integrity product item has no write id: {item_path}")
    return str(write_id), assets


def _committed_write_id(product: Path) -> str | None:
    """Return the current item write id, or ``None`` for a raw payload path."""
    metadata = _read_item_metadata(product)
    return None if metadata is None else metadata[0]


def _load_baseline(  # noqa: PLR0911
    path: Path, product: Path
) -> tuple[IntegrityBaseline | None, str]:
    """Return one baseline or ``(None, reason)`` when absent or invalid."""
    if not path.exists() and not path.is_symlink():
        return None, ""
    if path.is_symlink() or not path.is_file():
        return None, "integrity baseline is not a regular file"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"integrity baseline cannot be read: {error}"
    if not isinstance(payload, dict):
        return None, "integrity baseline is not a JSON object"
    if payload.get("schema") != BASELINE_SCHEMA:
        return None, "integrity baseline schema is unsupported"
    if payload.get("algorithm") != "sha256":
        return None, "integrity baseline algorithm is unsupported"
    if payload.get("product") != str(product):
        return None, "integrity baseline belongs to another product path"
    write_id = payload.get("write_id")
    if not _valid_write_id(write_id):
        return None, "integrity baseline write id is invalid"
    files = payload.get("files")
    if not isinstance(files, dict):
        return None, "integrity baseline files are missing"
    fingerprints: dict[str, FileFingerprint] = {}
    for key, value in files.items():
        if not _is_safe_relative(key):
            return None, "integrity baseline file key is unsafe"
        relative = str(key)
        if not isinstance(value, dict):
            return None, f"integrity baseline entry is invalid: {relative}"
        size = value.get("size")
        digest = value.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or _SHA256_HEX.fullmatch(digest) is None
        ):
            return None, f"integrity baseline entry is invalid: {relative}"
        fingerprints[relative] = FileFingerprint(size=size, sha256=digest)
    created_at = payload.get("created_at", "")
    if not isinstance(created_at, str):
        return None, "integrity baseline created_at is invalid"
    return (
        IntegrityBaseline(
            product=product,
            write_id=str(write_id),
            files=fingerprints,
            baseline_path=path,
            created_at=created_at,
        ),
        "",
    )


def _is_baseline_artifact(path: Path, baseline_path: Path) -> bool:
    """Return whether ``path`` is the baseline file or its atomic temp sibling."""
    if path == baseline_path:
        return True
    return path.parent == baseline_path.parent and path.name.startswith(
        f".{baseline_path.name}."
    )


def _walk_payload(
    product: Path,
    assets: Mapping[str, Any] | None,
    baseline_path: Path,
) -> tuple[dict[str, Path], list[str]]:
    """Return payload files and missing declared assets.

    Committed products hash only declared assets so reports, baselines and
    incidental workspace state are never swallowed into the fingerprint. A
    raw payload tree hashes every regular file under it, excluding the
    baseline artifact itself.
    """
    files: dict[str, Path] = {}
    missing: list[str] = []
    if assets is None:
        if product.is_file():
            if not _is_baseline_artifact(product, baseline_path):
                files[product.name] = product
            return files, missing
        for child in sorted(product.rglob("*")):
            if child.is_symlink():
                reject_invalid_state(
                    f"integrity payload must not contain symbolic links: {child}"
                )
            if not child.is_file() or _is_baseline_artifact(child, baseline_path):
                continue
            relative = child.relative_to(product).as_posix()
            files[relative] = child
        return files, missing

    for name, asset in sorted(assets.items()):
        if not isinstance(asset, dict):
            reject_invalid_state(f"integrity product asset {name!r} is invalid")
        href = _asset_href(asset.get("href"))
        source = product / href
        if source.is_symlink():
            reject_invalid_state(f"integrity product asset {name!r} is a symbolic link")
        if not source.exists():
            missing.append(href)
            continue
        if source.is_dir():
            for child in sorted(source.rglob("*")):
                if child.is_symlink():
                    reject_invalid_state(
                        "integrity product asset tree contains a symbolic link: "
                        f"{child}"
                    )
                if not child.is_file() or _is_baseline_artifact(child, baseline_path):
                    continue
                relative = child.relative_to(product).as_posix()
                files[relative] = child
        elif source.is_file():
            relative = source.relative_to(product).as_posix()
            files[relative] = source
        else:
            reject_invalid_state(
                f"integrity product asset {name!r} is not a file or directory"
            )
    return files, missing


def _hash_file(path: Path) -> tuple[str, int]:
    """Return ``(sha256_hex, size_bytes)`` for one regular file."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _fingerprint(files: Mapping[str, Path]) -> dict[str, FileFingerprint]:
    """Hash every payload file and return a sorted mapping."""
    fingerprints: dict[str, FileFingerprint] = {}
    for relative, path in sorted(files.items()):
        digest, size = _hash_file(path)
        fingerprints[relative] = FileFingerprint(size=size, sha256=digest)
    return fingerprints


def _provider_matches(
    files: Mapping[str, FileFingerprint], provider_digest: str | None
) -> bool:
    """Return whether an external SHA-256 matches a single payload file.

    A provider digest is trusted external evidence for actual bytes. It is
    accepted only when the product has exactly one payload file, so a
    canonical metadata manifest is never passed off as a file checksum.
    """
    if provider_digest is None:
        return True
    expected = _normalize_provider_digest(provider_digest)
    if len(files) != 1:
        message = "provider_digest requires a single-file product"
        raise ValueError(message)
    return next(iter(files.values())).sha256 == expected


def _normalize_provider_digest(value: str) -> str:
    """Return a lowercase SHA-256 hex digest from a qualified provider value."""
    if not isinstance(value, str):
        message = "provider_digest must be a string"
        raise TypeError(message)
    text = value.strip().lower().removeprefix("sha256:")
    if _SHA256_HEX.fullmatch(text) is None:
        message = "provider_digest must be 'sha256:<64 hex>' or 64 hex digits"
        raise ValueError(message)
    return text


def establish_integrity(
    root: str | Path,
    product: str | Path,
    *,
    write_id: str,
    provider_digest: str | None = None,
) -> IntegrityBaseline:
    """Record one strict payload baseline for a newly published product.

    Call this only when the caller has just published the product under
    ``write_id`` (for example immediately after ``ProductWriter.commit`` or
    ``UnitCheckpointStore.commit``). It always rereads the full declared
    payload, so it must stay outside the ordinary fast path.

    Parameters
    ----------
    root : str or pathlib.Path
        Caller-owned baseline directory, such as ``<work_dir>/integrity``.
    product : str or pathlib.Path
        Committed product directory, raw Zarr directory, or TIFF payload file.
    write_id : str
        Exact committed product write id. For an ``item.json`` product it must
        equal ``faninsar:write_id``; re-baselining an existing write id is
        refused so mutated bytes cannot be laundered into a fresh baseline.
    provider_digest : str, optional
        Independent ``sha256:<hex>`` evidence for one actual payload file.
        When supplied, the published single-file payload must match before the
        baseline is written. Multi-file products require a payload baseline.

    Returns
    -------
    IntegrityBaseline
        The written baseline, including its per-file fingerprints.

    Raises
    ------
    InvalidProcessingStateError
        If the product, metadata, paths, or provider digest are invalid, or a
        baseline already exists for ``write_id``.

    """
    baseline_root = _resolve_root(root)
    product_path = _resolve_product(product)
    if not _valid_write_id(write_id):
        reject_invalid_state(f"integrity write id is invalid: {write_id!r}")
    metadata = _read_item_metadata(product_path)
    if metadata is None:
        current_write_id = write_id
        committed_before = None
        assets: Mapping[str, Any] | None = None
    else:
        current_write_id, assets = metadata
        committed_before = current_write_id
        if current_write_id != write_id:
            reject_invalid_state(
                f"integrity write id {write_id!r} does not match committed "
                f"product write id {current_write_id!r}"
            )

    path = _baseline_path(baseline_root, product_path)
    existing, error = _load_baseline(path, product_path)
    if error:
        logger.warning("ignoring invalid prior integrity baseline %s: %s", path, error)
    if existing is not None and existing.write_id == write_id:
        reject_invalid_state(
            f"an integrity baseline already exists for product write id "
            f"{write_id!r}; publish a new product write id before baselining"
        )

    files, missing = _walk_payload(product_path, assets, path)
    if missing:
        reject_invalid_state(
            "integrity product has missing declared assets: " + ", ".join(missing)
        )
    fingerprints = _fingerprint(files)
    after_write_id = _committed_write_id(product_path)
    if after_write_id != committed_before:
        reject_invalid_state(
            "integrity product write id changed while its baseline was being read"
        )
    if provider_digest is not None and not _provider_matches(
        fingerprints, provider_digest
    ):
        reject_invalid_state(
            "integrity provider digest does not match the published payload"
        )
    created_at = datetime.now(UTC).isoformat()
    payload = {
        "schema": BASELINE_SCHEMA,
        "product": str(product_path),
        "write_id": write_id,
        "algorithm": "sha256",
        "files": {
            relative: {"size": value.size, "sha256": value.sha256}
            for relative, value in sorted(fingerprints.items())
        },
        "created_at": created_at,
    }
    _write_baseline_atomic(path, payload)
    logger.info(
        "established strict integrity baseline for %s (write_id=%s, files=%d)",
        product_path,
        write_id,
        len(fingerprints),
    )
    return IntegrityBaseline(
        product=product_path,
        write_id=write_id,
        files=fingerprints,
        baseline_path=path,
        created_at=created_at,
    )


def verify_integrity(  # noqa: PLR0911
    root: str | Path,
    product: str | Path,
    *,
    strict: bool = False,
    write_id: str | None = None,
    provider_digest: str | None = None,
) -> IntegrityVerdict:
    """Verify one product against its baseline or an independent provider digest.

    Fast mode (``strict=False``, the default) reads only ``item.json`` and the
    small baseline record; it never scans payload bytes. Strict mode rereads
    every declared asset. A product without a baseline is explicitly
    ``integrity_unverifiable`` unless an independent provider digest verifies
    it; strict verification never fabricates a baseline from current bytes.

    Parameters
    ----------
    root : str or pathlib.Path
        Caller-owned baseline directory used by :func:`establish_integrity`.
    product : str or pathlib.Path
        Product directory, raw Zarr directory, or TIFF payload file.
    strict : bool, optional
        Reread every payload file. Default ``False`` keeps the fast path
        metadata-only.
    write_id : str, optional
        Exact write id for a raw file/directory without ``item.json``. For an
        item product the committed metadata is authoritative and a differing
        caller value is reported as ``stale``.
    provider_digest : str, optional
        Independent ``sha256:<hex>`` evidence for one actual payload file,
        used only in strict mode. It can verify a single-file product that has
        no baseline. A canonical metadata manifest is never accepted as a
        file checksum, so multi-file products require a baseline.

    Returns
    -------
    IntegrityVerdict
        Status, checked file count, and mismatched relative paths. A missing
        baseline reports ``integrity_unverifiable``; a baseline for another
        write id reports ``stale``; changed bytes report ``mismatch``; a
        malformed baseline reports ``invalid_baseline``.

    Raises
    ------
    InvalidProcessingStateError
        If the product or a payload path is missing or unsafe, or a raw
        payload has no caller-supplied write id.
    ValueError or TypeError
        If ``provider_digest`` is not a SHA-256 value.

    """
    baseline_root = _resolve_root(root)
    product_path = _resolve_product(product)
    metadata = _read_item_metadata(product_path)
    if metadata is None:
        if not _valid_write_id(write_id):
            reject_invalid_state(
                "integrity verification of a payload without item.json requires "
                "the caller-supplied write id"
            )
        current_write_id = str(write_id)
        committed_before = None
        assets: Mapping[str, Any] | None = None
    else:
        current_write_id, assets = metadata
        committed_before = current_write_id
        if write_id is not None and write_id != current_write_id:
            return IntegrityVerdict(
                status="stale",
                product=product_path,
                write_id=current_write_id,
                strict=strict,
                detail=(
                    f"caller write id {write_id!r} differs from committed "
                    f"product write id {current_write_id!r}"
                ),
            )

    path = _baseline_path(baseline_root, product_path)
    baseline, error = _load_baseline(path, product_path)
    if error:
        return IntegrityVerdict(
            status="invalid_baseline",
            product=product_path,
            write_id=current_write_id,
            strict=strict,
            detail=error,
        )

    if baseline is None:
        if strict and provider_digest is not None:
            files, missing = _walk_payload(product_path, assets, path)
            fingerprints = _fingerprint(files)
            if _committed_write_id(product_path) != committed_before:
                return IntegrityVerdict(
                    status="stale",
                    product=product_path,
                    write_id=current_write_id,
                    strict=True,
                    detail="product write id changed while payload was being read",
                )
            if missing:
                return IntegrityVerdict(
                    status="mismatch",
                    product=product_path,
                    write_id=current_write_id,
                    strict=True,
                    checked_files=len(fingerprints),
                    mismatched=tuple(sorted(missing)),
                    basis="provider",
                    detail="declared payload assets are missing",
                )
            if not _provider_matches(fingerprints, provider_digest):
                return IntegrityVerdict(
                    status="mismatch",
                    product=product_path,
                    write_id=current_write_id,
                    strict=True,
                    checked_files=len(fingerprints),
                    basis="provider",
                    detail="provider digest does not match the payload",
                )
            return IntegrityVerdict(
                status="verified",
                product=product_path,
                write_id=current_write_id,
                strict=True,
                checked_files=len(fingerprints),
                basis="provider",
                detail="verified against an independent provider digest",
            )
        return IntegrityVerdict(
            status="integrity_unverifiable",
            product=product_path,
            write_id=current_write_id,
            baseline_write_id=None,
            strict=strict,
            detail="no integrity baseline exists for this product",
        )

    if baseline.write_id != current_write_id:
        return IntegrityVerdict(
            status="stale",
            product=product_path,
            write_id=current_write_id,
            baseline_write_id=baseline.write_id,
            strict=strict,
            detail=(
                f"baseline write id {baseline.write_id!r} differs from current "
                f"product write id {current_write_id!r}"
            ),
        )

    if not strict:
        return IntegrityVerdict(
            status="metadata_verified",
            product=product_path,
            write_id=current_write_id,
            baseline_write_id=baseline.write_id,
            strict=False,
            detail="metadata and baseline write id match; payload not reread",
        )

    files, missing = _walk_payload(product_path, assets, path)
    fingerprints = _fingerprint(files)
    if _committed_write_id(product_path) != committed_before:
        return IntegrityVerdict(
            status="stale",
            product=product_path,
            write_id=current_write_id,
            baseline_write_id=baseline.write_id,
            strict=True,
            detail="product write id changed while payload was being read",
        )
    mismatched = set(missing)
    for relative, expected in baseline.files.items():
        actual = fingerprints.get(relative)
        if actual is None or actual != expected:
            mismatched.add(relative)
    mismatched.update(set(fingerprints) - set(baseline.files))

    basis = "baseline"
    detail = ""
    if provider_digest is not None:
        if not _provider_matches(fingerprints, provider_digest):
            detail = "provider digest does not match the payload"
            mismatched.add("<provider_digest>")
        basis = "baseline+provider"
    if mismatched:
        return IntegrityVerdict(
            status="mismatch",
            product=product_path,
            write_id=current_write_id,
            baseline_write_id=baseline.write_id,
            strict=True,
            checked_files=len(fingerprints),
            mismatched=tuple(sorted(mismatched)),
            basis=basis,
            detail=detail or "payload bytes differ from the integrity baseline",
        )
    return IntegrityVerdict(
        status="verified",
        product=product_path,
        write_id=current_write_id,
        baseline_write_id=baseline.write_id,
        strict=True,
        checked_files=len(fingerprints),
        basis=basis,
        detail=detail,
    )
