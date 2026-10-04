"""Allowlisted local Stack input manifest (PROPOSAL-0049)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.stack.search import MissionPreparedInputs

logger = setup_logger(__name__)
MANIFEST_SCHEMA = "faninsar.stack.input_manifest.v2"


def _source_identity(
    path: str | Path,
    verification: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the lightweight source identity recorded by the manifest.

    PROPOSAL-0055: no byte hashing. The anchor is the download-time
    verification record (when the transfer layer supplied one) plus the file
    size, which reuse compares as a truncation tripwire. ``file_identity``
    remains available to callers that want an explicit byte snapshot.
    """
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    if candidate.is_dir():
        size = sum(
            item.stat().st_size for item in candidate.rglob("*") if item.is_file()
        )
        entry: dict[str, object] = {
            "path": str(candidate),
            "kind": "directory",
            "size": size,
        }
    else:
        entry = {
            "path": str(candidate),
            "kind": "file",
            "size": candidate.stat().st_size,
        }
    if verification:
        entry["verification"] = dict(verification)
    return entry


def file_identity(path: str | Path) -> dict[str, object]:
    """Return a bounded identity for one local file or directory."""
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    if candidate.is_dir():
        digest = hashlib.sha256()
        size = 0
        entries = 0
        for child in sorted(path for path in candidate.rglob("*") if path.is_file()):
            relative = child.relative_to(candidate).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
            entries += 1
        return {
            "path": str(candidate),
            "kind": "directory",
            "entries": entries,
            "size": size,
            "sha256": digest.hexdigest(),
        }
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(candidate),
        "kind": "file",
        "size": candidate.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _json_value(value: object) -> object:
    """Convert only safe scalar/container values to JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__geo_interface__"):
        return _json_value(value.__geo_interface__)
    return None


def build_manifest(
    *,
    mission: str,
    catalog_paths: Mapping[str, Sequence[str | Path]],
    prepared: MissionPreparedInputs,
    grid: object | None = None,
    dem: object | None = None,
    mask: object | None = None,
    processing_options: Mapping[str, object] | None = None,
    verification: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build the allowlisted manifest document without raw provider metadata.

    ``verification`` maps an input path to the download-time verification
    record `{algorithm, checksum, status, size}` supplied by the transfer
    layer (PROPOSAL-0055). No byte hashing happens here.
    """
    verification = verification or {}
    sources = {
        str(scene): [
            _source_identity(path, verification.get(str(Path(path)))) for path in paths
        ]
        for scene, paths in catalog_paths.items()
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "mission": mission,
        "sources": sources,
        "selected_footprints": [
            _json_value(footprint) for footprint in prepared.selected_footprints
        ],
        "orbits": {
            str(key): _source_identity(path, verification.get(str(Path(path))))
            for key, path in prepared.orbit_paths_by_acquisition.items()
        },
        "calibration": [
            _source_identity(path, verification.get(str(Path(path))))
            for path in prepared.calibration_paths
        ],
        "provenance": dict(prepared.provenance),
        "grid": _json_value(grid),
        "dem": _json_value(dem),
        "mask": _json_value(mask),
        "processing_options": _json_value(processing_options or {}),
    }


def validate_manifest(
    document: object, *, work_dir: str | Path | None = None
) -> dict[str, object]:
    """Validate and return a manifest with existing local bindings.

    PROPOSAL-0055: no byte hashing. Each bound input must exist and match its
    recorded size (the truncation tripwire); legacy v1 manifests carrying
    `sha256` fields load unchanged with those fields inert. Unknown
    top-level keys are rejected so arbitrary provider metadata cannot become
    part of the persistent Stack input contract.
    """
    schema = document.get("schema") if isinstance(document, dict) else None
    if schema not in {MANIFEST_SCHEMA, "faninsar.stack.input_manifest.v1"}:
        message = "unsupported Stack input manifest"
        raise ValueError(message)
    allowed = {
        "schema",
        "mission",
        "sources",
        "selected_footprints",
        "orbits",
        "calibration",
        "provenance",
        "grid",
        "dem",
        "mask",
        "processing_options",
    }
    if set(document) - allowed:
        message = "Stack input manifest contains unknown fields"
        raise ValueError(message)
    # Source SAFE and qualified orbit files may live in a configured cache
    # root outside ``work_dir``.  The caller that owns those roots is the only
    # layer able to apply confinement correctly, so this utility validates
    # existence and identity without inventing a work-directory boundary.
    del work_dir
    for group in ("sources", "orbits", "calibration"):
        values = document.get(group, {} if group != "calibration" else [])
        if group == "sources":
            if not isinstance(values, dict):
                message = "invalid manifest sources binding"
                raise ValueError(message)
            entries = [entry for paths in values.values() for entry in paths]
        else:
            entries = values.values() if isinstance(values, dict) else values
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                message = f"invalid manifest {group} binding"
                raise TypeError(message)
            path = Path(entry["path"]).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            # PROPOSAL-0055: size tripwire only. Legacy sha256 fields are
            # inert — never compared, never recomputed.
            recorded_size = entry.get("size")
            if isinstance(recorded_size, int):
                if entry.get("kind") == "directory":
                    found = sum(
                        item.stat().st_size
                        for item in path.rglob("*")
                        if item.is_file()
                    )
                else:
                    found = path.stat().st_size
                if found != recorded_size:
                    message = (
                        f"manifest size mismatch for {path}: recorded "
                        f"{recorded_size}, found {found}"
                    )
                    raise ValueError(message)
    return document


def read_manifest(
    path: str | Path, *, work_dir: str | Path | None = None
) -> dict[str, object]:
    """Read and validate one Stack input manifest."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_manifest(document, work_dir=work_dir)


def write_manifest(path: str | Path, document: Mapping[str, object]) -> None:
    """Atomically validate and publish one JSON manifest with ``os.replace``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, sort_keys=True, indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        validate_manifest(json.loads(temporary.read_text(encoding="utf-8")))
        os.replace(temporary, destination)  # noqa: PTH105
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "MANIFEST_SCHEMA",
    "build_manifest",
    "file_identity",
    "read_manifest",
    "validate_manifest",
    "write_manifest",
]
