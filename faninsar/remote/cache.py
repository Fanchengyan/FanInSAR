"""Private cache identity and manifest helpers for remote assets."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .records import RemoteAsset

_COMPLETION_SCHEMA = "p0082-completion-v1"


def _identity(asset: RemoteAsset) -> str:
    """Create a stable cache identity excluding mutable URL details."""
    fields = [
        asset.provider,
        asset.catalog,
        asset.collection,
        asset.item_id,
        asset.key,
        asset.version,
        asset.representation,
        asset.media_type,
        asset.checksum,
        asset.auth_profile,
    ]
    encoded = json.dumps(fields, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _manifest_path(path: Path, asset: RemoteAsset) -> Path:
    """Return a private manifest path qualified by the asset identity."""
    return path.with_name(f"{path.name}.faninsar.remote.{_identity(asset)}.json")


def _is_qualified(asset: RemoteAsset) -> bool:
    """Return whether an asset has a stable version or content checksum."""
    return asset.checksum is not None or asset.version is not None


def _manifest_is_complete(manifest: object, size: int) -> bool:
    """Return whether a parsed manifest is a reusable completion record.

    Completed reuse trusts identity (checked by the caller), expected size, and
    completion state. Local checksum strings are not compared and the
    destination is not rescanned. A failed provider-verification record is
    never reusable, including against an older same-sized destination.
    """
    if not isinstance(manifest, dict) or manifest.get("size") != size:
        return False
    verification = manifest.get("verification")
    if isinstance(verification, dict) and verification.get("status") == "failed":
        return False
    complete = manifest.get("complete")
    if complete is False:
        return False
    schema = manifest.get("schema")
    if schema is not None and schema != _COMPLETION_SCHEMA:
        return False
    if schema == _COMPLETION_SCHEMA:
        return complete is True
    return True


def _matching_manifest(path: Path, asset: RemoteAsset) -> bool:
    """Return whether an existing destination is a valid qualified reuse."""
    manifest_path = _manifest_path(path, asset)
    if not path.is_file() or not manifest_path.is_file():
        return False
    if not _is_qualified(asset):
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(manifest, dict) or manifest.get("identity") != _identity(asset):
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return _manifest_is_complete(manifest, size)


__all__ = [
    "_COMPLETION_SCHEMA",
    "_identity",
    "_is_qualified",
    "_manifest_path",
    "_matching_manifest",
]
