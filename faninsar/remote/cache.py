"""Private cache identity and manifest helpers for remote assets."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .records import RemoteAsset


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
    if manifest.get("identity") != _identity(asset):
        return False
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return False
    return manifest == {"identity": _identity(asset), "sha256": digest, "size": size}


__all__ = ["_identity", "_is_qualified", "_manifest_path", "_matching_manifest"]
