"""PROPOSAL-0082 transfer completion: reuse without payload self-hashing.

Public ``remote.download`` reuses a matching identity/size/completion record
without scanning destination bytes. Provider digest comparison, including a
resumed prefix, stays on the streaming path. A failed overwrite must not make
the previous same-sized destination reusable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from shapely.geometry import Polygon

from faninsar import remote
from faninsar.data.query import Points
from faninsar.remote.cache import _identity, _manifest_path
from faninsar.remote.catalog import _register_adapter

if TYPE_CHECKING:
    from pathlib import Path

PAYLOAD = b"fixture archive bytes"
OTHER = b"DIFFERENT archive byt"
assert len(PAYLOAD) == len(OTHER)


@dataclass
class _CountingAdapter:
    """In-memory adapter that records each complete-file fetch."""

    payload: bytes
    checksum: str | None
    fetches: list[int] = field(default_factory=list)
    provider: str = "fixture"
    origins: tuple[str, ...] = ("https://fixture.invalid",)
    path_prefixes: tuple[str, ...] = ("/",)
    redirect_origins: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ("anonymous",)

    def items(self) -> list[dict[object, object]]:
        """Return one fixture catalog record."""
        asset: dict[object, object] = {
            "href": "https://fixture.invalid/assets/data",
            "data": self.payload,
            "size": len(self.payload),
        }
        if self.checksum is not None:
            asset["checksum"] = self.checksum
        return [
            {
                "id": "item-1",
                "collection": "slc",
                "geometry": Polygon(((-1, -1), (1, -1), (1, 1), (-1, 1))),
                "assets": {"data": asset},
            }
        ]

    def fetch(self, asset: remote.RemoteAsset, budget: object) -> bytes:
        """Return the current payload and record the fetch."""
        del asset, budget
        self.fetches.append(len(self.payload))
        return self.payload


def _asset(name: str) -> remote.RemoteAsset:
    """Return the data asset of one registered counting catalog."""
    return remote.search(Points([(0, 0)], crs=4326), catalog=name)[0].assets["data"]


def _spy_file_digest(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record ``hashlib.file_digest`` algorithms used during the test."""
    scans: list[str] = []
    real = hashlib.file_digest

    def spy(stream: object, algorithm: str, *args: object, **kwargs: object) -> object:
        scans.append(str(algorithm))
        return real(stream, algorithm, *args, **kwargs)

    monkeypatch.setattr(hashlib, "file_digest", spy)
    return scans


def test_completed_download_reuses_without_requests_or_payload_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed destination is reused with zero fetches and no file digest."""
    scans = _spy_file_digest(monkeypatch)
    name = "p0082-reuse"
    adapter = _CountingAdapter(PAYLOAD, f"md5:{hashlib.md5(PAYLOAD).hexdigest()}")
    _register_adapter(name, adapter)
    asset = _asset(name)
    destination = tmp_path / "scene.zip"

    first = remote.download(asset, destination)
    assert first == destination
    assert destination.read_bytes() == PAYLOAD
    assert adapter.fetches == [len(PAYLOAD)]
    scans.clear()
    adapter.fetches.clear()

    reused = remote.download(asset, destination)

    assert reused == destination
    assert adapter.fetches == []
    assert scans == []
    manifest = json.loads(_manifest_path(destination, asset).read_text())
    assert manifest["identity"] == _identity(asset)
    assert manifest["size"] == len(PAYLOAD)
    assert manifest.get("complete") is True
    assert manifest["verification"]["status"] == "verified"
    assert (
        "sha256" not in manifest
        or manifest["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    )


def test_legacy_manifest_reuses_without_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity, size, and completion are enough; legacy sha256 is not compared."""
    scans = _spy_file_digest(monkeypatch)
    name = "p0082-legacy"
    adapter = _CountingAdapter(PAYLOAD, f"md5:{hashlib.md5(PAYLOAD).hexdigest()}")
    _register_adapter(name, adapter)
    asset = _asset(name)
    destination = tmp_path / "scene.zip"
    remote.download(asset, destination)
    manifest_path = _manifest_path(destination, asset)
    manifest_path.write_text(
        json.dumps(
            {
                "identity": _identity(asset),
                "sha256": "00" * 32,
                "size": len(PAYLOAD),
            }
        )
    )
    adapter.fetches.clear()
    scans.clear()

    assert remote.download(asset, destination) == destination
    assert adapter.fetches == []
    assert scans == []


def test_failed_overwrite_does_not_reuse_old_same_sized_destination(
    tmp_path: Path,
) -> None:
    """Failed provider verification must not authorize the previous destination."""
    name = "p0082-failed-overwrite"
    adapter = _CountingAdapter(PAYLOAD, f"md5:{hashlib.md5(PAYLOAD).hexdigest()}")
    _register_adapter(name, adapter)
    asset = _asset(name)
    destination = tmp_path / "scene.zip"
    remote.download(asset, destination)
    adapter.payload = OTHER
    adapter.fetches.clear()

    with pytest.raises(remote.RemoteIntegrityError) as raised:
        remote.download(asset, destination, overwrite=True)

    assert raised.value.reason == "checksum_mismatch"
    assert destination.read_bytes() == PAYLOAD
    manifest = json.loads(_manifest_path(destination, asset).read_text())
    assert manifest["verification"]["status"] == "failed"
    assert manifest.get("complete") is False
    adapter.fetches.clear()

    with pytest.raises(remote.RemoteIntegrityError, match="destination_conflict"):
        remote.download(asset, destination)
    assert adapter.fetches == []
    assert destination.read_bytes() == PAYLOAD


def test_same_size_mutation_after_completion_is_not_detected(tmp_path: Path) -> None:
    """Same-size edits after a valid completion record are intentionally trusted."""
    name = "p0082-same-size"
    adapter = _CountingAdapter(PAYLOAD, f"md5:{hashlib.md5(PAYLOAD).hexdigest()}")
    _register_adapter(name, adapter)
    asset = _asset(name)
    destination = tmp_path / "scene.zip"
    remote.download(asset, destination)
    destination.write_bytes(OTHER)
    adapter.fetches.clear()

    assert remote.download(asset, destination) == destination
    assert adapter.fetches == []
    assert destination.read_bytes() == OTHER


def test_torn_manifest_does_not_reuse(tmp_path: Path) -> None:
    """A torn completion record is not a cache hit."""
    name = "p0082-torn"
    adapter = _CountingAdapter(PAYLOAD, f"md5:{hashlib.md5(PAYLOAD).hexdigest()}")
    _register_adapter(name, adapter)
    asset = _asset(name)
    destination = tmp_path / "scene.zip"
    remote.download(asset, destination)
    _manifest_path(destination, asset).write_text("{")
    adapter.fetches.clear()

    with pytest.raises(remote.RemoteIntegrityError, match="destination_conflict"):
        remote.download(asset, destination)
    assert adapter.fetches == []


def test_unverifiable_versioned_download_reuses_without_self_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No provider checksum stays unverifiable; reuse still skips payload hashing."""
    scans = _spy_file_digest(monkeypatch)
    name = "p0082-unverifiable"

    @dataclass
    class _VersionedAdapter(_CountingAdapter):
        def items(self) -> list[dict[object, object]]:
            record = super().items()[0]
            record["assets"]["data"]["version"] = "v1"
            return [record]

    adapter = _VersionedAdapter(PAYLOAD, None)
    _register_adapter(name, adapter)
    asset = _asset(name)
    destination = tmp_path / "scene.zip"
    remote.download(asset, destination)
    manifest = json.loads(_manifest_path(destination, asset).read_text())
    assert manifest["verification"] == {"status": "unverifiable", "size": len(PAYLOAD)}
    assert "sha256" not in manifest
    adapter.fetches.clear()
    scans.clear()

    assert remote.download(asset, destination) == destination
    assert adapter.fetches == []
    assert scans == []
