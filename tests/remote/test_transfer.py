"""Download-time verification recording (PROPOSAL-0055 slice 2)."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
from shapely.geometry import Polygon

from faninsar import remote
from faninsar.data.query import Points

if TYPE_CHECKING:
    from pathlib import Path

PAYLOAD = b"fixture archive bytes"


def _register(name: str, checksum: str | None) -> None:
    """Register a one-asset fixture catalog with an optional checksum."""
    asset: dict[object, object] = {
        "href": "https://fixture.invalid/assets/data",
        "data": PAYLOAD,
        "size": len(PAYLOAD),
    }
    if checksum is not None:
        asset["checksum"] = checksum
    remote._register_fixture(
        [
            {
                "id": "item-1",
                "collection": "slc",
                "geometry": Polygon(((-1, -1), (1, -1), (1, 1), (-1, 1))),
                "assets": {"data": asset},
            }
        ],
        name=name,
    )


def _asset(name: str) -> remote.RemoteAsset:
    """Return the data asset of the registered fixture catalog."""
    item = remote.search(Points([(0, 0)], crs=4326), catalog=name)[0]
    return item.assets["data"]


def _manifest(destination: Path, asset: remote.RemoteAsset) -> dict[object, object]:
    """Read the private per-destination transfer manifest."""
    return json.loads(remote._manifest_path(destination, asset).read_text())


def test_verified_provider_checksum_is_recorded(tmp_path: Path) -> None:
    """A matching provider checksum records a verified entry with size."""
    name = "p0055-verified"
    md5 = hashlib.md5(PAYLOAD).hexdigest()
    _register(name, f"md5:{md5}")
    asset = _asset(name)
    destination = tmp_path / "scene.zip"

    result = remote.download(asset, destination)

    assert result == destination
    assert destination.read_bytes() == PAYLOAD
    manifest = _manifest(destination, asset)
    assert manifest["identity"] == remote._identity(asset)
    assert manifest["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert manifest["verification"] == {
        "algorithm": "md5",
        "checksum": md5,
        "status": "verified",
        "size": len(PAYLOAD),
    }


def test_failed_verification_is_persisted_before_integrity_error(
    tmp_path: Path,
) -> None:
    """A mismatch persists status failed in the manifest before raising."""
    name = "p0055-failed"
    wrong = "ab" * 32
    _register(name, f"sha256:{wrong}")
    asset = _asset(name)
    destination = tmp_path / "scene.zip"

    with pytest.raises(remote.RemoteIntegrityError) as raised:
        remote.download(asset, destination)

    assert raised.value.reason == "checksum_mismatch"
    assert not destination.exists()
    manifest = _manifest(destination, asset)
    assert manifest["verification"] == {
        "algorithm": "sha256",
        "checksum": wrong,
        "status": "failed",
        "size": len(PAYLOAD),
    }


def test_missing_provider_checksum_records_unverifiable(tmp_path: Path) -> None:
    """An asset without provider metadata records unverifiable with size only."""
    name = "p0055-unverifiable"
    _register(name, None)
    asset = _asset(name)
    destination = tmp_path / "scene.zip"

    remote.download(asset, destination)

    manifest = _manifest(destination, asset)
    assert manifest["verification"] == {
        "status": "unverifiable",
        "size": len(PAYLOAD),
    }


def test_extended_manifest_keeps_destination_reuse(tmp_path: Path) -> None:
    """Reuse works with the verification record and with legacy manifests."""
    name = "p0055-reuse"
    _register(name, f"md5:{hashlib.md5(PAYLOAD).hexdigest()}")
    asset = _asset(name)
    destination = tmp_path / "scene.zip"

    remote.download(asset, destination)
    assert remote.download(asset, destination) == destination

    manifest_path = remote._manifest_path(destination, asset)
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("verification")
    manifest_path.write_text(json.dumps(manifest))
    assert remote.download(asset, destination) == destination

    destination.write_bytes(b"corrupted")
    with pytest.raises(remote.RemoteIntegrityError, match="destination_conflict"):
        remote.download(asset, destination)


class _ChecksumAdapter:
    """Small adapter fixture for the direct streaming-contract test."""

    provider = "checksum-provider"
    origins = ("https://source.invalid",)
    path_prefixes = ("/assets",)
    redirect_origins = ()
    profiles = ("anonymous",)

    def items(self) -> list[object]:
        """Return no catalog records."""
        return []

    def fetch(self, asset: object, budget: object) -> bytes:
        """Return the fixture bytes without consulting the asset."""
        del asset, budget
        return PAYLOAD


def test_stream_download_keeps_pair_contract_on_mismatch(tmp_path: Path) -> None:
    """_stream_download still returns size/digest and raises on mismatch."""
    from faninsar.remote.transfer import _stream_download

    name = "p0055-stream-contract"
    remote._register_adapter(name, _ChecksumAdapter())
    asset = remote.RemoteAsset(
        provider="checksum-provider",
        catalog=name,
        collection=None,
        item_id="item-1",
        key="data",
        href="https://source.invalid/assets/data",
        checksum=f"md5:{'00' * 16}",
    )
    ledger = remote._CallLedger(remote.RemoteResourceBudget())
    staging = tmp_path / "staging.bin"

    with pytest.raises(remote.RemoteIntegrityError) as raised:
        _stream_download(asset, _ChecksumAdapter(), ledger.budget, ledger, staging)

    assert raised.value.reason == "checksum_mismatch"
    assert staging.read_bytes() == PAYLOAD


def test_read_verification_returns_record_from_qualified_sibling(
    tmp_path: Path,
) -> None:
    """The manifest reader returns the verification record for one input."""
    from faninsar.remote.transfer import read_verification

    destination = tmp_path / "S1A.zip"
    destination.write_bytes(b"payload")
    manifest = {
        "identity": "x",
        "sha256": "ab" * 32,
        "size": 7,
        "verification": {
            "algorithm": "md5",
            "checksum": "d4" * 16,
            "status": "verified",
            "size": 7,
        },
    }
    (tmp_path / "S1A.zip.faninsar.remote.abc123.json").write_text(json.dumps(manifest))

    record = read_verification(destination)

    assert record == {
        "algorithm": "md5",
        "checksum": "d4" * 16,
        "status": "verified",
        "size": 7,
    }
    assert read_verification(tmp_path / "absent.zip") is None
