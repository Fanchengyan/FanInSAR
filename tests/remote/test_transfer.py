"""Download-time verification recording (PROPOSAL-0055 slice 2).

Burst SAFE post-unzip ``manifest.safe`` member verification is
PROPOSAL-0060 seam 4: a pure manifest parser and a member-verify function
over the written zip, wired into the burst download/publication path.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
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


# --- PROPOSAL-0060 seam 4: burst SAFE post-unzip manifest.safe members ---


def test_parse_safe_manifest_returns_file_location_md5_pairs() -> None:
    """A pure manifest.safe parser returns href -> MD5 for checksummed members.

    The expected MD5 values are independent known-good digests of the literal
    member bytes (not recomputed by the code under test), so the assertion can
    disagree with a broken parser.
    """
    from faninsar.remote.transfer import parse_safe_manifest_checksums

    manifest = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1">\n'
        "  <xfdu:dataObjectSection>\n"
        '    <xfdu:dataObject ID="measurement-a">\n'
        "      <xfdu:byteStream>\n"
        '        <xfdu:fileLocation href="./measurement-a.tiff"/>\n'
        '        <xfdu:checksum checksumName="MD5">'
        "4ae69c457a6534ddaf0f511c69a37ca6</xfdu:checksum>\n"
        "      </xfdu:byteStream>\n"
        "    </xfdu:dataObject>\n"
        '    <xfdu:dataObject ID="measurement-b">\n'
        "      <xfdu:byteStream>\n"
        '        <xfdu:fileLocation href="./measurement-b.tiff"/>\n'
        '        <xfdu:checksum checksumName="SHA-1">deadbeef</xfdu:checksum>\n'
        "      </xfdu:byteStream>\n"
        "    </xfdu:dataObject>\n"
        "  </xfdu:dataObjectSection>\n"
        "</xfdu:XFDU>\n"
    )

    pairs = parse_safe_manifest_checksums(manifest)

    assert pairs == {"measurement-a.tiff": "4ae69c457a6534ddaf0f511c69a37ca6"}


def _write_safe_zip(path: Path, members: dict[str, bytes], manifest: str) -> None:
    """Write a tiny SAFE zip under the repository tmp/ tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.safe", manifest)


def _manifest_for(members: dict[str, bytes]) -> str:
    """Return a manifest.safe declaring the MD5 of each member (fixture only)."""
    blocks = []
    for name, payload in members.items():
        md5 = hashlib.md5(payload).hexdigest()
        blocks.append(
            "<xfdu:byteStream>"
            f'<xfdu:fileLocation href="./{name}"/>'
            f'<xfdu:checksum checksumName="MD5">{md5}</xfdu:checksum>'
            "</xfdu:byteStream>"
        )
    return (
        '<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1">'
        + "".join(blocks)
        + "</xfdu:XFDU>"
    )


def test_member_verify_reports_matched_summary_for_all_match(tmp_path: Path) -> None:
    """A SAFE whose members match the manifest reports checked/matched/verified."""
    from faninsar.remote.transfer import verify_safe_members

    members = {"measurement-a.tiff": b"tiff-a", "measurement-b.tiff": b"tiff-b"}
    safe_zip = tmp_path / "burst.zip"
    _write_safe_zip(safe_zip, members, _manifest_for(members))

    summary, all_matched = verify_safe_members(safe_zip)

    assert all_matched is True
    assert summary == {"checked": 2, "matched": 2, "status": "verified"}


def _safe_zip_bytes(members: dict[str, bytes], manifest: str) -> bytes:
    """Return the bytes of a tiny SAFE zip (in-memory fixture)."""
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.safe", manifest)
    return buffer.getvalue()


def _register_safe(name: str, payload: bytes, checksum: str | None = None) -> None:
    """Register a one-asset fixture catalog whose data payload is a SAFE zip."""
    asset: dict[object, object] = {
        "href": "https://fixture.invalid/assets/burst.zip",
        "data": payload,
        "size": len(payload),
    }
    if checksum is not None:
        asset["checksum"] = checksum
    remote._register_fixture(
        [
            {
                "id": "burst-1",
                "collection": "slc-burst",
                "geometry": Polygon(((-1, -1), (1, -1), (1, 1), (-1, 1))),
                "assets": {"data": asset},
            }
        ],
        name=name,
    )


def test_safe_all_match_gains_members_fact_container_unverifiable(
    tmp_path: Path,
) -> None:
    """A SAFE zip whose members match keeps the container unverifiable.

    The container gains an additive members sub-record.
    """
    name = "p0060-safe-match"
    members = {"measurement-a.tiff": b"tiff-a", "measurement-b.tiff": b"tiff-b"}
    payload = _safe_zip_bytes(members, _manifest_for(members))
    _register_safe(name, payload)
    asset = remote.search(Points([(0, 0)], crs=4326), catalog=name)[0].assets["data"]
    destination = tmp_path / "burst.zip"

    remote.download(asset, destination)

    manifest = json.loads(remote._manifest_path(destination, asset).read_text())
    verification = manifest["verification"]
    assert verification["status"] == "unverifiable"
    assert verification["size"] == len(payload)
    assert verification["members"] == {
        "checked": 2,
        "matched": 2,
        "status": "verified",
    }
    assert "checksum" not in verification


def test_safe_corrupted_member_records_failed_and_excludes(tmp_path: Path) -> None:
    """A SAFE member that mismatches its manifest MD5 records failed and raises."""
    name = "p0060-safe-corrupt"
    good = {"measurement-a.tiff": b"tiff-a", "measurement-b.tiff": b"tiff-b"}
    # The manifest declares the MD5 of the original bytes, but the zip carries
    # a corrupted measurement-a.tiff so the content check must fail.
    manifest = _manifest_for(good)
    corrupted = dict(good)
    corrupted["measurement-a.tiff"] = b"CORRUPTED"
    payload = _safe_zip_bytes(corrupted, manifest)
    _register_safe(name, payload)
    asset = remote.search(Points([(0, 0)], crs=4326), catalog=name)[0].assets["data"]
    destination = tmp_path / "burst.zip"

    with pytest.raises(remote.RemoteIntegrityError) as raised:
        remote.download(asset, destination)

    assert raised.value.reason == "checksum_mismatch"
    assert not destination.exists()
    manifest_record = json.loads(remote._manifest_path(destination, asset).read_text())
    verification = manifest_record["verification"]
    assert verification["status"] == "failed"
    assert verification["members"] == {"checked": 2, "matched": 1, "status": "failed"}


def test_safe_without_manifest_checksums_has_no_members_record(
    tmp_path: Path,
) -> None:
    """A SAFE whose manifest declares no MD5 keeps the container unverifiable.

    It gains no members sub-record.
    """
    name = "p0060-safe-nomanifest"
    members = {"measurement-a.tiff": b"tiff-a", "annotation.xml": b"<xml/>"}
    # A manifest.safe that lists members but declares no checksumName="MD5".
    manifest = (
        '<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1">'
        "<xfdu:byteStream>"
        '<xfdu:fileLocation href="./measurement-a.tiff"/>'
        "</xfdu:byteStream>"
        "</xfdu:XFDU>"
    )
    payload = _safe_zip_bytes(members, manifest)
    _register_safe(name, payload)
    asset = remote.search(Points([(0, 0)], crs=4326), catalog=name)[0].assets["data"]
    destination = tmp_path / "burst.zip"

    remote.download(asset, destination)

    manifest_record = json.loads(remote._manifest_path(destination, asset).read_text())
    verification = manifest_record["verification"]
    assert verification["status"] == "unverifiable"
    assert verification["size"] == len(payload)
    assert "members" not in verification
    assert "checksum" not in verification
