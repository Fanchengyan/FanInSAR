"""Contract tests for the anonymous s1-orbits catalog (PROPOSAL-0054).

Also covers the PROPOSAL-0060 seam 1 digest extraction: the s1-orbits bucket
publishes its MD5 as the S3 ``ETag`` of single-part uploads.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from faninsar import remote
from faninsar.missions.s1.search import ensure_asf_catalog
from faninsar.remote.catalog import _ADAPTERS
from faninsar.remote.fetch.complete import read_verification
from faninsar.remote.providers.s1_orbits import (
    S1_ORBITS_CATALOG,
    S1_ORBITS_ORIGIN,
    _etag_to_checksum,
    ensure_s1_orbits_catalog,
)

# Independent known-good MD5 digest of b"" (RFC 1321 test-suite value).
_EMPTY_MD5 = "d41d8cd98f00b204e9800998ecf8427e"


def test_etag_to_checksum_single_part_etag_is_md5() -> None:
    """A single-part S3 ETag (32 hex, no -N suffix) yields md5:<etag>."""
    assert _etag_to_checksum(_EMPTY_MD5) == f"md5:{_EMPTY_MD5}"


def test_etag_to_checksum_multipart_etag_is_unverifiable() -> None:
    """A multipart S3 ETag is never treated as an MD5 digest."""
    assert _etag_to_checksum(f"{_EMPTY_MD5}-21") is None


def test_etag_to_checksum_missing_and_malformed_are_unverifiable() -> None:
    """Absent, empty, quoted, short, and non-hex ETags yield no checksum."""
    assert _etag_to_checksum(None) is None
    assert _etag_to_checksum("") is None
    assert _etag_to_checksum(f'"{_EMPTY_MD5}"') is None
    assert _etag_to_checksum(_EMPTY_MD5[:-1]) is None
    assert _etag_to_checksum("z" * 32) is None


class _FakeOrbitResponse:
    """Minimal urllib response carrying one ETag header."""

    def __init__(self, payload: bytes, etag: str | None) -> None:
        self._payload = payload
        self.status = 200
        self.headers: dict[str, str] = {"Content-Encoding": "identity"}
        if etag is not None:
            self.headers["ETag"] = etag

    def read(self, _size: int = -1) -> bytes:
        """Return the remaining fixture bytes."""
        payload, self._payload = self._payload, b""
        return payload

    def __enter__(self) -> _FakeOrbitResponse:
        """Enter the response context."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Exit the response context."""
        return None


class _FakeOrbitOpener:
    """Opener serving one object GET with a mocked ETag header."""

    def __init__(self, payload: bytes, etag: str | None) -> None:
        self.payload = payload
        self.etag = etag

    def open(self, request: Any, timeout: float | None = None) -> _FakeOrbitResponse:
        """Serve the EOF object bytes plus its mocked ETag header."""
        del request, timeout
        return _FakeOrbitResponse(self.payload, self.etag)


def _download_eof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: bytes,
    etag: str | None,
) -> dict[str, object]:
    """Download one EOF object through the public seam and return its record."""
    ensure_s1_orbits_catalog()
    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *_handlers: _FakeOrbitOpener(payload, etag),
    )
    asset = remote.RemoteAsset(
        provider="s1-orbits",
        catalog=S1_ORBITS_CATALOG,
        collection="AUX_POEORB",
        item_id="eof",
        key="eof",
        href=f"{S1_ORBITS_ORIGIN}/AUX_POEORB/S1A_OPER_AUX_POEORB_TEST.EOF",
        auth_profile="anonymous",
    )
    destination = tmp_path / "orbit.EOF"
    remote.download(asset, destination)
    record = read_verification(destination)
    assert record is not None
    return record


def test_single_part_etag_is_verified_against_md5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A single-part ETag drives a verified md5 comparison at download time.

    The expected digest is the independent MD5 of the literal fixture bytes
    (not derived from the ETag by the code under test), so the assertion can
    disagree with a broken comparison.
    """
    payload = b"orbit fixture bytes"
    md5 = hashlib.md5(payload).hexdigest()

    record = _download_eof(monkeypatch, tmp_path, payload, md5)

    assert record == {
        "algorithm": "md5",
        "checksum": md5,
        "status": "verified",
        "size": len(payload),
    }


def test_multipart_etag_stays_unverifiable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A multipart ETag never drives a checksum comparison."""
    payload = b"orbit fixture bytes"

    record = _download_eof(monkeypatch, tmp_path, payload, f"{_EMPTY_MD5}-21")

    assert record == {"status": "unverifiable", "size": len(payload)}


def test_s1_orbits_catalog_is_anonymous_only() -> None:
    """The dedicated origin admits anonymous and rejects scene Earthdata."""
    name = ensure_s1_orbits_catalog()
    adapter = _ADAPTERS[name]
    assert name == S1_ORBITS_CATALOG
    assert adapter.profiles == ("anonymous",)
    assert S1_ORBITS_ORIGIN in adapter.origins
    assert getattr(adapter, "_allow_anonymous_delivery", False) is False
    asset = remote.RemoteAsset(
        provider="s1-orbits",
        catalog=name,
        collection="AUX_POEORB",
        item_id="eof",
        key="eof",
        href=f"{S1_ORBITS_ORIGIN}/AUX_POEORB/S1A_OPER_AUX_POEORB_TEST.EOF",
        auth_profile="earthdata-asf",
    )
    with pytest.raises(remote.RemoteAccessError, match="unknown_auth_profile"):
        remote.download(asset, destination=__import__("pathlib").Path("unused.EOF"))


def test_s1_orbits_origin_is_not_added_to_sentinel1_asf() -> None:
    """Scene ASF registration must not inherit the public orbit bucket."""
    ensure_asf_catalog(product_type="burst")
    ensure_s1_orbits_catalog()
    asf = _ADAPTERS["sentinel1-asf"]
    origins = {str(origin).rstrip("/") for origin in asf.origins}
    redirect = {str(origin).rstrip("/") for origin in asf.redirect_origins}
    assert S1_ORBITS_ORIGIN not in origins
    assert S1_ORBITS_ORIGIN not in redirect
    assert "s1qc.asf.alaska.edu" not in " ".join(origins | redirect)


def test_s1_orbits_unknown_profile_fails_closed() -> None:
    """Copying the Stack scene profile onto this catalog is rejected."""
    name = ensure_s1_orbits_catalog()
    adapter = _ADAPTERS[name]
    assert "earthdata-asf" not in adapter.profiles
    assert "sentinel1-asf" not in {adapter.provider, name}
