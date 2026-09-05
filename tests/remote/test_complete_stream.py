"""Complete-file streaming and publication tests."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
import requests

from faninsar import remote
from faninsar.data.query import Points

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class _ChunkAdapter:
    """Fixture adapter exposing a one-pass chunk stream."""

    provider = "chunks"
    origins = ("https://chunks.invalid",)
    path_prefixes = ("/",)
    redirect_origins = ()
    profiles = ("anonymous",)

    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.consumed = False

    def items(self) -> Iterable[Mapping[str, Any]]:
        payload = b"".join(self.chunks)
        return (
            {
                "id": "chunk-item",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
                "assets": {
                    "data": {
                        "href": "https://chunks.invalid/data.bin",
                        "size": len(payload),
                        "checksum": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                    }
                },
            },
        )

    def fetch(
        self, asset: remote.RemoteAsset, budget: remote.RemoteResourceBudget
    ) -> Iterable[bytes]:
        del asset, budget
        assert not self.consumed
        self.consumed = True
        return iter(self.chunks)


def _asset(adapter_name: str) -> remote.RemoteAsset:
    return remote.search(Points([(0.5, 0.5)], crs=4326), catalog=adapter_name)[
        0
    ].assets["data"]


def test_complete_download_streams_chunks_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    """A chunk stream is written, checked, and atomically published."""
    adapter = _ChunkAdapter((b"first-", b"second-", b"third"))
    remote._register_adapter("chunks", adapter)
    destination = tmp_path / "nested" / "data.bin"

    path = remote.download(_asset("chunks"), destination)

    assert path == destination
    assert destination.read_bytes() == b"first-second-third"
    assert not list(destination.parent.glob(f".{destination.name}.*"))
    assert adapter.consumed


def test_stream_length_failure_leaves_destination_unchanged(tmp_path: Path) -> None:
    """Length validation happens before replacing an existing destination."""
    adapter = _ChunkAdapter((b"short",))
    remote._register_adapter("short", adapter)
    asset = _asset("short")
    destination = tmp_path / "data.bin"
    destination.write_bytes(b"old content")
    object.__setattr__(asset, "size_bytes", 99)

    with pytest.raises(remote.RemoteIntegrityError, match="content_length_mismatch"):
        remote.download(asset, destination, overwrite=True)

    assert destination.read_bytes() == b"old content"
    assert not list(tmp_path.glob(f".{destination.name}.*"))


@dataclass
class _ASFResponse:
    """Minimal streaming response for the ASF authentication fixture."""

    url: str
    status_code: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)

    def iter_content(self, chunk_size: int) -> Iterable[bytes]:
        """Yield the body in bounded chunks."""
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def close(self) -> None:
        """Close the fixture response."""


class _ASFSession:
    """Operation-scoped ASF session fixture with two redirect chains."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.cookies = requests.cookies.RequestsCookieJar()
        self.headers: dict[str, str] = {}
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        data: Any = None,
        auth: Any = None,
        allow_redirects: bool,
        stream: bool,
        timeout: tuple[float, float],
    ) -> _ASFResponse:
        """Return the next provider response without following redirects."""
        del data, auth, allow_redirects, stream, timeout
        self.requests.append((method, url, dict(headers)))
        if url.endswith("/api/users/find_or_create_token"):
            return _ASFResponse(url, 200, b'{"access_token":"fixture-token"}')
        if "/oauth/authorize" in url:
            return _ASFResponse(
                url,
                302,
                body=b"oauth-redirect",
                headers={"Location": "https://cumulus.asf.alaska.edu/login"},
            )
        if url == "https://cumulus.asf.alaska.edu/login":
            self.cookies.set("asf-urs", "fixture-cookie", domain=".asf.alaska.edu")
            return _ASFResponse(url, 200)
        if url.startswith("https://datapool.asf.alaska.edu/"):
            return _ASFResponse(
                url,
                307,
                body=b"data-redirect",
                headers={"Location": "https://sentinel1.asf.alaska.edu/SLC/item.zip"},
            )
        if url == "https://sentinel1.asf.alaska.edu/SLC/item.zip":
            return _ASFResponse(
                url,
                303,
                body=b"sentinel-redirect",
                headers={
                    "Location": (
                        "https://dy4owt9f80bz7.cloudfront.net/"
                        "s3-06b/asf-ngap2w-p-s1-slc-7b420b89."
                        "s3.us-west-2.amazonaws.com/item.zip?X-Amz-Signature=fixture"
                    )
                },
            )
        if url.startswith("https://dy4owt9f80bz7.cloudfront.net/"):
            return _ASFResponse(url, 200, self.payload)
        message = f"unexpected fixture URL: {url}"
        raise AssertionError(message)

    def close(self) -> None:
        """Record session cleanup."""
        self.closed = True


def test_asf_download_pre_authenticates_and_follows_trusted_redirects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ASF transfer gets a bearer/cookie before the complete SAFE GET."""
    payload = b"PK\x03\x04complete-safe-zip"
    session = _ASFSession(payload)
    monkeypatch.setattr(remote.requests, "Session", lambda: session)
    monkeypatch.setattr(
        remote,
        "_netrc_credentials",
        lambda host: (
            ("fixture-user", "fixture-password")
            if host == "urs.earthdata.nasa.gov"
            else None
        ),
    )

    class _ASFAdapter:
        provider = "ASF"
        origins = ("https://datapool.asf.alaska.edu",)
        path_prefixes = ("/",)
        redirect_origins = (
            "https://datapool.asf.alaska.edu",
            "https://sentinel1.asf.alaska.edu",
            "https://urs.earthdata.nasa.gov",
            "https://cumulus.asf.alaska.edu",
        )
        profiles = ("earthdata-asf",)

    adapter = _ASFAdapter()
    asset = remote.RemoteAsset(
        "ASF",
        "asf-live",
        "sentinel-1",
        "S1_FIXTURE",
        "data",
        "https://datapool.asf.alaska.edu/SLC/item.zip",
        size_bytes=len(payload),
        auth_profile="earthdata-asf",
    )
    staging = tmp_path / "staging.zip"
    ledger = remote._CallLedger(remote.RemoteResourceBudget(max_redirects=4))

    size, digest = remote._stream_download(
        asset, adapter, ledger.budget, ledger, staging
    )

    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert staging.read_bytes() == payload
    assert ledger.requests == 6
    assert ledger.redirects == 3
    assert ledger.response_bytes_total == (
        len(b'{"access_token":"fixture-token"}')
        + len(b"oauth-redirect")
        + len(b"data-redirect")
        + len(b"sentinel-redirect")
        + len(payload)
    )
    assert "Authorization" not in session.requests[2][2]
    assert session.requests[4][2]["Authorization"] == "Bearer fixture-token"
    assert session.requests[-2][2]["Authorization"] == "Bearer fixture-token"
    assert "Authorization" not in session.requests[-1][2]
    assert "X-Amz-Signature=fixture" in session.requests[-1][1]
    assert session.closed


class _LPDAACSession:
    """Fixture for LPDAAC's data -> URS -> data -> CloudFront flow."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.cookies = requests.cookies.RequestsCookieJar()
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        data: Any = None,
        auth: Any = None,
        allow_redirects: bool,
        stream: bool,
        timeout: tuple[float, float],
    ) -> _ASFResponse:
        """Return one response for each explicitly followed request."""
        del data, auth, allow_redirects, stream, timeout
        self.requests.append((method, url, dict(headers)))
        if len(self.requests) == 1:
            return _ASFResponse(
                url,
                302,
                body=b"data-redirect",
                headers={
                    "Location": (
                        "https://urs.earthdata.nasa.gov/oauth/authorize?"
                        "client_id=fixture&state=object"
                    )
                },
            )
        if url.endswith("NASADEM_HGT_n36w121.zip"):
            return _ASFResponse(
                url,
                302,
                body=b"object-redirect",
                headers={
                    "Location": (
                        "https://d123example.cloudfront.net/"
                        "s3-0123456789abcdef0123456789abcdef/"
                        "lp-prod-protected.s3.us-west-2.amazonaws.com/"
                        "NASADEM_HGT.001/NASADEM_HGT_n36w121/"
                        "NASADEM_HGT_n36w121.zip?X-Amz-Signature=fixture"
                    )
                },
            )
        if url.startswith("https://urs.earthdata.nasa.gov/oauth/"):
            return _ASFResponse(
                url,
                302,
                body=b"urs-redirect",
                headers={
                    "Location": (
                        "https://data.lpdaac.earthdatacloud.nasa.gov/login?"
                        "code=fixture"
                    )
                },
            )
        if url.startswith("https://data.lpdaac.earthdatacloud.nasa.gov/login"):
            return _ASFResponse(
                url,
                302,
                body=b"login-redirect",
                headers={
                    "Location": (
                        "https://data.lpdaac.earthdatacloud.nasa.gov/"
                        "lp-prod-protected/NASADEM_HGT.001/"
                        "NASADEM_HGT_n36w121/NASADEM_HGT_n36w121.zip"
                    )
                },
            )
        if url.startswith("https://d123example.cloudfront.net/"):
            return _ASFResponse(
                url,
                200,
                self.payload,
                headers={"Content-Encoding": "identity"},
            )
        message = f"unexpected fixture URL: {url}"
        raise AssertionError(message)

    def close(self) -> None:
        """Record session cleanup."""
        self.closed = True


def test_lpdaac_download_meters_redirects_and_strips_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """LPDAAC auth credentials stop at URS and never reach CloudFront."""
    payload = b"PK\x03\x04nasadem-zip"
    session = _LPDAACSession(payload)
    monkeypatch.setattr(remote.requests, "Session", lambda: session)
    monkeypatch.setattr(
        remote,
        "_netrc_credentials",
        lambda host: (
            ("fixture-user", "fixture-password")
            if host == "urs.earthdata.nasa.gov"
            else None
        ),
    )

    class _LPDAACAdapter:
        provider = "LPCLOUD"
        origins = (
            "https://data.lpdaac.earthdatacloud.nasa.gov",
        )
        path_prefixes = ("/",)
        redirect_origins = (
            "https://data.lpdaac.earthdatacloud.nasa.gov",
            "https://urs.earthdata.nasa.gov",
        )
        profiles = ("earthdata-lpdaac",)

    source = (
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
        "NASADEM_HGT.001/NASADEM_HGT_n36w121/NASADEM_HGT_n36w121.zip"
    )
    asset = remote.RemoteAsset(
        "LPCLOUD",
        "lpdaac-live",
        "NASADEM",
        "G2816843744-LPCLOUD",
        "data",
        source,
        size_bytes=len(payload),
        auth_profile="earthdata-lpdaac",
    )
    adapter = _LPDAACAdapter()
    ledger = remote._CallLedger(remote.RemoteResourceBudget(max_redirects=5))
    staging = tmp_path / "NASADEM_HGT_n36w121.zip"

    size, digest = remote._stream_download(
        asset, adapter, ledger.budget, ledger, staging
    )

    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert ledger.requests == 5, [item[1] for item in session.requests]
    assert ledger.redirects == 4
    assert ledger.response_bytes_total == len(payload) + sum(
        len(body)
        for body in (
            b"data-redirect",
            b"urs-redirect",
            b"login-redirect",
            b"object-redirect",
        )
    )
    urs_headers = session.requests[1][2]
    assert urs_headers["Authorization"].startswith("Basic ")
    assert "Authorization" not in session.requests[-1][2]
    assert "Cookie" not in session.requests[-1][2]
    assert "X-Amz-Signature=fixture" in session.requests[-1][1]
    assert session.closed


def test_asf_anonymous_delivery_stays_sticky_across_unknown_hosts() -> None:
    """Credentials stay stripped through a CDN and object-host handoff."""

    class _AnonymousChainSession:
        def __init__(self) -> None:
            self.cookies = requests.cookies.RequestsCookieJar()
            self.cookies.set("session", "secret")
            self.requests: list[tuple[str, str, dict[str, str], object]] = []

        def request(
            self,
            method: str,
            url: str,
            *,
            headers: dict[str, str],
            data: Any = None,
            auth: Any = None,
            allow_redirects: bool,
            stream: bool,
            timeout: tuple[float, float],
        ) -> _ASFResponse:
            del data, allow_redirects, stream, timeout
            self.requests.append((method, url, dict(headers), auth))
            if url == "https://datapool.asf.alaska.edu/item.zip":
                return _ASFResponse(
                    url,
                    302,
                    body=b"gateway",
                    headers={"Location": "https://cdn.unknown.example/item.zip"},
                )
            if url == "https://cdn.unknown.example/item.zip":
                return _ASFResponse(
                    url,
                    302,
                    body=b"cdn",
                    headers={"Location": "https://objects.unknown.example/item.zip"},
                )
            if url == "https://objects.unknown.example/item.zip":
                return _ASFResponse(url, 200, body=b"payload")
            message = f"unexpected fixture URL: {url}"
            raise AssertionError(message)

        def close(self) -> None:
            return None

    class _ASFAdapter:
        provider = "ASF"
        origins = ("https://datapool.asf.alaska.edu",)
        path_prefixes = ("/",)
        redirect_origins = ("https://datapool.asf.alaska.edu",)
        profiles = ("earthdata-asf",)

    session = _AnonymousChainSession()
    asset = remote.RemoteAsset(
        "ASF",
        "asf-anonymous-chain",
        "sentinel-1",
        "G3964549387-ASF",
        "data",
        "https://datapool.asf.alaska.edu/item.zip",
        auth_profile="earthdata-asf",
    )
    ledger = remote._CallLedger(remote.RemoteResourceBudget(max_redirects=3))
    response = remote._asf_request(
        session,
        "GET",
        asset.href,
        adapter=_ASFAdapter(),
        asset=asset,
        budget=ledger.budget,
        ledger=ledger,
        headers={"Authorization": "Bearer fixture", "Cookie": "session=secret"},
    )

    assert response.status_code == 200
    assert len(session.requests) == 3
    for _method, _url, headers, auth in session.requests[1:]:
        assert "Authorization" not in headers
        assert "Cookie" not in headers
        assert auth is remote._no_auth


def test_lpdaac_anonymous_handoff_admits_changed_delivery_origins() -> None:
    """A trusted LPDAAC gateway may hand off to any safe HTTPS object URL."""

    class _LPDAACAdapter:
        origins = ("https://data.lpdaac.earthdatacloud.nasa.gov",)
        path_prefixes = ("/",)
        redirect_origins = (
            "https://data.lpdaac.earthdatacloud.nasa.gov",
            "https://urs.earthdata.nasa.gov",
        )

    source = (
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
        "NASADEM_HGT.001/NASADEM_HGT_n36w121/NASADEM_HGT_n36w121.zip"
    )
    asset = remote.RemoteAsset(
        "LPCLOUD",
        "lpdaac-live",
        "NASADEM",
        "G2816843744-LPCLOUD",
        "data",
        source,
        auth_profile="earthdata-lpdaac",
    )
    valid = (
        "https://d123example.cloudfront.net/"
        "s3-0123456789abcdef0123456789abcdef/"
        "lp-prod-protected.s3.us-west-2.amazonaws.com/"
        "NASADEM_HGT.001/NASADEM_HGT_n36w121/NASADEM_HGT_n36w121.zip"
    )
    assert remote._lpdaac_redirect_url(source, valid, asset, _LPDAACAdapter()) == valid
    for invalid in (
        valid.replace("https://d123example.cloudfront.net", "http://evil.example"),
        valid.replace("/s3-012", "/../"),
        valid.replace("d123example", "evil.example") + "#fragment",
    ):
        with pytest.raises(remote.RemoteAccessError):
            remote._lpdaac_redirect_url(source, invalid, asset, _LPDAACAdapter())


def test_asf_anonymous_handoff_admits_changed_delivery_origins() -> None:
    """A trusted ASF gateway may hand off to any safe HTTPS object URL."""

    class _ASFAdapter:
        origins = ("https://datapool.asf.alaska.edu",)
        path_prefixes = ("/",)
        redirect_origins = ("https://sentinel1.asf.alaska.edu",)

    asset = remote.RemoteAsset(
        "ASF",
        "asf-live",
        "sentinel-1",
        "G3964549387-ASF",
        "data",
        "https://datapool.asf.alaska.edu/SLC/item.zip",
        auth_profile="earthdata-asf",
    )
    valid = (
        "https://dy4owt9f80bz7.cloudfront.net/s3-06b/"
        "asf-ngap2w-p-s1-slc-7b420b89.s3.us-west-2.amazonaws.com/item.zip"
    )

    assert (
        remote._asf_redirect_url(
            "https://sentinel1.asf.alaska.edu/SLC/item.zip",
            valid,
            asset,
            _ASFAdapter(),
        )
        == valid
    )
    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        remote._asf_redirect_url(
            "https://datapool.asf.alaska.edu/SLC/item.zip",
            valid,
            asset,
            _ASFAdapter(),
        )
    changed = "https://objects.example/new-layout/other.zip"
    assert (
        remote._asf_redirect_url(
            "https://sentinel1.asf.alaska.edu/SLC/item.zip",
            changed,
            asset,
            _ASFAdapter(),
        )
        == changed
    )


def test_asf_nisar_anonymous_handoff_admits_changed_delivery_origins() -> None:
    """A trusted NISAR gateway may hand off to a changed object layout."""

    class _ASFAdapter:
        origins = ("https://nisar.asf.earthdatacloud.nasa.gov",)
        path_prefixes = ("/",)
        redirect_origins = ("https://nisar.asf.earthdatacloud.nasa.gov",)

    source = (
        "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/DEM/v1.2/"
        "EPSG4326/S90/S90_W180/DEM_S90_00_W180_00_C01.tif"
    )
    asset = remote.RemoteAsset(
        "ASF",
        "asf-live",
        "NISAR_DEM",
        "S90",
        "data",
        source,
        auth_profile="earthdata-asf",
    )
    valid = (
        "https://d1mv8zhcvry6x4.cloudfront.net/s3-7fdf/"
        "sds-n-cumulus-prod-nisar-products.s3.us-west-2.amazonaws.com/"
        "DEM/v1.2/EPSG4326/S90/S90_W180/DEM_S90_00_W180_00_C01.tif"
    )
    assert remote._asf_redirect_url(source, valid, asset, _ASFAdapter()) == valid
    changed = "https://objects.example/new-layout/dem.tif"
    assert remote._asf_redirect_url(source, changed, asset, _ASFAdapter()) == changed


@pytest.mark.parametrize(
    ("status", "headers", "reason"),
    [
        (206, {}, "partial_content"),
        (200, {"Content-Range": "bytes 0-3/4"}, "partial_content"),
        (404, {}, "unexpected_status"),
    ],
)
def test_complete_download_rejects_partial_or_non_success_responses(
    status: int, headers: dict[str, str], reason: str
) -> None:
    """Complete-file publication admits only an un-ranged HTTP 200."""
    response = type("Response", (), {"status_code": status, "headers": headers})()
    with pytest.raises(remote.RemoteAccessError, match=reason):
        remote._response_is_complete(response)


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        ("zip", b"PK\x03\x04payload"),
        ("zip", b"PK\x05\x06payload"),
        ("zip", b"PK\x07\x08payload"),
        ("tif", b"II*\x00payload"),
        ("tif", b"MM\x00*payload"),
        ("tif", b"II+\x00payload"),
        ("tif", b"MM\x00+payload"),
        ("h5", b"\x89HDF\r\n\x1a\npayload"),
    ],
)
def test_representation_magic_is_checked_before_publication(
    suffix: str, payload: bytes, tmp_path: Path
) -> None:
    """Known binary suffixes require their representation magic."""
    staging = tmp_path / f"asset.{suffix}"
    asset = remote.RemoteAsset(
        "fixture",
        "fixture",
        None,
        "item",
        "data",
        f"https://fixture.invalid/asset.{suffix}",
    )
    staging.write_bytes(payload)
    remote._validate_representation(staging, asset)

    staging.write_bytes(b"\xef\xbb\xbf  <HTML>login</HTML>")
    with pytest.raises(remote.RemoteIntegrityError, match="representation_mismatch"):
        remote._validate_representation(staging, asset)


def test_signed_query_redirect_is_terminal() -> None:
    """A request carrying a delivery capability cannot follow a 3xx."""
    adapter = _ChunkAdapter((b"payload",))
    handler = remote._RedirectHandler(
        adapter,
        remote.RemoteResourceBudget(max_redirects=1),
    )
    request = remote.urllib.request.Request(
        "https://chunks.invalid/data.bin?x-random-signature=fixture-secret"
    )
    response = io.BytesIO(b"redirect")
    response.headers = {"Location": "https://objects.example/data.bin"}  # type: ignore[attr-defined]
    with pytest.raises(remote.RemoteAccessError, match="signed_redirect"):
        handler.redirect_request(
            request,
            response,
            302,
            "Found",
            response.headers,
            response.headers["Location"],
        )


def test_transfer_failure_does_not_surface_signed_url(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Transport exception text and unknown query values stay private."""

    class _FailingAdapter:
        provider = "failing"
        origins = ("https://failing.invalid",)
        path_prefixes = ("/",)
        redirect_origins = ()
        profiles = ("anonymous",)

        def fetch(self, asset: object, budget: object) -> bytes:
            del asset, budget
            message = (
                "request failed at https://objects.example/file.bin?"
                "x-random-signature=fixture-secret"
            )
            raise OSError(message)

    remote._register_adapter("failing-p0047", _FailingAdapter())
    asset = remote.RemoteAsset(
        "failing",
        "failing-p0047",
        None,
        "item",
        "data",
        "https://failing.invalid/file.bin?x-random-signature=fixture-secret",
    )
    with caplog.at_level("WARNING"), pytest.raises(
        remote.RemoteAccessError, match="transfer_failed"
    ) as error:
        remote.download(
            asset,
            tmp_path / "file.bin",
            budget=remote.RemoteResourceBudget(max_retries=1),
        )
    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in caplog.text
    assert "objects.example" not in caplog.text
