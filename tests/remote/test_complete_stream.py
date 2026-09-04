"""Complete-file streaming and publication tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
import requests

from faninsar import remote
from faninsar.query import Points

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
    payload = b"complete-safe-zip"
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
    payload = b"nasadem-zip"
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


def test_lpdaac_cloudfront_handoff_is_bound_to_exact_object() -> None:
    """An LPDAAC CDN hostname cannot redirect to another bucket or object."""

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
        valid.replace("lp-prod-protected.s3", "other-bucket.s3"),
        valid.replace("NASADEM_HGT_n36w121.zip", "other.zip"),
        valid.replace("d123example", "evil.example"),
    ):
        with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
            remote._lpdaac_redirect_url(source, invalid, asset, _LPDAACAdapter())


def test_asf_cloudfront_handoff_is_bound_to_source_bucket_and_filename() -> None:
    """A dynamic CDN host is not a general redirect wildcard."""

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
    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        remote._asf_redirect_url(
            "https://sentinel1.asf.alaska.edu/SLC/item.zip",
            valid.replace("item.zip", "other.zip"),
            asset,
            _ASFAdapter(),
        )


def test_asf_nisar_cloudfront_handoff_is_bound_to_source_path() -> None:
    """The observed NISAR CDN handoff cannot be reused for another object."""

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
    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        remote._asf_redirect_url(
            source,
            valid.replace("/DEM/", "/NISAR/DEM/"),
            asset,
            _ASFAdapter(),
        )
    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        remote._asf_redirect_url(
            source, valid.replace("S90_W180", "S90_W179"), asset, _ASFAdapter()
        )
