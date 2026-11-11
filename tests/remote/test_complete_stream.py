"""Complete-file streaming and publication tests."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass, field
from email.message import Message
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self

import pytest
import requests

from faninsar import remote
from faninsar.data.query import Points
from faninsar.remote.access import (
    _asf_redirect_url,
    _lpdaac_redirect_url,
    _response_is_complete,
    _validate_representation,
)
from faninsar.remote.auth import _asf_request, _no_auth
from faninsar.remote.cache import _identity
from faninsar.remote.catalog import _register_adapter
from faninsar.remote.fetch.complete import (
    _part_path,
    _part_state_path,
    _PartState,
    _read_part_state,
    _RedirectHandler,
    _stream_download,
    _write_part_state,
    download,
    probe,
    read_verification,
)
from faninsar.remote.protocols import _CallLedger

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
    _register_adapter("chunks", adapter)
    destination = tmp_path / "nested" / "data.bin"

    path = remote.download(_asset("chunks"), destination)

    assert path == destination
    assert destination.read_bytes() == b"first-second-third"
    assert not list(destination.parent.glob(f".{destination.name}.*"))
    assert adapter.consumed


def test_stream_length_failure_leaves_destination_unchanged(tmp_path: Path) -> None:
    """Length validation happens before replacing an existing destination."""
    adapter = _ChunkAdapter((b"short",))
    _register_adapter("short", adapter)
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
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(
        "faninsar.remote.auth._netrc_credentials",
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
    ledger = _CallLedger(remote.RemoteResourceBudget(max_redirects=4))

    size, digest = _stream_download(asset, adapter, ledger.budget, ledger, staging)

    assert size == len(payload)
    assert digest == ""
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


def test_asf_burst_download_returns_pending_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Burst SAFE generation polls with the authenticated ASF request path."""

    class _Response:
        def __init__(
            self, status_code: int, *, location: str | None = None, body: bytes = b""
        ) -> None:
            self.status_code = status_code
            self.headers = {"Location": location} if location else {}
            self._body = body

        def iter_content(self, *, chunk_size: int) -> Iterable[bytes]:
            del chunk_size
            return iter((self._body,)) if self._body else iter(())

        def close(self) -> None:
            return None

    responses = iter(
        (
            _Response(
                202,
                location="https://sentinel1-burst.asf.alaska.edu/jobs/result.zip",
                body=b"pending",
            ),
        )
    )
    calls: list[str] = []

    class _Session:
        def close(self) -> None:
            return None

    def authenticated_session(*_args: Any) -> tuple[_Session, str]:
        return _Session(), "fixture-token"

    def asf_request(
        _session: _Session,
        _method: str,
        url: str,
        *,
        ledger: Any,
        **_kwargs: Any,
    ) -> _Response:
        calls.append(url)
        ledger.request()
        ledger.begin_response()
        return next(responses)

    monkeypatch.setattr(
        "faninsar.remote.fetch.complete._asf_authenticated_session",
        authenticated_session,
    )
    monkeypatch.setattr("faninsar.remote.fetch.complete._asf_request", asf_request)
    delays: list[float] = []
    monkeypatch.setattr("faninsar.remote.fetch.complete.time.sleep", delays.append)

    class _ASFAdapter:
        provider = "ASF"
        origins = ("https://sentinel1-burst.asf.alaska.edu",)
        path_prefixes = ("/",)
        redirect_origins = ("https://sentinel1-burst.asf.alaska.edu",)
        profiles = ("earthdata-asf",)

    asset = remote.RemoteAsset(
        "ASF",
        "asf-burst-transfer",
        "SLC-BURST",
        "burst-1",
        "data",
        "https://sentinel1-burst.asf.alaska.edu/jobs/request.zip",
        auth_profile="earthdata-asf",
    )
    ledger = _CallLedger(remote.RemoteResourceBudget(max_retries=2))
    staging = tmp_path / "burst.zip"

    with pytest.raises(remote.BurstPendingError) as raised:
        _stream_download(asset, _ASFAdapter(), ledger.budget, ledger, staging)

    assert raised.value.location.endswith("/jobs/result.zip")
    assert calls == [
        "https://sentinel1-burst.asf.alaska.edu/jobs/request.zip",
    ]
    assert ledger.requests == 1
    assert ledger.retries == 0
    assert ledger.response_bytes_total == len(b"pending")
    assert delays == []


class _BurstDeliveryAdapter:
    """Registered burst gateway without a delivery-host allowlist."""

    provider = "ASF"
    origins = ("https://sentinel1-burst.asf.alaska.edu",)
    path_prefixes = ("/",)
    redirect_origins = origins
    profiles = ("earthdata-asf",)


class _BurstDeliveryTransport(requests.adapters.BaseAdapter):
    """Capture real requests preparation without making network requests."""

    def __init__(self, responses: list[tuple[int, dict[str, str], bytes]]) -> None:
        self.responses = iter(responses)
        self.requests: list[requests.PreparedRequest] = []

    def send(
        self, request: requests.PreparedRequest, **_kwargs: Any
    ) -> requests.Response:
        self.requests.append(request)
        status, headers, body = next(self.responses)
        response = requests.Response()
        response.status_code = status
        response.url = request.url
        response.request = request
        response.headers.update(headers)
        response.raw = io.BytesIO(body)
        if "Set-Cookie" in headers:
            message = Message()
            message["Set-Cookie"] = headers["Set-Cookie"]
            response.raw._original_response = SimpleNamespace(msg=message)
        return response

    def close(self) -> None:
        return None


def _burst_delivery_asset(catalog: str) -> remote.RemoteAsset:
    return remote.RemoteAsset(
        "ASF",
        catalog,
        "SLC-BURST",
        "burst-1",
        "safe",
        "https://sentinel1-burst.asf.alaska.edu/jobs/request.zip",
        auth_profile="earthdata-asf",
    )


@pytest.mark.parametrize("status", [200, 202])
def test_asf_authenticated_probe_keeps_anonymous_delivery_handoff(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """Ready and pending observations preserve runtime URLs after gateway redirects."""
    asset = _burst_delivery_asset(f"asf-probe-delivery-{status}")
    _register_adapter(asset.catalog, _BurstDeliveryAdapter())
    delivery = "https://new-storage.example/layout/item.zip"
    signed = delivery + "?X-Amz-Signature=fixture-capability"
    # A pending response can be returned by the anonymous hop itself, with a
    # relative Location resolved against that hop, not the original gateway.
    target = signed if status == 200 else delivery
    transport = _BurstDeliveryTransport(
        [
            (307, {"Location": target}, b"handoff"),
            (
                status,
                {
                    "Location": "item.zip?X-Amz-Signature=fixture-capability",
                    "Retry-After": "3",
                },
                b"queued" if status == 202 else b"",
            ),
        ]
    )
    session = requests.Session()
    session.mount("https://", transport)
    monkeypatch.setattr(
        "faninsar.remote.fetch.complete._asf_authenticated_session",
        lambda *_args: (session, "fixture-token"),
    )
    ledger = _CallLedger(remote.RemoteResourceBudget())

    observation = probe(asset, budget=ledger.budget, ledger=ledger)

    assert observation.state == ("ready" if status == 200 else "pending")
    assert observation.url == signed
    assert observation.delay_seconds == (0.0 if status == 200 else 3.0)
    assert ledger.requests == 2
    assert ledger.redirects == 1
    assert ledger.response_bytes_total == len(b"handoff") + (
        len(b"queued") if status == 202 else 0
    )
    assert transport.requests[0].headers["Authorization"] == "Bearer fixture-token"
    assert "Authorization" not in transport.requests[1].headers
    assert "Cookie" not in transport.requests[1].headers
    assert "?" not in asset.href


def test_asf_pending_delivery_is_observed_anonymously_on_the_next_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 202 Location can hand off to an unregistered object without reauthentication."""
    asset = _burst_delivery_asset("asf-pending-delivery-next-probe")
    _register_adapter(asset.catalog, _BurstDeliveryAdapter())
    delivery = "https://pending-object.example/result.zip?X-Amz-Signature=fixture"
    transport = _BurstDeliveryTransport(
        [(202, {"Location": delivery, "Retry-After": "30"}, b"pending"), (200, {}, b"")]
    )
    authenticated = requests.Session()
    authenticated.mount("https://", transport)
    anonymous = requests.Session()
    anonymous.mount("https://", transport)
    auth_calls: list[bool] = []

    def authenticate(*_args: Any) -> tuple[requests.Session, str]:
        auth_calls.append(True)
        assert len(auth_calls) == 1
        return authenticated, "fixture-token"

    monkeypatch.setattr(
        "faninsar.remote.fetch.complete._asf_authenticated_session", authenticate
    )
    monkeypatch.setattr(requests, "Session", lambda: anonymous)
    ledger = _CallLedger(remote.RemoteResourceBudget())

    pending = probe(asset, budget=ledger.budget, ledger=ledger)
    assert pending.state == "pending"
    assert pending.url == delivery
    assert pending.delay_seconds == 10.0
    ready = probe(asset, url=pending.url, budget=ledger.budget, ledger=ledger)

    assert ready.state == "ready"
    assert ready.url == delivery
    assert len(transport.requests) == 2
    assert transport.requests[1].url == delivery
    assert "Authorization" not in transport.requests[1].headers
    assert "Cookie" not in transport.requests[1].headers
    assert auth_calls == [True]
    assert ledger.requests == 2
    assert ledger.retries == 0
    assert ledger.response_bytes_total == len(b"pending")


@pytest.mark.parametrize("session_auth", [False, True])
def test_asf_prepared_delivery_never_prepares_account_credentials(
    monkeypatch: pytest.MonkeyPatch, session_auth: bool
) -> None:
    """Prepared delivery strips explicit/default credentials before real preparation."""
    asset = _burst_delivery_asset("asf-prepared-credentials")
    delivery = "https://unlisted-object.example/new/item.zip?X-Amz-Signature=fixture"
    transport = _BurstDeliveryTransport([(200, {}, b"PK\x03\x04payload")])
    session = requests.Session()
    session.mount("https://", transport)
    session.headers.update(
        {
            "aUtHoRiZaTiOn": "Bearer session-default",
            "cOoKiE": "default=secret",
            "Proxy-Authorization": "Basic origin-header-secret",
            "X-Api-Key": "session-key",
        }
    )
    if session_auth:
        session.auth = requests.auth.HTTPBasicAuth("fixture-user", "fixture-password")
    session.cookies.set("account", "secret", domain="unlisted-object.example")
    netrc_calls: list[str] = []

    def netrc_auth(url: str) -> tuple[str, str]:
        netrc_calls.append(url)
        return "implicit-user", "implicit-password"

    monkeypatch.setattr("requests.sessions.get_netrc_auth", netrc_auth)
    ledger = _CallLedger(remote.RemoteResourceBudget())

    response = _asf_request(
        session,
        "GET",
        delivery,
        adapter=_BurstDeliveryAdapter(),
        asset=asset,
        budget=ledger.budget,
        ledger=ledger,
        headers={
            "authorization": "Bearer explicit-secret",
            "COOKIE": "explicit=secret",
            "x-credential": "explicit-secret",
            "Accept-Encoding": "identity",
            "Range": "bytes=4-",
        },
    )

    assert response.status_code == 200
    assert len(transport.requests) == 1
    outgoing = transport.requests[0]
    assert outgoing.url == delivery
    assert outgoing.headers["Range"] == "bytes=4-"
    assert outgoing.headers["Accept-Encoding"] == "identity"
    assert not any(
        name.lower()
        in {
            "authorization",
            "cookie",
            "proxy-authorization",
            "x-api-key",
            "x-credential",
        }
        for name in outgoing.headers
    )
    assert netrc_calls == []
    assert not session.cookies
    assert ledger.requests == 1
    response.close()
    session.close()


def test_asf_prepared_download_resumes_without_reauth_or_persisting_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A prepared signed object resumes and publishes with only canonical identity."""
    payload = b"PK\x03\x04prepared-burst-payload"
    asset = _burst_delivery_asset("asf-prepared-resume")
    object.__setattr__(asset, "size_bytes", len(payload))
    object.__setattr__(
        asset, "checksum", "sha256:" + hashlib.sha256(payload).hexdigest()
    )
    _register_adapter(asset.catalog, _BurstDeliveryAdapter())
    delivery = "https://unlisted-object.example/layout/item.zip?X-Amz-Signature=fixture"
    transport = _BurstDeliveryTransport(
        [
            (
                206,
                {"Content-Range": f"bytes 4-{len(payload) - 1}/{len(payload)}"},
                payload[4:],
            )
        ]
    )
    session = requests.Session()
    session.mount("https://", transport)
    monkeypatch.setattr(requests, "Session", lambda: session)

    def no_reauthentication(*_args: Any) -> Any:
        pytest.fail("A prepared anonymous object must not resolve account credentials")

    monkeypatch.setattr(
        "faninsar.remote.fetch.complete._asf_authenticated_session", no_reauthentication
    )
    destination = tmp_path / "item.zip"
    identity = _identity(asset)
    _part_path(destination, identity).write_bytes(payload[:4])
    _write_part_state(
        _part_state_path(destination, identity),
        _PartState(identity, len(payload), asset.href, "p70.1"),
    )
    ledger = _CallLedger(remote.RemoteResourceBudget())

    result = download(asset, destination, transfer_url=delivery, ledger=ledger)

    assert result == destination
    assert destination.read_bytes() == payload
    assert transport.requests[0].url == delivery
    assert transport.requests[0].headers["Range"] == "bytes=4-"
    assert "Authorization" not in transport.requests[0].headers
    assert ledger.requests == 1
    assert ledger.response_bytes_total == len(payload) - 4
    manifest_text = next(tmp_path.glob("*.json")).read_text()
    manifest = json.loads(manifest_text)
    assert manifest["complete"] is True
    assert manifest["verification"]["status"] == "verified"
    assert "X-Amz" not in manifest_text
    assert "unlisted-object" not in manifest_text
    assert not _part_state_path(destination, identity).exists()


def test_asf_anonymous_delivery_stays_cookieless_when_redirect_returns_to_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anonymous Set-Cookie and default headers never revive account state."""
    asset = _burst_delivery_asset("asf-anonymous-return")
    delivery = "https://cdn-return.example/item.zip"
    transport = _BurstDeliveryTransport(
        [
            (
                302,
                {
                    "Location": delivery,
                    "Set-Cookie": "login=secret; Domain=.asf.alaska.edu; Path=/",
                },
                b"gateway",
            ),
            (
                302,
                {
                    "Location": asset.href,
                    "Set-Cookie": "object=secret; Domain=.asf.alaska.edu; Path=/",
                },
                b"cdn",
            ),
            (200, {"Set-Cookie": "final=secret; Path=/"}, b"payload"),
        ]
    )
    session = requests.Session()
    session.mount("https://", transport)
    session.headers["authorization"] = "Bearer session-default"
    session.headers["COOKIE"] = "default=secret"
    monkeypatch.setattr(
        "requests.sessions.get_netrc_auth",
        lambda _url: ("fixture-user", "fixture-password"),
    )
    ledger = _CallLedger(remote.RemoteResourceBudget())

    response = _asf_request(
        session,
        "GET",
        asset.href,
        adapter=_BurstDeliveryAdapter(),
        asset=asset,
        budget=ledger.budget,
        ledger=ledger,
        headers={"authorization": "Bearer explicit", "cookie": "explicit=secret"},
    )

    assert response.status_code == 200
    assert len(transport.requests) == 3
    for outgoing in transport.requests[1:]:
        assert "Authorization" not in outgoing.headers
        assert "Cookie" not in outgoing.headers
    assert not session.cookies
    assert ledger.requests == 3
    assert ledger.redirects == 2
    response.close()
    session.close()


def test_asf_prepared_signed_delivery_rejects_redirect_before_second_request() -> None:
    """A signed object capability is terminal even on a prepared first hop."""
    asset = _burst_delivery_asset("asf-prepared-terminal")
    delivery = "https://terminal-object.example/item.zip?X-Amz-Signature=fixture"
    transport = _BurstDeliveryTransport(
        [(307, {"Location": "https://other-object.example/item.zip"}, b"redirect")]
    )
    session = requests.Session()
    session.mount("https://", transport)
    ledger = _CallLedger(remote.RemoteResourceBudget())

    with pytest.raises(remote.RemoteAccessError, match="signed_redirect"):
        _asf_request(
            session,
            "GET",
            delivery,
            adapter=_BurstDeliveryAdapter(),
            asset=asset,
            budget=ledger.budget,
            ledger=ledger,
        )

    assert len(transport.requests) == 1
    assert "Authorization" not in transport.requests[0].headers
    assert ledger.requests == 1
    assert ledger.redirects == 0
    assert ledger.response_bytes_total == len(b"redirect")
    session.close()


@pytest.mark.parametrize(
    "delivery",
    [
        "http://object.example/item.zip",
        "https://user:password@object.example/item.zip",
        "https://object.example/item.zip#fragment",
        "https://object.example/%2e%2e/item.zip",
        "https://object.example/a%2fb.zip",
        "https://object.example:invalid/item.zip",
    ],
)
def test_asf_prepared_delivery_rejects_invalid_url_before_request(
    delivery: str,
) -> None:
    """Prepared delivery admission still applies the structural HTTPS gate."""
    asset = _burst_delivery_asset("asf-prepared-invalid")
    transport = _BurstDeliveryTransport([])
    session = requests.Session()
    session.mount("https://", transport)
    ledger = _CallLedger(remote.RemoteResourceBudget())

    with pytest.raises(remote.RemoteAccessError, match="invalid_endpoint"):
        _asf_request(
            session,
            "GET",
            delivery,
            adapter=_BurstDeliveryAdapter(),
            asset=asset,
            budget=ledger.budget,
            ledger=ledger,
        )

    assert transport.requests == []
    assert ledger.requests == 0
    session.close()


def test_asf_prepared_delivery_does_not_admit_unregistered_asset_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anonymous runtime admission never relaxes the canonical asset registry."""
    asset = _burst_delivery_asset("asf-invalid-canonical")
    object.__setattr__(asset, "href", "https://unregistered.example/item.zip")
    _register_adapter(asset.catalog, _BurstDeliveryAdapter())
    session = requests.Session()
    transport = _BurstDeliveryTransport([])
    session.mount("https://", transport)
    monkeypatch.setattr(
        "faninsar.remote.fetch.complete._asf_authenticated_session",
        lambda *_args: pytest.fail("Invalid identity must fail before authentication"),
    )
    ledger = _CallLedger(remote.RemoteResourceBudget())

    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        probe(asset, ledger=ledger)
    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        _asf_request(
            session,
            "GET",
            "https://delivery.example/item.zip",
            adapter=_BurstDeliveryAdapter(),
            asset=asset,
            budget=ledger.budget,
            ledger=ledger,
        )

    assert transport.requests == []
    assert ledger.requests == 0
    session.close()


def test_asf_prepared_registered_auth_origin_cannot_bypass_route_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A registered authentication hostname does not authorize every route."""
    adapter = _BurstDeliveryAdapter()
    adapter.redirect_origins = (
        *adapter.redirect_origins,
        "https://urs.earthdata.nasa.gov",
    )
    adapter.redirect_path_prefixes = {
        "https://urs.earthdata.nasa.gov": ("/oauth/authorize",)
    }
    asset = _burst_delivery_asset("asf-prepared-forbidden-auth-route")
    delivery = "https://urs.earthdata.nasa.gov/unregistered-route"
    transport = _BurstDeliveryTransport([(200, {}, b"PK\x03\x04payload")])
    session = requests.Session()
    session.mount("https://", transport)
    monkeypatch.setattr(
        "faninsar.remote.fetch.complete._asf_authenticated_session",
        lambda *_args: pytest.fail("An unregistered auth route must fail before login"),
    )
    ledger = _CallLedger(remote.RemoteResourceBudget())

    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        _asf_request(
            session,
            "GET",
            delivery,
            adapter=adapter,
            asset=asset,
            budget=ledger.budget,
            ledger=ledger,
            headers={"Authorization": "Bearer fixture-secret"},
        )
    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        _stream_download(
            asset,
            adapter,
            ledger.budget,
            ledger,
            tmp_path / "item.zip",
            transfer_url_override=delivery,
        )

    assert transport.requests == []
    assert ledger.requests == 0
    session.close()


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
                        "https://data.lpdaac.earthdatacloud.nasa.gov/login?code=fixture"
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
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(
        "faninsar.remote.auth._netrc_credentials",
        lambda host: (
            ("fixture-user", "fixture-password")
            if host == "urs.earthdata.nasa.gov"
            else None
        ),
    )

    class _LPDAACAdapter:
        provider = "LPCLOUD"
        origins = ("https://data.lpdaac.earthdatacloud.nasa.gov",)
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
    ledger = _CallLedger(remote.RemoteResourceBudget(max_redirects=5))
    staging = tmp_path / "NASADEM_HGT_n36w121.zip"

    size, digest = _stream_download(asset, adapter, ledger.budget, ledger, staging)

    assert size == len(payload)
    assert digest == ""
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
    ledger = _CallLedger(remote.RemoteResourceBudget(max_redirects=3))
    response = _asf_request(
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
        assert auth is _no_auth


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
    assert _lpdaac_redirect_url(source, valid, asset, _LPDAACAdapter()) == valid
    for invalid in (
        valid.replace("https://d123example.cloudfront.net", "http://evil.example"),
        valid.replace("/s3-012", "/../"),
        valid.replace("d123example", "evil.example") + "#fragment",
    ):
        with pytest.raises(remote.RemoteAccessError):
            _lpdaac_redirect_url(source, invalid, asset, _LPDAACAdapter())


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
        _asf_redirect_url(
            "https://sentinel1.asf.alaska.edu/SLC/item.zip",
            valid,
            asset,
            _ASFAdapter(),
        )
        == valid
    )
    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        _asf_redirect_url(
            "https://datapool.asf.alaska.edu/SLC/item.zip",
            valid,
            asset,
            _ASFAdapter(),
        )
    changed = "https://objects.example/new-layout/other.zip"
    assert (
        _asf_redirect_url(
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
    assert _asf_redirect_url(source, valid, asset, _ASFAdapter()) == valid
    changed = "https://objects.example/new-layout/dem.tif"
    assert _asf_redirect_url(source, changed, asset, _ASFAdapter()) == changed


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
        _response_is_complete(response)


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
    _validate_representation(staging, asset)

    staging.write_bytes(b"\xef\xbb\xbf  <HTML>login</HTML>")
    with pytest.raises(remote.RemoteIntegrityError, match="representation_mismatch"):
        _validate_representation(staging, asset)


def test_signed_query_redirect_is_terminal() -> None:
    """A request carrying a delivery capability cannot follow a 3xx."""
    adapter = _ChunkAdapter((b"payload",))
    handler = _RedirectHandler(
        adapter,
        remote.RemoteResourceBudget(max_redirects=1),
    )
    request = urllib.request.Request(
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

    _register_adapter("failing-complete-stream", _FailingAdapter())
    asset = remote.RemoteAsset(
        "failing",
        "failing-complete-stream",
        None,
        "item",
        "data",
        "https://failing.invalid/file.bin?x-random-signature=fixture-secret",
    )
    with (
        caplog.at_level("WARNING"),
        pytest.raises(remote.RemoteAccessError, match="transfer_failed") as error,
    ):
        remote.download(
            asset,
            tmp_path / "file.bin",
            budget=remote.RemoteResourceBudget(max_retries=1),
        )
    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in caplog.text
    assert "objects.example" not in caplog.text


class _AnonymousRangeAdapter:
    """Anonymous fixture adapter with no ``fetch`` (urllib transport path)."""

    provider = "ranged"
    origins = ("https://ranged.invalid",)
    path_prefixes = ("/",)
    redirect_origins = ()
    profiles = ("anonymous",)


def _ranged_asset(
    catalog: str,
    *,
    size_bytes: int | None = 64,
    checksum: str | None = None,
    href: str = "https://ranged.invalid/data.bin",
) -> remote.RemoteAsset:
    """Build one anonymous asset for the registered ranged catalog."""
    return remote.RemoteAsset(
        "ranged",
        catalog,
        None,
        "item",
        "data",
        href,
        size_bytes=size_bytes,
        checksum=checksum,
    )


class _ResumeOpener:
    """Opener double that honors Range and can drop the first stream."""

    def __init__(
        self,
        payload: bytes,
        *,
        drop_first_after: int | None = None,
        answer_206_as_200: bool = False,
        content_range_total: int | None = None,
    ) -> None:
        self.payload = payload
        self.drop_first_after = drop_first_after
        self.answer_206_as_200 = answer_206_as_200
        self.content_range_total = content_range_total
        self.requests: list[dict[str, str]] = []
        self._dropped = False

    def open(self, request: urllib.request.Request, timeout: float) -> Any:
        del timeout
        headers = {key.lower(): value for key, value in request.header_items()}
        self.requests.append(headers)
        range_header = headers.get("range")
        total = self.content_range_total or len(self.payload)
        if range_header is None:
            body = self.payload
            if self.drop_first_after is not None and not self._dropped:
                self._dropped = True
                return _UrllibStreamResponse(
                    200, body[: self.drop_first_after], headers={}
                )
            return _UrllibStreamResponse(200, body, headers={})
        start_text = range_header.removeprefix("bytes=").split("-", 1)[0]
        start = int(start_text)
        if self.answer_206_as_200:
            # A server that ignores Range returns the whole object with 200.
            return _UrllibStreamResponse(200, self.payload, headers={})
        return _UrllibStreamResponse(
            206,
            self.payload[start:],
            headers={"Content-Range": f"bytes {start}-{total - 1}/{total}"},
        )


class _UrllibStreamResponse:
    """Minimal context-managed response for the urllib transport branch."""

    def __init__(self, status: int, body: bytes, *, headers: dict[str, str]) -> None:
        self.status = status
        self._body = body
        self.headers = headers

    def read(self, size: int) -> bytes:
        chunk, self._body = self._body[:size], self._body[size:]
        return chunk

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _stub_urllib_opener(monkeypatch: pytest.MonkeyPatch, opener: _ResumeOpener) -> None:
    """Route the module's opener construction through the fixture opener."""
    monkeypatch.setattr(
        "faninsar.remote.fetch.complete.urllib.request.build_opener",
        lambda *_handlers: opener,
    )


def test_resume_completes_with_one_range_request_after_mid_stream_kill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A killed download resumes with exactly one ``Range: bytes=<k>-`` GET.

    The retained ``.part`` prefix is re-streamed through the digest, the 206
    answer is accepted via the resume gate, and the published file plus its
    identity manifest verify byte-identically to a single-pass download.
    """
    payload = b"pk\x03\x04".join([bytes([i]) * 8 for i in range(16)])
    adapter = _AnonymousRangeAdapter()
    _register_adapter("ranged-resume", adapter)
    asset = _ranged_asset("ranged-resume", size_bytes=len(payload))
    destination = tmp_path / "data.bin"
    opener = _ResumeOpener(payload, drop_first_after=24)
    _stub_urllib_opener(monkeypatch, opener)

    with pytest.raises(remote.RemoteIntegrityError):
        remote.download(asset, destination)
    identity = _identity(asset)
    part = _part_path(destination, identity)
    part_state = _part_state_path(destination, identity)
    assert part.read_bytes() == payload[:24]
    state = _read_part_state(part_state)
    assert state is not None
    assert state.identity == identity
    assert state.expected_size == len(payload)
    # The persisted URL is the _safe_url-scrubbed canonical href.
    assert state.url == "https://ranged.invalid/data.bin"
    assert "?" not in state.url
    assert len(opener.requests) == 1  # truncated body fails the size gate

    # Second invocation resumes instead of restarting.
    resume_opener = _ResumeOpener(payload)
    _stub_urllib_opener(monkeypatch, resume_opener)
    result = remote.download(asset, destination)

    assert result == destination
    range_requests = [h["range"] for h in resume_opener.requests if "range" in h]
    assert range_requests == ["bytes=24-"]
    assert destination.read_bytes() == payload
    assert not part.exists()
    assert not part_state.exists()
    manifest = json.loads(
        (tmp_path / f"data.bin.faninsar.remote.{identity}.json").read_text()
    )
    assert manifest["complete"] is True
    assert "sha256" not in manifest
    assert manifest["size"] == len(payload)


def test_resume_provider_digest_covers_retained_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A resumed suffix is verified together with the retained prefix."""
    payload = b"pk\x03\x04".join([bytes([i]) * 8 for i in range(16)])
    digest = hashlib.sha256(payload).hexdigest()
    adapter = _AnonymousRangeAdapter()
    _register_adapter("ranged-resume-digest", adapter)
    asset = _ranged_asset(
        "ranged-resume-digest",
        size_bytes=len(payload),
        checksum=f"sha256:{digest}",
    )
    destination = tmp_path / "data.bin"
    opener = _ResumeOpener(payload, drop_first_after=24)
    _stub_urllib_opener(monkeypatch, opener)
    with pytest.raises(remote.RemoteIntegrityError):
        remote.download(asset, destination)

    resume_opener = _ResumeOpener(payload)
    _stub_urllib_opener(monkeypatch, resume_opener)
    remote.download(asset, destination)

    assert destination.read_bytes() == payload
    manifest = json.loads(
        destination.with_name(
            f"{destination.name}.faninsar.remote.{_identity(asset)}.json"
        ).read_text()
    )
    assert manifest["verification"]["status"] == "verified"
    assert manifest["sha256"] == digest
    assert resume_opener.requests[0]["range"] == "bytes=24-"


def test_corrupt_resume_prefix_fails_provider_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A mutated retained prefix cannot verify against the full-object digest."""
    payload = b"pk\x03\x04".join([bytes([i]) * 8 for i in range(16)])
    digest = hashlib.sha256(payload).hexdigest()
    adapter = _AnonymousRangeAdapter()
    _register_adapter("ranged-resume-corrupt", adapter)
    asset = _ranged_asset(
        "ranged-resume-corrupt",
        size_bytes=len(payload),
        checksum=f"sha256:{digest}",
    )
    destination = tmp_path / "data.bin"
    opener = _ResumeOpener(payload, drop_first_after=24)
    _stub_urllib_opener(monkeypatch, opener)
    with pytest.raises(remote.RemoteIntegrityError):
        remote.download(asset, destination)
    identity = _identity(asset)
    part = _part_path(destination, identity)
    corrupted = bytearray(part.read_bytes())
    corrupted[0] ^= 0xFF
    part.write_bytes(bytes(corrupted))

    resume_opener = _ResumeOpener(payload)
    _stub_urllib_opener(monkeypatch, resume_opener)
    with pytest.raises(remote.RemoteIntegrityError) as raised:
        remote.download(asset, destination)
    assert raised.value.reason == "checksum_mismatch"
    manifest = json.loads(
        destination.with_name(
            f"{destination.name}.faninsar.remote.{identity}.json"
        ).read_text()
    )
    assert manifest["verification"]["status"] == "failed"
    assert manifest["complete"] is False


def test_resume_false_performs_a_full_get(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``resume=False`` ignores a retained ``.part`` and refetches byte 0."""
    payload = b"pk\x03\x04" + bytes(range(48))
    adapter = _AnonymousRangeAdapter()
    _register_adapter("ranged-noresume", adapter)
    asset = _ranged_asset("ranged-noresume", size_bytes=len(payload))
    destination = tmp_path / "data.bin"
    identity = _identity(asset)

    # Pre-seed a partial download.
    _part_path(destination, identity).write_bytes(payload[:16])
    _write_part_state(
        _part_state_path(destination, identity),
        _PartState(
            identity=identity,
            expected_size=len(payload),
            url="https://ranged.invalid/data.bin",
            planner_version="p70.1",
        ),
    )

    opener = _ResumeOpener(payload)
    _stub_urllib_opener(monkeypatch, opener)
    remote.download(asset, destination, resume=False)

    assert all("range" not in h for h in opener.requests)
    assert destination.read_bytes() == payload


def test_identity_mismatch_part_state_restarts_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``.part-state`` naming another identity is ignored: clean restart."""
    payload = b"pk\x03\x04" + bytes(range(48))
    adapter = _AnonymousRangeAdapter()
    _register_adapter("ranged-mismatch", adapter)
    asset = _ranged_asset("ranged-mismatch", size_bytes=len(payload))
    destination = tmp_path / "data.bin"
    identity = _identity(asset)

    _part_path(destination, identity).write_bytes(payload[:16])
    _write_part_state(
        _part_state_path(destination, identity),
        _PartState(
            identity="different-identity",
            expected_size=len(payload),
            url="https://ranged.invalid/data.bin",
            planner_version="p70.1",
        ),
    )

    opener = _ResumeOpener(payload)
    _stub_urllib_opener(monkeypatch, opener)
    remote.download(asset, destination)

    assert all("range" not in h for h in opener.requests)
    assert destination.read_bytes() == payload


def test_resume_with_unknown_expected_size_uses_final_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``asset.size_bytes=None`` resumes unbounded; the size gate stays final."""
    payload = b"pk\x03\x04" + bytes(range(48))
    adapter = _AnonymousRangeAdapter()
    _register_adapter("ranged-unknown-size", adapter)
    asset = _ranged_asset("ranged-unknown-size", size_bytes=None)
    destination = tmp_path / "data.bin"
    identity = _identity(asset)

    _part_path(destination, identity).write_bytes(payload[:16])
    _write_part_state(
        _part_state_path(destination, identity),
        _PartState(
            identity=identity,
            expected_size=None,
            url="https://ranged.invalid/data.bin",
            planner_version="p70.1",
        ),
    )

    opener = _ResumeOpener(payload)
    _stub_urllib_opener(monkeypatch, opener)
    remote.download(asset, destination)

    range_requests = [h["range"] for h in opener.requests if "range" in h]
    assert range_requests == ["bytes=16-"]
    assert destination.read_bytes() == payload
    # The observed Content-Range total is persisted on the first 206.
    state_after = _read_part_state(_part_state_path(destination, identity))
    assert state_after is None  # cleaned after publish


def test_200_answer_to_range_request_restarts_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 200 response to a resume GET means the server ignored Range."""
    payload = b"pk\x03\x04" + bytes(range(48))
    adapter = _AnonymousRangeAdapter()
    _register_adapter("ranged-range-ignored", adapter)
    asset = _ranged_asset("ranged-range-ignored", size_bytes=len(payload))
    destination = tmp_path / "data.bin"
    identity = _identity(asset)

    _part_path(destination, identity).write_bytes(payload[:16])
    _write_part_state(
        _part_state_path(destination, identity),
        _PartState(
            identity=identity,
            expected_size=len(payload),
            url="https://ranged.invalid/data.bin",
            planner_version="p70.1",
        ),
    )

    opener = _ResumeOpener(payload, answer_206_as_200=True)
    _stub_urllib_opener(monkeypatch, opener)
    remote.download(asset, destination)

    # First request carried the Range header; after the 200 the transfer
    # restarted and re-GET the whole object.
    assert opener.requests[0].get("range") == "bytes=16-"
    assert destination.read_bytes() == payload


def test_read_verification_ignores_a_part_state_sibling(tmp_path: Path) -> None:
    """``.part-state`` sits outside the manifest glob; it is never a manifest."""
    destination = tmp_path / "data.bin"
    destination.write_bytes(b"payload")
    identity = "abc123"
    part_state = _part_state_path(destination, identity)
    _write_part_state(
        part_state,
        _PartState(
            identity=identity,
            expected_size=7,
            url="https://ranged.invalid/data.bin",
            planner_version="p70.1",
        ),
    )
    assert read_verification(destination) is None


def test_grown_object_between_resumes_restarts_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A persisted Content-Range total smaller than a later answer restarts."""
    payload = b"pk\x03\x04" + bytes(range(48))
    grown = payload + b"GROWN"
    adapter = _AnonymousRangeAdapter()
    _register_adapter("ranged-grown", adapter)
    asset = _ranged_asset("ranged-grown", size_bytes=None)
    destination = tmp_path / "data.bin"
    identity = _identity(asset)

    _part_path(destination, identity).write_bytes(payload[:16])
    _write_part_state(
        _part_state_path(destination, identity),
        _PartState(
            identity=identity,
            expected_size=None,
            url="https://ranged.invalid/data.bin",
            planner_version="p70.1",
            # Recorded during an earlier resume of the pre-growth object.
            content_range_total=len(payload),
        ),
    )

    opener = _ResumeOpener(grown)
    _stub_urllib_opener(monkeypatch, opener)
    remote.download(asset, destination)

    assert opener.requests[0].get("range") == "bytes=16-"
    assert destination.read_bytes() == grown


def test_asf_resume_prefix_is_rehashed_before_range_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ASF transport sends ``Range: bytes=<k>-`` and re-hashes the prefix."""
    payload = b"PK\x03\x04" + bytes(range(96))

    class _RangeASFSession:
        def __init__(self) -> None:
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
            del data, auth, allow_redirects, stream, timeout
            self.requests.append((method, url, dict(headers)))
            if url.endswith("/api/users/find_or_create_token"):
                return _ASFResponse(url, 200, b'{"access_token":"fixture-token"}')
            if "/oauth/authorize" in url:
                return _ASFResponse(
                    url,
                    302,
                    body=b"oauth-redirect",
                    headers={"Location": "https://datapool.asf.alaska.edu/item.zip"},
                )
            if url == "https://datapool.asf.alaska.edu/item.zip":
                range_header = headers.get("Range")
                if range_header is None:
                    return _ASFResponse(url, 200, payload)
                start = int(range_header.removeprefix("bytes=").split("-")[0])
                return _ASFResponse(
                    url,
                    206,
                    payload[start:],
                    headers={
                        "Content-Range": (
                            f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                        )
                    },
                )
            message = f"unexpected fixture URL: {url}"
            raise AssertionError(message)

        def close(self) -> None:
            self.closed = True

    session = _RangeASFSession()
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(
        "faninsar.remote.fetch.complete._asf_authenticated_session",
        lambda *_args: (session, "fixture-token"),
    )
    monkeypatch.setattr(
        "faninsar.remote.auth._netrc_credentials",
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
            "https://urs.earthdata.nasa.gov",
        )
        profiles = ("earthdata-asf",)

    adapter = _ASFAdapter()
    _register_adapter("asf-resume", adapter)
    asset = remote.RemoteAsset(
        "ASF",
        "asf-resume",
        "sentinel-1",
        "S1_RESUME",
        "data",
        "https://datapool.asf.alaska.edu/item.zip",
        size_bytes=len(payload),
        auth_profile="earthdata-asf",
    )
    identity = _identity(asset)
    destination = tmp_path / "item.zip"
    _part_path(destination, identity).write_bytes(payload[:32])
    _write_part_state(
        _part_state_path(destination, identity),
        _PartState(
            identity=identity,
            expected_size=len(payload),
            url="https://datapool.asf.alaska.edu/item.zip",
            planner_version="p70.1",
        ),
    )

    remote.download(asset, destination)

    data_requests = [
        headers
        for _method, url, headers in session.requests
        if url == "https://datapool.asf.alaska.edu/item.zip"
    ]
    assert data_requests
    assert data_requests[-1].get("Range") == "bytes=32-"
    assert destination.read_bytes() == payload
    manifest = json.loads(
        (tmp_path / f"item.zip.faninsar.remote.{identity}.json").read_text()
    )
    assert manifest["complete"] is True
    assert "sha256" not in manifest
