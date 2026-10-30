"""CDSE range-fetcher seam (PROPOSAL-0070): ``cdse_range_fetcher``.

The seam yields an open, credential-header ``RangeFetcher`` after the same
``_validate_url`` + ``_is_odata_value`` + ``_needs_cdse_bearer`` gate as
``fetch_cdse_ranges``.  The access token is fetched inside the context and
never returned to the caller — missions drive ``fetch``/``fetch_into``
through the fetcher alone.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any, Self

import pytest

from faninsar.remote.catalog import _ADAPTERS, _REGISTRY_LOCK
from faninsar.remote.errors import RemoteAccessError
from faninsar.remote.fetch import RangeFetcher, RangeSpan
from faninsar.remote.providers.cdse import (
    CDSE_CATALOG,
    cdse_range_fetcher,
    ensure_cdse_catalog,
)

_ODATA_SPAN_URL = (
    "https://zipper.dataspace.copernicus.eu/odata/v1/"
    "Products(078a8ae1-47f1-40ed-87f6-99a3e62ca665)/"
    "Nodes(S1C_IW_SLC__1SDV.SAFE)/Nodes(measurement)/"
    "Nodes(s1c-iw1-slc-vv-004.tiff)/$value"
)


@pytest.fixture
def isolated_cdse_catalog() -> Any:
    """Remove ``sentinel1-cdse`` for the test and restore afterwards."""
    with _REGISTRY_LOCK:
        previous = _ADAPTERS.pop(CDSE_CATALOG, None)
    try:
        yield
    finally:
        with _REGISTRY_LOCK:
            _ADAPTERS.pop(CDSE_CATALOG, None)
            if previous is not None:
                _ADAPTERS[CDSE_CATALOG] = previous


@pytest.fixture
def fake_cdse_token(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the Keycloak grant; return the list of issued tokens."""
    issued: list[str] = []

    def fake_token(*, force: bool = False) -> str:
        del force
        issued.append("cdse-access-token")
        return "cdse-access-token"

    monkeypatch.setattr(
        "faninsar.remote.providers.cdse.request_cdse_token",
        fake_token,
    )
    return issued


class _FakeStreamContent:
    """Chunk source shaped like ``response.content``."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def iter_chunked(self, n: int) -> Any:
        for offset in range(0, len(self._body), n):
            yield self._body[offset : offset + n]


class _FakeRangeResponse:
    """One 206 whose Content-Range echoes the requested interval."""

    def __init__(self, payload: bytes, request_start: int, request_end: int) -> None:
        self.status = 206
        self.headers = {
            "Content-Range": (f"bytes {request_start}-{request_end}/{len(payload)}"),
        }
        self.content = _FakeStreamContent(payload[request_start : request_end + 1])

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _RecordingRangeSession:
    """Stand-in ``ClientSession`` that parses Range/Authorization headers."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.requests: list[tuple[str, str | None]] = []

    def get(self, url: str, **kwargs: object) -> _FakeRangeResponse:
        del url
        headers = kwargs.get("headers") or {}
        start_s, end_s = str(headers.get("Range", "bytes=0-0"))[len("bytes=") :].split(
            "-"
        )
        self.requests.append((f"{start_s}-{end_s}", headers.get("Authorization")))
        return _FakeRangeResponse(self._payload, int(start_s), int(end_s))


def test_cdse_range_fetcher_yields_open_fetcher_for_empty_spans(
    isolated_cdse_catalog: None,
    fake_cdse_token: list[str],
) -> None:
    """Empty ``spans`` validates nothing and still yields an open fetcher."""
    del isolated_cdse_catalog
    ensure_cdse_catalog()

    async def _run() -> tuple[bool, bool]:
        async with cdse_range_fetcher(()) as fetcher:
            assert isinstance(fetcher, RangeFetcher)
            session_open = fetcher._session is not None
        return session_open, fetcher._session is not None

    inside, after = asyncio.run(_run())
    assert inside is True
    assert after is False
    assert fake_cdse_token == ["cdse-access-token"]


def test_cdse_range_fetcher_rejects_non_odata_span_before_token(
    isolated_cdse_catalog: None,
    fake_cdse_token: list[str],
) -> None:
    """A non-OData span fails ``invalid_endpoint`` before the grant runs."""
    del isolated_cdse_catalog
    ensure_cdse_catalog()
    span = RangeSpan(
        "https://zipper.dataspace.copernicus.eu/odata/v1/plain-object",
        0,
        15,
        "probe",
    )

    async def _run() -> None:
        async with cdse_range_fetcher((span,)):
            pass

    with pytest.raises(RemoteAccessError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "invalid_endpoint"
    assert fake_cdse_token == []


def test_cdse_range_fetcher_rejects_unregistered_origin(
    isolated_cdse_catalog: None,
    fake_cdse_token: list[str],
) -> None:
    """A foreign-host ``Nodes(...)/$value`` URL fails origin validation."""
    del isolated_cdse_catalog
    ensure_cdse_catalog()
    span = RangeSpan("https://example.invalid/Nodes(x)/$value", 0, 15, "probe")

    async def _run() -> None:
        async with cdse_range_fetcher((span,)):
            pass

    with pytest.raises(RemoteAccessError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "unregistered_endpoint"
    assert fake_cdse_token == []


def test_cdse_range_fetcher_rejects_non_bearer_odata_host(
    isolated_cdse_catalog: None,
    fake_cdse_token: list[str],
) -> None:
    """An OData-shaped URL on the STAC origin still fails the bearer gate."""
    del isolated_cdse_catalog
    ensure_cdse_catalog()
    span = RangeSpan(
        "https://stac.dataspace.copernicus.eu/Nodes(x)/$value",
        0,
        15,
        "probe",
    )

    async def _run() -> None:
        async with cdse_range_fetcher((span,)):
            pass

    with pytest.raises(RemoteAccessError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "invalid_endpoint"
    assert fake_cdse_token == []


def test_cdse_range_fetcher_fetch_into_sends_bearer_without_leaking_token(
    isolated_cdse_catalog: None,
    fake_cdse_token: list[str],
) -> None:
    """``fetch_into`` carries ``Authorization: Bearer``; caller never sees it."""
    del isolated_cdse_catalog
    ensure_cdse_catalog()
    payload = b"ABCDEFGHIJ"
    span = RangeSpan(_ODATA_SPAN_URL, 2, 9, "window")
    session = _RecordingRangeSession(payload)
    sink = io.BytesIO()

    async def _run() -> int:
        async with cdse_range_fetcher((span,)) as fetcher:
            # The context opens a real session; swap in the recording stub so
            # no network traffic leaves the test while the fetcher is open.
            fetcher._session = session
            try:
                return await fetcher.fetch_into(span, sink)
            finally:
                fetcher._session = None

    assert asyncio.run(_run()) == 8
    assert sink.getvalue() == payload[2:10]
    assert session.requests == [("2-9", "Bearer cdse-access-token")]
    assert fake_cdse_token == ["cdse-access-token"]


def test_cdse_range_fetcher_validates_every_span_url(
    isolated_cdse_catalog: None,
    fake_cdse_token: list[str],
) -> None:
    """One bad span in a mixed batch aborts the whole context."""
    del isolated_cdse_catalog
    ensure_cdse_catalog()
    good = RangeSpan(_ODATA_SPAN_URL, 0, 3, "good")
    bad = RangeSpan(
        "https://zipper.dataspace.copernicus.eu/odata/v1/not-odata",
        0,
        3,
        "bad",
    )

    async def _run() -> None:
        async with cdse_range_fetcher((good, bad)):
            pass

    with pytest.raises(RemoteAccessError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "invalid_endpoint"
    assert fake_cdse_token == []
