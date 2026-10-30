"""Byte-range fetch seam (PROPOSAL-0061): RangeSpan and RangeFetcher."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Self

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from faninsar.remote.errors import RemoteAccessError, RemoteIntegrityError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


def test_range_span_is_a_byte_interval_value() -> None:
    """A span is the inclusive byte interval the fetcher will request."""
    from faninsar.remote.fetch import RangeSpan

    span = RangeSpan(
        url="https://example.invalid/measurement.tiff",
        start=0,
        end=1023,
        label="hdr",
    )
    assert span.url == "https://example.invalid/measurement.tiff"
    assert span.start == 0
    assert span.end == 1023
    assert span.label == "hdr"


def test_fetch_ranges_async_returns_the_body_of_one_206() -> None:
    """One span comes back as the exact inclusive byte interval requested."""
    payload = b"ABCDEFGHIJ"

    async def handler(request: web.Request) -> web.StreamResponse:
        assert request.headers["Range"] == "bytes=2-5"
        body = payload[2:6]
        return web.Response(
            status=206,
            body=body,
            headers={"Content-Range": "bytes 2-5/10", "Content-Encoding": "identity"},
        )

    async def _run() -> dict[str, bytes]:
        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeSpan, fetch_ranges_async

            url = str(server.make_url("/obj"))
            return await fetch_ranges_async(
                (RangeSpan(url=url, start=2, end=5, label="a"),)
            )
        finally:
            await server.close()

    assert asyncio.run(_run()) == {"a": b"CDEF"}


def test_fetch_ranges_async_retries_the_whole_span_after_429() -> None:
    """A 429 does not split the span; the next attempt returns the interval."""
    payload = b"ABCDEFGHIJ"
    attempts = {"n": 0}

    async def handler(request: web.Request) -> web.StreamResponse:
        del request
        attempts["n"] += 1
        if attempts["n"] == 1:
            return web.Response(status=429)
        body = payload[2:6]
        return web.Response(
            status=206,
            body=body,
            headers={"Content-Range": "bytes 2-5/10", "Content-Encoding": "identity"},
        )

    async def _run() -> dict[str, bytes]:
        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeSpan, fetch_ranges_async

            url = str(server.make_url("/obj"))
            return await fetch_ranges_async(
                (RangeSpan(url=url, start=2, end=5, label="a"),)
            )
        finally:
            await server.close()

    assert asyncio.run(_run()) == {"a": b"CDEF"}


def test_fetch_ranges_async_does_not_follow_redirects() -> None:
    """A 302 is a hard failure; the Location body is never requested."""
    hits: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        hits.append(request.path)
        if request.path == "/obj":
            return web.Response(status=302, headers={"Location": "/other"})
        return web.Response(
            status=206,
            body=b"CDEF",
            headers={"Content-Range": "bytes 2-5/10", "Content-Encoding": "identity"},
        )

    async def _run() -> dict[str, bytes]:
        app = web.Application()
        app.router.add_get("/obj", handler)
        app.router.add_get("/other", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeSpan, fetch_ranges_async

            url = str(server.make_url("/obj"))
            return await fetch_ranges_async(
                (RangeSpan(url=url, start=2, end=5, label="a"),)
            )
        finally:
            await server.close()

    with pytest.raises(RemoteAccessError, match="redirect"):
        asyncio.run(_run())
    assert hits == ["/obj"]


def test_fetch_ranges_async_rejects_a_mismatched_content_range() -> None:
    """A 206 whose Content-Range is not the requested interval is not assembled."""
    hits = {"n": 0}

    async def handler(request: web.Request) -> web.StreamResponse:
        del request
        hits["n"] += 1
        return web.Response(
            status=206,
            body=b"CDEF",
            headers={"Content-Range": "bytes 0-3/10", "Content-Encoding": "identity"},
        )

    async def _run() -> dict[str, bytes]:
        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeSpan, fetch_ranges_async

            url = str(server.make_url("/obj"))
            return await fetch_ranges_async(
                (RangeSpan(url=url, start=2, end=5, label="a"),)
            )
        finally:
            await server.close()

    with pytest.raises(RemoteIntegrityError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "content_range_mismatch"
    assert hits["n"] == 1


def test_fetch_ranges_async_treats_401_as_access_error() -> None:
    """A 401 is auth failure, not a mismatched 206."""

    async def handler(request: web.Request) -> web.StreamResponse:
        del request
        return web.Response(status=401)

    async def _run() -> dict[str, bytes]:
        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeSpan, fetch_ranges_async

            url = str(server.make_url("/obj"))
            return await fetch_ranges_async(
                (RangeSpan(url=url, start=2, end=5, label="a"),)
            )
        finally:
            await server.close()

    with pytest.raises(RemoteAccessError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "unauthorized"


def test_concurrent_401s_share_one_token_refresh() -> None:
    """Two in-flight 401s wait on one refresh, then retry the whole span once."""
    payload = b"ABCDEFGHIJ"
    refresh_calls = {"n": 0}

    async def _run() -> tuple[dict[str, bytes], dict[str, bytes]]:
        first_wave = asyncio.Barrier(2)

        async def handler(request: web.Request) -> web.StreamResponse:
            auth = request.headers.get("Authorization", "")
            if auth != "Bearer fresh":
                await first_wave.wait()
                return web.Response(status=401)
            start_s, end_s = request.headers["Range"].removeprefix("bytes=").split("-")
            start, end = int(start_s), int(end_s)
            body = payload[start : end + 1]
            return web.Response(
                status=206,
                body=body,
                headers={
                    "Content-Range": f"bytes {start}-{end}/10",
                    "Content-Encoding": "identity",
                },
            )

        async def refresh() -> str:
            refresh_calls["n"] += 1
            return "fresh"

        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import (
                RangeFetcher,
                RangeSpan,
                fetch_ranges_async,
            )

            url = str(server.make_url("/obj"))
            async with RangeFetcher(token="stale", refresh=refresh) as fetcher:
                return await asyncio.gather(
                    fetch_ranges_async(
                        (RangeSpan(url=url, start=2, end=5, label="a"),),
                        fetcher=fetcher,
                    ),
                    fetch_ranges_async(
                        (RangeSpan(url=url, start=6, end=9, label="b"),),
                        fetcher=fetcher,
                    ),
                )
        finally:
            await server.close()

    first, second = asyncio.run(_run())
    assert first == {"a": b"CDEF"}
    assert second == {"b": b"GHIJ"}
    assert refresh_calls["n"] == 1


def test_second_401_after_refresh_fails_the_span() -> None:
    """A fresh token that still 401s is terminal; refresh is not a loop."""
    hits: list[str] = []
    refresh_calls = {"n": 0}

    async def handler(request: web.Request) -> web.StreamResponse:
        hits.append(request.headers.get("Authorization", ""))
        return web.Response(status=401)

    async def refresh() -> str:
        refresh_calls["n"] += 1
        return "fresh"

    async def _run() -> dict[str, bytes]:
        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import (
                RangeFetcher,
                RangeSpan,
                fetch_ranges_async,
            )

            url = str(server.make_url("/obj"))
            async with RangeFetcher(token="stale", refresh=refresh) as fetcher:
                return await fetch_ranges_async(
                    (RangeSpan(url=url, start=2, end=5, label="a"),),
                    fetcher=fetcher,
                )
        finally:
            await server.close()

    with pytest.raises(RemoteAccessError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "unauthorized"
    assert refresh_calls["n"] == 1
    assert hits == ["Bearer stale", "Bearer fresh"]


def test_range_fetcher_assembles_out_of_order_spans_by_label() -> None:
    """Two in-flight spans that finish out of order still assemble by label."""
    payload = b"ABCDEFGHIJ"
    finished: list[str] = []

    async def _run() -> dict[str, bytes]:
        arrived = asyncio.Barrier(2)

        async def handler(request: web.Request) -> web.StreamResponse:
            start_s, end_s = request.headers["Range"].removeprefix("bytes=").split("-")
            start, end = int(start_s), int(end_s)
            label = "slow" if start == 0 else "fast"
            await arrived.wait()
            if label == "slow":
                await asyncio.sleep(0.05)
            finished.append(label)
            body = payload[start : end + 1]
            return web.Response(
                status=206,
                body=body,
                headers={
                    "Content-Range": f"bytes {start}-{end}/10",
                    "Content-Encoding": "identity",
                },
            )

        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeFetcher, RangeSpan

            url = str(server.make_url("/obj"))
            async with RangeFetcher() as fetcher:
                return await asyncio.wait_for(
                    fetcher.fetch(
                        (
                            RangeSpan(url=url, start=0, end=3, label="slow"),
                            RangeSpan(url=url, start=4, end=7, label="fast"),
                        )
                    ),
                    timeout=2,
                )
        finally:
            await server.close()

    assembled = asyncio.run(_run())
    assert finished == ["fast", "slow"]
    assert assembled == {"slow": b"ABCD", "fast": b"EFGH"}


def test_aimd_window_ignores_401_and_halves_on_429() -> None:
    """401 is auth expiry, not congestion; 429 halves the shared AIMD window."""
    payload = b"ABCDEFGHIJ"
    rate_hits = {"n": 0}

    async def _run() -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        async def handler(request: web.Request) -> web.StreamResponse:
            auth = request.headers.get("Authorization", "")
            if auth == "Bearer stale":
                return web.Response(status=401)
            if request.path == "/rate":
                rate_hits["n"] += 1
                if rate_hits["n"] == 1:
                    return web.Response(status=429)
            start_s, end_s = request.headers["Range"].removeprefix("bytes=").split("-")
            start, end = int(start_s), int(end_s)
            body = payload[start : end + 1]
            return web.Response(
                status=206,
                body=body,
                headers={
                    "Content-Range": f"bytes {start}-{end}/10",
                    "Content-Encoding": "identity",
                },
            )

        async def refresh() -> str:
            return "fresh"

        app = web.Application()
        app.router.add_get("/auth", handler)
        app.router.add_get("/rate", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeFetcher, RangeSpan

            async with RangeFetcher(token="stale", refresh=refresh) as fetcher:
                seed = fetcher.window, fetcher._session.connector.limit
                auth_url = str(server.make_url("/auth"))
                after_401 = await fetcher.fetch(
                    (RangeSpan(url=auth_url, start=2, end=5, label="a"),)
                )
                assert after_401 == {"a": b"CDEF"}
                after_auth = fetcher.window, fetcher._session.connector.limit
                rate_url = str(server.make_url("/rate"))
                after_429 = await fetcher.fetch(
                    (RangeSpan(url=rate_url, start=6, end=9, label="b"),)
                )
                assert after_429 == {"b": b"GHIJ"}
                after_rate = fetcher.window, fetcher._session.connector.limit
                return seed, after_auth, after_rate
        finally:
            await server.close()

    seed, after_auth, after_rate = asyncio.run(_run())
    assert seed == (20, 20)
    assert after_auth == (20, 20)
    assert after_rate == (10, 10)


def test_range_fetcher_has_no_chunk_bytes() -> None:
    """RangeFetcher concatenates already-sized spans; it has no chunk_bytes."""
    import inspect
    from dataclasses import fields, is_dataclass

    from faninsar.remote.fetch import RangeFetcher

    assert "chunk_bytes" not in inspect.signature(RangeFetcher).parameters
    assert not hasattr(RangeFetcher, "chunk_bytes")
    fetcher = RangeFetcher()
    assert "chunk_bytes" not in vars(fetcher)
    assert not hasattr(fetcher, "chunk_bytes")
    if is_dataclass(RangeFetcher):
        assert all(field.name != "chunk_bytes" for field in fields(RangeFetcher))


def test_range_get_uses_system_https_proxy_without_trust_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A system HTTPS proxy receives the range GET; trust_env stays closed."""
    payload = b"ABCDEFGHIJ"
    seen: dict[str, object] = {}

    async def origin(request: web.Request) -> web.StreamResponse:
        del request
        pytest.fail("range GET bypassed the system HTTPS proxy")

    async def proxy(request: web.Request) -> web.StreamResponse:
        seen["method"] = request.method
        seen["host"] = request.headers.get("Host")
        seen["range"] = request.headers.get("Range")
        seen["authorization"] = request.headers.get("Authorization")
        body = payload[2:6]
        return web.Response(
            status=206,
            body=body,
            headers={"Content-Range": "bytes 2-5/10", "Content-Encoding": "identity"},
        )

    async def _run() -> dict[str, bytes]:
        proxy_app = web.Application()
        proxy_app.router.add_route("*", "/{tail:.*}", proxy)
        origin_app = web.Application()
        origin_app.router.add_get("/obj", origin)
        proxy_server = TestServer(proxy_app, host="127.0.0.1")
        origin_server = TestServer(origin_app, host="127.0.0.1")
        await proxy_server.start_server()
        await origin_server.start_server()
        try:
            from faninsar.remote.fetch import (
                RangeFetcher,
                RangeSpan,
                fetch_ranges_async,
            )

            proxy_url = f"http://{proxy_server.host}:{proxy_server.port}"
            monkeypatch.setattr(
                "urllib.request.getproxies",
                lambda: {"http": proxy_url, "https": proxy_url},
            )
            monkeypatch.setattr("urllib.request.proxy_bypass", lambda _host: False)
            async with RangeFetcher(token="cdse-token") as fetcher:
                seen["trust_env"] = fetcher._session.trust_env
                return await fetch_ranges_async(
                    (
                        RangeSpan(
                            url="http://zipper.example.invalid/obj",
                            start=2,
                            end=5,
                            label="a",
                        ),
                    ),
                    fetcher=fetcher,
                )
        finally:
            await origin_server.close()
            await proxy_server.close()

    assert asyncio.run(_run()) == {"a": b"CDEF"}
    assert seen["trust_env"] is False
    assert seen["range"] == "bytes=2-5"
    assert seen["authorization"] == "Bearer cdse-token"


def test_range_get_bypasses_system_proxy_for_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loopback origins stay direct even when a system HTTP proxy is set."""
    payload = b"ABCDEFGHIJ"

    async def origin(request: web.Request) -> web.StreamResponse:
        body = payload[2:6]
        return web.Response(
            status=206,
            body=body,
            headers={
                "Content-Range": "bytes 2-5/10",
                "Content-Encoding": "identity",
                "Range-Echo": request.headers.get("Range", ""),
            },
        )

    async def proxy(request: web.Request) -> web.StreamResponse:
        del request
        pytest.fail("loopback range GET was sent through the system proxy")

    async def _run() -> dict[str, bytes]:
        proxy_app = web.Application()
        proxy_app.router.add_route("*", "/{tail:.*}", proxy)
        origin_app = web.Application()
        origin_app.router.add_get("/obj", origin)
        proxy_server = TestServer(proxy_app, host="127.0.0.1")
        origin_server = TestServer(origin_app, host="127.0.0.1")
        await proxy_server.start_server()
        await origin_server.start_server()
        try:
            from faninsar.remote.fetch import (
                RangeFetcher,
                RangeSpan,
                fetch_ranges_async,
            )

            proxy_url = f"http://{proxy_server.host}:{proxy_server.port}"
            monkeypatch.setattr(
                "urllib.request.getproxies",
                lambda: {"http": proxy_url, "https": proxy_url},
            )
            monkeypatch.setattr("urllib.request.proxy_bypass", lambda _host: False)
            async with RangeFetcher(token="cdse-token") as fetcher:
                return await fetch_ranges_async(
                    (
                        RangeSpan(
                            url=f"http://{origin_server.host}:{origin_server.port}/obj",
                            start=2,
                            end=5,
                            label="a",
                        ),
                    ),
                    fetcher=fetcher,
                )
        finally:
            await origin_server.close()
            await proxy_server.close()

    assert asyncio.run(_run()) == {"a": b"CDEF"}


def test_https_proxy_url_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTPS proxy URLs are dropped because aiohttp cannot CONNECT through them."""
    from faninsar.remote.fetch.ranges import _proxy_for_url

    monkeypatch.setattr(
        "urllib.request.getproxies",
        lambda: {"https": "https://proxy.example.invalid:8443"},
    )
    monkeypatch.setattr("urllib.request.proxy_bypass", lambda _host: False)
    assert _proxy_for_url("https://zipper.example.invalid/obj") is None


class _FakeStreamContent:
    """Chunk source shaped like ``response.content`` for the fake response."""

    def __init__(self, body: bytes, *, fail_after: int | None = None) -> None:
        self._body = body
        self._fail_after = fail_after
        self._drop_message = "simulated mid-stream drop"

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        import aiohttp

        delivered = 0
        for offset in range(0, len(self._body), n):
            chunk = self._body[offset : offset + n]
            if self._fail_after is not None:
                remaining = self._fail_after - delivered
                if remaining <= 0:
                    raise aiohttp.ClientPayloadError(self._drop_message)
                if len(chunk) > remaining:
                    yield chunk[:remaining]
                    raise aiohttp.ClientPayloadError(self._drop_message)
            delivered += len(chunk)
            yield chunk
        if self._fail_after is not None and delivered <= self._fail_after:
            raise aiohttp.ClientPayloadError(self._drop_message)


class _FakeRangeResponse:
    """One 206 response with the exact Content-Range the span expects."""

    def __init__(self, body: bytes, *, fail_after: int | None = None) -> None:
        self._body = body
        self.status = 206
        self.headers = {
            "Content-Range": f"bytes 2-{2 + len(body) - 1}/{2 + len(body) - 1}",
        }
        self.content = _FakeStreamContent(body, fail_after=fail_after)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def read(self) -> bytes:
        return self._body


class _FlakySession:
    """Fake session whose first GETs drop the connector, then succeed."""

    def __init__(self, failures: int, error: Exception) -> None:
        self._failures = failures
        self._error = error
        self.calls = 0

    def get(self, url: str, **kwargs: object) -> _FakeRangeResponse:
        del url, kwargs
        self.calls += 1
        if self.calls <= self._failures:
            raise self._error
        return _FakeRangeResponse(b"CDEF")


class _RangedFakeResponse:
    """Fake 206 echoing the request's Range header through a chunked body."""

    def __init__(
        self,
        payload: bytes,
        request_start: int,
        request_end: int,
        *,
        fail_after: int | None = None,
        content_range: str | None = None,
    ) -> None:
        self.status = 206
        body = payload[request_start : request_end + 1]
        self.headers = {
            "Content-Range": content_range
            or f"bytes {request_start}-{request_end}/{len(payload)}"
        }
        self.content = _FakeStreamContent(body, fail_after=fail_after)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _RangedSession:
    """Session that parses Range headers and can drop one stream mid-body."""

    def __init__(
        self,
        payload: bytes,
        *,
        drop_at_call: int | None = None,
        fail_after: int | None = None,
        content_range: str | None = None,
    ) -> None:
        self._payload = payload
        self._drop_at_call = drop_at_call
        self._fail_after = fail_after
        self._content_range = content_range
        self.calls = 0
        self.requests: list[str] = []

    def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
        del url
        headers = kwargs.get("headers") or {}
        start_s, end_s = str(headers.get("Range", "bytes=0-0"))[len("bytes=") :].split(
            "-"
        )
        self.calls += 1
        self.requests.append(f"{start_s}-{end_s}")
        fail_after = (
            self._fail_after
            if self._drop_at_call is None or self.calls == self._drop_at_call
            else None
        )
        return _RangedFakeResponse(
            self._payload,
            int(start_s),
            int(end_s),
            fail_after=fail_after,
            content_range=self._content_range,
        )


def test_range_span_retries_transient_connector_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy CONNECT drop is retried and the span still completes."""
    import aiohttp

    from faninsar.remote.fetch import RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_CONNECT_BACKOFF_S", 0)
    error = aiohttp.ClientConnectorError(
        connection_key=None,
        os_error=OSError(61, "Connection refused"),
    )
    session = _FlakySession(failures=1, error=error)
    span = RangeSpan(
        url="https://zipper.example.invalid/obj", start=2, end=5, label="a"
    )

    async def _run() -> bytes:
        return await ranges_module._request_span(session, span)

    assert asyncio.run(_run()) == b"CDEF"
    assert session.calls == 2


def test_range_span_connector_failures_raise_after_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent connection drops surface the connector error, not a hang."""
    import aiohttp

    from faninsar.remote.fetch import RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_CONNECT_BACKOFF_S", 0)
    error = aiohttp.ClientOSError("SSL: UNEXPECTED_EOF_WHILE_READING")
    session = _FlakySession(failures=99, error=error)
    span = RangeSpan(
        url="https://zipper.example.invalid/obj", start=2, end=5, label="a"
    )

    async def _run() -> bytes:
        return await ranges_module._request_span(session, span)

    with pytest.raises(aiohttp.ClientOSError):
        asyncio.run(_run())
    assert session.calls == ranges_module._CONNECT_ATTEMPTS


def test_stream_span_resumes_suffix_after_mid_stream_payload_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-stream ClientPayloadError resumes with the missing suffix range."""
    from faninsar.remote.fetch import RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_CONNECT_BACKOFF_S", 0)
    payload = b"ABCDEFGHIJ"
    session = _RangedSession(payload, drop_at_call=1, fail_after=2)
    span = RangeSpan(
        url="https://zipper.example.invalid/obj", start=2, end=9, label="a"
    )
    body = bytearray()

    async def _run() -> None:
        await ranges_module._stream_span(session, span, body.extend, lambda: len(body))

    asyncio.run(_run())
    assert bytes(body) == payload[2:10]
    assert session.requests == ["2-9", "4-9"]
    assert session.calls == 2


def test_stream_span_persistent_payload_drops_raise_after_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Payload drops on every attempt surface after the bounded retry budget."""
    import aiohttp

    from faninsar.remote.fetch import RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_CONNECT_BACKOFF_S", 0)
    session = _RangedSession(b"ABCDEFGHIJ", fail_after=1)
    span = RangeSpan(
        url="https://zipper.example.invalid/obj", start=0, end=9, label="a"
    )
    body = bytearray()

    async def _run() -> None:
        await ranges_module._stream_span(session, span, body.extend, lambda: len(body))

    with pytest.raises(aiohttp.ClientPayloadError):
        asyncio.run(_run())
    assert session.calls == ranges_module._CONNECT_ATTEMPTS
    assert session.requests[0] == "0-9"
    # Each resumed attempt asks only for the missing suffix; the last
    # request narrows to the single byte still unreceived.
    assert session.requests[-1] == "9-9"
    assert bytes(body) == b"ABCDEFGHIJ"


def test_request_span_once_rejects_wrong_content_range_start() -> None:
    """A 206 whose Content-Range start differs is not a prefix match — it fails."""
    from faninsar.remote.fetch import RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    session = _RangedSession(b"ABCDEFGHIJ", content_range="bytes 2-9/10")
    span = RangeSpan(
        url="https://zipper.example.invalid/obj", start=0, end=9, label="a"
    )
    body = bytearray()

    async def _run() -> None:
        await ranges_module._request_span_once(session, span, body.extend, 0)

    with pytest.raises(RemoteIntegrityError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "content_range_mismatch"
    assert bytes(body) == b""


def test_request_span_once_rejects_non_206_status() -> None:
    """A 200 response masquerading as range support fails strict validation."""
    from faninsar.remote.fetch import RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    class _OkStatusSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            response = super().get(url, **kwargs)
            response.status = 200
            return response

    ok_session = _OkStatusSession(b"ABCDEFGHIJ")
    span = RangeSpan(
        url="https://zipper.example.invalid/obj", start=0, end=9, label="a"
    )
    body = bytearray()

    async def _run() -> None:
        await ranges_module._request_span_once(ok_session, span, body.extend, 0)

    with pytest.raises(RemoteIntegrityError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "content_range_mismatch"


def test_range_fetcher_fetch_into_writes_a_part_file(tmp_path: Path) -> None:
    """fetch_into streams the 206 body into a caller-provided part file."""
    from aiohttp import web

    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    payload = b"ABCDEFGHIJ"
    part = tmp_path / "span.part"

    async def handler(request: web.Request) -> web.StreamResponse:
        assert request.headers["Range"] == "bytes=2-9"
        return web.Response(
            status=206,
            body=payload[2:10],
            headers={
                "Content-Range": "bytes 2-9/10",
                "Content-Encoding": "identity",
            },
        )

    async def _run() -> int:
        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            url = str(server.make_url("/obj"))
            async with RangeFetcher() as fetcher:
                return await fetcher.fetch_into(
                    RangeSpan(url=url, start=2, end=9, label="a"), part
                )
        finally:
            await server.close()

    assert asyncio.run(_run()) == 8
    assert part.read_bytes() == payload[2:10]


def test_range_fetcher_fetch_into_resumes_an_existing_part_file(
    tmp_path: Path,
) -> None:
    """A pre-seeded part file resumes at start+received, not at span start."""
    from aiohttp import web

    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    payload = b"ABCDEFGHIJ"
    part = tmp_path / "span.part"
    part.write_bytes(payload[2:4])
    seen_ranges: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        seen_ranges.append(request.headers["Range"])
        assert request.headers["Range"] == "bytes=4-9"
        return web.Response(
            status=206,
            body=payload[4:10],
            headers={
                "Content-Range": "bytes 4-9/10",
                "Content-Encoding": "identity",
            },
        )

    async def _run() -> int:
        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            url = str(server.make_url("/obj"))
            async with RangeFetcher() as fetcher:
                return await fetcher.fetch_into(
                    RangeSpan(url=url, start=2, end=9, label="a"), part
                )
        finally:
            await server.close()

    assert asyncio.run(_run()) == 8
    assert seen_ranges == ["bytes=4-9"]
    assert part.read_bytes() == payload[2:10]


def test_range_fetcher_fetch_into_rejects_an_oversized_part_file(
    tmp_path: Path,
) -> None:
    """A part file already larger than the span is an integrity failure."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    part = tmp_path / "span.part"
    part.write_bytes(b"0123456789ABCDEF")

    async def _run() -> int:
        async with RangeFetcher() as fetcher:
            return await fetcher.fetch_into(
                RangeSpan(
                    url="https://zipper.example.invalid/obj",
                    start=0,
                    end=3,
                    label="a",
                ),
                part,
            )

    with pytest.raises(RemoteIntegrityError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "part_file_oversized"


def test_range_fetcher_fetch_into_accepts_a_binary_stream() -> None:
    """An io.BytesIO sink receives the body and reports its own position."""
    import io

    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    fetcher = RangeFetcher()
    fetcher._session = _RangedSession(b"ABCDEFGHIJ")
    sink = io.BytesIO()
    sink.write(b"AB")
    span = RangeSpan(
        url="https://zipper.example.invalid/obj", start=0, end=9, label="a"
    )

    async def _run() -> int:
        return await fetcher.fetch_into(span, sink)

    assert asyncio.run(_run()) == 10
    assert sink.getvalue() == b"ABCDEFGHIJ"


def test_range_fetcher_fetch_into_requires_a_tellable_writable_sink() -> None:
    """Opaque objects without write()/tell() fail before any request."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    fetcher = RangeFetcher()
    fetcher._session = object()
    span = RangeSpan(
        url="https://zipper.example.invalid/obj", start=0, end=3, label="a"
    )

    async def _run() -> int:
        return await fetcher.fetch_into(span, object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        asyncio.run(_run())
