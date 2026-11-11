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


def _aimd_206(payload: bytes, request: web.Request) -> web.StreamResponse:
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


def test_aimd_window_ignores_401_and_halves_on_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """401 is auth expiry, not congestion; 429 halves, then a success probes up."""
    monkeypatch.setattr("faninsar.remote.fetch.ranges._RATE_LIMIT_BACKOFF_S", 0)
    monkeypatch.setattr("faninsar.remote.fetch.ranges._AIMD_HOLD_S", 0)
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
            return _aimd_206(payload, request)

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
    # 20 triggered → 15 (3/4); the retry succeeded at 15 → probe toward 20 is 17.
    assert after_rate == (17, 17)


def test_aimd_window_sits_just_below_the_last_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a 429, successes binary-search up and then sit at hi-1."""
    monkeypatch.setattr("faninsar.remote.fetch.ranges._RATE_LIMIT_BACKOFF_S", 0)
    monkeypatch.setattr("faninsar.remote.fetch.ranges._AIMD_HOLD_S", 0)
    payload = b"ABCDEFGHIJ"

    async def _run() -> list[int]:
        hits = {"n": 0}

        async def handler(request: web.Request) -> web.StreamResponse:
            hits["n"] += 1
            if hits["n"] == 1:
                return web.Response(status=429)
            return _aimd_206(payload, request)

        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeFetcher, RangeSpan

            windows: list[int] = []
            async with RangeFetcher() as fetcher:
                url = str(server.make_url("/obj"))
                for i in range(6):
                    body = await fetcher.fetch(
                        (RangeSpan(url=url, start=2, end=5, label=f"s{i}"),)
                    )
                    assert body == {f"s{i}": b"CDEF"}
                    windows.append(fetcher.window)
            return windows
        finally:
            await server.close()

    # 429 at 20 → 15, success probes 17, 18, 19, then sit at 19 (hi=20).
    assert asyncio.run(_run()) == [17, 18, 19, 19, 19, 19]


def test_aimd_probe_429_searches_midpoint_not_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 on a probe cuts to (lo+hi)/2, not another multiplicative half."""
    monkeypatch.setattr("faninsar.remote.fetch.ranges._RATE_LIMIT_BACKOFF_S", 0)
    monkeypatch.setattr("faninsar.remote.fetch.ranges._AIMD_HOLD_S", 0)
    payload = b"ABCDEFGHIJ"

    async def _run() -> tuple[int, int]:
        hits = {"n": 0}

        async def handler(request: web.Request) -> web.StreamResponse:
            hits["n"] += 1
            # First fetch: 429 then 206 (20→15, success probes 17).
            # Second fetch: 429 at the probe, then 206 (walk back to 15,
            # success probes (15+17)//2 = 16).
            if hits["n"] in {1, 3}:
                return web.Response(status=429)
            return _aimd_206(payload, request)

        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeFetcher, RangeSpan

            async with RangeFetcher() as fetcher:
                url = str(server.make_url("/obj"))
                await fetcher.fetch((RangeSpan(url=url, start=2, end=5, label="a"),))
                after_first = fetcher.window
                await fetcher.fetch((RangeSpan(url=url, start=6, end=9, label="b"),))
                return after_first, fetcher.window
        finally:
            await server.close()

    assert asyncio.run(_run()) == (17, 16)


def test_concurrent_429s_cut_the_window_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-flight 429s at the same sending window share one decrease."""
    monkeypatch.setattr("faninsar.remote.fetch.ranges._RATE_LIMIT_BACKOFF_S", 0)
    monkeypatch.setattr("faninsar.remote.fetch.ranges._AIMD_HOLD_S", 0)
    payload = b"ABCDEFGHIJ"
    n = 8

    async def _run() -> int:
        wave = asyncio.Barrier(n)
        first = {"n": 0}

        async def handler(request: web.Request) -> web.StreamResponse:
            async with asyncio.Lock():
                # Count this arrival before the barrier so every first-wave
                # GET 429s together at window=20.
                first["n"] += 1
                idx = first["n"]
            if idx <= n:
                await wave.wait()
                return web.Response(status=429)
            return _aimd_206(payload, request)

        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeFetcher, RangeSpan

            url = str(server.make_url("/obj"))
            async with RangeFetcher() as fetcher:
                results = await asyncio.gather(
                    *[
                        fetcher.fetch(
                            (RangeSpan(url=url, start=2, end=5, label=f"s{i}"),)
                        )
                        for i in range(n)
                    ]
                )
                assert all(row == {f"s{i}": b"CDEF"} for i, row in enumerate(results))
                return fetcher.window
        finally:
            await server.close()

    # One shared 3/4 20→15, then one shared probe to 17 — not eight cuts to 1.
    assert asyncio.run(_run()) == 17


def test_repeated_429s_on_one_span_do_not_collapse_past_the_first_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retries of the same 429 wave do not keep halving 20→10→5→2→1."""
    monkeypatch.setattr("faninsar.remote.fetch.ranges._RATE_LIMIT_BACKOFF_S", 0)
    monkeypatch.setattr("faninsar.remote.fetch.ranges._AIMD_HOLD_S", 0)
    payload = b"ABCDEFGHIJ"

    async def _run() -> tuple[int, int]:
        hits = {"n": 0}

        async def handler(request: web.Request) -> web.StreamResponse:
            hits["n"] += 1
            if hits["n"] <= 4:
                return web.Response(status=429)
            return _aimd_206(payload, request)

        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeFetcher, RangeSpan

            async with RangeFetcher() as fetcher:
                url = str(server.make_url("/obj"))
                body = await fetcher.fetch(
                    (RangeSpan(url=url, start=2, end=5, label="a"),)
                )
                assert body == {"a": b"CDEF"}
                return hits["n"], fetcher.window
        finally:
            await server.close()

    hits, window = asyncio.run(_run())
    assert hits == 5
    # First 429 cuts 20→15; the next three retries do not cut again;
    # the 206 at 15 probes toward 20.
    assert window == 17


def test_inflight_success_at_the_old_window_does_not_reopen_the_429_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 206 already in flight at 20 must not let leftover 429s keep cutting."""
    monkeypatch.setattr("faninsar.remote.fetch.ranges._RATE_LIMIT_BACKOFF_S", 0)
    monkeypatch.setattr("faninsar.remote.fetch.ranges._AIMD_HOLD_S", 0)
    payload = b"ABCDEFGHIJ"

    async def _run() -> int:
        slow_hits = {"n": 0}
        first_429 = asyncio.Event()
        holder: dict[str, object] = {}

        async def slow(request: web.Request) -> web.StreamResponse:
            slow_hits["n"] += 1
            if slow_hits["n"] == 1:
                first_429.set()
            if slow_hits["n"] <= 4:
                return web.Response(status=429)
            return _aimd_206(payload, request)

        async def fast(request: web.Request) -> web.StreamResponse:
            await first_429.wait()
            fetcher = holder["fetcher"]
            while fetcher.window == 20:  # type: ignore[attr-defined]
                await asyncio.sleep(0)
            return _aimd_206(payload, request)

        app = web.Application()
        app.router.add_get("/slow", slow)
        app.router.add_get("/fast", fast)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeFetcher, RangeSpan

            async with RangeFetcher() as fetcher:
                holder["fetcher"] = fetcher
                slow_url = str(server.make_url("/slow"))
                fast_url = str(server.make_url("/fast"))
                slow_body, fast_body = await asyncio.gather(
                    fetcher.fetch(
                        (RangeSpan(url=slow_url, start=2, end=5, label="slow"),)
                    ),
                    fetcher.fetch(
                        (RangeSpan(url=fast_url, start=6, end=9, label="fast"),)
                    ),
                )
                assert slow_body == {"slow": b"CDEF"}
                assert fast_body == {"fast": b"GHIJ"}
                return fetcher.window
        finally:
            await server.close()

    # Fast 206 was sent at 20; it is not evidence that the reduced window is
    # safe, so Slow's leftover 429s must not keep cutting 15→10→7→1.
    assert asyncio.run(_run()) == 17


def test_aimd_hold_blocks_probe_while_429s_are_still_arriving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long hold keeps leftover 429s from looking like a failed probe."""
    monkeypatch.setattr("faninsar.remote.fetch.ranges._RATE_LIMIT_BACKOFF_S", 0)
    monkeypatch.setattr("faninsar.remote.fetch.ranges._AIMD_HOLD_S", 60)
    payload = b"ABCDEFGHIJ"

    async def _run() -> int:
        hits = {"n": 0}

        async def handler(request: web.Request) -> web.StreamResponse:
            hits["n"] += 1
            if hits["n"] == 1:
                return web.Response(status=429)
            return _aimd_206(payload, request)

        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeFetcher, RangeSpan

            async with RangeFetcher() as fetcher:
                url = str(server.make_url("/obj"))
                await fetcher.fetch((RangeSpan(url=url, start=2, end=5, label="a"),))
                return fetcher.window
        finally:
            await server.close()

    # 20→15, then the 206 must sit at 15 until the hold elapses.
    assert asyncio.run(_run()) == 15


def test_429_on_a_sitting_ceiling_walks_back_the_climb_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later 429 at 19 walks back 18, not another 3/4 or half."""
    monkeypatch.setattr("faninsar.remote.fetch.ranges._RATE_LIMIT_BACKOFF_S", 0)
    monkeypatch.setattr("faninsar.remote.fetch.ranges._AIMD_HOLD_S", 0)
    payload = b"ABCDEFGHIJ"

    async def _run() -> list[int]:
        hits = {"n": 0}

        async def handler(request: web.Request) -> web.StreamResponse:
            hits["n"] += 1
            if hits["n"] in {1, 8}:
                return web.Response(status=429)
            return _aimd_206(payload, request)

        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeFetcher, RangeSpan

            windows: list[int] = []
            async with RangeFetcher() as fetcher:
                url = str(server.make_url("/obj"))
                for i in range(8):
                    body = await fetcher.fetch(
                        (RangeSpan(url=url, start=2, end=5, label=f"s{i}"),)
                    )
                    assert body == {f"s{i}": b"CDEF"}
                    windows.append(fetcher.window)
            return windows
        finally:
            await server.close()

    # Climb 15→17→18→19, sit, then 429 at 19 walks back to 18.
    assert asyncio.run(_run()) == [17, 18, 19, 19, 19, 19, 18, 18]


def test_second_virgin_429_halves_the_original_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no climb path, 20→15, and a 429 at 15 goes to 10 (half of 20)."""
    monkeypatch.setattr("faninsar.remote.fetch.ranges._RATE_LIMIT_BACKOFF_S", 0)
    monkeypatch.setattr("faninsar.remote.fetch.ranges._AIMD_HOLD_S", 60)
    payload = b"ABCDEFGHIJ"

    async def _run() -> tuple[int, int]:
        hits = {"n": 0}

        async def handler(request: web.Request) -> web.StreamResponse:
            hits["n"] += 1
            if hits["n"] in {1, 3}:
                return web.Response(status=429)
            return _aimd_206(payload, request)

        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            from faninsar.remote.fetch import RangeFetcher, RangeSpan

            async with RangeFetcher() as fetcher:
                url = str(server.make_url("/obj"))
                await fetcher.fetch((RangeSpan(url=url, start=2, end=5, label="a"),))
                after_first = fetcher.window
                fetcher._hold_until = 0.0
                await fetcher.fetch((RangeSpan(url=url, start=6, end=9, label="b"),))
                return after_first, fetcher.window
        finally:
            await server.close()

    assert asyncio.run(_run()) == (15, 10)


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

    def __init__(
        self,
        body: bytes,
        *,
        fail_after: int | None = None,
        drop_exc: BaseException | None = None,
    ) -> None:
        self._body = body
        self._fail_after = fail_after
        self._drop_exc = drop_exc
        self._drop_message = "simulated mid-stream drop"

    def _drop(self) -> None:
        import aiohttp

        if self._drop_exc is not None:
            raise self._drop_exc
        raise aiohttp.ClientPayloadError(self._drop_message)

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        delivered = 0
        for offset in range(0, len(self._body), n):
            chunk = self._body[offset : offset + n]
            if self._fail_after is not None:
                remaining = self._fail_after - delivered
                if remaining <= 0:
                    self._drop()
                if len(chunk) > remaining:
                    yield chunk[:remaining]
                    self._drop()
            delivered += len(chunk)
            yield chunk
        if self._fail_after is not None and delivered <= self._fail_after:
            self._drop()


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
        drop_exc: BaseException | None = None,
    ) -> None:
        self.status = 206
        body = payload[request_start : request_end + 1]
        self.headers = {
            "Content-Range": content_range
            or f"bytes {request_start}-{request_end}/{len(payload)}"
        }
        self.content = _FakeStreamContent(
            body, fail_after=fail_after, drop_exc=drop_exc
        )

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
        drop_exc: BaseException | None = None,
    ) -> None:
        self._payload = payload
        self._drop_at_call = drop_at_call
        self._fail_after = fail_after
        self._content_range = content_range
        self._drop_exc = drop_exc
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
            drop_exc=self._drop_exc,
        )


class _HeldRangeSession:
    """Async fake whose live responses wait until the test releases them."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.releases: dict[str, asyncio.Event] = {}
        self.active = 0
        self.by_url: dict[str, int] = {}
        self.peak = 0
        self.peaks: dict[str, int] = {}
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: object) -> _HeldRangeResponse:
        del kwargs
        self.calls.append(url)
        return _HeldRangeResponse(self, url)


class _HeldRangeResponse(_RangedFakeResponse):
    """Hold an actual request-context await, including cancellation cleanup."""

    def __init__(self, session: _HeldRangeSession, url: str) -> None:
        super().__init__(b"ABCDEFGHIJ", 2, 5)
        self.session = session
        self.url = url

    async def __aenter__(self) -> Self:
        session = self.session
        session.active += 1
        session.by_url[self.url] = session.by_url.get(self.url, 0) + 1
        session.peak = max(session.peak, session.active)
        session.peaks[self.url] = max(
            session.peaks.get(self.url, 0), session.by_url[self.url]
        )
        try:
            await session.releases.get(self.url, session.release).wait()
        except BaseException:
            self._finish()
            raise
        return self

    def _finish(self) -> None:
        self.session.active -= 1
        self.session.by_url[self.url] -= 1

    async def __aexit__(self, *exc_info: object) -> None:
        self._finish()


@pytest.mark.parametrize("files", [1, 4])
def test_metadata_slots_overlap_payload_then_restore_all_payload_files(
    files: int,
) -> None:
    """Pending annotations reserve client file slots until every XML is drained."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        metadata_slots = min(2, files)
        session = _HeldRangeSession()
        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(
                max_workers=16,
                max_ranges_per_file=4,
                max_active_files=files,
                max_metadata_files=metadata_slots,
            )
        )
        fetcher._session = session
        # URLs deliberately say nothing about type; only annotation labels classify.
        xmls = [f"https://example.test/object-{i}" for i in range(metadata_slots + 1)]
        payloads = [f"https://example.test/object-{i + 10}" for i in range(files)]
        session.releases = {url: asyncio.Event() for url in xmls}
        spans = [
            RangeSpan(url, 2, 5, f"payload-{i}-{j}")
            for i, url in enumerate(payloads)
            for j in range(4)
        ] + [RangeSpan(url, 2, 5, f"annotation-{i}") for i, url in enumerate(xmls)]
        task = asyncio.create_task(fetcher.fetch(spans))
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            active = {url for url, count in session.by_url.items() if count}
            assert len(active) == files
            assert len(active.intersection(xmls)) == metadata_slots
            assert len(active.intersection(payloads)) == files - metadata_slots
            for url in xmls:
                session.releases[url].set()
            for _ in range(15):
                await asyncio.sleep(0)
            assert {url for url, count in session.by_url.items() if count} == set(
                payloads
            )
            assert all(session.by_url[url] == 4 for url in payloads)
            assert session.active == files * 4
            assert session.peak <= 16
        finally:
            session.release.set()
            for event in session.releases.values():
                event.set()
            assert len(await asyncio.wait_for(task, 2)) == len(spans)

    asyncio.run(_run())


def test_metadata_http_slots_cannot_be_taken_by_payload_first_ranges() -> None:
    """H4/R4 leaves room for both XMLs even when a payload offers four ranges."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        session = _HeldRangeSession()
        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(
                max_workers=4,
                max_active_files=4,
                max_metadata_files=2,
                max_ranges_per_file=4,
            )
        )
        fetcher._session = session
        payload = "https://example.test/p"
        xmls = ["https://example.test/x1", "https://example.test/x2"]
        session.releases = {url: asyncio.Event() for url in xmls}
        task = asyncio.create_task(
            fetcher.fetch(
                [RangeSpan(payload, 2, 5, f"payload-{i}") for i in range(8)]
                + [
                    RangeSpan(url, 2, 5, f"annotation-{i}")
                    for i, url in enumerate(xmls)
                ]
            )
        )
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert all(session.by_url.get(url) == 1 for url in xmls)
            assert session.active == 4
            assert session.by_url[payload] == 2
            for event in session.releases.values():
                event.set()
            for _ in range(15):
                await asyncio.sleep(0)
            assert session.by_url[payload] == 4
            assert session.active == 4
        finally:
            session.release.set()
            for event in session.releases.values():
                event.set()
            await asyncio.wait_for(task, 2)

    asyncio.run(_run())


@pytest.mark.parametrize("workers", [1, 4])
def test_metadata_slots_respect_a_small_live_http_window(workers: int) -> None:
    """A shared H1 or H4 window admits XML before payload can refill it."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        session = _HeldRangeSession()
        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(
                max_workers=workers,
                max_active_files=4,
                max_metadata_files=2,
                max_ranges_per_file=4,
            )
        )
        fetcher._session = session
        xmls = ["https://example.test/x1", "https://example.test/x2"]
        task = asyncio.create_task(
            fetcher.fetch(
                [
                    RangeSpan("https://example.test/p", 2, 5, f"payload-{i}")
                    for i in range(4)
                ]
                + [
                    RangeSpan(url, 2, 5, f"annotation-{i}")
                    for i, url in enumerate(xmls)
                ]
            )
        )
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.active == workers
            assert session.by_url.get(xmls[0]) == 1
            assert session.by_url.get("https://example.test/p", 0) == max(
                0, workers - 2
            )
        finally:
            session.release.set()
            await asyncio.wait_for(task, 2)
        assert session.peak <= workers

    asyncio.run(_run())


@pytest.mark.parametrize("workers", [4, 16])
def test_late_metadata_waits_for_payload_to_drain_without_preemption(workers: int) -> None:
    """Four in-flight payload URLs drain naturally before two XMLs are admitted."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        session = _HeldRangeSession()
        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(
                max_workers=16,
                max_active_files=4,
                max_metadata_files=2,
                max_ranges_per_file=4,
            )
        )
        fetcher._session = session
        payloads = [f"https://example.test/p-{i}" for i in range(4)]
        session.releases = {url: asyncio.Event() for url in payloads}
        main = asyncio.create_task(
            fetcher.fetch(
                [RangeSpan(url, 2, 5, f"payload-{i}") for i, url in enumerate(payloads)]
            )
        )
        tasks = [main]
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.active == 4
            xmls = ["https://example.test/a", "https://example.test/b"]
            metadata = asyncio.create_task(
                fetcher.fetch(
                    [
                        RangeSpan(url, 2, 5, "annotation" if i == 0 else "annotation-1")
                        for i, url in enumerate(xmls)
                    ]
                )
            )
            replacement = asyncio.create_task(
                fetcher.fetch(
                    [
                        RangeSpan(
                            "https://example.test/replacement", 2, 5, "payload-next"
                        )
                    ]
                )
            )
            tasks.extend((metadata, replacement))
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.calls == payloads
            assert not main.done()
            for url in payloads[:2]:
                session.releases[url].set()
            for _ in range(15):
                await asyncio.sleep(0)
            active = {url for url, count in session.by_url.items() if count}
            assert active == set(payloads[2:] + xmls)
            assert "https://example.test/replacement" not in session.calls
            assert session.active == 4
            assert session.peak == 4
        finally:
            session.release.set()
            for event in session.releases.values():
                event.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 2)

    asyncio.run(_run())


@pytest.mark.parametrize("cancel_metadata", [False, True])
def test_metadata_reservations_clear_after_failure_or_cancellation(
    cancel_metadata: bool,
) -> None:
    """Queued and active XML cleanup wakes payload without leaked file slots."""
    import io

    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        class PolicySession(_HeldRangeSession):
            def get(self, url: str, **kwargs: object) -> _HeldRangeResponse:
                response = super().get(url, **kwargs)
                if not cancel_metadata and url == xml:
                    response.headers["Content-Range"] = "invalid"
                return response

        session = PolicySession()
        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(
                max_workers=4,
                max_active_files=1,
                max_metadata_files=1,
                max_ranges_per_file=4,
            )
        )
        fetcher._session = session
        xml = "https://example.test/opaque-a"
        session.releases[xml] = asyncio.Event()
        spans = [
            RangeSpan(xml, 2, 5, "annotation"),
            RangeSpan("https://example.test/opaque-b", 2, 5, "annotation-1"),
        ]
        metadata = asyncio.create_task(fetcher.fetch(spans))
        payload = asyncio.create_task(
            fetcher.fetch_into(
                RangeSpan("https://example.test/opaque-c", 2, 5, "payload"),
                io.BytesIO(),
            )
        )
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.calls == [xml]
            if cancel_metadata:
                metadata.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await metadata
            else:
                session.releases[xml].set()
                with pytest.raises(RemoteIntegrityError):
                    await metadata
            for _ in range(15):
                await asyncio.sleep(0)
            assert session.by_url["https://example.test/opaque-c"] == 1
            session.release.set()
            assert await asyncio.wait_for(payload, 2) == 4
            assert session.active == 0
        finally:
            session.release.set()
            metadata.cancel()
            payload.cancel()
            await asyncio.gather(metadata, payload, return_exceptions=True)

    asyncio.run(_run())


def test_none_metadata_policy_keeps_distinct_url_admission_unchanged() -> None:
    """Opt-out preserves all four active URLs even with F1 and XML labels."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        session = _HeldRangeSession()
        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(
                max_workers=4,
                max_active_files=1,
                max_metadata_files=None,
            )
        )
        fetcher._session = session
        task = asyncio.create_task(
            fetcher.fetch(
                [
                    RangeSpan(f"https://example.test/{i}", 2, 5, f"annotation-{i}")
                    for i in range(4)
                ]
            )
        )
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.active == 4
            assert len(session.by_url) == 4
        finally:
            session.release.set()
            await asyncio.wait_for(task, 2)

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("workers", "final_stage"), [(16, False), (6, False), (16, True)]
)
def test_per_file_ranges_share_one_global_http_window(
    workers: int, final_stage: bool
) -> None:
    """Four exact URLs can reach R4/H16, never per-file times global workers."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        session = _HeldRangeSession()
        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(max_workers=workers, max_ranges_per_file=4)
        )
        fetcher._session = session
        if final_stage:
            fetcher.configure_rate_limit_policy(max_retries_per_object=0, max_workers=1)
        ceiling = 1 if final_stage else workers
        urls = [f"https://example.test/measurement-{i}" for i in range(4)]
        spans = [RangeSpan(url, 2, 5, f"{url}-{i}") for url in urls for i in range(8)]
        task = asyncio.create_task(fetcher.fetch(spans))
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.active == ceiling
            assert session.peak == ceiling
            assert all(count <= 4 for count in session.by_url.values())
            if ceiling == 16:
                assert session.by_url == dict.fromkeys(urls, 4)
        finally:
            session.release.set()
            results = await asyncio.wait_for(task, timeout=2)
        assert len(results) == 32
        assert session.active == 0
        assert all(count == 0 for count in session.by_url.values())
        assert session.peak == ceiling
        assert set(session.peaks) == set(urls)
        assert all(peak <= min(4, ceiling) for peak in session.peaks.values())
        if ceiling == 16:
            assert session.peaks == dict.fromkeys(urls, 4)

    asyncio.run(_run())


@pytest.mark.parametrize("budget_mode", ["omitted", "default", "explicit_none"])
def test_none_per_file_limit_preserves_global_only_admission(budget_mode: str) -> None:
    """Legacy same-URL callers retain the full shared window, not R4."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        budget = None
        if budget_mode == "default":
            budget = RemoteResourceBudget(max_workers=8)
        elif budget_mode == "explicit_none":
            budget = RemoteResourceBudget(max_workers=8, max_ranges_per_file=None)
        fetcher = RangeFetcher(budget=budget)
        session = _HeldRangeSession()
        fetcher._session = session
        ceiling = 20 if budget_mode == "omitted" else 8
        url = "https://example.test/measurement"
        spans = [RangeSpan(url, 2, 5, str(i)) for i in range(24)]
        task = asyncio.create_task(fetcher.fetch(spans))
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.active == ceiling
            assert session.by_url == {url: ceiling}
        finally:
            session.release.set()
            assert len(await asyncio.wait_for(task, timeout=2)) == 24
        assert session.peak == ceiling

    asyncio.run(_run())


def test_per_file_waiters_do_not_block_other_exact_urls() -> None:
    """A busy URL cannot reserve global slots ahead of XML or distinct queries."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        session = _HeldRangeSession()
        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(max_workers=6, max_ranges_per_file=2)
        )
        fetcher._session = session
        urls = [
            "https://example.test/measurement?part=one",
            "https://example.test/measurement?part=two",
            "https://example.test/annotation.xml",
        ]
        task = asyncio.create_task(
            fetcher.fetch(
                [RangeSpan(url, 2, 5, f"{i}-{url}") for url in urls for i in range(5)]
            )
        )
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.by_url == dict.fromkeys(urls, 2)
            assert session.active == 6
        finally:
            session.release.set()
            assert len(await asyncio.wait_for(task, timeout=2)) == 15
        assert session.peaks == dict.fromkeys(urls, 2)

    asyncio.run(_run())


@pytest.mark.parametrize("cancel_active", [False, True])
def test_per_file_cancellation_releases_only_owned_slots(cancel_active: bool) -> None:
    """Active or queued cancellation leaves both URL and global slots reusable."""
    import io

    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        session = _HeldRangeSession()
        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(max_workers=2, max_ranges_per_file=1)
        )
        fetcher._session = session
        url = "https://example.test/measurement"
        span = RangeSpan(url, 2, 5, "a")
        first = asyncio.create_task(fetcher.fetch((span,)))
        tasks = [first]
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.by_url == {url: 1}
            queued = asyncio.create_task(fetcher.fetch_into(span, io.BytesIO()))
            survivor = asyncio.create_task(fetcher.fetch((span,)))
            tasks.extend((queued, survivor))
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.calls == [url]
            cancelled = first if cancel_active else queued
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.by_url == {url: 1}
            assert len(session.calls) == (2 if cancel_active else 1)
            assert not survivor.done()
            session.release.set()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
            assert session.active == 0
            assert survivor.result() == {"a": b"CDEF"}
            session.release.clear()
            followup = asyncio.create_task(
                fetcher.fetch((span, RangeSpan(f"{url}-other", 2, 5, "b")))
            )
            tasks.append(followup)
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.active == 2
            assert session.by_url == {url: 1, f"{url}-other": 1}
            session.release.set()
            assert await asyncio.wait_for(followup, 2) == {"a": b"CDEF", "b": b"CDEF"}
            assert session.active == 0
        finally:
            session.release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(_run())


def test_object_rate_limit_exhaustion_does_not_stop_other_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eleven actual 429s defer only the exact object, not the shared session."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)
    failed_url = "https://example.test/measurement?signature=secret"

    class RateSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            response = super().get(url, **kwargs)
            if url == failed_url:
                response.status = 429
            return response

    async def _run() -> None:
        session = RateSession(b"ABCDEFGHIJ")
        fetcher = RangeFetcher()
        fetcher._session = session
        fetcher.configure_rate_limit_policy(max_retries_per_object=10)
        span = RangeSpan(failed_url, 2, 5, "a")
        for _ in range(2):
            with pytest.raises(RemoteAccessError) as raised:
                await fetcher.fetch((span,))
            assert raised.value.reason == "rate_limit_exhausted"
            assert "secret" not in str(raised.value)
            assert session.calls == 11
        assert await fetcher.fetch(
            (RangeSpan("https://example.test/annotation", 2, 5, "b"),)
        ) == {"b": b"CDEF"}
        assert session.calls == 12

    asyncio.run(_run())


def test_exhausted_object_skips_already_queued_http_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eleventh 429 fences old waiters but another URL still uses the slot."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)
    failed_url = "https://example.test/measurement"

    async def _run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[str] = []

        class RateResponse(_RangedFakeResponse):
            async def __aenter__(self) -> Self:
                if len(calls) == 1:
                    started.set()
                    await release.wait()
                return self

        class RateSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> RateResponse:
                del kwargs
                calls.append(url)
                response = RateResponse(b"ABCDEFGHIJ", 2, 5)
                if url == failed_url:
                    response.status = 429
                return response

        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(max_workers=1, max_retries=100)
        )
        fetcher._session = RateSession(b"ABCDEFGHIJ")
        fetcher.configure_rate_limit_policy(max_retries_per_object=10)
        span = RangeSpan(failed_url, 2, 5, "a")
        first = asyncio.create_task(fetcher.fetch((span,)))
        await started.wait()
        queued = [asyncio.create_task(fetcher.fetch((span,))) for _ in range(20)]
        healthy = asyncio.create_task(
            fetcher.fetch((RangeSpan("https://example.test/other", 2, 5, "b"),))
        )
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, *queued, return_exceptions=True)
        assert all(isinstance(result, RemoteAccessError) for result in results)
        assert {result.reason for result in results} == {"rate_limit_exhausted"}
        assert await healthy == {"b": b"CDEF"}
        assert calls.count(failed_url) == 11
        assert calls.count("https://example.test/other") == 1

    asyncio.run(_run())


def test_retry_stage_configuration_rejects_batch_before_children_start() -> None:
    """A started batch fences stage changes before its child spans are scheduled."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        session = _RangedSession(b"ABCDEFGHIJ")
        fetcher = RangeFetcher()
        fetcher._session = session
        fetcher.configure_rate_limit_policy(max_retries_per_object=10)
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        batch = asyncio.create_task(fetcher.fetch((span,)))
        await asyncio.sleep(0)
        try:
            assert session.calls == 0
            assert not batch.done()
            with pytest.raises(RuntimeError, match="drain"):
                fetcher.configure_rate_limit_policy(
                    max_retries_per_object=0, max_workers=1, reset_counters=True
                )
            assert await batch == {"a": b"CDEF"}
            fetcher.configure_rate_limit_policy(
                max_retries_per_object=0, max_workers=1, reset_counters=True
            )
        finally:
            batch.cancel()
            await asyncio.gather(batch, return_exceptions=True)

    asyncio.run(_run())


@pytest.mark.parametrize("blocked_stage", ["active", "queued", "cooldown"])
def test_retry_stage_configuration_requires_all_spans_to_drain(
    monkeypatch: pytest.MonkeyPatch, blocked_stage: str
) -> None:
    """Stage changes cannot reset counters beneath active or waiting callers."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    async def _run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        real_sleep = asyncio.sleep

        class HeldResponse(_RangedFakeResponse):
            async def __aenter__(self) -> Self:
                if blocked_stage != "cooldown":
                    started.set()
                    await release.wait()
                return self

        class HeldSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> HeldResponse:
                del url, kwargs
                response = HeldResponse(b"ABCDEFGHIJ", 2, 5)
                if blocked_stage == "cooldown":
                    response.status = 429
                return response

        async def sleep(delay: float) -> None:
            if delay:
                started.set()
                await release.wait()
            await real_sleep(0)

        monkeypatch.setattr(ranges_module.asyncio, "sleep", sleep)
        fetcher = RangeFetcher()
        fetcher._session = HeldSession(b"ABCDEFGHIJ")
        fetcher.configure_rate_limit_policy(max_retries_per_object=10, max_workers=1)
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        tasks = [asyncio.create_task(fetcher.fetch((span,)))]
        await started.wait()
        if blocked_stage == "queued":
            tasks.append(asyncio.create_task(fetcher.fetch((span,))))
            await real_sleep(0)
        try:
            with pytest.raises(RuntimeError, match="drain"):
                fetcher.configure_rate_limit_policy(
                    max_retries_per_object=0, max_workers=1, reset_counters=True
                )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if fetcher._cool_task is not None:
                fetcher._cool_task.cancel()
                await asyncio.gather(fetcher._cool_task, return_exceptions=True)
        fetcher.configure_rate_limit_policy(
            max_retries_per_object=0, max_workers=1, reset_counters=True
        )

    asyncio.run(_run())


def test_concurrent_spans_share_actual_object_429_counts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Eleven simultaneous 429s count once each, without per-span retries."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)

    async def _run() -> None:
        wave = asyncio.Barrier(11)

        class RateResponse(_RangedFakeResponse):
            async def __aenter__(self) -> Self:
                await wave.wait()
                return self

        class RateSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> RateResponse:
                del url, kwargs
                self.calls += 1
                response = RateResponse(b"ABCDEFGHIJ", 2, 5)
                response.status = 429
                return response

        fetcher = RangeFetcher()
        session = RateSession(b"ABCDEFGHIJ")
        fetcher._session = session
        fetcher.configure_rate_limit_policy(max_retries_per_object=10)
        span = RangeSpan("https://example.test/measurement?token=hidden", 2, 5, "a")
        results = await asyncio.gather(
            *(fetcher.fetch((span,)) for _ in range(11)), return_exceptions=True
        )
        assert all(isinstance(result, RemoteAccessError) for result in results)
        assert {result.reason for result in results} == {"rate_limit_exhausted"}
        assert session.calls == 11

    asyncio.run(_run())
    messages = [record.getMessage() for record in caplog.records]
    assert messages.count("range request was rate-limited") == 11
    assert all("hidden" not in message for message in messages)


def test_object_429_count_survives_successes_and_separates_metadata_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful span boundaries never reset an exact measurement URL's count."""
    import io

    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)
    measurement = "https://example.test/measurement"
    annotation = "https://example.test/annotation"
    hits: dict[str, int] = {}

    class RateSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            response = super().get(url, **kwargs)
            hits[url] = hits.get(url, 0) + 1
            if url == measurement and hits[url] in {1, 2, 4}:
                response.status = 429
            if url == annotation and hits[url] <= 2:
                response.status = 429
            return response

    async def _run() -> None:
        fetcher = RangeFetcher()
        fetcher._session = RateSession(b"ABCDEFGHIJ")
        fetcher.configure_rate_limit_policy(max_retries_per_object=2)
        assert await fetcher.fetch((RangeSpan(measurement, 2, 5, "hdr"),)) == {
            "hdr": b"CDEF"
        }
        assert await fetcher.fetch((RangeSpan(annotation, 2, 5, "xml"),)) == {
            "xml": b"CDEF"
        }
        with pytest.raises(RemoteAccessError) as raised:
            await fetcher.fetch_into(RangeSpan(measurement, 6, 9, "rows"), io.BytesIO())
        assert raised.value.reason == "rate_limit_exhausted"
        assert hits == {measurement: 4, annotation: 3}

    asyncio.run(_run())


def test_final_retry_stage_keeps_shared_retry_after_and_fails_first_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset counts only, retain the session cooldown, then allow no 429 retry."""
    import io

    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module
    from faninsar.remote.protocols import _CallLedger

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        real_sleep = asyncio.sleep
        cooling = asyncio.Event()
        release = asyncio.Event()
        calls: list[float] = []

        async def sleep(delay: float) -> None:
            if delay:
                cooling.set()
                await release.wait()
                clock[0] += delay
            await real_sleep(0)

        class RateSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
                response = super().get(url, **kwargs)
                calls.append(clock[0])
                response.status = 429
                response.headers["Retry-After"] = "4"
                return response

        monkeypatch.setattr(loop, "time", lambda: clock[0])
        monkeypatch.setattr(ranges_module.asyncio, "sleep", sleep)
        ledger = _CallLedger(RemoteResourceBudget(max_workers=15, max_retries=100))
        fetcher = RangeFetcher(ledger=ledger)
        session = RateSession(b"ABCDEFGHIJ")
        fetcher._session = session
        fetcher.configure_rate_limit_policy(max_retries_per_object=10)
        span = RangeSpan("https://example.test/measurement", 2, 5, "a")
        main = asyncio.create_task(fetcher.fetch((span,)))
        await cooling.wait()
        main.cancel()
        await asyncio.gather(main, return_exceptions=True)
        fetcher.configure_rate_limit_policy(
            max_retries_per_object=0, max_workers=1, reset_counters=True
        )
        final = asyncio.create_task(fetcher.fetch_into(span, io.BytesIO()))
        try:
            for _ in range(5):
                await real_sleep(0)
            assert session.calls == 1
            assert fetcher.window == 1
            release.set()
            with pytest.raises(RemoteAccessError) as raised:
                await final
            assert raised.value.reason == "rate_limit_exhausted"
            assert session.calls == 2
            assert calls[1] - calls[0] >= 4
            assert fetcher._session is session
            assert ledger.requests == 2
            assert ledger.retries == 1
        finally:
            release.set()
            final.cancel()
            await asyncio.gather(final, return_exceptions=True)
            if fetcher._cool_task is not None:
                fetcher._cool_task.cancel()
                await asyncio.gather(fetcher._cool_task, return_exceptions=True)

    asyncio.run(_run())


@pytest.mark.parametrize("cancel_batch", [False, True])
def test_failed_fetch_drains_only_its_own_span_tasks(
    monkeypatch: pytest.MonkeyPatch,
    cancel_batch: bool,
) -> None:
    """Batch error/cancellation drains siblings without cancelling another call."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)

    async def _run() -> None:
        slow_started = asyncio.Event()
        independent_started = asyncio.Event()
        release = asyncio.Event()
        slow_closed = asyncio.Event()

        class HeldResponse(_RangedFakeResponse):
            def __init__(self, url: str) -> None:
                super().__init__(b"ABCDEFGHIJ", 2, 5)
                self.url = url

            async def __aenter__(self) -> Self:
                if self.url.endswith("slow"):
                    slow_started.set()
                    try:
                        await release.wait()
                    finally:
                        slow_closed.set()
                elif self.url.endswith("independent"):
                    independent_started.set()
                    await release.wait()
                else:
                    await slow_started.wait()
                    self.status = 429
                return self

        class HeldSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> HeldResponse:
                del kwargs
                return HeldResponse(url)

        fetcher = RangeFetcher()
        fetcher._session = HeldSession(b"ABCDEFGHIJ")
        fetcher.configure_rate_limit_policy(max_retries_per_object=0)
        slow = RangeSpan("https://example.test/slow", 2, 5, "slow")
        bad = RangeSpan("https://example.test/bad", 2, 5, "bad")
        other = RangeSpan("https://example.test/independent", 2, 5, "other")
        independent = asyncio.create_task(fetcher.fetch((other,)))
        await independent_started.wait()
        batch = asyncio.create_task(
            fetcher.fetch((slow,) if cancel_batch else (slow, bad))
        )
        await slow_started.wait()
        try:
            if cancel_batch:
                batch.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await batch
            else:
                with pytest.raises(RemoteAccessError) as raised:
                    await batch
                assert raised.value.reason == "rate_limit_exhausted"
            assert slow_closed.is_set(), "fetch left one of its own spans alive"
            assert not independent.done()
            release.set()
            assert await independent == {"other": b"CDEF"}
            fetcher.configure_rate_limit_policy(
                max_retries_per_object=0, max_workers=1, reset_counters=True
            )
        finally:
            release.set()
            for task in (batch, independent):
                task.cancel()
            await asyncio.gather(batch, independent, return_exceptions=True)

    asyncio.run(_run())


def test_object_policy_allows_ten_retries_with_bounded_client_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eleventh request may succeed; client backoff never poisons the session."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        real_sleep = asyncio.sleep
        sleeps: list[float] = []

        async def sleep(delay: float) -> None:
            if delay:
                sleeps.append(delay)
                clock[0] += delay
            await real_sleep(0)

        class RateSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
                response = super().get(url, **kwargs)
                if self.calls <= 10:
                    response.status = 429
                return response

        monkeypatch.setattr(loop, "time", lambda: clock[0])
        monkeypatch.setattr(ranges_module.asyncio, "sleep", sleep)
        session = RateSession(b"ABCDEFGHIJ")
        fetcher = RangeFetcher()
        fetcher._session = session
        fetcher.configure_rate_limit_policy(max_retries_per_object=10)
        assert await fetcher.fetch(
            (RangeSpan("https://example.test/o", 2, 5, "a"),)
        ) == {"a": b"CDEF"}
        assert session.calls == 11
        assert sleeps == [1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 60.0, 60.0, 60.0, 60.0]

    asyncio.run(_run())


def test_none_policy_retains_five_attempts_per_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default and explicitly disabled policy retain the existing retry contract."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)

    class RateSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            response = super().get(url, **kwargs)
            response.status = 429
            return response

    async def _run() -> None:
        fetcher = RangeFetcher()
        session = RateSession(b"ABCDEFGHIJ")
        fetcher._session = session
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        for expected in (5, 10):
            with pytest.raises(RemoteAccessError) as raised:
                await fetcher.fetch((span,))
            assert raised.value.reason == "range_unavailable"
            assert session.calls == expected
            fetcher.configure_rate_limit_policy(max_retries_per_object=None)

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [({"max_requests": 2}, "max_requests"), ({"max_retries": 1}, "max_retries")],
)
def test_object_policy_preserves_operation_request_and_retry_budgets(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, int],
    reason: str,
) -> None:
    """An object retry allowance never overrides stricter operation accounting."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.errors import RemoteLimitError
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)

    class RateSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            response = super().get(url, **kwargs)
            response.status = 429
            return response

    async def _run() -> None:
        fetcher = RangeFetcher(budget=RemoteResourceBudget(**overrides))
        session = RateSession(b"ABCDEFGHIJ")
        fetcher._session = session
        fetcher.configure_rate_limit_policy(max_retries_per_object=10)
        with pytest.raises(RemoteLimitError) as raised:
            await fetcher.fetch((RangeSpan("https://example.test/o", 2, 5, "a"),))
        assert raised.value.reason == reason
        assert session.calls == 2

    asyncio.run(_run())


def test_object_policy_cannot_raise_global_http_worker_budget() -> None:
    """All object URLs share the operation's fifteen-request admission ceiling."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    async def _run() -> None:
        release = asyncio.Event()
        started: list[str] = []

        class HeldResponse(_RangedFakeResponse):
            async def __aenter__(self) -> Self:
                await release.wait()
                return self

        class HeldSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> HeldResponse:
                del kwargs
                started.append(url)
                return HeldResponse(b"ABCDEFGHIJ", 2, 5)

        fetcher = RangeFetcher(budget=RemoteResourceBudget(max_workers=15))
        fetcher._session = HeldSession(b"ABCDEFGHIJ")
        fetcher.configure_rate_limit_policy(max_retries_per_object=10, max_workers=100)
        spans = [
            RangeSpan(f"https://example.test/{i}", 2, 5, str(i)) for i in range(20)
        ]
        task = asyncio.create_task(fetcher.fetch(spans))
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert len(started) == 15
            assert fetcher.window == 15
        finally:
            release.set()
            result = await task
        assert len(result) == 20
        fetcher.configure_rate_limit_policy(max_retries_per_object=0, max_workers=1)
        fetcher.configure_rate_limit_policy(max_retries_per_object=0, max_workers=15)
        assert fetcher.window == 1

    asyncio.run(_run())


@pytest.mark.parametrize(
    "retry_after",
    ["1", "Wed, 11 Nov 2026 12:00:02 GMT", "Wed Nov 11 12:00:02 2026"],
)
def test_429_retry_after_delays_the_next_request(
    monkeypatch: pytest.MonkeyPatch, retry_after: str
) -> None:
    """Seconds and HTTP dates both postpone retries by the server's delay."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    class RateSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            response = super().get(url, **kwargs)
            if self.calls == 1:
                response.status = 429
                response.headers["Retry-After"] = retry_after
            return response

    async def _run() -> dict[str, bytes]:
        loop = asyncio.get_running_loop()
        epoch = loop.time()
        clock = [epoch]
        sleeps: list[float] = []
        real_sleep = asyncio.sleep

        async def sleep(delay: float) -> None:
            if delay:
                sleeps.append(delay)
                clock[0] += delay
            await real_sleep(0)

        monkeypatch.setattr(loop, "time", lambda: clock[0])
        monkeypatch.setattr(ranges_module.asyncio, "sleep", sleep)
        monkeypatch.setattr("time.time", lambda: 1794398400.0)
        monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)
        fetcher = RangeFetcher()
        fetcher._session = RateSession(b"ABCDEFGHIJ")
        result = await fetcher.fetch((RangeSpan("https://example.test/o", 2, 5, "a"),))
        assert sum(sleeps) >= (1 if retry_after == "1" else 2)
        return result

    assert asyncio.run(_run()) == {"a": b"CDEF"}


def test_429_cooldown_pauses_new_fetch_and_fetch_into_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One rate-limited request pauses new work on the reusable fetcher."""
    import io

    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    class RateSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            response = super().get(url, **kwargs)
            if self.calls == 1:
                response.status = 429
            return response

    async def _run() -> None:
        cooling = asyncio.Event()
        release = asyncio.Event()
        real_sleep = asyncio.sleep
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        monkeypatch.setattr(loop, "time", lambda: clock[0])

        async def sleep(delay: float) -> None:
            if delay:
                cooling.set()
                await release.wait()
                clock[0] += delay
            else:
                await real_sleep(0)

        monkeypatch.setattr(ranges_module.asyncio, "sleep", sleep)
        session = RateSession(b"ABCDEFGHIJ")
        fetcher = RangeFetcher()
        fetcher._session = session
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        first = asyncio.create_task(fetcher.fetch((span,)))
        await cooling.wait()
        sink = io.BytesIO()
        second = asyncio.create_task(fetcher.fetch((span,)))
        third = asyncio.create_task(fetcher.fetch_into(span, sink))
        try:
            for _ in range(5):
                await real_sleep(0)
            assert session.calls == 1
        finally:
            release.set()
            results = await asyncio.gather(first, second, third)
        assert results == [{"a": b"CDEF"}, {"a": b"CDEF"}, 4]
        assert sink.getvalue() == b"CDEF"

    asyncio.run(_run())


def test_reduced_window_drains_inflight_before_admitting_queued_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4-to-3 cut admits nothing while three old requests are still active."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_AIMD_SEED", 4)
    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)

    async def _run() -> None:
        started: list[int] = []
        releases = [asyncio.Event() for _ in range(4)]
        wave = asyncio.Event()
        extra = asyncio.Event()

        class ControlledResponse(_RangedFakeResponse):
            async def __aenter__(self) -> Self:
                index = len(started)
                started.append(index)
                if len(started) == 4:
                    wave.set()
                if index < 4:
                    await releases[index].wait()
                    if index == 0:
                        self.status = 429
                else:
                    extra.set()
                return self

        class ControlledSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> ControlledResponse:
                del url, kwargs
                return ControlledResponse(b"ABCDEFGHIJ", 2, 5)

        fetcher = RangeFetcher()
        fetcher._session = ControlledSession(b"ABCDEFGHIJ")
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        task = asyncio.create_task(fetcher.fetch([span] * 6))
        try:
            await wave.wait()
            releases[0].set()
            for _ in range(20):
                await asyncio.sleep(0)
            assert fetcher.window == 3
            assert started == [0, 1, 2, 3]
            releases[1].set()
            await asyncio.wait_for(extra.wait(), timeout=2)
        finally:
            for release in releases:
                release.set()
            await task

    asyncio.run(_run())


def test_later_429_extends_the_shared_retry_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-flight 429 with a longer delay cannot reuse an earlier wakeup."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        monkeypatch.setattr(loop, "time", lambda: clock[0])
        monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)
        real_sleep = asyncio.sleep
        cooling = asyncio.Event()
        release_sleep = asyncio.Event()
        second_429 = asyncio.Event()
        both_started = asyncio.Event()
        drain = [False]

        async def sleep(delay: float) -> None:
            if delay:
                cooling.set()
                if not drain[0]:
                    await release_sleep.wait()
                    release_sleep.clear()
                clock[0] += delay
            await real_sleep(0)

        class RateResponse(_RangedFakeResponse):
            async def __aenter__(self) -> Self:
                if self.headers.get("Retry-After") == "1":
                    await both_started.wait()
                if self.headers.get("Retry-After") == "4":
                    both_started.set()
                    await second_429.wait()
                return self

        class RateSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> RateResponse:
                del url, kwargs
                self.calls += 1
                response = RateResponse(b"ABCDEFGHIJ", 2, 5)
                if self.calls <= 2:
                    response.status = 429
                    response.headers["Retry-After"] = "1" if self.calls == 1 else "4"
                return response

        monkeypatch.setattr(ranges_module.asyncio, "sleep", sleep)
        session = RateSession(b"ABCDEFGHIJ")
        fetcher = RangeFetcher()
        fetcher._session = session
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        task = asyncio.create_task(fetcher.fetch([span] * 2))
        await cooling.wait()
        second_429.set()
        try:
            for _ in range(10):
                await real_sleep(0)
            release_sleep.set()
            for _ in range(10):
                await real_sleep(0)
            assert session.calls == 2
        finally:
            drain[0] = True
            release_sleep.set()
            await task

    asyncio.run(_run())


def test_429_cooldown_starts_before_response_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting for response cleanup cannot let another request slip through."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0.05)

    async def _run() -> None:
        closing = asyncio.Event()
        release = asyncio.Event()

        class RateResponse(_RangedFakeResponse):
            async def __aexit__(self, *exc_info: object) -> None:
                if self.status == 429:
                    closing.set()
                    await release.wait()

        class RateSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> RateResponse:
                del url, kwargs
                self.calls += 1
                response = RateResponse(b"ABCDEFGHIJ", 2, 5)
                if self.calls == 1:
                    response.status = 429
                return response

        session = RateSession(b"ABCDEFGHIJ")
        fetcher = RangeFetcher()
        fetcher._session = session
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        first = asyncio.create_task(fetcher.fetch((span,)))
        await closing.wait()
        second = asyncio.create_task(fetcher.fetch((span,)))
        try:
            for _ in range(5):
                await asyncio.sleep(0)
            assert session.calls == 1
        finally:
            release.set()
            await asyncio.gather(first, second)

    asyncio.run(_run())


def test_excessive_retry_after_fails_without_retrying_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server delay beyond the supported wait fails this session closed."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    class RateSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            response = super().get(url, **kwargs)
            response.status = 429
            response.headers["Retry-After"] = "301"
            return response

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        real_sleep = asyncio.sleep

        async def sleep(delay: float) -> None:
            clock[0] += delay
            await real_sleep(0)

        monkeypatch.setattr(loop, "time", lambda: clock[0])
        monkeypatch.setattr(ranges_module.asyncio, "sleep", sleep)
        session = RateSession(b"ABCDEFGHIJ")
        fetcher = RangeFetcher()
        fetcher._session = session
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        for _ in range(2):
            with pytest.raises(RemoteAccessError, match="rate-limited"):
                await fetcher.fetch((span,))
        assert session.calls == 1

    asyncio.run(_run())


def test_cancelled_request_releases_admission_for_other_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling an active stream releases its slot for already-queued work."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_AIMD_SEED", 1)

    async def _run() -> None:
        started = asyncio.Event()
        blocked = asyncio.Event()

        class HeldResponse(_RangedFakeResponse):
            async def __aenter__(self) -> Self:
                if not started.is_set():
                    started.set()
                    await blocked.wait()
                return self

        class HeldSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> HeldResponse:
                del url, kwargs
                return HeldResponse(b"ABCDEFGHIJ", 2, 5)

        fetcher = RangeFetcher()
        fetcher._session = HeldSession(b"ABCDEFGHIJ")
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        first = asyncio.create_task(fetcher.fetch((span,)))
        await started.wait()
        second = asyncio.create_task(fetcher.fetch((span,)))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await asyncio.wait_for(second, timeout=2) == {"a": b"CDEF"}

    asyncio.run(_run())


def test_cancelled_queued_request_does_not_consume_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling a waiting caller leaves the active stream and next slot intact."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_AIMD_SEED", 1)

    async def _run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class HeldResponse(_RangedFakeResponse):
            async def __aenter__(self) -> Self:
                started.set()
                await release.wait()
                return self

        class HeldSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> HeldResponse:
                del url, kwargs
                return HeldResponse(b"ABCDEFGHIJ", 2, 5)

        fetcher = RangeFetcher()
        fetcher._session = HeldSession(b"ABCDEFGHIJ")
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        first = asyncio.create_task(fetcher.fetch((span,)))
        await started.wait()
        queued = asyncio.create_task(fetcher.fetch((span,)))
        for _ in range(3):
            await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        assert await first == {"a": b"CDEF"}
        assert await asyncio.wait_for(fetcher.fetch((span,)), timeout=2) == {
            "a": b"CDEF"
        }

    asyncio.run(_run())


@pytest.mark.parametrize("wait_kind", ["retry_cooldown", "new_cooldown", "admission"])
def test_range_waits_obey_the_remaining_operation_deadline(
    monkeypatch: pytest.MonkeyPatch, wait_kind: str
) -> None:
    """A caller times out without cancelling the shared cooldown or active work."""
    from faninsar.remote import RemoteResourceBudget
    from faninsar.remote.errors import RemoteLimitError
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)

    async def _run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class ControlledResponse(_RangedFakeResponse):
            async def __aenter__(self) -> Self:
                if wait_kind == "admission":
                    started.set()
                    await release.wait()
                return self

            async def __aexit__(self, *exc_info: object) -> None:
                if self.status == 429:
                    started.set()

        class ControlledSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> ControlledResponse:
                del url, kwargs
                self.calls += 1
                response = ControlledResponse(b"ABCDEFGHIJ", 2, 5)
                if wait_kind != "admission":
                    response.status = 429
                    response.headers["Retry-After"] = "1"
                return response

        session = ControlledSession(b"ABCDEFGHIJ")
        fetcher = RangeFetcher(
            budget=RemoteResourceBudget(max_workers=1, max_elapsed_seconds=0.03)
        )
        fetcher._session = session
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        first = asyncio.create_task(fetcher.fetch((span,)))
        await started.wait()
        selected = (
            first
            if wait_kind == "retry_cooldown"
            else asyncio.create_task(fetcher.fetch((span,)))
        )
        try:
            done, _ = await asyncio.wait({selected}, timeout=0.2)
            assert selected in done, "operation deadline did not bound the wait"
            with pytest.raises(RemoteLimitError) as raised:
                await selected
            assert raised.value.reason == "max_elapsed_seconds"
            assert session.calls == 1
            if wait_kind == "admission":
                assert not first.done()
            else:
                assert fetcher._cool_task is not None
                assert not fetcher._cool_task.done()
                assert fetcher._cool is not None and not fetcher._cool.is_set()
        finally:
            tasks = {first, selected}
            if fetcher._cool_task is not None:
                tasks.add(fetcher._cool_task)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(_run())


def test_terminal_retry_after_wakes_existing_cooldown_waiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal 429 wakes old retry and new callers without another request."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_RATE_LIMIT_BACKOFF_S", 0)

    async def _run() -> None:
        both_started = asyncio.Event()
        cooling = asyncio.Event()
        terminal = asyncio.Event()

        class ControlledResponse(_RangedFakeResponse):
            async def __aenter__(self) -> Self:
                if self.headers["Retry-After"] == "1":
                    await both_started.wait()
                else:
                    both_started.set()
                    await terminal.wait()
                return self

            async def __aexit__(self, *exc_info: object) -> None:
                if self.headers["Retry-After"] == "1":
                    cooling.set()

        class ControlledSession(_RangedSession):
            def get(self, url: str, **kwargs: object) -> ControlledResponse:
                del url, kwargs
                self.calls += 1
                response = ControlledResponse(b"ABCDEFGHIJ", 2, 5)
                response.status = 429
                response.headers["Retry-After"] = "1" if self.calls == 1 else "301"
                return response

        session = ControlledSession(b"ABCDEFGHIJ")
        fetcher = RangeFetcher()
        fetcher._session = session
        span = RangeSpan("https://example.test/o", 2, 5, "a")
        first = asyncio.create_task(fetcher.fetch((span,)))
        cap = asyncio.create_task(fetcher.fetch((span,)))
        await cooling.wait()
        queued = asyncio.create_task(fetcher.fetch((span,)))
        await asyncio.sleep(0)
        terminal.set()
        try:
            with pytest.raises(RemoteAccessError, match="rate-limited"):
                await cap
            done, _ = await asyncio.wait({first, queued}, timeout=0.2)
            assert done == {first, queued}, (
                "terminal state left old cooldown waiters asleep"
            )
            for task in (first, queued):
                with pytest.raises(RemoteAccessError, match="rate-limited"):
                    await task
            assert session.calls == 2
        finally:
            tasks = {first, cap, queued}
            if fetcher._cool_task is not None:
                tasks.add(fetcher._cool_task)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(_run())


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


def test_stream_span_resumes_suffix_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-stream TimeoutError is the same class of drop as a payload cut."""
    from faninsar.remote.fetch import RangeSpan
    from faninsar.remote.fetch import ranges as ranges_module

    monkeypatch.setattr(ranges_module, "_CONNECT_BACKOFF_S", 0)
    payload = b"ABCDEFGHIJ"
    session = _RangedSession(
        payload, drop_at_call=1, fail_after=2, drop_exc=TimeoutError()
    )
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


def test_range_fetcher_session_has_no_total_timeout() -> None:
    """A live range GET is bounded by idle sockets and suffix retries, not 300s."""
    from faninsar.remote.fetch import RangeFetcher
    from faninsar.remote.fetch import ranges as ranges_module

    async def _run() -> tuple[float | None, float | None, float | None]:
        async with RangeFetcher() as fetcher:
            timeout = fetcher._session.timeout
            return timeout.total, timeout.sock_connect, timeout.sock_read

    total, sock_connect, sock_read = asyncio.run(_run())
    assert total is None
    assert sock_connect == ranges_module._SOCK_CONNECT_S
    assert sock_read == ranges_module._SOCK_READ_S


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


def test_fetch_into_flushes_small_network_chunks_to_part(tmp_path: Path) -> None:
    """A 4 KiB network prefix is visible on disk before the span ends."""
    from aiohttp import web

    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    payload = b"A" * 8192
    first = 4096
    part = tmp_path / "span.part"
    proceed = asyncio.Event()
    first_sent = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=206,
            headers={
                "Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}",
                "Content-Encoding": "identity",
                "Content-Length": str(len(payload)),
            },
        )
        await response.prepare(request)
        await response.write(payload[:first])
        first_sent.set()
        await proceed.wait()
        await response.write(payload[first:])
        await response.write_eof()
        return response

    async def _run() -> int:
        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            url = str(server.make_url("/obj"))
            span = RangeSpan(url=url, start=0, end=len(payload) - 1, label="a")
            async with RangeFetcher() as fetcher:
                task = asyncio.create_task(fetcher.fetch_into(span, part))
                await asyncio.wait_for(first_sent.wait(), timeout=5)
                size = 0
                loop = asyncio.get_running_loop()
                deadline = loop.time() + 2
                while loop.time() < deadline:
                    if part.exists():
                        size = part.stat().st_size
                        if size >= first:
                            break
                    await asyncio.sleep(0.02)
                proceed.set()
                await task
                return size
        finally:
            proceed.set()
            await server.close()

    size_mid = asyncio.run(_run())
    assert size_mid >= first
    assert part.read_bytes() == payload


# ---------------------------------------------------------------------------
# Resume revision pinning (PROPOSAL-0071): ETag recorded at byte 0, If-Range
# on resume, a changed object surfaces as ``revision_changed``.
# ---------------------------------------------------------------------------


async def _serve_versioned(
    versions: dict[str, bytes],
    current: list[str],
    seen: list[dict[str, str]],
) -> TestServer:
    """Serve ``versions[current[0]]`` with RFC 9110 ``If-Range`` semantics."""

    async def handler(request: web.Request) -> web.StreamResponse:
        seen.append(dict(request.headers))
        tag = current[0]
        body = versions[tag]
        etag = f'"{tag}"'
        if_range = request.headers.get("If-Range")
        if if_range is not None and if_range != etag:
            return web.Response(status=200, body=body, headers={"ETag": etag})
        first, last = request.headers["Range"].removeprefix("bytes=").split("-")
        start, end = int(first), int(last)
        return web.Response(
            status=206,
            body=body[start : end + 1],
            headers={
                "Content-Range": f"bytes {start}-{end}/{len(body)}",
                "ETag": etag,
            },
        )

    app = web.Application()
    app.router.add_get("/obj", handler)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    return server


def test_fetch_into_reports_the_etag_of_a_span_started_at_byte_zero(
    tmp_path: Path,
) -> None:
    """A fresh span reports its strong ETag and sends no If-Range."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    seen: list[dict[str, str]] = []
    learned: list[str] = []
    part = tmp_path / "span.part"

    async def _run() -> None:
        server = await _serve_versioned({"v1": b"ABCDEFGHIJ"}, ["v1"], seen)
        try:
            span = RangeSpan(str(server.make_url("/obj")), 2, 9, "a")
            async with RangeFetcher() as fetcher:
                await fetcher.fetch_into(span, part, on_etag=learned.append)
        finally:
            await server.close()

    asyncio.run(_run())
    assert learned == ['"v1"']
    assert "If-Range" not in seen[0]
    assert part.read_bytes() == b"CDEFGHIJ"


def test_fetch_into_sends_if_range_when_resuming_a_pinned_prefix(
    tmp_path: Path,
) -> None:
    """A recorded ETag rides the resume as If-Range; 206 appends the suffix."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    seen: list[dict[str, str]] = []
    part = tmp_path / "span.part"
    part.write_bytes(b"CD")

    async def _run() -> None:
        server = await _serve_versioned({"v1": b"ABCDEFGHIJ"}, ["v1"], seen)
        try:
            span = RangeSpan(str(server.make_url("/obj")), 2, 9, "a")
            async with RangeFetcher() as fetcher:
                await fetcher.fetch_into(span, part, etag='"v1"')
        finally:
            await server.close()

    asyncio.run(_run())
    assert [(h["Range"], h.get("If-Range")) for h in seen] == [("bytes=4-9", '"v1"')]
    assert part.read_bytes() == b"CDEFGHIJ"


def test_fetch_into_rejects_a_changed_revision_without_appending(
    tmp_path: Path,
) -> None:
    """A 200 answer to If-Range raises revision_changed; the prefix is untouched."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    seen: list[dict[str, str]] = []
    part = tmp_path / "span.part"
    part.write_bytes(b"CD")

    async def _run() -> None:
        server = await _serve_versioned({"v2": b"abcdefghij"}, ["v2"], seen)
        try:
            span = RangeSpan(str(server.make_url("/obj")), 2, 9, "a")
            async with RangeFetcher() as fetcher:
                await fetcher.fetch_into(span, part, etag='"v1"')
        finally:
            await server.close()

    with pytest.raises(RemoteIntegrityError) as raised:
        asyncio.run(_run())
    assert raised.value.reason == "revision_changed"
    assert len(seen) == 1
    assert part.read_bytes() == b"CD"


def test_fetch_into_without_a_recorded_etag_resumes_unvalidated(
    tmp_path: Path,
) -> None:
    """No recorded ETag keeps today's contract: no If-Range on resume."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    seen: list[dict[str, str]] = []
    learned: list[str] = []
    part = tmp_path / "span.part"
    part.write_bytes(b"CD")

    async def _run() -> None:
        server = await _serve_versioned({"v1": b"ABCDEFGHIJ"}, ["v1"], seen)
        try:
            span = RangeSpan(str(server.make_url("/obj")), 2, 9, "a")
            async with RangeFetcher() as fetcher:
                await fetcher.fetch_into(span, part, on_etag=learned.append)
        finally:
            await server.close()

    asyncio.run(_run())
    assert [(h["Range"], h.get("If-Range")) for h in seen] == [("bytes=4-9", None)]
    # A resumed (non-zero) start never records a validator for the prefix.
    assert learned == []
    assert part.read_bytes() == b"CDEFGHIJ"


def test_fetch_into_rejects_changed_etag_even_on_exact_206(tmp_path: Path) -> None:
    """A server ignoring If-Range cannot append a new revision's suffix."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    seen: list[dict[str, str]] = []

    class ChangedSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            seen.append(dict(kwargs["headers"]))
            response = super().get(url, **kwargs)
            response.headers["ETag"] = '"v2"'
            return response

    part = tmp_path / "span.part"
    part.write_bytes(b"AB")
    fetcher = RangeFetcher()
    fetcher._session = ChangedSession(b"abcdefghij")
    with pytest.raises(RemoteIntegrityError, match="revision"):
        asyncio.run(
            fetcher.fetch_into(
                RangeSpan("https://example.test/o", 0, 9, "x"), part, etag='"v1"'
            )
        )
    assert seen[0]["If-Range"] == '"v1"'
    assert part.read_bytes() == b"AB"


def test_fetch_into_does_not_send_saved_weak_validator(tmp_path: Path) -> None:
    """A weak validator loaded from old progress is never sent as If-Range."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    seen: list[dict[str, str]] = []

    class WeakSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            seen.append(dict(kwargs["headers"]))
            return super().get(url, **kwargs)

    part = tmp_path / "span.part"
    part.write_bytes(b"AB")
    fetcher = RangeFetcher()
    fetcher._session = WeakSession(b"ABCDEFGHIJ")
    asyncio.run(
        fetcher.fetch_into(
            RangeSpan("https://example.test/o", 0, 9, "x"), part, etag='W/"v1"'
        )
    )
    assert "If-Range" not in seen[0]
    assert part.read_bytes() == b"ABCDEFGHIJ"


def test_fetch_into_ignores_weak_etags() -> None:
    """Weak validators cannot pin byte ranges (RFC 9110 If-Range)."""
    from faninsar.remote.fetch import RangeFetcher, RangeSpan

    class _WeakSession(_RangedSession):
        def get(self, url: str, **kwargs: object) -> _RangedFakeResponse:
            response = super().get(url, **kwargs)
            response.headers["ETag"] = 'W/"v1"'
            return response

    import io

    fetcher = RangeFetcher()
    fetcher._session = _WeakSession(b"ABCDEFGHIJ")
    learned: list[str] = []
    span = RangeSpan("https://zipper.example.invalid/obj", 0, 9, "a")

    async def _run() -> int:
        return await fetcher.fetch_into(span, io.BytesIO(), on_etag=learned.append)

    assert asyncio.run(_run()) == 10
    assert learned == []
