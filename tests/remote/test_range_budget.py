"""Optional range operation accounting under PROPOSAL-0073."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from faninsar.remote import RemoteResourceBudget
from faninsar.remote.errors import RemoteLimitError
from faninsar.remote.fetch import RangeFetcher, RangeSpan
from faninsar.remote.protocols import _CallLedger


@pytest.mark.parametrize(
    "guard",
    [
        None,
        "max_requests",
        "max_response_bytes",
        "max_operation_bytes",
        "max_elapsed_seconds",
        "max_retries",
    ],
)
def test_range_budget_is_enforced_on_actual_requests(guard):
    async def run():
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if guard == "max_retries":
                return web.Response(status=429)
            response = web.StreamResponse(
                status=206, headers={"Content-Range": "bytes 0-3/4"}
            )
            await response.prepare(request)
            try:
                await response.write(b"ab")
                await asyncio.sleep(0.01)
                await response.write(b"cd")
                await response.write_eof()
            except ConnectionResetError:
                pass
            return response

        options = {"max_workers": 2, "max_response_bytes": 4}
        if guard is not None:
            options[guard] = (
                0.001
                if guard == "max_elapsed_seconds"
                else (
                    2 if guard in {"max_response_bytes", "max_operation_bytes"} else 1
                )
            )
        budget = RemoteResourceBudget(**options)
        ledger = _CallLedger(budget)
        app = web.Application()
        app.router.add_get("/obj", handler)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            url = str(server.make_url("/obj"))
            async with RangeFetcher(ledger=ledger) as fetcher:
                assert fetcher.window == 2
                result = await fetcher.fetch(
                    [RangeSpan(url, 0, 3, "a"), RangeSpan(url, 0, 3, "b")]
                )
                assert result == {"a": b"abcd", "b": b"abcd"}
                assert ledger.response_bytes_total == 8
                assert ledger.requests == 2
        finally:
            await server.close()

    if guard is None:
        asyncio.run(run())
    else:
        with pytest.raises(RemoteLimitError) as error:
            asyncio.run(run())
        assert error.value.reason == guard


def test_conflicting_range_budget_is_rejected():
    with pytest.raises(ValueError, match="budget must match"):
        RangeFetcher(
            budget=RemoteResourceBudget(max_workers=1),
            ledger=_CallLedger(RemoteResourceBudget()),
        )
