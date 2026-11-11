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


def test_range_budget_separates_active_files_from_http_workers() -> None:
    """File tasks have their own cap; per-file HTTP admission is opt-in."""
    budget = RemoteResourceBudget()
    assert budget.max_active_files == 4
    assert budget.max_ranges_per_file is None
    assert budget.max_metadata_files is None
    assert budget.max_workers == 8
    configured = RemoteResourceBudget(
        max_active_files=6, max_ranges_per_file=4, max_workers=16
    )
    assert configured.max_active_files == 6
    assert configured.max_ranges_per_file == 4
    assert configured.max_workers == 16


@pytest.mark.parametrize(
    "field", ["max_active_files", "max_ranges_per_file", "max_metadata_files"]
)
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, 1.5, "4"])
def test_range_budget_rejects_invalid_file_limits(field: str, value: object) -> None:
    """The new concurrency guards accept positive integers, never booleans."""
    with pytest.raises(ValueError, match=field):
        RemoteResourceBudget(**{field: value})


def test_active_file_budget_cannot_be_unbounded() -> None:
    """Only the per-file HTTP limit has a backwards-compatible None value."""
    with pytest.raises(ValueError, match="max_active_files"):
        RemoteResourceBudget(max_active_files=None)


def test_metadata_file_budget_fits_the_shared_file_limit() -> None:
    """One metadata slot may own the whole F1 policy but cannot exceed it."""
    assert RemoteResourceBudget(max_active_files=1, max_metadata_files=1)
    with pytest.raises(ValueError, match="max_metadata_files"):
        RemoteResourceBudget(max_active_files=1, max_metadata_files=2)


def test_conflicting_range_budget_is_rejected():
    with pytest.raises(ValueError, match="budget must match"):
        RangeFetcher(
            budget=RemoteResourceBudget(max_workers=1),
            ledger=_CallLedger(RemoteResourceBudget()),
        )
