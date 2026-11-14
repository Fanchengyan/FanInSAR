"""ROI date workers share a loop-owned HTTP session (P0061/P0070)."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from faninsar.missions.s1 import acquisition, preparation
from faninsar.remote import RemoteResourceBudget
from faninsar.remote.fetch import RangeFetcher, RangeSpan
from tests.missions.s1.test_acquire_stage_parallel import (
    _context,
    _make_stack,
    _Recorder,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence


@pytest.mark.parametrize("dates", [2, 4])
def test_roi_dates_share_real_http_fetcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dates: int
) -> None:
    """Distinct date payload URLs overlap, with one provider session and budget."""
    active = peak = 0
    urls = set()
    opened = []
    closed = []
    validated = set()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal active, peak

            with lock:
                active += 1
                peak = max(peak, active)
                urls.add(self.path)
            try:
                time.sleep(0.08)
                self.send_response(206)
                self.send_header("Content-Length", "4")
                self.send_header("Content-Range", "bytes 0-3/4")
                self.end_headers()
                self.wfile.write(b"data")
            finally:
                with lock:
                    active -= 1

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    budget = RemoteResourceBudget(
        max_workers=4, max_active_files=4, max_ranges_per_file=4, max_metadata_files=2
    )
    from faninsar.remote.providers import cdse

    async def validate(urls: Sequence[str]) -> None:
        validated.update(urls)

    @asynccontextmanager
    async def provider(
        _spans: Sequence[RangeSpan], **policy: Any
    ) -> AsyncIterator[RangeFetcher]:
        opened.append(policy)
        async with RangeFetcher(**policy) as fetcher:
            yield fetcher
        closed.append(True)

    monkeypatch.setattr(cdse, "validate_cdse_range_urls", validate)
    monkeypatch.setattr(cdse, "cdse_range_fetcher", provider)
    monkeypatch.setattr(acquisition, "_cdse_range_fetcher", provider)
    monkeypatch.setattr(acquisition, "_cdse_range_host", lambda _url: True)
    recorder = _Recorder()
    sinks = []

    def download(day: int) -> Callable[[], None]:
        async def run() -> None:
            url = f"http://127.0.0.1:{server.server_port}/{day}.tiff"
            span = RangeSpan(url, 0, 3, "measurement")
            sink = Path(tmp_path) / f"{day}.part"
            async with acquisition._span_fetcher_cm([span]) as fetcher:
                await fetcher.fetch_into(span, sink)
            sinks.append(sink)

        return lambda: asyncio.run(run())

    stack = _make_stack(
        recorder,
        tmp_path,
        {f"201612{day:02}": download(day) for day in range(1, dates + 1)},
    )
    # Build the immutable search request through the real public constructor;
    # only materialization/qualification is isolated from scientific processing.
    from faninsar.missions.s1.s1search import S1Search
    from faninsar.missions.s1.stack import S1Stack
    from tests.missions.s1.test_from_roi import ROI, _cdse_item

    items = tuple(
        replace(
            _cdse_item(),
            item_id=f"item-{day}",
            acquisition=replace(
                _cdse_item().acquisition,
                acquisition_id=f"item-{day}",
                start=datetime(2016, 12, day, tzinfo=UTC),
                end=datetime(2016, 12, day, tzinfo=UTC),
            ),
        )
        for day in range(1, dates + 1)
    )
    public_stack = S1Stack.from_search(
        S1Search(items=items, catalog="sentinel1-cdse", spatial=ROI, query={}),
        work_dir=tmp_path / "public-stack",
        extra={"orbit_auto_fetch": False, "aux_cal_auto_fetch": False},
    )
    context = _context(granularity="roi")
    context.request = public_stack.request
    context.budget = budget
    try:
        result = preparation.prepare_s1_inputs(stack, context)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.failures == ()
    assert len(opened) == 1
    assert opened[0]["budget"] is budget
    assert closed == [True]
    assert 2 <= peak <= budget.max_workers
    assert len(urls) == dates
    assert len(validated) == dates
    assert all(sink.read_bytes() == b"data" for sink in sinks)


def test_roi_cancel_waits_for_owned_sink_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling a caller cannot leave writes running after its sink is released."""
    from faninsar.remote.providers import cdse

    entered = threading.Event()
    cleaned = threading.Event()
    closed = []

    class Fetcher:
        async def fetch_into(self, *_args: Any, **_kwargs: Any) -> int:
            entered.set()
            try:
                await asyncio.sleep(3600)
            finally:
                await asyncio.sleep(0.02)
                cleaned.set()
            return 4

    @asynccontextmanager
    async def provider(*_args: Any, **_kwargs: Any) -> AsyncIterator[Fetcher]:
        try:
            yield Fetcher()
        finally:
            assert cleaned.is_set()
            closed.append(True)

    async def validate(_urls: Sequence[str]) -> None:
        return None

    monkeypatch.setattr(cdse, "cdse_range_fetcher", provider)
    monkeypatch.setattr(cdse, "validate_cdse_range_urls", validate)
    operation = acquisition._CdseBurstTransfer()
    span = RangeSpan("https://download.dataspace.copernicus.eu/object", 0, 3, "x")

    async def run() -> None:
        token = acquisition._burst_transfer.set(operation)
        try:
            async with acquisition._span_fetcher_cm([span]) as fetcher:
                task = asyncio.create_task(fetcher.fetch_into(span, object()))
                assert await asyncio.to_thread(entered.wait, 2)
                task.cancel()
                await asyncio.sleep(0.005)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert cleaned.is_set()
        finally:
            acquisition._burst_transfer.reset(token)

    try:
        asyncio.run(run())
    finally:
        operation.close()
    assert closed == [True]
