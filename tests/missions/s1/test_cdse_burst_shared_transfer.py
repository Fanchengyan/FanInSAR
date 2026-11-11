"""Regression tests for the shared CDSE burst transfer operation.

Contract (PROPOSAL-0061 D2, PROPOSAL-0070 operation-scoped fetcher):

- one operation-scoped provider ``RangeFetcher`` per
  :func:`prepare_s1_inputs` call — shared by annotation fetches, TIFF/IFD
  probes, and measurement spans across all items and dates; a direct
  :func:`materialize_cdse_burst_safes` call owns one temporarily;
- every CDSE URL is provider-validated through
  ``faninsar.remote.providers.cdse.validate_cdse_range_urls`` before its
  first credentialed request; non-CDSE spans keep the anonymous per-batch
  path and mixed-origin batches are rejected;
- the AIMD window no longer resets per batch (one session, one token);
- useful-data progress is logged per item;
- shutdown is ordered: a mid-flight cancellation closes every opened
  session, joins the loop thread, and publishes no torn path-SAFE.

All tests are offline: HTTP is answered by an in-process range server.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as concurrent_futures
import hashlib
import json
import struct
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from faninsar.missions.s1.acquisition import materialize_cdse_burst_safes
from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.preparation import prepare_s1_inputs
from faninsar.remote.fetch import RangeFetcher

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import faninsar.missions.s1.acquisition as acquisition_mod
import faninsar.missions.s1.preparation as preparation_mod
import faninsar.remote.providers.cdse as cdse_mod

N_BURSTS = 2
LINES_PER_BURST = 4
SAMPLES_PER_BURST = 16
IFD_OFFSET = 16
DATA_START = 16_384  # keep the IFD probe (>= 16 KiB) inside the object
_STRIP_BYTES = LINES_PER_BURST * SAMPLES_PER_BURST * 4

_CDSE_URL = (
    "https://zipper.dataspace.copernicus.eu/odata/v1/"
    "Products({stem})/Nodes({stem}.SAFE)/Nodes(measurement)/"
    "Nodes({stem}-iw1-vv.tiff)/$value"
)


def _parent_xml() -> str:
    """Annotation for a 2-burst synthetic IW1 parent, with geolocation."""
    burst_els: list[str] = []
    grid_els: list[str] = []
    first = " ".join(["0"] * LINES_PER_BURST)
    last = " ".join([str(SAMPLES_PER_BURST - 1)] * LINES_PER_BURST)
    for k in range(N_BURSTS):
        burst_els.append(
            f"""
      <burst>
        <azimuthTime>2016-12-07T11:18:{53 + k:02d}.426276</azimuthTime>
        <azimuthAnxTime>{607.4725 + 2.5731 * k}</azimuthAnxTime>
        <sensingTime>2016-12-07T11:18:{54 + k:02d}.543845</sensingTime>
        <byteOffset>{k * _STRIP_BYTES}</byteOffset>
        <firstValidSample>{first}</firstValidSample>
        <lastValidSample>{last}</lastValidSample>
      </burst>"""
        )
        for local in (0, LINES_PER_BURST - 1):
            grid_els.append(
                "        <geolocationGridPoint>"
                f"<line>{k * LINES_PER_BURST + local}</line><pixel>0</pixel>"
                f"<latitude>{40.0 + k + 0.01 * local:.4f}</latitude>"
                f"<longitude>{100.0 + 0.01 * local:.4f}</longitude>"
                "</geolocationGridPoint>"
            )
    points = "\n".join(grid_els)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <adsHeader>
    <missionId>S1A</missionId>
    <productType>SLC</productType>
    <polarisation>VV</polarisation>
    <mode>IW</mode>
    <swath>IW1</swath>
    <startTime>2016-12-07T11:18:53.426275</startTime>
    <stopTime>2016-12-07T11:18:55.571896</stopTime>
  </adsHeader>
  <generalAnnotation>
    <productInformation>
      <rangeSamplingRate>64345238.12571428</rangeSamplingRate>
      <radarFrequency>5405000454.33435</radarFrequency>
      <azimuthSteeringRate>1.397440</azimuthSteeringRate>
    </productInformation>
    <orbitList count="2">
      <orbit>
        <time>2016-12-07T11:17:45.950212</time>
        <frame>Earth Fixed</frame>
        <position><x>-572093.4</x><y>5939124.4</y><z>3795662.9</z></position>
        <velocity><x>2075.0</x><y>-3800.0</y><z>6240.0</z></velocity>
      </orbit>
      <orbit>
        <time>2016-12-07T11:17:55.950213</time>
        <frame>Earth Fixed</frame>
        <position><x>-551342.0</x><y>5901124.4</y><z>6239.7</z></position>
        <velocity><x>2075.0</x><y>-3800.0</y><z>6240.0</z></velocity>
      </orbit>
    </orbitList>
    <azimuthFmRateList count="1">
      <azimuthFmRate>
        <azimuthTime>2016-12-07T11:18:52.178090</azimuthTime>
        <t0>0.0053317936</t0>
        <azimuthFmRatePolynomial>-2330.5 451066.6</azimuthFmRatePolynomial>
      </azimuthFmRate>
    </azimuthFmRateList>
  </generalAnnotation>
  <imageAnnotation>
    <imageInformation>
      <slantRangeTime>0.0053317936</slantRangeTime>
      <rangePixelSpacing>2.329562</rangePixelSpacing>
      <azimuthPixelSpacing>13.96706</azimuthPixelSpacing>
      <azimuthTimeInterval>0.0020555563</azimuthTimeInterval>
      <numberOfSamples>{SAMPLES_PER_BURST}</numberOfSamples>
      <numberOfLines>{N_BURSTS * LINES_PER_BURST}</numberOfLines>
    </imageInformation>
  </imageAnnotation>
  <dopplerCentroid>
    <dcEstimateList count="1">
      <dcEstimate>
        <azimuthTime>2016-12-07T11:18:53.140830</azimuthTime>
        <t0>0.0053422084</t0>
        <geometryDcPolynomial>-3.992857 -287.3943 134566.6</geometryDcPolynomial>
      </dcEstimate>
    </dcEstimateList>
  </dopplerCentroid>
  <swathTiming>
    <linesPerBurst>{LINES_PER_BURST}</linesPerBurst>
    <samplesPerBurst>{SAMPLES_PER_BURST}</samplesPerBurst>
    <burstList count="{N_BURSTS}">{"".join(burst_els)}
    </burstList>
  </swathTiming>
  <geolocationGrid>
    <geolocationGridPointList count="{len(grid_els)}">
{points}
    </geolocationGridPointList>
  </geolocationGrid>
</product>"""


_TAG_COMPRESSION = 259
_TAG_STRIP_OFFSETS = 273
_TAG_ROWS_PER_STRIP = 278
_TAG_STRIP_BYTE_COUNTS = 279
_TYPE_SHORT = 3
_TYPE_LONG = 4


def _measurement_blob() -> bytes:
    """Classic LE strip TIFF: one uncompressed strip per burst at DATA_START."""
    strip_offsets = tuple(DATA_START + k * _STRIP_BYTES for k in range(N_BURSTS))
    strip_counts = tuple(_STRIP_BYTES for _ in range(N_BURSTS))
    entries = [
        (_TAG_COMPRESSION, _TYPE_SHORT, 1, (1,)),
        (_TAG_ROWS_PER_STRIP, _TYPE_SHORT, 1, (LINES_PER_BURST,)),
        (_TAG_STRIP_OFFSETS, _TYPE_LONG, N_BURSTS, strip_offsets),
        (_TAG_STRIP_BYTE_COUNTS, _TYPE_LONG, N_BURSTS, strip_counts),
    ]
    ifd_size = 2 + len(entries) * 12 + 4
    cursor = IFD_OFFSET + ifd_size
    packed: list[tuple[int, int, int, int]] = []
    extras = bytearray()
    for tag, typ, count, values in entries:
        type_size = 2 if typ == _TYPE_SHORT else 4
        if type_size * count <= 4:
            raw = b"".join(
                struct.pack("<H" if typ == _TYPE_SHORT else "<I", v) for v in values
            )
            packed.append(
                (tag, typ, count, int.from_bytes(raw.ljust(4, b"\x00"), "little"))
            )
        else:
            packed.append((tag, typ, count, cursor))
            extras.extend(b"".join(struct.pack("<I", v) for v in values))
            cursor += 4 * count

    ifd = bytearray(struct.pack("<H", len(entries)))
    for tag, typ, count, value_field in packed:
        ifd.extend(struct.pack("<HHII", tag, typ, count, value_field))
    ifd.extend(struct.pack("<I", 0))
    head = bytearray(b"II" + struct.pack("<HI", 42, IFD_OFFSET))
    head.extend(b"\x00" * (IFD_OFFSET - len(head)))
    head.extend(ifd)
    head.extend(extras)
    strips = bytearray()
    for k in range(N_BURSTS):
        rows = np.arange(LINES_PER_BURST, dtype="<i2")[:, None] + k * 100
        cols = np.arange(SAMPLES_PER_BURST, dtype="<i2")[None, :]
        block = np.stack(np.broadcast_arrays(rows, cols), axis=-1)
        strips.extend(block.tobytes())
    assert len(head) <= DATA_START
    blob = bytearray(head)
    blob.extend(b"\x00" * (DATA_START - len(head)))
    blob.extend(strips)
    return bytes(blob)


class _RangeServer:
    """In-process 206 server: exact byte ranges per URL, no network."""

    def __init__(self, blobs: dict[str, bytes], *, fail_after: int | None = None):
        self.blobs = blobs
        self.requests: list[tuple[str, int, int]] = []
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()
        self._count = 0
        self.fail_after = fail_after

    def requested(self, url_prefix: str) -> list[tuple[str, int, int]]:
        return [r for r in self.requests if r[0].startswith(url_prefix)]

    def install(self, patch: Any) -> None:
        """Install the fake ``aiohttp.ClientSession.get`` via a patch object."""
        server = self

        class _Content:
            def __init__(self, data: bytes) -> None:
                self._data = data

            async def iter_chunked(self, size: int) -> AsyncIterator[bytes]:
                for i in range(0, len(self._data), size):
                    yield self._data[i : i + size]

        class _Response:
            def __init__(self, data: bytes, start: int, end: int, total: int) -> None:
                self.status = 206
                self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}
                self.content = _Content(data)

            async def __aenter__(self) -> _Response:
                with server._lock:
                    server.active += 1
                    server.peak = max(server.peak, server.active)
                return self

            async def __aexit__(self, *args: object) -> None:
                with server._lock:
                    server.active -= 1

        def get(session: Any, request_url: str, **kwargs: Any) -> _Response:
            with server._lock:
                server._count += 1
                nth = server._count
            if server.fail_after is not None and nth > server.fail_after:
                raise asyncio.CancelledError("simulated cancellation")
            span_range = str((kwargs.get("headers") or {}).get("Range", ""))
            start_s, end_s = span_range.removeprefix("bytes=").split("-")
            start, end = int(start_s), int(end_s)
            blob = server.blobs[request_url]
            server.requests.append((request_url, start, end))
            return _Response(blob[start : end + 1], start, end, len(blob))

        patch.setattr("aiohttp.ClientSession.get", get)


def _binding(stem: str, xml_bytes: bytes, blob: bytes) -> dict[str, Any]:
    return {
        "product_id": f"S1A_IW_SLC__1SDV_{stem}",
        "annotation_url": _CDSE_URL.format(stem=f"{stem}-annotation"),
        "annotation_size": len(xml_bytes),
        "measurement_url": _CDSE_URL.format(stem=f"{stem}-measurement"),
        "measurement_size": len(blob),
        "polarization": "VV",
        "swath": "IW1",
    }


@pytest.fixture
def offline_transfer(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Two CDSE bindings on separate dates answered by the range server."""
    xml_bytes = _parent_xml().encode()
    xml_bytes_next_day = _parent_xml().replace("2016-12-07", "2016-12-08").encode()
    blob = _measurement_blob()
    items = [
        _binding("TEST1", xml_bytes, blob),
        _binding("TEST2", xml_bytes_next_day, blob),
    ]
    server = _RangeServer(
        {
            items[0]["annotation_url"]: xml_bytes,
            items[0]["measurement_url"]: blob,
            items[1]["annotation_url"]: xml_bytes_next_day,
            items[1]["measurement_url"]: blob,
        }
    )
    server.install(monkeypatch)
    monkeypatch.setattr(cdse_mod, "request_cdse_token", lambda: "fixture-token")
    return {"items": items, "server": server, "xml_bytes": xml_bytes, "blob": blob}


@pytest.fixture
def fetcher_opens(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count RangeFetcher session open/close across the whole test process."""
    seen = {"enter": 0, "exit": 0}
    original_enter = RangeFetcher.__aenter__
    original_exit = RangeFetcher.__aexit__

    async def enter(self: RangeFetcher) -> RangeFetcher:
        seen["enter"] += 1
        return await original_enter(self)

    async def exit_(self: RangeFetcher, *args: Any) -> None:
        seen["exit"] += 1
        await original_exit(self, *args)

    monkeypatch.setattr(RangeFetcher, "__aenter__", enter)
    monkeypatch.setattr(RangeFetcher, "__aexit__", exit_)
    return seen


class TestSharedOperationFetcher:
    """One provider RangeFetcher serves the whole materialization."""

    def test_single_session_across_dates_probes_and_bursts(
        self,
        offline_transfer: dict[str, Any],
        fetcher_opens: dict[str, int],
        tmp_path: Path,
    ) -> None:
        paths = materialize_cdse_burst_safes(
            None, items=offline_transfer["items"], work_dir=tmp_path
        )
        assert len(paths) == 2
        assert fetcher_opens["enter"] == 1, (
            "one operation-scoped RangeFetcher expected, got "
            f"{fetcher_opens['enter']}"
        )
        assert fetcher_opens["exit"] == fetcher_opens["enter"], "session leaked"

    def test_prepare_two_dates_share_one_session(
        self,
        offline_transfer: dict[str, Any],
        fetcher_opens: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The real per-date prepare loop keeps one session across dates."""
        items = offline_transfer["items"]

        @dataclass
        class _FakeSearch:
            items: tuple
            catalog: str = "fake"
            spatial: object = None

        catalog_items = tuple(
            SimpleNamespace(
                acquisition=SimpleNamespace(
                    start=datetime(2016, 12, day, tzinfo=UTC)
                )
            )
            for day in (7, 8)
        )
        monkeypatch.setattr(
            preparation_mod,
            "restore_search",
            lambda record, spatial: _FakeSearch(items=catalog_items),
        )

        @dataclass
        class _FakeRequestConfig:
            roi: object = None
            polarizations: tuple = ("VV",)
            coreg_mode: str = "golden"
            extra: dict = field(default_factory=dict)

        materialized: list[int] = []

        class _FakeStack:
            _request_config = _FakeRequestConfig(
                extra={"polarizations": ("VV",), "orbit_auto_fetch": False}
            )
            _inputs_state = None
            config = SimpleNamespace(work_dir=tmp_path)
            catalog = SimpleNamespace(paths={}, paths_for=lambda target: [])

            @classmethod
            def _materialize_from_search(cls, search: Any, **kwargs: Any) -> Any:
                day_items = [
                    items[0] if it.acquisition.start.day == 7 else items[1]
                    for it in search.items
                ]
                paths = materialize_cdse_burst_safes(
                    None, items=day_items, work_dir=tmp_path
                )
                materialized.append(len(paths))
                return SimpleNamespace(request={"paths": []})

        context = SimpleNamespace(
            budget=None,
            request={
                "kind": "search",
                "provider": "cdse",
                "granularity": "burst",
                "orbit_group": None,
                "orbit_paths": None,
                "aux_cal_paths": None,
                "search": {},
            },
            offline=False,
        )
        prepare_s1_inputs(_FakeStack(), context)
        assert materialized == [1, 1]
        assert fetcher_opens["enter"] == 1, (
            f"one session expected across the two prepare dates, got "
            f"{fetcher_opens['enter']}"
        )
        assert fetcher_opens["exit"] == fetcher_opens["enter"]

    def test_every_url_provider_validated_before_credentialed_request(
        self,
        offline_transfer: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        validated: list[str] = []
        original = cdse_mod.validate_cdse_range_urls

        async def recording(urls: Any) -> None:
            validated.extend(str(url) for url in urls)
            await original(urls)

        monkeypatch.setattr(cdse_mod, "validate_cdse_range_urls", recording)
        server = offline_transfer["server"]
        materialize_cdse_burst_safes(
            None, items=offline_transfer["items"], work_dir=tmp_path
        )
        fetched_urls = {url for url, _start, _end in server.requests}
        assert fetched_urls, "no credentialed requests observed"
        assert fetched_urls <= set(validated), (
            "every fetched URL must be provider-validated before its first "
            f"credentialed request; unvalidated: {fetched_urls - set(validated)}"
        )

    def test_mixed_origin_binding_rejected_not_pooled(
        self,
        offline_transfer: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        item = dict(offline_transfer["items"][0])
        item["measurement_url"] = "https://fixture.example/measurement.tiff"
        with pytest.raises(ValueError, match=r"origin|mixed|pool"):
            materialize_cdse_burst_safes(None, items=[item], work_dir=tmp_path)


class TestProgressAndLifecycle:
    """Useful-data progress and clean cancellation semantics."""

    def test_useful_data_progress_logged(
        self,
        offline_transfer: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        import logging

        with caplog.at_level(logging.INFO, logger="faninsar.missions.s1.acquisition"):
            materialize_cdse_burst_safes(
                None, items=offline_transfer["items"], work_dir=tmp_path
            )
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "useful_bytes=" in messages, messages
        assert "bursts=" in messages, messages

    def test_real_thread_cancellation_closes_sessions(
        self,
        offline_transfer: dict[str, Any],
        fetcher_opens: dict[str, int],
        tmp_path: Path,
    ) -> None:
        """Cancellation reaches a real caller thread and unwinds ordered.

        The fetcher close must happen only after pending bridge futures
        are cancelled and the loop settles; a close-before-join bug would
        strand the caller on a dead loop or leave the session open.
        """
        server = _RangeServer(offline_transfer["server"].blobs, fail_after=2)
        errors: list[BaseException] = []

        def run() -> None:
            with pytest.MonkeyPatch.context() as mp:
                server.install(mp)
                try:
                    materialize_cdse_burst_safes(
                        None, items=offline_transfer["items"], work_dir=tmp_path
                    )
                except BaseException as exc:  # noqa: BLE001 - the point
                    errors.append(exc)

        worker = threading.Thread(target=run, name="materialize-cancel-probe")
        worker.start()
        worker.join(timeout=120)
        assert not worker.is_alive(), "materialize thread hung on cancellation"
        assert errors, "cancellation did not propagate"
        assert isinstance(
            errors[0],
            asyncio.CancelledError | RuntimeError | concurrent_futures.CancelledError,
        )
        assert fetcher_opens["enter"] >= 1
        assert fetcher_opens["exit"] == fetcher_opens["enter"], "session leaked"
        assert list(tmp_path.rglob("*.SAFE")) == [], "torn path-SAFE published"
        alive = [
            thread.name
            for thread in threading.enumerate()
            if "faninsar-s1" in thread.name
        ]
        assert alive == [], f"operation threads left running: {alive}"


class TestBridgeInterruption:
    """SIGINT during a bridge wait must cancel the unowned coroutine."""

    def test_bridge_interrupt_cancels_leftover_coroutine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A KeyboardInterrupt in ``future.result()`` cancels the task.

        The real SIGINT path parks the caller inside
        ``concurrent.futures.Future.result()``; the exception must cancel
        the bridged future, and ``close`` must then cancel every remaining
        task owned by the dedicated loop, gather the cancellations, run
        the provider ``__aexit__``, and stop promptly — never wait a fixed
        drain timeout for a cooldown timer it is about to cancel anyway.
        """
        operation = acquisition_mod._CdseBurstTransfer()
        started = threading.Event()
        running = threading.Event()
        inners: list[Any] = []
        original_bridger = asyncio.run_coroutine_threadsafe

        async def sleeper() -> None:
            running.set()
            await asyncio.sleep(3600)

        class _KBIOnResult:
            """Future wrapper whose first ``result`` is interrupted mid-wait."""

            def __init__(self, inner: Any, interrupt: bool) -> None:
                self._inner = inner
                self._interrupt = interrupt

            def cancel(self) -> bool:
                return self._inner.cancel()

            def result(self, timeout: float | None = None) -> Any:
                if not self._interrupt:
                    return self._inner.result(timeout)
                self._interrupt = False
                started.set()
                assert running.wait(timeout=10), "coroutine never started"
                raise KeyboardInterrupt

        interrupt_flags = [True]

        def bridging(coro: Any, loop: Any) -> _KBIOnResult:
            inner = original_bridger(coro, loop)
            inners.append(inner)
            interrupt = bool(interrupt_flags)
            if interrupt_flags:
                interrupt_flags.pop(0)
            return _KBIOnResult(inner, interrupt=interrupt)

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", bridging)
        errors: list[BaseException] = []

        def caller() -> None:
            try:
                operation._bridge(sleeper())
            except BaseException as exc:  # noqa: BLE001 - the point
                errors.append(exc)

        thread = threading.Thread(target=caller, name="bridge-kbi-probe")
        thread.start()
        assert running.wait(10), "coroutine never ran on the operation loop"
        thread.join(timeout=10)
        assert not thread.is_alive(), "caller hung on interrupted bridge"
        assert isinstance(errors[0], KeyboardInterrupt)

        inner = inners[0]
        deadline = time.monotonic() + 10
        while not inner.cancelled() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert inner.cancelled(), "unowned coroutine left running on the loop"

        # A live provider session plus a cooldown-shaped timer on the
        # dedicated loop: close must cancel-and-gather, not wait it out.

        class _DummyCM:
            def __init__(self) -> None:
                self.exited = False
                self.cooldown_cancelled_at_exit = False

            async def __aexit__(self, *args: object) -> None:
                self.exited = True
                self.cooldown_cancelled_at_exit = cooldown.cancelled()

        cooldown = original_bridger(asyncio.sleep(3600), operation._loop)
        time.sleep(0.1)  # let the timer task start on the loop
        dummy = _DummyCM()
        operation._cm = dummy
        operation._fetcher = object()
        close_started = time.monotonic()
        operation.close()
        elapsed = time.monotonic() - close_started
        assert elapsed < 10, (
            f"close blocked {elapsed:.1f}s on leftover loop tasks it "
            "should have cancelled"
        )
        assert dummy.exited, "provider __aexit__ never ran"
        assert dummy.cooldown_cancelled_at_exit, (
            "cooldown task was not cancelled before the session closed"
        )
        assert cooldown.cancelled() or cooldown.done()
        assert not any(
            "faninsar-s1" in thread.name for thread in threading.enumerate()
        ), "operation loop thread left running"

    def test_close_closes_fetcher_opened_during_drain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fetcher opened while ``close`` drains is still closed.

        The lazy open runs on the operation loop, so a close racing an
        in-flight open (SIGINT while the first token grant blocks the
        loop) must resolve the fetcher context inside the drain
        coroutine, not from a caller-thread snapshot taken before the
        open landed.
        """
        operation = acquisition_mod._CdseBurstTransfer()
        original = asyncio.run_coroutine_threadsafe
        opened = threading.Event()

        class _LazyCM:
            def __init__(self) -> None:
                self.exited = False

            async def __aexit__(self, *args: object) -> None:
                self.exited = True

        dummy = _LazyCM()

        def bridging(coro: Any, loop: Any) -> Any:
            if not opened.is_set():
                opened.set()
                # Emulate the lazy open landing on the loop side while a
                # close started on the caller thread with no fetcher yet.
                operation._cm = dummy
                operation._fetcher = object()
            return original(coro, loop)

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", bridging)
        operation._ensure_loop()
        operation.close()
        assert opened.is_set()
        assert dummy.exited, (
            "fetcher opened during close was never closed — the drain must "
            "read the context on the loop after cancelling tasks"
        )
        assert not any(
            "faninsar-s1" in thread.name for thread in threading.enumerate()
        ), "operation loop thread left running"


def test_range_server_fixture_serves_parent_geometry() -> None:
    """Guard: the synthetic parent parses and the blob matches geometry."""
    swath = parse_annotation_xml(
        _parent_xml(),
        annotation_path="memory://parent.xml",
        measurement_path="memory://parent.tiff",
    )
    assert swath.lines_per_burst == LINES_PER_BURST
    assert swath.samples_per_burst == SAMPLES_PER_BURST
    assert len(swath.bursts) == N_BURSTS
    assert len(_measurement_blob()) == DATA_START + N_BURSTS * _STRIP_BYTES
