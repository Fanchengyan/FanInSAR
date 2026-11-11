"""Async byte-range spans. RangeFetcher lives here; aiohttp is lazy."""

from __future__ import annotations

import asyncio
import os
import re
import time
import urllib.parse
import urllib.request
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Self

from faninsar.logging import reject, setup_logger
from faninsar.remote.errors import (
    RemoteAccessError,
    RemoteIntegrityError,
    RemoteLimitError,
    _fail,
)
from faninsar.remote.protocols import _CallLedger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from types import TracebackType

    from faninsar.remote.records import RemoteResourceBudget

logger = setup_logger(__name__)

# Sit at this concurrency until a 429. A virgin 429 (no climb path) first
# drops to 3/4, then to 1/2 of that origin. After a climb, a later 429
# walks back the same steps (19→18→17→15) instead of cutting in half.
# Concurrent 429s share one decrease. 401 is not congestion.
_AIMD_SEED = 20
_CONNECT_ATTEMPTS = 10
_CONNECT_BACKOFF_S = 2.0
_RATE_LIMIT_ATTEMPTS = 5
_RATE_LIMIT_BACKOFF_S = 1.5
_RETRY_AFTER_MAX_S = 300.0
# After a cut, leftover 429s from the same wave keep arriving at the new
# window. Ignore them — and do not probe up — until this hold elapses.
_AIMD_HOLD_S = 15.0
_SOCK_CONNECT_S = 30
_SOCK_READ_S = 60
_STREAM_CHUNK_BYTES = 16 * 1024

# RFC 9110 single-range Content-Range: ``bytes <first>-<last>/<complete|*>``.
# Exact match against the requested span is required; anything looser would
# admit a prefix/suffix window that silently corrupts offset assembly.
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")
_ACTIVE_RANGE: ContextVar[_RangeAttempt | None] = ContextVar(
    "range_fetcher", default=None
)


class _RateLimitError(RemoteAccessError):
    """Carry a server delay without exposing response headers or URLs."""

    def __init__(self, retry_after: float = 0.0) -> None:
        self.retry_after = retry_after
        super().__init__("range_unavailable", "range request was rate-limited")


def _retry_after_seconds(value: str) -> float:
    """Parse Retry-After seconds or an HTTP date; invalid values use backoff."""
    value = value.strip()
    if value.isascii() and value.isdecimal():
        return float(value)
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        return max(0.0, date.timestamp() - time.time())
    except (ValueError, TypeError, OverflowError):
        pass
    return 0.0


@dataclass(slots=True)
class _RangeAttempt:
    """Task-local admission bookkeeping without changing getter signatures."""

    fetcher: RangeFetcher
    attempt: int = 0
    origin: int | None = None
    sent: int = 0
    cooldown_started: bool = False
    metadata: bool = False


@dataclass(frozen=True, slots=True)
class RangeSpan:
    """Inclusive byte interval on one object URL.

    Parameters
    ----------
    url : str
        Object URL the fetcher will request. Mission adapters emit
        already-validated provider URLs.
    start, end : int
        Inclusive byte offsets (HTTP ``Range: bytes=start-end``).
    label : str
        Caller-owned identity for resume/assembly. With metadata admission
        enabled, ``annotation`` and ``annotation-*`` identify metadata work;
        every other label is payload. Object URLs are never used to classify.

    """

    url: str
    start: int
    end: int
    label: str


@dataclass(slots=True)
class _Revision:
    """Strong ETag pinning one span's received prefix to one object revision.

    ``etag`` is sent as ``If-Range`` on every request that resumes a
    non-empty prefix.  A request that starts at the span's first byte
    replaces it with the response's strong ``ETag`` and reports that value
    through ``on_etag``.  With no ``etag`` no validator is sent.
    """

    etag: str | None = None
    on_etag: Callable[[str], None] | None = None


class RangeFetcher:
    """Reusable async session that concatenates already-sized spans.

    The AIMD window stays at ``_AIMD_SEED`` until a 429. With no climb path
    it drops to 3/4, then 1/2 of that origin; after a climb it walks back
    the same steps. ``TCPConnector.limit`` tracks the same integer; queued
    requests also wait for in-flight work to drain below that window. A 429
    pauses every new attempt until the shared backoff/``Retry-After`` deadline.
    Server delays above 300 seconds fail this context closed rather than retry
    early. 401 is auth refresh, not a window event.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        refresh: Callable[[], Awaitable[str]] | None = None,
        budget: RemoteResourceBudget | None = None,
        ledger: _CallLedger | None = None,
    ) -> None:
        """Create a fetcher; the session is opened by ``async with``.

        Parameters
        ----------
        token : str or None
            Optional access token sent on each range request.
        refresh : callable or None
            Async callable returning a replacement token. Concurrent 401
            responses share one in-flight refresh; each span is requeued
            once after that result.
        budget : RemoteResourceBudget, optional
            Network guards shared by all spans in this context. ``max_workers``
            caps all active HTTP requests; ``max_ranges_per_file`` optionally
            caps each exact URL within that same pool, including metadata URLs.
            ``max_metadata_files`` enables distinct active-URL admission bounded
            by ``max_active_files``, reserving metadata slots while annotation
            spans are pending. In-flight payload is never preempted. Item-count
            and disk/cache quotas remain outside range transfer. Omission retains
            the AIMD and socket defaults; this is client policy, not a server quota.
        ledger : _CallLedger, optional
            Existing operation accounting. Its budget is authoritative; a
            conflicting explicit budget is rejected.

        """
        if ledger is not None and budget is not None and ledger.budget != budget:
            reject(
                "budget must match the supplied range ledger", ValueError, logger=logger
            )
        self._ledger = (
            ledger
            if ledger is not None
            else (_CallLedger(budget) if budget is not None else None)
        )
        self._budget = self._ledger.budget if self._ledger is not None else None
        self._ceiling = (
            min(_AIMD_SEED, self._budget.max_workers)
            if self._budget is not None
            else _AIMD_SEED
        )
        self._session: Any = None
        self._connector: Any = None
        self._token = token
        self._refresh = refresh
        self._refresh_lock = asyncio.Lock()
        self._generation = 0
        self._window = self._ceiling
        self._lo = 0
        self._hi = self._ceiling + 1
        self._origin = self._ceiling
        self._ladder: list[int] = []
        self._virgin = 0
        self._congested = False
        self._hold_until = 0.0
        self._cool: asyncio.Event | None = None
        self._cool_task: asyncio.Task[None] | None = None
        self._cool_waiters = 0
        self._cool_until = 0.0
        self._rate_limit_error: RemoteAccessError | None = None
        self._inflight = 0
        self._by_url_inflight: dict[str, int] = {}
        self._max_ranges_per_file = (
            self._budget.max_ranges_per_file if self._budget is not None else None
        )
        self._metadata_limit = (
            self._budget.max_metadata_files if self._budget is not None else None
        )
        self._metadata_pending = 0
        self._metadata_inflight: dict[str, int] = {}
        self._available = asyncio.Event()
        self._max_retries_per_object: int | None = None
        self._url_429_counts: dict[str, int] = {}
        self._active_spans = 0
        self._active_batches = 0

    def configure_rate_limit_policy(
        self,
        *,
        max_retries_per_object: int | None = None,
        max_workers: int | None = None,
        reset_counters: bool = False,
    ) -> None:
        """Configure an operation-owned URL retry stage without reopening HTTP.

        Parameters
        ----------
        max_retries_per_object : int or None, optional
            Retry allowance shared by every span on an exact URL. Ten allows
            ten retries; its eleventh actual 429 exhausts that URL. Zero makes
            its first 429 terminal. None retains five attempts per span.
        max_workers : int or None, optional
            Lower the shared HTTP concurrency ceiling; never increase the
            original budget or a previously lowered ceiling.
        reset_counters : bool, optional
            Clear URL 429 counts when entering a new stage. The caller must
            first drain every span from the preceding stage. Session, ledger,
            AIMD history and any current Retry-After cooldown are retained.

        Raises
        ------
        RuntimeError
            If any active, queued or cooling span has not drained.
        ValueError
            If a retry allowance or worker ceiling is invalid.

        Notes
        -----
        An exhausted URL raises ``RemoteAccessError`` with reason
        ``rate_limit_exhausted`` without stopping other URLs. Only actual 429
        responses consume its allowance; metadata URLs have separate counts.

        """
        if self._active_spans or self._active_batches:
            reject(
                "drain all range spans before changing retry stage",
                RuntimeError,
                logger=logger,
            )
        if max_retries_per_object is not None and (
            isinstance(max_retries_per_object, bool)
            or not isinstance(max_retries_per_object, int)
            or max_retries_per_object < 0
        ):
            reject(
                "object retry allowance must be a non-negative integer",
                ValueError,
                logger=logger,
            )
        if max_workers is not None and (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers < 1
        ):
            reject(
                "range worker ceiling must be a positive integer",
                ValueError,
                logger=logger,
            )
        self._max_retries_per_object = max_retries_per_object
        if reset_counters:
            self._url_429_counts.clear()
        if max_workers is not None:
            self._ceiling = min(self._ceiling, max_workers)
            self._hi = min(self._hi, self._ceiling + 1)
            self._set_window(self._window)

    def _check_object_rate_limit(self, url: str) -> None:
        """Reject exhausted exact URLs without logging credentials or recounting."""
        maximum = self._max_retries_per_object
        if maximum is not None and self._url_429_counts.get(url, 0) > maximum:
            reason = "rate_limit_exhausted"
            raise RemoteAccessError(
                reason, "range object exhausted its 429 retry allowance"
            )

    def _rate_limit_delay(self, attempt: int, retry_after: float) -> float:
        """Bound enabled-policy client backoff, never the server's Retry-After."""
        if self._max_retries_per_object is None:
            backoff = _RATE_LIMIT_BACKOFF_S * (2**attempt)
        else:
            backoff = min(60.0, _RATE_LIMIT_BACKOFF_S * (2 ** min(attempt, 6)))
        return max(backoff, retry_after)

    async def _wait_event(self, event: asyncio.Event) -> None:
        """Bound this waiter by the operation deadline, not the shared task."""
        ledger = self._ledger
        if ledger is None:
            await event.wait()
        else:
            ledger.check_elapsed()
            remaining = ledger.budget.max_elapsed_seconds - (
                time.monotonic() - ledger.started
            )
            try:
                async with asyncio.timeout(max(0.0, remaining)):
                    await event.wait()
            except TimeoutError:
                _fail(RemoteLimitError, "max_elapsed_seconds")
            ledger.check_elapsed()
        if self._rate_limit_error is not None:
            logger.error("range session stopped after an excessive Retry-After")
            raise self._rate_limit_error

    def _metadata_span(self, span: RangeSpan) -> bool:
        """Classify opt-in metadata by public label, never by secret object URL."""
        return self._metadata_limit is not None and (
            span.label == "annotation" or span.label.startswith("annotation-")
        )

    def _finish_metadata(self) -> None:
        """Release a pending reservation on success, failure or cancellation."""
        self._metadata_pending -= 1
        self._available.set()

    def _file_available(self, url: str, metadata: bool) -> bool:
        """Reserve client file slots while metadata is pending; never preempt."""
        limit = self._metadata_limit
        if limit is None:
            return True
        assert self._budget is not None
        files = self._by_url_inflight
        annotations = self._metadata_inflight
        if not metadata and self._metadata_pending:
            payload_http = self._inflight - sum(annotations.values())
            if payload_http >= max(0, self._window - limit):
                return False
        if url not in files and len(files) >= self._budget.max_active_files:
            return False
        if metadata:
            return url in annotations or len(annotations) < limit
        payload_limit = self._budget.max_active_files - (
            limit if self._metadata_pending else 0
        )
        payloads = files.keys() - annotations.keys()
        return len(payloads) < payload_limit or (
            url in payloads and len(payloads) == payload_limit
        )

    async def _acquire(self, url: str) -> int:
        """Admit below HTTP, URL and optional file caps, outside cooldown."""
        state = _ACTIVE_RANGE.get()
        metadata = state.metadata if state is not None else False
        while True:
            self._check_object_rate_limit(url)
            if self._rate_limit_error is not None:
                logger.error("range session stopped after an excessive Retry-After")
                raise self._rate_limit_error
            if self._cool is not None:
                await self._wait_event(self._cool)
                continue
            by_url = self._by_url_inflight.get(url, 0)
            if (
                self._inflight < self._window
                and self._file_available(url, metadata)
                and (
                    self._max_ranges_per_file is None
                    or by_url < self._max_ranges_per_file
                )
            ):
                self._inflight += 1
                self._by_url_inflight[url] = by_url + 1
                if metadata:
                    annotations = self._metadata_inflight
                    annotations[url] = annotations.get(url, 0) + 1
                return self._window
            self._available.clear()
            await self._wait_event(self._available)

    def _release(self, url: str) -> None:
        """Release global and exact-URL slots and wake every eligible waiter."""
        self._inflight -= 1
        remaining = self._by_url_inflight[url] - 1
        if remaining:
            self._by_url_inflight[url] = remaining
        else:
            del self._by_url_inflight[url]
        state = _ACTIVE_RANGE.get()
        if state is not None and state.metadata:
            self._metadata_inflight[url] -= 1
            if not self._metadata_inflight[url]:
                del self._metadata_inflight[url]
        self._available.set()

    @property
    def window(self) -> int:
        """Current AIMD concurrency window, shared with ``TCPConnector.limit``."""
        return self._window

    def _set_window(self, value: int) -> None:
        """Keep the AIMD window and connector limit on the same integer."""
        self._window = max(1, min(value, self._ceiling))
        if self._connector is not None:
            self._connector._limit = self._window

    def _holding(self) -> bool:
        """Return whether leftover 429s from the last cut should be ignored."""
        try:
            now = asyncio.get_running_loop().time()
        except RuntimeError:
            return False
        return now < self._hold_until

    def _retreat(self, sent: int) -> int:
        """Walk back a remembered climb, else 3/4 then 1/2 of the origin."""
        while self._ladder and self._ladder[-1] >= sent:
            self._ladder.pop()
        if self._ladder:
            return self._ladder[-1]
        if self._virgin == 0:
            self._origin = sent
            self._virgin = 1
            nxt = max(1, (sent * 3) // 4)
            self._ladder = [nxt]
            return nxt
        nxt = max(1, self._origin // 2)
        if nxt >= self._window:
            nxt = max(1, self._window // 2)
        self._ladder = [nxt]
        self._virgin = 2
        return nxt

    def _on_rate_limit(self, sent: int) -> None:
        """Shrink once per congestion epoch; 3/4 then 1/2, or walk the climb.

        Every 429 extends a quiet hold. Leftover 429s from the same wave are
        not a new ceiling — ignore them until ``_AIMD_HOLD_S`` elapses with
        no further 429, then a success may probe up.
        """
        holding = self._holding()
        try:
            self._hold_until = asyncio.get_running_loop().time() + _AIMD_HOLD_S
        except RuntimeError:
            self._hold_until = 0.0
        if sent != self._window or holding or self._congested:
            return
        self._congested = True
        self._hi = min(self._hi, sent)
        nxt = self._retreat(sent)
        nxt = max(1, min(nxt, self._window, self._ceiling))
        if self._hi > 1:
            nxt = min(nxt, self._hi - 1)
        if self._lo >= nxt:
            self._lo = max(0, nxt - 1)
        if nxt != self._window:
            logger.warning(
                "range AIMD window %d -> %d (lo=%d hi=%d)",
                self._window,
                nxt,
                self._lo,
                self._hi,
            )
        self._set_window(nxt)

    def _on_success(self, sent: int) -> None:
        """Mark ``sent`` safe and binary-search toward the last trigger.

        A 206 that left at a higher window is leftover in-flight. During the
        post-cut hold, successes may raise ``lo`` but must not probe up —
        leftover 429s would otherwise look like a failed probe. Each probe
        is remembered so a later 429 can walk the same path back.
        """
        if sent > self._window:
            return
        if sent < self._hi:
            self._lo = max(self._lo, sent)
        if sent == self._window:
            self._congested = False
        if self._holding() or self._cool is not None:
            return
        if sent != self._window or self._window >= self._hi - 1:
            return
        nxt = min(self._ceiling, (self._window + self._hi) // 2)
        if nxt != self._window:
            logger.info(
                "range AIMD window %d -> %d (lo=%d hi=%d)",
                self._window,
                nxt,
                self._lo,
                self._hi,
            )
            if not self._ladder or self._ladder[-1] != nxt:
                self._ladder.append(nxt)
        self._set_window(nxt)

    def _start_rate_limit(self, sent: int, delay: float) -> asyncio.Event:
        """Publish a shared cooldown before releasing a rate-limited response."""
        self._on_rate_limit(sent)
        if delay > _RETRY_AFTER_MAX_S:
            self._rate_limit_error = _RateLimitError(delay)
            self._available.set()
            if self._cool is not None:
                self._cool.set()
                self._cool = None
            if self._cool_task is not None:
                self._cool_task.cancel()
            cool = asyncio.Event()
            cool.set()
            return cool
        self._cool_until = max(
            self._cool_until, asyncio.get_running_loop().time() + delay
        )
        self._cool_waiters += 1
        cool = self._cool
        if cool is None:
            cool = asyncio.Event()
            self._cool = cool

            async def _release() -> None:
                try:
                    prev = -1
                    while prev != self._cool_waiters:
                        prev = self._cool_waiters
                        await asyncio.sleep(0)
                    while True:
                        remaining = self._cool_until - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            break
                        await asyncio.sleep(remaining)
                finally:
                    if self._cool is cool:
                        self._cool = None
                    self._cool_task = None
                    self._cool_waiters = 0
                    cool.set()

            self._cool_task = asyncio.create_task(_release())
        return cool

    async def _wait_rate_limit(
        self, sent: int, attempt: int, retry_after: float = 0.0
    ) -> None:
        """Decrease once, then wait for the shared congestion deadline."""
        cool = self._start_rate_limit(
            sent, self._rate_limit_delay(attempt, retry_after)
        )
        await self._wait_event(cool)

    async def __aenter__(self) -> Self:
        """Open one ClientSession; live streams have no total timeout."""
        import aiohttp

        self._connector = aiohttp.TCPConnector(limit=self._window)
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            auto_decompress=False,
            headers={"Accept-Encoding": "identity"},
            trust_env=False,
            timeout=aiohttp.ClientTimeout(
                total=self._budget.max_elapsed_seconds if self._budget else None,
                sock_connect=self._budget.connect_timeout_seconds
                if self._budget
                else _SOCK_CONNECT_S,
                sock_read=self._budget.read_timeout_seconds
                if self._budget
                else _SOCK_READ_S,
            ),
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the session."""
        del exc_type, exc, tb
        task = self._cool_task
        self._cool_task = None
        if task is not None:
            task.cancel()
        session = self._session
        self._session = None
        self._connector = None
        if session is not None:
            await session.close()

    async def _refresh_token(self, generation: int) -> None:
        """Refresh once; waiters that 401'd on the same generation share it."""
        refresh = self._refresh
        if refresh is None:
            return
        async with self._refresh_lock:
            if self._generation != generation:
                return
            self._token = await refresh()
            self._generation += 1

    async def fetch(self, spans: Iterable[RangeSpan]) -> dict[str, bytes]:
        """GET each span as ``Range: bytes=start-end`` and return bodies."""
        session = self._session
        if session is None:
            msg = "RangeFetcher must be used as an async context manager"
            reject(msg, RuntimeError, logger=logger)
        span_list = tuple(spans)
        tasks: list[asyncio.Task[bytes]] = []
        self._active_batches += 1
        self._metadata_pending += sum(self._metadata_span(span) for span in span_list)
        try:
            for span in span_list:
                task = asyncio.create_task(_get_span(self, span))
                if self._metadata_span(span):
                    task.add_done_callback(lambda _: self._finish_metadata())
                tasks.append(task)
            bodies = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            self._active_batches -= 1
        return {span.label: body for span, body in zip(span_list, bodies, strict=True)}

    async def fetch_into(
        self,
        span: RangeSpan,
        sink: str | Path | BinaryIO,
        *,
        on_bytes: Callable[[int], None] | None = None,
        etag: str | None = None,
        on_etag: Callable[[str], None] | None = None,
    ) -> int:
        """Stream one span body into a caller-provided part file or byte sink.

        ``sink`` is either a filesystem path (a partial-download ``.part``
        file, opened in append-binary mode) or a binary file-like object.
        The number of already-received prefix bytes is measured from the
        destination itself: the file size for a path, ``tell()`` for a
        stream.  Those bytes are treated as the span's already-received
        prefix and the GET resumes at ``start + received``, so an
        interrupted earlier call continues instead of re-downloading.

        Mid-stream drops (:class:`aiohttp.ClientPayloadError`, connection
        failures, truncated payloads) resume with a suffix ``Range``
        request for the missing tail under the same bounded attempt budget
        as whole-span retries.  A sink positioned past the span length is
        an integrity failure, never a silent truncation.

        Parameters
        ----------
        span : RangeSpan
            The inclusive byte interval to fetch.
        sink : str, Path, or binary file-like
            Destination for the span body.  Paths are opened in
            append-binary mode for the duration of the call and flushed
            closed before returning; streams must support ``write`` and
            ``tell`` and stay owned by the caller.
        on_bytes : callable, optional
            Called with the byte count of each streamed chunk, for download
            progress reporting.  Purely observational; it never affects the
            fetch.
        etag : str, optional
            Strong ``ETag`` recorded when the existing prefix was started.
            Every request that resumes a non-empty prefix sends it as
            ``If-Range``; a response other than the exact expected 206 then
            raises ``RemoteIntegrityError("revision_changed")`` without
            consuming the body.  ``None`` sends no validator and assumes the
            object is immutable.
        on_etag : callable, optional
            Called with the strong ``ETag`` of a response to a request that
            starts at ``span.start``; later resumes in this call pin to it.

        Returns
        -------
        int
            The span length in bytes (``end - start + 1``); the sink now
            holds the complete span body.

        Raises
        ------
        RuntimeError
            If the fetcher is not inside ``async with``.
        TypeError
            If ``sink`` is neither a path nor a writable, tell-able stream.
        RemoteIntegrityError
            If the destination already holds more than the span length, or
            a resumed response fails the strict 206/Content-Range check
            (reason ``revision_changed`` when ``If-Range`` was sent).
        RemoteAccessError
            For the same redirect/401/429 semantics as :meth:`fetch`.

        """
        session = self._session
        if session is None:
            msg = "RangeFetcher must be used as an async context manager"
            reject(msg, RuntimeError, logger=logger)
        write, received, close = _open_sink(span, sink)
        if on_bytes is not None:
            sink_write = write

            def write(chunk: bytes) -> object:  # type: ignore[misc]
                on_bytes(len(chunk))
                return sink_write(chunk)

        revision = _Revision(
            etag=etag.strip() if etag and not etag.strip().startswith("W/") else None,
            on_etag=on_etag,
        )
        metadata = self._metadata_span(span)
        self._metadata_pending += metadata
        try:
            await _attempt_span_stream(
                self, span, write, received, allow_refresh=True, revision=revision
            )
        finally:
            if metadata:
                self._finish_metadata()
            close()
        return span.end - span.start + 1


async def fetch_ranges_async(
    spans: Iterable[RangeSpan],
    *,
    fetcher: RangeFetcher | None = None,
) -> dict[str, bytes]:
    """Fetch already-sized spans, optionally reusing a ``RangeFetcher``."""
    if fetcher is not None:
        return await fetcher.fetch(spans)
    async with RangeFetcher() as owned:
        return await owned.fetch(spans)


def _open_sink(
    span: RangeSpan, sink: str | Path | BinaryIO
) -> tuple[Callable[[bytes], object], Callable[[], int], Callable[[], None]]:
    """Adapt ``sink`` to (write, received, close) for the streaming path.

    A path is opened in append-binary mode and its size is the received
    prefix; a stream uses its own ``tell()``.  Each path write is flushed
    so a few kilobytes of body are visible on disk without waiting for
    Python's file buffer.  The returned ``received`` reflects live
    position so a resumed retry continues where the last write stopped.
    """
    expected = span.end - span.start + 1
    if isinstance(sink, str | Path):
        part = Path(sink)
        part.parent.mkdir(parents=True, exist_ok=True)
        stream = part.open("ab")
        received0 = stream.tell()
        if received0 > expected:
            stream.close()
            logger.error(
                "range part file is larger than the requested span",
                extra={"label": span.label, "size": received0},
            )
            _fail(
                RemoteIntegrityError,
                "part_file_oversized",
                "range part file is larger than the requested span",
            )

        def write(chunk: bytes) -> object:
            result = stream.write(chunk)
            stream.flush()
            return result

        def close() -> None:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()

        return write, stream.tell, close
    write = getattr(sink, "write", None)
    tell = getattr(sink, "tell", None)
    if not callable(write) or not callable(tell):
        msg = "sink must be a path or a binary stream supporting write() and tell()"
        reject(msg, TypeError, logger=logger)
    if tell() > expected:
        logger.error(
            "range sink position is past the requested span",
            extra={"label": span.label},
        )
        _fail(
            RemoteIntegrityError,
            "part_file_oversized",
            "range sink position is past the requested span",
        )
    return write, tell, lambda: None


async def _get_span(fetcher: RangeFetcher, span: RangeSpan) -> bytes:
    """Return one 206 body whose Content-Range matches the span."""
    return await _attempt_span(fetcher, span, allow_refresh=True)


async def _attempt_span(
    fetcher: RangeFetcher, span: RangeSpan, *, allow_refresh: bool
) -> bytes:
    """Retry a whole span once after 429; one 401 requeues after a shared refresh."""
    body = bytearray()

    def received() -> int:
        return len(body)

    await _attempt_span_stream(
        fetcher, span, body.extend, received, allow_refresh=allow_refresh
    )
    return bytes(body)


async def _attempt_span_stream(
    fetcher: RangeFetcher,
    span: RangeSpan,
    write: Callable[[bytes], object],
    received: Callable[[], int],
    *,
    allow_refresh: bool,
    revision: _Revision | None = None,
) -> None:
    """Track the entire span lifetime, including admission, refresh and cooldown."""
    fetcher._active_spans += 1
    try:
        await _retry_span_stream(
            fetcher,
            span,
            write,
            received,
            allow_refresh=allow_refresh,
            revision=revision,
        )
    finally:
        fetcher._active_spans -= 1


async def _retry_span_stream(
    fetcher: RangeFetcher,
    span: RangeSpan,
    write: Callable[[bytes], object],
    received: Callable[[], int],
    *,
    allow_refresh: bool,
    revision: _Revision | None = None,
) -> None:
    """Drive the streaming span through 429 requeue and shared 401 refresh."""
    last_access_error: RemoteAccessError | None = None
    state = _RangeAttempt(fetcher, metadata=fetcher._metadata_span(span))
    maximum = fetcher._max_retries_per_object
    attempts = _RATE_LIMIT_ATTEMPTS if maximum is None else maximum + 1
    for attempt in range(attempts):
        state.attempt = attempt
        state.sent = fetcher._window
        state.cooldown_started = False
        generation = fetcher._generation
        try:
            context = _ACTIVE_RANGE.set(state)
            try:
                fetcher._check_object_rate_limit(span.url)
                if fetcher._rate_limit_error is not None:
                    logger.error("range session stopped after an excessive Retry-After")
                    raise fetcher._rate_limit_error
                if fetcher._cool is not None:
                    await fetcher._wait_event(fetcher._cool)
                await _stream_span(
                    fetcher._session,
                    span,
                    write,
                    received,
                    token=fetcher._token,
                    revision=revision,
                    ledger=fetcher._ledger,
                )
            finally:
                _ACTIVE_RANGE.reset(context)
        except RemoteAccessError as error:
            last_access_error = error
            if (
                error.reason == "unauthorized"
                and allow_refresh
                and fetcher._refresh is not None
            ):
                if fetcher._ledger is not None:
                    fetcher._ledger.retry()
                await fetcher._refresh_token(generation)
                await _attempt_span_stream(
                    fetcher,
                    span,
                    write,
                    received,
                    allow_refresh=False,
                    revision=revision,
                )
                return
            if error.reason != "range_unavailable":
                raise
            retry_after = (
                error.retry_after if isinstance(error, _RateLimitError) else 0.0
            )
            if retry_after > _RETRY_AFTER_MAX_S:
                raise
            # Rate-limited: pin the window this span first sent at. Retries
            # after a cut would otherwise look like the new window 429'd.
            origin_window = state.origin if state.origin is not None else state.sent
            if attempt < attempts - 1:
                if fetcher._ledger is not None:
                    fetcher._ledger.retry()
                if state.cooldown_started:
                    if fetcher._cool is not None:
                        await fetcher._wait_event(fetcher._cool)
                else:
                    await fetcher._wait_rate_limit(origin_window, attempt, retry_after)
            elif not state.cooldown_started:
                fetcher._start_rate_limit(origin_window, retry_after)
        else:
            fetcher._on_success(state.sent)
            return
    assert last_access_error is not None
    raise last_access_error


async def _request_span(
    session: Any, span: RangeSpan, *, token: str | None = None
) -> bytes:
    """GET one span; transient connection drops are retried before failing.

    Proxy CONNECT failures, TLS handshake EOFs, and server disconnects are
    bounded retries of an idempotent span GET; 429/401 semantics are owned
    by :func:`_attempt_span` above this seam.
    """
    body = bytearray()
    await _stream_span(session, span, body.extend, lambda: len(body), token=token)
    return bytes(body)


async def _stream_span(
    session: Any,
    span: RangeSpan,
    write: Callable[[bytes], object],
    received: Callable[[], int],
    *,
    token: str | None = None,
    revision: _Revision | None = None,
    ledger: _CallLedger | None = None,
) -> None:
    """Stream one span into ``write``; mid-stream drops resume the suffix.

    Each attempt issues ``Range: bytes=(start+received)-end`` so a
    connection, payload, or timeout failure that arrives after some
    bytes were already sunk resumes from the live position instead of
    restarting the span.  Whole-request failures resume identically
    because ``received()`` is still zero.  Attempts are bounded by
    ``_CONNECT_ATTEMPTS`` with the same linear backoff as before.
    ``revision`` pins resumed suffixes to one strong ETag (``If-Range``).
    """
    import aiohttp

    retryable = (
        aiohttp.ClientConnectionError,
        aiohttp.ClientPayloadError,
        asyncio.IncompleteReadError,
        asyncio.TimeoutError,
        TimeoutError,
    )
    expected = span.end - span.start + 1
    last_error: Exception | None = None
    attempt = 0
    while attempt < _CONNECT_ATTEMPTS:
        attempt += 1
        if attempt > 1:
            if ledger is not None:
                ledger.retry()
            await asyncio.sleep(_CONNECT_BACKOFF_S * (attempt - 1))
        if received() >= expected:
            return
        state = _ACTIVE_RANGE.get()
        if state is not None:
            state.sent = await state.fetcher._acquire(span.url)
            if state.origin is None:
                state.origin = state.sent
        try:
            try:
                await _request_span_once(
                    session,
                    span,
                    write,
                    received(),
                    token=token,
                    revision=revision,
                    ledger=ledger,
                )
            finally:
                if state is not None:
                    state.fetcher._release(span.url)
        except retryable as error:
            if ledger is not None:
                ledger.check_elapsed()
            last_error = error
            logger.warning(
                "range request stream failed on attempt %d/%d "
                "(label=%s, received=%d/%d): %r; resuming",
                attempt,
                _CONNECT_ATTEMPTS,
                span.label,
                received(),
                expected,
                error,
            )
            continue
        if received() != expected:
            logger.error(
                "range response ended before the requested span",
                extra={
                    "label": span.label,
                    "received": received(),
                    "expected": expected,
                },
            )
            _fail(
                RemoteIntegrityError,
                "content_range_mismatch",
                "range response ended before the requested span",
            )
        return
    assert last_error is not None
    raise last_error


async def _request_span_once(
    session: Any,
    span: RangeSpan,
    write: Callable[[bytes], object],
    received: int,
    *,
    token: str | None = None,
    revision: _Revision | None = None,
    ledger: _CallLedger | None = None,
) -> None:
    """GET the missing suffix once; 429 is retryable, a mismatched 206 is not.

    The request covers ``start + received`` through ``end`` so a resumed
    attempt validates the same strict contract: status 206 and an exact
    ``Content-Range: bytes <start+received>-<end>/<complete>`` match.
    Body bytes are streamed to ``write`` in small socket-sized chunks.

    With a pinned ``revision`` a resumed request (``received > 0``) sends
    ``If-Range``; any answer other than the expected 206 then raises
    ``revision_changed`` before the body is read.  A request from the
    span's first byte records the response's strong ``ETag``.
    """
    request_start = span.start + received
    headers = {"Range": f"bytes={request_start}-{span.end}"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if_range = revision.etag if revision is not None and received > 0 else None
    if if_range:
        headers["If-Range"] = if_range
    if ledger is not None:
        ledger.check_elapsed()
        ledger.request()
    async with session.get(
        span.url,
        headers=headers,
        allow_redirects=False,
        proxy=_proxy_for_url(span.url),
    ) as response:
        if response.status in {301, 302, 303, 307, 308}:
            logger.error(
                "range request returned a redirect",
                extra={"label": span.label, "status": response.status},
            )
            _fail(
                RemoteAccessError,
                "redirect",
                "range request followed a redirect",
            )
        if response.status == 401:
            logger.error(
                "range request was unauthorized",
                extra={"label": span.label},
            )
            _fail(
                RemoteAccessError,
                "unauthorized",
                "range request was unauthorized",
            )
        if response.status == 429:
            logger.error(
                "range request was rate-limited",
                extra={"label": span.label},
            )
            delay = _retry_after_seconds(response.headers.get("Retry-After", ""))
            state = _ACTIVE_RANGE.get()
            if state is not None:
                state.fetcher._start_rate_limit(
                    state.origin if state.origin is not None else state.sent,
                    state.fetcher._rate_limit_delay(state.attempt, delay),
                )
                state.cooldown_started = True
                if state.fetcher._max_retries_per_object is not None:
                    counts = state.fetcher._url_429_counts
                    counts[span.url] = counts.get(span.url, 0) + 1
                    state.fetcher._check_object_rate_limit(span.url)
            raise _RateLimitError(delay)
        content_range = response.headers.get("Content-Range", "")
        parsed = _CONTENT_RANGE_RE.match(content_range.strip())
        if (
            response.status != 206
            or parsed is None
            or int(parsed.group(1)) != request_start
            or int(parsed.group(2)) != span.end
        ):
            if if_range:
                logger.error(
                    "range resume was not served from the pinned revision",
                    extra={"label": span.label, "status": response.status},
                )
                _fail(
                    RemoteIntegrityError,
                    "revision_changed",
                    "range resume did not match the recorded object revision",
                )
            logger.error(
                "range response did not match requested span",
                extra={"label": span.label, "status": response.status},
            )
            _fail(
                RemoteIntegrityError,
                "content_range_mismatch",
                "range response did not match the requested span",
            )
        returned_etag = response.headers.get("ETag", "").strip()
        if if_range and returned_etag and returned_etag != if_range:
            logger.error("range resume returned a changed ETag (label=%s)", span.label)
            _fail(
                RemoteIntegrityError,
                "revision_changed",
                "range resume returned a different object revision",
            )
        if received == 0 and revision is not None:
            etag = returned_etag
            if etag and not etag.startswith("W/"):
                revision.etag = etag
                if revision.on_etag is not None:
                    revision.on_etag(etag)
        response_total = 0
        async for chunk in response.content.iter_chunked(_STREAM_CHUNK_BYTES):
            response_total += len(chunk)
            if ledger is not None:
                ledger.response_bytes(len(chunk), response_total=response_total)
            write(bytes(chunk))
        if ledger is not None:
            ledger.check_elapsed()


def _proxy_for_url(url: str) -> str | None:
    """Return the system HTTP proxy for ``url``, or None when bypassed.

    Uses :func:`urllib.request.getproxies` (environment and the macOS
    system proxy) and :func:`urllib.request.proxy_bypass`. Loopback hosts
    always bypass so local TestServers stay direct. ``trust_env`` stays
    closed so ``.netrc`` is never mixed with a Bearer header. HTTPS proxy
    URLs are ignored because aiohttp cannot CONNECT through them.
    """
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname
    if not host or _is_loopback(host) or urllib.request.proxy_bypass(host):
        return None
    proxies = urllib.request.getproxies()
    scheme = (parsed.scheme or "https").casefold()
    candidate = proxies.get(scheme) or proxies.get("https") or proxies.get("http")
    if not candidate:
        return None
    proxy_scheme = urllib.parse.urlsplit(candidate).scheme.casefold()
    if proxy_scheme in {"https", "wss"}:
        return None
    return candidate


def _is_loopback(host: str) -> bool:
    """Return whether ``host`` is a local loopback name."""
    return host.casefold() in {"localhost", "127.0.0.1", "::1", "[::1]"}
