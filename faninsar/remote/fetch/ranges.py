"""Async byte-range spans. RangeFetcher lives here; aiohttp is lazy."""

from __future__ import annotations

import asyncio
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Self

from faninsar.logging import setup_logger
from faninsar.remote.errors import (
    RemoteAccessError,
    RemoteIntegrityError,
    _fail,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from types import TracebackType

logger = setup_logger(__name__)

_AIMD_SEED = 20
_CONNECT_ATTEMPTS = 10
_CONNECT_BACKOFF_S = 2.0
_STREAM_CHUNK_BYTES = 256 * 1024

# RFC 9110 single-range Content-Range: ``bytes <first>-<last>/<complete|*>``.
# Exact match against the requested span is required; anything looser would
# admit a prefix/suffix window that silently corrupts offset assembly.
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


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
        Caller-owned identity for resume/assembly; the fetcher does not
        interpret it.

    """

    url: str
    start: int
    end: int
    label: str


class RangeFetcher:
    """Reusable async session that concatenates already-sized spans."""

    def __init__(
        self,
        *,
        token: str | None = None,
        refresh: Callable[[], Awaitable[str]] | None = None,
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

        """
        self._session: Any = None
        self._connector: Any = None
        self._token = token
        self._refresh = refresh
        self._refresh_lock = asyncio.Lock()
        self._generation = 0
        self._window = _AIMD_SEED

    @property
    def window(self) -> int:
        """Current AIMD concurrency window, shared with ``TCPConnector.limit``."""
        return self._window

    def _set_window(self, value: int) -> None:
        """Keep the AIMD window and connector limit on the same integer."""
        self._window = max(1, value)
        if self._connector is not None:
            self._connector._limit = self._window

    async def __aenter__(self) -> Self:
        """Open one ClientSession with identity encoding and closed trust_env."""
        import aiohttp

        self._connector = aiohttp.TCPConnector(limit=self._window)
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            auto_decompress=False,
            headers={"Accept-Encoding": "identity"},
            trust_env=False,
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
            logger.error(msg)
            raise RuntimeError(msg)
        span_list = tuple(spans)
        bodies = await asyncio.gather(*(_get_span(self, span) for span in span_list))
        return {span.label: body for span, body in zip(span_list, bodies, strict=True)}

    async def fetch_into(self, span: RangeSpan, sink: str | Path | BinaryIO) -> int:
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
            a resumed response fails the strict 206/Content-Range check.
        RemoteAccessError
            For the same redirect/401/429 semantics as :meth:`fetch`.

        """
        session = self._session
        if session is None:
            msg = "RangeFetcher must be used as an async context manager"
            logger.error(msg)
            raise RuntimeError(msg)
        write, received, close = _open_sink(span, sink)
        try:
            await _attempt_span_stream(self, span, write, received, allow_refresh=True)
        finally:
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
    prefix; a stream uses its own ``tell()``.  The returned ``received``
    reflects live position so a resumed retry continues where the last
    write stopped.
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

        def close() -> None:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()

        return stream.write, stream.tell, close
    write = getattr(sink, "write", None)
    tell = getattr(sink, "tell", None)
    if not callable(write) or not callable(tell):
        msg = "sink must be a path or a binary stream supporting write() and tell()"
        logger.error(msg)
        raise TypeError(msg)
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
) -> None:
    """Drive the streaming span through 429 requeue and shared 401 refresh."""
    last_access_error: RemoteAccessError | None = None
    for _attempt in range(2):
        generation = fetcher._generation
        try:
            await _stream_span(
                fetcher._session, span, write, received, token=fetcher._token
            )
        except RemoteAccessError as error:
            last_access_error = error
            if (
                error.reason == "unauthorized"
                and allow_refresh
                and fetcher._refresh is not None
            ):
                await fetcher._refresh_token(generation)
                await _attempt_span_stream(
                    fetcher,
                    span,
                    write,
                    received,
                    allow_refresh=False,
                )
                return
            if error.reason != "range_unavailable":
                raise
            fetcher._set_window(fetcher._window // 2)
        else:
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
) -> None:
    """Stream one span into ``write``; mid-stream drops resume the suffix.

    Each attempt issues ``Range: bytes=(start+received)-end`` so a
    connection or payload failure that arrives after some bytes were
    already sunk resumes from the live position instead of restarting
    the span.  Whole-request failures resume identically because
    ``received()`` is still zero.  Attempts are bounded by
    ``_CONNECT_ATTEMPTS`` with the same linear backoff as before.
    """
    import aiohttp

    retryable = (
        aiohttp.ClientConnectionError,
        aiohttp.ClientPayloadError,
        asyncio.IncompleteReadError,
    )
    expected = span.end - span.start + 1
    last_error: Exception | None = None
    attempt = 0
    while attempt < _CONNECT_ATTEMPTS:
        attempt += 1
        if attempt > 1:
            await asyncio.sleep(_CONNECT_BACKOFF_S * (attempt - 1))
        if received() >= expected:
            return
        try:
            await _request_span_once(session, span, write, received(), token=token)
        except retryable as error:
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
) -> None:
    """GET the missing suffix once; 429 is retryable, a mismatched 206 is not.

    The request covers ``start + received`` through ``end`` so a resumed
    attempt validates the same strict contract: status 206 and an exact
    ``Content-Range: bytes <start+received>-<end>/<complete>`` match.
    Body bytes are streamed to ``write`` in bounded chunks.
    """
    request_start = span.start + received
    headers = {"Range": f"bytes={request_start}-{span.end}"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
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
            _fail(
                RemoteAccessError,
                "range_unavailable",
                "range request was rate-limited",
            )
        content_range = response.headers.get("Content-Range", "")
        parsed = _CONTENT_RANGE_RE.match(content_range.strip())
        if (
            response.status != 206
            or parsed is None
            or int(parsed.group(1)) != request_start
            or int(parsed.group(2)) != span.end
        ):
            logger.error(
                "range response did not match requested span",
                extra={"label": span.label, "status": response.status},
            )
            _fail(
                RemoteIntegrityError,
                "content_range_mismatch",
                "range response did not match the requested span",
            )
        async for chunk in response.content.iter_chunked(_STREAM_CHUNK_BYTES):
            write(bytes(chunk))


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
