"""Async byte-range spans. RangeFetcher lives here; aiohttp is lazy."""

from __future__ import annotations

import asyncio
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

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


async def _get_span(fetcher: RangeFetcher, span: RangeSpan) -> bytes:
    """Return one 206 body whose Content-Range matches the span."""
    return await _attempt_span(fetcher, span, allow_refresh=True)


async def _attempt_span(
    fetcher: RangeFetcher, span: RangeSpan, *, allow_refresh: bool
) -> bytes:
    """Retry a whole span once after 429; one 401 requeues after a shared refresh."""
    last_access_error: RemoteAccessError | None = None
    for _attempt in range(2):
        generation = fetcher._generation
        try:
            return await _request_span(fetcher._session, span, token=fetcher._token)
        except RemoteAccessError as error:
            last_access_error = error
            if (
                error.reason == "unauthorized"
                and allow_refresh
                and fetcher._refresh is not None
            ):
                await fetcher._refresh_token(generation)
                return await _attempt_span(fetcher, span, allow_refresh=False)
            if error.reason != "range_unavailable":
                raise
            fetcher._set_window(fetcher._window // 2)
    assert last_access_error is not None
    raise last_access_error


async def _request_span(
    session: Any, span: RangeSpan, *, token: str | None = None
) -> bytes:
    """GET one span once; 429 is retryable, a mismatched 206 is not."""
    expected_length = span.end - span.start + 1
    expected_range = f"bytes {span.start}-{span.end}/"
    headers = {"Range": f"bytes={span.start}-{span.end}"}
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
        body = await response.read()
        if (
            response.status != 206
            or not content_range.startswith(expected_range)
            or len(body) != expected_length
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
        return bytes(body)


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
