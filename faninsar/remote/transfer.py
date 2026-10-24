"""Complete-file transfer and atomic publication helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

from faninsar.logging import setup_logger

from .access import (
    _SECRET_KEY,
    _has_signed_query,
    _redirect_headers,
    _registered_origin,
    _response_is_complete,
    _safe_url,
    _strip_redirect_credentials,
    _url_origin,
    _validate_anonymous_delivery_url,
    _validate_representation,
    _validate_url,
)
from .auth import (
    _asf_authenticated_session,
    _asf_request,
    _drain_urllib_response,
    _lpdaac_request,
    _netrc_authorization,
)
from .cache import _identity, _is_qualified, _manifest_path, _matching_manifest
from .catalog import _ADAPTERS, _DESTINATION_LOCKS, _REGISTRY_LOCK
from .errors import (
    BurstPendingError,
    RemoteAccessError,
    RemoteError,
    RemoteIntegrityError,
    RemoteLimitError,
    _fail,
)
from .protocols import _accepts_ledger, _Adapter, _CallLedger
from .records import RemoteAsset, RemoteResourceBudget

logger = setup_logger(__name__)

_POLL_INTERVAL_CAP_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class _RemotePollObservation:
    """Result of one provider preparation observation."""

    state: str
    url: str
    delay_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class _FetchedPayload:
    """One adapter ``fetch`` result that also carries a provider checksum.

    Adapters that discover a provider digest while fetching (for example the
    s1-orbits single-part S3 ``ETag``, PROPOSAL-0060 seam 1) return this
    wrapper instead of bare bytes so the transfer layer can compare the
    streamed bytes against the digest and record ``verified``/``failed``.
    ``checksum`` is the qualified ``algorithm:hex`` spelling accepted by
    :class:`~faninsar.remote.records.RemoteAsset`; ``None`` means the fetch
    found no trustworthy digest and the download stays ``unverifiable``.
    """

    payload: Any
    checksum: str | None = None


def _poll_delay_seconds(
    retry_after: str | None,
    attempt: int,
    max_interval_seconds: float,
) -> float:
    """Return the bounded delay before one provider-poll observation.

    A valid numeric or HTTP-date ``Retry-After`` value takes precedence over
    the local backoff.  Missing and malformed values use 2, 4, 8, then 10
    seconds.  The returned delay is always bounded by the operation budget.
    """
    if retry_after:
        try:
            value = float(retry_after)
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value) and value >= 0:
            return min(value, max_interval_seconds, _POLL_INTERVAL_CAP_SECONDS)
        try:
            retry_date = parsedate_to_datetime(retry_after)
            value = retry_date.timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            value = math.nan
        if math.isfinite(value) and value >= 0:
            return min(value, max_interval_seconds, _POLL_INTERVAL_CAP_SECONDS)
    return min(
        float(2 ** min(max(attempt, 0) + 1, 4)),
        max_interval_seconds,
        _POLL_INTERVAL_CAP_SECONDS,
    )


class _BurstPollToken:
    """Identity and stable priority for one Stack burst transfer."""

    __slots__ = ("index", "submitted_at")

    def __init__(self, index: int, submitted_at: float) -> None:
        self.index = index
        self.submitted_at = submitted_at


class _BurstPollScheduler:
    """Coordinate retries for concurrently requested ASF burst products.

    Initial requests are made by each worker immediately.  Only subsequent
    observations enter this coordinator.  An observation that is due is
    selected by original submission time, with selection index as the stable
    tie-breaker, while workers whose products are already ready continue to
    download without waiting for another burst.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        wait: Callable[[threading.Condition, float], None] | None = None,
    ) -> None:
        """Initialize a scheduler, optionally with deterministic test hooks."""
        self._clock = clock or time.monotonic
        self._wait = wait or (lambda condition, timeout: condition.wait(timeout))
        self._condition = threading.Condition()
        self._priorities: dict[_BurstPollToken, tuple[float, int]] = {}
        self._due: dict[_BurstPollToken, float] = {}
        self._states: dict[_BurstPollToken, str] = {}
        self._observations: dict[_BurstPollToken, int] = {}

    def register(self, index: int) -> _BurstPollToken:
        """Register one transfer before its first provider request."""
        token = _BurstPollToken(index, self._clock())
        with self._condition:
            self._priorities[token] = (token.submitted_at, index)
            self._states[token] = "not_started"
            self._observations[token] = 0
        return token

    def observe(self, token: _BurstPollToken) -> int:
        """Count one completed provider preparation observation."""
        with self._condition:
            self._observations[token] = self._observations.get(token, 0) + 1
            return self._observations[token]

    def mark_waiting(self, token: _BurstPollToken, delay: float) -> None:
        """Put one pending product into the next-due waiting set."""
        with self._condition:
            self._states[token] = "waiting"
            self._due[token] = self._clock() + max(0.0, delay)
            self._condition.notify_all()

    def claim_due(self) -> _BurstPollToken | None:
        """Claim the oldest currently due waiting product."""
        with self._condition:
            now = self._clock()
            candidates = [token for token, due in self._due.items() if due <= now]
            if not candidates:
                return None
            token = min(candidates, key=self._priorities.__getitem__)
            self._due.pop(token, None)
            self._states[token] = "submitted"
            return token

    def next_due(self) -> float | None:
        """Return the next scheduled retry time, if any."""
        with self._condition:
            return min(self._due.values(), default=None)

    def set_state(self, token: _BurstPollToken | None, state: str) -> None:
        """Record one in-memory transfer state for Stack sweep inspection."""
        if token is None:
            return
        with self._condition:
            if token in self._priorities:
                self._states[token] = state
                self._condition.notify_all()

    def snapshot(self) -> tuple[tuple[int, str], ...]:
        """Return states in deterministic selection order."""
        with self._condition:
            return tuple(
                (token.index, self._states[token])
                for token in sorted(self._states, key=self._priorities.__getitem__)
            )

    def details(self) -> tuple[dict[str, object], ...]:
        """Return deterministic scheduler state for diagnostics and tests."""
        with self._condition:
            return tuple(
                {
                    "index": token.index,
                    "submitted_at": token.submitted_at,
                    "state": self._states[token],
                    "next_due": self._due.get(token),
                    "observations": self._observations[token],
                }
                for token in sorted(self._states, key=self._priorities.__getitem__)
            )

    def wait_for_retry(self, token: _BurstPollToken, delay: float) -> None:
        """Wait until this token is the oldest currently due retry."""
        with self._condition:
            self._states[token] = "submitted"
            self._due[token] = self._clock() + max(0.0, delay)
            self._condition.notify_all()
            while True:
                now = self._clock()
                due = [
                    candidate
                    for candidate, scheduled in self._due.items()
                    if scheduled <= now
                ]
                if due and min(due, key=self._priorities.__getitem__) is token:
                    self._due.pop(token, None)
                    self._states[token] = "submitted"
                    self._condition.notify_all()
                    return
                next_due = min(self._due.values(), default=now)
                self._wait(self._condition, max(0.0, next_due - now))

    def complete(self, token: _BurstPollToken | None) -> None:
        """Remove a transfer from the waiting set after it finishes."""
        if token is None:
            return
        with self._condition:
            self._due.pop(token, None)
            if self._states.get(token) != "failed":
                self._states[token] = "complete"
            self._condition.notify_all()


def _facade_hook(name: str, fallback: Any) -> Any:
    """Resolve a legacy monkeypatch hook from the public remote facade."""
    from faninsar import remote

    return getattr(remote, name, fallback)


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow only redirects admitted by a registered adapter policy."""

    def __init__(
        self,
        adapter: _Adapter,
        budget: RemoteResourceBudget,
        ledger: _CallLedger | None = None,
    ) -> None:
        """Initialize a handler with one operation-wide redirect budget."""
        super().__init__()
        self._adapter = adapter
        self._budget = budget
        self._ledger = ledger
        self._redirects = 0
        self._anonymous_delivery = False

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Validate and charge one redirect before following it."""
        self._redirects += 1
        if self._ledger is not None:
            self._ledger.redirect()
            # urllib follows the returned request internally, so account for
            # that target request here before it leaves the public boundary.
            self._ledger.request()
        elif self._redirects > self._budget.max_redirects:
            _fail(RemoteLimitError, "max_redirects")
        _drain_urllib_response(fp, self._ledger, self._budget.max_response_bytes)
        if _has_signed_query(req.full_url):
            _fail(RemoteAccessError, "signed_redirect")
        target = urllib.parse.urljoin(req.full_url, newurl)
        try:
            target = _validate_url(target, self._adapter, redirect=True)
        except RemoteAccessError:
            # Anonymous object delivery is an explicit capability of built-in
            # complete-file adapters.  Keep the generic urllib redirect
            # policy strict for custom adapters and legacy callers.
            if not getattr(self._adapter, "_allow_anonymous_delivery", False):
                raise
            target = _validate_anonymous_delivery_url(
                req.full_url,
                target,
                self._adapter,
                source_is_anonymous=self._anonymous_delivery,
            )
        redirected = super().redirect_request(req, fp, code, msg, headers, target)
        if redirected is None:
            return None
        # A signed query is deliberately preserved on an approved redirect,
        # but ordinary request credentials must not cross origins.  The
        # standard urllib handler copies headers verbatim, including custom
        # Authorization/Cookie headers, so enforce the boundary here.
        try:
            source_origin = _url_origin(req.full_url)
            target_origin = _url_origin(target)
        except ValueError:
            _fail(RemoteAccessError, "invalid_endpoint")
        if source_origin != target_origin:
            # urllib copies custom headers verbatim, so apply the same
            # minimal policy as the ASF requests transport.  Never attach a
            # target netrc credential here: that would forward Basic EDL
            # credentials from URS to another origin.
            headers = _redirect_headers(
                dict(redirected.headers), source_origin, target_origin
            )
            redirected.headers.clear()
            redirected.headers.update(headers)
            for name in list(getattr(redirected, "unredirected_hdrs", {})):
                if _SECRET_KEY.search(str(name)):
                    del redirected.unredirected_hdrs[name]
        if not _registered_origin(target, self._adapter, redirect=True):
            # Once credentials have been removed at the first object-delivery
            # handoff, every later structurally valid HTTPS hop stays
            # anonymous, regardless of how many storage/CDN hosts are used.
            self._anonymous_delivery = True
            _strip_redirect_credentials(redirected)
        return redirected


def _stream_and_verify(
    asset: RemoteAsset,
    adapter: _Adapter,
    budget: RemoteResourceBudget,
    ledger: _CallLedger,
    staging: Path,
    poll_scheduler: _BurstPollScheduler | None = None,
    poll_token: _BurstPollToken | None = None,
    transfer_url_override: str | None = None,
) -> tuple[int, str, dict[str, object]]:
    """Stream one asset into ``staging`` and return size/digest/verification.

    The verification record follows PROPOSAL-0055: ``{algorithm, checksum,
    status, size}`` for assets that carry a provider checksum and
    ``{status, size}`` for assets without one.  A bytes-vs-checksum mismatch
    is returned with ``status: failed`` instead of being raised here so the
    caller can persist the failed attempt in the per-destination manifest
    before raising :class:`RemoteIntegrityError`.
    """
    fetcher = getattr(adapter, "fetch", None)
    supplied_ledger = fetcher is not None and _accepts_ledger(fetcher)
    redirect_handler = _RedirectHandler(adapter, budget, ledger)
    # Anonymous delivery uses no cookie processor at all.  This deliberately
    # ignores Set-Cookie and prevents a response on one object hop from
    # affecting every later hop in the operation.
    opener = urllib.request.build_opener(redirect_handler)
    resolver = getattr(adapter, "_transfer_url", None)
    transfer_url = transfer_url_override or asset.href
    if (
        transfer_url_override is None
        and fetcher is None
        and callable(resolver)
        and asset.auth_profile != "earthdata-asf"
    ):
        try:
            resolver_kwargs: dict[str, Any] = {"budget": budget, "ledger": ledger}
            transfer_url = resolver(asset, **resolver_kwargs)
        except RemoteError:
            raise
        except Exception as error:
            logger.warning("Remote transfer setup failed (%s)", type(error).__name__)
            _fail(RemoteAccessError, "transfer_setup_failed")
    if not isinstance(transfer_url, str):
        _fail(RemoteAccessError, "invalid_endpoint")
    hasher = hashlib.sha256()
    checksum_algorithm = asset.checksum.split(":", 1)[0] if asset.checksum else None
    checksum_hasher = (
        hashlib.new(checksum_algorithm) if checksum_algorithm is not None else None
    )

    def check_output(total: int) -> None:
        """Enforce output, temporary, and cache limits while streaming."""
        if total > budget.max_output_bytes:
            _fail(RemoteLimitError, "max_output_bytes")
        if total > budget.max_temporary_bytes:
            _fail(RemoteLimitError, "max_temporary_bytes")
        if total > budget.max_cache_bytes:
            _fail(RemoteLimitError, "max_cache_bytes")
        ledger.check_elapsed()

    def consume(chunks: Iterable[bytes], *, meter: bool) -> int:
        """Consume chunks directly into the staging file."""
        response_bytes = 0
        with staging.open("ab") as stream:
            for chunk in chunks:
                part = bytes(chunk)
                response_bytes += len(part)
                if meter:
                    ledger.response_bytes(len(part))
                if response_bytes > budget.max_response_bytes:
                    _fail(RemoteLimitError, "max_response_bytes")
                stream.write(part)
                hasher.update(part)
                if checksum_hasher is not None:
                    checksum_hasher.update(part)
                check_output(stream.tell())
            stream.flush()
            os.fsync(stream.fileno())
        return response_bytes

    def result_chunks(result: Any) -> Iterable[bytes]:
        """Adapt provider result forms to a one-pass chunk iterable."""
        if isinstance(result, _FetchedPayload):
            return result_chunks(result.payload)
        if isinstance(result, (bytes, bytearray)):
            # Slice fixture bytes so publication follows the same bounded
            # sequential path as a real HTTP response.
            payload = bytes(result)
            return (
                payload[offset : offset + 1024 * 1024]
                for offset in range(0, len(payload), 1024 * 1024)
            )
        if hasattr(result, "read"):
            return iter(lambda: result.read(1024 * 1024), b"")
        if isinstance(result, Iterable):
            return result
        return ()

    for attempt in range(budget.max_retries + 1):
        if attempt:
            ledger.retry()
        staging.write_bytes(b"")
        hasher = hashlib.sha256()
        if checksum_algorithm is not None:
            checksum_hasher = hashlib.new(checksum_algorithm)
        try:
            if fetcher is None:
                result = None
            elif supplied_ledger:
                result = fetcher(asset, budget, ledger=ledger)
            else:
                ledger.request()
                result = fetcher(asset, budget)
            if isinstance(result, _FetchedPayload) and result.checksum:
                # The adapter discovered a provider digest during fetch
                # (PROPOSAL-0060).  Adopt it for the comparison below; the
                # streaming hasher is (re)built for its algorithm so the
                # already-known asset checksum keeps precedence.
                discovered = result.checksum
                if not asset.checksum:
                    asset = replace(asset, checksum=discovered)
                    checksum_algorithm = discovered.split(":", 1)[0]
                    checksum_hasher = hashlib.new(checksum_algorithm)

            if result is None and asset.auth_profile == "earthdata-asf":
                authenticated_session = _facade_hook(
                    "_asf_authenticated_session", _asf_authenticated_session
                )
                session, token = authenticated_session(
                    asset,
                    adapter,
                    budget,
                    ledger,
                )
                response: requests.Response | None = None
                try:
                    asf_request = _facade_hook("_asf_request", _asf_request)
                    request_url = transfer_url
                    if poll_scheduler is not None:
                        poll_scheduler.set_state(poll_token, "submitted")
                    response = asf_request(
                        session,
                        "GET",
                        request_url,
                        adapter=adapter,
                        asset=asset,
                        budget=budget,
                        ledger=ledger,
                        headers={
                            "Accept-Encoding": "identity",
                            "Authorization": f"Bearer {token}",
                        },
                    )
                    if response.status_code == 202:
                        ledger.begin_response()
                        pending_bytes = 0
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            pending_bytes += len(chunk)
                            ledger.response_bytes(len(chunk))
                            if pending_bytes > budget.max_response_bytes:
                                _fail(RemoteLimitError, "max_response_bytes")
                        location = response.headers.get("Location")
                        retry_after = response.headers.get("Retry-After")
                        response.close()
                        response = None
                        raise BurstPendingError(
                            urllib.parse.urljoin(request_url, location)
                            if location
                            else request_url,
                            _poll_delay_seconds(
                                retry_after,
                                0,
                                budget.max_poll_interval_seconds,
                            ),
                        )
                    if poll_scheduler is not None:
                        poll_scheduler.set_state(poll_token, "ready")
                    _response_is_complete(response)
                    if (
                        response.headers.get("Content-Encoding", "identity")
                        != "identity"
                    ):
                        _fail(RemoteAccessError, "unexpected_content_encoding")
                    if poll_scheduler is not None:
                        poll_scheduler.set_state(poll_token, "downloading")
                    consume(response.iter_content(chunk_size=1024 * 1024), meter=True)
                finally:
                    if response is not None:
                        response.close()
                    session.close()
            elif result is None and asset.auth_profile == "earthdata-lpdaac":
                session = requests.Session()
                response: requests.Response | None = None
                try:
                    lpdaac_request = _facade_hook("_lpdaac_request", _lpdaac_request)
                    response = lpdaac_request(
                        session,
                        "GET",
                        asset.href,
                        adapter=adapter,
                        asset=asset,
                        budget=budget,
                        ledger=ledger,
                    )
                    _response_is_complete(response)
                    if (
                        response.headers.get("Content-Encoding", "identity")
                        != "identity"
                    ):
                        _fail(RemoteAccessError, "unexpected_content_encoding")
                    consume(response.iter_content(chunk_size=1024 * 1024), meter=True)
                finally:
                    if response is not None:
                        response.close()
                    session.close()
            elif result is None:
                ledger.request()
                ledger.begin_response()
                request = urllib.request.Request(
                    transfer_url,
                    headers={
                        "Accept-Encoding": "identity",
                        **(
                            {"Authorization": auth}
                            if asset.auth_profile == "earthdata-lpdaac"
                            and (auth := _netrc_authorization(asset.href))
                            else {}
                        ),
                    },
                )
                with opener.open(
                    request, timeout=budget.read_timeout_seconds
                ) as response:
                    _response_is_complete(response)
                    if (
                        response.headers.get("Content-Encoding", "identity")
                        != "identity"
                    ):
                        _fail(RemoteAccessError, "unexpected_content_encoding")
                    if poll_scheduler is not None:
                        poll_scheduler.set_state(poll_token, "downloading")
                    consume(
                        iter(lambda response=response: response.read(1024 * 1024), b""),
                        meter=True,
                    )
            else:
                if not supplied_ledger:
                    ledger.begin_response()
                if poll_scheduler is not None:
                    poll_scheduler.set_state(poll_token, "downloading")
                consume(result_chunks(result), meter=not supplied_ledger)
            break
        except RemoteError:
            raise
        except Exception as error:
            if attempt >= budget.max_retries:
                logger.warning("Remote transfer failed (%s)", type(error).__name__)
                _fail(RemoteAccessError, "transfer_failed")
    size = staging.stat().st_size
    if asset.size_bytes is not None and size != asset.size_bytes:
        _fail(RemoteIntegrityError, "content_length_mismatch")
    # Provider checksums are normalized upstream at search time
    # (``records._normalize_checksum``); this layer never fetches a checksum
    # sidecar itself.  A failed sidecar fetch (for example an ASF ``.md5``
    # endpoint returning HTTP 503) is therefore an upstream retryable
    # download failure and can never surface here as a verification state.
    verification: dict[str, object] = {"status": "unverifiable", "size": size}
    if asset.checksum:
        algorithm, expected = asset.checksum.split(":", 1)
        actual = checksum_hasher.hexdigest() if checksum_hasher is not None else ""
        verification = {
            "algorithm": algorithm,
            "checksum": expected,
            "status": "verified" if actual == expected else "failed",
            "size": size,
        }
        if actual != expected:
            # A bytes-vs-checksum mismatch is the only verification failure.
            # Return it so the caller records ``status: failed`` in the
            # per-destination manifest before raising the integrity error.
            return size, hasher.hexdigest(), verification
    _validate_representation(staging, asset)
    return size, hasher.hexdigest(), verification


def _stream_download(
    asset: RemoteAsset,
    adapter: _Adapter,
    budget: RemoteResourceBudget,
    ledger: _CallLedger,
    staging: Path,
    poll_scheduler: _BurstPollScheduler | None = None,
    poll_token: _BurstPollToken | None = None,
    transfer_url_override: str | None = None,
) -> tuple[int, str]:
    """Stream one complete asset into ``staging`` and return size/digest."""
    size, digest, verification = _stream_and_verify(
        asset,
        adapter,
        budget,
        ledger,
        staging,
        poll_scheduler,
        poll_token,
        transfer_url_override,
    )
    if verification["status"] == "failed":
        _fail(RemoteIntegrityError, "checksum_mismatch")
    return size, digest


def probe(
    asset: RemoteAsset,
    *,
    budget: RemoteResourceBudget | None = None,
    url: str | None = None,
    attempt: int = 0,
    ledger: _CallLedger | None = None,
) -> _RemotePollObservation:
    """Perform exactly one observation of a generated remote asset.

    The Sentinel-1 burst adapter uses this seam to separate provider
    preparation from file transfer.  A pending observation returns its next
    URL and bounded delay; a ready observation returns the final URL.  No
    sleep or retry loop occurs here.
    """
    budget = budget or RemoteResourceBudget()
    adapter = _ADAPTERS.get(asset.catalog)
    if adapter is None or adapter.provider != asset.provider:
        _fail(RemoteAccessError, "unknown_catalog")
    ledger = ledger or _CallLedger(budget)
    current = url or asset.href
    if asset.auth_profile == "earthdata-asf" and "sentinel1-burst.asf.alaska.edu" in (
        urllib.parse.urlsplit(current).hostname or ""
    ):
        authenticated_session = _facade_hook(
            "_asf_authenticated_session", _asf_authenticated_session
        )
        session, token = authenticated_session(asset, adapter, budget, ledger)
        response: requests.Response | None = None
        try:
            asf_request = _facade_hook("_asf_request", _asf_request)
            response = asf_request(
                session,
                "GET",
                current,
                adapter=adapter,
                asset=asset,
                budget=budget,
                ledger=ledger,
                headers={
                    "Accept-Encoding": "identity",
                    "Authorization": f"Bearer {token}",
                },
            )
            if response.status_code == 202:
                pending_bytes = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    pending_bytes += len(chunk)
                    ledger.response_bytes(len(chunk))
                    if pending_bytes > budget.max_response_bytes:
                        _fail(RemoteLimitError, "max_response_bytes")
                location = response.headers.get("Location")
                next_url = (
                    urllib.parse.urljoin(current, location) if location else current
                )
                next_url = _validate_url(next_url, adapter, redirect=True)
                delay = _poll_delay_seconds(
                    response.headers.get("Retry-After"),
                    attempt,
                    budget.max_poll_interval_seconds,
                )
                return _RemotePollObservation("pending", next_url, delay)
            if response.status_code == 200:
                return _RemotePollObservation(
                    "ready",
                    _validate_url(
                        getattr(response, "url", current), adapter, redirect=True
                    ),
                )
            _fail(RemoteAccessError, "unexpected_status")
        finally:
            if response is not None:
                response.close()
            session.close()
    resolver = getattr(adapter, "_probe", None)
    if callable(resolver):
        return resolver(
            asset,
            budget=budget,
            ledger=ledger,
            url=current,
            attempt=attempt,
        )
    return _RemotePollObservation("ready", current)


def _prepare_burst_for_standalone_download(
    asset: RemoteAsset,
    budget: RemoteResourceBudget,
    ledger: _CallLedger,
) -> str:
    """Drive the single-probe primitive synchronously for legacy callers."""
    current = asset.href
    for attempt in range(budget.max_poll_attempts):
        observation = probe(
            asset,
            budget=budget,
            url=current,
            attempt=attempt,
            ledger=ledger,
        )
        if observation.state == "ready":
            return observation.url
        if attempt >= budget.max_poll_attempts - 1:
            _fail(RemoteLimitError, "max_poll_attempts")
        ledger.poll_retry()
        current = observation.url
        time.sleep(observation.delay_seconds)
    return _fail(RemoteLimitError, "max_poll_attempts")


def parse_safe_manifest_checksums(manifest_xml: str | bytes) -> dict[str, str]:
    """Return the MD5 member checksums declared by one SAFE ``manifest.safe``.

    A SAFE ``manifest.safe`` lists each internal file with a ``fileLocation``
    ``href`` (a relative member path) and, when the provider supplies one, a
    ``checksum checksumName="MD5"`` element.  This pure parser collects only
    the MD5-declared members; members without an MD5 entry (or with a
    different ``checksumName``) are omitted so they never produce a spurious
    failure.  The href is normalized to a bare zip member path (leading
    ``./`` stripped) to match the names inside the SAFE zip.

    Parameters
    ----------
    manifest_xml : str or bytes
        The decoded (or raw UTF-8) text of a SAFE ``manifest.safe`` document.

    Returns
    -------
    dict of str to str
        Mapping of normalized member path to its lowercase hex MD5 digest.
        Empty when the manifest declares no MD5 checksums.

    Examples
    --------
    >>> pairs = parse_safe_manifest_checksums(
    ...     "<byteStream>"
    ...     '<fileLocation href="./a.tiff"/>'
    ...     '<checksum checksumName="MD5">'
    ...     "4ae69c457a6534ddaf0f511c69a37ca6</checksum>"
    ...     "</byteStream>"
    ... )
    >>> pairs
    {'a.tiff': '4ae69c457a6534ddaf0f511c69a37ca6'}

    """
    if isinstance(manifest_xml, bytes):
        text = manifest_xml.decode("utf-8", errors="replace")
    else:
        text = manifest_xml

    def local_name(tag: str) -> str:
        """Return the namespace-free local name of an element tag."""
        return tag.rsplit("}", 1)[-1]

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return {}
    pairs: dict[str, str] = {}
    for byte_stream in root.iter():
        if local_name(byte_stream.tag) != "byteStream":
            continue
        href: str | None = None
        md5: str | None = None
        for child in byte_stream:
            name = local_name(child.tag)
            if name == "fileLocation":
                href = child.get("href") or child.get("HREF")
            elif name == "checksum":
                algorithm = (child.get("checksumName") or "").strip().upper()
                if algorithm == "MD5":
                    value = (child.text or "").strip().lower()
                    if value:
                        md5 = value
        if href and md5:
            member = href.strip().removeprefix("./")
            pairs[member] = md5
    return pairs


def verify_safe_members(zip_path: Path) -> tuple[dict[str, object], bool]:
    """Verify every ``manifest.safe``-checksummed member of a written SAFE zip.

    The zip container itself carries no provider digest and stays
    ``unverifiable``; this is a content check on the SAFE members listed in
    the embedded ``manifest.safe`` (PROPOSAL-0060 seam 4).  Each member that
    the manifest declares with an MD5 is streamed and compared; members
    without a manifest MD5 are not checked.  The returned summary is the
    additive ``members`` fact recorded alongside the unchanged P0055
    three-state verification envelope.

    Parameters
    ----------
    zip_path : Path
        Path to the written SAFE zip.

    Returns
    -------
    summary : dict
        ``{'checked': N, 'matched': M, 'status': 'verified' | 'failed'}`` where
        ``N`` is the number of manifest-MD5 members checked and ``M`` the
        number whose bytes matched.  ``status`` is ``'verified'`` only when
        every checked member matched (and at least one was checked).
    all_matched : bool
        ``True`` when at least one member was checked and every checked member
        matched; ``False`` when a member mismatched or none were checkable.

    Examples
    --------
    For a SAFE zip whose two members match the manifest MD5s, the summary is
    ``{'checked': 2, 'matched': 2, 'status': 'verified'}`` and ``all_matched``
    is ``True``.

    """
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        manifest_name = next(
            (name for name in names if name.rsplit("/", 1)[-1] == "manifest.safe"),
            None,
        )
        if manifest_name is None:
            return {"checked": 0, "matched": 0, "status": "unverifiable"}, False
        pairs = parse_safe_manifest_checksums(archive.read(manifest_name))
        checked = 0
        matched = 0
        for member, expected in pairs.items():
            name = (
                member
                if member in names
                else next(
                    (
                        candidate
                        for candidate in names
                        if candidate.endswith("/" + member) or candidate == member
                    ),
                    None,
                )
            )
            if name is None:
                checked += 1
                continue
            digest = hashlib.md5()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            checked += 1
            if digest.hexdigest().lower() == expected.lower():
                matched += 1
        status = "verified" if checked > 0 and matched == checked else "failed"
        summary = {"checked": checked, "matched": matched, "status": status}
        return summary, bool(checked and matched == checked)


def _apply_safe_member_verification(
    staging: Path, verification: dict[str, object]
) -> tuple[dict[str, object], bool]:
    """Annotate a burst SAFE verification record with its member check.

    After a SAFE zip is written and before publication, the embedded
    ``manifest.safe`` MD5 members are verified (PROPOSAL-0060 seam 4).  The
    container record keeps its P0055 three-state ``status`` (``unverifiable``,
    since a SAFE zip has no container digest) and gains an additive
    ``members`` sub-record only when the manifest declares at least one
    checksummed member.  A member mismatch marks the container ``failed`` and
    is returned as the second element so the caller applies P0055 skip
    semantics (persist ``failed`` and raise).

    Parameters
    ----------
    staging : Path
        The written SAFE zip to check.
    verification : dict
        The P0055 verification record for the container.

    Returns
    -------
    verification : dict
        The record, possibly annotated with ``{'members': {...}}``.
    member_failed : bool
        ``True`` when a checksummed member mismatched its manifest MD5.

    """
    try:
        is_zip = zipfile.is_zipfile(staging)
    except OSError:
        is_zip = False
    if not is_zip:
        return verification, False
    with zipfile.ZipFile(staging) as archive:
        names = archive.namelist()
        has_manifest = any(name.rsplit("/", 1)[-1] == "manifest.safe" for name in names)
    if not has_manifest:
        return verification, False
    summary, all_matched = verify_safe_members(staging)
    if summary.get("checked", 0) == 0:
        # No checksummed members: leave the record untouched (no members
        # sub-record); the container stays unverifiable.
        return verification, False
    verification = dict(verification)
    verification["members"] = summary
    if not all_matched:
        # A member mismatch is a content-verification failure: the record
        # becomes ``failed`` (P0055 skip semantics) even though the container
        # digest itself stays ``unverifiable``.
        verification["status"] = "failed"
    return verification, not all_matched


def _publish_manifest(
    manifest: Path,
    identity: str,
    digest: str,
    size: int,
    verification: Mapping[str, object],
) -> None:
    """Atomically publish one per-destination transfer manifest.

    The manifest keeps the cache identity, the streaming self digest, and the
    downloaded size, and carries the PROPOSAL-0055 verification record
    (``{algorithm, checksum, status, size}`` or ``{status, size}``).
    """
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=manifest.parent,
            prefix=f".{manifest.name}.",
            mode="w",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                {
                    "identity": identity,
                    "sha256": digest,
                    "size": size,
                    "verification": dict(verification),
                },
                stream,
                separators=(",", ":"),
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(manifest)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _extended_manifest_matches(path: Path, asset: RemoteAsset) -> bool:
    """Return whether a verification-extended manifest qualifies for reuse.

    ``cache._matching_manifest`` compares the manifest mapping exactly, so a
    manifest carrying the PROPOSAL-0055 verification record cannot match it.
    This fallback keeps the same identity, self-digest, and size gate while
    ignoring the added record; the strict check is still tried first.
    """
    if not _is_qualified(asset) or not path.is_file():
        return False
    manifest_path = _manifest_path(path, asset)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    if (
        not isinstance(manifest, dict)
        or not manifest_path.is_file()
        or manifest.get("identity") != _identity(asset)
    ):
        return False
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return False
    return manifest.get("sha256") == digest and manifest.get("size") == size


def download(
    asset: RemoteAsset,
    destination: Path,
    *,
    overwrite: bool = False,
    budget: RemoteResourceBudget | None = None,
    poll_scheduler: _BurstPollScheduler | None = None,
    poll_token: _BurstPollToken | None = None,
    transfer_url: str | None = None,
    ledger: _CallLedger | None = None,
) -> Path:
    """Download one complete asset and publish it atomically.

    Existing files are reused only with a qualified checksum or immutable
    version and a matching private manifest.  Each download records a
    PROPOSAL-0055 checksum verification entry (``{algorithm, checksum,
    status, size}``, or ``{status, size}`` when the provider offers no
    checksum) in the private per-destination manifest.  Concurrent publishers
    targeting one destination are serialized by a private process-local guard.
    """
    if not isinstance(asset, RemoteAsset):
        msg = "asset must be a RemoteAsset"
        raise TypeError(msg)
    budget = budget or RemoteResourceBudget()
    adapter = _ADAPTERS.get(asset.catalog)
    if adapter is None or adapter.provider != asset.provider:
        _fail(RemoteAccessError, "unknown_catalog")
    if asset.auth_profile not in adapter.profiles:
        _fail(RemoteAccessError, "unknown_auth_profile")
    _safe_url(asset.href, adapter)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_key = str(destination.resolve())
    with _REGISTRY_LOCK:
        lock = _DESTINATION_LOCKS.setdefault(lock_key, threading.Lock())
    with lock:
        if destination.exists() and not overwrite:
            if _matching_manifest(destination, asset) or _extended_manifest_matches(
                destination, asset
            ):
                if poll_scheduler is not None:
                    poll_scheduler.complete(poll_token)
                return destination
            if _is_qualified(asset):
                _fail(RemoteIntegrityError, "destination_conflict")
        ledger = ledger or _CallLedger(budget)
        if destination.exists() and not overwrite:
            if _matching_manifest(destination, asset) or _extended_manifest_matches(
                destination, asset
            ):
                return destination
            _fail(RemoteIntegrityError, "destination_conflict")
        temporary: Path | None = None
        manifest = _manifest_path(destination, asset)
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
            prepared_url = transfer_url
            if (
                prepared_url is None
                and asset.auth_profile == "earthdata-asf"
                and "sentinel1-burst.asf.alaska.edu"
                in (urllib.parse.urlsplit(asset.href).hostname or "")
            ):
                prepared_url = _prepare_burst_for_standalone_download(
                    asset, budget, ledger
                )
            size, digest, verification = _stream_and_verify(
                asset,
                adapter,
                budget,
                ledger,
                temporary,
                poll_scheduler,
                poll_token,
                prepared_url,
            )
            if verification["status"] == "failed":
                # Persist the failed verification before raising so the
                # failed attempt is recorded and the mission layer can
                # exclude this acquisition from processing.
                _publish_manifest(
                    manifest, _identity(asset), digest, size, verification
                )
                _fail(RemoteIntegrityError, "checksum_mismatch")
            verification, member_failed = _apply_safe_member_verification(
                temporary, verification
            )
            if member_failed:
                # A burst SAFE member mismatched its manifest.safe MD5.  Record
                # status failed (P0055 skip semantics) before raising so the
                # mission layer excludes this acquisition from processing.
                _publish_manifest(
                    manifest, _identity(asset), digest, size, verification
                )
                _fail(RemoteIntegrityError, "checksum_mismatch")
            temporary.replace(destination)
            temporary = None
            _publish_manifest(manifest, _identity(asset), digest, size, verification)
        except BaseException:
            if poll_scheduler is not None:
                poll_scheduler.set_state(poll_token, "failed")
            raise
        finally:
            if poll_scheduler is not None:
                poll_scheduler.complete(poll_token)
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination


def read_verification(path: str | Path) -> dict[str, object] | None:
    """Return the recorded PROPOSAL-0055 verification record for one input.

    The transfer layer writes one qualified manifest per destination
    (``<name>.faninsar.remote.<identity>.json``).  Callers that only hold the
    destination path (e.g. manifest build) read the verification record from
    the single qualified sibling when one exists.
    """
    destination = Path(path)
    try:
        siblings = sorted(
            destination.parent.glob(f"{destination.name}.faninsar.remote.*.json"),
            key=lambda item: item.stat().st_mtime,
        )
    except OSError:
        return None
    for candidate in reversed(siblings):
        try:
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        verification = manifest.get("verification")
        if isinstance(verification, dict):
            return verification
    return None


__all__ = ["_RedirectHandler", "_stream_download", "download", "read_verification"]
