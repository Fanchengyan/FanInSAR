"""Complete-file transfer and atomic publication helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import urllib.request
from collections.abc import Iterable
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
    RemoteAccessError,
    RemoteError,
    RemoteIntegrityError,
    RemoteLimitError,
    _fail,
)
from .protocols import _accepts_ledger, _Adapter, _CallLedger
from .records import RemoteAsset, RemoteResourceBudget

logger = setup_logger(__name__)


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


def _stream_download(
    asset: RemoteAsset,
    adapter: _Adapter,
    budget: RemoteResourceBudget,
    ledger: _CallLedger,
    staging: Path,
) -> tuple[int, str]:
    """Stream one complete asset into ``staging`` and return size/digest."""
    fetcher = getattr(adapter, "fetch", None)
    supplied_ledger = fetcher is not None and _accepts_ledger(fetcher)
    redirect_handler = _RedirectHandler(adapter, budget, ledger)
    # Anonymous delivery uses no cookie processor at all.  This deliberately
    # ignores Set-Cookie and prevents a response on one object hop from
    # affecting every later hop in the operation.
    opener = urllib.request.build_opener(redirect_handler)
    resolver = getattr(adapter, "_transfer_url", None)
    transfer_url = asset.href
    if fetcher is None and callable(resolver):
        try:
            transfer_url = resolver(asset, budget=budget, ledger=ledger)
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
                    response = asf_request(
                        session,
                        "GET",
                        asset.href,
                        adapter=adapter,
                        asset=asset,
                        budget=budget,
                        ledger=ledger,
                        headers={
                            "Accept-Encoding": "identity",
                            "Authorization": f"Bearer {token}",
                        },
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
                    consume(
                        iter(lambda response=response: response.read(1024 * 1024), b""),
                        meter=True,
                    )
            else:
                if not supplied_ledger:
                    ledger.begin_response()
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
    if asset.checksum:
        _, expected = asset.checksum.split(":", 1)
        actual = checksum_hasher.hexdigest() if checksum_hasher is not None else ""
        if actual != expected:
            _fail(RemoteIntegrityError, "checksum_mismatch")
    _validate_representation(staging, asset)
    return size, hasher.hexdigest()


def download(
    asset: RemoteAsset,
    destination: Path,
    *,
    overwrite: bool = False,
    budget: RemoteResourceBudget | None = None,
) -> Path:
    """Download one complete asset and publish it atomically.

    Existing files are reused only with a qualified checksum or immutable
    version and a matching private manifest.  Concurrent publishers targeting
    one destination are serialized by a private process-local guard.
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
            if _matching_manifest(destination, asset):
                return destination
            if _is_qualified(asset):
                _fail(RemoteIntegrityError, "destination_conflict")
        ledger = _CallLedger(budget)
        if destination.exists() and not overwrite:
            if _matching_manifest(destination, asset):
                return destination
            _fail(RemoteIntegrityError, "destination_conflict")
        temporary: Path | None = None
        manifest = _manifest_path(destination, asset)
        manifest_temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
            size, digest = _stream_download(
                asset,
                adapter,
                budget,
                ledger,
                temporary,
            )
            temporary.replace(destination)
            temporary = None
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{manifest.name}.",
                mode="w",
                delete=False,
            ) as stream:
                manifest_temp = Path(stream.name)
                json.dump(
                    {
                        "identity": _identity(asset),
                        "sha256": digest,
                        "size": size,
                    },
                    stream,
                    separators=(",", ":"),
                )
                stream.flush()
                os.fsync(stream.fileno())
            manifest_temp.replace(manifest)
            manifest_temp = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if manifest_temp is not None:
                manifest_temp.unlink(missing_ok=True)
    return destination


__all__ = ["_RedirectHandler", "_stream_download", "download"]
