"""Authentication and provider-specific request helpers."""

from __future__ import annotations

import base64
import contextlib
import json
import netrc
import urllib.parse
from typing import TYPE_CHECKING, Any

import requests

from .access import (
    _ASF_AUTH_ORIGIN,
    _ASF_EDL_CLIENT_ID,
    _ASF_EDL_ORIGIN,
    _ASF_REDIRECT_CODES,
    _LPDAAC_DATA_ORIGIN,
    _asf_redirect_url,
    _has_signed_query,
    _lpdaac_redirect_url,
    _redirect_headers,
    _registered_origin,
    _url_origin,
    _validate_url,
)
from .errors import RemoteAccessError, RemoteLimitError, _fail

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .protocols import _Adapter, _CallLedger
    from .records import RemoteAsset, RemoteResourceBudget


def _netrc_authorization(url: str) -> str | None:
    """Return a Basic authorization value from the user's netrc, if present.

    ``urllib`` does not apply ``~/.netrc`` automatically (unlike ``requests``).
    Earthdata providers commonly challenge after a redirect, so the matching
    credential is attached only to the original host and is removed by the
    redirect handler before any cross-origin request.
    """
    try:
        host = urllib.parse.urlsplit(url).hostname
        if not host:
            return None
        entry = netrc.netrc().authenticators(host)
    except (OSError, netrc.NetrcParseError):
        return None
    if entry is None or entry[0] is None or entry[2] is None:
        return None
    token = f"{entry[0]}:{entry[2]}".encode()
    return "Basic " + base64.b64encode(token).decode("ascii")


def _netrc_credentials(host: str) -> tuple[str, str] | None:
    """Resolve one host's username and password without persisting either."""
    try:
        entry = netrc.netrc().authenticators(host)
    except (OSError, netrc.NetrcParseError):
        return None
    if entry is None or entry[0] is None or entry[2] is None:
        return None
    return entry[0], entry[2]


def _facade_netrc_credentials(host: str) -> tuple[str, str] | None:
    """Resolve credentials through the facade compatibility hook."""
    from faninsar import remote

    resolver = getattr(remote, "_netrc_credentials", _netrc_credentials)
    return resolver(host)


def _no_auth(request: requests.PreparedRequest) -> requests.PreparedRequest:
    """Prevent implicit netrc lookup without changing requests' environment."""
    return request


def _asf_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    adapter: _Adapter,
    asset: RemoteAsset,
    budget: RemoteResourceBudget,
    ledger: _CallLedger,
    headers: Mapping[str, str] | None = None,
    data: Mapping[str, str] | None = None,
) -> requests.Response:
    """Issue one metered ASF request while following redirects explicitly."""
    current_method = method.upper()
    current_url = url
    current_headers = dict(headers or {})
    current_data = data
    first_request = True
    anonymous = False
    while True:
        if first_request:
            current_url = _validate_url(current_url, adapter, redirect=True)
            first_request = False
        ledger.request()
        ledger.begin_response()
        response = session.request(
            current_method,
            current_url,
            headers=current_headers,
            data=current_data,
            # A truthy no-op auth hook prevents requests from consulting
            # ``.netrc`` for an otherwise anonymous delivery request.
            auth=_no_auth,
            allow_redirects=False,
            stream=True,
            timeout=(
                budget.connect_timeout_seconds,
                budget.read_timeout_seconds,
            ),
        )
        if response.status_code not in _ASF_REDIRECT_CODES:
            if anonymous:
                with contextlib.suppress(AttributeError):
                    session.cookies.clear()
            return response
        if _has_signed_query(current_url):
            _drain_asf_response(response, ledger, budget.max_response_bytes)
            _fail(RemoteAccessError, "signed_redirect")
        location = response.headers.get("Location")
        if not location:
            _drain_asf_response(response, ledger, budget.max_response_bytes)
            response.close()
            _fail(RemoteAccessError, "invalid_redirect")
        target = urllib.parse.urljoin(current_url, location)
        _drain_asf_response(response, ledger, budget.max_response_bytes)
        response.close()
        ledger.redirect()
        target = _asf_redirect_url(
            current_url,
            target,
            asset,
            adapter,
            anonymous=anonymous,
        )
        try:
            source_origin = _url_origin(current_url)
            target_origin = _url_origin(target)
        except ValueError:
            _fail(RemoteAccessError, "invalid_endpoint")
        if source_origin != target_origin:
            current_headers = _redirect_headers(
                current_headers, source_origin, target_origin
            )
        if not _registered_origin(target, adapter, redirect=True):
            anonymous = True
            current_headers.pop("Authorization", None)
            current_headers.pop("Cookie", None)
            with contextlib.suppress(AttributeError):
                session.cookies.clear()
        if response.status_code == 303 or (
            response.status_code in {301, 302} and current_method == "POST"
        ):
            current_method = "GET"
            current_data = None
        current_url = target


def _lpdaac_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    adapter: _Adapter,
    asset: RemoteAsset,
    budget: RemoteResourceBudget,
    ledger: _CallLedger,
) -> requests.Response:
    """Issue one metered LPDAAC request through its explicit auth redirects.

    Basic credentials are attached solely to the URS hop.  Every other hop,
    including the signed CloudFront transfer, receives no Authorization or
    Cookie header.  Redirect responses are drained before being closed so
    their bodies count against the operation budget.
    """
    credentials = _facade_netrc_credentials("urs.earthdata.nasa.gov")
    if credentials is None:
        _fail(RemoteAccessError, "missing_earthdata_credentials")
    username, password = credentials
    basic = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
    current_method = method.upper()
    current_url = _validate_url(url, adapter, redirect=True)
    current_headers: dict[str, str] = {"Accept-Encoding": "identity"}
    current_data: Mapping[str, str] | None = None
    anonymous = False
    while True:
        current_origin = _url_origin(current_url)
        if current_origin == _ASF_EDL_ORIGIN and not anonymous:
            current_headers["Authorization"] = basic
        else:
            current_headers.pop("Authorization", None)
        ledger.request()
        ledger.begin_response()
        response = session.request(
            current_method,
            current_url,
            headers=current_headers,
            data=current_data,
            # A truthy no-op auth hook suppresses requests' implicit netrc
            # lookup while preserving proxy and CA environment settings.
            auth=_no_auth,
            allow_redirects=False,
            stream=True,
            timeout=(budget.connect_timeout_seconds, budget.read_timeout_seconds),
        )
        if response.status_code not in _ASF_REDIRECT_CODES:
            if anonymous:
                with contextlib.suppress(AttributeError):
                    session.cookies.clear()
            return response
        if _has_signed_query(current_url):
            _drain_asf_response(response, ledger, budget.max_response_bytes)
            _fail(RemoteAccessError, "signed_redirect")
        location = response.headers.get("Location")
        if not location:
            _drain_asf_response(response, ledger, budget.max_response_bytes)
            _fail(RemoteAccessError, "invalid_redirect")
        target = urllib.parse.urljoin(current_url, location)
        _drain_asf_response(response, ledger, budget.max_response_bytes)
        ledger.redirect()
        try:
            target = _validate_url(target, adapter, redirect=True)
        except RemoteAccessError:
            target = _lpdaac_redirect_url(
                current_url,
                target,
                asset,
                adapter,
                anonymous=anonymous,
            )
        source_origin = _url_origin(current_url)
        target_origin = _url_origin(target)
        if source_origin != target_origin:
            current_headers = _redirect_headers(
                current_headers, source_origin, target_origin
            )
        if target_origin == _ASF_EDL_ORIGIN and not anonymous:
            current_headers["Authorization"] = basic
        else:
            current_headers.pop("Authorization", None)
        if target_origin != _ASF_EDL_ORIGIN:
            # Cookie headers are never carried to the data or CDN origin.  A
            # requests cookie jar still supplies only cookies scoped to the
            # target domain, but clearing it at the CDN boundary makes that
            # policy explicit for custom sessions and test doubles alike.
            current_headers.pop("Cookie", None)
            if target_origin != _LPDAAC_DATA_ORIGIN:
                with contextlib.suppress(AttributeError, KeyError):
                    session.cookies.clear()
        if not _registered_origin(target, adapter, redirect=True):
            anonymous = True
            current_headers.pop("Authorization", None)
            current_headers.pop("Cookie", None)
            with contextlib.suppress(AttributeError):
                session.cookies.clear()
        if response.status_code == 303 or (
            response.status_code in {301, 302} and current_method == "POST"
        ):
            current_method = "GET"
            current_data = None
        current_url = target


def _asf_response_body(
    response: requests.Response,
    ledger: _CallLedger,
    *,
    max_bytes: int,
) -> bytes:
    """Read and meter a small ASF authentication response body."""
    content = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        part = bytes(chunk)
        ledger.response_bytes(len(part))
        content.extend(part)
        if len(content) > max_bytes:
            _fail(RemoteLimitError, "max_response_bytes")
    return bytes(content)


def _drain_asf_response(
    response: requests.Response, ledger: _CallLedger, max_bytes: int
) -> None:
    """Drain and meter a bounded redirect response before closing it."""
    iterator = getattr(response, "iter_content", None)
    if not callable(iterator):
        return
    ledger.begin_response()
    total = 0
    try:
        for chunk in iterator(chunk_size=64 * 1024):
            part_size = len(bytes(chunk))
            total += part_size
            if total > max_bytes:
                _fail(RemoteLimitError, "max_response_bytes")
            ledger.response_bytes(part_size)
    finally:
        response.close()


def _drain_urllib_response(
    response: Any, ledger: _CallLedger | None, max_bytes: int
) -> None:
    """Drain and meter one urllib redirect body before it is closed."""
    reader = getattr(response, "read", None)
    if not callable(reader):
        return
    if ledger is not None:
        ledger.begin_response()
    total = 0
    try:
        while True:
            try:
                chunk = reader(64 * 1024)
            except (OSError, ValueError):
                # Some urllib test doubles (and already-closed error paths)
                # expose a closed file object.  There is no body left to
                # account for.
                break
            if not chunk:
                break
            part_size = len(bytes(chunk))
            total += part_size
            if total > max_bytes:
                _fail(RemoteLimitError, "max_response_bytes")
            if ledger is not None:
                ledger.response_bytes(part_size)
    finally:
        closer = getattr(response, "close", None)
        if callable(closer):
            closer()


def _asf_authenticated_session(
    asset: RemoteAsset,
    adapter: _Adapter,
    budget: RemoteResourceBudget,
    ledger: _CallLedger,
) -> tuple[requests.Session, str]:
    """Create one provider-scoped ASF bearer/cookie session from ``.netrc``."""
    credentials = _facade_netrc_credentials("urs.earthdata.nasa.gov")
    if credentials is None:
        _fail(RemoteAccessError, "missing_earthdata_credentials")
    username, password = credentials
    basic = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
    session = requests.Session()
    try:
        token_url = f"{_ASF_EDL_ORIGIN}/api/users/find_or_create_token"
        _validate_url(token_url, adapter, redirect=True)
        token_response = _asf_request(
            session,
            "POST",
            token_url,
            adapter=adapter,
            asset=asset,
            budget=budget,
            ledger=ledger,
            headers={"Authorization": basic, "Accept-Encoding": "identity"},
        )
        try:
            if not 200 <= token_response.status_code < 300:
                _fail(RemoteAccessError, "asf_auth_failed")
            body = _asf_response_body(
                token_response,
                ledger,
                max_bytes=min(1024 * 1024, budget.max_response_bytes),
            )
            try:
                token = json.loads(body).get("access_token")
            except (UnicodeDecodeError, ValueError):
                token = None
            if not isinstance(token, str) or not token:
                _fail(RemoteAccessError, "asf_auth_failed")
        finally:
            token_response.close()

        oauth_query = urllib.parse.urlencode(
            {
                "splash": "false",
                "client_id": _ASF_EDL_CLIENT_ID,
                "response_type": "code",
                "redirect_uri": f"{_ASF_AUTH_ORIGIN}/login",
            }
        )
        oauth_url = f"{_ASF_EDL_ORIGIN}/oauth/authorize?{oauth_query}"
        oauth_response = _asf_request(
            session,
            "GET",
            oauth_url,
            adapter=adapter,
            asset=asset,
            budget=budget,
            ledger=ledger,
            headers={"Authorization": basic, "Accept-Encoding": "identity"},
        )
        try:
            if not 200 <= oauth_response.status_code < 300:
                _fail(RemoteAccessError, "asf_auth_failed")
        finally:
            oauth_response.close()
        if "asf-urs" not in session.cookies:
            _fail(RemoteAccessError, "asf_auth_cookie_missing")
    except Exception:
        session.close()
        raise
    return session, token


__all__ = [name for name in globals() if name.startswith("_")]
