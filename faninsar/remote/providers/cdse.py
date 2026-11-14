"""CDSE Sentinel-1 STAC adapter (PROPOSAL-0061).

Discovery is anonymous STAC against ``sentinel-1-slc``.  Each measurement and
annotation asset is rewritten from ``s3://eodata`` to the HTTPS OData
``Nodes(...)/$value`` URL carried in ``alternate.https``.  Byte-range transfer
is owned by :class:`~faninsar.remote.fetch.RangeFetcher`; ``fetch`` stays
closed.  OAuth2 tokens come from a Keycloak password grant using ``.netrc``
host ``identity.dataspace.copernicus.eu`` and are never logged.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from faninsar.logging import reject, setup_logger
from faninsar.remote.access import _safe_url, _validate_url
from faninsar.remote.auth import _netrc_credentials, _no_auth
from faninsar.remote.catalog import (
    _ADAPTERS,
    _REGISTRY_LOCK,
    _normalize_record,
    _register_adapter,
)
from faninsar.remote.errors import RemoteAccessError, RemoteLimitError, _fail
from faninsar.remote.fetch import RangeFetcher, RangeSpan, fetch_ranges_async
from faninsar.remote.fetch.complete import _RedirectHandler
from faninsar.remote.protocols import _CallLedger
from faninsar.remote.records import CatalogItem, RemoteAsset, RemoteResourceBudget

logger = setup_logger(__name__)

CDSE_CATALOG = "sentinel1-cdse"
CDSE_COLLECTION = "sentinel-1-slc"
CDSE_STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1"
CDSE_STAC_ORIGIN = "https://stac.dataspace.copernicus.eu"
CDSE_CATALOGUE_ORIGIN = "https://catalogue.dataspace.copernicus.eu"
CDSE_ZIPPER_ORIGIN = "https://zipper.dataspace.copernicus.eu"
CDSE_DOWNLOAD_ORIGIN = "https://download.dataspace.copernicus.eu"
CDSE_IDENTITY_ORIGIN = "https://identity.dataspace.copernicus.eu"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_TOKEN_HOST = "identity.dataspace.copernicus.eu"
CDSE_CLIENT_ID = "cdse-public"
_MEASUREMENT_KEY = "measurement"
_ANNOTATION_KEY = "annotation"
_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)
# Bound one page's outage even when a campaign ledger allows many retries.
_MAX_RETRIES_PER_PAGE = 3
# CDSE OData product names and their child node names.  The provider validates
# both independently of the mission so a response cannot smuggle another
# product, swath, or polarization into the exact-parent fallback.
_S1_SAFE_NAME_RE = re.compile(
    r"^(?P<mission>S1[ABCD])_(?P<beam>IW|EW)_SLC__1S(?P<pols>[A-Z]{2})_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_"
    r"(?P<orbit>\d{6})_(?P<datatake>[0-9A-F]{6})_(?P<uid>[0-9A-F]{4})"
    r"\.SAFE$"
)
_S1_NODE_NAME_RE = re.compile(
    r"^(?P<mission>s1[abcd])-(?P<swath>iw[1-3]|ew[1-5])-slc-"
    r"(?P<pol>vv|vh|hh|hv)-"
    r"(?P<start>\d{8}t\d{6})-(?P<stop>\d{8}t\d{6})-"
    r"(?P<orbit>\d{6})-(?P<datatake>[0-9a-f]{6})-(?P<seq>\d{3,4})"
    r"\.(?P<ext>tiff|tif|xml)$",
    re.IGNORECASE,
)
_SWATH_POL = (
    "iw1-vv",
    "iw2-vv",
    "iw3-vv",
    "iw1-vh",
    "iw2-vh",
    "iw3-vh",
    "iw1-hh",
    "iw2-hh",
    "iw3-hh",
    "iw1-hv",
    "iw2-hv",
    "iw3-hv",
)
_token_cache: dict[str, Any] = {"token": None, "expires_at": 0.0}
_token_lock = threading.Lock()


def _origin(url: str) -> str:
    """Return a normalized HTTPS origin."""
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    port = parts.port
    return f"https://{host}" + (f":{port}" if port not in (None, 443) else "")


def _is_odata_value(href: str) -> bool:
    """Return True when ``href`` is an HTTPS OData ``Nodes(...)/$value`` URL."""
    parsed = urllib.parse.urlsplit(href)
    if parsed.scheme.casefold() != "https" or not parsed.netloc:
        return False
    host = (parsed.hostname or "").casefold()
    path = parsed.path
    if "eodata" in host or "eodata" in path.casefold():
        return False
    return "/Nodes(" in path and path.rstrip("/").endswith("/$value")


def _odata_href(asset: Mapping[str, Any]) -> str | None:
    """Prefer ``alternate.https`` OData ``$value`` over a primary s3 href."""
    href = str(asset.get("href") or "")
    if _is_odata_value(href):
        return href
    alternate = asset.get("alternate")
    if not isinstance(alternate, Mapping):
        return None
    https = alternate.get("https")
    if not isinstance(https, Mapping):
        return None
    candidate = str(https.get("href") or "")
    return candidate if _is_odata_value(candidate) else None


def _swath_from_key(key: str) -> str:
    """Return ``IW1`` from an asset key such as ``iw1-vv``."""
    prefix = key.casefold().split("-", 1)[0]
    return prefix.upper() if prefix.startswith("iw") else "IW1"


def _annotation_key(measurement_key: str) -> str:
    """Return the CDSE annotation asset key paired with one measurement."""
    return f"schema-product-{measurement_key.casefold()}"


def _bounds(spatial: Any) -> tuple[float, float, float, float]:
    """Return a WGS84 bbox from a Shapely geometry or FanInSAR query."""
    if spatial is None:
        return (-180.0, -90.0, 180.0, 90.0)
    if hasattr(spatial, "bounds") and not hasattr(spatial, "crs"):
        return tuple(spatial.bounds)  # type: ignore[return-value]
    from faninsar.remote.catalog import _query_geometry

    geometry, _kind, _points = _query_geometry(spatial)
    return tuple(geometry.bounds)  # type: ignore[return-value]


def _datetime_interval(
    datetime_range: tuple[datetime, datetime] | None,
) -> str | None:
    """Format a STAC datetime interval, or None when unbounded."""
    if datetime_range is None:
        return None
    start, end = datetime_range
    return f"{start.astimezone(UTC).isoformat()}/{end.astimezone(UTC).isoformat()}"


def _filter_orbit(filters: Mapping[str, Any] | None) -> int | None:
    """Read ``relativeOrbit`` / ``sat:relative_orbit`` from search filters."""
    if not filters:
        return None
    value = filters.get("relativeOrbit", filters.get("sat:relative_orbit"))
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _retry_after_seconds(value: str | None) -> float | None:
    """Return a server-requested delay, or None when it is unusable."""
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = math.nan
    if math.isfinite(seconds) and seconds >= 0:
        return seconds
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(retry_at.timestamp() - time.time(), 0.0)


def _retry_delay_seconds(retry_after: str | None, attempt: int) -> float:
    """Prefer a server ``Retry-After``; otherwise use bounded backoff."""
    requested = _retry_after_seconds(retry_after)
    if requested is not None:
        return requested
    return _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]


def _is_retryable_transport_error(error: BaseException) -> bool:
    """Return True for a transient timeout or connection failure."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in _RETRYABLE_HTTP_STATUS
    if isinstance(error, urllib.error.URLError):
        return isinstance(error.reason, (TimeoutError, ConnectionError))
    return isinstance(error, (TimeoutError, ConnectionError))


def _wait_before_retry(
    ledger: _CallLedger, retry_after: str | None, attempt: int
) -> None:
    """Charge one retry and wait without undercutting ``Retry-After``."""
    delay = _retry_delay_seconds(retry_after, attempt)
    remaining = ledger.budget.max_elapsed_seconds - (time.monotonic() - ledger.started)
    if delay > remaining:
        _fail(RemoteLimitError, "max_elapsed_seconds")
    ledger.retry()
    time.sleep(delay)


def request_cdse_token(*, force: bool = False) -> str:
    """Return a CDSE Keycloak access token from ``.netrc``.

    The password grant is posted to the registered identity origin.  Basic
    credentials are never attached as an ``Authorization`` header; they travel
    only in the form body.  Tokens are cached in memory for this process and
    are never written to the logger.

    Parameters
    ----------
    force : bool, optional
        Ignore the in-memory cache and request a new token.

    Returns
    -------
    str
        Access token suitable for ``Authorization: Bearer``.

    Raises
    ------
    RemoteAccessError
        If ``.netrc`` has no CDSE entry or the token endpoint rejects the grant.

    """
    now = time.monotonic()
    with _token_lock:
        cached = _token_cache.get("token")
        expires_at = float(_token_cache.get("expires_at") or 0.0)
        if not force and isinstance(cached, str) and cached and now < expires_at:
            return cached
    credentials = _netrc_credentials(CDSE_TOKEN_HOST)
    if credentials is None:
        message = (
            "CDSE OAuth2 requires netrc authenticators for "
            f"{CDSE_TOKEN_HOST}; cannot obtain a range-fetch Bearer token"
        )
        _fail(RemoteAccessError, "missing_cdse_credentials", message)
    username, password = credentials
    ensure_cdse_catalog()
    with _REGISTRY_LOCK:
        adapter = _ADAPTERS.get(CDSE_CATALOG)
    if adapter is None:
        message = f"CDSE catalog {CDSE_CATALOG!r} is not registered"
        _fail(RemoteAccessError, "unknown_catalog", message)
    _validate_url(CDSE_TOKEN_URL, adapter)
    try:
        response = requests.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "password",
                "client_id": CDSE_CLIENT_ID,
                "username": username,
                "password": password,
            },
            auth=_no_auth,
            headers={"Accept-Encoding": "identity"},
            timeout=30,
        )
    except requests.RequestException as error:
        logger.exception("CDSE token request failed")
        _fail(RemoteAccessError, "request_failed", str(error))
    if response.status_code != 200:
        logger.error("CDSE token request returned HTTP %s", response.status_code)
        reason = (
            "unauthorized" if response.status_code in {401, 403} else "request_failed"
        )
        _fail(RemoteAccessError, reason, "CDSE token request failed")
    try:
        payload = response.json()
    except ValueError:
        logger.exception("CDSE token response was not JSON")
        _fail(RemoteAccessError, "request_failed", "CDSE token response was not JSON")
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        _fail(
            RemoteAccessError,
            "request_failed",
            "CDSE token response omitted access_token",
        )
    expires_in = payload.get("expires_in")
    lifetime = float(expires_in) if isinstance(expires_in, (int, float)) else 1800.0
    with _token_lock:
        _token_cache["token"] = token
        _token_cache["expires_at"] = time.monotonic() + max(lifetime - 60.0, 1.0)
    return token


async def refresh_cdse_token() -> str:
    """Force-refresh the in-memory CDSE token (RangeFetcher 401 single-flight).

    Returns
    -------
    str
        Replacement access token.  The previous cache entry is discarded.

    """
    return await asyncio.to_thread(request_cdse_token, force=True)


def _needs_cdse_bearer(url: str) -> bool:
    """Return True when ``url`` is a zipper/download OData origin."""
    host = (urllib.parse.urlsplit(url).hostname or "").casefold()
    return host in {
        urllib.parse.urlsplit(CDSE_ZIPPER_ORIGIN).hostname,
        urllib.parse.urlsplit(CDSE_DOWNLOAD_ORIGIN).hostname,
    }


@asynccontextmanager
async def cdse_range_fetcher(
    spans: Sequence[RangeSpan],
    *,
    budget: RemoteResourceBudget | None = None,
    ledger: _CallLedger | None = None,
) -> AsyncIterator[RangeFetcher]:
    """Validate OData range URLs and yield a credential RangeFetcher.

    Ensures the CDSE catalog adapter, validates every ``span.url`` with
    ``_validate_url`` + ``_is_odata_value`` + ``_needs_cdse_bearer`` (same rules
    as ``fetch_cdse_ranges``), obtains the CDSE access token, and yields an
    open ``RangeFetcher`` with 401-refresh wired to ``refresh_cdse_token``.
    The token never leaves this module's scope — callers only see the fetcher.
    Empty ``spans`` still validates nothing and yields a working fetcher.

    Parameters
    ----------
    spans : sequence of RangeSpan
        Already-sized inclusive byte intervals on CDSE OData ``$value`` URLs.
    budget : RemoteResourceBudget, optional
        Network guards forwarded to the shared range fetcher, not item or
        disk quotas. Omitting it retains the existing transfer defaults.
    ledger : _CallLedger, optional
        Existing operation meter; its budget must match an explicit budget.

    Yields
    ------
    RangeFetcher
        An open ``RangeFetcher`` whose ``Authorization: Bearer`` header is
        managed inside this module; callers drive ``fetch``/``fetch_into``
        without seeing the token.

    Raises
    ------
    RemoteAccessError
        If a span URL is not a registered OData origin, or the token grant
        fails.

    """
    span_tuple = tuple(spans)
    ensure_cdse_catalog()
    with _REGISTRY_LOCK:
        adapter = _ADAPTERS.get(CDSE_CATALOG)
    if adapter is None:
        message = f"CDSE catalog {CDSE_CATALOG!r} is not registered"
        _fail(RemoteAccessError, "unknown_catalog", message)
    for span in span_tuple:
        _validate_url(span.url, adapter)
        if not _is_odata_value(span.url) or not _needs_cdse_bearer(span.url):
            message = "CDSE range URL is not an OData Nodes(...)/$value href"
            _fail(RemoteAccessError, "invalid_endpoint", message)
    token = request_cdse_token()
    async with RangeFetcher(
        token=token, refresh=refresh_cdse_token, budget=budget, ledger=ledger
    ) as fetcher:
        yield fetcher


async def fetch_cdse_ranges(
    spans: Sequence[RangeSpan],
    *,
    budget: RemoteResourceBudget | None = None,
    ledger: _CallLedger | None = None,
) -> dict[str, bytes]:
    """Fetch OData byte ranges after validating each URL, then attach Bearer.

    Parameters
    ----------
    spans : sequence of RangeSpan
        Already-sized inclusive byte intervals on CDSE OData ``$value`` URLs.
    budget : RemoteResourceBudget, optional
        Network guards for the range transfer. Omission retains defaults.
    ledger : _CallLedger, optional
        Existing operation meter; its budget must match an explicit budget.

    Returns
    -------
    dict of str to bytes
        Body for each span, keyed by ``RangeSpan.label``.

    Raises
    ------
    RemoteAccessError
        If a span URL is not a registered OData origin, or the token grant fails.

    """
    span_tuple = tuple(spans)
    if not span_tuple:
        return {}
    async with cdse_range_fetcher(span_tuple, budget=budget, ledger=ledger) as fetcher:
        return await fetch_ranges_async(span_tuple, fetcher=fetcher)


async def validate_cdse_range_urls(urls: Sequence[str]) -> None:
    """Validate OData range URLs for a shared provider-owned fetcher.

    Applies exactly the same rules as :func:`fetch_cdse_ranges` applies at
    fetch time — catalog registration, ``_validate_url`` allow-listing, and
    the OData ``Nodes(...)/$value`` + Bearer-origin checks — so an
    operation-scoped fetcher opened once can be reused across batches while
    every new URL is still provider-validated before its first credentialed
    request.  This is a private seam for
    ``faninsar.missions.s1.acquisition``; per-call transfers should keep
    using :func:`fetch_cdse_ranges`.

    Parameters
    ----------
    urls : sequence of str
        Candidate range URLs to validate.  Duplicates collapse silently.

    Raises
    ------
    RemoteAccessError
        If any URL is not a registered OData origin.

    """
    url_tuple = tuple(dict.fromkeys(str(url) for url in urls))
    if not url_tuple:
        return
    ensure_cdse_catalog()
    with _REGISTRY_LOCK:
        adapter = _ADAPTERS.get(CDSE_CATALOG)
    if adapter is None:
        message = f"CDSE catalog {CDSE_CATALOG!r} is not registered"
        _fail(RemoteAccessError, "unknown_catalog", message)
    for url in url_tuple:
        _validate_url(url, adapter)
        if not _is_odata_value(url) or not _needs_cdse_bearer(url):
            message = "CDSE range URL is not an OData Nodes(...)/$value href"
            _fail(RemoteAccessError, "invalid_endpoint", message)


def _normalize_safe_name(parent_id: Any) -> tuple[str, re.Match[str]]:
    """Return the exact CDSE SAFE name and its grammar match for one parent id."""
    text = str(parent_id or "").strip()
    if text.upper().endswith(".SAFE"):
        text = text[:-5]
    text = text.upper()
    name = f"{text}.SAFE"
    match = _S1_SAFE_NAME_RE.fullmatch(name)
    if match is None:
        message = (
            f"CDSE exact-parent lookup rejected {parent_id!r}: expected an "
            "ESA Sentinel-1 IW/EW SLC SAFE product id"
        )
        _fail(RemoteAccessError, "invalid_product", message)
    return name, match


def _canonical_product_uuid(value: Any) -> str:
    """Return a canonical UUID, rejecting opaque or route-changing product ids."""
    text = str(value or "")
    try:
        parsed = uuid.UUID(text)
    except (AttributeError, TypeError, ValueError):
        _fail(
            RemoteAccessError,
            "invalid_product",
            "CDSE OData product Id is not a UUID",
        )
    canonical = str(parsed)
    if canonical != text.casefold():
        _fail(
            RemoteAccessError,
            "invalid_product",
            "CDSE OData product Id is not a canonical UUID",
        )
    return canonical


def _product_attributes(product: Mapping[str, Any]) -> dict[str, Any]:
    """Return expanded OData product attributes keyed by attribute name."""
    raw = product.get("Attributes")
    if not isinstance(raw, list):
        _fail(
            RemoteAccessError,
            "invalid_product",
            "CDSE OData product omitted expanded Attributes",
        )
    attributes: dict[str, Any] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            _fail(
                RemoteAccessError,
                "invalid_product",
                "CDSE OData product has a malformed attribute",
            )
        name = entry.get("Name")
        if not isinstance(name, str) or not name:
            _fail(
                RemoteAccessError,
                "invalid_product",
                "CDSE OData product attribute has no name",
            )
        attributes[name] = entry.get("Value")
    return attributes


def _required_int_attribute(attributes: Mapping[str, Any], name: str) -> int:
    """Return one integer product attribute or fail closed."""
    value = attributes.get(name)
    if type(value) is not int or value < 0:
        message = f"CDSE OData product attribute {name!r} is not a non-negative integer"
        _fail(RemoteAccessError, "invalid_product", message)
    return value


def _required_text_attribute(attributes: Mapping[str, Any], name: str) -> str:
    """Return one non-empty product attribute or fail closed."""
    value = str(attributes.get(name) or "").strip()
    if not value:
        message = f"CDSE OData product attribute {name!r} is missing"
        _fail(RemoteAccessError, "invalid_product", message)
    return value


def _odata_datetime(value: Any) -> datetime:
    """Parse one OData timestamp as timezone-aware UTC."""
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            _fail(
                RemoteAccessError,
                "invalid_product",
                "CDSE OData product has an invalid ContentDate",
            )
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def _product_window(
    product: Mapping[str, Any], name_match: re.Match[str]
) -> tuple[datetime, datetime]:
    """Return the product sensing interval, falling back to the SAFE name."""
    content = product.get("ContentDate")
    if isinstance(content, Mapping) and content.get("Start") and content.get("End"):
        start = _odata_datetime(content["Start"])
        end = _odata_datetime(content["End"])
    else:
        fmt = "%Y%m%dT%H%M%S"
        start = datetime.strptime(name_match.group("start"), fmt).replace(tzinfo=UTC)
        end = datetime.strptime(name_match.group("stop"), fmt).replace(tzinfo=UTC)
    if end < start:
        _fail(
            RemoteAccessError,
            "invalid_product",
            "CDSE OData product ContentDate ends before it starts",
        )
    return start, end


def _odata_products_url(name: str) -> str:
    """Return the exact-name catalogue Products query URL."""
    escaped = urllib.parse.quote(f"Name eq '{name}'", safe="")
    return (
        f"{CDSE_CATALOGUE_ORIGIN}/odata/v1/Products"
        f"?$filter={escaped}&$top=2&$expand=Attributes"
    )


def _odata_nodes_url(product_uuid: str, name: str, role: str) -> str:
    """Return the node directory listing URL for annotation or measurement."""
    return (
        f"{CDSE_DOWNLOAD_ORIGIN}/odata/v1/Products({product_uuid})"
        f"/Nodes({name})/Nodes({role})/Nodes"
    )


def _odata_node_value_url(
    product_uuid: str, name: str, role: str, node_name: str
) -> str:
    """Return the authenticated OData ``$value`` URL for one metadata node."""
    return (
        f"{CDSE_DOWNLOAD_ORIGIN}/odata/v1/Products({product_uuid})"
        f"/Nodes({name})/Nodes({role})/Nodes({node_name})/$value"
    )


def _request_odata_json(
    url: str,
    adapter: CDSESearchAdapter,
    ledger: _CallLedger,
) -> Mapping[str, Any]:
    """GET one anonymous OData document within the shared operation meter.

    The URL is validated against the registered CDSE origins before every
    attempt.  Retries, response bytes, and elapsed time are charged to the
    caller's ledger, which is the same operation meter used by preparation.
    """
    _safe_url(url, adapter)
    opener = urllib.request.build_opener(
        _RedirectHandler(adapter, ledger.budget, ledger)
    )
    max_retries = min(ledger.budget.max_retries, _MAX_RETRIES_PER_PAGE)
    attempt = 0
    while True:
        ledger.check_elapsed()
        ledger.request()
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with opener.open(
                request, timeout=ledger.budget.read_timeout_seconds
            ) as response:
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    _fail(RemoteAccessError, "unexpected_content_encoding")
                ledger.begin_response()
                chunks: list[bytes] = []
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    ledger.response_bytes(len(chunk))
                    chunks.append(chunk)
                document = json.loads(b"".join(chunks))
            break
        except urllib.error.HTTPError as error:
            retry_after = error.headers.get("Retry-After")
            error.close()
            if not _is_retryable_transport_error(error):
                logger.exception("CDSE OData request failed")
                _fail(RemoteAccessError, "request_failed", str(error))
            if attempt >= max_retries:
                _fail(
                    RemoteLimitError,
                    "max_retries",
                    "CDSE OData lookup exhausted retries",
                )
            logger.warning(
                "CDSE OData request returned HTTP %s; retrying (%d/%d)",
                error.code,
                attempt + 1,
                max_retries,
            )
            _wait_before_retry(ledger, retry_after, attempt)
            attempt += 1
        except json.JSONDecodeError as error:
            logger.exception("CDSE OData response was not JSON")
            _fail(RemoteAccessError, "malformed_results", str(error))
        except (urllib.error.URLError, OSError) as error:
            if not _is_retryable_transport_error(error):
                logger.exception("CDSE OData request failed")
                _fail(RemoteAccessError, "request_failed", str(error))
            if attempt >= max_retries:
                _fail(
                    RemoteLimitError,
                    "max_retries",
                    "CDSE OData lookup exhausted retries",
                )
            logger.warning(
                "CDSE OData transport error (%s); retrying (%d/%d)",
                type(error).__name__,
                attempt + 1,
                max_retries,
            )
            _wait_before_retry(ledger, None, attempt)
            attempt += 1
    if not isinstance(document, Mapping):
        _fail(
            RemoteAccessError,
            "malformed_results",
            "CDSE OData response is not a JSON object",
        )
    return document


def _node_leaves(document: Mapping[str, Any], *, role: str) -> list[tuple[str, int]]:
    """Return positive-size leaf nodes of one annotation/measurement directory."""
    entries = document.get("result", document.get("value"))
    if not isinstance(entries, list):
        _fail(
            RemoteAccessError,
            "malformed_results",
            f"CDSE OData {role} node list is missing",
        )
    expected_ext = {"xml"} if role == "annotation" else {"tif", "tiff"}
    leaves: list[tuple[str, int]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            _fail(
                RemoteAccessError,
                "malformed_results",
                f"CDSE OData {role} node is malformed",
            )
        name = entry.get("Name")
        if not isinstance(name, str) or not name:
            _fail(
                RemoteAccessError,
                "invalid_product",
                f"CDSE OData {role} node has no name",
            )
        children = entry.get("ChildrenNumber")
        if isinstance(children, int) and children > 0:
            continue
        extension = name.rsplit(".", 1)[-1].casefold()
        if extension not in expected_ext:
            continue
        size = entry.get("ContentLength")
        if type(size) is not int or size <= 0:
            message = f"CDSE OData {role} node {name!r} has no positive ContentLength"
            _fail(RemoteAccessError, "invalid_product", message)
        leaves.append((name, size))
    return leaves


def _node_pairs(
    annotation_document: Mapping[str, Any],
    measurement_document: Mapping[str, Any],
    product_match: re.Match[str],
    *,
    channels: frozenset[str],
    swaths: frozenset[str],
) -> list[tuple[str, int, str, int, str, str]]:
    """Pair exact annotation/measurement leaves for every swath/polarization."""
    annotations: dict[str, tuple[str, int, re.Match[str]]] = {}
    measurements: dict[str, tuple[str, int, re.Match[str]]] = {}
    for role, document, target in (
        ("annotation", annotation_document, annotations),
        ("measurement", measurement_document, measurements),
    ):
        for name, size in _node_leaves(document, role=role):
            match = _S1_NODE_NAME_RE.fullmatch(name)
            if match is None:
                message = (
                    f"CDSE OData {role} node {name!r} is not a Sentinel-1 "
                    "SLC measurement/annotation name"
                )
                _fail(RemoteAccessError, "invalid_product", message)
            stem = name.rsplit(".", 1)[0]
            if stem in target:
                message = f"CDSE OData {role} node {name!r} is duplicated"
                _fail(RemoteAccessError, "ambiguous_product", message)
            target[stem] = (name, size, match)
    if not annotations or set(annotations) != set(measurements):
        _fail(
            RemoteAccessError,
            "invalid_product",
            "CDSE OData SAFE annotation/measurement nodes do not pair exactly",
        )
    pairs: list[tuple[str, int, str, int, str, str]] = []
    for stem in sorted(annotations):
        annotation_name, annotation_size, annotation_match = annotations[stem]
        measurement_name, measurement_size, measurement_match = measurements[stem]
        if annotation_match.group("swath").casefold() != (
            measurement_match.group("swath").casefold()
        ) or annotation_match.group("pol").casefold() != (
            measurement_match.group("pol").casefold()
        ):
            _fail(
                RemoteAccessError,
                "invalid_product",
                f"CDSE OData node stem {stem!r} disagrees about swath or polarization",
            )
        if annotation_match.group("seq") != measurement_match.group("seq"):
            _fail(
                RemoteAccessError,
                "invalid_product",
                f"CDSE OData node stem {stem!r} disagrees about sequence",
            )
        for field_name in ("mission", "orbit", "datatake"):
            expected = product_match.group(field_name).casefold()
            for actual in (
                annotation_match.group(field_name),
                measurement_match.group(field_name),
            ):
                if actual.casefold() != expected:
                    _fail(
                        RemoteAccessError,
                        "invalid_product",
                        f"CDSE OData node {stem!r} disagrees about {field_name}",
                    )
        swath = annotation_match.group("swath").upper()
        polarization = annotation_match.group("pol").upper()
        if swaths and swath not in swaths:
            _fail(
                RemoteAccessError,
                "invalid_product",
                f"CDSE OData node {stem!r} is not in the product swath list",
            )
        if channels and polarization not in channels:
            _fail(
                RemoteAccessError,
                "invalid_product",
                f"CDSE OData node {stem!r} is not in the product polarization list",
            )
        pairs.append(
            (
                annotation_name,
                annotation_size,
                measurement_name,
                measurement_size,
                swath,
                polarization,
            )
        )
    pairs.sort(key=_node_pair_sort_key)
    return pairs


def _node_pair_sort_key(pair: tuple[str, int, str, int, str, str]) -> tuple[int, str]:
    """Order fallback bindings by the registered swath/polarization order."""
    suffix = f"{pair[4].casefold()}-{pair[5].casefold()}"
    try:
        return (_SWATH_POL.index(suffix), suffix)
    except ValueError:
        return (len(_SWATH_POL), suffix)


def lookup_cdse_safe_record(
    parent_id: str,
    *,
    ledger: _CallLedger,
) -> CatalogItem | None:
    """Resolve one exact ASF parent SAFE to its CDSE OData metadata nodes.

    This is the missing-only bridge used when the CDSE STAC index omits a
    parent that the catalogue OData Products endpoint still knows.  The
    lookup is anonymous metadata only: it returns a normal ``CatalogItem``
    whose measurement/annotation assets are authenticated OData ``$value``
    URLs, never a complete SAFE payload.  Every request is charged to the
    caller's shared operation ledger.

    Parameters
    ----------
    parent_id : str
        Exact ESA Sentinel-1 SAFE product id (with or without ``.SAFE``).
    ledger : _CallLedger
        Shared preparation/operation meter.  The budget is never reset here.

    Returns
    -------
    CatalogItem or None
        The exact parent item, or ``None`` when no product has that name.

    Raises
    ------
    RemoteAccessError
        If the product name, UUID, collection, attributes, or child nodes are
        missing, ambiguous, malformed, or on an unregistered origin.

    """
    ensure_cdse_catalog()
    with _REGISTRY_LOCK:
        adapter = _ADAPTERS.get(CDSE_CATALOG)
    if not isinstance(adapter, CDSESearchAdapter):
        message = f"CDSE catalog {CDSE_CATALOG!r} is not registered"
        _fail(RemoteAccessError, "unknown_catalog", message)
    name, name_match = _normalize_safe_name(parent_id)
    document = _request_odata_json(_odata_products_url(name), adapter, ledger)
    values = document.get("value")
    if not isinstance(values, list):
        _fail(
            RemoteAccessError,
            "malformed_results",
            "CDSE OData Products response is missing value",
        )
    if not values:
        return None
    if len(values) != 1:
        message = f"CDSE OData returned {len(values)} products for exact name {name!r}"
        _fail(RemoteAccessError, "ambiguous_product", message)
    product = values[0]
    if not isinstance(product, Mapping):
        _fail(
            RemoteAccessError,
            "malformed_results",
            "CDSE OData product entry is not an object",
        )
    product_uuid = _canonical_product_uuid(product.get("Id"))
    actual_name = str(product.get("Name") or "")
    if actual_name != name:
        message = (
            f"CDSE OData exact-name query returned {actual_name!r}, expected {name!r}; "
            "refusing to substitute another parent or version"
        )
        _fail(RemoteAccessError, "invalid_product", message)
    s3_path = str(product.get("S3Path") or "")
    if s3_path and (
        not s3_path.startswith("/eodata/Sentinel-1/SAR/SLC/")
        or not s3_path.endswith(f"/{name}")
    ):
        message = f"CDSE OData product {name!r} is not the registered SLC collection"
        _fail(RemoteAccessError, "invalid_product", message)
    footprint = product.get("GeoFootprint")
    if not isinstance(footprint, Mapping):
        _fail(
            RemoteAccessError,
            "invalid_product",
            f"CDSE OData product {name!r} has no GeoFootprint",
        )
    attributes = _product_attributes(product)
    relative_orbit = _required_int_attribute(attributes, "relativeOrbitNumber")
    orbit_direction = _required_text_attribute(attributes, "orbitDirection").upper()
    if orbit_direction not in {"ASCENDING", "DESCENDING"}:
        _fail(
            RemoteAccessError,
            "invalid_product",
            f"CDSE OData product {name!r} has an invalid orbitDirection",
        )
    serial = str(attributes.get("platformSerialIdentifier") or "").strip().upper()
    if serial and f"S1{serial}" != name_match.group("mission"):
        _fail(
            RemoteAccessError,
            "invalid_product",
            f"CDSE OData product {name!r} has a mismatched platform serial",
        )
    channels = frozenset(
        part.strip().upper()
        for part in re.split(
            r"[\s&,;+/]+", str(attributes.get("polarisationChannels") or "")
        )
        if part.strip()
    )
    swaths = frozenset(
        part.strip().upper()
        for part in re.split(r"[\s,;+/]+", str(attributes.get("swathIdentifier") or ""))
        if part.strip()
    )
    start, end = _product_window(product, name_match)
    annotation_document = _request_odata_json(
        _odata_nodes_url(product_uuid, name, "annotation"), adapter, ledger
    )
    measurement_document = _request_odata_json(
        _odata_nodes_url(product_uuid, name, "measurement"), adapter, ledger
    )
    pairs = _node_pairs(
        annotation_document,
        measurement_document,
        name_match,
        channels=channels,
        swaths=swaths,
    )
    assets: dict[str, Any] = {}
    polarizations: list[str] = []
    for (
        annotation_name,
        annotation_size,
        measurement_name,
        measurement_size,
        swath,
        polarization,
    ) in pairs:
        suffix = f"{swath.casefold()}-{polarization.casefold()}"
        polarizations.append(polarization)
        assets[f"{_MEASUREMENT_KEY}-{suffix}"] = {
            "href": _odata_node_value_url(
                product_uuid, name, "measurement", measurement_name
            ),
            "media_type": "image/tiff",
            "size": measurement_size,
            "properties": {
                "swath": swath,
                "polarization": polarization,
                "file:size": measurement_size,
            },
        }
        assets[f"{_ANNOTATION_KEY}-{suffix}"] = {
            "href": _odata_node_value_url(
                product_uuid, name, "annotation", annotation_name
            ),
            "media_type": "application/xml",
            "size": annotation_size,
            "properties": {
                "swath": swath,
                "polarization": polarization,
                "file:size": annotation_size,
            },
        }
    if not assets:
        _fail(
            RemoteAccessError,
            "invalid_product",
            f"CDSE OData product {name!r} has no measurement nodes",
        )
    record = {
        "id": name.removesuffix(".SAFE"),
        "collection": CDSE_COLLECTION,
        "geometry": footprint,
        "start": start,
        "end": end,
        "platform": f"sentinel-1{serial.casefold()}" if serial else None,
        "polarization": polarizations[0],
        "relative_orbit": relative_orbit,
        "orbit_direction": orbit_direction,
        "ipf_version": attributes.get("processingSoftwareVersion")
        or attributes.get("ipfVersion"),
        "assets": assets,
    }
    logger.info(
        "CDSE exact-parent OData resolved %s: %d measurement pairs",
        name,
        len(pairs),
    )
    return _normalize_record(record, CDSE_CATALOG, adapter, "cdse")


@dataclass(slots=True)
class CDSESearchAdapter:
    """Registered remote adapter for CDSE Sentinel-1 SLC STAC.

    Parameters
    ----------
    client : object, optional
        Injected search client exposing ``search(**kwargs)``.  Intended for
        tests; live discovery uses anonymous HTTPS POST.
    endpoint : str, default=CDSE_STAC_ENDPOINT
        STAC API root.  Only the registered CDSE STAC origin is admitted.

    """

    client: object | None = None
    endpoint: str = CDSE_STAC_ENDPOINT
    collection: str = CDSE_COLLECTION
    provider: str = "cdse"
    origins: tuple[str, ...] = field(init=False)
    path_prefixes: tuple[str, ...] = ("/",)
    redirect_origins: tuple[str, ...] = field(init=False)
    redirect_path_prefixes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    profiles: tuple[str, ...] = ("anonymous", "cdse")

    def __post_init__(self) -> None:
        """Validate the STAC endpoint and configure origin allow-lists."""
        if _origin(self.endpoint) != CDSE_STAC_ORIGIN:
            message = "CDSE STAC endpoint is not approved"
            reject(message, ValueError, logger=logger)
        if self.collection != CDSE_COLLECTION:
            message = "CDSE Sentinel-1 collection is registry-owned"
            reject(message, ValueError, logger=logger)
        object.__setattr__(
            self,
            "origins",
            (
                CDSE_STAC_ORIGIN,
                CDSE_CATALOGUE_ORIGIN,
                CDSE_ZIPPER_ORIGIN,
                CDSE_DOWNLOAD_ORIGIN,
                CDSE_IDENTITY_ORIGIN,
            ),
        )
        object.__setattr__(self, "redirect_origins", self.origins)

    def register(self, name: str = CDSE_CATALOG) -> CDSESearchAdapter:
        """Register this adapter in the provider-neutral remote registry."""
        _register_adapter(name, self)
        return self

    def prepare_query(
        self,
        *,
        datetime_range: tuple[datetime, datetime] | None,
        flight_direction: str | None,
        filters: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Pass datetime, orbit, and provider filters through to ``items``."""
        prepared = dict(filters)
        if flight_direction is not None:
            prepared["sat:orbit_state"] = flight_direction
        return {"datetime_range": datetime_range, "filters": prepared}

    def fetch(
        self,
        asset: RemoteAsset,
        budget: RemoteResourceBudget,
        *,
        ledger: _CallLedger | None = None,
    ) -> bytes:
        """Reject complete-object fetch; ROI canvases use byte ranges."""
        del asset, budget, ledger
        message = (
            "CDSE ROI materialization uses byte range transfer, "
            "not complete-object fetch"
        )
        _fail(RemoteAccessError, "unsupported_representation", message)

    def items(
        self,
        *,
        ledger: _CallLedger | None = None,
        spatial: Any | None = None,
        datetime_range: tuple[datetime, datetime] | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
        collections: tuple[str, ...] | None = None,
        auth_profile: str = "anonymous",
        **_kwargs: Any,
    ) -> Iterable[Mapping[str, Any]]:
        """Yield provider-neutral records with OData ``$value`` asset hrefs."""
        del auth_profile
        if collections is not None and tuple(collections) != (self.collection,):
            message = (
                "CDSE Sentinel-1 collection is registry-owned; "
                f"expected {(self.collection,)!r}, got {tuple(collections)!r}"
            )
            _fail(RemoteAccessError, "invalid_collection", message)
        bounds = _bounds(spatial)
        features = self._search_features(
            list(bounds),
            ledger,
            datetime_range=datetime_range,
            limit=limit,
            collection=self.collection,
        )
        orbit = _filter_orbit(filters)
        emitted = 0
        for feature in features:
            if not isinstance(feature, Mapping):
                continue
            record = self._record_from_feature(feature, orbit=orbit)
            if record is not None:
                yield record
                emitted += 1
            if limit is not None and emitted >= limit:
                break

    def _search_features(
        self,
        bbox: list[float],
        ledger: _CallLedger | None,
        *,
        datetime_range: tuple[datetime, datetime] | None,
        limit: int | None,
        collection: str,
    ) -> Iterable[Mapping[str, Any]]:
        """Search one WGS84 bbox through an injected client or HTTPS POST."""
        if self.client is not None:
            search = getattr(self.client, "search", None)
            if not callable(search):
                message = "injected CDSE client has no callable search"
                _fail(RemoteAccessError, "unobservable_discovery", message)
            kwargs: dict[str, Any] = {
                "collections": [collection],
                "bbox": bbox,
                "limit": min(limit or 100, 100),
                "max_items": limit,
            }
            interval = _datetime_interval(datetime_range)
            if interval is not None:
                kwargs["datetime"] = interval
            result = search(**kwargs)
            items = getattr(result, "items", None)
            values = items() if callable(items) else result
            if values is None or isinstance(values, (str, bytes, Mapping)):
                return ()
            return values
        return self._http_search(
            bbox,
            ledger,
            datetime_range=datetime_range,
            limit=limit,
            collection=collection,
        )

    def _http_search(
        self,
        bbox: list[float],
        ledger: _CallLedger | None,
        *,
        datetime_range: tuple[datetime, datetime] | None,
        limit: int | None,
        collection: str,
    ) -> Iterable[Mapping[str, Any]]:
        """Follow anonymous STAC pages within the registered endpoint trust."""
        url = f"{self.endpoint.rstrip('/')}/search"
        ledger = ledger or _CallLedger(RemoteResourceBudget())
        body: dict[str, Any] | None = {
            "collections": [collection],
            "bbox": bbox,
            "limit": min(limit or 100, 100),
        }
        interval = _datetime_interval(datetime_range)
        if interval is not None:
            body["datetime"] = interval
        method = "POST"
        seen: set[tuple[str, str, str]] = set()
        opener = urllib.request.build_opener(
            _RedirectHandler(self, ledger.budget, ledger)
        )
        while True:
            _safe_url(url, self)
            identity = (url, method, json.dumps(body, sort_keys=True))
            if identity in seen:
                _fail(
                    RemoteAccessError,
                    "pagination_loop",
                    "CDSE pagination did not advance",
                )
            seen.add(identity)
            max_retries = min(ledger.budget.max_retries, _MAX_RETRIES_PER_PAGE)
            attempt = 0
            while True:
                ledger.check_elapsed()
                ledger.request()
                request = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode() if body is not None else None,
                    method=method,
                    headers={
                        "Accept": "application/geo+json, application/json",
                        "Accept-Encoding": "identity",
                        "Content-Type": "application/json",
                    },
                )
                try:
                    with opener.open(
                        request, timeout=ledger.budget.read_timeout_seconds
                    ) as response:
                        if (
                            response.headers.get("Content-Encoding", "identity")
                            != "identity"
                        ):
                            _fail(RemoteAccessError, "unexpected_content_encoding")
                        ledger.begin_response()
                        chunks: list[bytes] = []
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            ledger.response_bytes(len(chunk))
                            chunks.append(chunk)
                        document = json.loads(b"".join(chunks))
                    break
                except urllib.error.HTTPError as error:
                    retry_after = error.headers.get("Retry-After")
                    error.close()
                    if not _is_retryable_transport_error(error):
                        logger.exception("CDSE STAC request failed")
                        _fail(RemoteAccessError, "request_failed", str(error))
                    if attempt >= max_retries:
                        _fail(
                            RemoteLimitError,
                            "max_retries",
                            "CDSE STAC search exhausted retries",
                        )
                    logger.warning(
                        "CDSE STAC request returned HTTP %s; retrying (%d/%d)",
                        error.code,
                        attempt + 1,
                        max_retries,
                    )
                    _wait_before_retry(ledger, retry_after, attempt)
                    attempt += 1
                except json.JSONDecodeError as error:
                    logger.exception("CDSE STAC request failed")
                    _fail(RemoteAccessError, "request_failed", str(error))
                except (urllib.error.URLError, OSError) as error:
                    if not _is_retryable_transport_error(error):
                        logger.exception("CDSE STAC request failed")
                        _fail(RemoteAccessError, "request_failed", str(error))
                    if attempt >= max_retries:
                        _fail(
                            RemoteLimitError,
                            "max_retries",
                            "CDSE STAC search exhausted retries",
                        )
                    logger.warning(
                        "CDSE STAC transport error (%s); retrying (%d/%d)",
                        type(error).__name__,
                        attempt + 1,
                        max_retries,
                    )
                    _wait_before_retry(ledger, None, attempt)
                    attempt += 1
            if not isinstance(document, Mapping) or not isinstance(
                document.get("features"), list
            ):
                _fail(
                    RemoteAccessError,
                    "malformed_results",
                    "CDSE returned no feature list",
                )
            yield from document["features"]
            links = document.get("links", [])
            if not isinstance(links, list):
                _fail(
                    RemoteAccessError,
                    "malformed_results",
                    "CDSE returned invalid links",
                )
            continuations = [
                link
                for link in links
                if isinstance(link, Mapping) and link.get("rel") == "next"
            ]
            if not continuations:
                return
            if len(continuations) != 1:
                _fail(
                    RemoteAccessError,
                    "malformed_results",
                    "CDSE returned ambiguous continuation",
                )
            link = continuations[0]
            href = link.get("href")
            method = str(link.get("method", "GET")).upper()
            if not isinstance(href, str) or not href or method not in {"GET", "POST"}:
                _fail(
                    RemoteAccessError,
                    "malformed_results",
                    "CDSE returned invalid continuation",
                )
            url = urllib.parse.urljoin(url, href)
            next_body = link.get("body")
            if next_body is not None and not isinstance(next_body, Mapping):
                _fail(
                    RemoteAccessError,
                    "malformed_results",
                    "CDSE returned invalid continuation body",
                )
            if method == "GET":
                body = None
            elif link.get("merge", False):
                body = {**(body or {}), **(next_body or {})}
            else:
                body = dict(next_body or {})

    def _record_from_feature(
        self,
        feature: Mapping[str, Any],
        *,
        orbit: int | None,
    ) -> dict[str, Any] | None:
        """Rewrite one STAC feature to OData measurement/annotation assets."""
        item_id = str(feature.get("id") or "")
        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            properties = {}
        relative = properties.get("sat:relative_orbit")
        assets = feature.get("assets")
        pairs = self._select_pairs(assets) if isinstance(assets, Mapping) else ()
        geometry = feature.get("geometry")
        if (
            not item_id
            or (orbit is not None and relative != orbit)
            or not pairs
            or not isinstance(geometry, Mapping)
        ):
            return None
        start = properties.get("datetime") or properties.get("start_datetime")
        end = properties.get("end_datetime") or start
        rewritten: dict[str, Any] = {}
        polarizations: list[str] = []
        for measurement_key, measurement, annotation in pairs:
            measurement_href = _odata_href(measurement)
            annotation_href = _odata_href(annotation)
            if measurement_href is None or annotation_href is None:
                logger.warning(
                    "CDSE item %s asset %s has no OData Nodes(...)/$value; skipped",
                    item_id,
                    measurement_key,
                )
                continue
            measurement_size = measurement.get("file:size")
            annotation_size = annotation.get("file:size")
            swath = _swath_from_key(measurement_key)
            polarization = measurement_key.rsplit("-", 1)[-1].upper()
            polarizations.append(polarization)
            rewritten[f"{_MEASUREMENT_KEY}-{measurement_key}"] = {
                "href": measurement_href,
                "media_type": measurement.get("type", "image/tiff"),
                "size": measurement_size,
                "properties": {
                    "swath": swath,
                    "polarization": polarization,
                    "file:size": measurement_size,
                },
            }
            rewritten[f"{_ANNOTATION_KEY}-{measurement_key}"] = {
                "href": annotation_href,
                "media_type": annotation.get("type", "application/xml"),
                "size": annotation_size,
                "properties": {
                    "swath": swath,
                    "polarization": polarization,
                    "file:size": annotation_size,
                },
            }
        if not rewritten:
            logger.warning(
                "CDSE item %s has no OData Nodes(...)/$value alternate; skipped",
                item_id,
            )
            return None
        unique_pols = tuple(dict.fromkeys(polarizations))
        software = properties.get("processing:software")
        ipf_version = (
            software.get("Sentinel-1 IPF") if isinstance(software, Mapping) else None
        )
        return {
            "id": item_id,
            "collection": self.collection,
            "geometry": geometry,
            "start": start,
            "end": end,
            "platform": properties.get("platform"),
            "polarization": unique_pols[0] if unique_pols else "VV",
            "relative_orbit": relative,
            "orbit_direction": properties.get("sat:orbit_state"),
            "ipf_version": str(ipf_version) if ipf_version else None,
            "assets": rewritten,
        }

    def _select_pairs(
        self, assets: Mapping[str, Any]
    ) -> tuple[tuple[str, Mapping[str, Any], Mapping[str, Any]], ...]:
        """Return every measurement/annotation pair with an OData rewrite."""
        pairs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
        for key in _SWATH_POL:
            measurement = assets.get(key)
            annotation = assets.get(_annotation_key(key))
            if isinstance(measurement, Mapping) and isinstance(annotation, Mapping):
                pairs.append((key, measurement, annotation))
        return tuple(pairs)


def register_cdse_catalog(name: str = CDSE_CATALOG, **kwargs: Any) -> CDSESearchAdapter:
    """Construct and register the CDSE Sentinel-1 catalog adapter.

    Parameters
    ----------
    name : str, optional
        Registry name.  Only the stable ``sentinel1-cdse`` name is admitted.
    **kwargs : Any
        Forwarded to :class:`CDSESearchAdapter` (injected ``client`` in tests).

    Returns
    -------
    CDSESearchAdapter
        The registered adapter.

    Raises
    ------
    ValueError
        If ``name`` is not the stable CDSE catalog, the STAC origin is
        unapproved, or the collection is not registry-owned.

    """
    if name != CDSE_CATALOG:
        message = (
            f"CDSE Sentinel-1 must use the stable registration name {CDSE_CATALOG!r}"
        )
        reject(message, ValueError, logger=logger)
    return CDSESearchAdapter(**kwargs).register(name)


def ensure_cdse_catalog(name: str | None = None) -> str:
    """Register or reuse the stable ``sentinel1-cdse`` catalog.

    Parameters
    ----------
    name : str or None, optional
        Registry name.  ``None`` uses :data:`CDSE_CATALOG`.

    Returns
    -------
    str
        The registered catalog name.

    Raises
    ------
    ValueError
        If ``name`` is not the stable catalog, or an incompatible adapter
        already occupies that name.

    """
    catalog = name or CDSE_CATALOG
    if catalog != CDSE_CATALOG:
        message = (
            f"CDSE Sentinel-1 must use the stable registration name {CDSE_CATALOG!r}"
        )
        reject(message, ValueError, logger=logger)
    expected = CDSESearchAdapter()
    identity_fields = (
        "provider",
        "origins",
        "path_prefixes",
        "redirect_origins",
        "profiles",
        "collection",
        "endpoint",
    )
    with _REGISTRY_LOCK:
        if catalog in _ADAPTERS:
            adapter = _ADAPTERS[catalog]
            if any(
                getattr(adapter, field_name, None) != getattr(expected, field_name)
                for field_name in identity_fields
            ):
                message = f"catalog {catalog!r} has an incompatible CDSE registration"
                reject(message, ValueError, logger=logger)
            return catalog
        _ADAPTERS[catalog] = expected
    return catalog


__all__ = [
    "CDSE_CATALOG",
    "CDSE_COLLECTION",
    "CDSE_STAC_ENDPOINT",
    "CDSESearchAdapter",
    "cdse_range_fetcher",
    "ensure_cdse_catalog",
    "fetch_cdse_ranges",
    "refresh_cdse_token",
    "register_cdse_catalog",
    "request_cdse_token",
]
