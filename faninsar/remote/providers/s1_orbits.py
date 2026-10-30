"""Anonymous public catalog for the s1-orbits S3 bucket (PROPOSAL-0054).

The adapter is a dedicated origin.  It never admits ``earthdata-asf`` and is
not registered on ``sentinel1-asf``.  Covering-file selection stays
mission-owned; this module only lists and GET objects on the public bucket.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

from faninsar.logging import setup_logger
from faninsar.remote.access import _response_is_complete, _url_origin, _validate_url
from faninsar.remote.catalog import _ADAPTERS, _REGISTRY_LOCK
from faninsar.remote.errors import RemoteAccessError, _fail
from faninsar.remote.fetch.complete import _FetchedPayload, _RedirectHandler
from faninsar.remote.protocols import _CallLedger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from faninsar.remote.records import RemoteAsset, RemoteResourceBudget

logger = setup_logger(__name__)

S1_ORBITS_CATALOG = "s1-orbits"
S1_ORBITS_ORIGIN = "https://s1-orbits.s3.amazonaws.com"
S1_ORBITS_REGIONAL_ORIGIN = "https://s1-orbits.s3.us-west-2.amazonaws.com"
_LIST_PREFIX = re.compile(
    r"^AUX_(?P<family>POEORB|RESORB)/S1[ABCD]_OPER_AUX_(?P=family)_$"
)
_OBJECT_KEY = re.compile(
    r"^AUX_(POEORB|RESORB)/S1[ABCD]_OPER_AUX_(POEORB|RESORB)_OPOD_"
    r"\d{8}T\d{6}_V\d{8}T\d{6}_\d{8}T\d{6}\.EOF$",
    re.IGNORECASE,
)


class S1OrbitsHttpError(RemoteAccessError):
    """An s1-orbits list or object GET returned a non-success HTTP status."""

    def __init__(self, status: int, host: str, url: str) -> None:
        """Record the failing host, URL, and status code."""
        self.status = status
        self.host = host
        self.url = url
        super().__init__("http_error", f"HTTP {status} from {host}")


def _etag_to_checksum(etag: str | None) -> str | None:
    """Return the provider checksum for one S3 ``ETag`` value.

    A single-part upload's ETag (a 32-hex string with no ``-N`` suffix) equals
    the object MD5, so it is returned as the qualified ``md5:<hex>``
    checksum.  Multipart ETags (``<hex>-N``), absent values, and malformed
    values yield ``None``: they must never drive a checksum comparison.

    Parameters
    ----------
    etag : str or None
        The raw ``ETag`` response header value.

    Returns
    -------
    str or None
        ``md5:<hex>`` for a single-part ETag, otherwise ``None``.

    Examples
    --------
    >>> _etag_to_checksum("d41d8cd98f00b204e9800998ecf8427e")
    'md5:d41d8cd98f00b204e9800998ecf8427e'
    >>> _etag_to_checksum("d41d8cd98f00b204e9800998ecf8427e-21") is None
    True

    """
    if not isinstance(etag, str):
        return None
    value = etag.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{32}", value):
        return None
    return f"md5:{value.lower()}"


def _response_etag(response: Any) -> str | None:
    """Return the raw ``ETag`` header of one urllib response, if present."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    if callable(get):
        value = get("ETag") or get("etag")
        return value if isinstance(value, str) else None
    try:
        for name, value in headers.items():
            if str(name).lower() == "etag":
                return value if isinstance(value, str) else None
    except (TypeError, ValueError):
        return None
    return None


def _local_tag(tag: str) -> str:
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _parse_list_result(payload: bytes) -> tuple[list[str], str | None, bool]:
    """Parse a ListObjects v2 page into keys, continuation token, and truncation."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        logger.exception("s1-orbits listing is not XML")
        _fail(RemoteAccessError, "invalid_listing", str(error))
    keys: list[str] = []
    token: str | None = None
    truncated = False
    for element in root.iter():
        name = _local_tag(element.tag)
        text = (element.text or "").strip()
        if name == "Key" and text:
            keys.append(text)
        elif name == "NextContinuationToken" and text:
            token = text
        elif name == "IsTruncated":
            truncated = text.lower() == "true"
    return keys, token, truncated


def _safe_object_key(key: str) -> str | None:
    """Admit basename-only AUX_POEORB/AUX_RESORB EOF keys."""
    if not isinstance(key, str) or "\\" in key or ".." in key.split("/"):
        return None
    if key.startswith("/") or not _OBJECT_KEY.fullmatch(key):
        return None
    family_dir, filename = key.split("/", 1)
    expected_family = family_dir.removeprefix("AUX_")
    if expected_family not in filename.upper():
        return None
    return f"{family_dir}/{filename}"


@dataclass(slots=True)
class S1OrbitsAdapter:
    """Anonymous-only adapter for the public s1-orbits S3 bucket."""

    provider: str = "s1-orbits"
    origins: tuple[str, ...] = (S1_ORBITS_ORIGIN, S1_ORBITS_REGIONAL_ORIGIN)
    path_prefixes: tuple[str, ...] = ("/AUX_POEORB", "/AUX_RESORB")
    redirect_origins: tuple[str, ...] = (
        S1_ORBITS_ORIGIN,
        S1_ORBITS_REGIONAL_ORIGIN,
    )
    redirect_path_prefixes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    profiles: tuple[str, ...] = ("anonymous",)
    _allow_anonymous_delivery: bool = False

    def items(self, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
        """Spatial catalog search is not used for covering-orbit discovery."""
        del kwargs
        return ()

    def fetch(
        self,
        asset: RemoteAsset,
        budget: RemoteResourceBudget,
        *,
        ledger: _CallLedger | None = None,
    ) -> bytes:
        """GET one EOF object without attaching scene-path credentials.

        A single-part S3 ``ETag`` response header equals the object MD5 and is
        surfaced as the download-time provider checksum (PROPOSAL-0060 seam
        1); multipart and malformed ETags yield no checksum, so the transfer
        layer records ``unverifiable`` rather than comparing a wrong digest.
        """
        operation = ledger or _CallLedger(budget)
        if asset.auth_profile != "anonymous":
            _fail(RemoteAccessError, "unknown_auth_profile")
        payload, etag = self._anonymous_get(
            asset.href, budget, operation, listing=False
        )
        checksum = _etag_to_checksum(etag)
        return _FetchedPayload(payload, checksum)

    def list_object_keys(
        self,
        prefix: str,
        *,
        budget: RemoteResourceBudget,
        ledger: _CallLedger | None = None,
    ) -> list[str]:
        """Return AUX_POEORB/AUX_RESORB keys under one mission prefix."""
        if _LIST_PREFIX.fullmatch(prefix) is None:
            message = f"unsupported s1-orbits listing prefix: {prefix}"
            logger.error(message)
            _fail(RemoteAccessError, "invalid_listing_prefix", message)
        keys: list[str] = []
        operation = ledger or _CallLedger(budget)
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            query = {
                "list-type": "2",
                "prefix": prefix,
                "max-keys": "1000",
            }
            if token is not None:
                query["continuation-token"] = token
            url = f"{S1_ORBITS_ORIGIN}/?{urllib.parse.urlencode(query)}"
            payload, _etag = self._anonymous_get(url, budget, operation, listing=True)
            page, token, truncated = _parse_list_result(payload)
            for key in page:
                safe = _safe_object_key(key)
                if safe is not None:
                    keys.append(safe)
            if not truncated or not token or token in seen_tokens:
                break
            seen_tokens.add(token)
        return keys

    def _validate_listing_url(self, url: str) -> str:
        """Admit only ListObjects v2 queries on the registered virtual-host origin."""
        try:
            parsed = urllib.parse.urlsplit(url)
            origin = _url_origin(url)
        except (TypeError, ValueError):
            _fail(RemoteAccessError, "invalid_endpoint")
        if origin != S1_ORBITS_ORIGIN or parsed.path not in {"", "/"}:
            _fail(RemoteAccessError, "unregistered_endpoint")
        query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
        prefix = query.get("prefix", "")
        if query.get("list-type") != "2" or _LIST_PREFIX.fullmatch(prefix) is None:
            _fail(RemoteAccessError, "invalid_listing_prefix")
        return url

    def _anonymous_get(
        self,
        url: str,
        budget: RemoteResourceBudget,
        ledger: _CallLedger,
        *,
        listing: bool,
    ) -> tuple[bytes, str | None]:
        """GET one listing page or object, returning bytes and the ``ETag``."""
        if listing:
            target = self._validate_listing_url(url)
        else:
            target = _validate_url(url, self)
        ledger.request()
        ledger.begin_response()
        opener = urllib.request.build_opener(_RedirectHandler(self, budget, ledger))
        request = urllib.request.Request(
            target,
            headers={"Accept-Encoding": "identity"},
        )
        try:
            with opener.open(request, timeout=budget.read_timeout_seconds) as response:
                _response_is_complete(response)
                payload = response.read()
                etag = _response_etag(response)
        except urllib.error.HTTPError as error:
            host = urllib.parse.urlsplit(url).hostname or ""
            try:
                body = error.read() or b""
                ledger.response_bytes(len(body))
            except (OSError, TypeError, ValueError):
                logger.warning("s1-orbits error body is unreadable")
            raise S1OrbitsHttpError(int(error.code), host, url) from error
        except urllib.error.URLError as error:
            host = urllib.parse.urlsplit(url).hostname or ""
            logger.exception("s1-orbits request failed")
            raise S1OrbitsHttpError(0, host, url) from error
        ledger.response_bytes(len(payload))
        return bytes(payload), etag


def get_s1_orbits_adapter() -> S1OrbitsAdapter:
    """Return the validated adapter registered for the public orbit bucket.

    Mission code uses this boundary instead of reaching into the remote
    catalog registry.  Registration remains private to the remote package.
    """
    ensure_s1_orbits_catalog()
    with _REGISTRY_LOCK:
        adapter = _ADAPTERS.get(S1_ORBITS_CATALOG)
    if not isinstance(adapter, S1OrbitsAdapter):
        message = "catalog 's1-orbits' has an incompatible registration"
        logger.error(message)
        raise TypeError(message)
    return adapter


def ensure_s1_orbits_catalog(name: str | None = None) -> str:
    """Register or reuse the dedicated anonymous s1-orbits catalog."""
    catalog = name or S1_ORBITS_CATALOG
    if catalog != S1_ORBITS_CATALOG:
        message = (
            f"s1-orbits must use the stable registration name {S1_ORBITS_CATALOG!r}"
        )
        logger.error(message)
        raise ValueError(message)
    expected = S1OrbitsAdapter()
    identity_fields = (
        "provider",
        "origins",
        "path_prefixes",
        "redirect_origins",
        "profiles",
    )
    with _REGISTRY_LOCK:
        if catalog in _ADAPTERS:
            adapter = _ADAPTERS[catalog]
            if any(
                getattr(adapter, field_name, None) != getattr(expected, field_name)
                for field_name in identity_fields
            ):
                message = (
                    f"catalog {catalog!r} has an incompatible s1-orbits registration"
                )
                logger.error(message)
                raise ValueError(message)
            return catalog
        _ADAPTERS[catalog] = expected
    return catalog


__all__ = [
    "S1_ORBITS_CATALOG",
    "S1_ORBITS_ORIGIN",
    "S1_ORBITS_REGIONAL_ORIGIN",
    "S1OrbitsAdapter",
    "S1OrbitsHttpError",
    "ensure_s1_orbits_catalog",
    "get_s1_orbits_adapter",
]
