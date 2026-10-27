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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import requests

from faninsar.logging import setup_logger
from faninsar.remote.access import _safe_url, _validate_url
from faninsar.remote.auth import _netrc_credentials, _no_auth
from faninsar.remote.catalog import _ADAPTERS, _REGISTRY_LOCK, _register_adapter
from faninsar.remote.errors import RemoteAccessError, _fail
from faninsar.remote.fetch import RangeFetcher, RangeSpan, fetch_ranges_async
from faninsar.remote.fetch.complete import _RedirectHandler
from faninsar.remote.protocols import _CallLedger
from faninsar.remote.records import RemoteAsset, RemoteResourceBudget

logger = setup_logger(__name__)

CDSE_CATALOG = "sentinel1-cdse"
CDSE_COLLECTION = "sentinel-1-slc"
CDSE_STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1"
CDSE_STAC_ORIGIN = "https://stac.dataspace.copernicus.eu"
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
    from faninsar.remote import _query_geometry

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
        logger.error(message)
        _fail(RemoteAccessError, "missing_cdse_credentials", message)
    username, password = credentials
    ensure_cdse_catalog()
    with _REGISTRY_LOCK:
        adapter = _ADAPTERS.get(CDSE_CATALOG)
    if adapter is None:
        message = f"CDSE catalog {CDSE_CATALOG!r} is not registered"
        logger.error(message)
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
        logger.error("CDSE token response omitted access_token")
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


async def fetch_cdse_ranges(spans: Sequence[RangeSpan]) -> dict[str, bytes]:
    """Fetch OData byte ranges after validating each URL, then attach Bearer.

    Parameters
    ----------
    spans : sequence of RangeSpan
        Already-sized inclusive byte intervals on CDSE OData ``$value`` URLs.

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
    ensure_cdse_catalog()
    with _REGISTRY_LOCK:
        adapter = _ADAPTERS.get(CDSE_CATALOG)
    if adapter is None:
        message = f"CDSE catalog {CDSE_CATALOG!r} is not registered"
        logger.error(message)
        _fail(RemoteAccessError, "unknown_catalog", message)
    for span in span_tuple:
        _validate_url(span.url, adapter)
        if not _is_odata_value(span.url) or not _needs_cdse_bearer(span.url):
            message = "CDSE range URL is not an OData Nodes(...)/$value href"
            logger.error(message)
            _fail(RemoteAccessError, "invalid_endpoint", message)
    token = request_cdse_token()
    async with RangeFetcher(token=token, refresh=refresh_cdse_token) as fetcher:
        return await fetch_ranges_async(span_tuple, fetcher=fetcher)


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
            logger.error(message)
            raise ValueError(message)
        if self.collection != CDSE_COLLECTION:
            message = "CDSE Sentinel-1 collection is registry-owned"
            logger.error(message)
            raise ValueError(message)
        object.__setattr__(
            self,
            "origins",
            (
                CDSE_STAC_ORIGIN,
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
        logger.error(message)
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
            logger.error(message)
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
        records: list[Mapping[str, Any]] = []
        for feature in features:
            if not isinstance(feature, Mapping):
                continue
            record = self._record_from_feature(feature, orbit=orbit)
            if record is not None:
                records.append(record)
            if limit is not None and len(records) >= limit:
                break
        return records

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
                logger.error(message)
                _fail(RemoteAccessError, "unobservable_discovery", message)
            kwargs: dict[str, Any] = {
                "collections": [collection],
                "bbox": bbox,
                "limit": limit or 100,
            }
            interval = _datetime_interval(datetime_range)
            if interval is not None:
                kwargs["datetime"] = interval
            result = search(**kwargs)
            items = getattr(result, "items", None)
            values = items() if callable(items) else result
            if values is None or isinstance(values, (str, bytes, Mapping)):
                return ()
            return list(values)
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
        """POST one anonymous STAC search against the registered endpoint."""
        url = f"{self.endpoint.rstrip('/')}/search"
        _safe_url(url, self)
        if ledger is None:
            ledger = _CallLedger(RemoteResourceBudget())
        body: dict[str, Any] = {
            "collections": [collection],
            "bbox": bbox,
            "limit": min(limit or 100, ledger.budget.max_items),
        }
        interval = _datetime_interval(datetime_range)
        if interval is not None:
            body["datetime"] = interval
        payload = json.dumps(body).encode("utf-8")
        ledger.request()
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Accept": "application/geo+json, application/json",
                "Accept-Encoding": "identity",
                "Content-Type": "application/json",
            },
        )
        opener = urllib.request.build_opener(
            _RedirectHandler(self, ledger.budget, ledger)
        )
        try:
            with opener.open(
                request, timeout=ledger.budget.read_timeout_seconds
            ) as response:
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    message = "CDSE STAC response used a compressed Content-Encoding"
                    logger.error(message)
                    _fail(RemoteAccessError, "unexpected_content_encoding", message)
                ledger.begin_response()
                chunks: list[bytes] = []
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    ledger.response_bytes(len(chunk))
                    chunks.append(chunk)
                document = json.loads(b"".join(chunks))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            logger.exception("CDSE STAC request failed")
            _fail(RemoteAccessError, "request_failed", str(error))
        features = document.get("features", []) if isinstance(document, Mapping) else []
        return features if isinstance(features, list) else ()

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
        return {
            "id": item_id,
            "collection": self.collection,
            "geometry": geometry,
            "start": start,
            "end": end,
            "platform": properties.get("platform"),
            "polarization": unique_pols[0] if unique_pols else "VV",
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
        logger.error(message)
        raise ValueError(message)
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
        logger.error(message)
        raise ValueError(message)
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
                logger.error(message)
                raise ValueError(message)
            return catalog
        _ADAPTERS[catalog] = expected
    return catalog


__all__ = [
    "CDSE_CATALOG",
    "CDSE_COLLECTION",
    "CDSE_STAC_ENDPOINT",
    "CDSESearchAdapter",
    "ensure_cdse_catalog",
    "fetch_cdse_ranges",
    "refresh_cdse_token",
    "register_cdse_catalog",
    "request_cdse_token",
]
