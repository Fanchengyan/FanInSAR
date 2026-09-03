"""Planetary Computer Copernicus DEM adapter (PROPOSAL-0045).

The adapter deliberately keeps provider details at the remote boundary.  STAC
discovery signs item assets in memory; the provider-neutral remote boundary
then owns identity, retries, atomic publication, and complete-file semantics.
No signed URL is retained in a plan or cache identity.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from faninsar.logging import setup_logger
from faninsar.remote import (
    RemoteAccessError,
    RemoteResourceBudget,
    _CallLedger,
    _fail,
    _RedirectHandler,
    _register_adapter,
    _safe_url,
)
from faninsar.remote.standards import MalformedSTACItemError, normalize_stac_item

logger = setup_logger(__name__)

PC_STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_STAC_HOST = "planetarycomputer.microsoft.com"
PC_ASSET_HOST = "elevationeuwest.blob.core.windows.net"
COP_DEM_GLO30_COLLECTION = "cop-dem-glo-30"
PC_ASSET_KEY = "data"


def _origin(url: str) -> str:
    """Return a normalized HTTPS origin."""
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    port = parts.port
    return f"https://{host}" + (f":{port}" if port not in (None, 443) else "")


def _as_mapping(item: object) -> Mapping[str, Any]:
    """Convert a pystac Item or test item into a JSON-like mapping."""
    if isinstance(item, Mapping):
        return item
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    assets = getattr(item, "assets", {})
    normalized_assets: dict[str, dict[str, Any]] = {}
    for key, asset in dict(assets).items():
        href = str(getattr(asset, "href", ""))
        fields = dict(getattr(asset, "extra_fields", {}) or {})
        fields["href"] = href
        media_type = getattr(asset, "media_type", None)
        if media_type is not None:
            fields["type"] = media_type
        normalized_assets[str(key)] = fields
    geometry = getattr(item, "geometry", None)
    collection = getattr(item, "collection_id", None)
    result: dict[str, Any] = {
        "type": "Feature",
        "id": str(getattr(item, "id", "")),
        "geometry": geometry,
        "properties": dict(getattr(item, "properties", {}) or {}),
        "assets": normalized_assets,
    }
    stac_version = getattr(item, "stac_version", None)
    if stac_version is not None:
        result["stac_version"] = str(stac_version)
    stac_extensions = getattr(item, "stac_extensions", None)
    if stac_extensions is not None:
        result["stac_extensions"] = list(stac_extensions)
    if collection is not None:
        result["collection"] = str(collection)
    return result


def _signed_href(item: object, key: str) -> str:
    """Read one asset href after the item has been signed in place."""
    if isinstance(item, Mapping):
        asset = item.get("assets", {}).get(key, {})
        return str(asset.get("href", ""))
    assets = getattr(item, "assets", {})
    asset = assets.get(key)
    return str(getattr(asset, "href", ""))


def _meter_response(response: Any, ledger: _CallLedger) -> None:
    """Charge one requests response, including its already-loaded body."""
    if getattr(response, "_faninsar_metered", False):
        return
    if response.headers.get("Content-Encoding", "identity") != "identity":
        _fail(RemoteAccessError, "unexpected_content_encoding")
    try:
        content = response.content
    except Exception as error:
        logger.exception("Planetary Computer STAC response is unreadable")
        _fail(RemoteAccessError, "unobservable_response", str(error))
    if not isinstance(content, (bytes, bytearray)):
        _fail(RemoteAccessError, "unobservable_response")
    ledger.begin_response()
    ledger.response_bytes(len(content))
    with suppress(AttributeError, TypeError):
        # A requests.Response accepts arbitrary attributes.  Test doubles
        # with slots do not, but their response is still fully metered for
        # this invocation.
        response._faninsar_metered = True


def _instrument_session(
    session: Any,
    ledger: _CallLedger,
    adapter: PlanetaryComputerAdapter,
) -> Callable[[], None]:
    """Instrument one operation-scoped requests-like session.

    ``pystac-client`` uses ``requests.Session.send`` after preparing each
    STAC request.  Wrapping that concrete boundary lets the remote operation
    account for the request, redirect, response-byte, and elapsed-time
    portions that are otherwise invisible to the provider-neutral ledger.
    Retries are disabled on the operation session by the caller; this keeps
    retry accounting from occurring inside urllib3 without an observable
    boundary.
    """
    send = getattr(session, "send", None)
    if not callable(send):
        message = "Planetary Computer STAC session cannot be instrumented"
        logger.error(message)
        _fail(RemoteAccessError, "unobservable_discovery", message)
    previous_send = send

    def metered_send(request: Any, *args: Any, **kwargs: Any) -> Any:
        """Meter one requests send and every response in its redirect chain."""
        url = getattr(request, "url", "")
        _safe_url(str(url), adapter)
        ledger.check_elapsed()
        requests_before = ledger.requests
        ledger.request()
        response = previous_send(request, *args, **kwargs)
        history = getattr(response, "history", ())
        history_values = list(history) if isinstance(history, Iterable) else []
        # Real requests recursively calls this wrapped ``send`` for each
        # redirect, so those requests are already charged by nested wrappers.
        # Fixture transports may instead return a complete ``history`` list;
        # charge only the redirect entries that were not observed recursively.
        nested_requests = ledger.requests - requests_before - 1
        missing_redirects = max(0, len(history_values) - nested_requests)
        for _ in range(missing_redirects):
            ledger.redirect()
            ledger.request()
        for previous in history_values:
            _meter_response(previous, ledger)
        _meter_response(response, ledger)
        ledger.check_elapsed()
        return response

    try:
        session.send = metered_send
    except (AttributeError, TypeError) as error:
        message = "Planetary Computer STAC session cannot be instrumented"
        logger.exception(message)
        _fail(RemoteAccessError, "unobservable_discovery", str(error))

    def restore() -> None:
        """Restore the caller-owned session after the operation."""
        session.send = previous_send

    return restore


@dataclass(slots=True)
class PlanetaryComputerAdapter:
    """Registered remote adapter for the PC Copernicus GLO-30 collection.

    Parameters
    ----------
    client : object, optional
        Injected STAC client.  It must expose ``search`` and is intended for
        deterministic tests or applications that already own a client.
    signer : callable, optional
        In-memory item signer, normally ``planetary_computer.sign_inplace``.
    endpoint : str, default=PC_STAC_ENDPOINT
        STAC API endpoint.  Only the registered endpoint origin is admitted.

    """

    client: object | None = None
    signer: Callable[[object], object] | None = None
    endpoint: str = PC_STAC_ENDPOINT
    collection: str = COP_DEM_GLO30_COLLECTION
    asset_key: str = PC_ASSET_KEY
    provider: str = "pc"
    origins: tuple[str, ...] = field(init=False)
    path_prefixes: tuple[str, ...] = field(init=False)
    redirect_origins: tuple[str, ...] = field(init=False)
    profiles: tuple[str, ...] = ("anonymous",)
    _signed: dict[tuple[str, str], str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Validate the endpoint and configure URL allowlists."""
        parts = urllib.parse.urlsplit(self.endpoint)
        if parts.scheme.lower() != "https" or parts.hostname != PC_STAC_HOST:
            message = "Planetary Computer endpoint is not approved"
            logger.error(message)
            raise ValueError(message)
        object.__setattr__(
            self,
            "origins",
            (_origin(self.endpoint), f"https://{PC_ASSET_HOST}"),
        )
        object.__setattr__(self, "redirect_origins", self.origins)
        object.__setattr__(self, "path_prefixes", (parts.path.rstrip("/") or "/", "/"))
        if (
            self.collection != COP_DEM_GLO30_COLLECTION
            or self.asset_key != PC_ASSET_KEY
        ):
            message = (
                "Planetary Computer GLO-30 collection and asset are registry-owned"
            )
            logger.error(message)
            raise ValueError(message)

    def register(self, name: str = "pc") -> PlanetaryComputerAdapter:
        """Register this adapter in the provider-neutral remote registry."""
        _register_adapter(name, self)
        return self

    def _sign_item(self, item: object) -> object:
        """Sign one STAC item without writing credentials to disk."""
        signer = self.signer
        if signer is None:
            try:
                import planetary_computer
            except ImportError as error:
                message = "Planetary Computer support requires planetary-computer"
                logger.exception(message)
                try:
                    _fail(RemoteAccessError, "missing_optional_dependency", message)
                except RemoteAccessError as raised:
                    raise raised from error
            signer = planetary_computer.sign_inplace
        signed = signer(item)
        return item if signed is None else signed

    def _client(self, ledger: _CallLedger) -> tuple[object, Callable[[], None] | None]:
        """Return a STAC client and cleanup callback for one operation.

        Injected clients must expose their ``pystac-client`` ``_stac_io``
        session so it can be instrumented.  A deterministic fixture may opt
        into the explicit ``_faninsar_offline`` marker; such a client performs
        no network I/O and therefore needs no fabricated request accounting.
        """
        if self.client is not None:
            if getattr(self.client, "_faninsar_offline", False):
                return self.client, None
            stac_io = getattr(self.client, "_stac_io", None)
            session = getattr(stac_io, "session", None)
            if session is None:
                message = (
                    "injected Planetary Computer client has no observable "
                    "STAC transport"
                )
                logger.error(message)
                _fail(RemoteAccessError, "unobservable_discovery", message)
            return self.client, _instrument_session(session, ledger, self)
        try:
            import pystac_client
        except ImportError as error:
            message = "Planetary Computer support requires pystac-client"
            logger.exception(message)
            try:
                _fail(RemoteAccessError, "missing_optional_dependency", message)
            except RemoteAccessError as raised:
                raise raised from error
        try:
            from pystac_client.stac_api_io import StacApiIO

            # urllib3 retries are intentionally disabled.  If retries are
            # enabled internally, they cannot be charged at this boundary.
            stac_io = StacApiIO(
                timeout=(
                    ledger.budget.connect_timeout_seconds,
                    ledger.budget.read_timeout_seconds,
                ),
                max_retries=0,
            )
            restore = _instrument_session(stac_io.session, ledger, self)
            try:
                client = pystac_client.Client.open(self.endpoint, stac_io=stac_io)
            except Exception:
                restore()
                stac_io.session.close()
                raise
        except RemoteAccessError:
            raise
        except Exception as error:
            logger.exception("Planetary Computer STAC client initialization failed")
            _fail(RemoteAccessError, "request_failed", str(error))
        return client, lambda: (restore(), stac_io.session.close())

    def _search_items(
        self,
        bbox: list[float],
        ledger: _CallLedger | None,
        *,
        datetime_range: tuple[datetime, datetime] | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterable[object]:
        """Search one WGS84 bbox through an instrumented STAC client."""
        if ledger is None:
            ledger = _CallLedger(RemoteResourceBudget())
        client, restore = self._client(ledger)
        search = getattr(client, "search", None)
        if not callable(search):
            if restore is not None:
                restore()
            message = "Planetary Computer client exposes no search method"
            logger.error(message)
            _fail(RemoteAccessError, "unobservable_discovery", message)

        parameters: dict[str, Any] = {
            "collections": [self.collection],
            "bbox": bbox,
            "max_items": min(limit, ledger.budget.max_items)
            if limit is not None
            else ledger.budget.max_items,
        }
        if datetime_range is not None:
            parameters["datetime"] = "/".join(
                value.astimezone(UTC).isoformat() for value in datetime_range
            )
        if filters:
            parameters["query"] = dict(filters)
        try:
            result = search(**parameters)
            items = getattr(result, "items", None)
            values = items() if callable(items) else result
            if values is None or isinstance(values, (str, bytes, Mapping)):
                message = "Planetary Computer search returned no item iterator"
                logger.error(message)
                _fail(RemoteAccessError, "unobservable_discovery", message)
            try:
                yield from values
            finally:
                if restore is not None:
                    restore()
        except RemoteAccessError:
            if restore is not None:
                restore()
            raise
        except Exception as error:
            if restore is not None:
                restore()
            logger.exception("Planetary Computer STAC search failed")
            _fail(RemoteAccessError, "request_failed", str(error))

    def _http_items(
        self, bbox: list[float], ledger: _CallLedger | None
    ) -> Iterable[Mapping[str, Any]]:
        """Perform a bounded anonymous STAC GET when no client is injected."""
        query = urllib.parse.urlencode(
            {
                "collections": self.collection,
                "bbox": ",".join(map(str, bbox)),
                "limit": "100",
            }
        )
        url = f"{self.endpoint.rstrip('/')}/search?{query}"
        _safe_url(url, self)
        if ledger is None:
            ledger = _CallLedger(RemoteResourceBudget())
        ledger.request()
        request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
        opener = urllib.request.build_opener(
            _RedirectHandler(self, ledger.budget, ledger)
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
                payload = json.loads(b"".join(chunks))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            logger.exception("Planetary Computer STAC request failed")
            _fail(RemoteAccessError, "request_failed", str(error))
        features = payload.get("features", []) if isinstance(payload, Mapping) else []
        return features if isinstance(features, list) else ()

    def items(
        self,
        *,
        ledger: _CallLedger | None = None,
        spatial: Any | None = None,
        datetime_range: tuple[datetime, datetime] | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterable[Mapping[str, Any]]:
        """Yield normalized PC records, signing each item in memory."""
        if spatial is None:
            bounds = (-180.0, -90.0, 180.0, 90.0)
        # The public boundary passes its already-projected Shapely query
        # geometry.  Keep compatibility with direct adapter callers that
        # still provide a FanInSAR query object.
        elif hasattr(spatial, "bounds") and not hasattr(spatial, "crs"):
            bounds = spatial.bounds
        else:
            from faninsar.remote import _query_geometry

            geometry, _kind, _points = _query_geometry(spatial)
            bounds = geometry.bounds
        return self.search(
            bounds,
            ledger=ledger,
            datetime_range=datetime_range,
            filters=filters,
            limit=limit,
        )

    def search(
        self,
        bounds: tuple[float, float, float, float],
        *,
        ledger: _CallLedger | None = None,
        datetime_range: tuple[datetime, datetime] | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[Mapping[str, Any]]:
        """Discover one WGS84 bbox and return provider-neutral records."""
        records: list[Mapping[str, Any]] = []
        item_stream = self._search_items(
            list(bounds),
            ledger,
            datetime_range=datetime_range,
            filters=filters,
            limit=limit,
        )
        try:
            for raw_item in item_stream:
                raw_mapping = dict(_as_mapping(raw_item))
                # Validate the provider response before invoking an optional
                # signer.  This keeps malformed identity/asset failures typed
                # even when the signer only accepts fully formed STAC items.
                normalize_stac_item(
                    raw_mapping,
                    provider=self.provider,
                    catalog=self.provider,
                    collection=self.collection,
                )
                item = self._sign_item(raw_item)
                item_mapping = dict(_as_mapping(item))
                item_id = item_mapping.get("id")
                assets = item_mapping.get("assets")
                if not isinstance(assets, Mapping):
                    _fail(
                        MalformedSTACItemError,
                        "missing_assets",
                        "Planetary Computer STAC Item must contain assets",
                    )
                asset = assets.get(self.asset_key)
                href = str(asset.get("href", "")) if isinstance(asset, Mapping) else ""
                if not isinstance(item_id, str) or not item_id:
                    _fail(
                        MalformedSTACItemError,
                        "invalid_item_id",
                        "Planetary Computer STAC Item id must be non-empty",
                    )
                if not isinstance(asset, Mapping) or not href:
                    _fail(
                        MalformedSTACItemError,
                        "missing_data_asset",
                        f"STAC Item {item_id!r} has no {self.asset_key!r} asset",
                    )
                unsigned = urllib.parse.urlsplit(href)
                if (
                    unsigned.scheme.lower() != "https"
                    or unsigned.hostname != PC_ASSET_HOST
                ):
                    _fail(RemoteAccessError, "unregistered_endpoint")
                self._signed[(item_id, self.asset_key)] = href
                persisted_assets = dict(assets)
                persisted_asset = dict(asset)
                persisted_asset["href"] = urllib.parse.urlunsplit(
                    (unsigned.scheme, unsigned.netloc, unsigned.path, "", "")
                )
                persisted_assets[self.asset_key] = persisted_asset
                persisted_item = dict(item_mapping)
                persisted_item["collection"] = self.collection
                persisted_item["assets"] = persisted_assets
                # Every item passes the pinned profile normalizer.  In
                # particular, malformed identities, geometry, temporal fields,
                # and any asset (not just the selected DEM asset) fail with
                # typed errors.
                record = normalize_stac_item(
                    persisted_item,
                    provider=self.provider,
                    catalog=self.provider,
                    collection=self.collection,
                )
                records.append(record)
        finally:
            close = getattr(item_stream, "close", None)
            if callable(close):
                close()
        return records

    def fetch(
        self,
        asset: object,
        budget: RemoteResourceBudget,
        *,
        ledger: _CallLedger | None = None,
    ) -> Iterator[bytes]:
        """Stream one complete signed COG using the P0044 accounting seam."""
        item_id = str(getattr(asset, "item_id", ""))
        key = str(getattr(asset, "key", self.asset_key))
        url = self._signed.get((item_id, key), str(getattr(asset, "href", "")))
        unsigned = urllib.parse.urlunsplit((*urllib.parse.urlsplit(url)[:3], "", ""))
        _safe_url(unsigned, self)
        if ledger is None:
            ledger = _CallLedger(budget)
        ledger.request()
        ledger.begin_response()
        request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
        opener = urllib.request.build_opener(_RedirectHandler(self, budget, ledger))
        try:
            response = self._open(opener, request, budget.read_timeout_seconds)
            with response:
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    _fail(RemoteAccessError, "unexpected_content_encoding")
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    ledger.response_bytes(len(chunk))
                    yield bytes(chunk)
        except (urllib.error.URLError, OSError):
            logger.exception("Planetary Computer asset transfer failed")
            # Preserve the standard-library exception so the P0044 public
            # boundary can apply its operation-wide retry policy.
            raise

    def _open(
        self, opener: object, request: urllib.request.Request, timeout: float
    ) -> object:
        """Open a request; isolated for deterministic transfer tests."""
        return opener.open(request, timeout=timeout)  # type: ignore[attr-defined]


def register_planetary_computer(
    name: str = "pc", **kwargs: Any
) -> PlanetaryComputerAdapter:
    """Construct and register one Planetary Computer adapter."""
    return PlanetaryComputerAdapter(**kwargs).register(name)


# Keep naming parallel with the registered CMR collection adapter while
# retaining the concise class used by the DEM integration.
PlanetaryComputerCollectionAdapter = PlanetaryComputerAdapter
register_pc_catalog = register_planetary_computer


__all__ = [
    "COP_DEM_GLO30_COLLECTION",
    "PC_ASSET_HOST",
    "PC_STAC_ENDPOINT",
    "PlanetaryComputerAdapter",
    "PlanetaryComputerCollectionAdapter",
    "register_pc_catalog",
    "register_planetary_computer",
]
