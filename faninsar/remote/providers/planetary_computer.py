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
from dataclasses import dataclass, field
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
        object.__setattr__(
            self, "path_prefixes", (parts.path.rstrip("/") or "/", "/")
        )
        if (
            self.collection != COP_DEM_GLO30_COLLECTION
            or self.asset_key != PC_ASSET_KEY
        ):
            message = (
                "Planetary Computer GLO-30 collection and asset are "
                "registry-owned"
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

    def _client(self) -> object:
        """Return an injected or lazily-created STAC client."""
        if self.client is not None:
            return self.client
        try:
            import pystac_client
        except ImportError as error:
            message = "Planetary Computer support requires pystac-client"
            logger.exception(message)
            try:
                _fail(RemoteAccessError, "missing_optional_dependency", message)
            except RemoteAccessError as raised:
                raise raised from error
        return pystac_client.Client.open(self.endpoint)

    def _search_items(
        self, bbox: list[float], ledger: _CallLedger | None
    ) -> Iterable[object]:
        """Search one WGS84 bbox through an injected or HTTP STAC client."""
        client = self._client()
        search = getattr(client, "search", None)
        if callable(search):
            result = search(collections=[self.collection], bbox=bbox)
            items = getattr(result, "items", None)
            return items() if callable(items) else result
        return self._http_items(bbox, ledger)

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
        request = urllib.request.Request(
            url, headers={"Accept-Encoding": "identity"}
        )
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
        self, *, ledger: _CallLedger | None = None
    ) -> Iterable[Mapping[str, Any]]:
        """Yield normalized PC records, signing each item in memory."""
        # ``remote.search`` supplies no spatial bbox to this adapter protocol;
        # callers that need spatially narrow discovery should use ``search``.
        return self.search((-180.0, -90.0, 180.0, 90.0), ledger=ledger)

    def search(
        self,
        bounds: tuple[float, float, float, float],
        *,
        ledger: _CallLedger | None = None,
    ) -> list[Mapping[str, Any]]:
        """Discover one WGS84 bbox and return provider-neutral records."""
        records: list[Mapping[str, Any]] = []
        for raw_item in self._search_items(list(bounds), ledger):
            item = self._sign_item(raw_item)
            mapping = dict(_as_mapping(item))
            item_id = str(mapping.get("id", ""))
            assets = dict(mapping.get("assets", {}))
            asset = assets.get(self.asset_key)
            href = _signed_href(item, self.asset_key)
            if not item_id or not href or not isinstance(asset, Mapping):
                continue
            unsigned = urllib.parse.urlsplit(href)
            if unsigned.scheme.lower() != "https" or unsigned.hostname != PC_ASSET_HOST:
                _fail(RemoteAccessError, "unregistered_endpoint")
            self._signed[(item_id, self.asset_key)] = href
            record = dict(mapping)
            record["collection"] = self.collection
            assets[self.asset_key] = dict(asset)
            assets[self.asset_key]["href"] = href
            record["assets"] = assets
            records.append(record)
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
