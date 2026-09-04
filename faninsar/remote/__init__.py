"""Provider-neutral catalog search and complete-file download.

The module deliberately owns transport mechanics only.  Domain adapters retain
the meaning of DEMs, scenes, or interferograms and may use the two public
operations without introducing a second scientific object model.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.cookiejar
import importlib
import inspect
import json
import netrc
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, Protocol

import requests
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from faninsar.logging import setup_logger
from faninsar.query import BoundingBox, Points, Polygons

logger = setup_logger(__name__)

__all__ = [
    "AcquisitionMetadata",
    "CatalogItem",
    "RemoteAccessError",
    "RemoteAsset",
    "RemoteIntegrityError",
    "RemoteLimitError",
    "RemoteQueryError",
    "RemoteResourceBudget",
    "download",
    "search",
]

# Provider and profile modules are deliberately loaded only when one of their
# public names is requested.  In particular, ``import faninsar.remote`` is a
# supported offline operation and must not require any optional provider
# package.  Keeping the map here also makes the existing ``catalog=`` registry
# the one public integration point: adapters only register a name, and users
# continue to call :func:`search` and :func:`download`.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_CMR_ENDPOINT": ("faninsar.remote.cmr", "DEFAULT_CMR_ENDPOINT"),
    "CMRAdapter": ("faninsar.remote.cmr", "CMRAdapter"),
    "CMRCollectionAdapter": ("faninsar.remote.cmr", "CMRCollectionAdapter"),
    "CMRDiscoveryError": ("faninsar.remote.cmr", "CMRDiscoveryError"),
    "CMRRegistrationError": ("faninsar.remote.cmr", "CMRRegistrationError"),
    "RegisteredCMRCollection": (
        "faninsar.remote.cmr",
        "RegisteredCMRCollection",
    ),
    "discover_cmr": ("faninsar.remote.cmr", "discover_cmr"),
    "iter_cmr_records": ("faninsar.remote.cmr", "iter_cmr_records"),
    "normalize_cmr_granule": ("faninsar.remote.cmr", "normalize_cmr_granule"),
    "normalize_umm_granule": ("faninsar.remote.cmr", "normalize_umm_granule"),
    "register_cmr_catalog": ("faninsar.remote.cmr", "register_cmr_catalog"),
    "register_cmr_collection": ("faninsar.remote.cmr", "register_cmr_collection"),
    "STAC_CORE_VERSION": ("faninsar.remote.standards", "STAC_CORE_VERSION"),
    "STAC_PROFILES": ("faninsar.remote.standards", "STAC_PROFILES"),
    "STAC_PROFILE_VERSIONS": (
        "faninsar.remote.standards",
        "STAC_PROFILE_VERSIONS",
    ),
    "SUPPORTED_STAC_EXTENSIONS": (
        "faninsar.remote.standards",
        "SUPPORTED_STAC_EXTENSIONS",
    ),
    "SUPPORTED_STAC_PROFILES": (
        "faninsar.remote.standards",
        "SUPPORTED_STAC_PROFILES",
    ),
    "MalformedSTACItemError": (
        "faninsar.remote.standards",
        "MalformedSTACItemError",
    ),
    "STACProfileError": ("faninsar.remote.standards", "STACProfileError"),
    "UnknownSTACProfileError": (
        "faninsar.remote.standards",
        "UnknownSTACProfileError",
    ),
    "map_stac_item": ("faninsar.remote.standards", "map_stac_item"),
    "normalize_stac_item": ("faninsar.remote.standards", "normalize_stac_item"),
    "stac_item_to_record": ("faninsar.remote.standards", "stac_item_to_record"),
    "validate_stac_item": ("faninsar.remote.standards", "validate_stac_item"),
    # ASF and Planetary Computer remain lazy optional provider integrations.
    "ASF_SEARCH_CERTIFIED_VERSION": (
        "faninsar.remote.providers.asf_search",
        "ASF_SEARCH_CERTIFIED_VERSION",
    ),
    "ASF_SEARCH_COMPATIBILITY": (
        "faninsar.remote.providers.asf_search",
        "ASF_SEARCH_COMPATIBILITY",
    ),
    "ASFSearchAdapter": (
        "faninsar.remote.providers.asf_search",
        "ASFSearchAdapter",
    ),
    "ASFSearchEngineAdapter": (
        "faninsar.remote.providers.asf_search",
        "ASFSearchEngineAdapter",
    ),
    "ASFSearchError": ("faninsar.remote.providers.asf_search", "ASFSearchError"),
    "ASFSearchUnavailableError": (
        "faninsar.remote.providers.asf_search",
        "ASFSearchUnavailableError",
    ),
    "ASFSearchVersionError": (
        "faninsar.remote.providers.asf_search",
        "ASFSearchVersionError",
    ),
    "AsfSearchAdapter": (
        "faninsar.remote.providers.asf_search",
        "AsfSearchAdapter",
    ),
    "EngineUnavailableError": (
        "faninsar.remote.providers.asf_search",
        "EngineUnavailableError",
    ),
    "UnobservableASFSearchError": (
        "faninsar.remote.providers.asf_search",
        "UnobservableASFSearchError",
    ),
    "UnsupportedASFSearchVersionError": (
        "faninsar.remote.providers.asf_search",
        "UnsupportedASFSearchVersionError",
    ),
    "register_asf_catalog": (
        "faninsar.remote.providers.asf_search",
        "register_asf_catalog",
    ),
    "register_asf_search_catalog": (
        "faninsar.remote.providers.asf_search",
        "register_asf_search_catalog",
    ),
    "COP_DEM_GLO30_COLLECTION": (
        "faninsar.remote.providers.planetary_computer",
        "COP_DEM_GLO30_COLLECTION",
    ),
    "PC_ASSET_HOST": (
        "faninsar.remote.providers.planetary_computer",
        "PC_ASSET_HOST",
    ),
    "PC_STAC_ENDPOINT": (
        "faninsar.remote.providers.planetary_computer",
        "PC_STAC_ENDPOINT",
    ),
    "PlanetaryComputerAdapter": (
        "faninsar.remote.providers.planetary_computer",
        "PlanetaryComputerAdapter",
    ),
    "PlanetaryComputerCollectionAdapter": (
        "faninsar.remote.providers.planetary_computer",
        "PlanetaryComputerCollectionAdapter",
    ),
    "register_pc_catalog": (
        "faninsar.remote.providers.planetary_computer",
        "register_pc_catalog",
    ),
    "register_planetary_computer": (
        "faninsar.remote.providers.planetary_computer",
        "register_planetary_computer",
    ),
}
__all__ += list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve profile and provider exports without eager optional imports."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


class RemoteError(Exception):
    """Base class for errors raised by the remote boundary."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        """Initialize an error with a stable machine-readable reason."""
        self.reason = reason
        super().__init__(message or reason)


def _fail(
    error_type: type[RemoteError], reason: str, message: str | None = None
) -> NoReturn:
    """Raise a remote error while keeping reason construction centralized."""
    raise error_type(reason, message)


class RemoteQueryError(RemoteError):
    """A spatial query or catalog selection is invalid."""


class RemoteAccessError(RemoteError):
    """A catalog, profile, endpoint, or transfer is unavailable."""


class RemoteLimitError(RemoteError):
    """A finite operation budget was exceeded."""


class RemoteIntegrityError(RemoteError):
    """Transferred bytes or a published destination failed validation."""


@dataclass(frozen=True, slots=True)
class RemoteResourceBudget:
    """Finite limits for remote search and complete-file transfer.

    Parameters
    ----------
    max_items : int
        Maximum number of catalog items returned.
    max_requests : int
        Maximum number of transfer attempts.
    max_response_bytes : int
        Maximum bytes accepted from one response.
    max_output_bytes : int
        Maximum bytes written for one asset.
    connect_timeout_seconds, read_timeout_seconds : float
        Standard-library request timeout values.

    """

    max_items: int = 100
    max_requests: int = 256
    max_redirects: int = 5
    max_retries: int = 4
    max_elapsed_seconds: float = 900.0
    max_workers: int = 8
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 120.0
    max_response_bytes: int = 2**31
    max_operation_bytes: int = 2**33
    max_output_bytes: int = 2**31
    max_temporary_bytes: int = 2**33
    max_cache_bytes: int = 2**33

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive resource limits."""
        for name in (
            "max_items",
            "max_requests",
            "max_redirects",
            "max_retries",
            "max_workers",
            "max_response_bytes",
            "max_operation_bytes",
            "max_output_bytes",
            "max_temporary_bytes",
            "max_cache_bytes",
        ):
            if getattr(self, name) <= 0:
                msg = f"{name} must be positive"
                raise ValueError(msg)
        for name in (
            "max_elapsed_seconds",
            "connect_timeout_seconds",
            "read_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                msg = f"{name} must be positive"
                raise ValueError(msg)


def _freeze(value: Any) -> Any:
    """Recursively make provider metadata immutable and JSON-compatible."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_SECRET_KEY = re.compile(
    r"(?:authorization|cookie|token|password|passwd|secret|signature|credential|api.?key)",
    re.IGNORECASE,
)
_SIGNED_QUERY_KEY = re.compile(
    r"^(?:sig(?:nature)?|token|expires?|se(?:curity)?|x-amz-|authorization|"
    r"sp|st|sv|sr|spr|sip|si|skoid|sktid|skt|ske|sks|skv|rscc|rscd|rsce|rscl|rsct|"
    r"ss|srt|sdd)$",
    re.IGNORECASE,
)
_AZURE_SAS_KEY = re.compile(
    r"^(?:sp|st|se|sv|sr|spr|sip|si|sig|skoid|sktid|skt|ske|sks|skv|"
    r"rscc|rscd|rsce|rscl|rsct|ss|srt|sdd)$",
    re.IGNORECASE,
)
_ASF_EDL_ORIGIN = "https://urs.earthdata.nasa.gov"
_ASF_AUTH_ORIGIN = "https://cumulus.asf.alaska.edu"
_ASF_DATA_ORIGIN = "https://datapool.asf.alaska.edu"
_ASF_SENTINEL1_ORIGIN = "https://sentinel1.asf.alaska.edu"
_ASF_NISAR_ORIGIN = "https://nisar.asf.earthdatacloud.nasa.gov"
_ASF_NISAR_BUCKET = "sds-n-cumulus-prod-nisar-products.s3.us-west-2.amazonaws.com"
_ASF_EDL_CLIENT_ID = "BO_n7nTIlMljdvU6kRRB3g"
_ASF_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_ASF_CLOUDFRONT_HOST = re.compile(r"^[a-z0-9]+\.cloudfront\.net$")
_ASF_SLC_CLOUDFRONT_PATH = re.compile(
    r"^/s3-[^/]+/asf-ngap2w-p-s1-slc-[a-z0-9]+\."
    r"s3\.us-west-2\.amazonaws\.com/[^/]+\.zip$",
    re.IGNORECASE,
)
_ASF_NISAR_CLOUDFRONT_PATH = re.compile(
    r"^/s3-[^/]+/sds-n-cumulus-prod-nisar-products\.s3\.us-west-2\.amazonaws\.com"
    r"(?P<key>/DEM/v1\.2/.+)$",
    re.IGNORECASE,
)
_ASF_NISAR_GATEWAY_PREFIX = "/NISAR"
_LPDAAC_DATA_ORIGIN = "https://data.lpdaac.earthdatacloud.nasa.gov"
_LPDAAC_BUCKET = "lp-prod-protected"
_LPDAAC_BUCKET_HOST = "lp-prod-protected.s3.us-west-2.amazonaws.com"
_LPDAAC_CLOUDFRONT_PATH = re.compile(
    rf"^/s3-[0-9a-f]{{32}}/{re.escape(_LPDAAC_BUCKET_HOST)}"
    r"(?P<key>/[^/].*)$"
)


def _sanitize(value: Any, key: str | None = None) -> Any:  # noqa: PLR0911
    """Remove credential-bearing values from nested provider data."""
    if key is not None and (_SECRET_KEY.search(key) or _SIGNED_QUERY_KEY.search(key)):
        return None
    if isinstance(value, Mapping):
        return {
            str(k): _sanitize(v, str(k))
            for k, v in value.items()
            if not (_SECRET_KEY.search(str(k)) or _SIGNED_QUERY_KEY.search(str(k)))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme and parsed.netloc:
            if parsed.username or parsed.password:
                return urllib.parse.urlunsplit(
                    (parsed.scheme, parsed.hostname or "", parsed.path, "", "")
                )
            query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            # An Azure SAS URL signs the complete query string.  Retaining an
            # unrecognised parameter alongside the SAS fields can still leak
            # signed request material, so drop the entire query whenever a
            # SAS marker is present.  The provider may retain a request-local
            # signed URL separately for the immediate transfer.
            if any(_AZURE_SAS_KEY.fullmatch(k) for k, _ in query):
                return urllib.parse.urlunsplit(
                    (parsed.scheme, parsed.netloc, parsed.path, "", "")
                )
            query = [(k, v) for k, v in query if not _SIGNED_QUERY_KEY.search(k)]
            clean_query = urllib.parse.urlencode(query)
            return urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, clean_query, "")
            )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _utc(value: Any) -> datetime | None:
    """Normalize an optional datetime to timezone-aware UTC."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            _fail(RemoteQueryError, "invalid_datetime", str(exc))
    if not isinstance(value, datetime):
        msg = "acquisition time is not a datetime"
        _fail(RemoteQueryError, "invalid_datetime", msg)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class AcquisitionMetadata:
    """Small normalized description of an acquisition."""

    acquisition_id: str
    start: datetime | None = None
    end: datetime | None = None
    platform: str | None = None
    instrument: str | None = None
    mode: str | None = None
    orbit: int | None = None
    processing_level: str | None = None
    geometry: Mapping[str, object] | None = None
    properties: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize times and protect nested metadata from mutation."""
        object.__setattr__(self, "start", _utc(self.start))
        object.__setattr__(self, "end", _utc(self.end))
        if self.start is not None and self.end is not None and self.end < self.start:
            _fail(RemoteQueryError, "invalid_acquisition_interval")
        geometry = (
            _freeze(_sanitize(self.geometry)) if self.geometry is not None else None
        )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "properties", _freeze(_sanitize(self.properties)))


@dataclass(frozen=True, slots=True)
class RemoteAsset:
    """An immutable, registered remote asset descriptor."""

    provider: str
    catalog: str
    collection: str | None
    item_id: str
    key: str
    href: str
    media_type: str | None = None
    size_bytes: int | None = None
    checksum: str | None = None
    version: str | None = None
    representation: str = "complete"
    auth_profile: str = "anonymous"
    properties: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize checksum and protect asset metadata from mutation."""
        object.__setattr__(self, "checksum", _normalize_checksum(self.checksum))
        # The request-local signed URL, when one exists, belongs to the
        # provider adapter.  A public asset descriptor must never retain its
        # query credentials (including Azure SAS fields).
        object.__setattr__(self, "href", _sanitize(self.href))
        object.__setattr__(self, "properties", _freeze(_sanitize(self.properties)))


@dataclass(frozen=True, slots=True)
class CatalogItem:
    """An immutable normalized catalog item and its assets."""

    provider: str
    catalog: str
    collection: str | None
    item_id: str
    acquisition: AcquisitionMetadata
    assets: Mapping[str, RemoteAsset]
    raw_metadata: Mapping[str, object]
    matched_point_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Protect asset and metadata mappings from mutation."""
        object.__setattr__(self, "assets", MappingProxyType(dict(self.assets)))
        object.__setattr__(self, "raw_metadata", _freeze(_sanitize(self.raw_metadata)))
        object.__setattr__(
            self, "matched_point_indices", tuple(self.matched_point_indices)
        )


class _Adapter(Protocol):
    """Private catalog adapter protocol."""

    provider: str
    origins: tuple[str, ...]
    path_prefixes: tuple[str, ...]
    redirect_origins: tuple[str, ...]
    redirect_path_prefixes: Mapping[str, tuple[str, ...]]
    profiles: tuple[str, ...]

    def items(
        self,
        *,
        spatial: Any | None = None,
        spatial_kind: str | None = None,
        point_geometries: tuple[Any, ...] = (),
        datetime_range: tuple[datetime, datetime] | None = None,
        collections: tuple[str, ...] | None = None,
        auth_profile: str = "anonymous",
        limit: int = 100,
        budget: RemoteResourceBudget | None = None,
        ledger: _CallLedger | None = None,
    ) -> Iterable[Mapping[str, Any]]: ...

    def fetch(
        self,
        asset: RemoteAsset,
        budget: RemoteResourceBudget,
        *,
        ledger: _CallLedger | None = None,
    ) -> Any: ...


@dataclass(slots=True)
class _CallLedger:
    """Private meter for exactly one public remote operation.

    Adapters receive this object only for the duration of one ``search`` or
    ``download`` call.  Provider code must charge every request, retry,
    redirect, and response byte it performs through the corresponding
    methods.  The public API intentionally exposes neither this object nor a
    cross-call session.
    """

    budget: RemoteResourceBudget
    started: float = 0.0
    requests: int = 0
    retries: int = 0
    redirects: int = 0
    response_bytes_total: int = 0
    _response_bytes: int = 0

    def __post_init__(self) -> None:
        """Capture the operation start before provider work begins."""
        self.started = time.monotonic()

    def check_elapsed(self) -> None:
        """Enforce the operation-wide elapsed-time limit."""
        if time.monotonic() - self.started > self.budget.max_elapsed_seconds:
            _fail(RemoteLimitError, "max_elapsed_seconds")

    def request(self) -> None:
        """Charge one provider request."""
        self.requests += 1
        if self.requests > self.budget.max_requests:
            _fail(RemoteLimitError, "max_requests")

    def retry(self) -> None:
        """Charge one retry before the next request attempt."""
        self.check_elapsed()
        self.retries += 1
        if self.retries > self.budget.max_retries:
            _fail(RemoteLimitError, "max_retries")

    def redirect(self) -> None:
        """Charge one redirect followed by a provider request."""
        self.check_elapsed()
        self.redirects += 1
        if self.redirects > self.budget.max_redirects:
            _fail(RemoteLimitError, "max_redirects")

    def begin_response(self) -> None:
        """Start accounting for one response body."""
        self._response_bytes = 0

    def response_bytes(self, count: int) -> None:
        """Charge bytes from the current response body.

        Parameters
        ----------
        count : int
            Number of newly consumed response bytes.

        """
        if count < 0:
            msg = "response byte count must be non-negative"
            raise ValueError(msg)
        self._response_bytes += count
        self.response_bytes_total += count
        if self._response_bytes > self.budget.max_response_bytes:
            _fail(RemoteLimitError, "max_response_bytes")
        if self.response_bytes_total > self.budget.max_operation_bytes:
            _fail(RemoteLimitError, "max_operation_bytes")
        self.check_elapsed()


def _accepts_ledger(method: Any) -> bool:
    """Return whether an adapter method accepts the private ledger keyword."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == "ledger"
        for parameter in parameters
    )


def _adapter_items(
    adapter: _Adapter,
    *,
    spatial: Any,
    spatial_kind: str,
    point_geometries: tuple[Any, ...],
    datetime_range: tuple[datetime, datetime] | None,
    collections: tuple[str, ...] | None,
    auth_profile: str,
    limit: int,
    budget: RemoteResourceBudget,
    ledger: _CallLedger,
) -> Iterable[Mapping[str, Any]]:
    """Invoke an adapter with the normalized query it advertises.

    The registry is private, but adapters from the preceding remote MVP are
    intentionally kept source-compatible.  Keyword arguments are therefore
    limited to parameters accepted by the concrete method; an adapter that
    exposes ``**kwargs`` receives the complete query contract.
    """
    method = adapter.items
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_any = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    values: dict[str, Any] = {
        "spatial": spatial,
        "spatial_kind": spatial_kind,
        "point_geometries": point_geometries,
        "datetime_range": datetime_range,
        "collections": collections,
        "auth_profile": auth_profile,
        "limit": limit,
        "budget": budget,
        "ledger": ledger,
    }
    kwargs = (
        values
        if accepts_any
        else {name: value for name, value in values.items() if name in parameters}
    )
    if "ledger" not in kwargs:
        # P0044 adapters predate the private ledger keyword.  Their one
        # catalog operation still receives a conservative request charge.
        ledger.request()
    return method(**kwargs)


@dataclass
class _FixtureAdapter:
    """Private in-memory adapter used by contract tests and examples."""

    records: list[Mapping[str, Any]]
    provider: str = "fixture"
    origins: tuple[str, ...] = ("https://fixture.invalid",)
    path_prefixes: tuple[str, ...] = ("/",)
    redirect_origins: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ("anonymous",)

    def items(self) -> Iterable[Mapping[str, Any]]:
        """Return fixture records."""
        return self.records

    def fetch(self, asset: RemoteAsset, budget: RemoteResourceBudget) -> bytes:
        """Return fixture bytes associated with an asset."""
        del budget
        for record in self.records:
            if str(record.get("id", record.get("item_id"))) != asset.item_id:
                continue
            for key, value in dict(record.get("assets", {})).items():
                if key == asset.key:
                    payload = value.get("data", value.get("content", b""))
                    if isinstance(payload, str):
                        payload = payload.encode()
                    return bytes(payload)
        _fail(RemoteAccessError, "asset_not_found")


_ADAPTERS: dict[str, _Adapter] = {}
_DESTINATION_LOCKS: dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _register_fixture(
    records: Iterable[Mapping[str, Any]],
    *,
    name: str = "fixture",
    provider: str = "fixture",
) -> None:
    """Register an in-memory catalog for local contract tests.

    This helper is intentionally private; production providers register their
    adapters in their own integration work, not through the public API.
    """
    adapter = _FixtureAdapter(list(records), provider=provider)
    with _REGISTRY_LOCK:
        _ADAPTERS[name] = adapter


def _register_adapter(name: str, adapter: _Adapter) -> None:
    """Register a private catalog adapter."""
    if not name or not isinstance(name, str):
        msg = "catalog name must be a non-empty string"
        raise ValueError(msg)
    with _REGISTRY_LOCK:
        if name in _ADAPTERS:
            msg = f"catalog {name!r} is already registered"
            raise ValueError(msg)
        _ADAPTERS[name] = adapter


def _query_geometry(
    spatial: Points | BoundingBox | Polygons,
) -> tuple[Any, str, tuple[Any, ...]]:
    """Convert a FanInSAR query to WGS84 Shapely geometry."""
    crs = getattr(spatial, "crs", None)
    if crs is None:
        _fail(RemoteQueryError, "missing_crs", "spatial queries must declare a CRS")
    try:
        source = CRS.from_user_input(crs)
        transformer = Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)
    except (CRSError, TypeError, ValueError) as exc:
        _fail(RemoteQueryError, "invalid_crs", str(exc))

    def project(geometry: Any) -> Any:
        """Project one geometry to WGS84."""
        return shapely_transform(transformer.transform, geometry)

    if isinstance(spatial, Points):
        point_geometries = tuple(project(Point(x, y)) for x, y in spatial.values)
        return unary_union(point_geometries), "points", point_geometries
    if isinstance(spatial, BoundingBox):
        return project(box(*spatial)), "bbox", ()
    if isinstance(spatial, Polygons):
        desired = [
            project(g)
            for g, typ in zip(spatial.geometry, spatial.types, strict=True)
            if typ == "desired"
        ]
        undesired = [
            project(g)
            for g, typ in zip(spatial.geometry, spatial.types, strict=True)
            if typ == "undesired"
        ]
        if not desired:
            _fail(RemoteQueryError, "missing_desired_geometry")
        excluded = unary_union(undesired) if undesired else None
        geometry = (
            unary_union(desired).difference(excluded)
            if excluded
            else unary_union(desired)
        )
        return geometry, "polygons", ()
    msg = "expected Points, BoundingBox, or Polygons"
    _fail(RemoteQueryError, "unsupported_geometry", msg)


def _url_origin(url: str) -> str:
    """Return the normalized origin for an HTTPS URL."""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname
    if not host:
        message = "URL has no hostname"
        raise ValueError(message)
    port = parsed.port
    return f"https://{host.lower()}" + (f":{port}" if port and port != 443 else "")


def _validate_url(
    url: str,
    adapter: _Adapter,
    *,
    redirect: bool = False,
) -> str:
    """Validate a URL while preserving request-local query credentials.

    Query strings are intentionally not inspected or rewritten here.  A
    provider may need a short-lived signed query on an immediate transfer or
    redirect.  Call :func:`_safe_url` at persistence boundaries to obtain the
    scrubbed canonical representation instead.
    """
    if not isinstance(url, str) or not url:
        _fail(RemoteAccessError, "invalid_endpoint")
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        _fail(RemoteAccessError, "invalid_endpoint")
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        _fail(RemoteAccessError, "invalid_endpoint")
    path_lower = parsed.path.lower()
    if "\\" in parsed.path or "%2f" in path_lower or "%5c" in path_lower:
        _fail(RemoteAccessError, "invalid_endpoint")
    segments = urllib.parse.unquote(parsed.path).split("/")
    if any(segment in {".", ".."} for segment in segments):
        _fail(RemoteAccessError, "invalid_endpoint")
    try:
        origin = _url_origin(url)
    except ValueError:
        _fail(RemoteAccessError, "invalid_endpoint")
    registered_origins: list[str] = []
    origins = getattr(adapter, "redirect_origins", ()) if redirect else adapter.origins
    for item in origins:
        try:
            registered_origin = _url_origin(item)
        except ValueError:
            continue
        registered_origins.append(registered_origin)
    if origin not in registered_origins:
        _fail(RemoteAccessError, "unregistered_endpoint")
    path = parsed.path or "/"
    origin_prefixes = getattr(adapter, "redirect_path_prefixes", {})
    scoped_redirect_paths = (
        redirect
        and isinstance(origin_prefixes, Mapping)
        and origin in origin_prefixes
    )
    prefixes = (
        origin_prefixes.get(origin, adapter.path_prefixes)
        if scoped_redirect_paths
        else adapter.path_prefixes
    ) or ("/",)
    allowed_path = False
    for prefix in prefixes:
        normalized_prefix = prefix.rstrip("/") or "/"
        # An explicitly scoped root is an exact root route.  The unscoped
        # default ``/`` remains a wildcard for registered data origins.
        path_matches = (
            path == normalized_prefix
            if scoped_redirect_paths
            else normalized_prefix in {"/", path}
            or path.startswith(normalized_prefix + "/")
        )
        if path_matches:
            allowed_path = True
            break
    if not allowed_path:
        _fail(RemoteAccessError, "unregistered_endpoint")
    # Return the original URL, including its query, for immediate network use.
    return url


def _safe_url(
    url: str,
    adapter: _Adapter,
    *,
    redirect: bool = False,
) -> str:
    """Validate and return a scrubbed canonical URL for persistence.

    This compatibility wrapper retains the historical private helper name for
    provider adapters.  Network redirect code uses :func:`_validate_url`
    directly so signed ``Location`` queries survive the hop.
    """
    validated = _validate_url(url, adapter, redirect=redirect)
    parsed = urllib.parse.urlsplit(validated)
    origin = _url_origin(validated)
    return urllib.parse.urlunsplit(
        ("https", origin.removeprefix("https://"), parsed.path or "/", "", "")
    )


def _canonicalize_url(
    url: str,
    adapter: _Adapter,
    *,
    redirect: bool = False,
) -> str:
    """Return the persistence-safe URL after endpoint validation."""
    return _safe_url(url, adapter, redirect=redirect)


def _strip_redirect_credentials(
    request: urllib.request.Request,
) -> None:
    """Remove credential headers before an approved cross-origin redirect."""
    for name in list(request.headers):
        if _SECRET_KEY.search(str(name)):
            del request.headers[name]


def _redirect_headers(
    headers: Mapping[str, str], source_origin: str, target_origin: str
) -> dict[str, str]:
    """Apply the minimal credential policy to one cross-origin hop.

    Bearer authentication is retained only for the ASF datapool-to-Sentinel-1
    handoff.  In particular, Basic Earthdata Login credentials never cross
    URS or reach a CloudFront object URL.
    """
    filtered = {
        key: value for key, value in headers.items() if not _SECRET_KEY.search(key)
    }
    authorization = next(
        (value for key, value in headers.items() if key.lower() == "authorization"),
        None,
    )
    if (
        source_origin == _ASF_DATA_ORIGIN
        and target_origin == _ASF_SENTINEL1_ORIGIN
        and isinstance(authorization, str)
        and authorization.lower().startswith("bearer ")
    ):
        filtered["Authorization"] = authorization
    return filtered


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


def _no_auth(request: requests.PreparedRequest) -> requests.PreparedRequest:
    """Prevent implicit netrc lookup without changing requests' environment."""
    return request


def _asf_redirect_url(
    source_url: str,
    target_url: str,
    asset: RemoteAsset,
    adapter: _Adapter,
) -> str:
    """Validate one ASF redirect, including its scoped CloudFront handoff.

    ASF's Sentinel-1 service issues a short-lived CloudFront URL whose host is
    deployment-specific.  That URL is admitted only when it comes directly
    from the registered Sentinel-1 origin, names the approved ASF SLC bucket,
    and retains the exact SAFE ZIP filename selected during discovery.
    """
    try:
        return _validate_url(target_url, adapter, redirect=True)
    except RemoteAccessError:
        pass
    try:
        source_origin = _url_origin(source_url)
        source_path = urllib.parse.unquote(
            urllib.parse.urlsplit(source_url).path or "/"
        )
        parsed = urllib.parse.urlsplit(target_url)
        host = (parsed.hostname or "").lower()
        source_name = urllib.parse.unquote(
            urllib.parse.urlsplit(asset.href).path.rsplit("/", 1)[-1]
        )
        target_name = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
        asset_path = urllib.parse.unquote(urllib.parse.urlsplit(asset.href).path or "/")
        target_path = urllib.parse.unquote(parsed.path)
    except (TypeError, ValueError):
        _fail(RemoteAccessError, "invalid_endpoint")
    is_sentinel1 = source_origin == _ASF_SENTINEL1_ORIGIN
    is_nisar = source_origin == _ASF_NISAR_ORIGIN
    nisar_key = (
        source_path.removeprefix(_ASF_NISAR_GATEWAY_PREFIX)
        if source_path.startswith(f"{_ASF_NISAR_GATEWAY_PREFIX}/DEM/v1.2/")
        else None
    )
    nisar_target = (
        _ASF_NISAR_CLOUDFRONT_PATH.fullmatch(urllib.parse.unquote(parsed.path))
        if is_nisar
        else None
    )
    if (
        not (is_sentinel1 or is_nisar)
        or parsed.scheme.lower() != "https"
        or not _ASF_CLOUDFRONT_HOST.fullmatch(host)
        or parsed.username
        or parsed.password
        or parsed.fragment
        or "\\" in parsed.path
        or "%2f" in parsed.path.lower()
        or "%5c" in parsed.path.lower()
        or any(segment in {".", ".."} for segment in target_path.split("/"))
        or not (
            _ASF_SLC_CLOUDFRONT_PATH.fullmatch(parsed.path)
            if is_sentinel1
            else nisar_target is not None
            and nisar_key is not None
            and source_path == asset_path
            and nisar_target.group("key") == nisar_key
        )
        or not source_name
        or target_name != source_name
    ):
        _fail(RemoteAccessError, "unregistered_endpoint")
    return target_url


def _lpdaac_redirect_url(
    source_url: str,
    target_url: str,
    asset: RemoteAsset,
    adapter: _Adapter,
) -> str:
    """Validate one LPDAAC redirect, including its scoped CloudFront hop.

    LPDAAC's Earthdata Cloud delivery authenticates through URS and returns a
    short-lived CloudFront URL for the original object.  The CDN hostname is
    deployment-specific, so it is admitted only when the redirect originated
    at the registered LPDAAC data URL and the target names the exact S3 key
    and filename selected during CMR discovery.
    """
    try:
        return _validate_url(target_url, adapter, redirect=True)
    except RemoteAccessError:
        pass
    try:
        source_origin = _url_origin(source_url)
        source_path = urllib.parse.unquote(
            urllib.parse.urlsplit(source_url).path or "/"
        )
        asset_path = urllib.parse.unquote(urllib.parse.urlsplit(asset.href).path or "/")
        parsed = urllib.parse.urlsplit(target_url)
        target_path = urllib.parse.unquote(parsed.path)
        target_host = (parsed.hostname or "").lower()
        target_match = _LPDAAC_CLOUDFRONT_PATH.fullmatch(target_path)
    except (TypeError, ValueError):
        _fail(RemoteAccessError, "invalid_endpoint")
    source_key = asset_path.removeprefix(f"/{_LPDAAC_BUCKET}")
    target_key = target_match.group("key") if target_match is not None else None
    target_name = target_key.rsplit("/", 1)[-1] if target_key else ""
    source_name = source_key.rsplit("/", 1)[-1]
    if (
        source_origin != _LPDAAC_DATA_ORIGIN
        or source_path != asset_path
        or not asset_path.startswith(f"/{_LPDAAC_BUCKET}/")
        or parsed.scheme.lower() != "https"
        or not _ASF_CLOUDFRONT_HOST.fullmatch(target_host)
        or parsed.username
        or parsed.password
        or parsed.fragment
        or "\\" in parsed.path
        or "%2f" in parsed.path.lower()
        or "%5c" in parsed.path.lower()
        or any(segment in {".", ".."} for segment in target_path.split("/"))
        or target_match is None
        or target_key != source_key
        or not source_name
        or target_name != source_name
    ):
        _fail(RemoteAccessError, "unregistered_endpoint")
    return target_url


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
            auth=None,
            allow_redirects=False,
            stream=True,
            timeout=(
                budget.connect_timeout_seconds,
                budget.read_timeout_seconds,
            ),
        )
        if response.status_code not in _ASF_REDIRECT_CODES:
            return response
        location = response.headers.get("Location")
        if not location:
            _drain_asf_response(response, ledger, budget.max_response_bytes)
            response.close()
            _fail(RemoteAccessError, "invalid_redirect")
        target = urllib.parse.urljoin(current_url, location)
        _drain_asf_response(response, ledger, budget.max_response_bytes)
        response.close()
        ledger.redirect()
        target = _asf_redirect_url(current_url, target, asset, adapter)
        try:
            source_origin = _url_origin(current_url)
            target_origin = _url_origin(target)
        except ValueError:
            _fail(RemoteAccessError, "invalid_endpoint")
        if source_origin != target_origin:
            current_headers = _redirect_headers(
                current_headers, source_origin, target_origin
            )
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
    credentials = _netrc_credentials("urs.earthdata.nasa.gov")
    if credentials is None:
        _fail(RemoteAccessError, "missing_earthdata_credentials")
    username, password = credentials
    basic = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
    current_method = method.upper()
    current_url = _validate_url(url, adapter, redirect=True)
    current_headers: dict[str, str] = {"Accept-Encoding": "identity"}
    current_data: Mapping[str, str] | None = None
    while True:
        current_origin = _url_origin(current_url)
        if current_origin == _ASF_EDL_ORIGIN:
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
            return response
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
            target = _lpdaac_redirect_url(current_url, target, asset, adapter)
        source_origin = _url_origin(current_url)
        target_origin = _url_origin(target)
        if source_origin != target_origin:
            current_headers = _redirect_headers(
                current_headers, source_origin, target_origin
            )
        if target_origin == _ASF_EDL_ORIGIN:
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
    credentials = _netrc_credentials("urs.earthdata.nasa.gov")
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
        target = _validate_url(
            urllib.parse.urljoin(req.full_url, newurl),
            self._adapter,
            redirect=True,
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
        return redirected


def _normalize_checksum(value: Any) -> str | None:
    """Accept only the qualified checksum grammar used by the cache."""
    if not isinstance(value, str) or ":" not in value:
        return None
    algorithm, digest = value.split(":", 1)
    algorithm = algorithm.lower()
    if algorithm not in {"sha256", "sha512"} or not re.fullmatch(
        r"[0-9a-fA-F]+", digest
    ):
        return None
    expected = 64 if algorithm == "sha256" else 128
    return f"{algorithm}:{digest.lower()}" if len(digest) == expected else None


def _normalize_record(
    record: Mapping[str, Any], catalog: str, adapter: _Adapter, profile: str
) -> CatalogItem:
    """Normalize one adapter record into immutable public records."""
    item_id = str(record.get("id", record.get("item_id", "")))
    if not item_id:
        _fail(RemoteQueryError, "invalid_catalog_item")
    collection = record.get("collection")
    geometry = record.get("geometry")
    if geometry is None:
        _fail(RemoteQueryError, "missing_footprint")
    footprint = (
        mapping(shape(geometry)) if not isinstance(geometry, Mapping) else geometry
    )
    safe_footprint = _sanitize(footprint)
    acquisition_data = record.get("acquisition", {})
    acquisition = AcquisitionMetadata(
        acquisition_id=str(acquisition_data.get("id", item_id)),
        start=_utc(acquisition_data.get("start", record.get("start"))),
        end=_utc(acquisition_data.get("end", record.get("end"))),
        platform=acquisition_data.get("platform"),
        instrument=acquisition_data.get("instrument"),
        mode=acquisition_data.get("mode"),
        orbit=acquisition_data.get("orbit"),
        processing_level=acquisition_data.get("processing_level"),
        geometry=_freeze(safe_footprint),
        properties=_freeze(_sanitize(acquisition_data.get("properties", {}))),
    )
    assets: dict[str, RemoteAsset] = {}
    for key, value in dict(record.get("assets", {})).items():
        href = _canonicalize_url(str(value.get("href", "")), adapter)
        assets[str(key)] = RemoteAsset(
            provider=adapter.provider,
            catalog=catalog,
            collection=str(collection) if collection is not None else None,
            item_id=item_id,
            key=str(key),
            href=href,
            media_type=value.get("media_type", value.get("type")),
            size_bytes=value.get("size_bytes", value.get("size")),
            checksum=_normalize_checksum(value.get("checksum")),
            version=value.get("version"),
            representation=str(value.get("representation", "complete")),
            auth_profile=profile,
            properties=_freeze(_sanitize(value.get("properties", {}))),
        )
    return CatalogItem(
        provider=adapter.provider,
        catalog=catalog,
        collection=str(collection) if collection is not None else None,
        item_id=item_id,
        acquisition=acquisition,
        assets=MappingProxyType(assets),
        raw_metadata=_freeze(_sanitize(record)),
    )


def search(
    spatial: Points | BoundingBox | Polygons,
    *,
    catalog: str,
    collections: Sequence[str] | None = None,
    datetime_range: tuple[datetime, datetime] | None = None,
    auth_profile: str | None = None,
    limit: int = 100,
    budget: RemoteResourceBudget | None = None,
) -> list[CatalogItem]:
    """Search a registered catalog using exact FanInSAR spatial semantics."""
    budget = budget or RemoteResourceBudget()
    if limit <= 0 or limit > budget.max_items:
        _fail(RemoteQueryError, "invalid_limit")
    query, query_kind, point_geometries = _query_geometry(spatial)
    adapter = _ADAPTERS.get(catalog)
    if adapter is None:
        _fail(RemoteAccessError, "unknown_catalog")
    profile = auth_profile or "anonymous"
    if profile not in adapter.profiles:
        _fail(RemoteAccessError, "unknown_auth_profile")
    if datetime_range is not None:
        if len(datetime_range) != 2:
            _fail(RemoteQueryError, "invalid_datetime_range")
        start_end = (_utc(datetime_range[0]), _utc(datetime_range[1]))
        if start_end[0] is None or start_end[1] is None:
            _fail(RemoteQueryError, "invalid_datetime_range")
        if start_end[1] < start_end[0]:
            _fail(RemoteQueryError, "invalid_datetime_range")
    else:
        start_end = None
    ledger = _CallLedger(budget)
    raw_items = _adapter_items(
        adapter,
        spatial=query,
        spatial_kind=query_kind,
        point_geometries=point_geometries,
        datetime_range=start_end,
        collections=tuple(collections) if collections is not None else None,
        auth_profile=profile,
        limit=limit,
        budget=budget,
        ledger=ledger,
    )
    results: list[CatalogItem] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for raw in raw_items:
        ledger.check_elapsed()
        if collections is not None and raw.get("collection") not in collections:
            continue
        item = _normalize_record(raw, catalog, adapter, profile)
        identity = (item.provider, item.catalog, item.collection, item.item_id)
        if identity in seen:
            continue
        candidate = shape(item.acquisition.geometry or {})
        if not candidate.intersects(query):
            continue
        if start_end:
            acquired = item.acquisition.start
            if acquired is None or acquired < start_end[0] or acquired > start_end[1]:
                continue
        matched = tuple(
            i for i, point in enumerate(point_geometries) if candidate.intersects(point)
        )
        if query_kind == "points" and not matched:
            continue
        item = CatalogItem(
            item.provider,
            item.catalog,
            item.collection,
            item.item_id,
            item.acquisition,
            item.assets,
            item.raw_metadata,
            matched,
        )
        results.append(item)
        seen.add(identity)
        if len(results) >= limit:
            break
    return results


def _identity(asset: RemoteAsset) -> str:
    """Create a stable cache identity excluding mutable URL details."""
    fields = [
        asset.provider,
        asset.catalog,
        asset.collection,
        asset.item_id,
        asset.key,
        asset.version,
        asset.representation,
        asset.media_type,
        asset.checksum,
        asset.auth_profile,
    ]
    encoded = json.dumps(fields, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _manifest_path(path: Path, asset: RemoteAsset) -> Path:
    """Return a private manifest path qualified by the asset identity."""
    return path.with_name(f"{path.name}.faninsar.remote.{_identity(asset)}.json")


def _is_qualified(asset: RemoteAsset) -> bool:
    """Return whether an asset has a stable version or content checksum."""
    return asset.checksum is not None or asset.version is not None


def _matching_manifest(path: Path, asset: RemoteAsset) -> bool:
    """Return whether an existing destination is a valid qualified reuse."""
    manifest_path = _manifest_path(path, asset)
    if not path.is_file() or not manifest_path.is_file():
        return False
    if not _is_qualified(asset):
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    if manifest.get("identity") != _identity(asset):
        return False
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return False
    return manifest == {"identity": _identity(asset), "sha256": digest, "size": size}


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
    # Earthdata authentication completes through an OAuth redirect chain that
    # sets short-lived cookies.  Keep them scoped to this one transfer so the
    # chain can finish without persisting credentials between operations.
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        redirect_handler,
        urllib.request.HTTPCookieProcessor(cookie_jar),
    )
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
                session, token = _asf_authenticated_session(
                    asset,
                    adapter,
                    budget,
                    ledger,
                )
                response: requests.Response | None = None
                try:
                    response = _asf_request(
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
                    if not 200 <= response.status_code < 300:
                        _fail(RemoteAccessError, "asf_transfer_denied")
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
                    response = _lpdaac_request(
                        session,
                        "GET",
                        asset.href,
                        adapter=adapter,
                        asset=asset,
                        budget=budget,
                        ledger=ledger,
                    )
                    if not 200 <= response.status_code < 300:
                        _fail(RemoteAccessError, "lpdaac_transfer_denied")
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
                    asset.href,
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
        except (requests.RequestException, urllib.error.URLError, OSError):
            if attempt >= budget.max_retries:
                logger.exception("Remote transfer failed")
                _fail(RemoteAccessError, "transfer_failed")
    size = staging.stat().st_size
    if asset.size_bytes is not None and size != asset.size_bytes:
        _fail(RemoteIntegrityError, "content_length_mismatch")
    if asset.checksum:
        _, expected = asset.checksum.split(":", 1)
        actual = checksum_hasher.hexdigest() if checksum_hasher is not None else ""
        if actual != expected:
            _fail(RemoteIntegrityError, "checksum_mismatch")
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


def _default_fixture() -> None:
    """Install an empty fixture registry for deterministic local use."""
    _register_fixture([], name="fixture")


_default_fixture()
