"""Provider-neutral catalog search and complete-file download.

The module deliberately owns transport mechanics only.  Domain adapters retain
the meaning of DEMs, scenes, or interferograms and may use the two public
operations without introducing a second scientific object model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, Protocol

from pyproj import CRS, Transformer
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
    "RemoteError",
    "RemoteIntegrityError",
    "RemoteLimitError",
    "RemoteQueryError",
    "RemoteResourceBudget",
    "download",
    "search",
]


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
    r"(?:sig(?:nature)?|token|expires?|se(?:curity)?|x-amz-|authorization)",
    re.IGNORECASE,
)


def _sanitize(value: Any, key: str | None = None) -> Any:  # noqa: PLR0911
    """Remove credential-bearing values from nested provider data."""
    if key is not None and _SECRET_KEY.search(key):
        return None
    if isinstance(value, Mapping):
        return {
            str(k): _sanitize(v, str(k))
            for k, v in value.items()
            if not _SECRET_KEY.search(str(k))
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
        value = datetime.fromisoformat(value)
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
    profiles: tuple[str, ...]

    def items(self) -> Iterable[Mapping[str, Any]]: ...

    def fetch(self, asset: RemoteAsset, budget: RemoteResourceBudget) -> Any: ...


@dataclass
class _FixtureAdapter:
    """Private in-memory adapter used by contract tests and examples."""

    records: list[Mapping[str, Any]]
    provider: str = "fixture"
    origins: tuple[str, ...] = ("https://fixture.invalid",)
    path_prefixes: tuple[str, ...] = ("/",)
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
    source = CRS.from_user_input(crs)
    transformer = Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)

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


def _safe_url(url: str, adapter: _Adapter) -> str:
    """Validate and canonicalize an asset URL against its registration."""
    parsed = urllib.parse.urlsplit(url)
    raw_lower = url.lower()
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        _fail(RemoteAccessError, "invalid_endpoint")
    if "\\" in url or "%2f" in raw_lower or "%5c" in raw_lower:
        _fail(RemoteAccessError, "invalid_endpoint")
    segments = urllib.parse.unquote(parsed.path).split("/")
    if any(segment in {".", ".."} for segment in segments):
        _fail(RemoteAccessError, "invalid_endpoint")
    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError:
        _fail(RemoteAccessError, "invalid_endpoint")
    origin = f"https://{host}" + (f":{port}" if port and port != 443 else "")
    if origin not in tuple(item.lower().rstrip("/") for item in adapter.origins):
        _fail(RemoteAccessError, "unregistered_endpoint")
    path = parsed.path or "/"
    prefixes = adapter.path_prefixes or ("/",)
    if not any(
        path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")
        for prefix in prefixes
    ):
        _fail(RemoteAccessError, "unregistered_endpoint")
    return urllib.parse.urlunsplit(
        ("https", origin.removeprefix("https://"), path, "", "")
    )


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
        href = _safe_url(str(value.get("href", "")), adapter)
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
    start_end = tuple(_utc(item) for item in datetime_range) if datetime_range else None
    results: list[CatalogItem] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for raw in adapter.items():
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


def _matching_manifest(path: Path, asset: RemoteAsset) -> bool:
    """Return whether an existing destination is a valid qualified reuse."""
    manifest_path = path.with_name(path.name + ".faninsar.remote.json")
    if not path.is_file() or not manifest_path.is_file():
        return False
    if asset.checksum is None and asset.version is None:
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


def _fetch(
    asset: RemoteAsset, adapter: _Adapter, budget: RemoteResourceBudget
) -> bytes:
    """Fetch complete bytes through an adapter or standard HTTPS."""

    def bounded(chunks: Iterable[bytes]) -> bytes:
        """Collect chunks while enforcing the response and output limits."""
        parts: list[bytes] = []
        total = 0
        for chunk in chunks:
            part = bytes(chunk)
            total += len(part)
            if total > budget.max_response_bytes:
                _fail(RemoteLimitError, "max_response_bytes")
            if total > budget.max_output_bytes:
                _fail(RemoteLimitError, "max_output_bytes")
            parts.append(part)
        return b"".join(parts)

    fetcher = getattr(adapter, "fetch", None)
    result = None if fetcher is None else fetcher(asset, budget)
    if isinstance(result, (bytes, bytearray)):
        return bytes(result)
    if hasattr(result, "read"):
        return bounded(iter(lambda: result.read(1024 * 1024), b""))
    if isinstance(result, Iterable):
        return bounded(result)
    request = urllib.request.Request(
        asset.href, headers={"Accept-Encoding": "identity"}
    )
    try:
        with urllib.request.urlopen(
            request, timeout=budget.read_timeout_seconds
        ) as response:
            if response.headers.get("Content-Encoding", "identity") != "identity":
                _fail(RemoteAccessError, "unexpected_content_encoding")
            return bounded(iter(lambda: response.read(1024 * 1024), b""))
    except urllib.error.URLError:
        logger.exception("Remote transfer failed")
        _fail(RemoteAccessError, "transfer_failed")


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
            _fail(RemoteIntegrityError, "destination_conflict")
    payload = _fetch(asset, adapter, budget)
    if len(payload) > budget.max_response_bytes:
        _fail(RemoteLimitError, "max_response_bytes")
    if len(payload) > budget.max_output_bytes:
        _fail(RemoteLimitError, "max_output_bytes")
    if asset.size_bytes is not None and len(payload) != asset.size_bytes:
        _fail(RemoteIntegrityError, "content_length_mismatch")
    digest = hashlib.sha256(payload).hexdigest()
    if asset.checksum:
        algorithm, expected = asset.checksum.split(":", 1)
        actual = hashlib.new(algorithm, payload).hexdigest()
        if actual != expected:
            _fail(RemoteIntegrityError, "checksum_mismatch")
    with lock:
        if destination.exists() and not overwrite:
            if _matching_manifest(destination, asset):
                return destination
            _fail(RemoteIntegrityError, "destination_conflict")
        temporary: Path | None = None
        manifest = destination.with_name(destination.name + ".faninsar.remote.json")
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
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
                        "size": len(payload),
                    },
                    stream,
                    separators=(",", ":"),
                )
                stream.flush()
                os.fsync(stream.fileno())
            manifest_temp.replace(manifest)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination


def _default_fixture() -> None:
    """Install an empty fixture registry for deterministic local use."""
    _register_fixture([], name="fixture")


_default_fixture()
