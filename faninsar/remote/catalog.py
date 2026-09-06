"""Catalog registry, query geometry, and normalized search."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from faninsar.data.query import BoundingBox, Points, Polygons

from .access import _canonicalize_url
from .errors import RemoteAccessError, RemoteQueryError, _fail
from .protocols import _Adapter, _adapter_items, _CallLedger
from .records import (
    AcquisitionMetadata,
    CatalogItem,
    RemoteAsset,
    RemoteResourceBudget,
    _freeze,
    _normalize_checksum,
    _sanitize,
    _utc,
)

if TYPE_CHECKING:
    from datetime import datetime


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
        return _fail(RemoteAccessError, "asset_not_found")


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
    return _fail(RemoteQueryError, "unsupported_geometry", msg)


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


__all__ = [
    "CatalogItem",
    "RemoteResourceBudget",
    "_normalize_record",
    "_query_geometry",
    "_register_adapter",
    "_register_fixture",
    "search",
]
