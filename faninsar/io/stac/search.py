"""STAC Item Search filters, exact geometry, and generation-bound pagination.

The search implementation is shared by the HTTP adapter and the in-memory
index. It validates every parameter, intersects Item geometry exactly through
``shapely``, and binds pagination cursors to one index generation and one query
fingerprint.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NoReturn

from shapely.errors import ShapelyError
from shapely.geometry import box, shape

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from faninsar.io.stac.index import StacIndex
    from faninsar.io.stac.record import StacRecord, TemporalCoverage

__all__ = [
    "SearchPage",
    "SearchRequest",
    "StacSearchError",
    "StaleSearchTokenError",
    "search",
]

_MAX_LIMIT = 1000
_INSTANT_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$"
)
_ALLOWED_GEOMETRY_TYPES = frozenset(
    {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
    }
)


class StacSearchError(ValueError):
    """Raised for an invalid STAC Item Search parameter (HTTP 400)."""


class StaleSearchTokenError(StacSearchError):
    """Raised for a cursor issued for another generation or query."""


def _reject(message: str) -> NoReturn:
    """Raise an invalid-request error with a diagnostic."""
    raise StacSearchError(message)


def _reject_stale(message: str) -> NoReturn:
    """Raise a stale-cursor error with a diagnostic."""
    raise StaleSearchTokenError(message)


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """One Item Search request before execution."""

    ids: tuple[str, ...] | None = None
    collections: tuple[str, ...] | None = None
    bbox: tuple[float, ...] | None = None
    intersects: Mapping[str, Any] | None = None
    datetime: str | None = None
    limit: int = 10
    token: str | None = None


@dataclass(frozen=True, slots=True)
class SearchPage:
    """One page of matched records and its generation-bound cursor."""

    items: tuple[StacRecord, ...]
    number_matched: int
    next_token: str | None


@dataclass(frozen=True, slots=True)
class _Filters:
    """Validated filters used for matching and cursor fingerprints."""

    ids: tuple[str, ...] | None
    collections: tuple[str, ...] | None
    bbox: tuple[float, float, float, float] | None
    intersects: Mapping[str, Any] | None
    geometry: BaseGeometry | None
    datetime_text: str | None
    start: datetime | None
    end: datetime | None


def search(index: StacIndex, request: SearchRequest) -> SearchPage:
    """Execute one validated Item Search request.

    Parameters
    ----------
    index : StacIndex
        Immutable index generation to search.
    request : SearchRequest
        Requested filters, limit, and optional cursor.

    Returns
    -------
    SearchPage
        Records ordered by public id, the full match count, and a next cursor.

    Raises
    ------
    StacSearchError
        If any filter is invalid, or ``bbox`` and ``intersects`` are combined.
    StaleSearchTokenError
        If the cursor belongs to another generation or query.

    """
    filters = _validate(index, request)
    limit = _validate_limit(request.limit)
    fingerprint = _fingerprint(filters, limit)
    offset = 0
    if request.token is not None:
        offset = _decode_token(request.token, index.generation, fingerprint)
    matched = _match(index, filters)
    page = matched[offset : offset + limit]
    next_offset = offset + len(page)
    next_token = None
    if next_offset < len(matched):
        next_token = _encode_token(index.generation, next_offset, fingerprint)
    return SearchPage(
        items=tuple(page),
        number_matched=len(matched),
        next_token=next_token,
    )


def _validate(index: StacIndex, request: SearchRequest) -> _Filters:
    """Validate all request filters and return their normalized form."""
    if request.bbox is not None and request.intersects is not None:
        _reject("bbox and intersects are mutually exclusive")
    ids = _string_tuple(request.ids, "ids")
    collections = _string_tuple(request.collections, "collections")
    if collections is not None:
        known = {collection.collection_id for collection in index.collections}
        unknown = sorted(set(collections) - known)
        if unknown:
            _reject(f"unknown collections: {', '.join(unknown)}")
    bbox = None
    if request.bbox is not None:
        bbox = _parse_bbox(request.bbox)
    geometry = None
    intersects = None
    if request.intersects is not None:
        intersects = _geometry_mapping(request.intersects)
        geometry = _shape_geojson(intersects)
    start, end = _parse_datetime(request.datetime)
    return _Filters(
        ids=ids,
        collections=collections,
        bbox=bbox,
        intersects=intersects,
        geometry=geometry,
        datetime_text=request.datetime,
        start=start,
        end=end,
    )


def _validate_limit(limit: object) -> int:
    """Return the clamped page size or reject a non-integer or zero value."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        _reject("limit must be an integer")
    if limit < 1:
        _reject("limit must be at least 1")
    return min(limit, _MAX_LIMIT)


def _string_tuple(value: object, name: str) -> tuple[str, ...] | None:
    """Validate an optional sequence of non-empty strings."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _reject(f"{name} must be a sequence of strings")
    items = tuple(value)
    if any(not isinstance(item, str) or not item for item in items):
        _reject(f"{name} must be a sequence of non-empty strings")
    return items


def _parse_bbox(value: object) -> tuple[float, float, float, float]:
    """Validate a 4- or 6-number CRS84 bbox and drop elevation."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _reject("bbox must be a sequence of 4 or 6 numbers")
    if len(value) not in {4, 6}:
        _reject("bbox must contain 4 or 6 numbers")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            _reject("bbox values must be numbers")
        number = float(item)
        if not math.isfinite(number):
            _reject("bbox values must be finite")
        numbers.append(number)
    west, south = numbers[0], numbers[1]
    east, north = (
        (numbers[2], numbers[3])
        if len(numbers) == 4
        else (
            numbers[3],
            numbers[4],
        )
    )
    if not -180.0 <= west <= 180.0 or not -180.0 <= east <= 180.0:
        _reject("bbox longitude is outside the CRS84 domain")
    if not -90.0 <= south <= 90.0 or not -90.0 <= north <= 90.0:
        _reject("bbox latitude is outside the CRS84 domain")
    if south > north:
        _reject("bbox latitude ordering is inverted")
    return (west, south, east, north)


def _parse_datetime(
    value: str | None,
) -> tuple[datetime | None, datetime | None]:
    """Parse an RFC 3339 instant or ``/``-separated interval."""
    if value is None:
        return None, None
    if not isinstance(value, str):
        _reject("datetime must be an RFC 3339 string")
    parts = value.split("/")
    if len(parts) == 1:
        instant = _parse_instant(parts[0])
        return instant, instant
    if len(parts) != 2:
        _reject("datetime must be an instant or one interval")
    start = None if parts[0] == ".." else _parse_instant(parts[0])
    end = None if parts[1] == ".." else _parse_instant(parts[1])
    if start is not None and end is not None and start > end:
        _reject("datetime interval start is after its end")
    return start, end


def _parse_instant(value: str) -> datetime:
    """Parse one RFC 3339 timestamp with a mandatory offset."""
    if not _INSTANT_PATTERN.match(value):
        _reject(f"datetime {value!r} is not an RFC 3339 timestamp with an offset")
    try:
        parsed = datetime.fromisoformat(value.replace("t", "T").replace("z", "Z"))
    except ValueError as error:
        _reject(f"datetime {value!r} is invalid: {error}")
    return parsed.astimezone(UTC)


def _geometry_mapping(value: object) -> Mapping[str, Any]:
    """Validate and return one GeoJSON Geometry mapping."""
    if not isinstance(value, Mapping):
        _reject("intersects must be a GeoJSON Geometry object")
    _validate_geojson_geometry(value)
    return value


def _validate_geojson_geometry(value: object) -> None:
    """Structurally validate every GeoJSON geometry type recursively."""
    if not isinstance(value, Mapping):
        _reject("intersects must be a GeoJSON Geometry object")
    geometry_type = value.get("type")
    if not isinstance(geometry_type, str) or geometry_type not in (
        _ALLOWED_GEOMETRY_TYPES
    ):
        _reject(f"unsupported GeoJSON geometry type: {geometry_type!r}")
    if geometry_type == "GeometryCollection":
        members = value.get("geometries")
        if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
            _reject("GeometryCollection requires a geometries sequence")
        for member in members:
            _validate_geojson_geometry(member)
        return
    coordinates = value.get("coordinates")
    if geometry_type == "Point":
        _validate_position(coordinates)
    elif geometry_type == "MultiPoint":
        _validate_positions(coordinates)
    elif geometry_type == "LineString":
        positions = _validate_positions(coordinates)
        if len(positions) < 2:
            _reject("LineString requires at least two positions")
    elif geometry_type == "MultiLineString":
        for line in _sequence(coordinates, "MultiLineString coordinates"):
            positions = _validate_positions(line)
            if len(positions) < 2:
                _reject("LineString requires at least two positions")
    elif geometry_type == "Polygon":
        _validate_rings(coordinates)
    elif geometry_type == "MultiPolygon":
        for polygon in _sequence(coordinates, "MultiPolygon coordinates"):
            _validate_rings(polygon)


def _sequence(value: object, name: str) -> Sequence[Any]:
    """Return a JSON coordinate sequence or reject it."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _reject(f"{name} must be a sequence")
    return value


def _validate_position(value: object) -> tuple[float, ...]:
    """Validate one finite position with at least two numbers."""
    position = _sequence(value, "geometry position")
    if len(position) < 2:
        _reject("geometry position requires at least two numbers")
    numbers = []
    for coordinate in position:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            _reject("geometry positions must contain numbers")
        number = float(coordinate)
        if not math.isfinite(number):
            _reject("geometry positions must be finite")
        numbers.append(number)
    return tuple(numbers)


def _validate_positions(value: object) -> list[tuple[float, ...]]:
    """Validate a sequence of positions."""
    positions = _sequence(value, "geometry coordinates")
    return [_validate_position(position) for position in positions]


def _validate_rings(value: object) -> None:
    """Validate a polygon's linear rings, including closure."""
    for ring in _sequence(value, "polygon coordinates"):
        positions = _validate_positions(ring)
        if len(positions) < 4:
            _reject("polygon rings require at least four positions")
        if positions[0] != positions[-1]:
            _reject("polygon rings must be closed")


def _shape_geojson(value: Mapping[str, Any]) -> BaseGeometry:
    """Convert a validated GeoJSON geometry into a valid shapely geometry."""
    try:
        geometry = shape(value)
    except (KeyError, TypeError, ValueError, ShapelyError) as error:
        _reject(f"invalid intersects geometry: {error}")
    if not geometry.is_valid:
        _reject("intersects geometry is not valid")
    return geometry


def _match(index: StacIndex, filters: _Filters) -> list[StacRecord]:
    """Return records matching every active filter, ordered by public id."""
    requested_ids = None if filters.ids is None else set(filters.ids)
    requested_collections = (
        None if filters.collections is None else set(filters.collections)
    )
    matched = []
    for record in sorted(index.records, key=lambda item: item.public_id):
        if requested_ids is not None and record.public_id not in requested_ids:
            continue
        if (
            requested_collections is not None
            and record.collection_id not in requested_collections
        ):
            continue
        if not _matches_temporal(record.temporal, filters.start, filters.end):
            continue
        if filters.bbox is not None and not _matches_bbox(record, filters.bbox):
            continue
        if filters.geometry is not None and not _matches_intersects(
            record, filters.geometry
        ):
            continue
        matched.append(record)
    return matched


def _matches_temporal(
    temporal: TemporalCoverage,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    """Return whether an Item interval intersects the requested interval."""
    item_start = _utc(temporal.start)
    item_end = _utc(temporal.end)
    if start is not None and item_end < start:
        return False
    return end is None or item_start <= end


def _utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime for comparisons."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _matches_bbox(
    record: StacRecord,
    bbox: tuple[float, float, float, float],
) -> bool:
    """Return whether Item geometry intersects a possibly antimeridian bbox."""
    geometry = _record_geometry(record)
    if geometry is None:
        return False
    west, south, east, north = bbox
    if west <= east:
        return geometry.intersects(box(west, south, east, north))
    return geometry.intersects(box(west, south, 180.0, north)) or (
        geometry.intersects(box(-180.0, south, east, north))
    )


def _matches_intersects(record: StacRecord, geometry: BaseGeometry) -> bool:
    """Return whether Item geometry intersects the query geometry exactly."""
    item_geometry = _record_geometry(record)
    return item_geometry is not None and item_geometry.intersects(geometry)


def _record_geometry(record: StacRecord) -> BaseGeometry | None:
    """Return valid Item geometry, or ``None`` for a geometry-less Item."""
    value = record.item.get("geometry")
    if value is None:
        return None
    try:
        geometry = shape(value)
    except (KeyError, TypeError, ValueError, ShapelyError):
        return None
    return geometry if geometry.is_valid else None


def _fingerprint(filters: _Filters, limit: int) -> str:
    """Return a stable digest covering every filter that affects the result."""
    payload = {
        "ids": None if filters.ids is None else sorted(filters.ids),
        "collections": (
            None if filters.collections is None else sorted(filters.collections)
        ),
        "bbox": None if filters.bbox is None else list(filters.bbox),
        "intersects": filters.intersects,
        "datetime": filters.datetime_text,
        "limit": limit,
    }
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        _reject(f"search query is not JSON-safe: {error}")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _encode_token(generation: str, offset: int, fingerprint: str) -> str:
    """Encode generation, next offset, and query fingerprint as a cursor."""
    payload = {
        "v": 1,
        "g": generation,
        "o": offset,
        "q": fingerprint,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _decode_token(token: str, generation: str, fingerprint: str) -> int:
    """Decode a cursor and reject a malformed, stale, or foreign-query token."""
    if not isinstance(token, str) or not token:
        _reject("token must be a non-empty cursor string")
    padding = "=" * (-len(token) % 4)
    try:
        payload = base64.b64decode(
            (token + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, binascii.Error):
        _reject("token is not a valid cursor")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, ValueError):
        _reject("token is not a valid cursor")
    if not isinstance(decoded, dict) or decoded.get("v") != 1:
        _reject("token is not a valid cursor")
    stored_generation = decoded.get("g")
    offset = decoded.get("o")
    stored_fingerprint = decoded.get("q")
    if (
        not isinstance(stored_generation, str)
        or isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or not isinstance(stored_fingerprint, str)
    ):
        _reject("token is not a valid cursor")
    if stored_generation != generation or stored_fingerprint != fingerprint:
        _reject_stale("token is stale for this index generation")
    return offset
