"""Generic, registered-collection adapter for NASA CMR.

CMR exposes both a compact JSON feed and UMM-JSON granule records.  This
adapter deliberately handles only discovery: it turns either representation
into the small record shape consumed by :mod:`faninsar.remote`, and leaves
geometry filtering and publication to that boundary.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from faninsar.logging import setup_logger

from . import (
    RemoteAccessError,
    RemoteLimitError,
    RemoteQueryError,
    RemoteResourceBudget,
    _CallLedger,
    _fail,
    _RedirectHandler,
    _register_adapter,
    _safe_url,
)

logger = setup_logger(__name__)

DEFAULT_CMR_ENDPOINT = "https://cmr.earthdata.nasa.gov/search/granules.json"


class CMRDiscoveryError(RemoteAccessError):
    """A registered CMR collection could not be decoded safely."""


class CMRRegistrationError(RemoteQueryError):
    """A CMR adapter registration is incomplete or ambiguous."""


def _error(error_type: type[RemoteAccessError], reason: str, message: str) -> None:
    """Log and raise a typed CMR error."""
    logger.error("CMR discovery failed: %s (%s)", message, reason)
    _fail(error_type, reason, message)


def _origin(url: str) -> str:
    """Return the origin part of an HTTPS URL."""
    parsed = urllib.parse.urlsplit(url)
    return f"https://{parsed.hostname}" + (
        f":{parsed.port}" if parsed.port and parsed.port != 443 else ""
    )


def _path(url: str) -> str:
    """Return a non-empty URL path."""
    return urllib.parse.urlsplit(url).path or "/"


def _as_datetime(value: Any) -> datetime | None:
    """Parse CMR's ISO timestamp variants as aware UTC values."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _point(value: Mapping[str, Any]) -> list[float] | None:
    """Read one UMM point object."""
    try:
        longitude = value.get("Longitude", value.get("PointLongitude"))
        latitude = value.get("Latitude", value.get("PointLatitude"))
        return [float(longitude), float(latitude)]
    except (KeyError, TypeError, ValueError):
        return None


def _geometry_from_umm(record: Mapping[str, Any]) -> Mapping[str, Any] | None:  # noqa: PLR0911
    """Convert UMM horizontal spatial domain to GeoJSON."""
    spatial = record.get("SpatialExtent")
    if not isinstance(spatial, Mapping):
        return None
    domain = spatial.get("HorizontalSpatialDomain")
    if not isinstance(domain, Mapping):
        return None
    geometry = domain.get("Geometry")
    if not isinstance(geometry, Mapping):
        return None
    polygons: list[Mapping[str, Any]] = []
    polygon = geometry.get("GPolygon")
    if isinstance(polygon, Mapping):
        polygons.append(polygon)
    polygon_values = geometry.get("GPolygons")
    if isinstance(polygon_values, Mapping):
        polygons.append(polygon_values)
    elif isinstance(polygon_values, list):
        polygons.extend(value for value in polygon_values if isinstance(value, Mapping))
    rings: list[list[list[float]]] = []
    for polygon in polygons:
        boundary = polygon.get("Boundary")
        points = boundary.get("Points") if isinstance(boundary, Mapping) else None
        if isinstance(points, list):
            coords = [_point(point) for point in points if isinstance(point, Mapping)]
            coords = [point for point in coords if point is not None]
            if len(coords) >= 3:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                rings.append(coords)
    if rings:
        return {"type": "Polygon", "coordinates": rings}
    mbr = geometry.get("BoundingRect")
    if isinstance(mbr, Mapping):
        try:
            west = float(mbr["WestBoundingCoordinate"])
            east = float(mbr["EastBoundingCoordinate"])
            south = float(mbr["SouthBoundingCoordinate"])
            north = float(mbr["NorthBoundingCoordinate"])
        except (KeyError, TypeError, ValueError):
            return None
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [west, south],
                    [east, south],
                    [east, north],
                    [west, north],
                    [west, south],
                ]
            ],
        }
    return None


def _geometry_from_cmr(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Convert compact CMR polygons or geometry to GeoJSON."""
    geometry = record.get("geometry")
    if isinstance(geometry, Mapping):
        return geometry
    polygons = record.get("polygons")
    boxes = record.get("boxes")
    if isinstance(boxes, list) and boxes and isinstance(boxes[0], str):
        try:
            south, west, north, east = (float(value) for value in boxes[0].split())
        except (ValueError, TypeError):
            pass
        else:
            return {
                "type": "Polygon",
                "coordinates": [
                    [
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south],
                    ]
                ],
            }
    if isinstance(polygons, str):
        polygons = [polygons]
    if not isinstance(polygons, list) or not polygons:
        return None
    rings: list[list[list[float]]] = []
    for polygon in polygons:
        if isinstance(polygon, str):
            # FedSearch compact polygons are sometimes encoded as one
            # whitespace/comma-delimited latitude/longitude string rather
            # than a list of point strings.
            values = [value for value in re.split(r"[,\s]+", polygon.strip()) if value]
            if len(values) < 6 or len(values) % 2:
                continue
            try:
                pairs = [
                    (float(values[index]), float(values[index + 1]))
                    for index in range(0, len(values), 2)
                ]
            except ValueError:
                continue
            ring = [[longitude, latitude] for latitude, longitude in pairs]
        elif isinstance(polygon, (list, tuple)):
            # A flat numeric list is another compact CMR representation.
            if polygon and all(
                isinstance(point, (int, float)) and not isinstance(point, bool)
                for point in polygon
            ):
                values = [str(value) for value in polygon]
                if len(values) < 6 or len(values) % 2:
                    continue
                try:
                    pairs = [
                        (float(values[index]), float(values[index + 1]))
                        for index in range(0, len(values), 2)
                    ]
                except ValueError:
                    continue
                ring = [[longitude, latitude] for latitude, longitude in pairs]
            else:
                ring = []
                for point in polygon:
                    if isinstance(point, str):
                        values = [
                            value
                            for value in re.split(r"[,\s]+", point.strip())
                            if value
                        ]
                        if len(values) != 2:
                            continue
                        try:
                            latitude, longitude = (float(value) for value in values)
                        except ValueError:
                            continue
                        ring.append([longitude, latitude])
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        try:
                            # CMR compact polygons are latitude/longitude pairs.
                            ring.append([float(point[1]), float(point[0])])
                        except (ValueError, TypeError):
                            continue
        else:
            continue
        if len(ring) >= 3:
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            rings.append(ring)
    if not rings:
        return None
    return {"type": "Polygon", "coordinates": rings}


def _umm_times(record: Mapping[str, Any]) -> tuple[datetime | None, datetime | None]:
    """Extract start/end from a UMM temporal extent."""
    temporal = record.get("TemporalExtents", record.get("TemporalExtent"))
    if isinstance(temporal, Mapping):
        temporal = [temporal]
    if not isinstance(temporal, list) or not temporal:
        return None, None
    extent = temporal[0]
    if not isinstance(extent, Mapping):
        return None, None
    range_value = extent.get("RangeDateTime")
    if isinstance(range_value, Mapping):
        return _as_datetime(range_value.get("BeginningDateTime")), _as_datetime(
            range_value.get("EndingDateTime")
        )
    return _as_datetime(extent.get("SingleDateTime")), _as_datetime(
        extent.get("SingleDateTime")
    )


def _umm_urls(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return only UMM RelatedUrls that explicitly represent GET DATA."""
    granule = record.get("DataGranule")
    related = granule.get("RelatedUrls") if isinstance(granule, Mapping) else None
    if not isinstance(related, list):
        related = record.get("RelatedUrls")
    if not isinstance(related, list):
        return []
    return [
        value
        for value in related
        if isinstance(value, Mapping)
        and str(value.get("Type", "")).strip().upper() == "GET DATA"
        and isinstance(value.get("URL"), str)
    ]


def _cmr_links(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return compact CMR links representing data downloads."""
    links = record.get("links")
    if not isinstance(links, list):
        return []
    selected: list[Mapping[str, Any]] = []
    for link in links:
        if not isinstance(link, Mapping) or not isinstance(link.get("href"), str):
            continue
        rel = str(link.get("rel", "")).strip().lower()
        # FedSearch uses a URI such as
        # ``http://esipfed.org/ns/fedsearch/1.1/data#``.  Prefix matching
        # would also admit browse/service links, which are not data assets.
        if (
            rel == "data#"
            or rel.endswith("/data#")
            or rel
            in {
                "enclosure",
                "download",
            }
        ):
            selected.append(link)
    return selected


def _asset_candidates(record: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Build deterministic data asset candidates from CMR metadata."""
    values = _umm_urls(record)
    if values:
        return [(str(value.get("Name") or "data"), value) for value in values]
    return [(str(value.get("title") or "data"), value) for value in _cmr_links(record)]


def _cmr_number(value: Any) -> str:
    """Format a WGS84 coordinate compactly for CMR query parameters."""
    return format(float(value), ".12g")


def _cmr_spatial_parameters(
    spatial: Any | None,
    spatial_kind: str | None,
    point_geometries: tuple[Any, ...],
) -> dict[str, str]:
    """Translate the normalized WGS84 query into CMR spatial parameters.

    CMR has a point parameter for one point, a bounding-box parameter for
    rectangular (and multi-point) coverage, and a polygon parameter for a
    single polygon.  Complex polygons are deliberately widened to their
    envelope; the public boundary still applies the exact geometry predicate
    after records are normalized.
    """
    if spatial is None or not hasattr(spatial, "bounds"):
        return {}
    if spatial_kind == "points" and len(point_geometries) == 1:
        point = point_geometries[0]
        if hasattr(point, "x") and hasattr(point, "y"):
            return {"point": f"{_cmr_number(point.x)},{_cmr_number(point.y)}"}
    geometry = spatial
    if spatial_kind == "polygons" and getattr(geometry, "geom_type", "") == "Polygon":
        exterior = getattr(geometry, "exterior", None)
        coordinates = getattr(exterior, "coords", None)
        if coordinates is not None:
            values = [
                f"{_cmr_number(longitude)} {_cmr_number(latitude)}"
                for longitude, latitude, *_ in coordinates
            ]
            if len(values) >= 4:
                return {"polygon": ",".join(values)}
    west, south, east, north = geometry.bounds
    return {
        "bounding_box": ",".join(
            _cmr_number(value) for value in (west, south, east, north)
        )
    }


def _cmr_collection_parameters(
    collection: str,
    collection_concept_id: str | None,
    collections: tuple[str, ...] | None,
) -> dict[str, str]:
    """Translate public collection identities into CMR filters."""
    selected = collections or (collection,)
    # CMR concept identifiers are conventionally C-prefixed.  Other
    # collection identities are short names; retaining this distinction avoids
    # accidentally treating a provider's human-readable name as an ID.
    if all(_is_concept_id(value) for value in selected):
        return {"collection_concept_id": ",".join(selected)}
    if collections is None and collection_concept_id:
        return {"collection_concept_id": collection_concept_id}
    return {"short_name": ",".join(selected)}


def _is_concept_id(value: str) -> bool:
    """Return whether a CMR collection value has concept-id syntax."""
    return re.fullmatch(r"C\d+(?:-[A-Za-z0-9_-]+)?", value, re.IGNORECASE) is not None


def _metadata_values(record: Mapping[str, Any], *names: str) -> list[str]:
    """Collect non-empty scalar values from a record and its CMR metadata."""
    values: list[str] = []
    metadata = record.get("meta")
    mappings = [record, metadata] if isinstance(metadata, Mapping) else [record]
    for source in mappings:
        for name in names:
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    return values


def _collection_claims(
    record: Mapping[str, Any], *, is_umm: bool
) -> tuple[list[str], list[str]]:
    """Return explicit collection concept and human-readable identity claims."""
    concepts = _metadata_values(
        record,
        "collection_concept_id",
        "collection-concept-id",
        "CollectionConceptId",
        "collectionConceptId",
    )
    identities = _metadata_values(record, "collection", "short_name", "ShortName")
    if is_umm:
        reference = record.get("CollectionReference")
        if isinstance(reference, Mapping):
            concept = reference.get("CollectionConceptId")
            if isinstance(concept, str) and concept.strip():
                concepts.append(concept.strip())
            for key in ("ShortName", "EntryTitle"):
                value = reference.get(key)
                if isinstance(value, str) and value.strip():
                    identities.append(value.strip())
    return concepts, identities


@dataclass(slots=True)
class CMRCollectionAdapter:
    """Registered CMR collection with bounded JSON pagination.

    Parameters
    ----------
    provider : str
        CMR provider identity (for example ``"ASF"``).
    collection : str
        Product collection identity.  Every emitted record is bound to this
        value; a record claiming another collection is rejected.
    endpoint : str, default=DEFAULT_CMR_ENDPOINT
        CMR granule search endpoint.
    page_size : int, default=100
        Number of granules requested in each page.
    max_pages : int, default=100
        Maximum pages consumed by one search operation.
    collection_concept_id : str, optional
        CMR concept id used as an additional server-side collection filter.
    profiles : sequence of str, default=("anonymous",)
        Authentication profile names accepted by this registration.
    auth_resolver : callable, optional
        Operation-scoped callback receiving a profile name and returning
        transient request headers.  Resolved values are never persisted.

    """

    provider: str
    collection: str
    endpoint: str = DEFAULT_CMR_ENDPOINT
    page_size: int = 100
    max_pages: int = 100
    collection_concept_id: str | None = None
    search_url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    data_origins: tuple[str, ...] = ()
    data_path_prefixes: tuple[str, ...] = ("/",)
    origins: tuple[str, ...] = field(init=False)
    path_prefixes: tuple[str, ...] = field(init=False)
    redirect_origins: tuple[str, ...] = field(init=False)
    profiles: tuple[str, ...] = ("anonymous",)
    auth_resolver: Callable[[str], Mapping[str, str] | None] | None = None

    def __post_init__(self) -> None:
        """Validate collection identity and configure endpoint allowlists."""
        if self.search_url is not None:
            object.__setattr__(self, "endpoint", self.search_url)
        if not isinstance(self.provider, str) or not self.provider.strip():
            _error(
                CMRRegistrationError, "invalid_provider", "provider must be non-empty"
            )
        if not isinstance(self.collection, str) or not self.collection.strip():
            _error(
                CMRRegistrationError,
                "invalid_collection",
                "collection must be non-empty",
            )
        if self.page_size <= 0 or self.max_pages <= 0:
            _error(
                CMRRegistrationError,
                "invalid_pagination",
                "page_size and max_pages must be positive",
            )
        if not self.profiles or any(
            not isinstance(profile, str) or not profile.strip()
            for profile in self.profiles
        ):
            _error(
                CMRRegistrationError,
                "invalid_auth_profiles",
                "auth profiles must contain non-empty names",
            )
        try:
            parsed = urllib.parse.urlsplit(self.endpoint)
            valid_endpoint = parsed.scheme.lower() == "https" and bool(parsed.hostname)
            if valid_endpoint:
                endpoint_port = parsed.port
                del endpoint_port
        except (TypeError, ValueError):
            valid_endpoint = False
        if not valid_endpoint:
            _error(
                CMRRegistrationError,
                "invalid_endpoint",
                "endpoint must be an HTTPS URL",
            )
        try:
            origins = (_origin(self.endpoint), *self.data_origins)
            for data_origin in self.data_origins:
                data_parsed = urllib.parse.urlsplit(data_origin)
                valid_origin = data_parsed.scheme.lower() == "https" and bool(
                    data_parsed.hostname
                )
                if valid_origin:
                    data_port = data_parsed.port
                    del data_port
                else:
                    origins = ()
                    break
        except (TypeError, ValueError):
            _error(
                CMRRegistrationError,
                "invalid_data_origin",
                "data origins must be valid HTTPS URLs",
            )
        if not origins:
            _error(
                CMRRegistrationError,
                "invalid_data_origin",
                "data origins must be valid HTTPS URLs",
            )
        object.__setattr__(self, "origins", origins)
        object.__setattr__(
            self,
            "path_prefixes",
            (_path(self.endpoint), *self.data_path_prefixes),
        )
        object.__setattr__(
            self,
            "redirect_origins",
            (_origin(self.endpoint), *self.data_origins),
        )

    def _auth_headers(self, profile: str) -> Mapping[str, str]:
        """Resolve one profile into transient request headers.

        The resolver runs once per operation.  Its result is retained only in
        the local iterator state and is never included in normalized records,
        exceptions, or logs.
        """
        if profile not in self.profiles:
            _error(
                CMRRegistrationError,
                "unknown_auth_profile",
                "unknown CMR auth profile",
            )
        if self.auth_resolver is None:
            return {}
        try:
            resolved = self.auth_resolver(profile)
        except Exception:
            # Resolver exceptions may contain credential material.  Keep the
            # diagnostic deliberately generic and never stringify the cause.
            logger.warning("CMR auth profile resolution failed")
            _error(
                CMRDiscoveryError,
                "auth_resolution_failed",
                "CMR auth profile could not be resolved",
            )
        if resolved is None:
            return {}
        if not isinstance(resolved, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in resolved.items()
        ):
            _error(
                CMRDiscoveryError,
                "auth_resolution_failed",
                "CMR auth resolver must return string headers",
            )
        return dict(resolved)

    def _request_page(
        self, url: str, headers: Mapping[str, str], ledger: _CallLedger
    ) -> tuple[Mapping[str, Any], Mapping[str, str]]:
        """Fetch and decode one CMR page with ledger accounting."""
        ledger.check_elapsed()
        ledger.request()
        request_headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            **self.headers,
            **headers,
        }
        request = urllib.request.Request(url, headers=request_headers)
        opener = urllib.request.build_opener(
            _RedirectHandler(self, ledger.budget, ledger)
        )
        try:
            with opener.open(
                request, timeout=ledger.budget.read_timeout_seconds
            ) as response:
                content_encoding = response.headers.get("Content-Encoding", "identity")
                if content_encoding != "identity":
                    _error(
                        CMRDiscoveryError,
                        "unexpected_content_encoding",
                        content_encoding,
                    )
                ledger.begin_response()
                parts: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    ledger.response_bytes(len(chunk))
                    if total > ledger.budget.max_response_bytes:
                        _error(
                            RemoteLimitError,
                            "max_response_bytes",
                            "CMR response exceeded max_response_bytes",
                        )
                    parts.append(chunk)
                try:
                    decoded = json.loads(b"".join(parts))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    _error(CMRDiscoveryError, "malformed_json", str(exc))
                if not isinstance(decoded, Mapping):
                    _error(
                        CMRDiscoveryError,
                        "malformed_json",
                        "CMR response must be a JSON object",
                    )
                response_headers = {
                    str(key).lower(): str(value)
                    for key, value in response.headers.items()
                }
                return decoded, response_headers
        except (urllib.error.URLError, OSError) as exc:
            logger.exception("CMR request failed")
            _error(CMRDiscoveryError, "request_failed", str(exc))

    def _pages(
        self,
        ledger: _CallLedger,
        *,
        spatial: Any | None = None,
        spatial_kind: str | None = None,
        point_geometries: tuple[Any, ...] = (),
        datetime_range: tuple[datetime, datetime] | None = None,
        collections: tuple[str, ...] | None = None,
        limit: int | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Yield CMR granule mappings while following Search-After."""
        page_size = min(
            self.page_size,
            limit if limit is not None else self.page_size,
            ledger.budget.max_items,
        )
        params: dict[str, str] = {"page_size": str(page_size)}
        if self.provider:
            params["provider"] = self.provider
        params.update(
            _cmr_collection_parameters(
                self.collection, self.collection_concept_id, collections
            )
        )
        params.update(_cmr_spatial_parameters(spatial, spatial_kind, point_geometries))
        if datetime_range is not None:
            start, end = datetime_range
            params["temporal"] = (
                f"{start.astimezone(UTC).isoformat().replace('+00:00', 'Z')},"
                f"{end.astimezone(UTC).isoformat().replace('+00:00', 'Z')}"
            )
        search_after: str | None = None
        for _ in range(self.max_pages):
            query = urllib.parse.urlencode(params)
            url = f"{self.endpoint}?{query}"
            page_headers = dict(request_headers or {})
            if search_after:
                page_headers["CMR-Search-After"] = search_after
            response, response_headers = self._request_page(url, page_headers, ledger)
            entries = (
                response.get("feed", {}).get("entry", [])
                if isinstance(response.get("feed"), Mapping)
                else response.get("items", response.get("entries", []))
            )
            if isinstance(entries, Mapping):
                entries = [entries]
            if not isinstance(entries, list):
                _error(
                    CMRDiscoveryError,
                    "malformed_entries",
                    "CMR response entries must be a list",
                )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    _error(
                        CMRDiscoveryError,
                        "malformed_entry",
                        "CMR entry must be an object",
                    )
                yield entry
            token = response_headers.get("cmr-search-after") or response_headers.get(
                "search-after"
            )
            if not token or not entries or len(entries) < page_size:
                return
            if token == search_after:
                _error(
                    CMRDiscoveryError,
                    "pagination_loop",
                    "CMR Search-After token did not advance",
                )
            search_after = token
        _error(RemoteLimitError, "max_pages", "CMR pagination exceeded max_pages")

    def _normalize(self, entry: Mapping[str, Any]) -> Mapping[str, Any]:
        """Normalize one compact or UMM granule into a remote record."""
        if isinstance(entry.get("umm"), Mapping):
            entry = entry["umm"]
        is_umm = "GranuleUR" in entry or "DataGranule" in entry
        item_id = entry.get("id") if not is_umm else entry.get("GranuleUR")
        if not isinstance(item_id, str) or not item_id:
            _error(CMRDiscoveryError, "invalid_item_id", "CMR granule has no stable id")
        provider_claims = _metadata_values(
            entry, "provider", "Provider", "provider_id", "provider-id", "ProviderId"
        )
        if provider_claims and any(
            claim.casefold() != self.provider.casefold() for claim in provider_claims
        ):
            _error(
                CMRDiscoveryError,
                "provider_mismatch",
                f"granule {item_id!r} is not from registered provider",
            )
        concept_claims, identity_claims = _collection_claims(entry, is_umm=is_umm)
        expected_concepts = {
            value.casefold()
            for value in (self.collection_concept_id, self.collection)
            if isinstance(value, str) and _is_concept_id(value)
        }
        expected_identity = self.collection.casefold()
        concept_matches = any(
            claim.casefold() in expected_concepts for claim in concept_claims
        )
        # UMM commonly reports a concept id in ``meta`` while the
        # CollectionReference.ShortName is a provider-specific alias.  Once
        # an explicitly registered concept id matches, the alias is not a
        # contradictory identity and must not reject the granule.
        if expected_concepts and concept_claims and not concept_matches:
            mismatch = True
        elif concept_matches:
            mismatch = False
        elif identity_claims:
            mismatch = all(
                claim.casefold() != expected_identity for claim in identity_claims
            )
        else:
            mismatch = False
        if mismatch:
            _error(
                CMRDiscoveryError,
                "collection_mismatch",
                f"granule {item_id!r} is not in registered collection",
            )
        geometry = _geometry_from_umm(entry) if is_umm else _geometry_from_cmr(entry)
        if geometry is None:
            _error(
                CMRDiscoveryError,
                "missing_footprint",
                f"granule {item_id!r} has no usable geometry",
            )
        if is_umm:
            start, end = _umm_times(entry)
            platforms = entry.get("Platforms")
            platform_data = (
                platforms[0]
                if isinstance(platforms, list)
                and platforms
                and isinstance(platforms[0], Mapping)
                else {}
            )
            platform = platform_data.get("ShortName") or platform_data.get("LongName")
            instrument_data = platform_data.get("Instruments")
            instrument = (
                instrument_data[0].get("ShortName")
                if isinstance(instrument_data, list)
                and instrument_data
                and isinstance(instrument_data[0], Mapping)
                else None
            )
            candidates = _asset_candidates(entry)
            properties: dict[str, Any] = {"provider": self.provider}
        else:
            start = _as_datetime(entry.get("time_start"))
            end = _as_datetime(entry.get("time_end"))
            platform = entry.get("platform")
            instrument = entry.get("instrument")
            candidates = _asset_candidates(entry)
            properties = {}
        if not candidates:
            _error(
                CMRDiscoveryError,
                "missing_data_asset",
                f"granule {item_id!r} has no GET DATA candidate",
            )
        assets: dict[str, dict[str, Any]] = {}
        safe_candidates: list[tuple[str, Mapping[str, Any], str]] = []
        for name, candidate in candidates:
            href = str(candidate.get("URL", candidate.get("href", "")))
            try:
                clean_href = _safe_url(href, self)
            except RemoteAccessError:
                continue
            safe_candidates.append((name, candidate, clean_href))
        if not safe_candidates:
            _error(
                CMRDiscoveryError,
                "unsafe_data_url",
                f"granule {item_id!r} contains no registered data URL",
            )
        # One complete-file asset is the portable remote contract.  If a
        # provider advertises several GET DATA URLs, use the first registered
        # HTTPS candidate and leave alternate representations to the provider
        # adapter instead of silently publishing ambiguous choices.
        _name, candidate, clean_href = safe_candidates[0]
        key = "data"
        asset = {"href": clean_href}
        if candidate.get("MimeType") or candidate.get("type"):
            asset["media_type"] = candidate.get("MimeType", candidate.get("type"))
        size = candidate.get("Size", candidate.get("size"))
        if isinstance(size, int) and size >= 0:
            asset["size"] = size
        checksum = candidate.get("Checksum", candidate.get("checksum"))
        if isinstance(checksum, str):
            asset["checksum"] = checksum
        assets[key] = asset
        return {
            "id": item_id,
            "collection": self.collection,
            "geometry": geometry,
            "assets": assets,
            "acquisition": {
                "id": item_id,
                "start": start,
                "end": end,
                "platform": platform,
                "instrument": instrument,
                "properties": properties,
            },
            "properties": properties,
            "provider": self.provider,
        }

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
    ) -> Iterable[Mapping[str, Any]]:
        """Yield normalized records from the registered CMR collection."""
        if ledger is None:
            ledger = _CallLedger(budget or RemoteResourceBudget())
        if budget is not None and budget is not ledger.budget:
            # The public boundary supplies one operation budget.  A direct
            # adapter call may provide only a ledger or only a budget, but a
            # mismatched pair would make accounting ambiguous.
            _error(
                CMRRegistrationError,
                "budget_mismatch",
                "budget must match the supplied operation ledger",
            )
        request_headers = self._auth_headers(auth_profile)
        if limit <= 0:
            _error(CMRRegistrationError, "invalid_limit", "limit must be positive")
        for entry in self._pages(
            ledger,
            spatial=spatial,
            spatial_kind=spatial_kind,
            point_geometries=point_geometries,
            datetime_range=datetime_range,
            collections=collections,
            limit=limit,
            request_headers=request_headers,
        ):
            yield self._normalize(entry)


CMRAdapter = CMRCollectionAdapter
RegisteredCMRCollection = CMRCollectionAdapter


def normalize_cmr_granule(
    entry: Mapping[str, Any],
    *,
    provider: str,
    collection: str,
    data_origins: tuple[str, ...] = (),
    data_path_prefixes: tuple[str, ...] = ("/",),
) -> Mapping[str, Any]:
    """Normalize one CMR JSON or UMM granule without performing I/O."""
    adapter = CMRCollectionAdapter(
        provider=provider,
        collection=collection,
        data_origins=data_origins,
        data_path_prefixes=data_path_prefixes,
    )
    return adapter._normalize(entry)


normalize_umm_granule = normalize_cmr_granule


def discover_cmr(
    adapter: CMRCollectionAdapter,
    *,
    budget: RemoteResourceBudget | None = None,
) -> list[Mapping[str, Any]]:
    """Discover normalized records from a registered adapter.

    This convenience seam is useful to provider integrations that need to
    inspect records before handing them to :func:`faninsar.remote.search`.
    It performs no work until called and accepts the same private ledger used
    by the public remote operation.
    """
    if not isinstance(adapter, CMRCollectionAdapter):
        _error(
            CMRRegistrationError,
            "invalid_adapter",
            "adapter must be a CMRCollectionAdapter",
        )
    operation_budget = budget or RemoteResourceBudget()
    ledger = _CallLedger(operation_budget)
    return list(adapter.items(ledger=ledger))


iter_cmr_records = discover_cmr


def register_cmr_catalog(
    name: str,
    *,
    provider: str,
    collection: str,
    endpoint: str = DEFAULT_CMR_ENDPOINT,
    page_size: int = 100,
    max_pages: int = 100,
    collection_concept_id: str | None = None,
    headers: Mapping[str, str] | None = None,
    profiles: Sequence[str] = ("anonymous",),
    auth_resolver: Callable[[str], Mapping[str, str] | None] | None = None,
    data_origins: tuple[str, ...] = (),
    data_path_prefixes: tuple[str, ...] = ("/",),
) -> CMRCollectionAdapter:
    """Register one CMR collection in the private remote adapter registry.

    ``auth_resolver`` is invoked only when an operation selects one of the
    registered ``profiles``.  Credentials therefore remain in the caller's
    resolver and transient request state rather than in the catalog registry.
    """
    adapter = CMRCollectionAdapter(
        provider=provider,
        collection=collection,
        endpoint=endpoint,
        page_size=page_size,
        max_pages=max_pages,
        collection_concept_id=collection_concept_id,
        headers=headers or {},
        profiles=tuple(profiles),
        auth_resolver=auth_resolver,
        data_origins=data_origins,
        data_path_prefixes=data_path_prefixes,
    )
    _register_adapter(name, adapter)
    return adapter


register_cmr_collection = register_cmr_catalog

__all__ = [
    "DEFAULT_CMR_ENDPOINT",
    "CMRAdapter",
    "CMRCollectionAdapter",
    "CMRDiscoveryError",
    "CMRRegistrationError",
    "RegisteredCMRCollection",
    "discover_cmr",
    "iter_cmr_records",
    "normalize_cmr_granule",
    "normalize_umm_granule",
    "register_cmr_catalog",
    "register_cmr_collection",
]
