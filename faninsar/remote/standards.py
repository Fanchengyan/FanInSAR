"""Validation and normalization for the FanInSAR remote STAC profile.

The remote boundary accepts a deliberately small, version-pinned subset of
STAC.  Keeping this policy in one module prevents individual providers from
quietly inventing extension names or passing provider-specific metadata into
the scientific data model.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlparse, urlsplit, urlunsplit

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

try:  # Importing this module while ``remote`` is being initialized is safe.
    from faninsar.remote import RemoteQueryError
except ImportError:  # pragma: no cover - useful when imported standalone

    class RemoteQueryError(ValueError):
        """Fallback error used only during isolated module loading."""

        def __init__(self, reason: str, message: str | None = None) -> None:
            self.reason = reason
            super().__init__(message or reason)


STAC_CORE_VERSION = "1.1.0"
"""Version of the STAC core supported by this boundary."""

STAC_PROFILE_VERSIONS: dict[str, str] = {
    "sar": "1.3.2",
    "satellite": "1.2.0",
    "projection": "2.0.0",
    "file": "2.1.0",
    "insar": "1.0.0",
}
"""Supported STAC extension versions, keyed by their short names."""

SUPPORTED_STAC_PROFILES = STAC_PROFILE_VERSIONS
STAC_PROFILES = STAC_PROFILE_VERSIONS

_EXTENSION_URLS = {
    "sar": "https://stac-extensions.github.io/sar/v1.3.2/schema.json",
    "satellite": "https://stac-extensions.github.io/sat/v1.2.0/schema.json",
    "projection": "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
    "file": "https://stac-extensions.github.io/file/v2.1.0/schema.json",
    "insar": "https://stac-extensions.github.io/insar/v1.0.0/schema.json",
}
SUPPORTED_STAC_EXTENSIONS = _EXTENSION_URLS

_PREFIX_TO_PROFILE = {
    "sar": "sar",
    "sat": "satellite",
    "proj": "projection",
    "file": "file",
    "insar": "insar",
}
_SIGNED_QUERY_KEYS = frozenset(
    {
        "sig",
        "signature",
        "token",
        "expires",
        "se",
        "sp",
        "st",
        "sv",
        "sr",
        "spr",
        "sip",
        "si",
        "skoid",
        "sktid",
        "skt",
        "ske",
        "sks",
        "skv",
        "rscc",
        "rscd",
        "rsce",
        "rscl",
        "rsct",
    }
)
_KNOWN_TOP_LEVEL = {
    "type",
    "stac_version",
    "stac_extensions",
    "id",
    "geometry",
    "bbox",
    "properties",
    "assets",
    "collection",
    "links",
}


class STACProfileError(RemoteQueryError):
    """A STAC document does not meet the registered remote profile."""


class UnknownSTACProfileError(STACProfileError):
    """A document declares an extension outside the supported profile."""


class MalformedSTACItemError(STACProfileError):
    """A STAC item is structurally invalid or has an invalid value."""


def _fail(error_type: type[STACProfileError], reason: str, message: str) -> None:
    """Log and raise one typed profile error."""
    logger.error("STAC profile validation failed: %s (%s)", message, reason)
    raise error_type(reason, message)


def _scrub_signed(value: Any) -> Any:
    """Remove Azure SAS material from values retained by STAC normalization."""
    if isinstance(value, Mapping):
        return {
            str(key): _scrub_signed(item)
            for key, item in value.items()
            if str(key).lower() not in _SIGNED_QUERY_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_scrub_signed(item) for item in value]
    if isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            query = parse_qsl(parsed.query, keep_blank_values=True)
            if any(key.lower() in _SIGNED_QUERY_KEYS for key, _ in query):
                return urlunsplit(
                    (parsed.scheme, parsed.netloc, parsed.path, "", "")
                )
    return value


def _extension_name(value: str) -> str | None:
    """Resolve one STAC extension URL or short name to its profile name."""
    if value in _EXTENSION_URLS.values():
        return next(name for name, url in _EXTENSION_URLS.items() if url == value)
    # The schema URI is an origin-qualified trust anchor.  Matching only the
    # path would allow an unrelated host to claim one of our approved
    # extensions (for example ``https://attacker.invalid/sar/...``).
    return None


def _validate_extensions(item: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate declared extension URLs and return their short names."""
    declared = item.get("stac_extensions", ())
    if declared is None:
        declared = ()
    if not isinstance(declared, (list, tuple)) or any(
        not isinstance(value, str) for value in declared
    ):
        _fail(
            MalformedSTACItemError,
            "invalid_stac_extensions",
            "stac_extensions must be strings",
        )
    profiles: list[str] = []
    for value in declared:
        name = _extension_name(value)
        if name is None:
            _fail(UnknownSTACProfileError, "unknown_stac_extension", value)
        if name in profiles:
            continue
        profiles.append(name)
    return tuple(profiles)


def _utc_datetime(value: Any, field: str) -> datetime | None:
    """Parse one STAC datetime field into aware UTC time."""
    if value is None:
        return None
    if not isinstance(value, str):
        _fail(
            MalformedSTACItemError, "invalid_datetime", f"{field} must be an ISO string"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        _fail(MalformedSTACItemError, "invalid_datetime", f"{field}: {exc}")
    if parsed.tzinfo is None:
        _fail(
            MalformedSTACItemError, "naive_datetime", f"{field} must include a timezone"
        )
    return parsed.astimezone(UTC)


def _validate_geometry(geometry: Any) -> Mapping[str, Any] | None:
    """Require a GeoJSON geometry object and return it unchanged."""
    if geometry is None:
        return None
    if not isinstance(geometry, Mapping):
        _fail(
            MalformedSTACItemError,
            "invalid_geometry",
            "geometry must be a GeoJSON object",
        )
    if not isinstance(geometry.get("type"), str) or "coordinates" not in geometry:
        _fail(
            MalformedSTACItemError,
            "invalid_geometry",
            "geometry must contain type and coordinates",
        )
    return geometry


def _asset_to_record(asset: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Convert one STAC asset into the provider-neutral asset shape."""
    href = asset.get("href")
    if not isinstance(href, str) or not href:
        _fail(
            MalformedSTACItemError, "invalid_asset_href", f"asset {key!r} has no href"
        )
    parsed = urlparse(href)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        _fail(
            MalformedSTACItemError,
            "invalid_asset_href",
            f"asset {key!r} href is not HTTP(S)",
        )
    media_type = asset.get("type")
    if media_type is not None and not isinstance(media_type, str):
        _fail(
            MalformedSTACItemError,
            "invalid_asset_media_type",
            f"asset {key!r} type must be a string",
        )
    size = asset.get("file:size")
    if size is not None and (
        isinstance(size, bool) or not isinstance(size, int) or size < 0
    ):
        _fail(
            MalformedSTACItemError,
            "invalid_asset_size",
            f"asset {key!r} file:size must be non-negative",
        )
    checksum = asset.get("file:checksum")
    if checksum is not None and not isinstance(checksum, str):
        _fail(
            MalformedSTACItemError,
            "invalid_asset_checksum",
            f"asset {key!r} checksum must be a string",
        )
    properties = {
        name: value
        for name, value in asset.items()
        if name in {"roles", "title", "description"}
    }
    result: dict[str, Any] = {"href": _scrub_signed(href)}
    if media_type is not None:
        result["media_type"] = media_type
    if size is not None:
        result["size"] = size
    if checksum is not None:
        result["checksum"] = checksum
    if properties:
        result["properties"] = properties
    return result


def validate_stac_item(
    item: Mapping[str, Any], *, require_insar: bool = False
) -> tuple[str, ...]:
    """Validate one STAC 1.1 item against the registered extension subset.

    Parameters
    ----------
    item : Mapping
        JSON-like STAC Item mapping.
    require_insar : bool, default=False
        Require the InSAR extension and explicit units when InSAR fields are
        present.  This is intended for products that opt into the policy.

    Returns
    -------
    tuple[str, ...]
        Recognized extension names in declaration order.

    Raises
    ------
    STACProfileError
        If the item is unknown, malformed, or outside the pinned profile.

    """
    if not isinstance(item, Mapping):
        _fail(MalformedSTACItemError, "invalid_stac_item", "item must be a mapping")
    if item.get("type") != "Feature":
        _fail(
            MalformedSTACItemError,
            "invalid_stac_type",
            "STAC Item type must be Feature",
        )
    version = item.get("stac_version")
    if version != STAC_CORE_VERSION:
        _fail(
            STACProfileError,
            "unsupported_stac_version",
            f"expected {STAC_CORE_VERSION}, got {version!r}",
        )
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id:
        _fail(
            MalformedSTACItemError, "invalid_item_id", "STAC Item id must be non-empty"
        )
    geometry = _validate_geometry(item.get("geometry"))
    if geometry is None:
        _fail(
            MalformedSTACItemError,
            "missing_geometry",
            "STAC Item needs a geometry-ready GeoJSON geometry",
        )
    properties = item.get("properties")
    if not isinstance(properties, Mapping):
        _fail(
            MalformedSTACItemError, "invalid_properties", "properties must be a mapping"
        )
    datetime_value = properties.get("datetime")
    start = _utc_datetime(properties.get("start_datetime"), "start_datetime")
    end = _utc_datetime(properties.get("end_datetime"), "end_datetime")
    if datetime_value is not None:
        _utc_datetime(datetime_value, "datetime")
    elif start is None and end is None:
        _fail(
            MalformedSTACItemError,
            "missing_datetime",
            "properties needs datetime or an interval",
        )
    if start is not None and end is not None and end < start:
        _fail(
            MalformedSTACItemError,
            "invalid_datetime_range",
            "end_datetime precedes start_datetime",
        )
    profiles = _validate_extensions(item)
    if require_insar and "insar" not in profiles:
        _fail(
            STACProfileError,
            "missing_insar_profile",
            "InSAR products must declare the InSAR extension",
        )
    for key in properties:
        if key.startswith("insar:") and "insar" not in profiles:
            _fail(STACProfileError, "unregistered_insar_field", key)
    if "insar" in profiles:
        units = properties.get("insar:units")
        measured_fields = {
            "insar:baseline_perpendicular",
            "insar:baseline_parallel",
            "insar:temporal_baseline",
            "insar:off_nadir_angle",
            "insar:incidence_angle",
        }
        if measured_fields.intersection(properties) and (
            not isinstance(units, str) or not units.strip()
        ):
            _fail(
                STACProfileError,
                "missing_insar_units",
                "numeric InSAR measurements require explicit insar:units",
            )
    assets = item.get("assets")
    if not isinstance(assets, Mapping) or not assets:
        _fail(MalformedSTACItemError, "missing_assets", "STAC Item must contain assets")
    for key, asset in assets.items():
        if not isinstance(key, str) or not isinstance(asset, Mapping):
            _fail(
                MalformedSTACItemError,
                "invalid_asset",
                "assets must map string names to objects",
            )
        _asset_to_record(asset, key)
    return profiles


def normalize_stac_item(
    item: Mapping[str, Any],
    *,
    provider: str,
    catalog: str,
    collection: str | None = None,
    require_insar: bool = False,
) -> dict[str, Any]:
    """Validate and convert a STAC Item to the remote adapter record shape.

    Only STAC core/common and the pinned SAR, Satellite, Projection, File,
    and optional InSAR fields are copied.  No FanInSAR-specific properties are
    introduced here.
    """
    if not provider or not catalog:
        _fail(STACProfileError, "invalid_identity", "provider and catalog are required")
    profiles = validate_stac_item(item, require_insar=require_insar)
    item_collection = item.get("collection")
    if collection is not None and item_collection not in {None, collection}:
        _fail(
            STACProfileError,
            "collection_mismatch",
            f"expected collection {collection!r}",
        )
    properties = _scrub_signed(dict(item["properties"]))
    start = _utc_datetime(properties.get("start_datetime"), "start_datetime")
    end = _utc_datetime(properties.get("end_datetime"), "end_datetime")
    instant = _utc_datetime(properties.get("datetime"), "datetime")
    if instant is not None:
        start = end = instant
    acquisition_properties: dict[str, Any] = {}
    # Common STAC properties are retained under acquisition metadata.  They
    # are all namespaced or core-defined; arbitrary provider fields are kept
    # only in raw_metadata by the outer boundary.
    for name in (
        "platform",
        "constellation",
        "instruments",
        "mission",
        "sar:frequency_band",
        "sar:center_frequency",
        "sar:polarizations",
        "sar:product_type",
        "sar:resolution_range",
        "sar:resolution_azimuth",
        "sat:orbit_state",
        "sat:relative_orbit",
        "sat:orbit_cycle",
        "proj:epsg",
        "proj:shape",
        "proj:transform",
        "proj:bbox",
    ):
        if name in properties:
            acquisition_properties[name] = properties[name]
    instrument = properties.get("instruments")
    if isinstance(instrument, (list, tuple)):
        instrument = instrument[0] if instrument else None
    orbit = properties.get("sat:relative_orbit")
    if isinstance(orbit, bool) or not isinstance(orbit, int):
        orbit = None
    assets = {
        str(key): _asset_to_record(value, str(key))
        for key, value in item["assets"].items()
    }
    record: dict[str, Any] = {
        "id": item["id"],
        "collection": collection if collection is not None else item_collection,
        "geometry": item.get("geometry"),
        "assets": assets,
        "acquisition": {
            "id": item["id"],
            "start": start,
            "end": end,
            "platform": properties.get("platform"),
            "instrument": instrument,
            "mode": properties.get("sar:instrument_mode"),
            "orbit": orbit,
            "processing_level": properties.get("sar:product_type"),
            "properties": acquisition_properties,
        },
        "properties": properties,
        "stac_profiles": profiles,
        "provider": provider,
        "catalog": catalog,
    }
    if item.get("bbox") is not None:
        record["bbox"] = item["bbox"]
    return record


stac_item_to_record = normalize_stac_item
map_stac_item = normalize_stac_item

__all__ = [
    "STAC_CORE_VERSION",
    "STAC_PROFILES",
    "STAC_PROFILE_VERSIONS",
    "SUPPORTED_STAC_EXTENSIONS",
    "SUPPORTED_STAC_PROFILES",
    "MalformedSTACItemError",
    "STACProfileError",
    "UnknownSTACProfileError",
    "map_stac_item",
    "normalize_stac_item",
    "stac_item_to_record",
    "validate_stac_item",
]
