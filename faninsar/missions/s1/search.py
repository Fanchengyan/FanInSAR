"""Sentinel-1 search profile and local input preparation helpers.

The profile translates the small Sentinel-1 vocabulary into the existing ASF
search adapter.  Transfer and authentication remain owned by
``faninsar.remote``; this module only selects a qualified SAFE representation
and hands it to the local SAFE reader.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from xml.etree.ElementTree import ParseError

from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from faninsar.remote.records import CatalogItem, RemoteAsset

logger = setup_logger(__name__)

S1ProductType = Literal["burst", "scene"]
S1SinglePolarization = Literal["VV", "VH", "HH", "HV"]
S1DualPolarization = Literal["VV+VH", "HH+HV"]
S1Polarization = S1SinglePolarization | S1DualPolarization
S1PolarizationRequest = tuple[S1Polarization, ...] | Literal["all"]
S1Mission = Literal["S1A", "S1B", "S1C", "S1D"]

_SINGLE_POLARIZATIONS = frozenset({"VV", "VH", "HH", "HV"})
_DUAL_POLARIZATIONS = frozenset({"VV+VH", "HH+HV"})
_POLARIZATIONS = _SINGLE_POLARIZATIONS | _DUAL_POLARIZATIONS
_MISSIONS = frozenset({"S1A", "S1B", "S1C", "S1D"})
_SAFE_SUFFIXES = frozenset({".zip"})
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def normalize_missions(
    value: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Validate a Sentinel-1 satellite selection to ``"A"``..``"D"`` letters.

    Accepts the mission letters — ``"A"``, ``"ACD"``, ``("A", "C", "D")`` —
    and tolerates the ``"S1"``/``"SENTINEL-1"`` prefixes (``"S1A"``,
    ``"Sentinel-1D"``), normalizing everything to the bare letters
    ``("A", "C", "D")``.  ``None`` returns ``None`` (no satellite
    constraint).  Provider filters translate each letter to the ASF
    ``platform="Sentinel-1X"`` spelling.
    """
    if value is None:
        return None
    if isinstance(value, str):
        tokens: tuple[str, ...] = (value,)
    else:
        tokens = tuple(str(item) for item in value)
    normalized: list[str] = []
    for raw in tokens:
        text = raw.strip().upper().replace("-", "").replace("_", "")
        for prefix in ("SENTINEL1", "S1"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        if not text or any(letter not in "ABCD" for letter in text):
            message = f"missions entries must be Sentinel-1 letters A..D; got {raw!r}"
            reject(message, ValueError, logger=logger)
        for letter in text:
            if letter not in normalized:
                normalized.append(letter)
    if not normalized:
        message = "missions must contain at least one satellite"
        reject(message, ValueError, logger=logger)
    return tuple(normalized)


def mission_platform_filter(missions: Sequence[str]) -> list[str]:
    """Translate ``"A"``-style mission letters to ASF ``platform`` values."""
    return [f"Sentinel-1{letter}" for letter in missions]


def normalize_polarizations(
    value: S1PolarizationRequest | Sequence[str] | str | None,
) -> tuple[str, ...] | Literal["all"]:
    """Validate and expand a Sentinel-1 polarization request.

    Parameters
    ----------
    value : sequence of str, str, or None
        Single/dual ASF values, local channel names, or ``"all"``.  ``None``
        selects the public default ``("VV",)``.

    Returns
    -------
    tuple of str or ``"all"``
        Individual channels in stable caller order, or the all-channels
        sentinel.

    Raises
    ------
    ValueError
        If a value is empty or is not supported by Sentinel-1 SLC readers.

    """
    if value is None:
        return ("VV",)
    if isinstance(value, str):
        values: tuple[str, ...] | Literal["all"] = value.upper()
        if values == "ALL":
            return "all"
        values = (values,)
    else:
        values = tuple(str(item).upper() for item in value)
    if not values:
        message = "polarizations must contain at least one value"
        reject(message, ValueError, logger=logger)
    if len(values) == 1 and values[0] == "ALL":
        return "all"
    invalid = set(values).difference(_POLARIZATIONS)
    if invalid:
        allowed = ", ".join(sorted(_POLARIZATIONS))
        message = (
            "unsupported Sentinel-1 polarization(s): "
            f"{sorted(invalid)}; allowed={allowed}"
        )
        reject(message, ValueError, logger=logger)
    expanded: list[str] = []
    for item in values:
        channels = item.split("+")
        for channel in channels:
            if channel not in expanded:
                expanded.append(channel)
    return tuple(expanded)


def provider_polarization_filter(
    value: tuple[str, ...] | Literal["all"],
    *,
    allow_dual: bool = False,
) -> str | None:
    """Return an ASF polarization filter without over-restricting searches.

    ASF's burst endpoint accepts only one channel per request.  Full-scene
    search supports the combined dual values, so callers identify that case
    with ``allow_dual``.
    """
    if value == "all":
        return None
    channels = set(value)
    if allow_dual and channels == {"VV", "VH"}:
        return "VV+VH"
    if allow_dual and channels == {"HH", "HV"}:
        return "HH+HV"
    if len(channels) == 1:
        return next(iter(channels))
    # ASF's combined values are finite and mission-specific.  An unusual
    # combination is filtered after normalization rather than losing valid
    # products through a provider-side conjunction.
    return None


def product_filters(
    product_type: S1ProductType,
    polarizations: tuple[str, ...] | Literal["all"],
    filters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose ASF defaults and explicit advanced filters."""
    if product_type not in ("burst", "scene"):
        message = "product_type must be 'burst' or 'scene'"
        reject(message, ValueError, logger=logger)
    defaults: dict[str, Any]
    if product_type == "burst":
        defaults = {"dataset": "SLC-BURST", "processingLevel": "BURST"}
    else:
        defaults = {"dataset": "SENTINEL-1", "processingLevel": "SLC"}
    provider_pol = provider_polarization_filter(
        polarizations,
        allow_dual=product_type == "scene",
    )
    if provider_pol is not None:
        defaults["polarization"] = provider_pol
    defaults.update(dict(filters or {}))
    return defaults


def complete_safe_asset(item: CatalogItem) -> RemoteAsset:
    """Select exactly one qualified SAFE ZIP asset from a catalog item."""
    candidates = []
    for asset in item.assets.values():
        suffix = urllib.parse.urlsplit(asset.href).path.casefold()
        media_type = (asset.media_type or "").casefold()
        is_zip = suffix.endswith(tuple(_SAFE_SUFFIXES)) or (
            "zip" in media_type and not suffix.endswith((".tif", ".tiff", ".xml"))
        )
        if asset.representation == "complete" and is_zip:
            candidates.append(asset)
    if len(candidates) != 1:
        message = (
            f"catalog item {item.item_id!r} must expose exactly one complete SAFE ZIP; "
            f"found {len(candidates)}"
        )
        reject(message, ValueError, logger=logger)
    return candidates[0]


def canonical_items(items: Sequence[CatalogItem]) -> list[CatalogItem]:
    """Return catalog items in deterministic acquisition order."""
    return sorted(
        items,
        key=lambda item: (
            item.acquisition.start or datetime.min.replace(tzinfo=UTC),
            item.item_id,
            tuple(sorted(asset.href for asset in item.assets.values())),
        ),
    )


def filter_burst_polarizations(
    items: Sequence[CatalogItem],
    requested: tuple[str, ...] | Literal["all"],
) -> list[CatalogItem]:
    """Filter burst records by explicit catalog polarization metadata.

    ASF burst search accepts only one channel filter.  A multi-channel request
    therefore searches without that filter and requires every returned item to
    carry an unambiguous polarization declaration before accepting it.
    """
    if requested == "all" or len(requested) <= 1:
        return list(items)
    expected = set(requested)
    selected: list[CatalogItem] = []
    for item in items:
        values = _metadata_polarizations(item.raw_metadata)
        values.update(_metadata_polarizations(item.acquisition.properties))
        if not values:
            message = (
                f"burst catalog item {item.item_id!r} has no unambiguous polarization"
            )
            reject(message, ValueError, logger=logger)
        if values.intersection(expected) and values.issubset(expected):
            selected.append(item)
    return selected


def _metadata_polarizations(value: object) -> set[str]:
    """Extract Sentinel-1 channel values from bounded nested metadata."""
    if isinstance(value, Mapping):
        result: set[str] = set()
        for key, child in value.items():
            if str(key).casefold() in {
                "polarization",
                "polarizations",
            } or isinstance(child, Mapping):
                result.update(_metadata_polarizations(child))
        return result
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for child in value:
            result.update(_metadata_polarizations(child))
        return result
    if isinstance(value, str):
        return {
            part.strip().upper()
            for part in value.replace(",", "+").split("+")
            if part.strip().upper() in _SINGLE_POLARIZATIONS
        }
    return set()


def _item_track_key(item: CatalogItem) -> str | None:
    """Return the ``track_group_key`` label for one catalog item, if derivable.

    ASF items carry ``relativeOrbit``/``orbitDirection``/``mission`` inside
    ``acquisition.properties`` (stage-1 pass-through); CDSE items keep
    ``relative_orbit``/``orbit_direction`` on ``raw_metadata``.  Both
    shapes reduce to the same ``track_group_key`` inputs.
    """
    from faninsar.missions.s1.types import track_group_key

    props = getattr(item.acquisition, "properties", {}) or {}
    raw = getattr(item, "raw_metadata", {}) or {}
    relative_orbit = (
        props.get("relativeOrbit")
        or props.get("relative_orbit")
        or raw.get("relative_orbit")
        or raw.get("sat:relative_orbit")
    )
    direction = (
        props.get("orbitDirection")
        or props.get("orbit_direction")
        or raw.get("orbit_direction")
        or raw.get("sat:orbit_state")
        or props.get("sat:orbit_state")
    )
    mission = (
        props.get("mission")
        or getattr(item.acquisition, "platform", "")
        or raw.get("platform")
        or ""
    )
    return track_group_key(
        str(mission),
        relative_orbit,
        str(direction or ""),
    )


def local_destination(
    work_dir: str | Path,
    item: CatalogItem,
    asset: RemoteAsset,
    *,
    track_key: str | None = None,
) -> Path:
    """Build a stable, confined local SAFE destination for one catalog item.

    Track-grouped workspaces stage downloads under
    ``tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>``; when the item's orbit
    group cannot be resolved and no ``track_key`` is passed the legacy
    ``inputs/`` root is used so older workspaces reopen.
    """
    from faninsar.stack.layout import acquisition_staging_dir, staged_acquisition_path

    key = track_key or _item_track_key(item)
    safe_id = _SAFE_NAME.sub("_", item.item_id).strip("._") or "s1-product"
    suffix = Path(urllib.parse.urlsplit(asset.href).path).suffix.casefold()
    if suffix not in _SAFE_SUFFIXES:
        suffix = ".zip"
    return staged_acquisition_path(
        acquisition_staging_dir(work_dir, key),
        f"{safe_id}{suffix}",
    )


def select_orbit_bindings(
    candidates: Mapping[str, object] | Sequence[str | Path] | None,
    intervals: Mapping[str, tuple[datetime, datetime]],
    *,
    mission_ids: Mapping[str, str] | None = None,
    require_coverage: bool = True,
) -> dict[str, Path]:
    """Select covering POEORB files before RESORB files for each acquisition.

    Parameters
    ----------
    candidates : mapping, sequence, or None
        Explicit local EOF candidates. A mapping may bind candidates to date
        IDs; a sequence is considered for every acquisition.
    intervals : mapping
        Acquisition sensing intervals keyed by Stack date ID.
    mission_ids : mapping, optional
        SAFE ``mission_id`` (``S1A``/``S1B``/``S1C``/``S1D``) per date ID.
        When provided, filenames that do not start with that prefix are
        ignored.
    require_coverage : bool, default True
        When true, a configured candidate set must cover every interval.
        Auto-fetch uses ``False`` so uncovered dates keep SAFE-embedded
        restituted vectors.

    Returns
    -------
    dict
        Date-to-orbit bindings. An empty mapping means embedded SAFE vectors
        remain authoritative because no external candidates were configured.

    Raises
    ------
    ValueError
        If ``require_coverage`` is true and configured candidates do not
        provide one covering valid orbit for every acquisition.

    """
    if candidates is None:
        return {}
    from faninsar.missions.s1.orbit import read_eof_orbit

    if isinstance(candidates, Mapping):
        by_date = {str(key): _as_paths(value) for key, value in candidates.items()}
    else:
        shared = _as_paths(candidates)
        by_date = dict.fromkeys(intervals, shared)
    selected: dict[str, Path] = {}
    missions = {
        str(key): str(value).upper() for key, value in (mission_ids or {}).items()
    }
    for date_id, interval in intervals.items():
        values = by_date.get(date_id, ())
        qualified: list[tuple[int, float, str, Path]] = []
        sensing_midpoint = interval[0] + (interval[1] - interval[0]) / 2
        mission = missions.get(date_id, "")
        for candidate in values:
            try:
                metadata = read_eof_orbit(candidate)
            except (OSError, ParseError, RuntimeError, ValueError):
                logger.warning("ignoring invalid S1 orbit candidate %s", candidate)
                continue
            if not metadata.vectors:
                continue
            start = min(vector.time for vector in metadata.vectors)
            stop = max(vector.time for vector in metadata.vectors)
            if start <= interval[0] and stop >= interval[1]:
                name = candidate.name.upper()
                if mission in {"S1A", "S1B", "S1C", "S1D"} and not name.startswith(
                    mission
                ):
                    continue
                if "POEORB" in name or "POE_ORB" in name:
                    rank = 0
                elif "RESORB" in name or "RES_ORB" in name:
                    rank = 1
                else:
                    continue
                orbit_midpoint = start + (stop - start) / 2
                distance = abs((orbit_midpoint - sensing_midpoint).total_seconds())
                qualified.append((rank, distance, candidate.as_posix(), candidate))
        if not qualified:
            if not require_coverage:
                continue
            message = (
                f"configured S1 orbit candidates do not cover acquisition {date_id}"
            )
            reject(message, ValueError, logger=logger)
        selected[date_id] = min(qualified)[3]
    return selected


def _as_paths(value: object) -> tuple[Path, ...]:
    """Normalize one orbit candidate binding to path values."""
    if isinstance(value, (str, Path)):
        return (Path(value),)
    if isinstance(value, Sequence):
        return tuple(Path(item) for item in value)
    message = f"unsupported S1 orbit candidate binding: {value!r}"
    return reject(message, TypeError, logger=logger)


def catalog_for_product(product_type: S1ProductType) -> str:
    """Return the stable ASF registration shared by S1 product families."""
    if product_type not in ("burst", "scene"):
        message = "product_type must be 'burst' or 'scene'"
        reject(message, ValueError, logger=logger)
    return "sentinel1-asf"


def ensure_asf_catalog(
    name: str | None = None,
    *,
    product_type: S1ProductType = "burst",
) -> str:
    """Ensure the stable Sentinel-1 ASF catalog registration exists."""
    from faninsar.remote.catalog import _ADAPTERS, _REGISTRY_LOCK
    from faninsar.remote.providers.asf_search import ASFSearchAdapter

    stable_name = catalog_for_product(product_type)
    if name is not None and name != stable_name:
        message = (
            f"Sentinel-1 ASF must use the stable registration name {stable_name!r}"
        )
        reject(message, ValueError, logger=logger)
    name = stable_name
    collection = "sentinel-1"

    expected = ASFSearchAdapter(collection=collection)
    identity_fields = (
        "provider",
        "collection",
        "query_collection",
        "engine",
        "endpoint",
        "profiles",
        "data_origins",
        "path_prefixes",
        "redirect_origins",
    )
    with _REGISTRY_LOCK:
        if name in _ADAPTERS:
            adapter = _ADAPTERS[name]
            if any(
                getattr(adapter, field, None) != getattr(expected, field)
                for field in identity_fields
            ):
                message = f"catalog {name!r} has an incompatible ASF registration"
                reject(message, ValueError, logger=logger)
            return name
        _ADAPTERS[name] = expected
    return name


__all__ = [
    "S1DualPolarization",
    "S1Polarization",
    "S1PolarizationRequest",
    "S1ProductType",
    "canonical_items",
    "catalog_for_product",
    "complete_safe_asset",
    "ensure_asf_catalog",
    "filter_burst_polarizations",
    "local_destination",
    "normalize_polarizations",
    "product_filters",
    "provider_polarization_filter",
    "select_orbit_bindings",
]
