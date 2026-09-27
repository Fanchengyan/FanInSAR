"""Sentinel-1 search profile and local input preparation helpers.

The profile translates the small Sentinel-1 vocabulary into the existing ASF
search adapter.  Transfer and authentication remain owned by
``faninsar.remote``; this module only selects a qualified SAFE representation
and hands it to the local SAFE reader.
"""

from __future__ import annotations

import re
import urllib.parse
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from xml.etree.ElementTree import ParseError

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.remote.records import CatalogItem, RemoteAsset

logger = setup_logger(__name__)

S1ProductType = Literal["burst", "scene"]
S1SinglePolarization = Literal["VV", "VH", "HH", "HV"]
S1DualPolarization = Literal["VV+VH", "HH+HV"]
S1Polarization = S1SinglePolarization | S1DualPolarization
S1PolarizationRequest = tuple[S1Polarization, ...] | Literal["all"]

_SINGLE_POLARIZATIONS = frozenset({"VV", "VH", "HH", "HV"})
_DUAL_POLARIZATIONS = frozenset({"VV+VH", "HH+HV"})
_POLARIZATIONS = _SINGLE_POLARIZATIONS | _DUAL_POLARIZATIONS
_SAFE_SUFFIXES = frozenset({".zip"})
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


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
        logger.error(message)
        raise ValueError(message)
    if len(values) == 1 and values[0] == "ALL":
        return "all"
    invalid = set(values).difference(_POLARIZATIONS)
    if invalid:
        allowed = ", ".join(sorted(_POLARIZATIONS))
        message = (
            "unsupported Sentinel-1 polarization(s): "
            f"{sorted(invalid)}; allowed={allowed}"
        )
        logger.error(message)
        raise ValueError(message)
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
        logger.error(message)
        raise ValueError(message)
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
        logger.error(message)
        raise ValueError(message)
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
            logger.error(message)
            raise ValueError(message)
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


def local_destination(
    work_dir: str | Path,
    item: CatalogItem,
    asset: RemoteAsset,
) -> Path:
    """Build a stable, confined local SAFE destination for one catalog item."""
    safe_id = _SAFE_NAME.sub("_", item.item_id).strip("._") or "s1-product"
    suffix = Path(urllib.parse.urlsplit(asset.href).path).suffix.casefold()
    if suffix not in _SAFE_SUFFIXES:
        suffix = ".zip"
    return Path(work_dir) / "inputs" / f"{safe_id}{suffix}"


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
        SAFE ``mission_id`` (``S1A``/``S1B``/``S1C``) per date ID. When
        provided, filenames that do not start with that prefix are ignored.
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
                if mission in {"S1A", "S1B", "S1C"} and not name.startswith(mission):
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
            logger.error(message)
            raise ValueError(message)
        selected[date_id] = min(qualified)[3]
    return selected


def safe_requires_aux_cal(path: str | Path) -> bool:
    """Return whether a SAFE was produced by Sentinel-1 IPF 002.36."""
    source = Path(path)
    payload: bytes | None = None
    if source.is_dir():
        candidates = [source / "manifest.safe", *source.glob("*.SAFE/manifest.safe")]
        manifest = next((item for item in candidates if item.is_file()), None)
        if manifest is not None:
            payload = manifest.read_bytes()
    elif source.suffix.casefold() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = [
                name for name in archive.namelist() if name.endswith("manifest.safe")
            ]
            if len(names) == 1:
                payload = archive.read(names[0])
    if payload is None:
        return False
    from xml.etree import ElementTree as ET

    root = ET.fromstring(payload)
    return any(
        element.tag.rsplit("}", 1)[-1] == "software"
        and "002.36" in str(element.attrib.get("version", ""))
        for element in root.iter()
    )


def _as_paths(value: object) -> tuple[Path, ...]:
    """Normalize one orbit candidate binding to path values."""
    if isinstance(value, (str, Path)):
        return (Path(value),)
    if isinstance(value, Sequence):
        return tuple(Path(item) for item in value)
    message = f"unsupported S1 orbit candidate binding: {value!r}"
    logger.error(message)
    raise TypeError(message)


def validate_calibration_paths(
    paths: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    """Validate explicitly bound local AUX_CAL/calibration files."""
    if paths is None:
        return ()
    result = tuple(Path(path) for path in paths)
    missing = [path for path in result if not path.is_file()]
    if missing:
        message = f"configured S1 calibration file is missing: {missing[0]}"
        logger.error(message)
        raise FileNotFoundError(message)
    return result


def catalog_for_product(product_type: S1ProductType) -> str:
    """Return the stable ASF registration shared by S1 product families."""
    if product_type not in ("burst", "scene"):
        message = "product_type must be 'burst' or 'scene'"
        logger.error(message)
        raise ValueError(message)
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
        logger.error(message)
        raise ValueError(message)
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
                logger.error(message)
                raise ValueError(message)
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
    "safe_requires_aux_cal",
    "select_orbit_bindings",
    "validate_calibration_paths",
]
