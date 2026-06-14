"""Metadata types and schema for FanInSAR frame products."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from os import PathLike

logger = setup_logger(__name__)

GeometryAssetName = Literal[
    "incidence",
    "azimuth",
    "heading",
    "dem",
    "water_mask",
]

InterferogramAssetName = Literal[
    "unw_phase",
    "coherence",
    "los_disp",
    "vert_disp",
    "wrapped_phase",
    "amplitude",
    "conncomp",
]

GEOMETRY_ASSETS: tuple[GeometryAssetName, ...] = (
    "incidence",
    "azimuth",
    "heading",
    "dem",
    "water_mask",
)

INTERFEROGRAM_ASSETS: tuple[InterferogramAssetName, ...] = (
    "unw_phase",
    "coherence",
    "los_disp",
    "vert_disp",
    "wrapped_phase",
    "amplitude",
    "conncomp",
)

CONTINUOUS_ASSETS: frozenset[str] = frozenset(
    {
        "incidence",
        "azimuth",
        "heading",
        "dem",
        "unw_phase",
        "los_disp",
        "vert_disp",
        "wrapped_phase",
        "amplitude",
    }
)

CATEGORICAL_ASSETS: frozenset[str] = frozenset({"water_mask", "conncomp"})

METADATA_VERSION = "0.1.0"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_geometry_metadata(
    *,
    crs: Any,
    width: int,
    height: int,
    transform: Any,
    bounds: tuple[float, float, float, float],
    resolution: tuple[float, float],
    angle_unit: str = "degree",
    azimuth_convention: str = "look_azimuth_from_north_clockwise",
    heading_convention: str = "satellite_heading_from_north_clockwise",
    assets: dict[str, dict[str, Any]],
    source_assets: dict[str, dict[str, Any]] | None = None,
    processing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a geometry.json metadata dictionary."""
    meta: dict[str, Any] = {
        "type": "FrameGeometry",
        "version": METADATA_VERSION,
        "crs": str(crs),
        "width": width,
        "height": height,
        "transform": list(transform)[:6]
        if hasattr(transform, "__iter__")
        else str(transform),
        "bounds": list(bounds),
        "resolution": list(resolution),
        "angle_unit": angle_unit,
        "azimuth_convention": azimuth_convention,
        "heading_convention": heading_convention,
        "assets": assets,
        "created_at": _utc_now_iso(),
    }
    if source_assets is not None:
        meta["source_assets"] = source_assets
    if processing is not None:
        meta["processing"] = processing
    return meta


def build_item_metadata(
    *,
    pair_name: str,
    reference_date: str,
    secondary_date: str,
    grid: dict[str, Any],
    assets: dict[str, Any],
    geometry_href: str = "../../geometry/geometry.json",
    source_product_id: str | None = None,
    source_processor: str | None = None,
    baseline: float | None = None,
    temporal_baseline_days: int | None = None,
    reference_granule: str | None = None,
    secondary_granule: str | None = None,
    pass_direction: str | None = None,
    orbit_number: int | None = None,
    heading: float | None = None,
    looks: str | None = None,
    reference_point: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an item.json metadata dictionary for a pair."""
    item: dict[str, Any] = {
        "type": "FrameInterferogramItem",
        "version": METADATA_VERSION,
        "pair_name": pair_name,
        "reference_date": reference_date,
        "secondary_date": secondary_date,
        "grid": grid,
        "geometry_href": geometry_href,
        "assets": assets,
        "created_at": _utc_now_iso(),
    }
    if source_product_id is not None:
        item["source_product_id"] = source_product_id
    if source_processor is not None:
        item["source_processor"] = source_processor
    if baseline is not None:
        item["baseline"] = baseline
    if temporal_baseline_days is not None:
        item["temporal_baseline_days"] = temporal_baseline_days
    if reference_granule is not None:
        item["reference_granule"] = reference_granule
    if secondary_granule is not None:
        item["secondary_granule"] = secondary_granule
    if pass_direction is not None:
        item["pass_direction"] = pass_direction
    if orbit_number is not None:
        item["orbit_number"] = orbit_number
    if heading is not None:
        item["heading"] = heading
    if looks is not None:
        item["looks"] = looks
    if reference_point is not None:
        item["reference_point"] = reference_point
    return item


def build_interferograms_index(
    *,
    pair_count: int,
    pairs: list[str],
    assets_by_pair: dict[str, list[str]],
    common_grid: bool,
    geometry_href: str | None = None,
) -> dict[str, Any]:
    """Build an interferograms_index.json metadata dictionary."""
    index: dict[str, Any] = {
        "type": "FrameInterferogramIndex",
        "version": METADATA_VERSION,
        "pair_count": pair_count,
        "pairs": pairs,
        "assets_by_pair": assets_by_pair,
        "common_grid": common_grid,
        "created_at": _utc_now_iso(),
    }
    if geometry_href is not None:
        index["geometry_href"] = geometry_href
    return index


def save_json(data: dict[str, Any], path: str | PathLike) -> None:
    """Write a JSON-serializable dictionary to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | PathLike) -> dict[str, Any]:
    """Load a JSON file and return the parsed dictionary."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_geometry_href(ifgs_dir: str | Path) -> str | None:
    """Return the relative href from an interferograms dir to its sibling geometry.

    This is the single source of truth for the ``geometry_href`` pointer that
    appears in ``item.json`` and ``interferograms_index.json``. It replaces
    every previously-hardcoded ``"../../geometry/geometry.json"`` string.

    Parameters
    ----------
    ifgs_dir : str or Path
        Path to the ``frame/interferograms`` directory.

    Returns
    -------
    str or None
        POSIX relative path (e.g. ``"../geometry/geometry.json"``) to the
        sibling ``geometry/geometry.json`` when it exists, otherwise ``None``.

    """
    ifgs_dir = Path(ifgs_dir).resolve()
    geom_json = ifgs_dir.parent / "geometry" / "geometry.json"
    if not geom_json.exists():
        return None
    # geom_json is a sibling of ifgs_dir, so use os.path.relpath (handles "..").
    import os

    return Path(os.path.relpath(geom_json, ifgs_dir)).as_posix()


# Backwards-compat alias — older code may import build_ifg_index.
build_ifg_index = build_interferograms_index
