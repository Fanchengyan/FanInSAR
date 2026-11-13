"""Normalize committed products into public STAC Items (PROPOSAL-0089).

Normalization is metadata-only. It reads the canonical Item already carried by
a :class:`~faninsar.io.storage.product.Product`, copies its scientific
properties and assets, and derives public identity, temporal coverage, and true
geographic geometry. It never opens, reads, hashes, or rewrites a payload or a
canonical ``item.json``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from faninsar.io.stac.record import (
    PROJECTION_EXTENSION,
    NormalizationReason,
    StacNormalizationError,
    StacRecord,
    TemporalCoverage,
    public_item_id,
)

if TYPE_CHECKING:
    from faninsar.io.storage.product import Product

__all__ = [
    "StacNormalizationError",
    "normalize_product",
]

_FAMILY_BY_KIND: dict[str, str] = {
    "scene": "scenes",
    "ifg": "interferograms",
    "unwrap": "interferograms",
    "complex_ion": "interferograms",
    "unwrap_ion": "interferograms",
    "ion": "interferograms",
    "ion_corrected": "interferograms",
    "timeseries": "time-series",
    "mask": "masks",
}

_GEOCODED = "geocoded"
_GEO = "geo"
_RADAR = "radar"
_PAIR_ROLES = ("primary", "secondary")

_INVALID: NormalizationReason = "invalid-item"
_UNSUPPORTED: NormalizationReason = "unsupported-kind"
_MISSING_TEMPORAL: NormalizationReason = "missing-temporal"
_MISSING_SOURCE: NormalizationReason = "missing-source"
_CYCLIC: NormalizationReason = "cyclic-inheritance"
_UNSAFE: NormalizationReason = "unsafe-asset"
_OUTSIDE: NormalizationReason = "outside-work-dir"


def normalize_product(
    product: Product,
    *,
    work_dir: Path,
    products_by_write_id: Mapping[str, Product] | None = None,
) -> StacRecord:
    """Normalize one committed product into a public STAC record.

    Parameters
    ----------
    product : Product
        Physically committed product with a parsed canonical ``item.json``.
    work_dir : pathlib.Path
        Workspace root. The product directory and every registered asset
        must resolve inside it.
    products_by_write_id : mapping, optional
        Metadata-only current products keyed by ``faninsar:write_id``. It is
        used to inherit a missing ``geocoded`` family and missing temporal
        evidence through recorded ``faninsar:inputs``. ``None`` disables
        inheritance so direct unit tests stay local.

    Returns
    -------
    StacRecord
        Immutable normalized record. Its Item is a strict-JSON STAC Feature and
        its assets point at registered product paths, never at absolute paths.

    Raises
    ------
    StacNormalizationError
        If the product is outside ``work_dir``, has an unsupported kind or an
        unsafe asset href, lacks resolvable temporal evidence, or inherits from
        a missing or cyclic source.

    """
    work_root = Path(work_dir).resolve()
    return _normalize(
        product,
        work_root=work_root,
        products_by_write_id=products_by_write_id,
        chain=(),
        cache={},
    )


def _normalize(
    product: Product,
    *,
    work_root: Path,
    products_by_write_id: Mapping[str, Product] | None,
    chain: tuple[str, ...],
    cache: dict[str, StacRecord],
) -> StacRecord:
    """Normalize one product, resolving metadata dependencies recursively."""
    properties = _product_properties(product)
    write_id = _required_string(properties, "faninsar:write_id")
    if write_id in chain:
        message = f"write id {write_id!r} is already on the inheritance chain"
        raise StacNormalizationError(_CYCLIC, message)
    if write_id in cache:
        return cache[write_id]
    kind = _required_string(properties, "faninsar:kind")
    next_chain = (*chain, write_id)
    inherited_pair = None
    if kind == _GEOCODED:
        source = _geocoded_source(
            properties,
            write_id=write_id,
            chain=next_chain,
            work_root=work_root,
            products_by_write_id=products_by_write_id,
            cache=cache,
        )
        family = source.family
        temporal = source.temporal
        inherited_pair = source.item["properties"].get("faninsar:pair")
    else:
        family = _family_for(kind, write_id)
        temporal = _temporal_coverage(
            properties,
            write_id=write_id,
            chain=next_chain,
            work_root=work_root,
            products_by_write_id=products_by_write_id,
            cache=cache,
        )
    source_item_path, source_relative_path = _source_paths(product.path, work_root)

    item = _canonical_copy(product.item)
    _validate_extensions(item, product.path)
    raw_properties = item.get("properties")
    if not isinstance(raw_properties, dict):
        message = f"product {product.path} has no properties mapping"
        raise StacNormalizationError(_INVALID, message)
    assets = item.get("assets")
    if not isinstance(assets, dict):
        message = f"product {product.path} has no assets mapping"
        raise StacNormalizationError(_INVALID, message)
    source_item_id = item.get("id")
    if not isinstance(source_item_id, str) or not source_item_id:
        message = f"product {product.path} has no canonical Item id"
        raise StacNormalizationError(_INVALID, message)

    public_id = public_item_id(
        kind=kind,
        source_relative_path=source_relative_path,
        write_id=write_id,
    )
    asset_paths = _normalized_assets(
        assets,
        product_root=source_item_path.parent,
        work_root=work_root,
    )
    geometry, bbox = _spatial_metadata(raw_properties, kind=kind, write_id=write_id)

    raw_properties.pop("collection", None)
    raw_properties["faninsar:family"] = family
    raw_properties["faninsar:source_item"] = _source_item_href(source_relative_path)
    raw_properties["faninsar:source_item_id"] = source_item_id
    if inherited_pair is not None and "faninsar:pair" not in raw_properties:
        raw_properties["faninsar:pair"] = _json_copy(inherited_pair)
    _apply_temporal(raw_properties, temporal)

    item["type"] = "Feature"
    item["stac_version"] = "1.0.0"
    item["id"] = public_id
    item["geometry"] = geometry
    if bbox is None:
        item.pop("bbox", None)
    else:
        item["bbox"] = list(bbox)
    item.pop("collection", None)
    item["properties"] = raw_properties
    if geometry is not None:
        _add_projection_extension(item, product.path)

    _validate_strict_json(item, product.path)
    record = StacRecord(
        public_id=public_id,
        write_id=write_id,
        kind=kind,
        family=family,
        source_item_path=source_item_path,
        source_relative_path=source_relative_path,
        collection_id=None,
        temporal=temporal,
        bbox=bbox,
        item=item,
        asset_paths=asset_paths,
    )
    cache[write_id] = record
    return record


def _product_properties(product: Product) -> dict[str, Any]:
    """Return a validated copy of the canonical properties mapping."""
    item = product.item
    properties = item.get("properties") if isinstance(item, Mapping) else None
    if not isinstance(properties, Mapping):
        message = f"product {product.path} has no properties mapping"
        raise StacNormalizationError(_INVALID, message)
    return dict(properties)


def _required_string(properties: Mapping[str, Any], field: str) -> str:
    """Return one non-empty string property or reject the product."""
    value = properties.get(field)
    if not isinstance(value, str) or not value:
        message = f"property {field!r} must be a non-empty string"
        raise StacNormalizationError(_INVALID, message)
    return value


def _family_for(kind: str, write_id: str) -> str:
    """Return the logical family for one public kind."""
    family = _FAMILY_BY_KIND.get(kind)
    if family is None:
        message = f"kind {kind!r} is not a public product kind ({write_id})"
        raise StacNormalizationError(_UNSUPPORTED, message)
    return family


def _inputs(properties: Mapping[str, Any]) -> dict[str, str]:
    """Return the recorded upstream write ids as a string mapping."""
    raw = properties.get("faninsar:inputs", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        message = "property 'faninsar:inputs' must be a mapping"
        raise StacNormalizationError(_INVALID, message)
    inputs: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            message = "property 'faninsar:inputs' has an invalid key"
            raise StacNormalizationError(_INVALID, message)
        if not isinstance(value, str) or not value:
            message = f"input {name!r} must record a non-empty write id"
            raise StacNormalizationError(_INVALID, message)
        inputs[name] = value
    return inputs


def _dependency_record(
    write_id: str,
    *,
    name: str,
    chain: tuple[str, ...],
    work_root: Path,
    products_by_write_id: Mapping[str, Product] | None,
    cache: dict[str, StacRecord],
) -> StacRecord:
    """Normalize one recorded current input, or report why it is inadmissible."""
    if write_id in chain:
        message = f"input {name!r} write id {write_id!r} starts a cycle"
        raise StacNormalizationError(_CYCLIC, message)
    if write_id in cache:
        return cache[write_id]
    if products_by_write_id is None:
        message = (
            f"input {name!r} write id {write_id!r} is missing and inheritance "
            "is disabled"
        )
        raise StacNormalizationError(_MISSING_SOURCE, message)
    dependency = products_by_write_id.get(write_id)
    if dependency is None:
        message = f"input {name!r} write id {write_id!r} is missing"
        raise StacNormalizationError(_MISSING_SOURCE, message)
    try:
        return _normalize(
            dependency,
            work_root=work_root,
            products_by_write_id=products_by_write_id,
            chain=chain,
            cache=cache,
        )
    except StacNormalizationError as error:
        if error.reason == "cyclic-inheritance":
            raise
        message = (
            f"input {name!r} write id {write_id!r} failed normalization: "
            f"{error.reason}: {error.detail}"
        )
        raise StacNormalizationError(_MISSING_SOURCE, message) from error


def _geocoded_source(
    properties: Mapping[str, Any],
    *,
    write_id: str,
    chain: tuple[str, ...],
    work_root: Path,
    products_by_write_id: Mapping[str, Product] | None,
    cache: dict[str, StacRecord],
) -> StacRecord:
    """Return the admitted source record of one geocoded child."""
    source_id = _inputs(properties).get("source")
    if not isinstance(source_id, str) or not source_id:
        message = f"geocoded product {write_id!r} has no recorded source input"
        raise StacNormalizationError(_MISSING_SOURCE, message)
    return _dependency_record(
        source_id,
        name="source",
        chain=chain,
        work_root=work_root,
        products_by_write_id=products_by_write_id,
        cache=cache,
    )


def _temporal_coverage(
    properties: Mapping[str, Any],
    *,
    write_id: str,
    chain: tuple[str, ...],
    work_root: Path,
    products_by_write_id: Mapping[str, Product] | None,
    cache: dict[str, StacRecord],
) -> TemporalCoverage:
    """Resolve persisted temporal evidence for one non-geocoded product."""
    precise = _precise_evidence(properties)
    if precise is not None and precise[1]:
        return precise[0]
    date_coverage = _date_field_coverage(properties)
    if date_coverage is not None:
        return date_coverage
    pair_coverage = _pair_input_coverage(
        properties,
        chain=chain,
        work_root=work_root,
        products_by_write_id=products_by_write_id,
        cache=cache,
    )
    if pair_coverage is not None:
        return pair_coverage
    if precise is not None:
        return precise[0]
    return _inherited_coverage(
        properties,
        write_id=write_id,
        chain=chain,
        work_root=work_root,
        products_by_write_id=products_by_write_id,
        cache=cache,
    )


def _precise_evidence(
    properties: Mapping[str, Any],
) -> tuple[TemporalCoverage, bool] | None:
    """Return persisted timestamp coverage and whether it is precise.

    The boolean is true when any persisted UTC time is not exactly midnight, so
    the interval is trusted over date-only metadata.
    """
    candidates: list[tuple[TemporalCoverage, bool]] = []
    raw_datetime = properties.get("datetime")
    if raw_datetime is not None:
        instant = _parse_timestamp(raw_datetime, "datetime")
        candidates.append(
            (
                TemporalCoverage(start=instant, end=instant, precision="timestamp"),
                not _is_midnight(instant),
            )
        )
    raw_start = properties.get("start_datetime")
    raw_end = properties.get("end_datetime")
    if raw_start is not None or raw_end is not None:
        if raw_start is None or raw_end is None:
            message = (
                "properties 'start_datetime' and 'end_datetime' must appear together"
            )
            raise StacNormalizationError(_INVALID, message)
        start = _parse_timestamp(raw_start, "start_datetime")
        end = _parse_timestamp(raw_end, "end_datetime")
        candidates.append(
            (
                TemporalCoverage(
                    start=min(start, end),
                    end=max(start, end),
                    precision="timestamp",
                ),
                not (_is_midnight(start) and _is_midnight(end)),
            )
        )
    if not candidates:
        return None
    for candidate in candidates:
        if candidate[1]:
            return candidate
    return candidates[0]


def _date_field_coverage(
    properties: Mapping[str, Any],
) -> TemporalCoverage | None:
    """Return date-only coverage from date, dates, or a date-shaped pair."""
    days: list[date] = []
    raw_date = properties.get("faninsar:date")
    if raw_date is not None:
        day = _parse_day(raw_date)
        if day is None:
            message = (
                f"property 'faninsar:date' is not a UTC calendar day: {raw_date!r}"
            )
            raise StacNormalizationError(_INVALID, message)
        days.append(day)
    raw_dates = properties.get("faninsar:dates")
    if raw_dates is not None:
        if not isinstance(raw_dates, list | tuple) or not raw_dates:
            message = "property 'faninsar:dates' must be a non-empty list"
            raise StacNormalizationError(_INVALID, message)
        for value in raw_dates:
            day = _parse_day(value)
            if day is None:
                message = (
                    f"property 'faninsar:dates' contains a non-date value: {value!r}"
                )
                raise StacNormalizationError(_INVALID, message)
            days.append(day)
    raw_pair = properties.get("faninsar:pair")
    if raw_pair is not None:
        pair_days = _pair_days(raw_pair)
        if pair_days is not None:
            days.extend(pair_days)
    if not days:
        return None
    return _day_coverage(days)


def _pair_input_coverage(
    properties: Mapping[str, Any],
    *,
    chain: tuple[str, ...],
    work_root: Path,
    products_by_write_id: Mapping[str, Product] | None,
    cache: dict[str, StacRecord],
) -> TemporalCoverage | None:
    """Resolve a non-date-shaped pair through its primary/secondary inputs."""
    raw_pair = properties.get("faninsar:pair")
    if raw_pair is None or _pair_days(raw_pair) is not None:
        return None
    if products_by_write_id is None:
        return None
    inputs = _inputs(properties)
    coverages: list[TemporalCoverage] = []
    for role in _PAIR_ROLES:
        write_id = inputs.get(role)
        if write_id is None:
            continue
        coverages.append(
            _dependency_record(
                write_id,
                name=role,
                chain=chain,
                work_root=work_root,
                products_by_write_id=products_by_write_id,
                cache=cache,
            ).temporal
        )
    if not coverages:
        return None
    return _union_coverage(coverages)


def _inherited_coverage(
    properties: Mapping[str, Any],
    *,
    write_id: str,
    chain: tuple[str, ...],
    work_root: Path,
    products_by_write_id: Mapping[str, Product] | None,
    cache: dict[str, StacRecord],
) -> TemporalCoverage:
    """Inherit temporal coverage through every recorded current input."""
    inputs = _inputs(properties)
    if not inputs:
        message = f"product {write_id!r} has no temporal evidence"
        raise StacNormalizationError(_MISSING_TEMPORAL, message)
    if products_by_write_id is None:
        message = (
            f"product {write_id!r} has no temporal evidence and inheritance is disabled"
        )
        raise StacNormalizationError(_MISSING_TEMPORAL, message)
    coverages = [
        _dependency_record(
            input_write_id,
            name=name,
            chain=chain,
            work_root=work_root,
            products_by_write_id=products_by_write_id,
            cache=cache,
        ).temporal
        for name, input_write_id in sorted(inputs.items())
    ]
    return _union_coverage(coverages)


def _union_coverage(coverages: list[TemporalCoverage]) -> TemporalCoverage:
    """Return the union of contributing coverage, day only when all are day."""
    precision = (
        "day"
        if all(coverage.precision == "day" for coverage in coverages)
        else "timestamp"
    )
    return TemporalCoverage(
        start=min(coverage.start for coverage in coverages),
        end=max(coverage.end for coverage in coverages),
        precision=precision,
    )


def _pair_days(value: object) -> list[date] | None:
    """Return both pair days, or ``None`` when the pair is not date-shaped."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        message = "property 'faninsar:pair' must contain two date identifiers"
        raise StacNormalizationError(_INVALID, message)
    if any(not isinstance(item, str) or not item for item in value):
        message = "property 'faninsar:pair' must contain two non-empty strings"
        raise StacNormalizationError(_INVALID, message)
    first = _parse_day(value[0])
    second = _parse_day(value[1])
    if first is None or second is None:
        return None
    return [first, second]


def _parse_day(value: object) -> date | None:
    """Return the UTC calendar day in ``value``, or ``None`` when not a date."""
    if not isinstance(value, str):
        return None
    if len(value) == 8 and value.isdigit():
        try:
            return date(int(value[:4]), int(value[4:6]), int(value[6:]))
        except ValueError:
            return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_timestamp(value: object, field: str) -> datetime:
    """Parse one persisted RFC 3339 timestamp and normalize it to UTC."""
    if not isinstance(value, str) or not value or "T" not in value:
        message = f"property {field!r} must be an RFC 3339 timestamp"
        raise StacNormalizationError(_INVALID, message)
    text = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        message = f"property {field!r} is not a timestamp: {value!r}"
        raise StacNormalizationError(_INVALID, message) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_midnight(value: datetime) -> bool:
    """Return whether a UTC timestamp is exactly midnight."""
    return value.time() == time(0, 0, 0, 0)


def _day_coverage(days: list[date]) -> TemporalCoverage:
    """Return the explicit UTC calendar-day interval covering ``days``."""
    first = min(days)
    last = max(days)
    start = datetime.combine(first, time(0, 0, 0, 0), tzinfo=UTC)
    end = datetime.combine(last, time(23, 59, 59, 999999), tzinfo=UTC)
    return TemporalCoverage(start=start, end=end, precision="day")


def _source_paths(product_path: Path, work_root: Path) -> tuple[Path, str]:
    """Return the canonical Item path and POSIX product path under ``work_root``."""
    if product_path.is_symlink() or _has_lexical_traversal(product_path):
        message = f"product path is unsafe: {product_path}"
        raise StacNormalizationError(_OUTSIDE, message)
    _reject_symlinked_components(product_path, work_root)
    resolved = product_path.resolve()
    if not resolved.is_relative_to(work_root):
        message = f"product path is outside work_dir: {product_path}"
        raise StacNormalizationError(_OUTSIDE, message)
    relative = resolved.relative_to(work_root).as_posix()
    return resolved / "item.json", relative


def _has_lexical_traversal(path: Path) -> bool:
    """Return whether a lexical path component is ``..``."""
    return any(part == ".." for part in path.parts)


def _reject_symlinked_components(path: Path, work_root: Path) -> None:
    """Reject a product reached through a symlinked directory under the root."""
    try:
        relative = path.absolute().relative_to(work_root)
    except ValueError:
        return
    current = work_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            message = f"product path crosses a symbolic link: {current}"
            raise StacNormalizationError(_OUTSIDE, message)


def _source_item_href(source_relative_path: str) -> str:
    """Return the relative canonical Item pointer."""
    if source_relative_path in {"", "."}:
        return "item.json"
    return str(PurePosixPath(source_relative_path) / "item.json")


def _normalized_assets(
    assets: Mapping[str, Any], *, product_root: Path, work_root: Path
) -> dict[str, Path]:
    """Validate canonical assets and register their absolute paths."""
    paths: dict[str, Path] = {}
    for name, asset in assets.items():
        if not isinstance(asset, Mapping):
            message = f"asset {name!r} is not a mapping"
            raise StacNormalizationError(_INVALID, message)
        href = asset.get("href")
        if not isinstance(href, str) or not _is_safe_basename(href):
            message = f"asset {name!r} has an unsafe href: {href!r}"
            raise StacNormalizationError(_UNSAFE, message)
        path = product_root / href
        if path.is_symlink():
            message = f"asset {name!r} is a symbolic link: {path}"
            raise StacNormalizationError(_UNSAFE, message)
        paths[str(name)] = path
    for name, path in paths.items():
        try:
            relative = path.relative_to(work_root)
        except ValueError as error:
            message = f"asset {name!r} is outside work_dir: {path}"
            raise StacNormalizationError(_OUTSIDE, message) from error
        assets[name]["href"] = relative.as_posix()
    return paths


def _is_safe_basename(href: str) -> bool:
    """Return whether ``href`` is one plain local basename."""
    if (
        not href
        or href in {".", ".."}
        or "\\" in href
        or "://" in href
        or "\x00" in href
    ):
        return False
    candidate = Path(href)
    return (
        not candidate.is_absolute()
        and len(candidate.parts) == 1
        and candidate.name == href
    )


def _spatial_metadata(
    properties: dict[str, Any], *, kind: str, write_id: str
) -> tuple[dict[str, Any] | None, tuple[float, float, float, float] | None]:
    """Derive true geographic geometry or preserve radar/unknown coordinates."""
    domain = _domain(properties, kind)
    if domain == _RADAR:
        _drop_projection(properties)
        return None, None
    if domain == _GEO:
        grid = _resolve_grid(properties, geocoded=kind == _GEOCODED)
        if grid is None:
            message = f"geo product {write_id!r} has no complete spatial metadata"
            raise StacNormalizationError(_INVALID, message)
        return _geometry_from_grid(properties, grid)
    if domain is not None:
        message = f"product {write_id!r} has unsupported domain {domain!r}"
        raise StacNormalizationError(_INVALID, message)
    if kind == _GEOCODED:
        message = f"geocoded product {write_id!r} has no geographic grid"
        raise StacNormalizationError(_INVALID, message)
    grid = _optional_grid_from_payload(properties.get("faninsar:grid"))
    if grid is not None:
        return _geometry_from_grid(properties, grid)
    _drop_projection(properties)
    return None, None


def _domain(properties: Mapping[str, Any], kind: str) -> str | None:
    """Return the persisted domain, using the mask-specific field for masks."""
    raw: object = None
    if kind == "mask" and "faninsar:mask_domain" in properties:
        raw = properties.get("faninsar:mask_domain")
    elif "faninsar:domain" in properties:
        raw = properties.get("faninsar:domain")
    elif kind == _GEOCODED:
        return _GEO
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in {_GEO, _RADAR}:
        message = f"property 'faninsar:domain' is invalid: {raw!r}"
        raise StacNormalizationError(_INVALID, message)
    return raw


def _resolve_grid(
    properties: Mapping[str, Any], *, geocoded: bool
) -> tuple[str, Any, tuple[int, int]] | None:
    """Resolve CRS, affine, and shape in the documented precedence order."""
    if geocoded:
        return _grid_from_payload(properties.get("faninsar:grid"))
    if (
        "proj:code" in properties
        and "proj:transform" in properties
        and "faninsar:shape" in properties
    ):
        return _grid_from_values(
            properties["proj:code"],
            properties["proj:transform"],
            shape=properties["faninsar:shape"],
        )
    if (
        "faninsar:crs" in properties
        and "faninsar:transform" in properties
        and "faninsar:shape" in properties
    ):
        return _grid_from_values(
            properties["faninsar:crs"],
            properties["faninsar:transform"],
            shape=properties["faninsar:shape"],
        )
    if "faninsar:grid" in properties:
        return _grid_from_payload(properties["faninsar:grid"])
    return None


def _optional_grid_from_payload(
    value: object,
) -> tuple[str, Any, tuple[int, int]] | None:
    """Return a complete grid payload, or ``None`` when it is not recognized."""
    try:
        return _grid_from_payload(value)
    except StacNormalizationError:
        return None


def _grid_from_payload(
    value: object,
) -> tuple[str, Any, tuple[int, int]]:
    """Parse one ``faninsar:grid`` mapping, accepting both shape spellings."""
    if not isinstance(value, Mapping):
        message = "property 'faninsar:grid' must be a mapping"
        raise StacNormalizationError(_INVALID, message)
    if "crs" not in value or "transform" not in value:
        message = "property 'faninsar:grid' needs crs and transform"
        raise StacNormalizationError(_INVALID, message)
    height = value.get("height")
    width = value.get("width")
    shape = value.get("shape")
    if shape is not None:
        parsed = _parse_shape(shape)
        if height is not None or width is not None:
            if height is None or width is None:
                message = "grid height and width must appear together"
                raise StacNormalizationError(_INVALID, message)
            separate = _parse_shape([height, width])
            if separate != parsed:
                message = "grid shape and height/width disagree"
                raise StacNormalizationError(_INVALID, message)
        return _grid_from_values(value["crs"], value["transform"], shape=parsed)
    if height is None or width is None:
        message = "property 'faninsar:grid' has no shape"
        raise StacNormalizationError(_INVALID, message)
    return _grid_from_values(value["crs"], value["transform"], shape=[height, width])


def _grid_from_values(
    crs: object, transform: object, *, shape: object
) -> tuple[str, Any, tuple[int, int]]:
    """Validate one CRS, affine transform, and height/width shape."""
    if not isinstance(crs, str) or not crs:
        message = "grid CRS must be a non-empty string"
        raise StacNormalizationError(_INVALID, message)
    parsed_shape = _parse_shape(shape)
    parsed_transform = _parse_transform(transform)
    return crs, parsed_transform, parsed_shape


def _parse_shape(value: object) -> tuple[int, int]:
    """Return a positive integer ``(height, width)`` shape."""
    if isinstance(value, Mapping):
        message = "grid shape must not be a mapping"
        raise StacNormalizationError(_INVALID, message)
    if not isinstance(value, list | tuple) or len(value) != 2:
        message = f"grid shape must contain exactly height and width: {value!r}"
        raise StacNormalizationError(_INVALID, message)
    height, width = value
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in (height, width)
    ):
        message = f"grid shape must be two positive integers: {value!r}"
        raise StacNormalizationError(_INVALID, message)
    return int(height), int(width)


def _parse_transform(value: object) -> Any:
    """Parse six or nine finite affine coefficients with ``rasterio.Affine``."""
    from rasterio import Affine

    if not isinstance(value, list | tuple) or len(value) not in {6, 9}:
        message = f"grid transform must contain six or nine numbers: {value!r}"
        raise StacNormalizationError(_INVALID, message)
    if any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in value
    ):
        message = f"grid transform contains a non-number: {value!r}"
        raise StacNormalizationError(_INVALID, message)
    numbers = [float(item) for item in value]
    if not all(math.isfinite(item) for item in numbers):
        message = f"grid transform is not finite: {value!r}"
        raise StacNormalizationError(_INVALID, message)
    try:
        parsed = Affine(*numbers)
    except Exception as error:
        message = f"grid transform is not an affine: {value!r}"
        raise StacNormalizationError(_INVALID, message) from error
    if parsed.a == 0.0 or parsed.e == 0.0:
        message = "grid transform is not invertible on the raster axes"
        raise StacNormalizationError(_INVALID, message)
    return parsed


def _geometry_from_grid(
    properties: dict[str, Any], grid: tuple[str, Any, tuple[int, int]]
) -> tuple[dict[str, Any], tuple[float, float, float, float]]:
    """Map the four pixel corners to a closed WGS84 ring and numeric bbox."""
    from pyproj import CRS, Transformer
    from shapely.geometry import Polygon

    crs, affine, (height, width) = grid
    try:
        source_crs = CRS.from_user_input(crs)
        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    except Exception as error:
        message = f"grid CRS is not recognized: {crs!r}"
        raise StacNormalizationError(_INVALID, message) from error
    ring: list[list[float]] = []
    for column, row in ((0, 0), (width, 0), (width, height), (0, height)):
        x = affine.a * column + affine.b * row + affine.c
        y = affine.d * column + affine.e * row + affine.f
        try:
            longitude, latitude = transformer.transform(x, y)
        except Exception as error:
            message = f"grid corner cannot be transformed: {(x, y)!r}"
            raise StacNormalizationError(_INVALID, message) from error
        if not (math.isfinite(longitude) and math.isfinite(latitude)):
            message = f"grid corner is not finite: {(x, y)!r}"
            raise StacNormalizationError(_INVALID, message)
        ring.append([float(longitude), float(latitude)])
    ring.append(list(ring[0]))
    polygon = Polygon(ring)
    if not polygon.is_valid or polygon.is_empty:
        message = "transformed grid footprint is not a valid polygon"
        raise StacNormalizationError(_INVALID, message)
    longitudes = [point[0] for point in ring]
    latitudes = [point[1] for point in ring]
    bbox = (
        min(longitudes),
        min(latitudes),
        max(longitudes),
        max(latitudes),
    )
    properties["proj:code"] = source_crs.to_string()
    properties["proj:shape"] = [height, width]
    properties["proj:transform"] = [
        affine.a,
        affine.b,
        affine.c,
        affine.d,
        affine.e,
        affine.f,
        0.0,
        0.0,
        1.0,
    ]
    return {"type": "Polygon", "coordinates": [ring]}, bbox


def _drop_projection(properties: dict[str, Any]) -> None:
    """Remove projection fields from a non-geographic Item."""
    for key in [key for key in properties if str(key).startswith("proj:")]:
        properties.pop(key, None)


def _validate_extensions(item: Mapping[str, Any], path: Path) -> None:
    """Reject a canonical extension list that is not a JSON array."""
    extensions = item.get("stac_extensions")
    if extensions is not None and not isinstance(extensions, list):
        message = f"product {path} has invalid stac_extensions"
        raise StacNormalizationError(_INVALID, message)


def _add_projection_extension(item: dict[str, Any], path: Path) -> None:
    """Add the projection v2.0.0 extension URI to one geographic Item."""
    extensions = item.get("stac_extensions")
    if extensions is None:
        extensions = []
    elif not isinstance(extensions, list):
        message = f"product {path} has invalid stac_extensions"
        raise StacNormalizationError(_INVALID, message)
    if PROJECTION_EXTENSION not in extensions:
        extensions.append(PROJECTION_EXTENSION)
    item["stac_extensions"] = extensions


def _canonical_copy(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict-JSON deep copy of the canonical Item."""
    try:
        return json.loads(json.dumps(item, allow_nan=False))
    except (TypeError, ValueError) as error:
        message = f"canonical product Item is not strict JSON: {error}"
        raise StacNormalizationError(_INVALID, message) from error


def _json_copy(value: object) -> Any:
    """Return a plain strict-JSON deep copy of one metadata value."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        message = f"inherited metadata is not strict JSON: {error}"
        raise StacNormalizationError(_INVALID, message) from error


def _apply_temporal(properties: dict[str, Any], temporal: TemporalCoverage) -> None:
    """Write the Item temporal fields for one coverage."""
    properties["faninsar:temporal_precision"] = temporal.precision
    if temporal.precision == "day":
        properties["datetime"] = None
        properties["start_datetime"] = _format_day_timestamp(temporal.start)
        properties["end_datetime"] = _format_day_timestamp(temporal.end)
        return
    properties["datetime"] = _format_timestamp(temporal.start)
    properties["start_datetime"] = _format_timestamp(temporal.start)
    properties["end_datetime"] = _format_timestamp(temporal.end)
    if temporal.start == temporal.end:
        properties.pop("start_datetime", None)
        properties.pop("end_datetime", None)
    else:
        properties["datetime"] = None


def _format_timestamp(value: datetime) -> str:
    """Return an RFC 3339 UTC timestamp with a trailing ``Z``."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _format_day_timestamp(value: datetime) -> str:
    """Return a day-convention UTC timestamp with six fractional digits."""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _validate_strict_json(item: Mapping[str, Any], path: Path) -> None:
    """Reject a normalized Item that cannot be serialized as strict JSON."""
    try:
        json.dumps(item, allow_nan=False)
    except (TypeError, ValueError) as error:
        message = f"normalized Item for {path} is not strict JSON: {error}"
        raise StacNormalizationError(_INVALID, message) from error
