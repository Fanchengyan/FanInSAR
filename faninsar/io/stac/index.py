"""Derive one immutable read-only STAC index from committed products (P0089).

The index is a disposable view over canonical ``item.json`` files. It reads
metadata only, never opens or hashes a payload, and never writes to the
workspace. Discovery, stale detection, semantic normalization, and assembly run
as four deterministic passes so an identical workspace rebuilds to the same
generation digest.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from faninsar.io.stac.normalize import normalize_product
from faninsar.io.stac.record import StacNormalizationError
from faninsar.io.storage.product import ITEM_NAME, Product, is_write_id

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from faninsar.io.stac.record import StacRecord
    from faninsar.io.stac.search import SearchPage, SearchRequest

__all__ = [
    "IndexExclusion",
    "StacCollection",
    "StacIndex",
    "build_index",
]

_MAX_ITEM_BYTES = 4 * 1024 * 1024
_NORMALIZATION_SCHEMA_VERSION = "1"
_PRUNED_DIRECTORY_NAMES = frozenset({"stac", "cache", "tmp"})
ExclusionReason = Literal[
    "stale",
    "invalid-item",
    "unsupported-kind",
    "missing-temporal",
    "missing-source",
    "cyclic-inheritance",
    "duplicate-write-id",
    "unsafe-asset",
    "outside-work-dir",
]
_EXCLUSION_REASONS: frozenset[str] = frozenset(
    {
        "stale",
        "invalid-item",
        "unsupported-kind",
        "missing-temporal",
        "missing-source",
        "cyclic-inheritance",
        "duplicate-write-id",
        "unsafe-asset",
        "outside-work-dir",
    }
)


@dataclass(frozen=True, slots=True)
class IndexExclusion:
    """One product omitted from the public index, with its reason."""

    source_relative_path: str
    reason: ExclusionReason
    detail: str


@dataclass(frozen=True, slots=True)
class StacCollection:
    """A real extent-backed Collection derived from geographic family members."""

    collection_id: str
    family: str
    item_ids: tuple[str, ...]
    bbox: tuple[float, float, float, float]
    interval: tuple[datetime, datetime]


@dataclass(frozen=True, slots=True)
class StacIndex:
    """One immutable generation of normalized records and their catalog view."""

    generation: str
    created_at: str
    work_dir: Path
    records: tuple[StacRecord, ...]
    collections: tuple[StacCollection, ...]
    exclusions: tuple[IndexExclusion, ...]

    def record(self, public_id: str) -> StacRecord | None:
        """Return the record with ``public_id``, or ``None``.

        Parameters
        ----------
        public_id : str
            Public Item id.

        """
        for record in self.records:
            if record.public_id == public_id:
                return record
        return None

    def records_for_family(self, family: str) -> tuple[StacRecord, ...]:
        """Return every record in ``family`` ordered by public id."""
        return tuple(record for record in self.records if record.family == family)

    def collection(self, collection_id: str) -> StacCollection | None:
        """Return the Collection with ``collection_id``, or ``None``."""
        for collection in self.collections:
            if collection.collection_id == collection_id:
                return collection
        return None

    def families(self) -> tuple[str, ...]:
        """Return every non-empty family name in deterministic order."""
        return tuple(sorted({record.family for record in self.records}))

    def search(self, request: SearchRequest) -> SearchPage:
        """Execute one Item Search request against this generation.

        Parameters
        ----------
        request : SearchRequest
            Validated-or-validating search parameters.

        Raises
        ------
        faninsar.io.stac.search.StacSearchError
            If a parameter is invalid or a cursor is stale.

        """
        from faninsar.io.stac.search import search

        return search(self, request)

    def report(self) -> dict[str, Any]:
        """Return a strict-JSON refresh report for this generation."""
        families = {
            family: len(self.records_for_family(family)) for family in self.families()
        }
        return {
            "generation": self.generation,
            "item_count": len(self.records),
            "collection_count": len(self.collections),
            "family_count": len(families),
            "families": families,
            "collections": [
                collection.collection_id for collection in self.collections
            ],
            "exclusions": [
                {
                    "source_relative_path": exclusion.source_relative_path,
                    "reason": exclusion.reason,
                    "detail": exclusion.detail,
                }
                for exclusion in self.exclusions
            ],
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One physically valid raw product candidate."""

    product: Product
    source_relative_path: str
    write_id: str
    inputs: Mapping[str, str]


def build_index(
    work_dir: str | Path,
    *,
    products: Sequence[Product] | None = None,
) -> StacIndex:
    """Build one immutable STAC index from committed products.

    The build is read-only. It discovers products without following symlinks,
    reads each ``item.json`` with a 4 MiB bound, excludes stale and duplicate
    write ids, normalizes every remaining candidate, and derives families and
    real geographic Collections.

    Parameters
    ----------
    work_dir : str or path-like
        Stack workspace root.
    products : sequence of Product, optional
        Complete candidate set supplied by a test or caller. When omitted the
        filesystem is scanned. Every supplied product is still validated under
        ``work_dir``.

    Returns
    -------
    StacIndex
        Immutable generation with records, Collections, and exclusions.

    """
    root = Path(work_dir).expanduser().resolve()
    candidates: list[_Candidate | IndexExclusion] = []
    for path, supplied in _candidate_paths(root, products):
        candidates.append(_read_candidate(path, root, supplied))

    exclusions: list[IndexExclusion] = [
        candidate for candidate in candidates if isinstance(candidate, IndexExclusion)
    ]
    admitted = [
        candidate
        for candidate in candidates
        if not isinstance(candidate, IndexExclusion)
    ]

    by_write_id: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in admitted:
        by_write_id[candidate.write_id].append(candidate)

    unique: list[_Candidate] = []
    for write_id in sorted(by_write_id):
        group = by_write_id[write_id]
        if len(group) == 1:
            unique.append(group[0])
            continue
        locations = ", ".join(
            sorted(candidate.source_relative_path for candidate in group)
        )
        exclusions.extend(
            IndexExclusion(
                source_relative_path=candidate.source_relative_path,
                reason="duplicate-write-id",
                detail=f"write id {write_id} is shared by {locations}",
            )
            for candidate in group
        )

    unique.sort(key=lambda candidate: candidate.source_relative_path)
    physical_write_ids = set(by_write_id)
    raw_products = {candidate.write_id: candidate.product for candidate in unique}

    stale: list[_Candidate] = []
    stale_write_ids: set[str] = set()
    for candidate in unique:
        missing = sorted(
            f"{name}={write_id}"
            for name, write_id in candidate.inputs.items()
            if write_id not in physical_write_ids
        )
        if missing:
            stale_write_ids.add(candidate.write_id)
            exclusions.append(
                IndexExclusion(
                    source_relative_path=candidate.source_relative_path,
                    reason="stale",
                    detail="missing current inputs: " + ", ".join(missing),
                )
            )
        else:
            stale.append(candidate)

    # A stale dependency is absent from the inheritance map so a dependent
    # cannot revive excluded metadata; it must fail as missing-source instead.
    inheritance_products = {
        write_id: product
        for write_id, product in raw_products.items()
        if write_id not in stale_write_ids
    }

    records: list[StacRecord] = []
    for candidate in stale:
        try:
            record = normalize_product(
                candidate.product,
                work_dir=root,
                products_by_write_id=inheritance_products,
            )
        except StacNormalizationError as error:
            reason = (
                error.reason if error.reason in _EXCLUSION_REASONS else "invalid-item"
            )
            exclusions.append(
                IndexExclusion(
                    source_relative_path=candidate.source_relative_path,
                    reason=reason,
                    detail=error.detail,
                )
            )
        else:
            records.append(record)

    records.sort(key=lambda record: record.public_id)
    collections = _build_collections(records)
    collection_by_family = {
        collection.family: collection.collection_id for collection in collections
    }
    records = [
        replace(
            record,
            collection_id=(
                collection_by_family.get(record.family)
                if record.bbox is not None
                else None
            ),
        )
        for record in records
    ]
    exclusions.sort(
        key=lambda exclusion: (
            exclusion.source_relative_path,
            exclusion.reason,
            exclusion.detail,
        )
    )
    return StacIndex(
        generation=_generation(records),
        created_at=_timestamp(datetime.now(UTC)),
        work_dir=root,
        records=tuple(records),
        collections=collections,
        exclusions=tuple(exclusions),
    )


def _candidate_paths(
    work_dir: Path,
    products: Sequence[Product] | None,
) -> list[tuple[Path, Mapping[str, Any] | None]]:
    """Return deterministic candidate directories and any supplied item maps."""
    if products is not None:
        supplied = []
        for product in products:
            path = product.path
            if not path.is_absolute():
                path = work_dir / path
            supplied.append((path, product.item))
        return sorted(supplied, key=lambda entry: entry[0].as_posix())
    found: list[tuple[Path, Mapping[str, Any] | None]] = []
    pending: list[tuple[Path, frozenset[str]]] = [(work_dir, frozenset())]
    while pending:
        current, pruned = pending.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(".") or entry.name in _PRUNED_DIRECTORY_NAMES:
                continue
            if entry.name in pruned:
                continue
            if entry.is_symlink():
                continue
            if not entry.is_dir():
                continue
            item_path = entry / ITEM_NAME
            if item_path.exists():
                found.append((entry, None))
                pending.append((entry, _asset_directory_names(item_path)))
            else:
                pending.append((entry, frozenset()))
    return sorted(found, key=lambda entry: entry[0].as_posix())


def _asset_directory_names(item_path: Path) -> frozenset[str]:
    """Return registered asset basenames so discovery skips payload trees.

    Discovery still enters the product directory itself so a committed child
    product (for example ``unwrap`` or ``geocoded``) remains discoverable, but
    it does not descend into a registered COG or Zarr payload directory. This
    is a bounded metadata read of ``item.json`` only; no payload is opened.
    """
    if item_path.is_symlink():
        return frozenset()
    try:
        with item_path.open("rb") as stream:
            payload = stream.read(_MAX_ITEM_BYTES + 1)
    except OSError:
        return frozenset()
    if len(payload) > _MAX_ITEM_BYTES:
        return frozenset()
    try:
        item = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError):
        return frozenset()
    assets = item.get("assets") if isinstance(item, dict) else None
    if not isinstance(assets, dict):
        return frozenset()
    names: set[str] = set()
    for asset in assets.values():
        if not isinstance(asset, dict):
            continue
        href = asset.get("href")
        if not isinstance(href, str) or not href or "\x00" in href:
            continue
        candidate = Path(href)
        if candidate.is_absolute() or len(candidate.parts) != 1:
            continue
        if candidate.name != href or href in {".", "..", ITEM_NAME}:
            continue
        names.add(href)
    return frozenset(names)


def _read_candidate(  # noqa: PLR0911 - one explicit exclusion per validation stage
    path: Path,
    work_dir: Path,
    supplied: Mapping[str, Any] | None = None,
) -> _Candidate | IndexExclusion:
    """Validate one raw candidate directory and return metadata-only state."""
    display = _display_path(path, work_dir)
    if path.is_symlink():
        return IndexExclusion(display, "outside-work-dir", "product is a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(work_dir)
    except ValueError:
        return IndexExclusion(
            resolved.as_posix(),
            "outside-work-dir",
            f"product resolves outside work_dir: {resolved}",
        )
    display = resolved.relative_to(work_dir).as_posix()
    if supplied is None:
        item_path = resolved / ITEM_NAME
        if item_path.is_symlink():
            return IndexExclusion(display, "invalid-item", "item.json is a symlink")
        try:
            item = _read_item(item_path)
        except _ItemError as error:
            return IndexExclusion(display, error.reason, error.detail)
    else:
        item = dict(supplied)
        try:
            _validate_item(item)
        except _ItemError as error:
            return IndexExclusion(display, error.reason, error.detail)

    properties = item["properties"]
    write_id = properties["faninsar:write_id"]
    raw_inputs = properties.get("faninsar:inputs", {})
    if not isinstance(raw_inputs, dict):
        return IndexExclusion(display, "invalid-item", "faninsar:inputs is not a map")
    inputs: dict[str, str] = {}
    for name, value in raw_inputs.items():
        if not isinstance(name, str) or not isinstance(value, str):
            return IndexExclusion(
                display,
                "invalid-item",
                "faninsar:inputs must map names to write ids",
            )
        inputs[name] = value
    unsafe = _unsafe_asset(item["assets"])
    if unsafe is not None:
        return IndexExclusion(display, "unsafe-asset", unsafe)
    return _Candidate(
        product=Product(path=resolved, item=item),
        source_relative_path=display,
        write_id=write_id,
        inputs=inputs,
    )


class _ItemError(Exception):
    """Internal rejection of one unreadable or non-product ``item.json``."""

    def __init__(self, reason: ExclusionReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _reject_item(reason: ExclusionReason, detail: str) -> NoReturn:
    """Raise one internal item rejection with a reason and diagnostic."""
    raise _ItemError(reason, detail)


def _read_item(item_path: Path) -> dict[str, Any]:
    """Read one bounded strict-JSON product item and validate its envelope."""
    try:
        with item_path.open("rb") as stream:
            payload = stream.read(_MAX_ITEM_BYTES + 1)
    except OSError as error:
        _reject_item("invalid-item", f"item.json cannot be read: {error}")
    if len(payload) > _MAX_ITEM_BYTES:
        _reject_item(
            "invalid-item",
            f"item.json exceeds the {_MAX_ITEM_BYTES} byte bound",
        )
    try:
        item = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as error:
        _reject_item(
            "invalid-item",
            f"item.json is not strict JSON: {error}",
        )
    if not isinstance(item, dict):
        _reject_item("invalid-item", "item.json is not a JSON object")
    _validate_item(item)
    return item


def _validate_item(item: Mapping[str, Any]) -> None:
    """Validate the minimal product envelope shared by files and injections."""
    for name in ("stac_version", "type", "id"):
        if not isinstance(item.get(name), str) or not item[name]:
            _reject_item("invalid-item", f"item.json has no valid {name!r}")
    properties = item.get("properties")
    if not isinstance(properties, dict):
        _reject_item("invalid-item", "item.json has no properties map")
    write_id = properties.get("faninsar:write_id")
    if not is_write_id(write_id):
        _reject_item(
            "invalid-item",
            "item.json has no valid faninsar:write_id",
        )
    assets = item.get("assets")
    if not isinstance(assets, dict):
        _reject_item("invalid-item", "item.json has no assets map")


def _reject_json_constant(value: str) -> NoReturn:
    """Reject NaN and Infinity while parsing strict JSON."""
    message = f"non-finite JSON constant {value!r}"
    raise ValueError(message)


def _unsafe_asset(assets: Mapping[str, Any]) -> str | None:
    """Return a detail string when any asset href escapes the product root."""
    for name, asset in assets.items():
        if not isinstance(asset, dict):
            return f"asset {name!r} is not a map"
        href = asset.get("href")
        if not isinstance(href, str) or not href or "\x00" in href:
            return f"asset {name!r} has no safe href"
        candidate = Path(href)
        if (
            candidate.is_absolute()
            or len(candidate.parts) != 1
            or candidate.name != href
            or href in {".", ".."}
            or href.startswith(".")
            or href == ITEM_NAME
        ):
            return f"asset {name!r} href is not a plain basename: {href!r}"
    return None


def _display_path(path: Path, work_dir: Path) -> str:
    """Return a workspace-relative diagnostic path without resolving symlinks."""
    try:
        return path.relative_to(work_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _build_collections(records: Sequence[StacRecord]) -> tuple[StacCollection, ...]:
    """Derive real geographic Collections from geographic family members."""
    grouped: dict[str, list[StacRecord]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)
    collections = []
    for family in sorted(grouped):
        members = [record for record in grouped[family] if record.bbox is not None]
        if not members:
            continue
        bbox = (
            min(record.bbox[0] for record in members if record.bbox is not None),
            min(record.bbox[1] for record in members if record.bbox is not None),
            max(record.bbox[2] for record in members if record.bbox is not None),
            max(record.bbox[3] for record in members if record.bbox is not None),
        )
        start = min(record.temporal.start for record in members)
        end = max(record.temporal.end for record in members)
        collections.append(
            StacCollection(
                collection_id=f"{family}-geographic",
                family=family,
                item_ids=tuple(sorted(record.public_id for record in members)),
                bbox=bbox,
                interval=(start, end),
            )
        )
    return tuple(collections)


def _generation(records: Sequence[StacRecord]) -> str:
    """Return the deterministic digest of admitted public identities."""
    payload = [
        _NORMALIZATION_SCHEMA_VERSION,
        sorted(
            (record.public_id, record.write_id, record.source_relative_path)
            for record in records
        ),
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: datetime) -> str:
    """Return one UTC RFC 3339 timestamp with microsecond precision."""
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )
