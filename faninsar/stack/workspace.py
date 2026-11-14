"""Durable Stack workspace intent (PROPOSAL-0088).

``stack.json`` is the single authoritative intent document. A supported
factory publishes it before returning, :meth:`faninsar.stack.Stack.open`
reopens it locally, and :meth:`faninsar.stack.Stack.extend` adds targets by
union. Completion and readiness remain derived from committed products and
receipts, never from a persisted configured/prepared state machine.

The module owns only intent normalization, atomic persistence, supported
legacy adoption and target union. Transfer journals, product receipts and
scientific manifests remain the authorities for the work they describe.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from faninsar.logging import reject, setup_logger
from faninsar.remote.records import _sanitize
from faninsar.stack.identity import (
    PREPARATION_EXCLUDED_SCIENCE_FIELDS,
    SAVED_EXTRA_KEYS,
    STACK_IDENTITY_SCHEMA,
    STACK_JSON_NAME,
    WORKSPACE_SCHEMA_VERSION,
    _write_stack_document,
    decode_value,
    stack_identity,
)
from faninsar.stack.preparation import freeze, reentrant_writer_guard

if TYPE_CHECKING:
    from faninsar.stack.session import Stack

logger = setup_logger(__name__)

BACKUP_SUFFIX = ".bak"
"""Suffix used to preserve the prior identity document before adoption."""

REQUEST_RECORD = Path("inputs") / "request.json"
"""Legacy request-only document adopted into the authoritative intent."""

MUTABLE_EXTRA_KEYS = frozenset(
    {
        "preparation_report",
        "preparation_local_inputs",
        "track_metadata",
        "selected_footprints",
    }
)
"""Science-extra keys derived from execution, excluded from intent equality."""

REQUEST_EXTRA_KEYS = SAVED_EXTRA_KEYS - MUTABLE_EXTRA_KEYS
"""Science-extra keys fixed by the caller's mission request.

Execution-derived keys are excluded. The remaining values (for example
``polarizations``) are workspace science invariants that ``extend`` never
changes, so a supported legacy record may restore them into the intent.
"""

REQUEST_PATH_EXTRA_KEYS = frozenset({"orbit_paths", "aux_cal_paths"})
"""Request extras whose value is also a resolved preparation artifact.

Preparation resolves or downloads covering orbit and AUX_CAL files even
when the caller fixed none, so ``science_extra`` records both caller pins and
derived bindings without distinguishing them. The authoritative ``request``
is the only record of a caller pin; equality and repair therefore read these
two keys from the request instead of the possibly-derived extra copy.
"""

Verification = Literal["fast", "strict"]
ErrorPolicy = Literal["warn", "raise"]


class WorkspaceConflictError(ValueError, RuntimeError):
    """A different durable intent already occupies the work directory."""


class WorkspaceIntegrityError(ValueError, RuntimeError):
    """A workspace intent is unreadable, ambiguous or unsupported."""


@dataclass(frozen=True)
class IntentState:
    """One validated authoritative workspace intent.

    Parameters
    ----------
    root : pathlib.Path
        Work directory holding the identity.
    document : mapping
        Parsed ``stack.json`` document.
    revision : int
        Lost-update token recorded by the latest intent publication.
    mission : str or None
        Concrete Stack class name.
    request : mapping
        Normalized mission source request.
    configured : bool
        Whether the document holds no resolved input membership yet.
    conflicts : tuple of str
        Legacy adoption conflicts retained for diagnostics. A nonempty value
        never authorizes scientific reuse of the affected pins.

    """

    root: Path
    document: Mapping[str, Any]
    revision: int
    mission: str | None
    request: Mapping[str, Any] = field(default_factory=dict)
    configured: bool = True
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtensionReport:
    """Additive targets accepted by one :meth:`Stack.extend` call.

    Parameters
    ----------
    revision : int
        Intent revision after the call.
    changed : bool
        Whether the union added any target or request identity.
    added_sources : tuple of str
        Newly admitted source identities.
    added_roi, added_bursts, added_swaths : bool
        Whether the corresponding union grew.
    reference : str or None
        Reference pin after the call.

    """

    revision: int
    changed: bool = False
    added_sources: tuple[str, ...] = ()
    added_roi: bool = False
    added_bursts: bool = False
    added_swaths: bool = False
    reference: str | None = None


def _validate_policy(verification: str, on_error: str) -> None:
    """Reject an unknown execution policy before reading the workspace."""
    if verification not in ("fast", "strict"):
        message = f"verification must be 'fast' or 'strict'; got {verification!r}"
        reject(message, ValueError, logger=logger)
    if on_error not in ("warn", "raise"):
        message = f"on_error must be 'warn' or 'raise'; got {on_error!r}"
        reject(message, ValueError, logger=logger)


def _schema_version(document: Mapping[str, Any]) -> int:
    """Return the recorded workspace schema version (legacy v1 when absent)."""
    value = document.get("schema_version")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 1


def _state(
    root: Path, document: Mapping[str, Any], *, conflicts: Sequence[str] = ()
) -> IntentState:
    """Build one immutable intent view from a parsed document."""
    revision = document.get("intent_revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        revision = 0
    request = document.get("request")
    if isinstance(request, Mapping):
        from faninsar.stack.identity import decode_value

        decoded = decode_value(dict(request))
        request = decoded if isinstance(decoded, Mapping) else {}
    else:
        request = {}
    status = document.get("identity_status")
    sources = document.get("inputs")
    resolved_sources = sources.get("sources") if isinstance(sources, Mapping) else None
    configured = (
        status == "configured"
        if isinstance(status, str)
        else not resolved_sources and not document.get("pairs")
    )
    return IntentState(
        root=Path(root),
        document=document,
        revision=revision,
        mission=(None if document.get("mission") is None else str(document["mission"])),
        request=request,
        configured=configured,
        conflicts=tuple(conflicts),
    )


def read_intent(root: str | Path) -> IntentState | None:
    """Read ``stack.json`` without requiring a current schema.

    Parameters
    ----------
    root : str or pathlib.Path
        Work directory that may hold ``stack.json``.

    Returns
    -------
    IntentState or None
        The parsed intent, or ``None`` when the identity is absent or is not
        a JSON object.

    """
    path = Path(root) / STACK_JSON_NAME
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    return _state(Path(root), document)


def open_intent(
    root: str | Path,
    *,
    verification: Verification = "fast",
    on_error: ErrorPolicy = "warn",
) -> IntentState | None:
    """Validate and, when needed, adopt one local workspace intent.

    Only supported legacy metadata is upgraded. No source is opened, no
    remote request runs and no scientific stage reruns. A newer schema is
    rejected rather than downgraded.

    Parameters
    ----------
    root : str or pathlib.Path
        Work directory to inspect.
    verification : {"fast", "strict"}, default="fast"
        Execution policy recorded by the caller, never persisted as science.
    on_error : {"warn", "raise"}, default="warn"
        Whether a supported but ambiguous legacy adoption warns or raises.

    Returns
    -------
    IntentState or None
        The current intent, or ``None`` when no identity exists yet.

    """
    _validate_policy(verification, on_error)
    target = Path(root)
    if target.exists() and not target.is_dir():
        message = f"Stack work_dir is not a directory: {target}"
        reject(message, ValueError, logger=logger)
    with reentrant_writer_guard(target):
        return _read_locked(target, on_error=on_error)


def publish_intent(
    stack: Stack,
    *,
    allow_existing: bool = False,
    on_error: ErrorPolicy = "raise",
) -> IntentState:
    """Atomically publish the configured intent for one factory result.

    The same normalized intent is idempotent. A different existing intent
    raises instead of overwriting unless ``allow_existing`` is set by the
    explicit extension path.

    Parameters
    ----------
    stack : Stack
        Configured Stack whose request and science contract are authoritative.
    allow_existing : bool, default=False
        Accept an existing different intent (extension only).
    on_error : {"warn", "raise"}, default="raise"
        Legacy adoption policy used while comparing the existing identity.

    Returns
    -------
    IntentState
        The published or already matching intent.

    """
    root = Path(stack.config.work_dir)
    if root.exists() and not root.is_dir():
        message = f"Stack work_dir is not a directory: {root}"
        reject(message, ValueError, logger=logger)
    with reentrant_writer_guard(root):
        existing = _read_locked(root, on_error=on_error)
        candidate = build_intent(
            stack, revision=(1 if existing is None else existing.revision + 1)
        )
        if existing is not None:
            if _same_intent(existing.document, candidate):
                stack.intent_revision = existing.revision
                _log_conflicts(existing)
                return existing
            if not allow_existing:
                message = (
                    f"{root} already holds a different Stack intent "
                    f"(revision {existing.revision}); open it and call extend() "
                    "or choose a different work_dir"
                )
                reject(message, WorkspaceConflictError, logger=logger)
        path = root / STACK_JSON_NAME
        if path.is_file():
            _backup(path)
        _write_stack_document(path, candidate)
        _fsync_dir(root)
        stack.intent_revision = int(candidate["intent_revision"])
        state = _state(root, candidate)
        _log_conflicts(state)
        return state


def build_intent(stack: Stack, *, revision: int = 1) -> dict[str, Any]:
    """Return the version-2 identity document for one Stack.

    Parameters
    ----------
    stack : Stack
        Stack whose request and science contract are serializable.
    revision : int, default=1
        Intent revision token to record.

    Returns
    -------
    dict
        The normalized intent document ready for atomic publication.

    """
    document = stack_identity(stack)
    document["schema"] = STACK_IDENTITY_SCHEMA
    document["schema_version"] = WORKSPACE_SCHEMA_VERSION
    document["intent_revision"] = int(revision)
    document["request"] = normalize_request(
        str(document.get("mission") or ""),
        decode_value(document.get("request") or {}),
    )
    targets = document.get("targets")
    document["targets"] = _normalize_targets(
        targets if isinstance(targets, Mapping) else {}
    )
    return document


def extend_intent(
    stack: Stack,
    *,
    search: Any = None,
    paths: Sequence[str | Path] | None = None,
    roi: object | None = None,
    bursts: object | None = None,
    swaths: Sequence[str] | None = None,
    on_error: ErrorPolicy = "warn",
) -> ExtensionReport:
    """Union additive targets into the durable intent and publish it.

    The Stack object is updated in place. Reference, grid/CRS/resolution,
    alignment, method and pinned source versions are never changed here.

    Parameters
    ----------
    stack : Stack
        Reopened or configured Stack owning the workspace.
    search : Any, optional
        Mission search result to union (Sentinel-1 only).
    paths : sequence of path-like, optional
        Additional local source paths to union.
    roi : spatial, optional
        Additional region of interest; the existing ROI is a union, never
        replaced.
    bursts : BurstSelection, optional
        Additional burst selections.
    swaths : sequence of str, optional
        Additional sub-swaths.
    on_error : {"warn", "raise"}, default="warn"
        Legacy adoption policy.

    Returns
    -------
    ExtensionReport
        Accepted additions and the current intent revision.

    """
    if search is not None and paths is not None:
        message = "extend accepts search or paths, not both source modes"
        reject(message, ValueError, logger=logger)
    if (
        search is None
        and paths is None
        and roi is None
        and bursts is None
        and swaths is None
    ):
        message = "extend requires at least one of search, paths, roi, bursts or swaths"
        reject(message, ValueError, logger=logger)
    root = Path(stack.config.work_dir)
    with reentrant_writer_guard(root):
        existing = _read_locked(root, on_error=on_error)
        if existing is None:
            message = (
                f"{root} holds no durable Stack intent; create it with a "
                "supported factory before calling extend()"
            )
            reject(message, WorkspaceIntegrityError, logger=logger)
        _log_conflicts(existing)
        old_request = dict(existing.request)
        old_roi = stack._request_config.roi
        old_bursts = stack._request_config.bursts
        old_swaths = stack._request_config.swaths
        new_request = stack._extend_request(old_request, search=search, paths=paths)
        new_roi = union_roi(old_roi, roi) if roi is not None else old_roi
        new_bursts = (
            union_bursts(old_bursts, bursts) if bursts is not None else old_bursts
        )
        new_swaths = (
            union_swaths(old_swaths, swaths) if swaths is not None else old_swaths
        )
        stack.request = freeze(new_request)
        stack.config.roi = new_roi
        stack.config.bursts = new_bursts
        stack.config.swaths = new_swaths
        stack._request_config = replace(
            stack._request_config,
            roi=new_roi,
            bursts=new_bursts,
            swaths=new_swaths,
            extra=dict(stack._request_config.extra),
        )
        candidate = build_intent(stack, revision=existing.revision + 1)
        _preserve_reference(candidate, existing.document)
        if _same_intent(existing.document, candidate):
            stack.intent_revision = existing.revision
            return ExtensionReport(
                revision=existing.revision,
                reference=_document_reference(existing.document),
            )
        path = root / STACK_JSON_NAME
        _backup(path)
        _write_stack_document(path, candidate)
        _fsync_dir(root)
        stack.intent_revision = int(candidate["intent_revision"])
        # The old ready set no longer satisfies the enlarged target. Keep its
        # catalog/report as compatible history, but force the next prepare to
        # reconcile support instead of returning the prior complete report.
        stack._prepared = False
        return ExtensionReport(
            revision=stack.intent_revision,
            changed=True,
            added_sources=_added_sources(old_request, new_request),
            added_roi=roi is not None and not _equal_target(old_roi, new_roi),
            added_bursts=(
                bursts is not None and not _equal_target(old_bursts, new_bursts)
            ),
            added_swaths=(
                swaths is not None and not _equal_target(old_swaths, new_swaths)
            ),
            reference=_document_reference(candidate),
        )


def normalize_request(mission: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Return a provider-order-independent canonical source request.

    Sentinel-1 search items are deduplicated by stable catalog identity and
    sorted; local source paths are canonicalized and sorted; explicit pair
    networks are sorted. List position therefore never becomes scientific
    identity.

    Parameters
    ----------
    mission : str
        Concrete Stack class name.
    request : mapping
        Mission source request.

    Returns
    -------
    dict
        Canonical request with credentials removed.

    """
    del mission
    plain = _plain(decode_value(request))
    kind = plain.get("kind") if isinstance(plain, dict) else None
    if isinstance(plain, dict) and kind == "search":
        search = plain.get("search")
        if isinstance(search, dict):
            items = search.get("items")
            if isinstance(items, list):
                search["items"] = _canonical_search_items(items)
    if isinstance(plain, dict):
        paths = plain.get("paths")
        if isinstance(paths, list):
            plain["paths"] = _canonical_paths(paths)
        topology = plain.get("topology")
        if isinstance(topology, dict):
            for key in ("pairs", "misreg_pairs"):
                values = topology.get(key)
                if isinstance(values, list):
                    topology[key] = sorted({str(item) for item in values})
            reference = topology.get("reference")
            if reference is not None:
                topology["reference"] = str(reference)
    return _sanitize(plain)


def union_roi(current: object | None, addition: object | None) -> object | None:
    """Return the spatial union of two ROIs.

    ``None`` means the mission default and therefore the other operand.
    Bounding boxes union by extent; polygon selections combine without
    duplication. The ROI is never replaced or shrunk.
    """
    if addition is None:
        return current
    if current is None:
        return addition
    from faninsar.data.query.bbox import BoundingBox

    if isinstance(current, BoundingBox) and isinstance(addition, BoundingBox):
        return current | addition
    return _canonical_polygons(_as_polygons(current) + _as_polygons(addition))


def union_bursts(current: object | None, addition: object | None) -> object | None:
    """Return the additive burst selection union.

    ``None`` selects every burst, so it dominates any explicit addition.
    """
    if addition is None or current is None:
        # ``None`` selects every burst, so an explicit addition cannot shrink it.
        return None
    left = _plain_bursts(current)
    right = _plain_bursts(addition)
    if isinstance(left, dict) and isinstance(right, dict):
        result: dict[str, list[object]] = {}
        for key in sorted(set(left) | set(right)):
            values = _unique_values((left.get(key), right.get(key)))
            if values:
                result[str(key)] = values
        return result
    combined = _unique_values((left, right))
    return combined or None


def union_swaths(current: object | None, addition: object | None) -> object | None:
    """Return the additive sub-swath union; ``None`` means every swath."""
    if addition is None or current is None:
        # ``None`` selects every sub-swath, so an explicit addition cannot
        # shrink it.
        return None
    return tuple(sorted({str(item) for item in (*current, *addition)}))


def _preserve_reference(candidate: dict[str, Any], existing: Mapping[str, Any]) -> None:
    """Keep the durable Reference pin across an additive extension."""
    recorded = _document_reference(existing)
    proposed = _document_reference(candidate)
    if recorded is not None and proposed is not None and recorded != proposed:
        message = (
            "extend cannot change the workspace Reference "
            f"({recorded!r} -> {proposed!r}); start a new work_dir instead"
        )
        reject(message, WorkspaceConflictError, logger=logger)
    if recorded is not None and proposed is None:
        candidate["reference"] = recorded


def _document_reference(document: Mapping[str, Any]) -> str | None:
    """Return one document Reference, or ``None``."""
    value = document.get("reference")
    return None if value is None else str(value)


def _added_sources(old: Mapping[str, Any], new: Mapping[str, Any]) -> tuple[str, ...]:
    """Return source identities added by one request union."""
    old_keys = _source_identities(old)
    new_keys = _source_identities(new)
    return tuple(sorted(new_keys - old_keys))


def _source_identities(request: Mapping[str, Any]) -> set[str]:
    """Return stable source identities from one canonical request."""
    kind = request.get("kind")
    if kind == "search":
        search = request.get("search")
        items = search.get("items") if isinstance(search, Mapping) else None
        if isinstance(items, list):
            return {
                _search_item_identity(item)
                for item in items
                if isinstance(item, Mapping)
            }
    paths = request.get("paths")
    if isinstance(paths, list):
        return {str(path) for path in paths}
    return set()


def _canonical_search_items(items: Sequence[object]) -> list[object]:
    """Deduplicate and sort serialized search items by stable identity."""
    selected: dict[str, object] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        selected.setdefault(_search_item_identity(item), item)
    return [selected[key] for key in sorted(selected)]


def _search_item_identity(item: Mapping[str, Any]) -> str:
    """Return a stable identity string for one serialized catalog item."""
    acquisition = item.get("acquisition")
    acquisition = acquisition if isinstance(acquisition, Mapping) else {}
    start = acquisition.get("start")
    item_id = item.get("item_id") or acquisition.get("acquisition_id")
    return "|".join(
        (
            str(item.get("catalog") or ""),
            str(item.get("provider") or ""),
            str(item_id or ""),
            str(start or ""),
        )
    )


def _canonical_paths(values: Sequence[object]) -> list[str]:
    """Deduplicate and sort local source paths without resolving symlinks."""
    selected: dict[str, str] = {}
    for raw in values:
        path = Path(str(raw))
        key = path.resolve(strict=False).as_posix()
        selected.setdefault(key, str(raw))
    return [selected[key] for key in sorted(selected)]


def _plain(value: object) -> Any:
    """Return a JSON-shaped mutable copy of one nested value."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _plain_bursts(value: object) -> Any:
    """Replace ``range`` burst selectors with explicit lists."""
    if isinstance(value, Mapping):
        return {str(key): _plain_bursts(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_bursts(item) for item in value]
    if isinstance(value, range):
        return list(value)
    return value


def _unique_values(values: Sequence[object]) -> list[object]:
    """Return flattened, canonical-unique burst values."""
    result: list[object] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            if item is None:
                continue
            key = _canonical_json(item)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
    return sorted(result, key=_canonical_json)


def _as_polygons(value: object) -> Any:
    """Return one ROI as :class:`~faninsar.data.query.polygons.Polygons`."""
    import geopandas as gpd
    import shapely.geometry

    from faninsar.data.query.bbox import BoundingBox
    from faninsar.data.query.polygons import Polygons

    if isinstance(value, Polygons):
        return value
    if isinstance(value, BoundingBox):
        geometry = shapely.geometry.box(
            value.left, value.bottom, value.right, value.top
        )
        return Polygons(gpd.GeoSeries([geometry], crs=value.crs))
    message = f"ROI union does not support {type(value).__name__}"
    reject(message, TypeError, logger=logger)
    return None


def _canonical_polygons(value: object) -> Any:
    """Deduplicate and order polygons by type and geometry bytes."""
    from faninsar.data.query.polygons import Polygons

    polygons = _as_polygons(value)
    frame = polygons.geodataframe.copy()
    frame["_wkb"] = [geometry.wkb_hex for geometry in frame.geometry]
    frame["_key"] = [
        f"{kind}|{wkb}" for kind, wkb in zip(frame["types"], frame["_wkb"], strict=True)
    ]
    frame = (
        frame.sort_values("_key")
        .drop_duplicates(subset=["_key"])
        .reset_index(drop=True)
    )
    return Polygons(
        frame.drop(columns=["_wkb", "_key"]),
        types=[str(kind) for kind in frame["types"]],
        all_touched=polygons.all_touched,
        pad=polygons.pad,
    )


def _normalize_targets(targets: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize one encoded target mapping."""
    result = dict(targets)
    roi = result.get("roi")
    if isinstance(roi, Mapping) and isinstance(roi.get("faninsar:polygons"), Mapping):
        payload = dict(roi["faninsar:polygons"])
        features = payload.get("features")
        if isinstance(features, list):
            payload["features"] = sorted(
                {_canonical_json(feature): feature for feature in features}.values(),
                key=_canonical_json,
            )
        result["roi"] = {"faninsar:polygons": payload}
    return result


def _same_intent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two documents describe the same scientific intent."""
    return (
        str(left.get("mission") or "") == str(right.get("mission") or "")
        and _canonical_json(normalize_request("", left.get("request") or {}))
        == _canonical_json(normalize_request("", right.get("request") or {}))
        and _canonical_json(_normalize_targets(left.get("targets") or {}))
        == _canonical_json(_normalize_targets(right.get("targets") or {}))
        and _canonical_json(_fixed_science(left))
        == _canonical_json(_fixed_science(right))
        and _canonical_json(_fixed_extra(left)) == _canonical_json(_fixed_extra(right))
    )


def _fixed_science(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the requested science contract used for intent equality."""
    science = document.get("requested_science")
    if not isinstance(science, Mapping):
        science = document.get("science")
    if not isinstance(science, Mapping):
        return {}
    return {
        str(key): value
        for key, value in science.items()
        if key not in PREPARATION_EXCLUDED_SCIENCE_FIELDS
    }


def _fixed_extra(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return requested mission selectors used for intent equality."""
    extra = document.get("requested_science_extra")
    if not isinstance(extra, Mapping):
        extra = document.get("science_extra")
    if not isinstance(extra, Mapping):
        return {}
    fixed = {
        str(key): value for key, value in extra.items() if key not in MUTABLE_EXTRA_KEYS
    }
    request = document.get("request")
    if isinstance(request, Mapping):
        # A path binding appears in both the authoritative request and the
        # requested extras, which preparation may fill from resolved inputs.
        # The request is the caller pin; compare that value only so a derived
        # binding neither conflicts with its own request nor hides a change.
        pinned = _request_path_extras(request)
        for key in REQUEST_PATH_EXTRA_KEYS:
            fixed.pop(key, None)
            if key in pinned:
                fixed[key] = pinned[key]
    return fixed


def _request_path_extras(request: object) -> dict[str, Any]:
    """Return caller-fixed orbit/AUX_CAL bindings from one source request."""
    if not isinstance(request, Mapping):
        return {}
    return {
        key: request[key]
        for key in REQUEST_PATH_EXTRA_KEYS
        if request.get(key) is not None
    }


def requested_extras(extra: object) -> dict[str, Any]:
    """Return the mission-owned request extras of one encoded mapping.

    Values stay encoded exactly as ``stack.json`` records them, so a restored
    record compares equal to one a factory publishes. Only keys in
    :data:`REQUEST_EXTRA_KEYS` are selected; execution-derived keys and
    explicit ``null`` defaults are omitted.

    Parameters
    ----------
    extra : object
        Mapping that may hold ``science_extra`` or ``requested_science_extra``
        values.

    Returns
    -------
    dict
        Mission-owned request extras keyed by their recorded names.

    """
    if not isinstance(extra, Mapping):
        return {}
    return {
        str(key): value
        for key, value in extra.items()
        if str(key) in REQUEST_EXTRA_KEYS and value is not None
    }


def _request_extra_evidence(
    root: Path, document: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return request extras recoverable from records the workspace persisted.

    The preserved ``inputs/request.json`` backup is the original preparation
    request recorded before an adoption refreshed it, so it is preferred when
    it describes this exact mission, source request and requested science.
    The document's own ``science_extra`` is the fallback resolved
    configuration; its orbit and AUX_CAL bindings are caller pins only when
    the authoritative request records them. Evidence for another mission or
    another request is ignored; disagreeing values for one key are ambiguous
    and returned separately instead of being guessed.
    """
    recorded = requested_extras(document.get("science_extra"))
    preserved = _preserved_request_extras(root, document)
    evidence = dict(recorded if preserved is None else preserved)
    # Only the source request records a caller pin. Preparation may resolve or
    # download orbit and AUX_CAL files the caller never fixed, so a path
    # binding in ``science_extra`` is execution state, never requested
    # evidence. Pinned values are restored from the authoritative request.
    pinned = _request_path_extras(document.get("request"))
    for key in REQUEST_PATH_EXTRA_KEYS:
        evidence.pop(key, None)
        if key in pinned:
            evidence[key] = pinned[key]
    conflicts = tuple(
        key
        for key, value in evidence.items()
        if key in recorded and _canonical_json(recorded[key]) != _canonical_json(value)
    )
    return dict(evidence), conflicts


def _preserved_request_extras(
    root: Path, document: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return the request extras of this workspace's preserved request record.

    ``None`` means there is no usable evidence: the record is missing,
    unreadable, is not a supported preparation request, names another
    mission, or describes a different source request or requested science.
    """
    path = root / REQUEST_RECORD.parent / (REQUEST_RECORD.name + BACKUP_SUFFIX)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    mission = str(document.get("mission") or "")
    if not (
        isinstance(loaded, dict)
        and loaded.get("schema") == "faninsar.preparation.v1"
        and str(loaded.get("mission") or "") == mission
        and isinstance(loaded.get("source"), Mapping)
    ):
        return None
    try:
        recorded = normalize_request(mission, document.get("request") or {})
        preserved = normalize_request(mission, loaded["source"])
    except (TypeError, ValueError):
        return None
    if _canonical_json(recorded) != _canonical_json(preserved):
        return None
    science = loaded.get("science")
    expected = document.get("requested_science")
    if not isinstance(expected, Mapping):
        expected = document.get("science")
    if isinstance(science, Mapping) and isinstance(expected, Mapping):
        shared = (set(science) | set(expected)) - PREPARATION_EXCLUDED_SCIENCE_FIELDS
        for key in shared:
            if key not in science or key not in expected:
                continue
            if _canonical_json(science[key]) != _canonical_json(expected[key]):
                return None
    return requested_extras(loaded.get("extra"))


def _canonical_json(value: object) -> str:
    """Return an order-independent JSON key for one value."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))


def _equal_target(left: object, right: object) -> bool:
    """Return whether two target values are equal after encoding."""
    return _canonical_json(_comparable_target(left)) == _canonical_json(
        _comparable_target(right)
    )


def _comparable_target(value: object) -> object:
    """Return a JSON-comparable encoded target value."""
    if value is None:
        return None
    from faninsar.stack.identity import encode_value

    try:
        return encode_value(_plain_bursts(value))
    except (TypeError, ValueError):
        return _plain(value)


def _read_locked(root: Path, *, on_error: ErrorPolicy) -> IntentState | None:
    """Read or adopt one intent while the writer lock is held."""
    path = root / STACK_JSON_NAME
    if not path.is_file():
        request_path = root / REQUEST_RECORD
        if request_path.is_file():
            return _adopt_request_only(root, request_path, on_error=on_error)
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    version = _schema_version(document)
    if version != WORKSPACE_SCHEMA_VERSION:
        message = (
            f"{path} records workspace schema version {version}; this "
            f"FanInSAR supports only version {WORKSPACE_SCHEMA_VERSION}. "
            "Rebuild this work directory with the current FanInSAR version."
        )
        reject(message, WorkspaceIntegrityError, logger=logger)
    return _repair_request_extras(root, path, document, on_error=on_error)


def _repair_request_extras(
    root: Path,
    path: Path,
    document: Mapping[str, Any],
    *,
    on_error: ErrorPolicy,
) -> IntentState:
    """Restore request extras lost by an earlier supported adoption.

    Earlier adoption published ``requested_science_extra`` and
    ``science_extra`` empty, which loses mission-owned request values such as
    ``polarizations``; preparation then cannot read its own request. Only
    missing keys are restored, and only from evidence the workspace persisted
    itself. The source request, targets, requested science and every committed
    product stay unchanged, and a second open makes no write.

    Parameters
    ----------
    root : pathlib.Path
        Work directory holding ``stack.json``.
    path : pathlib.Path
        The current authoritative document.
    document : mapping
        Parsed current intent.
    on_error : {"warn", "raise"}
        Whether ambiguous evidence warns and skips the affected key or raises.

    Returns
    -------
    IntentState
        The repaired intent, or the unchanged one when nothing is missing.

    """
    recorded_requested = document.get("requested_science_extra")
    recorded_science = document.get("science_extra")
    requested = dict(requested_extras(recorded_requested))
    science = dict(requested_extras(recorded_science))
    evidence, conflicts = _request_extra_evidence(root, document)
    # Only a value the intent still needs can be ambiguous; a complete
    # document never fails to open because a stale copy disagrees.
    conflicts = tuple(
        key for key in conflicts if key not in requested or key not in science
    )
    if conflicts:
        message = (
            f"{path} needs request extras that disagree with its preserved "
            "preparation request: " + ", ".join(conflicts)
        )
        if on_error == "raise":
            reject(message, WorkspaceIntegrityError, logger=logger)
        logger.warning("%s; recording nothing for those keys", message)
        evidence = {
            key: value for key, value in evidence.items() if key not in conflicts
        }
    requested_additions = {
        key: value for key, value in evidence.items() if key not in requested
    }
    science_additions = {
        key: value for key, value in evidence.items() if key not in science
    }
    if not requested_additions and not science_additions:
        return _state(root, document)
    repaired = dict(document)
    repaired["requested_science_extra"] = {
        **(recorded_requested if isinstance(recorded_requested, Mapping) else {}),
        **requested_additions,
    }
    repaired["science_extra"] = {
        **(recorded_science if isinstance(recorded_science, Mapping) else {}),
        **science_additions,
    }
    repaired["intent_revision"] = _state(root, document).revision + 1
    _backup(path)
    _write_stack_document(path, repaired)
    _fsync_dir(root)
    logger.info(
        "Restored mission request extras %s in Stack workspace %s",
        ", ".join(sorted(requested_additions or science_additions)),
        root,
    )
    return _state(root, repaired)


def _adopt_request_only(
    root: Path, request_path: Path, *, on_error: ErrorPolicy
) -> IntentState | None:
    """Adopt one supported request-only workspace into the intent."""
    try:
        request_document = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        if on_error == "raise":
            message = f"{request_path} cannot be read; rebuild the workspace"
            reject(message, WorkspaceIntegrityError, logger=logger)
        logger.warning("Ignoring unreadable legacy request %s", request_path)
        return None
    if (
        not isinstance(request_document, dict)
        or request_document.get("schema") != "faninsar.preparation.v1"
        or not isinstance(request_document.get("source"), Mapping)
        or not isinstance(request_document.get("science"), Mapping)
    ):
        if on_error == "raise":
            message = f"{request_path} is not a supported preparation request"
            reject(message, WorkspaceIntegrityError, logger=logger)
        logger.warning("Ignoring unsupported legacy request %s", request_path)
        return None
    source = request_document["source"]
    science = request_document["science"]
    mission = str(request_document.get("mission") or "")
    # The request document is the only record of the caller's mission request;
    # its extras (for example ``polarizations``) are required to prepare the
    # workspace again and cannot be reconstructed from the source catalog.
    request_extra = requested_extras(request_document.get("extra"))
    document = {
        "schema": STACK_IDENTITY_SCHEMA,
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "intent_revision": 1,
        "mission": mission,
        "identity_status": "configured",
        "reference": _request_reference(source),
        "request": normalize_request(mission, source),
        "targets": _targets_from_science(science),
        "inputs": {"sources": {}, "orbits": {}, "aux_cal": {}},
        "pairs": [],
        "misreg_pairs": [],
        "science": dict(science),
        "science_extra": dict(request_extra),
        "requested_science": dict(science),
        "requested_science_extra": dict(request_extra),
    }
    _backup(request_path)
    _write_stack_document(root / STACK_JSON_NAME, document)
    _fsync_dir(root)
    logger.info("Adopted legacy request-only Stack workspace %s", root)
    return _state(root, document)


def _request_reference(request: Mapping[str, Any]) -> str | None:
    """Return the explicit request Reference without resolving a catalog."""
    topology = request.get("topology")
    value = topology.get("reference") if isinstance(topology, Mapping) else None
    if value is None:
        value = request.get("reference")
    if value is None:
        return None
    return str(value)


def _targets_from_science(science: Mapping[str, Any]) -> dict[str, Any]:
    """Return encoded targets from one request science document."""
    return {
        "roi": science.get("roi"),
        "swaths": science.get("swaths"),
        "bursts": _plain_bursts(science.get("bursts")),
    }


def _adopt_legacy(
    root: Path,
    path: Path,
    document: Mapping[str, Any],
    *,
    on_error: ErrorPolicy,
) -> IntentState | None:
    """Adopt one supported version-1 identity without rerunning science."""
    if not isinstance(document.get("mission"), str) or not isinstance(
        document.get("science"), Mapping
    ):
        # A drifted identity is left for the tolerant open path; adopting it
        # would rewrite metadata whose meaning is not established.
        logger.warning("Legacy identity %s is not adoptable; leaving it intact", path)
        return _state(root, document)
    request_document: dict[str, Any] | None = None
    request_path = root / REQUEST_RECORD
    if request_path.is_file():
        try:
            loaded = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        if (
            isinstance(loaded, dict)
            and loaded.get("schema") == "faninsar.preparation.v1"
        ):
            request_document = loaded
    conflicts = _legacy_conflicts(document, request_document)
    if conflicts and on_error == "raise":
        message = (
            f"{path} and {request_path} disagree on scientific identity: "
            + "; ".join(conflicts)
        )
        reject(message, WorkspaceIntegrityError, logger=logger)
    source = request_document.get("source") if request_document else {}
    source = source if isinstance(source, Mapping) else {}
    requested_science = request_document.get("science") if request_document else None
    requested_science = (
        dict(requested_science) if isinstance(requested_science, Mapping) else None
    )
    recorded_extra = document.get("science_extra")
    recorded_extra = dict(recorded_extra) if isinstance(recorded_extra, Mapping) else {}
    # A version-1 identity records the live science extra; the original
    # preparation request keeps the caller's fixed request extras. Prefer the
    # original snapshot and never drop a recorded value.
    request_extra = requested_extras(
        (request_document or {}).get("extra")
        if request_document is not None
        else recorded_extra
    )
    adopted = dict(document)
    adopted.update(
        {
            "schema": STACK_IDENTITY_SCHEMA,
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "intent_revision": 1,
            "identity_status": (
                "resolved"
                if document.get("inputs", {}).get("sources")
                else "configured"
            ),
            "request": normalize_request(str(document.get("mission") or ""), source),
            "targets": (
                _targets_from_science(requested_science)
                if requested_science is not None
                else _targets_from_science(document.get("science") or {})
            ),
            "requested_science": (
                requested_science
                if requested_science is not None
                else dict(document.get("science") or {})
            ),
            "requested_science_extra": request_extra,
            "science_extra": {**request_extra, **recorded_extra},
        }
    )
    adopted.setdefault("inputs", {"sources": {}, "orbits": {}, "aux_cal": {}})
    adopted.setdefault("pairs", [])
    adopted.setdefault("misreg_pairs", [])
    _backup(path)
    if request_document is not None:
        _backup(request_path)
    _write_stack_document(path, adopted)
    _fsync_dir(root)
    logger.info("Adopted legacy Stack workspace schema at %s", root)
    state = _state(root, adopted, conflicts=conflicts)
    if conflicts:
        logger.warning(
            "Legacy identity conflict retained; prepared reuse is not authorized: %s",
            "; ".join(conflicts),
        )
    return state


def _legacy_conflicts(
    document: Mapping[str, Any], request_document: Mapping[str, Any] | None
) -> tuple[str, ...]:
    """Return scientific-identity disagreements between two legacy records."""
    if request_document is None:
        return ()
    conflicts: list[str] = []
    if str(request_document.get("mission") or "") not in (
        "",
        str(document.get("mission") or ""),
    ):
        conflicts.append("mission")
    request_science = request_document.get("science")
    document_science = document.get("science")
    if isinstance(request_science, Mapping) and isinstance(document_science, Mapping):
        keys = (
            set(request_science) | set(document_science)
        ) - PREPARATION_EXCLUDED_SCIENCE_FIELDS
        for key in sorted(keys):
            if key not in request_science or key not in document_science:
                continue
            if _canonical_json(request_science[key]) != _canonical_json(
                document_science[key]
            ):
                conflicts.append(f"science.{key}")
    source = request_document.get("source")
    if isinstance(source, Mapping):
        requested_reference = _request_reference(source)
        recorded_reference = _document_reference(document)
        if (
            requested_reference is not None
            and recorded_reference is not None
            and requested_reference != recorded_reference
        ):
            conflicts.append("reference")
    return tuple(conflicts)


def _log_conflicts(state: IntentState) -> None:
    """Report retained legacy identity conflicts once per call."""
    for conflict in state.conflicts:
        logger.warning(
            "%s holds a legacy identity conflict on %s; prepared reuse is not "
            "authorized for affected pins",
            state.root,
            conflict,
        )


def _backup(path: Path) -> Path | None:
    """Preserve one prior metadata file before an atomic replacement."""
    if not path.is_file():
        return None
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    shutil.copyfile(path, backup)
    with backup.open("rb") as handle:
        os.fsync(handle.fileno())
    return backup


def _fsync_dir(path: Path) -> None:
    """Make a published directory entry durable on supported local filesystems."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BACKUP_SUFFIX",
    "MUTABLE_EXTRA_KEYS",
    "REQUEST_EXTRA_KEYS",
    "REQUEST_RECORD",
    "WORKSPACE_SCHEMA_VERSION",
    "ExtensionReport",
    "IntentState",
    "WorkspaceConflictError",
    "WorkspaceIntegrityError",
    "build_intent",
    "extend_intent",
    "normalize_request",
    "open_intent",
    "publish_intent",
    "read_intent",
    "requested_extras",
    "union_bursts",
    "union_roi",
    "union_swaths",
]
