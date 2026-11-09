"""``stack.json``: the identity of a Stack on disk (PROPOSAL-0059).

The document holds identity only: mission, Reference, local inputs, the
fully serializable science configuration, the DEM recipe, the mask plan,
and the planned Pair networks. Results, reports, and progress never enter
it. Every :class:`~faninsar.stack.config.StackConfig` field is either a
saved science value or listed in :data:`UNSAVED_FIELDS` (runtime
supplied per process, or derived from products); a field added without
classification fails at save time, and an unserializable science value
fails at save time instead of silently changing the Stack. ``extra``
keys in :data:`SAVED_EXTRA_KEYS` are persisted with the science config;
all other ``extra`` keys are session state and are ignored.

Filesystem paths are stored relative to the identity file's directory when
they live inside the workspace (``acquisitions/...``, ``../../dem.tif``)
and stay absolute when they point outside it.  On read they are resolved
back to absolute paths before the reopen path consumes the document.

Multi-track workspaces (orbitalized Sentinel-1 stacks, stage 6) split the
document: the root ``stack.json`` carries only the schema, mission, the
``tracks`` list, and the global science configuration, while each track
owns a full identity document at ``tracks/<trackKey>/stack.json`` recording
that track's scene catalog, planned pairs, Reference, multilook, and
DEM/mask references. Pair networks and look factors are per-track — a
mixed-track root graph is not a Sentinel-1 plan. A track directory is
independently openable through
:meth:`Stack.open`. On read, :func:`read_stack_identity` and
:func:`read_stack_identity_lenient` merge the per-track documents back
into the root-shaped document the reopen path already consumes. A Stack
occupying zero or one track keeps the legacy flat document unchanged.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from faninsar.logging import reject, setup_logger
from faninsar.processing.errors import InvalidProcessingStateError, reject_invalid_state
from faninsar.stack.layout import (
    TRACKS_DIRNAME,
    encode_identity_path,
    relativize_identity_paths,
    resolve_identity_paths,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from faninsar.stack.session import Stack

logger = setup_logger(__name__)

STACK_IDENTITY_SCHEMA = "faninsar.stack.v1"
STACK_JSON_NAME = "stack.json"

# StackConfig fields that never enter ``stack.json``: runtime values
# supplied per process at ``Stack.open`` and derived values recomputed
# from products. Every other config field is a science value that is
# saved and must round-trip exactly.
UNSAVED_FIELDS = frozenset(
    {
        "work_dir",
        "device",
        "invert_device",
        "executor",
        "n_jobs",
        "gpu_memory_reclaim",
        "resource_budget",
        "extra",
    }
)
# ``extra`` keys persisted with the science configuration. All other
# keys are session state — runtime caches, provider selections, search
# bookkeeping — and are ignored.
# Science fields that belong on ``tracks/<key>/stack.json`` rather than the
# multi-track workspace root.  Look factors can differ per track; a leftover
# root copy would silently override the track that actually formed products.
TRACK_OWNED_SCIENCE_FIELDS = frozenset({"multilook"})
# Unwrap-only science persisted on ``stack.json`` and recorded in
# ``inputs/request.json``, but stripped from preparation identity
# comparison. Changing these must not reject a workspace whose
# ``inputs/request.json`` predates the field, or invalidate source
# readiness after a live config mutation.
PREPARATION_EXCLUDED_SCIENCE_FIELDS = frozenset(
    {"nlooks_divisor", "ref_points", "ref_policy"}
)

SAVED_EXTRA_KEYS = frozenset(
    {
        "orbit_paths",
        "aux_cal_paths",
        "polarizations",
        "orbit_auto_fetch",
        "aux_cal_auto_fetch",
        "frame_id",
        "swath",
        "channel",
        "nisar_window",
        "nisar_tile_shape",
        "rslc_window",
        "height",
        "height_m",
        "track_metadata",
        "preparation_report",
        "preparation_dem_scope",
        "preparation_local_inputs",
        "nisar_provider",
        "source_lineage",
        "source_admission",
        "nisar_admission",
        "admission_policy",
        "mission",
        "product",
        "frequency",
        "polarization",
    }
)


def _fail_unserializable(value: object) -> NoReturn:
    message = (
        "stack.json cannot represent "
        f"{type(value).__name__!r}; classify the value or pass a serializable "
        "science configuration"
    )
    reject_invalid_state(message)


def encode_value(value: Any, *, identity_dir: Path | None = None) -> object:
    """Encode one strict-JSON science value.

    Tuples are tagged so they round-trip as tuples.  Paths become strings
    relative to ``identity_dir`` when they live inside the workspace,
    grids and ROI geometries become typed payloads, and anything else
    unknown is rejected instead of being dropped or coerced.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail_unserializable(value)
        return value  # finite float
    if isinstance(value, Path):
        return (
            str(value)
            if identity_dir is None
            else encode_identity_path(value, identity_dir)
        )
    if isinstance(value, (tuple, list)):
        encoded = [encode_value(item, identity_dir=identity_dir) for item in value]
        return {"faninsar:tuple": encoded} if isinstance(value, tuple) else encoded
    if isinstance(value, dict):
        return {
            str(key): encode_value(item, identity_dir=identity_dir)
            for key, item in value.items()
        }
    return _encode_geometry_or_dem(value, identity_dir=identity_dir)


def _encode_geometry_or_dem(value: Any, *, identity_dir: Path | None = None) -> object:
    """Encode an ROI geometry, a mask plan, a grid, or a DEM recipe."""
    from faninsar.data.query.bbox import BoundingBox
    from faninsar.data.query.points import Points
    from faninsar.processing.geometry.api import (
        ConstantDEM,
        RasterDEM,
        SourceDEM,
    )
    from faninsar.processing.geometry.grids import GridSpec
    from faninsar.stack.mask_plan import MaskPlan

    if isinstance(value, (Points, BoundingBox)):
        if isinstance(value, Points):
            from faninsar.data.reference import encode_points_payload

            encoded: object = {"faninsar:points": encode_points_payload(value)}
        else:
            from pyproj import CRS

            bounds = [value.left, value.bottom, value.right, value.top]
            crs = (
                None
                if value.crs is None
                else CRS.from_user_input(value.crs).to_string()
            )
            encoded = {"faninsar:bbox": {"bounds": bounds, "crs": crs}}
        return encoded
    if isinstance(value, GridSpec):
        return {
            "faninsar:grid": {
                "crs": str(value.crs),
                "transform": [float(item) for item in tuple(value.transform)[:6]],
                "height": int(value.height),
                "width": int(value.width),
            }
        }
    if isinstance(value, MaskPlan):
        mapping = value.to_mapping()
        if identity_dir is not None:
            relativize_identity_paths(mapping, identity_dir)
        return {"faninsar:mask_plan": mapping}
    if isinstance(value, (SourceDEM, RasterDEM, ConstantDEM)):
        return _encode_dem(value, identity_dir=identity_dir)
    from faninsar.data.query.polygons import Polygons

    if isinstance(value, Polygons):
        gdf = value.geodataframe
        crs = None if gdf.crs is None else str(gdf.crs)
        features = [
            {
                "geometry": geom.__geo_interface__,
                "properties": {"types": row["types"]},
            }
            for (_, row), geom in zip(gdf.iterrows(), gdf.geometry, strict=True)
        ]
        return {
            "faninsar:polygons": {
                "features": features,
                "crs": crs,
                "all_touched": value.all_touched,
                "pad": value.pad,
            }
        }
    if hasattr(value, "__geo_interface__"):
        from pyproj import CRS

        crs = getattr(value, "crs", None)
        return {
            "faninsar:geometry": dict(value.__geo_interface__),
            "faninsar:crs": (
                None if crs is None else CRS.from_user_input(crs).to_string()
            ),
        }
    _fail_unserializable(value)


def _encode_dem(value: Any, *, identity_dir: Path | None = None) -> dict[str, object]:
    """Encode a DEM recipe (source selection, raster path, or constant).

    Raster and constant recipes persist the admitted vertical datum with the
    existing path or height so reopen reconstructs the same request.  A
    payload that omits the datum is restored through the public factory
    without inventing one.
    """
    from faninsar.processing.geometry.api import ConstantDEM, RasterDEM, SourceDEM

    def _path(recorded: Path) -> str:
        if identity_dir is None:
            return str(recorded)
        return encode_identity_path(recorded, identity_dir)

    if isinstance(value, SourceDEM):
        return {
            "faninsar:dem": {
                "kind": "source",
                "product": value.product,
                "provider": value.provider,
                "cache_dir": (
                    None if value.cache_dir is None else _path(value.cache_dir)
                ),
            }
        }
    if isinstance(value, RasterDEM):
        if value.path is None:
            _fail_unserializable(value)
        return {
            "faninsar:dem": {
                "kind": "raster",
                "path": _path(value.path),
                "vertical_datum": value.vertical_datum,
            }
        }
    if isinstance(value, ConstantDEM):
        return {
            "faninsar:dem": {
                "kind": "constant",
                "height": float(value.height),
                "vertical_datum": value.vertical_datum,
            }
        }
    _fail_unserializable(value)


def decode_value(value: object) -> Any:
    """Decode one encoded science value back into its Python form."""
    if isinstance(value, dict):
        decoded = _decode_typed(value)
        if decoded is not _UNTYPED:
            return decoded
        return {str(key): decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    return value


_UNTYPED = object()


def _decode_typed(value: Mapping[str, object]) -> Any:
    """Decode one typed payload, or return the ``_UNTYPED`` sentinel."""
    if "faninsar:tuple" in value:
        return tuple(decode_value(item) for item in value["faninsar:tuple"])
    if "faninsar:grid" in value:
        from faninsar.processing.geometry.grids import GridSpec

        payload = value["faninsar:grid"]
        return GridSpec(
            crs=payload["crs"],
            transform=tuple(float(item) for item in payload["transform"]),
            height=int(payload["height"]),
            width=int(payload["width"]),
        )
    if (
        "faninsar:bbox" in value
        or "faninsar:geometry" in value
        or "faninsar:polygons" in value
    ):
        return _decode_geometry(value)
    if "faninsar:mask_plan" in value:
        from faninsar.stack.mask_plan import MaskPlan

        return MaskPlan.from_mapping(value["faninsar:mask_plan"])
    if "faninsar:dem" in value:
        decoded: Any = _decode_dem(value["faninsar:dem"])
    elif "faninsar:points" in value:
        from faninsar.data.reference import decode_points_payload

        decoded = decode_points_payload(value["faninsar:points"])
    else:
        decoded = _UNTYPED
    return decoded


def _decode_polygons(payload: Mapping[str, object]) -> Any:
    """Decode a ``faninsar:polygons`` payload back to :class:`Polygons`."""
    import geopandas as gpd
    import shapely.geometry

    from faninsar.data.query.polygons import Polygons

    features = payload["features"]
    geometries = [shapely.geometry.shape(f["geometry"]) for f in features]
    types = [f["properties"]["types"] for f in features]
    crs = payload.get("crs")
    gdf = gpd.GeoDataFrame({"types": types}, geometry=geometries, crs=crs)
    return Polygons(
        gdf,
        types=types,
        all_touched=bool(payload.get("all_touched", True)),
        pad=bool(payload.get("pad", False)),
    )


def _decode_geometry(value: Mapping[str, object]) -> Any:
    """Decode one ROI geometry payload."""
    if "faninsar:polygons" in value:
        return _decode_polygons(value["faninsar:polygons"])
    if "faninsar:bbox" in value:
        from faninsar.data.query.bbox import BoundingBox

        payload = value["faninsar:bbox"]
        return BoundingBox(
            *(float(item) for item in payload["bounds"]),
            crs=payload["crs"],
        )
    import shapely

    geometry = shapely.geometry.shape(value["faninsar:geometry"])
    crs = value.get("faninsar:crs")
    if crs is None:
        return geometry
    import geopandas as gpd

    from faninsar.data.query.polygons import Polygons

    return Polygons(gpd.GeoSeries([geometry], crs=crs))


def _decode_dem(payload: Mapping[str, object]) -> Any:
    from faninsar.processing.geometry.api import DEM

    kind = payload.get("kind")
    if kind == "source":
        from faninsar.processing.geometry.api import SourceDEM

        provider = payload.get("provider")
        selection = str(payload["product"])
        if provider is not None:
            selection = f"{selection}:{provider}"
        cache_dir = payload.get("cache_dir")
        return SourceDEM(
            selection,
            cache_dir=None if cache_dir is None else Path(str(cache_dir)),
        )
    if kind == "raster":
        path = Path(str(payload["path"]))
        datum = payload.get("vertical_datum")
        if datum is None:
            return DEM.from_raster(path)
        return DEM.from_raster(path, vertical_datum=str(datum))
    if kind == "constant":
        height = float(payload["height"])
        datum = payload.get("vertical_datum")
        if datum is None:
            return DEM.from_constant(height)
        return DEM.from_constant(height, vertical_datum=str(datum))
    message = f"stack.json DEM recipe is invalid: {kind!r}"
    return reject(message, InvalidProcessingStateError, logger=logger)


def _input_record(path: Path, identity_dir: Path) -> dict[str, object]:
    """Return one record for a local input: path, size, and verification.

    The record carries the download-time PROPOSAL-0055 verification outcome
    (``{status, algorithm, size}``) so the persisted ``stack.json`` surfaces
    whether each input was ``verified``, ``failed``, or ``unverifiable`` at
    download. An input with no download record (hand-acquired, or no provider
    digest) is surfaced as ``unverifiable`` — the honest outcome.
    """
    if not path.exists():
        message = f"Stack input is missing: {path}"
        reject_invalid_state(message)
    from faninsar.remote.fetch.complete import read_verification

    record = read_verification(path) or {}
    return {
        "path": encode_identity_path(path, identity_dir),
        "size": path.stat().st_size,
        "verification": {
            "status": record.get("status", "unverifiable"),
            "algorithm": record.get("algorithm"),
            "size": record.get("size", path.stat().st_size),
        },
    }


def stack_identity(stack: Stack) -> dict[str, object]:
    """Build the ``stack.json`` document for one Stack.

    Parameters
    ----------
    stack : Stack
        Any Stack whose science configuration is serializable.

    Returns
    -------
    dict
        The identity document: schema, mission, Reference, inputs, planned
        Pair networks, and science configuration.

    Raises
    ------
    InvalidProcessingStateError
        If a config field or ``extra`` key is unclassified, or a science
        value cannot be represented strictly.

    """
    identity_dir = Path(stack.config.work_dir)
    science, extra = _science_and_extra(stack, identity_dir=identity_dir)
    config = stack.config
    document = {
        "schema": STACK_IDENTITY_SCHEMA,
        "mission": type(stack).__name__,
        "reference": stack.reference,
        "inputs": {
            "sources": {
                date: [_input_record(Path(path), identity_dir) for path in paths]
                for date, paths in stack.catalog.paths.items()
            },
            "orbits": {
                date: _input_record(Path(path), identity_dir)
                for date, path in _orbit_inputs(config).items()
            },
            "aux_cal": {
                date: _input_record(Path(path), identity_dir)
                for date, path in _aux_cal_inputs(config).items()
            },
        },
        "pairs": [f"{first}_{second}" for first, second in _pair_dates(stack.pairs)],
        "misreg_pairs": [
            f"{first}_{second}" for first, second in _pair_dates(stack.misreg_pairs)
        ],
        "science": science,
        "science_extra": extra,
    }
    relativize_identity_paths(document, identity_dir)
    return document


def _science_and_extra(
    stack: Stack, *, identity_dir: Path | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    """Return the encoded ``(science, science_extra)`` pair for one Stack.

    The DEM recipe prefers the captured ``_dem_source`` over the config
    ``dem`` so a fetched/materialized DEM persists as its source recipe.
    """
    config = stack.config
    science: dict[str, object] = {}
    for field in fields(type(config)):
        if field.name in UNSAVED_FIELDS:
            continue
        value = getattr(config, field.name)
        if field.name == "dem" and stack._dem_source is not None:
            value = stack._dem_source
        science[field.name] = encode_value(value, identity_dir=identity_dir)
    extra = {
        key: encode_value(config.extra[key], identity_dir=identity_dir)
        for key in sorted(config.extra)
        if key in SAVED_EXTRA_KEYS
    }
    from faninsar.remote.records import _sanitize

    return _sanitize(science), _sanitize(extra)


def _orbit_inputs(config: Any) -> dict[str, object]:
    """Return the recorded per-date orbit paths, if any."""
    value = config.extra.get("orbit_paths")
    if isinstance(value, dict):
        return {str(date): path for date, path in value.items() if path is not None}
    return {}


def _aux_cal_inputs(config: Any) -> dict[str, object]:
    """Return the recorded per-date AUX_CAL paths, if any."""
    value = config.extra.get("aux_cal_paths")
    if isinstance(value, dict):
        return {str(date): path for date, path in value.items() if path is not None}
    return {}


def fields(config: Any) -> tuple[Any, ...]:
    """Return the dataclass fields of one StackConfig."""
    from dataclasses import fields as dataclass_fields

    return dataclass_fields(config)


def _pair_dates(pairs: Any) -> list[tuple[str, str]]:
    """Normalize one planned Pair network through the canonical ordering."""
    from faninsar.stack.session import _iter_pair_dates

    return _iter_pair_dates(pairs)


def save_stack_identity(stack: Stack) -> Path:
    """Write ``stack.json`` atomically into the Stack work directory.

    A Stack occupying more than one track group writes a split identity:
    each track gets a complete ``tracks/<trackKey>/stack.json`` (its own
    scene catalog, planned pairs, Reference, and DEM/mask references, so
    the directory is independently openable) and the root document keeps
    only the schema, mission, the sorted ``tracks`` list, and the global
    science configuration. Pair networks and Reference live on the track
    documents; a mixed-track root graph is not a Sentinel-1 plan. A Stack
    occupying zero or one track writes the legacy flat document unchanged.

    Parameters
    ----------
    stack : Stack
        The Stack whose identity is saved.

    Returns
    -------
    pathlib.Path
        The written root ``stack.json``.

    """
    work_dir = stack.config.work_dir
    # Read the recorded track map only — never resolve it here. Resolving
    # through ``track_groups``/``_catalog_track_infos`` would open SAFEs
    # and, worse, write ``track_metadata`` into ``config.extra`` mid-save,
    # changing the coregistration resume digest so every committed scene
    # would go stale on the next stage.
    track_keys = _saved_track_keys(stack)
    if len(track_keys) > 1:
        # Per-track documents land before the root document so a crash
        # mid-save never leaves a root doc naming a track file that does
        # not exist yet.
        for track_key in track_keys:
            sub_document = stack_identity(_track_scoped_stack(stack, track_key))
            _write_stack_document(
                work_dir / TRACKS_DIRNAME / track_key / STACK_JSON_NAME,
                sub_document,
            )
        science, extra = _science_and_extra(stack, identity_dir=work_dir)
        for field_name in TRACK_OWNED_SCIENCE_FIELDS:
            science.pop(field_name, None)
        document = {
            "schema": STACK_IDENTITY_SCHEMA,
            "mission": type(stack).__name__,
            "tracks": list(track_keys),
            "science": science,
            "science_extra": extra,
        }
        relativize_identity_paths(document, work_dir)
    else:
        document = stack_identity(stack)
    target = _write_stack_document(work_dir / STACK_JSON_NAME, document)
    logger.info("Saved Stack identity %s", target)
    return target


def _write_stack_document(path: Path, document: dict[str, object]) -> Path:
    """Write one ``stack.json`` document atomically (tmp + fsync + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _saved_track_dates(stack: Stack) -> dict[str, str]:
    """Return ``{date_id: track_key}`` from recorded metadata only.

    Reads ``config.extra["track_metadata"]["dates"]`` — the mapping a
    prepared Stack persists — without invoking mission track resolution.
    Values that are decoded ``TrackAcquisitionInfo``-like objects or
    persisted dict records both contribute their ``track_key`` field.
    """
    metadata = stack.config.extra.get("track_metadata")
    dates = metadata.get("dates") if isinstance(metadata, dict) else None
    if not isinstance(dates, dict):
        return {}
    keys: dict[str, str] = {}
    for date_id, record in dates.items():
        if isinstance(record, dict):
            key = record.get("track_key")
        else:
            key = getattr(record, "track_key", None)
        if key:
            keys[str(date_id)] = str(key)
    return keys


def _saved_track_keys(stack: Stack) -> tuple[str, ...]:
    """Return the sorted distinct track keys in the recorded metadata."""
    return tuple(sorted(set(_saved_track_dates(stack).values())))


def _track_scoped_stack(stack: Stack, track_key: str) -> Stack:
    """Return a lightweight Stack view scoped to one track group.

    The returned Stack shares the parent's class and runtime fields but
    restricts the catalog, planned pairs, misregistration pairs, and the
    persisted ``extra`` date keys (``orbit_paths``, ``aux_cal_paths``,
    ``track_metadata``) to the dates whose track metadata reports
    ``track_key``. Its
    ``config.work_dir`` points at ``tracks/<trackKey>`` so the identity
    document built from it describes an independently openable directory.
    """
    from dataclasses import replace

    from faninsar.core.pair import Pairs as PairsType
    from faninsar.stack.catalog import SceneCatalog

    recorded = _saved_track_dates(stack)
    dates = tuple(
        date_id for date_id in stack.catalog.dates if recorded.get(date_id) == track_key
    )
    selected = set(dates)
    catalog = SceneCatalog({date_id: stack.catalog.paths[date_id] for date_id in dates})

    def _scoped_pairs(network: object) -> PairsType:
        names = [
            f"{first}_{second}"
            for first, second in _pair_dates(network)
            if first in selected and second in selected
        ]
        return PairsType.from_names(names)

    reference = (
        stack.reference
        if stack.reference in selected
        else (dates[0] if dates else stack.reference)
    )

    extra = dict(stack.config.extra)
    for date_key in ("orbit_paths", "aux_cal_paths"):
        bindings = extra.get(date_key)
        if isinstance(bindings, dict):
            extra[date_key] = {
                date: path for date, path in bindings.items() if date in selected
            }
    metadata = extra.get("track_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("dates"), dict):
        scoped = dict(metadata)
        scoped["dates"] = {
            date: value for date, value in metadata["dates"].items() if date in selected
        }
        extra["track_metadata"] = scoped

    config = replace(
        stack.config,
        work_dir=stack.config.work_dir / TRACKS_DIRNAME / track_key,
        extra=extra,
    )
    scoped = type(stack)(
        catalog=catalog,
        config=config,
        pairs=_scoped_pairs(stack.pairs),
        misreg_pairs=_scoped_pairs(stack.misreg_pairs),
        reference=reference,
    )
    # The captured source recipe can differ from ``config.dem`` (it is
    # also bound when a DEM is supplied per call); copying it keeps the
    # per-track document's DEM reference identical to the parent's.
    scoped._dem_source = stack._dem_source
    return scoped


def read_stack_identity(work_dir: str | Path) -> dict[str, object]:
    """Read and validate the ``stack.json`` of one work directory.

    Raises
    ------
    InvalidProcessingStateError
        If the file is missing, is not a FanInSAR Stack identity, or carries
        an unsupported schema (an earlier layout must be rebuilt).

    """
    path = Path(work_dir) / STACK_JSON_NAME
    if not path.is_file():
        message = (
            f"{path} is missing; this work directory was not created by the "
            "current FanInSAR Stack — rebuild it"
        )
        reject_invalid_state(message)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        message = f"{path} cannot be read: {error}"
        logger.exception(message)
        reject_invalid_state(message)
    if (
        not isinstance(document, dict)
        or document.get("schema") != STACK_IDENTITY_SCHEMA
        or "mission" not in document
        or "science" not in document
    ):
        message = (
            f"{path} is not a FanInSAR Stack identity; rebuild this work "
            "directory with the current FanInSAR version"
        )
        reject_invalid_state(message)
    return merge_track_documents(Path(work_dir), document, strict=True)


def merge_track_documents(
    work_dir: Path,
    document: dict[str, object],
    *,
    strict: bool,
) -> dict[str, object]:
    """Expand a track-split root document into the flat identity shape.

    A root document carrying a ``tracks`` list holds no ``inputs`` of its
    own; each entry names a complete identity document at
    ``tracks/<trackKey>/stack.json``. This merges those documents —
    unioning inputs (sources, orbits, aux_cal) and deep-merging
    ``science_extra`` date-keyed sections (``orbit_paths``,
    ``track_metadata.dates``) — so the reopen path sees the same document
    a single-track workspace writes. Pair networks always come from the
    per-track documents (a leftover mixed-track root graph is ignored).
    ``science`` stays on the root; ``reference`` is kept when the root
    still records one, otherwise it is the earliest merged date.

    Parameters
    ----------
    work_dir : pathlib.Path
        Work directory holding the root ``stack.json``.
    document : dict
        The already-validated root document.
    strict : bool
        ``True`` rejects a missing/undecodable per-track document naming
        the track; ``False`` (lenient recovery path) skips it so the
        caller can decide what is recoverable.

    Returns
    -------
    dict
        The merged flat document — the input unchanged when it carries no
        ``tracks`` list.

    Raises
    ------
    InvalidProcessingStateError
        ``strict`` only: a listed track's ``stack.json`` is missing,
        undecodable, or lacks the identity schema.

    """
    tracks = document.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        resolve_identity_paths(document, work_dir)
        return document

    resolve_identity_paths(document, work_dir)
    merged = dict(document)
    sources: dict[str, object] = {}
    orbits: dict[str, object] = {}
    aux_cal: dict[str, object] = {}
    pairs: list[str] = []
    misreg_pairs: list[str] = []
    extra: dict[str, object] = {}
    track_science: dict[str, object] = {}
    root_extra = document.get("science_extra")

    for entry in tracks:
        track_key = str(entry)
        sub_path = work_dir / TRACKS_DIRNAME / track_key / STACK_JSON_NAME
        sub_document: object = None
        if sub_path.is_file():
            try:
                sub_document = json.loads(sub_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                if strict:
                    message = f"{sub_path} cannot be read: {error}"
                    logger.exception(message)
                    reject_invalid_state(message)
                continue
        if not isinstance(sub_document, dict):
            if strict:
                message = (
                    f"{sub_path} is missing or unreadable; track "
                    f"{track_key!r} is listed in {STACK_JSON_NAME} but its "
                    "own identity document cannot be loaded — restore it "
                    "or rebuild the work directory"
                )
                reject_invalid_state(message)
            continue
        if strict and (
            sub_document.get("schema") != STACK_IDENTITY_SCHEMA
            or "inputs" not in sub_document
        ):
            message = (
                f"{sub_path} is not a FanInSAR Stack identity for track "
                f"{track_key!r}; rebuild this work directory"
            )
            reject_invalid_state(message)

        resolve_identity_paths(sub_document, sub_path.parent)
        inputs = sub_document.get("inputs")
        if isinstance(inputs, dict):
            sub_sources = inputs.get("sources")
            if isinstance(sub_sources, dict):
                for date, entries in sub_sources.items():
                    bucket = sources.setdefault(str(date), [])
                    if isinstance(entries, list):
                        bucket.extend(entries)  # type: ignore[union-attr]
            sub_orbits = inputs.get("orbits")
            if isinstance(sub_orbits, dict):
                orbits.update(sub_orbits)
            sub_aux_cal = inputs.get("aux_cal")
            if isinstance(sub_aux_cal, dict):
                aux_cal.update(sub_aux_cal)
        for key, target in (("pairs", pairs), ("misreg_pairs", misreg_pairs)):
            values = sub_document.get(key)
            if isinstance(values, list):
                for name in values:
                    text = str(name)
                    if text not in target:
                        target.append(text)
        sub_extra = sub_document.get("science_extra")
        if isinstance(sub_extra, dict):
            _merge_extra_section(extra, sub_extra)
        sub_science = sub_document.get("science")
        if isinstance(sub_science, dict):
            for field_name in TRACK_OWNED_SCIENCE_FIELDS:
                if field_name in sub_science and field_name not in track_science:
                    track_science[field_name] = sub_science[field_name]

    if isinstance(root_extra, dict):
        _merge_extra_section(extra, root_extra, override=True)

    merged["inputs"] = {
        "sources": sources,
        "orbits": orbits,
        "aux_cal": aux_cal,
    }
    merged["pairs"] = pairs
    merged["misreg_pairs"] = misreg_pairs
    if not isinstance(merged.get("reference"), str) or not merged.get("reference"):
        dates = sorted(sources)
        if dates:
            merged["reference"] = dates[0]
    science = merged.get("science")
    if isinstance(science, dict):
        for field_name in TRACK_OWNED_SCIENCE_FIELDS:
            science.pop(field_name, None)
            if field_name in track_science:
                science[field_name] = track_science[field_name]
    merged["science_extra"] = extra
    return merged


def _merge_extra_section(
    target: dict[str, object],
    source: dict[str, object],
    *,
    override: bool = False,
) -> None:
    """Merge one ``science_extra`` mapping into the accumulated extra.

    ``orbit_paths``, ``aux_cal_paths``, and ``track_metadata.dates`` are
    merged per date so one track's recorded entries join the others; every
    other key keeps
    the first value seen unless ``override`` (the root document) wins.
    """
    for key, value in source.items():
        if key in {"orbit_paths", "aux_cal_paths"} and isinstance(value, dict):
            existing = target.setdefault(key, {})
            if isinstance(existing, dict):
                for date, entry in value.items():
                    existing.setdefault(str(date), entry)
            continue
        if key == "track_metadata" and isinstance(value, dict):
            existing_meta = target.setdefault("track_metadata", {"dates": {}})
            if isinstance(existing_meta, dict):
                existing_dates = existing_meta.setdefault("dates", {})
                new_dates = value.get("dates")
                if isinstance(existing_dates, dict) and isinstance(new_dates, dict):
                    for date, entry in new_dates.items():
                        existing_dates.setdefault(str(date), entry)
            continue
        if override or key not in target:
            target[key] = value


def read_stack_identity_lenient(work_dir: str | Path) -> dict[str, object] | None:
    """Read ``stack.json`` without enforcing the schema contract.

    This is the recovery-path reader used by :meth:`Stack.rescan`,
    :meth:`Stack.repair`, and the tolerant branch of :meth:`Stack.open`
    (PROPOSAL-0069).  Unlike :func:`read_stack_identity` it never rejects on
    schema/field drift: the JSON is parsed and returned verbatim so the
    caller can decide which sections are recoverable.  ``None`` signals that
    there is no document to recover — the file is absent or not JSON.

    Parameters
    ----------
    work_dir : str or pathlib.Path
        Work directory that may hold ``stack.json``.

    Returns
    -------
    dict or None
        The parsed document, or ``None`` when ``stack.json`` is missing or
        cannot be decoded as a JSON object.

    """
    path = Path(work_dir) / STACK_JSON_NAME
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    return merge_track_documents(Path(work_dir), document, strict=False)
