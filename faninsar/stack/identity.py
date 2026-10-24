"""``stack.json``: the identity of a Stack on disk (PROPOSAL-0059).

The document holds identity only: mission, Reference, local inputs, the
fully serializable science configuration, the DEM recipe, the mask plan,
and the planned Pair networks. Results, reports, and progress never enter
it. Every :class:`~faninsar.stack.config.StackConfig` field and every
``extra`` key is classified as science, runtime, or derived in the tables
below; a field added without classification fails at save time, and an
unserializable science value fails at save time instead of silently
changing the Stack.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from faninsar.logging import setup_logger
from faninsar.processing.errors import InvalidProcessingStateError, reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Mapping

    from faninsar.stack.session import Stack

logger = setup_logger(__name__)

STACK_IDENTITY_SCHEMA = "faninsar.stack.v1"
STACK_JSON_NAME = "stack.json"

# Science values survive reopen exactly; a changed value is rejected.
SCIENCE_FIELDS = frozenset(
    {
        "coreg_mode",
        "flatten_stage",
        "coregistration_grid",
        "esd_coherence_threshold",
        "ampcor_snr_threshold",
        "multilook",
        "goldstein_alpha",
        "dem",
        "geo_grid",
        "swaths",
        "bursts",
        "roi",
        "mask_plan",
        "control_spacing",
        "retain_pair_states",
        "record_scientific_lineage",
        "grid",
        "resolution_m",
        "dem_cache_dir",
        "on_network_failure",
    }
)
# Runtime values are supplied per process at ``Stack.open`` and never saved.
RUNTIME_FIELDS = frozenset(
    {
        "work_dir",
        "device",
        "invert_device",
        "executor",
        "n_jobs",
        "gpu_memory_reclaim",
        "resource_budget",
        "activation_binding",
        "activation_token",
        "activation_authority_root",
    }
)
# Derived values are recomputed from products and never saved.
DERIVED_FIELDS = frozenset({"extra", "activation_mode"})

SCIENCE_EXTRA_KEYS = frozenset(
    {
        "orbit_paths",
        "calibration_paths",
        "polarizations",
        "orbit_auto_fetch",
        "frame_id",
        "swath",
        "channel",
        "nisar_window",
        "nisar_tile_shape",
        "rslc_window",
        "height",
        "height_m",
    }
)
RUNTIME_EXTRA_KEYS = frozenset(
    {
        "orbit_cache_dir",
        "geo_lut_cache_dir",
        "geo_work_dir",
        "admission_policy",
    }
)
DERIVED_EXTRA_KEYS = frozenset(
    {
        "selected_footprints",
        "excluded_acquisitions",
        "search_product_type",
        "search_polarizations",
        "search_items",
    }
)


def _fail_unserializable(value: object) -> NoReturn:
    message = (
        "stack.json cannot represent "
        f"{type(value).__name__!r}; classify the value or pass a serializable "
        "science configuration"
    )
    logger.error(message)
    reject_invalid_state(message)


def encode_value(value: Any) -> object:
    """Encode one strict-JSON science value.

    Tuples are tagged so they round-trip as tuples.  Paths become strings,
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
        return str(value)
    if isinstance(value, (tuple, list)):
        encoded = [encode_value(item) for item in value]
        return {"faninsar:tuple": encoded} if isinstance(value, tuple) else encoded
    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in value.items()}
    return _encode_geometry_or_dem(value)


def _encode_geometry_or_dem(value: Any) -> object:
    """Encode an ROI geometry, a mask plan, a grid, or a DEM recipe."""
    from faninsar.data.query.bbox import BoundingBox
    from faninsar.processing.geometry.api import (
        ConstantDEM,
        RasterDEM,
        SourceDEM,
    )
    from faninsar.processing.geometry.grids import GridSpec
    from faninsar.stack.mask_plan import MaskPlan

    if isinstance(value, BoundingBox):
        from pyproj import CRS

        bounds = [value.left, value.bottom, value.right, value.top]
        crs = None if value.crs is None else CRS.from_user_input(value.crs).to_string()
        return {"faninsar:bbox": {"bounds": bounds, "crs": crs}}
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
        return {"faninsar:mask_plan": value.to_mapping()}
    if isinstance(value, (SourceDEM, RasterDEM, ConstantDEM)):
        return _encode_dem(value)
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


def _encode_dem(value: Any) -> dict[str, object]:
    """Encode a DEM recipe (source selection, raster path, or constant)."""
    from faninsar.processing.geometry.api import ConstantDEM, RasterDEM, SourceDEM

    if isinstance(value, SourceDEM):
        return {
            "faninsar:dem": {
                "kind": "source",
                "product": value.product,
                "provider": value.provider,
                "cache_dir": (
                    None if value.cache_dir is None else str(value.cache_dir)
                ),
            }
        }
    if isinstance(value, RasterDEM):
        if value.path is None:
            _fail_unserializable(value)
        return {"faninsar:dem": {"kind": "raster", "path": str(value.path)}}
    if isinstance(value, ConstantDEM):
        return {"faninsar:dem": {"kind": "constant", "height": float(value.height)}}
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
    if "faninsar:bbox" in value or "faninsar:geometry" in value:
        return _decode_geometry(value)
    if "faninsar:mask_plan" in value:
        from faninsar.stack.mask_plan import MaskPlan

        return MaskPlan.from_mapping(value["faninsar:mask_plan"])
    if "faninsar:dem" in value:
        return _decode_dem(value["faninsar:dem"])
    return _UNTYPED


def _decode_geometry(value: Mapping[str, object]) -> Any:
    """Decode one ROI geometry payload."""
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
        return DEM.from_raster(Path(str(payload["path"])))
    if kind == "constant":
        return DEM.from_constant(float(payload["height"]))
    message = f"stack.json DEM recipe is invalid: {kind!r}"
    logger.error(message)
    raise InvalidProcessingStateError(message)


def _input_record(path: Path) -> dict[str, object]:
    """Return one record for a local input: path, size, and verification.

    The record carries the download-time PROPOSAL-0055 verification outcome
    (``{status, algorithm, size}``) so the persisted ``stack.json`` surfaces
    whether each input was ``verified``, ``failed``, or ``unverifiable`` at
    download. An input with no download record (hand-acquired, or no provider
    digest) is surfaced as ``unverifiable`` — the honest outcome.
    """
    if not path.exists():
        message = f"Stack input is missing: {path}"
        logger.error(message)
        reject_invalid_state(message)
    from faninsar.remote.transfer import read_verification

    record = read_verification(path) or {}
    return {
        "path": str(path),
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
        Pair networks, activation mode, and science configuration.

    Raises
    ------
    InvalidProcessingStateError
        If a config field or ``extra`` key is unclassified, or a science
        value cannot be represented strictly.

    """
    config = stack.config
    classified = {field.name for field in fields(type(config))}
    unknown_fields = classified - SCIENCE_FIELDS - RUNTIME_FIELDS - DERIVED_FIELDS
    if unknown_fields:
        message = (
            "StackConfig fields are not classified in the stack.json tables: "
            + ", ".join(sorted(unknown_fields))
        )
        logger.error(message)
        reject_invalid_state(message)
    unclassified_extra = set(config.extra) - (
        SCIENCE_EXTRA_KEYS | RUNTIME_EXTRA_KEYS | DERIVED_EXTRA_KEYS
    )
    if unclassified_extra:
        message = (
            "StackConfig.extra keys are not classified in the stack.json tables: "
            + ", ".join(sorted(unclassified_extra))
        )
        logger.error(message)
        reject_invalid_state(message)
    science: dict[str, object] = {}
    for name in sorted(SCIENCE_FIELDS):
        value = getattr(config, name)
        if name == "dem" and stack._dem_source is not None:
            value = stack._dem_source
        science[name] = encode_value(value)
    extra = {
        key: encode_value(config.extra[key])
        for key in sorted(config.extra)
        if key in SCIENCE_EXTRA_KEYS
    }
    return {
        "schema": STACK_IDENTITY_SCHEMA,
        "mission": type(stack).__name__,
        "reference": stack.reference,
        "inputs": {
            "sources": {
                date: [_input_record(Path(path)) for path in paths]
                for date, paths in stack.catalog.paths.items()
            },
            "orbits": {
                date: _input_record(Path(path))
                for date, path in _orbit_inputs(config).items()
            },
            "calibrations": [
                _input_record(Path(path)) for path in _calibration_inputs(config)
            ],
        },
        "pairs": [f"{first}_{second}" for first, second in _pair_dates(stack.pairs)],
        "misreg_pairs": [
            f"{first}_{second}" for first, second in _pair_dates(stack.misreg_pairs)
        ],
        "activation_mode": config.activation_mode,
        "science": science,
        "science_extra": extra,
    }


def _orbit_inputs(config: Any) -> dict[str, object]:
    """Return the recorded per-date orbit paths, if any."""
    value = config.extra.get("orbit_paths")
    if isinstance(value, dict):
        return {str(date): path for date, path in value.items() if path is not None}
    return {}


def _calibration_inputs(config: Any) -> list[object]:
    """Return the recorded calibration paths, if any."""
    value = config.extra.get("calibration_paths", ())
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


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

    Parameters
    ----------
    stack : Stack
        The Stack whose identity is saved.

    Returns
    -------
    pathlib.Path
        The written ``stack.json``.

    """
    document = stack_identity(stack)
    target = stack.config.work_dir / STACK_JSON_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = target.with_name(f".{STACK_JSON_NAME}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Saved Stack identity %s", target)
    return target


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
        logger.error(message)
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
        logger.error(message)
        reject_invalid_state(message)
    return document
