"""Versioned local NISAR provider recipes, without eager reader reopening."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from faninsar.logging import reject


def encode_product(value: Any) -> Any:  # noqa: PLR0911
    """Serialize the closed set of normalized metadata used by local providers."""
    if is_dataclass(value):
        return {
            "type": type(value).__name__,
            "fields": {
                item.name: encode_product(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "value": value.value}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: encode_product(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_product(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return reject("NISAR provider metadata contains an unsupported value", TypeError)


def decode_product(value: Any) -> Any:
    """Decode only registered normalized metadata types, never arbitrary classes."""
    from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
    from faninsar.processing.geometry.coordinates import (
        ArrayDescriptor,
        ArrayRepresentation,
        RadarGrid,
    )
    from faninsar.processing.interferometry.products import (
        CalibrationState,
        CarrierState,
        CoregistrationState,
    )
    from faninsar.processing.slc.products import SLCProduct

    types = {
        item.__name__: item
        for item in (
            OrbitMetadata,
            OrbitStateVector,
            ArrayDescriptor,
            RadarGrid,
            SLCProduct,
            ArrayRepresentation,
            CalibrationState,
            CarrierState,
            CoregistrationState,
        )
    }
    if isinstance(value, (list, tuple)):
        return tuple(decode_product(item) for item in value)
    if not isinstance(value, dict):
        return value
    if "datetime" in value:
        return datetime.fromisoformat(value["datetime"])
    if "enum" in value:
        return types[value["enum"]](value["value"])
    if "type" in value:
        return types[value["type"]](
            **{key: decode_product(item) for key, item in value["fields"].items()}
        )
    return {key: decode_product(item) for key, item in value.items()}


class LazyHandles(dict):
    """Open an admitted local source only when a provider needs its reader."""

    def __init__(self, sensor: Any, admission: Any) -> None:
        """Retain runtime admission without opening a reader."""
        super().__init__()
        self.sensor = sensor
        self.admission = admission

    def __missing__(self, path: Path) -> Any:
        """Open a local admitted handle on first actual use."""
        handle = self.sensor.open_product(str(path), admission=self.admission)
        self[path] = handle
        return handle


def restore_provider(stack: Any) -> None:
    """Restore saved acquisition mapping and lazy handles without source I/O."""
    from faninsar.missions.nisar import NisarSensor
    from faninsar.missions.nisar.provider import make_nisar_scene_provider
    from faninsar.stack.provider import StackSceneProvider

    recipe = stack.config.extra.get("nisar_provider")
    if not isinstance(recipe, dict) or recipe.get("version") != 1:
        reject("NISAR identity lacks a supported local provider recipe", ValueError)
    sensor = NisarSensor()
    extra = stack.config.extra
    lineage = dict(extra["source_lineage"])
    products = {key: decode_product(item) for key, item in recipe["products"].items()}
    handles = LazyHandles(
        sensor, extra.get("nisar_admission", extra.get("admission_policy"))
    )
    # Defer provider stat checks until actual processing.
    callback = None

    def produce(*args: Any, **kwargs: Any) -> Any:
        nonlocal callback
        if callback is None:
            callback = make_nisar_scene_provider(
                sensor=sensor,
                handles=handles,
                products=products,
                lineage=lineage,
                reference=stack.reference,
                channel=tuple(recipe["channel"]),
                configured_window=extra.get("nisar_window", extra.get("rslc_window")),
                configured_tile_shape=extra.get("nisar_tile_shape"),
                configured_dem=stack.config.dem,
                configured_height=extra.get("height_m", extra.get("height")),
                admission_lineage=extra.get("source_admission", {}),
                flatten_stage=stack.config.flatten_stage,
            )
        return callback(*args, **kwargs)

    stack._nisar_sensor = sensor
    stack._nisar_handles = handles
    stack._nisar_lineage = lineage
    stack._nisar_admission = dict(extra.get("source_admission", {}))
    stack._nisar_channel = tuple(recipe["channel"])
    stack._nisar_products = products
    stack.scene_provider = StackSceneProvider(name="NISAR RSLC", produce_pair=produce)
