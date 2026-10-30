"""Reference-bound geometry raster assets on a scene product (PROPOSAL-0066).

The six radar-domain geometry fields solved in the same rdr2geo pass —
``latitude``, ``longitude``, ``height``, ``incidence``, ``azimuth``, and the
``geom_flags`` layover/shadow bitmask — are persisted as committed assets on
the Reference scene product (``coreg/<reference>/``). They share the scene's
``write_id`` and ``grid_identity`` because they are a property of the
Reference frame geometry already recorded on that item.

Assets are materialized lazily: the first ``SceneLayers.<name>`` read runs
the rdr2geo solve once over the full Reference grid and commits all six
payloads in one item.json rewrite. Reopen then reads them directly through
the usual ``Product.read``/``verify_product`` machinery, so
``Stack.status()``/``verify()`` pick them up with no extra wiring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from faninsar.io.storage._zarr_stac import (
    digest_path,
    read_zarr_array,
    write_zarr_array,
)
from faninsar.io.storage.product import ITEM_NAME, open_product
from faninsar.io.storage.scene_store import (
    SCENE_KIND,
    CoregisteredSceneStore,
)
from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)

#: Ordered committed geometry asset names on the Reference scene item.
SCENE_GEOMETRY_ASSETS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "height",
    "incidence",
    "azimuth",
    "geom_flags",
)

#: Virtual LOS layers derived at read time from ``incidence``/``azimuth``;
#: they carry no asset and persist nothing.
VIRTUAL_LOS_LAYERS: tuple[str, ...] = ("los_east", "los_north", "los_up")


def _committed_assets(store: CoregisteredSceneStore) -> frozenset[str]:
    """Return the asset names committed on this scene product's item."""
    return frozenset(open_product(store.root, kind=SCENE_KIND).assets)


def missing_geometry_assets(store: CoregisteredSceneStore) -> tuple[str, ...]:
    """Return the committed geometry assets absent from this scene product."""
    committed = _committed_assets(store)
    return tuple(name for name in SCENE_GEOMETRY_ASSETS if name not in committed)


def _geometry_field_map(store: CoregisteredSceneStore) -> dict[str, np.ndarray]:
    """Solve the six geometry fields on the Reference grid via rdr2geo.

    Reuses the persisted ``faninsar:radar_geometry`` record and the same
    ``run_rdr2geo_chunked`` production machinery used by geocoding, so the
    committed rasters are byte-for-byte the solver's own outputs rather than
    a derived recompute.
    """
    from faninsar.processing.geometry.prepare_production import (
        run_rdr2geo_chunked,
    )
    from faninsar.processing.geometry.record import radar_geometry_from_record

    record = store.radar_geometry
    if record is None:
        reject_invalid_state(
            "Reference scene has no radar_geometry record; run "
            "coregister_scenes with coregistration_grid='radar' first"
        )
    model = radar_geometry_from_record(record)
    rows, cols = store.grid_shape
    az, rg = np.meshgrid(
        np.arange(rows, dtype=np.float64),
        np.arange(cols, dtype=np.float64),
        indexing="ij",
    )
    result = run_rdr2geo_chunked(
        model,
        az,
        rg,
        _geometry_dem_for(store),
        device=_geometry_device(store),
    )
    return {
        "latitude": np.asarray(result.latitude_deg, dtype=np.float64),
        "longitude": np.asarray(result.longitude_deg, dtype=np.float64),
        "height": np.asarray(result.height_m, dtype=np.float64),
        "incidence": np.asarray(result.incidence_deg, dtype=np.float64),
        "azimuth": np.asarray(result.look_azimuth_deg, dtype=np.float64),
        "geom_flags": np.asarray(result.geom_flags, dtype=np.uint8),
    }


def _geometry_dem_for(store: CoregisteredSceneStore) -> object:
    """Return the Stack session's radar-geometry DEM, if a session is bound."""
    session = store._session
    dem = getattr(session, "_geometry_dem", None)
    return dem() if callable(dem) else None


def _geometry_device(store: CoregisteredSceneStore) -> object:
    """Return the Stack session's configured device, defaulting to ``cpu``."""
    session = store._session
    config = getattr(session, "config", None)
    return getattr(config, "device", "cpu") if config is not None else "cpu"


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync so a rename survives a crash."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(descriptor)


def commit_scene_geometry(
    store: CoregisteredSceneStore,
) -> CoregisteredSceneStore:
    """Materialize the six geometry assets on the Reference scene product.

    The rdr2geo solve runs once over the full Reference grid; the six zarr
    payloads are written into the product directory and ``item.json`` is
    rewritten atomically, preserving the scene ``write_id`` and properties
    while adding the new asset entries (with ``file:checksum`` and
    ``roles: ["data"]``). Keeping the ``write_id`` intact is required by the
    flat layout: secondaries bound to this Reference must not go stale when
    its geometry fields are materialized.

    Parameters
    ----------
    store : CoregisteredSceneStore
        The Reference scene product. Must be the Stack Reference
        (``is_reference``), radar domain, and bound to a session that can
        supply the radar-geometry record.

    Returns
    -------
    CoregisteredSceneStore
        A reopened store whose item exposes the geometry assets.

    Raises
    ------
    InvalidProcessingStateError
        If the store is not the radar Reference, already carries the assets,
        or lacks the radar-geometry record.

    """
    if not store.is_reference:
        reject_invalid_state(
            "geometry layers are bound to the Stack Reference; "
            f"scene {store.date_id!r} is a secondary"
        )
    if store.domain != "radar":
        reject_invalid_state("geometry layers require a radar-domain Reference scene")
    committed = _committed_assets(store)
    missing = [name for name in SCENE_GEOMETRY_ASSETS if name not in committed]
    if not missing:
        return store
    fields = _geometry_field_map(store)

    root = Path(store.root)
    # Write the payloads first; item.json stays the commit signal, so a crash
    # before the rewrite leaves orphaned (unreferenced) zarr dirs, never a
    # product that names missing assets.
    for name in missing:
        write_zarr_array(root / f"{name}.zarr", fields[name])

    item_path = root / ITEM_NAME
    item = json.loads(item_path.read_text(encoding="utf-8"))
    assets = dict(item.get("assets", {}))
    for name in missing:
        href = f"{name}.zarr"
        assets[name] = {
            "href": href,
            "roles": ["data"],
            "file:checksum": digest_path(root / href),
        }
    item["assets"] = assets
    text = json.dumps(item, sort_keys=True, separators=(",", ":"))
    temp_path = root / f".{ITEM_NAME}.tmp"
    with temp_path.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temp_path.replace(item_path)
    _fsync_directory(root)
    logger.info(
        "Committed %d geometry assets on Reference scene %s",
        len(missing),
        store.date_id,
    )
    return CoregisteredSceneStore.open(root)


def read_scene_geometry(store: CoregisteredSceneStore, name: str) -> np.ndarray:
    """Read one committed geometry asset, materializing it lazily if absent.

    On the first access the rdr2geo solve runs and all six payloads are
    committed; subsequent reads hit the persisted zarr directly.

    """
    if name not in SCENE_GEOMETRY_ASSETS:
        reject_invalid_state(f"unknown scene geometry layer {name!r}")
    if name not in _committed_assets(store):
        store = commit_scene_geometry(store)
    return read_zarr_array(store.root / f"{name}.zarr")


def los_unit_vector(store: CoregisteredSceneStore, name: str) -> np.ndarray:
    """Derive one LOS unit-vector component from stored incidence/azimuth.

    The convention matches
    ``NetworkGeometry.compute_los_unit_vectors``: the LOS unit vector points
    from ground to satellite, with ``incidence`` measured from the local
    ellipsoid normal and ``azimuth`` the clockwise-from-north look azimuth.

    """
    if name not in VIRTUAL_LOS_LAYERS:
        reject_invalid_state(f"unknown LOS layer {name!r}")
    incidence = np.deg2rad(read_scene_geometry(store, "incidence"))
    azimuth = np.deg2rad(read_scene_geometry(store, "azimuth"))
    if name == "los_east":
        return -np.sin(incidence) * np.sin(azimuth)
    if name == "los_north":
        return -np.sin(incidence) * np.cos(azimuth)
    return np.cos(incidence)


__all__ = [
    "SCENE_GEOMETRY_ASSETS",
    "VIRTUAL_LOS_LAYERS",
    "commit_scene_geometry",
    "los_unit_vector",
    "missing_geometry_assets",
    "read_scene_geometry",
]
