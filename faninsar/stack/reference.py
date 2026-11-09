"""Stack-owned radar mapping and inversion referencing (PROPOSAL-0085)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.data.reference import (
    ReferencePixelMap,
    applied_reference_record,
    mean_offsets,
    nearest_pixels,
    sample_at_pixels,
    select_valid_subsets,
    subtract_offsets,
)
from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from faninsar.data.query.points import Points
    from faninsar.io.storage.ifg_store import InterferogramArtifactStore
    from faninsar.network.layers import ProductLayer
    from faninsar.stack.session import Stack

logger = setup_logger(__name__)


def native_layer_origin(
    *,
    row_origin: int = 0,
    col_origin: int = 0,
    look_origin: tuple[int, int] = (0, 0),
    looks: tuple[int, int] = (1, 1),
) -> tuple[int, int]:
    """Return the native index of layer pixel ``(0, 0)``'s first sample.

    Matches :func:`faninsar.io.storage.ifg_store.ifg_sample_placement`:
    ``row_origin``/``col_origin`` are native crop offsets and
    ``look_origin`` is the look-grid origin on the parent looked grid.
    They are not two copies of the same origin.
    """
    return (
        int(row_origin) + int(look_origin[0]) * int(looks[0]),
        int(col_origin) + int(look_origin[1]) * int(looks[1]),
    )


def origin_from_store(store: InterferogramArtifactStore) -> tuple[int, int]:
    """Return the native radar origin recorded on an IFG product."""
    return native_layer_origin(
        row_origin=int(store.row_origin),
        col_origin=int(store.col_origin),
        look_origin=tuple(store.look_origin),
        looks=tuple(store.looks),
    )


def _placement_properties(layer: ProductLayer) -> dict[str, Any]:
    """Return IFG placement metadata, falling back to an upstream IFG product."""
    properties = dict(layer.source.properties)
    if "faninsar:row_origin" in properties or "faninsar:look_origin" in properties:
        return properties
    for parent in getattr(layer, "_upstream", ()):
        parent_props = getattr(parent, "properties", None)
        if isinstance(parent_props, dict) and (
            "faninsar:row_origin" in parent_props
            or "faninsar:look_origin" in parent_props
        ):
            return dict(parent_props)
    return properties


def origin_from_layer(layer: ProductLayer) -> tuple[int, int]:
    """Return the native radar origin from layer or parent IFG metadata."""
    properties = _placement_properties(layer)
    looks = tuple(layer.multilook)
    raw_looks = properties.get("faninsar:looks")
    if isinstance(raw_looks, (list, tuple)) and len(raw_looks) == 2:
        looks = (int(raw_looks[0]), int(raw_looks[1]))
    raw_look_origin = properties.get("faninsar:look_origin", (0, 0))
    look_origin = (int(raw_look_origin[0]), int(raw_look_origin[1]))
    return native_layer_origin(
        row_origin=int(properties.get("faninsar:row_origin", layer.origin[0])),
        col_origin=int(properties.get("faninsar:col_origin", layer.origin[1])),
        look_origin=look_origin,
        looks=looks,
    )


def map_radar_points(
    points: Points,
    *,
    geometry: object,
    dem: object | None,
    origin: tuple[int, int],
    looks: tuple[int, int],
    shape: tuple[int, int],
    device: str = "cpu",
) -> ReferencePixelMap:
    """Map geographic points into a cropped, multilooked radar layer.

    Heights come from the authoritative DEM sampler already prepared on
    the Stack. A missing DEM or a source recipe that would fetch is
    rejected; height 0 is never substituted. ``DEM.sample`` out-of-coverage
    values are NaN (or a raised coverage error) and become invalid points
    rather than a batch failure. Non-converged geo2rdr samples are
    invalid, not clipped to the array edge.

    Parameters
    ----------
    points : Points
        Normalized geographic points.
    geometry : RadarGeometryModel
        Reference radar geometry.
    dem : DEM or None
        Sampler exposing ``sample(latitude_deg, longitude_deg)``.
    origin, looks : tuple[int, int]
        Full-radar origin of the layer and ``(azimuth, range)`` looks.
    shape : tuple[int, int]
        Layer ``(height, width)``.
    device : str, optional
        geo2rdr execution device.

    Returns
    -------
    ReferencePixelMap
        Layer-frame centres and nearest pixels.

    Raises
    ------
    ValueError
        If the DEM is missing or cannot be sampled.

    """
    from faninsar.data.reference import coordinates_in_crs
    from faninsar.processing.geocoding.coordinates import RadarCoordinates
    from faninsar.processing.geometry.api import SourceDEM
    from faninsar.processing.geometry.prepare_production import run_geo2rdr

    if dem is None or isinstance(dem, SourceDEM) or not hasattr(dem, "sample"):
        message = (
            "DEM is not available for radar reference-point mapping; "
            "prepare a Stack DEM before sampling radar coordinates"
        )
        reject(message, ValueError, logger=logger)
    longitude, latitude = coordinates_in_crs(points, "EPSG:4326")
    n_points = int(longitude.size)
    height = np.full(n_points, np.nan, dtype=np.float64)
    azimuth = np.full(n_points, np.nan, dtype=np.float64)
    range_index = np.full(n_points, np.nan, dtype=np.float64)
    converged = np.zeros(n_points, dtype=bool)
    finite_ll = np.isfinite(longitude) & np.isfinite(latitude)
    if np.any(finite_ll):
        try:
            sampled = np.asarray(
                dem.sample(latitude[finite_ll], longitude[finite_ll]),
                dtype=np.float64,
            ).reshape(-1)
        except Exception as error:
            message = (
                "DEM could not be sampled at the requested reference points; "
                "out-of-coverage samples are invalid and this path does not "
                "fetch a DEM"
            )
            reject(message, ValueError, logger=logger, cause=error)
        if sampled.size != int(np.count_nonzero(finite_ll)):
            message = "DEM.sample must return one height per requested point"
            reject(message, ValueError, logger=logger)
        height[finite_ll] = sampled
    usable = finite_ll & np.isfinite(height)
    if np.any(usable):
        result = run_geo2rdr(
            geometry,
            latitude[usable],
            longitude[usable],
            height[usable],
            device=device,
            max_iter=40,
        )
        azimuth[usable] = np.asarray(result.azimuth_index, dtype=np.float64).reshape(-1)
        range_index[usable] = np.asarray(result.range_index, dtype=np.float64).reshape(
            -1
        )
        converged[usable] = np.asarray(result.converged, dtype=bool).reshape(-1)
    valid = usable & converged & np.isfinite(azimuth) & np.isfinite(range_index)
    coords = RadarCoordinates(
        azimuth.reshape(1, -1),
        range_index.reshape(1, -1),
        valid.reshape(1, -1),
    )
    layer = coords.to_layer_frame(origin=origin, looks=looks)
    row_center = np.asarray(layer.az, dtype=np.float64).reshape(-1)
    col_center = np.asarray(layer.rg, dtype=np.float64).reshape(-1)
    layer_valid = np.asarray(layer.valid, dtype=bool).reshape(-1)
    nearest_row, nearest_col, in_footprint = nearest_pixels(
        row_center, col_center, shape
    )
    return ReferencePixelMap(
        row_frac=row_center,
        col_frac=col_center,
        rows=nearest_row,
        cols=nearest_col,
        mapped=layer_valid & in_footprint,
    )


def map_points_for_store(
    stack: Stack,
    points: Points,
    store: InterferogramArtifactStore,
) -> ReferencePixelMap:
    """Map points onto the actual IFG/unwrap product grid."""
    from faninsar.data.reference import map_geographic_points

    if store.domain == "geo":
        if store.transform is None:
            message = "geographic reference-point mapping requires a layer transform"
            reject(message, ValueError, logger=logger)
        return map_geographic_points(
            points,
            transform=store.transform,
            shape=store.shape,
            crs=store.crs,
        )
    geometry, _grid_shape = stack._reference_radar_geometry()
    return map_radar_points(
        points,
        geometry=geometry,
        dem=stack._geometry_dem(),
        origin=origin_from_store(store),
        looks=tuple(store.looks),
        shape=store.shape,
        device=str(stack.config.device),
    )


def map_points_for_layer(
    session: object | None,
    points: Points,
    layer: ProductLayer,
) -> ReferencePixelMap:
    """Map points onto a product layer's scientific grid.

    Geographic layers with a CRS and affine do not need a session.
    Radar layers require a session that can run DEM sampling and geo2rdr.

    """
    from faninsar.data.reference import map_geographic_points

    properties = layer.source.properties
    domain = properties.get("faninsar:domain")
    if domain is None:
        domain = "geo" if layer.crs is not None else "radar"
    shape = layer.shape
    if domain == "geo":
        transform = properties.get("proj:transform")
        if transform is None:
            message = "geographic reference-point mapping requires a layer transform"
            reject(message, ValueError, logger=logger)
        return map_geographic_points(
            points,
            transform=transform,
            shape=shape,
            crs=layer.crs or properties.get("proj:code"),
        )
    mapper = getattr(session, "map_reference_points_to_radar", None)
    if session is None or mapper is None:
        message = (
            "radar reference-point mapping requires geometry context; "
            "use a Stack-bound layer or pass geographic points on a "
            "geographic layer"
        )
        reject(message, ValueError, logger=logger)
    looks = tuple(layer.multilook)
    if "faninsar:looks" in properties:
        raw_looks = properties["faninsar:looks"]
        looks = (int(raw_looks[0]), int(raw_looks[1]))
    return mapper(
        points,
        origin=origin_from_layer(layer),
        looks=looks,
        shape=shape,
    )


def reference_phase_arrays(
    phases: Mapping[str, np.ndarray],
    *,
    mapping: ReferencePixelMap,
    support_masks: Mapping[str, np.ndarray | None],
    points: Points,
    policy: str,
    grid: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Subtract per-image reference means from copies of ``phases``.

    Original arrays are not modified. Connected-component labels are not
    used as filters; the arithmetic mean of valid samples is subtracted.

    """
    samples: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    for key, array in phases.items():
        sampled = sample_at_pixels(array, mapping, support_masks.get(key))
        samples[key] = sampled
        valid[key] = np.isfinite(sampled)
    selected = select_valid_subsets(valid, policy)
    offsets = mean_offsets(samples, selected)
    referenced = subtract_offsets(phases, offsets)
    record = applied_reference_record(
        points=points,
        policy=policy,
        mapping=mapping,
        selected_by_image=selected,
        offsets=offsets,
        grid=grid,
    )
    return referenced, record


def support_mask_from_product(product: object) -> np.ndarray | None:
    """Read a boolean support mask from an unwrap product when present."""
    assets = getattr(product, "assets", None)
    if not isinstance(assets, dict) or "valid_mask" not in assets:
        return None
    read = getattr(product, "read", None)
    if read is None:
        return None
    mask = np.asarray(read("valid_mask"))
    if mask.dtype != np.bool_:
        mask = np.asarray(mask, dtype=bool)
    return mask


def grid_payload_from_store(store: InterferogramArtifactStore) -> dict[str, Any]:
    """Return the grid context stored with applied reference provenance."""
    transform = store.transform
    return {
        "domain": store.domain,
        "shape": [int(store.shape[0]), int(store.shape[1])],
        "looks": [int(store.looks[0]), int(store.looks[1])],
        "origin": list(origin_from_store(store)),
        "crs": store.crs,
        "transform": None if transform is None else [float(v) for v in transform],
        "grid_identity": store.grid_identity,
    }
