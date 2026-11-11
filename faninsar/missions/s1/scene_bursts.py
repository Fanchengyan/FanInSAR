"""Sentinel-1 source-coverage footprints for scene burst inspection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from shapely import polygons as shapely_polygons

from faninsar.logging import setup_logger
from faninsar.missions.s1.errors import Sentinel1ProductError
from faninsar.missions.s1.processing import _common_burst_pairs, _radar_model
from faninsar.missions.s1.safe import open_acquisition_products
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry.prepare_production import run_rdr2geo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapely.geometry import Polygon

    from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath
    from faninsar.network.readers.scene_bursts import SceneBurstObservation
    from faninsar.processing.geometry.transforms import RadarGeometryModel
    from faninsar.stack.session import Stack

logger = setup_logger(__name__)

_FOOTPRINT_COLUMNS = ("date", "track", "burst", "absolute_burst_id", "geometry")


def actual_source_polygon(burst: S1Burst) -> Polygon | None:
    """Return a reliable local actual burst polygon, or ``None``.

    ``S1Burst.footprint`` is the annotation min/max bbox
    (``annotation.py`` ``_parse_burst_footprints``) and is never accepted.
    Current S1 types do not persist a distinct actual per-burst polygon, so
    this hook is explicitly ``None`` until such a field exists.

    Parameters
    ----------
    burst : S1Burst
        Source burst whose optional actual polygon would be used.

    Returns
    -------
    shapely.geometry.Polygon or None
        Always ``None`` for the current S1 product types.

    """
    del burst
    return None


def official_absolute_burst_id(burst: object | None) -> int | None:
    """Return a source burst's official ESA identifier, if present.

    Parameters
    ----------
    burst : object or None
        Burst or payload exposing ``absolute_burst_id``.

    Returns
    -------
    int or None
        The official identifier, never an ANX-derived internal key.

    """
    if burst is None:
        return None
    value = getattr(burst, "absolute_burst_id", None)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def qualified_absolute_burst_id(
    source_id: int | None,
    other_id: int | None,
    *,
    tag: str,
) -> int | None:
    """Return the source official ID, failing closed on a qualified conflict.

    Compare only two qualified claims for the same source observation and
    identifier namespace. Cross-date Reference IDs are not counterpart claims
    for this purpose and must not be substituted for the source ID.

    Parameters
    ----------
    source_id : int or None
        Official ID of the burst being published.
    other_id : int or None
        Another qualified ID claim for the same source and namespace.
    tag : str
        Scene unit tag used in the rejection message.

    Returns
    -------
    int or None
        ``source_id`` when it is present and not in conflict.

    Raises
    ------
    InvalidProcessingStateError
        If both IDs are present and disagree.

    """
    source = None if source_id is None else int(source_id)
    other = None if other_id is None else int(other_id)
    if source is not None and other is not None and source != other:
        reject_invalid_state(
            f"{tag}: conflicting official absolute burst IDs {source} vs {other}"
        )
    return source


def _empty_footprints() -> Any:
    """Return an empty EPSG:4326 footprints table with the inspection schema."""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {
            "date": pd.Series(dtype="string"),
            "track": pd.Series(dtype="string"),
            "burst": pd.Series(dtype="string"),
            "absolute_burst_id": pd.Series(dtype="Int64"),
            "geometry": gpd.GeoSeries(dtype="geometry"),
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def parse_scene_unit_tag(tag: str) -> tuple[int, str, int] | None:
    """Parse ``f{frame}_{swath}_b{burst}`` or ``{swath}_b{burst}`` tags.

    Parameters
    ----------
    tag : str
        Product-local scene unit tag.

    Returns
    -------
    tuple of int, str, int or None
        ``(frame_index, swath, burst_index)`` when the tag matches the
        production spelling; otherwise ``None``.

    """
    parts = tag.split("_")
    try:
        if len(parts) == 3 and parts[0].startswith("f") and parts[2].startswith("b"):
            return int(parts[0][1:]), parts[1], int(parts[2][1:])
        if len(parts) == 2 and parts[1].startswith("b"):
            return 0, parts[0], int(parts[1][1:])
    except ValueError:
        return None
    return None


def _footprints_frame(
    dates: Sequence[str],
    tracks: Sequence[str | None],
    bursts: Sequence[str],
    absolute_ids: Sequence[int | None],
    geometries: Sequence[Polygon | None],
) -> Any:
    """Build a caller-owned EPSG:4326 GeoDataFrame of burst observations."""
    import geopandas as gpd

    frame = gpd.GeoDataFrame(
        {
            "date": pd.Series(list(dates), dtype="string"),
            "track": pd.Series(list(tracks), dtype="string"),
            "burst": pd.Series(list(bursts), dtype="string"),
            "absolute_burst_id": pd.array(list(absolute_ids), dtype="Int64"),
            "geometry": list(geometries),
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    return frame.loc[:, list(_FOOTPRINT_COLUMNS)]


def _products_for_date(
    stack: Stack,
    date_id: str,
    cache: dict[str, list[S1Product] | BaseException],
) -> list[S1Product] | None:
    """Open one date through ``open_acquisition_products``, caching the merged list."""
    cached = cache.get(date_id)
    if isinstance(cached, BaseException):
        return None
    if cached is not None:
        return cached
    catalog = getattr(stack, "catalog", None)
    if catalog is None:
        logger.warning("burst footprints require a Stack catalog")
        cache[date_id] = RuntimeError("missing catalog")
        return None
    try:
        paths = catalog.paths_for(date_id)
    except Exception as exc:
        logger.warning("burst footprints: no local source for %s (%s)", date_id, exc)
        cache[date_id] = exc
        return None
    try:
        products = open_acquisition_products(list(paths))
    except (OSError, Sentinel1ProductError, ValueError, TypeError) as exc:
        logger.warning(
            "cannot open merged acquisition products for %s: %s", date_id, exc
        )
        cache[date_id] = exc
        return None
    cache[date_id] = products
    return products


def _find_swath(
    products: Sequence[S1Product],
    swath_name: str,
    *,
    preferred_index: int,
) -> tuple[S1Product, S1Swath] | None:
    """Find ``swath_name`` across merged products, preferring the frame slot."""
    ordered = list(products)
    if 0 <= preferred_index < len(ordered):
        ordered.insert(0, ordered.pop(preferred_index))
    for product in ordered:
        swath = product.find_swath(swath_name)
        if swath is not None:
            return product, swath
    return None


def _match_secondary_burst(
    reference_swath: S1Swath,
    secondary_swath: S1Swath,
    reference_burst_index: int,
    *,
    absolute_burst_id: int | None,
) -> S1Burst | None:
    """Bind a secondary burst through qualified ANX matching."""
    _ = absolute_burst_id
    pairs = _common_burst_pairs(reference_swath, secondary_swath)
    matched = pairs.get(int(reference_burst_index))
    if matched is None:
        return None
    if matched < 0 or matched >= len(secondary_swath.bursts):
        return None
    return secondary_swath.bursts[matched]


def _model_for_swath(swath: S1Swath) -> RadarGeometryModel | None:
    """Return the production radar model for a swath, origin at the first burst."""
    if not swath.bursts:
        return None
    orbit = getattr(swath, "orbit", None)
    vectors = getattr(orbit, "vectors", None)
    if not vectors:
        return None
    burst = swath.bursts[0]
    try:
        return _radar_model(
            swath,
            burst,
            shape=(int(burst.lines), int(burst.samples)),
            row0=int(burst.index) * int(swath.lines_per_burst),
            col0=0,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("cannot build radar model for swath %s: %s", swath.swath, exc)
        return None


def _group_key(model: RadarGeometryModel, swath: S1Swath) -> tuple[object, ...]:
    """Identity of a compatible geometry/orbit/DEM=None group."""
    orbit = swath.orbit
    vectors = getattr(orbit, "vectors", None) or ()
    first_time = getattr(vectors[0], "time", None) if vectors else None
    return (
        swath.swath,
        swath.polarization,
        model.sensing_start,
        round(float(model.starting_slant_range_m), 6),
        round(float(model.range_spacing_m), 9),
        round(float(model.azimuth_time_interval_s), 12),
        round(float(model.wavelength_m), 12),
        model.look_direction,
        getattr(orbit, "source", None),
        first_time,
    )


def _burst_radar_corners(
    model: RadarGeometryModel,
    swath: S1Swath,
    burst: S1Burst,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return four ordered radar corners using ``burst.azimuth_time`` as line 0."""
    start = model.sensing_start
    azimuth_time = burst.azimuth_time
    try:
        offset_s = (azimuth_time - start).total_seconds()
    except TypeError:
        logger.warning(
            "burst azimuth_time %s is not compatible with model sensing_start %s",
            azimuth_time,
            start,
        )
        return None
    dt = float(swath.azimuth_time_interval_s)
    if not np.isfinite(dt) or dt == 0.0:
        return None
    line0 = float(offset_s) / dt
    line1 = line0 + int(burst.lines) - 1
    sample1 = int(burst.samples) - 1
    azimuth = np.array([line0, line0, line1, line1], dtype=np.float64)
    slant = np.array([0.0, sample1, sample1, 0.0], dtype=np.float64)
    return azimuth, slant


def _publishable_polygon(geom: Polygon | None) -> Polygon | None:
    """Reject empty, invalid, self-intersecting, or dateline-wrapping rings."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type != "Polygon" or not geom.is_valid or not geom.is_simple:
        return None
    xs = np.asarray(geom.exterior.coords, dtype=np.float64)[:, 0]
    if xs.size == 0 or not np.isfinite(xs).all():
        return None
    if float(np.max(xs) - np.min(xs)) > 180.0:
        return None
    return geom


@dataclass(slots=True)
class _BoundSolve:
    """One observation ready for a four-corner rdr2geo group."""

    index: int
    model: RadarGeometryModel
    group_key: tuple[object, ...]
    azimuth: np.ndarray
    range: np.ndarray


def _bind_observation(  # noqa: PLR0911
    stack: Stack,
    observation: SceneBurstObservation,
    *,
    products: dict[str, list[S1Product] | BaseException],
) -> tuple[S1Swath, S1Burst] | None:
    """Match one observation to a local source burst, or return None."""
    parsed = parse_scene_unit_tag(observation.burst)
    if parsed is None:
        logger.warning(
            "burst %s %s: local tag is not a scene unit spelling",
            observation.date,
            observation.burst,
        )
        return None
    frame_index, swath_name, burst_index = parsed
    date_products = _products_for_date(stack, observation.date, products)
    if date_products is None:
        return None
    found = _find_swath(date_products, swath_name, preferred_index=frame_index)
    if found is None:
        logger.warning(
            "burst %s %s: swath %s not in merged products",
            observation.date,
            observation.burst,
            swath_name,
        )
        return None
    _product, swath = found
    reference_id = getattr(stack, "reference", None)
    if reference_id is None or observation.date == reference_id:
        if observation.absolute_burst_id is not None:
            for burst in swath.bursts:
                if burst.absolute_burst_id == observation.absolute_burst_id:
                    return swath.source_for_burst(burst.index), burst
        if burst_index < 0 or burst_index >= len(swath.bursts):
            logger.warning(
                "burst %s %s: burst index %s out of range",
                observation.date,
                observation.burst,
                burst_index,
            )
            return None
        return swath.source_for_burst(burst_index), swath.bursts[burst_index]
    reference_products = _products_for_date(stack, str(reference_id), products)
    if reference_products is None:
        logger.warning(
            "burst %s %s: cannot open Reference source for qualified match",
            observation.date,
            observation.burst,
        )
        return None
    reference_found = _find_swath(
        reference_products, swath_name, preferred_index=frame_index
    )
    if reference_found is None:
        logger.warning(
            "burst %s %s: Reference swath %s missing",
            observation.date,
            observation.burst,
            swath_name,
        )
        return None
    reference_product, reference_swath = reference_found
    from faninsar.missions.s1.track_map import track_key_for_product

    reference_track = track_key_for_product(reference_product)
    if reference_track is None:
        logger.warning(
            "burst %s %s: Reference track is unknown",
            observation.date,
            observation.burst,
        )
        return None
    matches = []
    for candidate_product in date_products:
        if track_key_for_product(candidate_product) != reference_track:
            logger.warning(
                "burst %s %s: secondary track differs or is unknown",
                observation.date,
                observation.burst,
            )
            return None
        candidate = candidate_product.find_swath(
            swath_name, reference_swath.polarization
        )
        if candidate is None:
            continue
        candidate_burst = _match_secondary_burst(
            reference_swath,
            candidate,
            burst_index,
            absolute_burst_id=observation.absolute_burst_id,
        )
        if candidate_burst is not None:
            matches.append((candidate, candidate_burst))
    if len(matches) != 1:
        logger.warning(
            "burst %s %s: secondary source match count is %d",
            observation.date,
            observation.burst,
            len(matches),
        )
        return None
    swath, burst = matches[0]
    return swath.source_for_burst(burst.index), burst


def _polygons_from_groups(
    solves: Sequence[_BoundSolve], n_obs: int
) -> list[Polygon | None]:
    """Batch compatible models once and build per-observation polygons."""
    out: list[Polygon | None] = [None] * n_obs
    groups: dict[tuple[object, ...], list[_BoundSolve]] = defaultdict(list)
    for item in solves:
        groups[item.group_key].append(item)
    for items in groups.values():
        model = items[0].model
        azimuth = np.concatenate([item.azimuth for item in items])
        slant = np.concatenate([item.range for item in items])
        try:
            result = run_rdr2geo(model, azimuth, slant, None, device="cpu")
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("four-corner rdr2geo failed for a footprint group: %s", exc)
            continue
        lon = np.asarray(result.longitude_deg, dtype=np.float64).reshape(len(items), 4)
        lat = np.asarray(result.latitude_deg, dtype=np.float64).reshape(len(items), 4)
        coords = np.stack([lon, lat], axis=-1)
        finite = np.isfinite(coords).all(axis=(1, 2))
        span = np.nanmax(lon, axis=1) - np.nanmin(lon, axis=1)
        wrapped = np.isfinite(span) & (span > 180.0)
        raw_converged = getattr(result, "converged", None)
        if raw_converged is None:
            converged = np.zeros((len(items), 4), dtype=bool)
        else:
            converged = np.asarray(raw_converged, dtype=bool).reshape(len(items), 4)
        ok = finite & converged.all(axis=1) & ~wrapped
        ok_idx = np.nonzero(ok)[0]
        if ok_idx.size:
            closed = np.concatenate(
                [coords[ok_idx], coords[ok_idx][:, :1, :]],
                axis=1,
            )
            built = shapely_polygons(closed)
            geoms = np.asarray(built, dtype=object).ravel().tolist()
            for item, geom in zip(
                (items[int(i)] for i in ok_idx),
                geoms,
                strict=True,
            ):
                published = _publishable_polygon(geom)
                if published is None:
                    logger.warning(
                        "burst observation %s produced an invalid footprint polygon",
                        item.index,
                    )
                    continue
                out[item.index] = published
        for item in items:
            if out[item.index] is not None:
                continue
            logger.warning(
                "burst observation %s needs four finite converged corners; "
                "partial or wrapping solves are omitted",
                item.index,
            )
    return out


def resolve_s1_scene_burst_footprints(
    stack: Stack,
    observations: Sequence[SceneBurstObservation],
) -> Any:
    """Resolve per-observation source coverage as an EPSG:4326 GeoDataFrame.

    Missing local annotation or orbit yields a null geometry and a diagnostic.
    Annotation min/max bboxes are never used. Compatible models are batched
    through :func:`run_rdr2geo` with ``dem=None``; four finite *converged*
    corners are required.

    Parameters
    ----------
    stack : Stack
        Parent Stack whose catalog supplies local SAFE paths.
    observations : sequence of SceneBurstObservation
        Selected per-acquisition handles.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per observation, CRS EPSG:4326.

    """
    if not observations:
        return _empty_footprints()
    products: dict[str, list[S1Product] | BaseException] = {}
    geometries: list[Polygon | None] = [None] * len(observations)
    solves: list[_BoundSolve] = []
    models: dict[tuple[object, ...], RadarGeometryModel] = {}
    for index, observation in enumerate(observations):
        bound = _bind_observation(stack, observation, products=products)
        if bound is None:
            continue
        swath, burst = bound
        actual = _publishable_polygon(actual_source_polygon(burst))
        if actual is not None:
            geometries[index] = actual
            continue
        model_token = (swath.annotation_path, swath.swath, swath.polarization)
        model = models.get(model_token)
        if model is None:
            model = _model_for_swath(swath)
            if model is None:
                logger.warning(
                    "burst %s %s: missing orbit for four-corner solve",
                    observation.date,
                    observation.burst,
                )
                continue
            models[model_token] = model
        corners = _burst_radar_corners(model, swath, burst)
        if corners is None:
            continue
        azimuth, slant = corners
        solves.append(
            _BoundSolve(
                index=index,
                model=model,
                group_key=_group_key(model, swath),
                azimuth=azimuth,
                range=slant,
            )
        )
    if solves:
        resolved = _polygons_from_groups(solves, len(observations))
        for index, geom in enumerate(resolved):
            if geom is not None:
                geometries[index] = geom
    return _footprints_frame(
        [item.date for item in observations],
        [item.track for item in observations],
        [item.burst for item in observations],
        [item.absolute_burst_id for item in observations],
        geometries,
    )
