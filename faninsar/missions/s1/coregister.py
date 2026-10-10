"""Apply geometry plus inverted date constants to a Sentinel-1 secondary scene."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.io.storage.scene_store import scene_grid_identity, write_scene_unit
from faninsar.logging import setup_logger
from faninsar.missions.s1.measure import _resolve_common_bursts, _resolve_swaths
from faninsar.missions.s1.processing import (
    BurstSelection,
    ProductionPairState,
    _as_frame_paths,
    _common_burst_indices,
    _phase_state_manifest,
    _roi_burst_window,
    _scene_id,
    _select_bursts_by_roi,
    _swath_range_offsets,
    load_production_scene,
    stage_coregister,
    stage_deramp,
)
from faninsar.missions.s1.safe import open_safe_product
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.data.query import BoundingBox, Polygons
    from faninsar.missions.s1.types import S1Product
    from faninsar.processing.geometry import DEM
    from faninsar.processing.mosaicking.grid import GeoGridSpec

logger = setup_logger(__name__)

__all__ = ["coregister_s1_scene"]


def coregister_s1_scene(
    reference_path: Path | Sequence[Path],
    secondary_path: Path | Sequence[Path],
    *,
    output_scenes: Path,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    roi: BoundingBox | Polygons | None = None,
    dem: DEM,
    misreg_az_px: float,
    misreg_rg_px: float,
    coregistration_grid: Literal["radar", "geo"] = "radar",
    geo_grid: GeoGridSpec | None = None,
    primary_orbit_path: Path | None = None,
    secondary_orbit_path: Path | None = None,
    device: str = "auto",
) -> Path:
    """Resample the secondary onto the reference grid with geometry + date constants.

    Does not measure ESD or Ampcor. ``roi`` crops the apply window; the
    constants still come from full-overlap NESD. Writes a
    :class:`~faninsar.io.storage.scene_store.CoregisteredSceneStore`.
    Range-offset flattening remains the coregistration-stage behavior used
    by the shipping apply path.

    Parameters
    ----------
    reference_path, secondary_path : path or sequence of path
        One or more SAFE products per date.
    output_scenes : path
        Directory of the CoregisteredSceneStore to write.
    swaths : tuple of str, optional
        Sub-swaths to apply. ``None`` selects every sub-swath of the
        reference product. Ignored when ``roi`` is set.
    bursts : BurstSelection, optional
        Per-swath burst selection. ``None`` selects every burst. Ignored
        when ``roi`` is set.
    roi : BoundingBox or Polygons, optional
        Apply-window crop. Burst selection follows footprint intersection
        (including neighboring bursts at edges) when set.
    dem : DEM
        Height sampler for geometry offsets.
    misreg_az_px, misreg_rg_px : float
        Inverted per-date constants (or 0 for geometry-only). Must be finite.
    coregistration_grid : {"radar", "geo"}, optional
        Grid on which aligned SLCs are produced. Default ``radar``.
    geo_grid : GeoGridSpec, optional
        Required when ``coregistration_grid="geo"``.
    primary_orbit_path, secondary_orbit_path : path, optional
        Precise ESA EOF orbits.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Numerical device.

    Returns
    -------
    pathlib.Path
        ``output_scenes`` after the store has been written.

    Raises
    ------
    InvalidProcessingStateError
        If ``misreg_az_px`` or ``misreg_rg_px`` is non-finite, or the
        selection/grid configuration is invalid.

    """
    if not np.isfinite(misreg_az_px) or not np.isfinite(misreg_rg_px):
        reject_invalid_state("misreg_az_px and misreg_rg_px must be finite")
    if coregistration_grid not in {"radar", "geo"}:
        reject_invalid_state("coregistration_grid must be 'radar' or 'geo'")
    if coregistration_grid == "geo" and geo_grid is None:
        reject_invalid_state("geo coregistration requires geo_grid")

    output_root = Path(output_scenes)
    output_root.mkdir(parents=True, exist_ok=True)
    primary_paths = _as_frame_paths(reference_path, "reference_path")
    secondary_paths = _as_frame_paths(secondary_path, "secondary_path")
    if len(primary_paths) != len(secondary_paths):
        reject_invalid_state(
            "reference/secondary frame counts differ: "
            f"{len(primary_paths)} vs {len(secondary_paths)}"
        )
    frame_count = len(primary_paths)
    ref_orbits = [primary_orbit_path] * frame_count
    sec_orbits = [secondary_orbit_path] * frame_count
    primary_products = [open_safe_product(path) for path in primary_paths]
    secondary_products = [open_safe_product(path) for path in secondary_paths]
    swath_tuple, resolved = _resolve_apply_selection(
        primary_products,
        secondary_products,
        primary_paths=primary_paths,
        swaths=swaths,
        bursts=bursts,
        roi=roi,
        ref_orbits=ref_orbits,
        dem=dem,
    )
    range_offsets = _swath_range_offsets(swath_tuple, primary_products)
    units_by_swath, frame_rows, frame_cols = _unit_placement(
        primary_products,
        resolved,
        swath_tuple,
        range_offsets,
    )
    reference_id = _scene_id(primary_paths[0])
    date_id = _scene_id(secondary_paths[0])
    geo_work_root = output_root / ".geo_work" if coregistration_grid == "geo" else None
    if geo_work_root is not None:
        geo_work_root.mkdir(parents=True, exist_ok=True)

    wrote = 0
    for swath in swath_tuple:
        for frame_index, burst_index, azimuth_offset in units_by_swath[swath]:
            tag = f"f{frame_index}_{swath}_b{burst_index}"
            ref = load_production_scene(
                primary_paths[frame_index],
                swath=swath,
                burst_index=burst_index,
                dem=dem,
                orbit_path=ref_orbits[frame_index],
                coregistration_grid=coregistration_grid,
                full_range=True,
            )
            sec = load_production_scene(
                secondary_paths[frame_index],
                swath=swath,
                burst_index=burst_index,
                dem=dem,
                orbit_path=sec_orbits[frame_index],
                coregistration_grid=coregistration_grid,
                full_range=True,
            )
            state = ProductionPairState(
                pair_id=f"{ref.scene_id}_{sec.scene_id}_{tag}",
                primary=ref,
                secondary=sec,
                dem=dem,
                coregistration_grid=coregistration_grid,
            )
            state = stage_deramp(state, device=device)
            roi_window = None
            if coregistration_grid == "radar" and roi is not None:
                if state.primary_deramped is None:
                    reject_invalid_state(f"{tag}: deramp produced no primary samples")
                roi_window = _roi_burst_window(
                    roi,
                    ref.geometry,
                    dem,
                    state.primary_deramped.shape,
                    device=device,
                )
            burst_work_dir = None if geo_work_root is None else geo_work_root / tag
            state = stage_coregister(
                state,
                esd_enabled=False,
                amplitude_refinement_enabled=False,
                misreg_az_px=float(misreg_az_px),
                misreg_rg_px=float(misreg_rg_px),
                device=device,
                coregistration_grid=coregistration_grid,
                geo_grid=geo_grid,
                geo_work_dir=burst_work_dir,
                roi=roi,
                roi_window=roi_window,
            )
            if state.primary_deramped is None or state.secondary_aligned is None:
                reject_invalid_state(f"{tag}: coregistration produced no aligned SLCs")
            if coregistration_grid == "geo" and (
                np.asarray(state.primary_deramped).size == 0
                or np.asarray(state.secondary_aligned).size == 0
            ):
                logger.warning("%s: empty geocoded footprint; skipped", tag)
                continue
            burst_row0 = 0
            burst_col0 = 0
            if coregistration_grid == "geo":
                row_origin = state.geo_bbox[0] if state.geo_bbox is not None else 0
                col_origin = state.geo_bbox[2] if state.geo_bbox is not None else 0
                grid_shape = geo_grid.shape if geo_grid is not None else None
                grid_identity = (
                    scene_grid_identity(
                        "geo",
                        geo_grid.shape,
                        {
                            "crs": geo_grid.crs,
                            "transform": list(geo_grid.transform),
                            "resolution_m": list(geo_grid.resolution_m),
                        },
                    )
                    if geo_grid is not None
                    else None
                )
                crs = geo_grid.crs if geo_grid is not None else None
                transform = (
                    tuple(float(value) for value in tuple(geo_grid.transform)[:6])
                    if geo_grid is not None
                    else None
                )
            else:
                if state.radar_roi_origin is not None:
                    burst_row0, burst_col0 = state.radar_roi_origin
                row_origin = azimuth_offset + burst_row0
                col_origin = range_offsets[swath] + burst_col0
                grid_shape = (frame_rows, frame_cols)
                grid_identity = None
                crs = None
                transform = None
            write_scene_unit(
                output_root,
                date_id=date_id,
                reference_id=reference_id,
                domain=str(coregistration_grid),
                tag=tag,
                primary=np.asarray(state.primary_deramped, dtype=np.complex64),
                secondary=np.asarray(state.secondary_aligned, dtype=np.complex64),
                row_origin=row_origin,
                col_origin=col_origin,
                grid_shape=grid_shape,
                wavelength_m=float(state.primary.geometry.wavelength_m),
                grid_identity=grid_identity,
                crs=crs,
                transform=transform,
                scientific_lineage=state.scientific_lineage,
                phase_state=_phase_state_manifest(state),
            )
            wrote += 1
    if wrote == 0:
        reject_invalid_state("coregister_s1_scene wrote no scene units")
    logger.info(
        "Wrote CoregisteredSceneStore %s date=%s reference=%s n_units=%s "
        "(geometry + date constants, ESD/Ampcor off)",
        output_root,
        date_id,
        reference_id,
        wrote,
    )
    return output_root


def _resolve_apply_selection(
    primary_products: Sequence[S1Product],
    secondary_products: Sequence[S1Product],
    *,
    primary_paths: Sequence[Path],
    swaths: tuple[str, ...] | None,
    bursts: BurstSelection | None,
    roi: BoundingBox | Polygons | None,
    ref_orbits: Sequence[Path | None],
    dem: DEM,
) -> tuple[tuple[str, ...], dict[tuple[int, str], list[int]]]:
    """Resolve swaths and bursts; ROI wins over explicit selection."""
    if roi is not None:
        logger.info("ROI provided; explicit swaths/bursts selection ignored")
        ordered = sorted(
            primary_products[0].swaths,
            key=lambda item: item.slant_range_time_s,
        )
        swath_tuple = tuple(item.swath for item in ordered)
        resolved = _select_bursts_by_roi(
            roi,
            list(primary_paths),
            swath_tuple,
            orbits=ref_orbits,
            dem=dem,
        )
        common: dict[tuple[int, str], list[int]] = {}
        for frame_index in range(len(primary_products)):
            for swath in swath_tuple:
                primary_swath = primary_products[frame_index].swath(swath)
                secondary_swath = secondary_products[frame_index].swath(swath)
                shared = set(_common_burst_indices(primary_swath, secondary_swath))
                selected = [
                    index
                    for index in resolved.get((frame_index, swath), [])
                    if index in shared
                ]
                if selected:
                    common[(frame_index, swath)] = selected
        if not any(common.values()):
            reject_invalid_state("ROI selection contains no common bursts")
        return swath_tuple, common
    swath_tuple = _resolve_swaths(primary_products, secondary_products, swaths)
    resolved = _resolve_common_bursts(
        primary_products,
        secondary_products,
        swath_tuple,
        bursts,
    )
    return swath_tuple, resolved


def _unit_placement(
    primary_products: Sequence[S1Product],
    resolved: dict[tuple[int, str], list[int]],
    swath_tuple: tuple[str, ...],
    range_offsets: dict[str, int],
) -> tuple[dict[str, list[tuple[int, int, int]]], int, int]:
    """Place selected bursts on one shared absolute radar grid."""
    primary_swath0 = primary_products[0].swath(swath_tuple[0])
    dt = primary_swath0.azimuth_time_interval_s
    burst_lines = {
        swath: primary_products[0].swath(swath).lines_per_burst for swath in swath_tuple
    }
    burst_width = {
        swath: primary_products[0].swath(swath).samples_per_burst
        for swath in swath_tuple
    }
    azimuth_origin = min(
        primary_products[frame_index].swath(swath).bursts[burst_index].azimuth_time
        for (frame_index, swath), indices in resolved.items()
        for burst_index in indices
    )
    units_by_swath: dict[str, list[tuple[int, int, int]]] = {}
    for swath in swath_tuple:
        units: list[tuple[int, int, int]] = []
        for (frame_index, swath_key), indices in resolved.items():
            if swath_key != swath:
                continue
            swath_obj = primary_products[frame_index].swath(swath)
            for burst_index in indices:
                azimuth_offset = round(
                    (
                        swath_obj.bursts[burst_index].azimuth_time - azimuth_origin
                    ).total_seconds()
                    / dt
                )
                units.append((frame_index, burst_index, azimuth_offset))
        units.sort(key=lambda unit: unit[2], reverse=True)
        units_by_swath[swath] = units
    if not any(units_by_swath.values()):
        reject_invalid_state("selection contains no bursts")
    frame_rows = max(
        azimuth_offset + burst_lines[swath]
        for swath, units in units_by_swath.items()
        for _, _, azimuth_offset in units
    )
    frame_cols = max(range_offsets[swath] + burst_width[swath] for swath in swath_tuple)
    return units_by_swath, frame_rows, frame_cols
