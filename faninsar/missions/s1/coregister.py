"""Sentinel-1 Stack coregistration with the Reference computed once (P0059).

:func:`prepare_s1_reference` reads and deramps the Reference bursts exactly
once per Stack, commits the Reference scene product (radar: the cropped
Reference SLC; geo: the geocoded Reference SLC), and returns a
metadata-only :class:`S1ReferenceContext`. :func:`coregister_s1_secondary`
then processes one secondary date against that context; Reference pixels
are never re-read from SAFE nor re-deramped per date.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.io.storage.scene_store import (
    CoregisteredSceneStore,
    SceneWriter,
    scene_grid_identity,
)
from faninsar.logging import setup_logger
from faninsar.missions.s1.measure import _resolve_common_bursts, _resolve_swaths
from faninsar.missions.s1.processing import (
    SPEED_OF_LIGHT_M_S,
    BurstSelection,
    ProductionPairState,
    _as_frame_paths,
    _common_burst_indices,
    _phase_state_manifest,
    _roi_burst_window,
    _scene_id,
    _select_bursts_by_roi,
    _swath_range_offsets,
    build_reference_geo_lut,
    load_production_scene,
    reference_window_crop,
    stage_coregister,
    stage_deramp,
)
from faninsar.missions.s1.safe import open_acquisition_products
from faninsar.processing.coregistration.tops.deramp import deramp, reramp
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.record import radar_geometry_record

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from faninsar.core.orbit import OrbitMetadata
    from faninsar.data.query import BoundingBox, Polygons
    from faninsar.io.storage.product import Product
    from faninsar.missions.s1.processing import ProductionScene
    from faninsar.missions.s1.types import S1Product
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
    from faninsar.processing.geometry import DEM
    from faninsar.processing.mosaicking.grid import GeoGridSpec

logger = setup_logger(__name__)

__all__ = [
    "S1ReferenceContext",
    "S1ReferenceUnit",
    "coregister_s1_scene",
    "coregister_s1_secondary",
    "prepare_s1_reference",
]


@dataclass(slots=True)
class S1ReferenceUnit:
    """Metadata of one Reference burst unit; holds no Reference pixels.

    Attributes
    ----------
    tag : str
        Unit tag ``f<frame>_<swath>_b<burst>``.
    frame_index, swath, burst_index : int, str, int
        Burst identity within the Reference frames.
    scene : ProductionScene
        Reference burst geometry, carrier, and placement with zero-stride
        placeholder samples.
    native_shape : tuple[int, int]
        Full native burst shape.
    roi_window : tuple[int, int, int, int] or None
        Radar ROI window on the native burst (radar ROI runs only).
    crop : tuple[int, int, int, int] or None
        Pair-independent Reference crop of ``roi_window``.
    row_origin, col_origin : int
        Placement of the persisted unit on the scene grid.
    lut : Geo2RdrLUT or None
        Reference geo2rdr LUT (geo runs only).

    """

    tag: str
    frame_index: int
    swath: str
    burst_index: int
    scene: ProductionScene
    native_shape: tuple[int, int]
    roi_window: tuple[int, int, int, int] | None = None
    crop: tuple[int, int, int, int] | None = None
    row_origin: int = 0
    col_origin: int = 0
    lut: Geo2RdrLUT | None = None


@dataclass(slots=True)
class S1ReferenceContext:
    """Everything a secondary date needs from the once-computed Reference.

    Attributes
    ----------
    reference_id : str
        Reference acquisition identifier.
    product : CoregisteredSceneStore
        Committed Reference scene product.
    coregistration_grid : {"radar", "geo"}
        Product grid.
    grid_shape : tuple[int, int]
        Scene grid shape.
    units : dict[str, S1ReferenceUnit]
        Reference units by tag, in placement order.
    geo_grid : GeoGridSpec or None
        Geographic grid for geo runs.
    roi : BoundingBox or Polygons or None
        Apply-window ROI.
    dem : DEM
        Height sampler.

    """

    reference_id: str
    product: CoregisteredSceneStore
    coregistration_grid: Literal["radar", "geo"]
    grid_shape: tuple[int, int]
    units: dict[str, S1ReferenceUnit] = field(default_factory=dict)
    geo_grid: GeoGridSpec | None = None
    roi: BoundingBox | Polygons | None = None
    dem: DEM | None = None

    @property
    def write_id(self) -> str:
        """Return the Reference product write id secondaries bind."""
        return self.product.write_id


def prepare_s1_reference(
    reference_path: Path | Sequence[Path],
    *,
    output_scene: Path,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    roi: BoundingBox | Polygons | None = None,
    dem: DEM,
    coregistration_grid: Literal["radar", "geo"] = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_lut_cache_dir: Path | None = None,
    orbit_path: Path | None = None,
    device: str = "auto",
    overwrite: bool = False,
) -> S1ReferenceContext:
    """Compute the Reference scene once and return its context.

    When ``output_scene`` already holds a committed Reference product and
    ``overwrite`` is false, Reference pixels are not read: only burst
    metadata, carriers, geometry, and (geo) LUTs are loaded.

    Parameters
    ----------
    reference_path : path or sequence of path
        Reference SAFE product(s), one per frame.
    output_scene : pathlib.Path
        Reference scene product directory, ``<work_dir>/coreg/<date>``.
    swaths : tuple of str, optional
        Sub-swaths; ``None`` selects all (or the ROI footprint set).
    bursts : BurstSelection, optional
        Per-swath burst selection.
    roi : BoundingBox or Polygons, optional
        Apply-window crop; selects bursts when ``bursts`` is omitted.
    dem : DEM
        Height sampler for geometry.
    coregistration_grid : {"radar", "geo"}, optional
        Product grid.
    geo_grid : GeoGridSpec, optional
        Required for geo coregistration.
    geo_lut_cache_dir : pathlib.Path, optional
        Shared content-addressed LUT cache (geo).
    orbit_path : pathlib.Path, optional
        Precise orbit of the Reference.
    device : str, optional
        Numerical device.
    overwrite : bool, optional
        Recompute and replace an existing Reference product.

    Returns
    -------
    S1ReferenceContext
        Metadata-only Reference context.

    Raises
    ------
    InvalidProcessingStateError
        If the selection is empty, the configuration is invalid, or an
        existing Reference product does not match the current selection.

    """
    if coregistration_grid not in {"radar", "geo"}:
        reject_invalid_state("coregistration_grid must be 'radar' or 'geo'")
    if coregistration_grid == "geo" and geo_grid is None:
        reject_invalid_state("geo coregistration requires geo_grid")
    primary_paths = _as_frame_paths(reference_path, "reference_path")
    primary_products = open_acquisition_products(primary_paths)
    ref_orbits = [orbit_path] * len(primary_products)
    swath_tuple, resolved = _resolve_apply_selection(
        primary_products,
        primary_products,
        primary_paths=primary_paths,
        swaths=swaths,
        bursts=bursts,
        roi=roi,
        ref_orbits=ref_orbits,
        dem=dem,
    )
    range_offsets = _swath_range_offsets(swath_tuple, primary_products)
    units_by_swath, frame_rows, frame_cols = _unit_placement(
        primary_products, resolved, swath_tuple, range_offsets
    )
    reference_id = _scene_id(primary_paths[0])
    output_scene = Path(output_scene)
    reuse = (output_scene / "item.json").is_file() and not overwrite
    if coregistration_grid == "geo":
        grid_shape = (int(geo_grid.shape[0]), int(geo_grid.shape[1]))  # type: ignore[union-attr]
        header = _geo_header(geo_grid)  # type: ignore[arg-type]
    else:
        grid_shape = (int(frame_rows), int(frame_cols))
        header = {}
    work_root = output_scene.parent / f".{output_scene.name}.work"
    writer: SceneWriter | None = None
    units: dict[str, S1ReferenceUnit] = {}
    try:
        for swath in swath_tuple:
            for frame_index, burst_index, azimuth_offset in units_by_swath[swath]:
                tag = f"f{frame_index}_{swath}_b{burst_index}"
                scene = load_production_scene(
                    primary_paths[min(frame_index, len(primary_paths) - 1)],
                    swath=swath,
                    burst_index=burst_index,
                    dem=dem,
                    orbit_path=ref_orbits[frame_index],
                    coregistration_grid=coregistration_grid,
                    full_range=True,
                    read_samples=not reuse,
                    product=primary_products[frame_index],
                )
                if writer is None and not reuse:
                    writer = SceneWriter(
                        output_scene,
                        date_id=reference_id,
                        reference_id=reference_id,
                        domain=coregistration_grid,
                        grid_shape=grid_shape,
                        wavelength_m=float(scene.geometry.wavelength_m),
                        overwrite=overwrite,
                        **header,
                    )
                unit = S1ReferenceUnit(
                    tag=tag,
                    frame_index=frame_index,
                    swath=swath,
                    burst_index=burst_index,
                    scene=scene,
                    native_shape=(
                        int(scene.array.samples.shape[0]),
                        int(scene.array.samples.shape[1]),
                    ),
                    row_origin=int(azimuth_offset),
                    col_origin=int(range_offsets[swath]),
                )
                if coregistration_grid == "radar":
                    payload = _prepare_radar_unit(
                        unit, roi=roi, dem=dem, compute=not reuse, device=device
                    )
                else:
                    payload = _prepare_geo_unit(
                        unit,
                        dem=dem,
                        geo_grid=geo_grid,  # type: ignore[arg-type]
                        geo_lut_cache_dir=geo_lut_cache_dir,
                        work_dir=work_root / tag,
                        roi=roi,
                        compute=not reuse,
                        device=device,
                    )
                    if unit.lut is None:
                        logger.warning("%s: empty geocoded footprint; skipped", tag)
                        continue
                unit.scene = _without_pixels(scene)
                units[tag] = unit
                if writer is not None and payload is not None:
                    writer.add_unit(
                        tag=tag,
                        payload=payload,
                        row_origin=unit.row_origin,
                        col_origin=unit.col_origin,
                        scientific_lineage=[
                            {
                                "stage": "s1_reference_burst",
                                "source": str(primary_paths[frame_index]),
                            }
                        ],
                    )
                del payload, scene
        if not units:
            reject_invalid_state("Reference selection produced no scene units")
        if writer is not None:
            properties: dict[str, object] = {}
            if coregistration_grid == "radar":
                properties["faninsar:radar_geometry"] = frame_geometry_record(
                    primary_products,
                    resolved,
                    swath_tuple,
                    range_offsets,
                    grid_shape,
                    orbit=next(iter(units.values())).scene.swath.orbit,
                )
            product = writer.commit(properties=properties)
        else:
            product = CoregisteredSceneStore.open(output_scene)
            _check_reused_reference(product, reference_id, units, coregistration_grid)
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    logger.info(
        "Reference %s %s once: %d units",
        reference_id,
        "reused" if reuse else "computed",
        len(units),
    )
    return S1ReferenceContext(
        reference_id=reference_id,
        product=product,
        coregistration_grid=coregistration_grid,
        grid_shape=grid_shape,
        units=units,
        geo_grid=geo_grid,
        roi=roi,
        dem=dem,
    )


def _prepare_radar_unit(
    unit: S1ReferenceUnit,
    *,
    roi: BoundingBox | Polygons | None,
    dem: DEM,
    compute: bool,
    device: str,
) -> np.ndarray | None:
    """Resolve the Reference ROI crop and return the radar Reference payload.

    The payload is ``reramp(deramp(reference))`` on the Reference crop, the
    same pointwise values the per-pair path produced. Returns ``None`` when
    ``compute`` is false (the persisted product is reused).
    """
    scene = unit.scene
    if roi is not None:
        unit.roi_window = _roi_burst_window(
            roi, scene.geometry, dem, unit.native_shape, device=device
        )
        if unit.roi_window is not None:
            unit.crop = reference_window_crop(unit.roi_window, unit.native_shape)
            unit.row_origin += unit.crop[0]
            unit.col_origin += unit.crop[2]
    if not compute:
        return None
    deramped = np.where(
        scene.array.valid_mask, deramp(scene.array.samples, scene.carrier), 0
    )
    row0 = col0 = 0
    if unit.crop is not None:
        row0, row1, col0, col1 = unit.crop
        deramped = deramped[row0:row1, col0:col1]
    return np.asarray(
        reramp(
            deramped,
            scene.carrier,
            row0=row0,
            col0=col0,
            native_height=unit.native_shape[0],
        ),
        dtype=np.complex64,
    )


def _prepare_geo_unit(
    unit: S1ReferenceUnit,
    *,
    dem: DEM,
    geo_grid: GeoGridSpec,
    geo_lut_cache_dir: Path | None,
    work_dir: Path,
    roi: BoundingBox | Polygons | None,
    compute: bool,
    device: str,
) -> np.ndarray | None:
    """Build the Reference LUT and (unless reusing) the geocoded Reference."""
    from faninsar.processing.geocoding.geo_modes import (
        geocode_reference_slc_chunked,
    )

    scene = unit.scene
    lut = build_reference_geo_lut(
        scene,
        radar_shape=unit.native_shape,
        dem=dem,
        geo_grid=geo_grid,
        storage_dir=work_dir / "lut",
        geo_lut_cache_dir=geo_lut_cache_dir,
        roi=roi,
        device=device,
    )
    if lut.shape[0] == 0 or lut.shape[1] == 0 or not np.any(lut.valid):
        return None
    lut = _materialize_lut(lut)
    unit.lut = lut
    unit.row_origin = int(lut.row0)
    unit.col_origin = int(lut.col0)
    if not compute:
        return None
    deramped = np.where(
        scene.array.valid_mask, deramp(scene.array.samples, scene.carrier), 0
    )
    reference_geo, _ = geocode_reference_slc_chunked(
        deramped,
        reference_carrier=scene.carrier,
        reference_lut=lut,
        output_dir=work_dir / "slc",
        device=device,
    )
    return np.array(reference_geo, dtype=np.complex64)


def _materialize_lut(lut: Geo2RdrLUT) -> Geo2RdrLUT:
    """Return the LUT held in memory so its scratch directory can be removed."""
    return replace(
        lut,
        az_full=np.array(lut.az_full),
        rg_full=np.array(lut.rg_full),
        valid=np.array(lut.valid),
        height_full=None if lut.height_full is None else np.array(lut.height_full),
    )


def _without_pixels(scene: ProductionScene) -> ProductionScene:
    """Return ``scene`` with zero-stride placeholder samples (metadata only)."""
    shape = scene.array.samples.shape
    array = replace(
        scene.array,
        samples=np.broadcast_to(np.complex64(0), shape),
        valid_mask=np.broadcast_to(np.True_, shape),
    )
    return replace(scene, array=array)


def _check_reused_reference(
    product: CoregisteredSceneStore,
    reference_id: str,
    units: Mapping[str, S1ReferenceUnit],
    coregistration_grid: str,
) -> None:
    """Reject a persisted Reference that no longer matches the selection."""
    if not product.is_reference or product.reference_id != reference_id:
        reject_invalid_state(
            f"{product.root} is not the Reference scene of {reference_id}"
        )
    if product.domain != coregistration_grid:
        reject_invalid_state(
            f"persisted Reference is {product.domain!r}, requested "
            f"{coregistration_grid!r}; recoregister with overwrite=True"
        )
    stored = product.unit_map()
    if set(stored) != set(units):
        reject_invalid_state(
            "persisted Reference units differ from the current selection; "
            "recoregister with overwrite=True"
        )
    for tag, unit in units.items():
        placed = stored[tag]
        if (placed.row_origin, placed.col_origin) != (unit.row_origin, unit.col_origin):
            reject_invalid_state(
                f"persisted Reference unit {tag!r} is placed differently from "
                "the current selection; recoregister with overwrite=True"
            )


def coregister_s1_secondary(
    context: S1ReferenceContext,
    secondary_path: Path | Sequence[Path],
    *,
    output_scene: Path,
    misreg_az_px: float,
    misreg_rg_px: float,
    secondary_orbit_path: Path | None = None,
    device: str = "auto",
    inputs: Mapping[str, Product | str] | None = None,
    properties: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> CoregisteredSceneStore:
    """Coregister one secondary date onto the once-prepared Reference.

    Only the secondary is read, deramped, and resampled. Reference units the
    secondary does not cover are omitted; samples outside the secondary
    support stay invalid. ESD and Ampcor are not measured.

    Parameters
    ----------
    context : S1ReferenceContext
        Result of :func:`prepare_s1_reference`.
    secondary_path : path or sequence of path
        Secondary SAFE product(s), one per Reference frame.
    output_scene : pathlib.Path
        Secondary scene product directory, ``<work_dir>/coreg/<date>``.
    misreg_az_px, misreg_rg_px : float
        Inverted date constants (0 for geometry-only). Must be finite.
    secondary_orbit_path : pathlib.Path, optional
        Precise orbit of the secondary.
    device : str, optional
        Numerical device.
    inputs : mapping, optional
        Extra upstream bindings such as ``misreg``; ``reference`` is always
        bound to the Reference product.
    properties : mapping, optional
        Extra item properties.
    overwrite : bool, optional
        Replace an existing secondary product.

    Returns
    -------
    CoregisteredSceneStore
        The committed secondary scene product.

    Raises
    ------
    InvalidProcessingStateError
        If the constants are non-finite, the date is the Reference, or the
        secondary covers no Reference unit.

    """
    if not np.isfinite(misreg_az_px) or not np.isfinite(misreg_rg_px):
        reject_invalid_state("misreg_az_px and misreg_rg_px must be finite")
    secondary_paths = _as_frame_paths(secondary_path, "secondary_path")
    date_id = _scene_id(secondary_paths[0])
    if date_id == context.reference_id:
        reject_invalid_state("coregister_s1_secondary requires a non-Reference date")
    secondary_products = open_acquisition_products(secondary_paths)
    reference = context.product
    reference_units = reference.unit_map()
    domain = context.coregistration_grid
    header = _geo_header(context.geo_grid) if domain == "geo" else {}  # type: ignore[arg-type]
    output_scene = Path(output_scene)
    work_root = output_scene.parent / f".{output_scene.name}.work"
    wrote = 0
    try:
        with SceneWriter(
            output_scene,
            date_id=date_id,
            reference_id=context.reference_id,
            domain=domain,
            grid_shape=context.grid_shape,
            wavelength_m=reference.wavelength_m,
            overwrite=overwrite,
            **header,
        ) as writer:
            for tag, unit in context.units.items():
                if not _secondary_covers(unit, secondary_products, tag, date_id):
                    continue
                payload, state = _coregister_unit(
                    context,
                    unit,
                    secondary_paths[min(unit.frame_index, len(secondary_paths) - 1)],
                    reference=reference,
                    misreg_az_px=float(misreg_az_px),
                    misreg_rg_px=float(misreg_rg_px),
                    secondary_orbit_path=secondary_orbit_path,
                    work_dir=work_root / tag,
                    device=device,
                    product=secondary_products[unit.frame_index],
                )
                placed = reference_units[tag]
                if payload.shape != placed.shape:
                    reject_invalid_state(
                        f"{tag}: secondary shape {payload.shape} differs from "
                        f"the Reference {placed.shape}"
                    )
                writer.add_unit(
                    tag=tag,
                    payload=payload,
                    row_origin=placed.row_origin,
                    col_origin=placed.col_origin,
                    scientific_lineage=state.scientific_lineage,
                    phase_state=_phase_state_manifest(state),
                )
                del payload, state
                shutil.rmtree(work_root / tag, ignore_errors=True)
                wrote += 1
            if wrote == 0:
                reject_invalid_state(f"secondary {date_id} covers no Reference unit")
            bindings: dict[str, Product | str] = dict(inputs or {})
            bindings["reference"] = reference.write_id
            record = {
                "faninsar:coregistration": {
                    "misreg_az_px": float(misreg_az_px),
                    "misreg_rg_px": float(misreg_rg_px),
                },
                **dict(properties or {}),
            }
            product = writer.commit(inputs=bindings, properties=record)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    logger.info(
        "Coregistered %s onto Reference %s: %d/%d units",
        date_id,
        context.reference_id,
        wrote,
        len(context.units),
    )
    return product


def _secondary_covers(
    unit: S1ReferenceUnit,
    secondary_products: Sequence[S1Product],
    tag: str,
    date_id: str,
) -> bool:
    """Return whether the secondary acquisition contains the unit's burst."""
    if unit.frame_index >= len(secondary_products):
        logger.info("%s: secondary %s has no frame %d", tag, date_id, unit.frame_index)
        return False
    names = {item.swath for item in secondary_products[unit.frame_index].swaths}
    if unit.swath not in names:
        logger.info("%s: secondary %s has no sub-swath %s", tag, date_id, unit.swath)
        return False
    secondary_swath = secondary_products[unit.frame_index].swath(unit.swath)
    if unit.burst_index not in _common_burst_indices(unit.scene.swath, secondary_swath):
        logger.info("%s: secondary %s does not cover this burst", tag, date_id)
        return False
    return True


def _coregister_unit(
    context: S1ReferenceContext,
    unit: S1ReferenceUnit,
    secondary_path: Path,
    *,
    reference: CoregisteredSceneStore,
    misreg_az_px: float,
    misreg_rg_px: float,
    secondary_orbit_path: Path | None,
    work_dir: Path,
    device: str,
    product: S1Product | None = None,
) -> tuple[np.ndarray, ProductionPairState]:
    """Resample one secondary burst onto its Reference unit."""
    domain = context.coregistration_grid
    secondary = load_production_scene(
        secondary_path,
        swath=unit.swath,
        burst_index=unit.burst_index,
        dem=context.dem,
        orbit_path=secondary_orbit_path,
        coregistration_grid=domain,
        full_range=True,
        product=product,
    )
    state = ProductionPairState(
        pair_id=f"{context.reference_id}_{secondary.scene_id}_{unit.tag}",
        primary=unit.scene,
        secondary=secondary,
        dem=context.dem,  # type: ignore[arg-type]
        coregistration_grid=domain,
    )
    state = stage_deramp(state, device=device, secondary_only=True)
    reference_geo_valid = None
    if domain == "geo":
        reference_samples, _ = reference.read(unit.tag)
        reference_geo_valid = np.isfinite(reference_samples.real) & np.isfinite(
            reference_samples.imag
        )
        del reference_samples
    state = stage_coregister(
        state,
        misreg_az_px=misreg_az_px,
        misreg_rg_px=misreg_rg_px,
        device=device,
        coregistration_grid=domain,
        geo_grid=context.geo_grid,
        geo_work_dir=work_dir if domain == "geo" else None,
        roi=context.roi,
        roi_window=unit.roi_window,
        prepared_geo_lut=unit.lut,
        secondary_only=True,
        reference_shape=unit.native_shape,
        reference_geo_valid=reference_geo_valid,
    )
    if state.secondary_aligned is None:
        reject_invalid_state(f"{unit.tag}: coregistration produced no aligned SLC")
    payload = np.array(state.secondary_aligned, dtype=np.complex64)
    if domain == "radar" and unit.crop is not None:
        payload = _crop_to_reference_window(state, unit.crop, payload, tag=unit.tag)
    state.secondary_aligned = None
    return payload, state


def _crop_to_reference_window(
    state: ProductionPairState,
    crop: tuple[int, int, int, int],
    secondary: np.ndarray,
    *,
    tag: str,
) -> np.ndarray:
    """Slice a halo-cropped secondary to the Reference ROI crop."""
    origin = state.radar_roi_origin or (0, 0)
    row0 = crop[0] - origin[0]
    col0 = crop[2] - origin[1]
    row1 = row0 + crop[1] - crop[0]
    col1 = col0 + crop[3] - crop[2]
    if row0 < 0 or col0 < 0 or row1 > secondary.shape[0] or col1 > secondary.shape[1]:
        reject_invalid_state(
            f"{tag}: pair ROI crop does not contain the Reference ROI crop"
        )
    return np.ascontiguousarray(secondary[row0:row1, col0:col1])


def coregister_s1_scene(
    reference_path: Path | Sequence[Path],
    secondary_path: Path | Sequence[Path],
    *,
    output_scene: Path,
    reference_scene: Path,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    roi: BoundingBox | Polygons | None = None,
    dem: DEM,
    misreg_az_px: float,
    misreg_rg_px: float,
    coregistration_grid: Literal["radar", "geo"] = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_lut_cache_dir: Path | None = None,
    primary_orbit_path: Path | None = None,
    secondary_orbit_path: Path | None = None,
    device: str = "auto",
) -> CoregisteredSceneStore:
    """Coregister one secondary acquisition onto a Reference scene product.

    Convenience wrapper: prepares (or reuses) the Reference product at
    ``reference_scene`` and coregisters one secondary. Stacks call
    :func:`prepare_s1_reference` once and :func:`coregister_s1_secondary`
    per date instead.

    Parameters
    ----------
    reference_path, secondary_path : path or sequence of path
        SAFE products per date.
    output_scene, reference_scene : pathlib.Path
        Secondary and Reference scene product directories.
    swaths, bursts, roi, dem, coregistration_grid, geo_grid, geo_lut_cache_dir
        See :func:`prepare_s1_reference`.
    misreg_az_px, misreg_rg_px : float
        Inverted date constants (0 for geometry-only).
    primary_orbit_path, secondary_orbit_path : pathlib.Path, optional
        Precise orbits.
    device : str, optional
        Numerical device.

    Returns
    -------
    CoregisteredSceneStore
        The committed secondary scene product.

    """
    if not np.isfinite(misreg_az_px) or not np.isfinite(misreg_rg_px):
        reject_invalid_state("misreg_az_px and misreg_rg_px must be finite")
    context = prepare_s1_reference(
        reference_path,
        output_scene=reference_scene,
        swaths=swaths,
        bursts=bursts,
        roi=roi,
        dem=dem,
        coregistration_grid=coregistration_grid,
        geo_grid=geo_grid,
        geo_lut_cache_dir=geo_lut_cache_dir,
        orbit_path=primary_orbit_path,
        device=device,
    )
    return coregister_s1_secondary(
        context,
        secondary_path,
        output_scene=output_scene,
        misreg_az_px=misreg_az_px,
        misreg_rg_px=misreg_rg_px,
        secondary_orbit_path=secondary_orbit_path,
        device=device,
    )


def frame_geometry_record(
    primary_products: Sequence[S1Product],
    resolved: Mapping[tuple[int, str], Sequence[int]],
    swath_tuple: tuple[str, ...],
    range_offsets: Mapping[str, int],
    grid_shape: tuple[int, int],
    *,
    orbit: OrbitMetadata,
) -> dict[str, object]:
    """Return the merged Reference frame radar geometry as the units are placed.

    Follows ISCE2 ``topsStack`` (``mergeBursts``/``VRTManager``) and
    :func:`_unit_placement`: row 0 is the earliest selected burst sensing
    time with the first swath's azimuth interval; column 0 is the first
    swath's starting slant range with its range spacing; later swaths sit at
    integer column offsets. The orbit is trimmed to the frame span.

    Parameters
    ----------
    primary_products : sequence of S1Product
        Reference products, one per frame.
    resolved : mapping
        Selected burst indices per ``(frame, swath)``.
    swath_tuple : tuple of str
        Selected sub-swaths in range order.
    range_offsets : mapping
        Integer column offset of each sub-swath.
    grid_shape : tuple[int, int]
        Merged frame shape.
    orbit : OrbitMetadata
        Reference orbit (precise when configured).

    Returns
    -------
    dict
        Strict-JSON record for
        :func:`~faninsar.processing.geometry.record.radar_geometry_from_record`.

    """
    swath0 = primary_products[0].swath(swath_tuple[0])
    sensing_start = min(
        primary_products[frame_index].swath(swath).bursts[burst_index].azimuth_time
        for (frame_index, swath), indices in resolved.items()
        for burst_index in indices
    )
    grid = RadarGrid(
        shape=(int(grid_shape[0]), int(grid_shape[1])),
        starting_slant_range_m=float(
            swath0.slant_range_time_s * SPEED_OF_LIGHT_M_S / 2.0
        ),
        range_spacing_m=float(swath0.range_pixel_spacing_m),
        sensing_start=sensing_start,
        azimuth_time_interval_s=float(swath0.azimuth_time_interval_s),
        wavelength_m=float(SPEED_OF_LIGHT_M_S / swath0.radar_frequency_hz),
        look_direction="right",
    )
    return radar_geometry_record(
        grid,
        orbit,
        swath_column_offsets={name: int(range_offsets[name]) for name in swath_tuple},
    )


def _geo_header(geo_grid: GeoGridSpec) -> dict[str, object]:
    """Return the SceneWriter geo header for ``geo_grid``."""
    return {
        "grid_identity": scene_grid_identity(
            "geo",
            geo_grid.shape,
            {
                "crs": geo_grid.crs,
                "transform": list(geo_grid.transform),
                "resolution_m": list(geo_grid.resolution_m),
            },
        ),
        "crs": geo_grid.crs,
        "transform": tuple(float(value) for value in tuple(geo_grid.transform)[:6]),
    }


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
    """Resolve swaths and bursts; explicit selection wins over ROI re-select.

    ``roi`` still crops the apply window. NESD overlap is not required to
    fall inside that crop; the burst set is the one already resolved by
    the caller (or ROI footprints when the caller omitted ``bursts``).
    """
    if bursts is not None or swaths is not None:
        swath_tuple = _resolve_swaths(primary_products, secondary_products, swaths)
        resolved = _resolve_common_bursts(
            primary_products,
            secondary_products,
            swath_tuple,
            bursts,
        )
        return swath_tuple, resolved
    if roi is not None:
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
        kept = tuple(
            swath
            for swath in swath_tuple
            if any(
                common.get((frame_index, swath))
                for frame_index in range(len(primary_products))
            )
        )
        return kept, common
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
