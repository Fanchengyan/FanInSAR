"""Production Sentinel-1 pair workflows on radar or geographic grids."""

from __future__ import annotations

import gc
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.missions.sentinel1 import (
    open_safe_product,
    read_eof_orbit,
    read_full_burst,
    read_swath_bursts,
    stitch_bursts,
)
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.coreg import (
    combine_offset_fields,
    dense_geometry_offsets,
    estimate_azimuth_shift_esd,
    refine_shift_with_correlation,
    resample_complex,
    resample_complex_deramped_reramp,
)
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry import (
    ConstantHeightDEM,
    RadarGeometryModel,
    rdr2geo_with_dem_chunked,
)
from faninsar.processing.geometry.baseline import BaselineComponents
from faninsar.processing.geometry.dem import GeoidAdjustedDEM, RasterDEM
from faninsar.processing.interferometry.flatten import (
    compute_geometric_phase_from_geo,
    compute_topographic_phase,
    estimate_residual_azimuth_ramp,
    estimate_residual_topographic_scale,
    remove_azimuth_phase_ramp,
    remove_topographic_phase,
)
from faninsar.processing.interferometry.pair import (
    form_interferogram,
    goldstein_filter,
    mask_invalid_looks,
)
from faninsar.processing.memory import close_memmap, release_memmap_pages
from faninsar.processing.pipeline.products import (
    PairProductArrays,
    write_pair_stac_item,
    write_pair_zarr,
)
from faninsar.processing.tops.carrier import carrier_from_swath
from faninsar.processing.tops.deramp import TOPSCarrierModel, deramp, reramp
from faninsar.processing.unwrap import SnaphuConfig, UnwrapBackend
from faninsar.processing.unwrap import unwrap as unwrap_dispatch

if TYPE_CHECKING:
    from faninsar.missions.sentinel1.io import BurstArray
    from faninsar.missions.sentinel1.types import S1Burst, S1Product, S1Swath
    from faninsar.processing.coreg.offsets import OffsetFieldResult
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.memory import MemoryWatchdog
    from faninsar.processing.merge.grid import GeoGridSpec
    from faninsar.processing.pipeline.geo_lut import Geo2RdrLUT
    from faninsar.processing.unwrap.common import CommonUnwrapResult

logger = setup_logger(__name__)

SPEED_OF_LIGHT_M_S = 299_792_458.0
ScopeMode = Literal["burst", "swath"]
CoregistrationGrid = Literal["radar", "geo"]


def _scene_id(path: Path) -> str:
    stem = path.stem.replace(".SAFE", "")
    for part in stem.split("_"):
        if len(part) >= 8 and part[:8].isdigit():
            return part[:8]
    return stem


def _radar_model(
    swath: S1Swath,
    burst: S1Burst,
    *,
    shape: tuple[int, int],
    row0: int,
    col0: int,
) -> RadarGeometryModel:
    """Build geometry model whose (0,0) is the array origin."""
    local_line = row0 - burst.index * swath.lines_per_burst
    sensing_start = burst.azimuth_time + timedelta(
        seconds=float(local_line) * swath.azimuth_time_interval_s
    )
    starting_slant_range_m = (
        swath.slant_range_time_s * SPEED_OF_LIGHT_M_S / 2.0
        + float(col0) * swath.range_pixel_spacing_m
    )
    grid = RadarGrid(
        shape=shape,
        starting_slant_range_m=starting_slant_range_m,
        range_spacing_m=swath.range_pixel_spacing_m,
        sensing_start=sensing_start,
        azimuth_time_interval_s=swath.azimuth_time_interval_s,
        wavelength_m=SPEED_OF_LIGHT_M_S / swath.radar_frequency_hz,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(grid, swath.orbit)


@dataclass(frozen=True, slots=True)
class ProductionScene:
    """Full-burst or stitched-swath scene ready for pair processing."""

    scene_id: str
    path: Path
    product: S1Product
    swath: S1Swath
    burst: S1Burst
    array: BurstArray
    carrier: TOPSCarrierModel
    geometry: RadarGeometryModel


@dataclass
class ProductionPairState:
    """Mutable state for the production pair workflow with stage log."""

    pair_id: str
    reference: ProductionScene
    secondary: ProductionScene
    dem: DEMSampler
    reference_deramped: np.ndarray | None = None
    secondary_deramped: np.ndarray | None = None
    range_shift_px: float | None = None
    azimuth_shift_px: float | None = None
    esd_azimuth_shift_px: float | None = None
    secondary_aligned: np.ndarray | None = None
    secondary_aligned_is_flattened: bool = False
    range_offset_flatten_phase: np.ndarray | None = None
    complex_ifg: np.ndarray | None = None
    complex_ifg_flat: np.ndarray | None = None
    topo_phase: np.ndarray | None = None
    coherence: np.ndarray | None = None
    wrapped_phase: np.ndarray | None = None
    unwrapped_phase: np.ndarray | None = None
    connected_components: np.ndarray | None = None
    baseline: BaselineComponents | None = None
    geocoded: dict[str, np.ndarray] | None = None
    geo2rdr_lut: Geo2RdrLUT | None = None
    geo_height_field: np.ndarray | None = None
    geo_grid: GeoGridSpec | None = None
    reference_geocoded_slc: np.ndarray | None = None
    secondary_geocoded_slc: np.ndarray | None = None
    geocoded_slc_valid: np.ndarray | None = None
    zarr_path: Path | None = None
    stac_path: Path | None = None
    log: list[str] = field(default_factory=list)
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    coregistration_timings_s: dict[str, float] = field(default_factory=dict)
    coregistration_grid: CoregistrationGrid = "radar"
    dem_id: str = ""
    coreg_executor: str = "torch"
    coreg_device: str = "auto"
    multilook: tuple[int, int] = (4, 20)
    goldstein_alpha: float = 0.5
    unwrap_method: str = "snaphu"
    geo_grid_meta: dict[str, Any] | None = None
    geo_work_dir: Path | None = None
    memory_watchdog: MemoryWatchdog | None = None

    def note(self, message: str) -> None:
        """Append a stage log line."""
        logger.info("[%s] %s", self.pair_id, message)
        self.log.append(message)


def load_production_scene(
    path: str | Path,
    *,
    swath: str = "IW1",
    scope: ScopeMode = "burst",
    burst_index: int = 0,
    dem: DEMSampler | None = None,
    orbit_path: str | Path | None = None,
    coregistration_grid: CoregistrationGrid = "radar",
    full_range: bool = False,
) -> ProductionScene:
    """Load a full burst or full stitched sub-swath for production processing.

    Parameters
    ----------
    path : path
        SAFE ZIP or directory.
    swath : str, optional
        Sub-swath name.
    scope : {"burst", "swath"}, optional
        ``burst`` reads one full burst; ``swath`` reads and stitches all bursts.
    burst_index : int, optional
        Burst index when ``scope="burst"``.
    dem : DEMSampler, optional
        DEM used only to attach geometry context (not required for load).
    orbit_path : path, optional
        Precise ESA EOF orbit. Annotation orbit vectors are used when omitted.
    coregistration_grid : {"radar", "geo"}, optional
        Select the burst sample layout required by the coregistration grid.
    full_range : bool, optional
        Read the full swath range extent (column 0 through the burst width)
        instead of the valid-sample envelope, matching ISCE2's full-width
        burst layout. Invalid samples are still zeroed via the valid mask.

    Returns
    -------
    ProductionScene
        Full complex array with carrier and geometry model.

    """
    _ = dem
    product = open_safe_product(path)
    s1_swath = product.swath(swath)
    if orbit_path is not None:
        s1_swath = replace(s1_swath, orbit=read_eof_orbit(orbit_path))
    if scope == "burst":
        array = read_full_burst(
            s1_swath,
            burst_index=burst_index,
            geocoding_layout=coregistration_grid == "geo",
            full_range=full_range,
        )
        burst = s1_swath.bursts[burst_index]
    elif scope == "swath":
        bursts = read_swath_bursts(s1_swath)
        array = stitch_bursts(bursts, overlap_blend=True)
        burst = s1_swath.bursts[0]
    else:
        reject_invalid_state(f"unknown scope: {scope}")

    carrier = carrier_from_swath(s1_swath, burst, first_range_sample=array.col0)
    geometry = _radar_model(
        s1_swath,
        burst,
        shape=array.samples.shape,
        row0=array.row0,
        col0=array.col0,
    )
    scene_id = _scene_id(Path(path))
    logger.info(
        "Loaded production scene %s scope=%s shape=%s origin=(%s,%s)",
        scene_id,
        scope,
        array.samples.shape,
        array.row0,
        array.col0,
    )
    return ProductionScene(
        scene_id=scene_id,
        path=Path(path),
        product=product,
        swath=s1_swath,
        burst=burst,
        array=array,
        carrier=carrier,
        geometry=geometry,
    )


def stage_deramp(state: ProductionPairState) -> ProductionPairState:
    """Deramp both full scenes with annotation carriers."""
    state.reference_deramped = deramp(
        state.reference.array.samples, state.reference.carrier
    )
    state.secondary_deramped = deramp(
        state.secondary.array.samples, state.secondary.carrier
    )
    # mask invalid
    state.reference_deramped = np.where(
        state.reference.array.valid_mask,
        state.reference_deramped,
        0,
    )
    state.secondary_deramped = np.where(
        state.secondary.array.valid_mask,
        state.secondary_deramped,
        0,
    )
    state.note(
        f"DERAMP shape={state.reference_deramped.shape} "
        f"mean|z|={float(np.mean(np.abs(state.reference_deramped))):.3f}"
    )
    return state


def _apply_geo_topographic_phase_chunked(
    state: ProductionPairState,
    lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
    reference_geo: np.memmap,
    secondary_geo: np.memmap,
    valid: np.memmap,
    *,
    grid: GeoGridSpec,
    output_dir: Path,
    chunk_size: int,
    watchdog: MemoryWatchdog | None,
) -> tuple[np.memmap, np.ndarray]:
    """Apply geometric phase to disk-backed geographic SLC row tiles.

    Parameters
    ----------
    state : ProductionPairState
        Pair geometry and DEM state.
    lut : Geo2RdrLUT
        Geographic-to-reference-radar coordinates.
    offsets : OffsetFieldResult
        Dense reference-to-secondary radar offsets.
    reference_geo, secondary_geo, valid : numpy.memmap
        Disk-backed geographic SLCs and validity mask, updated in place.
    grid : GeoGridSpec
        Geographic grid defining the shared ground targets.
    output_dir : pathlib.Path
        Directory for disk-backed phase and height fields.
    chunk_size : int
        Number of geographic rows per tile.
    watchdog : MemoryWatchdog, optional
        Memory guard sampled after every completed tile.

    Returns
    -------
    topo_phase, height : tuple[numpy.memmap, numpy.ndarray]
        Complete disk-backed geographic fields.

    """
    from faninsar.processing.pipeline.geo_lut import Geo2RdrLUT, grid_lonlat_rows
    from faninsar.processing.pipeline.geo_resample import (
        compose_secondary_coordinates,
    )

    topo_phase = np.memmap(
        output_dir / "topographic_phase.float32",
        mode="w+",
        dtype=np.float32,
        shape=grid.shape,
    )
    height_field = lut.height_full
    if height_field is None:
        height_field = np.memmap(
            output_dir / "height.float64",
            mode="w+",
            dtype=np.float64,
            shape=grid.shape,
        )
    invalid = np.complex64(np.nan + 1j * np.nan)
    for row_start in range(0, grid.height, chunk_size):
        row_stop = min(row_start + chunk_size, grid.height)
        rows = slice(row_start, row_stop)
        tile_lut = Geo2RdrLUT(
            az_full=lut.az_full[rows],
            rg_full=lut.rg_full[rows],
            valid=lut.valid[rows],
            full_radar_shape=lut.full_radar_shape,
            height_m=lut.height_m,
            height_full=(None if lut.height_full is None else lut.height_full[rows]),
        )
        secondary_azimuth, _, coordinate_valid = compose_secondary_coordinates(
            tile_lut,
            offsets,
        )
        latitude, longitude = grid_lonlat_rows(grid, row_start, row_stop)
        if tile_lut.height_full is None:
            height = np.asarray(
                state.dem.sample(latitude, longitude),
                dtype=np.float64,
            )
        else:
            height = np.asarray(tile_lut.height_full, dtype=np.float64)
        phase = compute_geometric_phase_from_geo(
            state.reference.geometry,
            state.secondary.geometry,
            latitude,
            longitude,
            height,
            tile_lut.az_full,
            secondary_azimuth,
            reference_range_index=tile_lut.rg_full,
        )
        tile_valid = valid[rows] & coordinate_valid & np.isfinite(phase)
        corrected_secondary = secondary_geo[rows] * np.exp(1j * phase).astype(
            np.complex64
        )
        reference_geo[rows] = np.where(
            tile_valid,
            reference_geo[rows],
            invalid,
        ).astype(np.complex64)
        secondary_geo[rows] = np.where(
            tile_valid,
            corrected_secondary,
            invalid,
        ).astype(np.complex64)
        valid[rows] = tile_valid
        topo_phase[rows] = phase.astype(np.float32)
        if lut.height_full is None:
            height_field[rows] = height
        del (
            corrected_secondary,
            coordinate_valid,
            height,
            latitude,
            longitude,
            phase,
            secondary_azimuth,
            tile_valid,
        )
        for array in (
            reference_geo,
            secondary_geo,
            valid,
            topo_phase,
        ):
            release_memmap_pages(array)
        for array in (height_field, lut.az_full, lut.rg_full, lut.valid):
            if isinstance(array, np.memmap):
                release_memmap_pages(array)
        if watchdog is not None:
            watchdog.sample(f"geo_topographic_phase:{row_start}:{row_stop}")
    reference_geo.flush()
    secondary_geo.flush()
    valid.flush()
    topo_phase.flush()
    if isinstance(height_field, np.memmap):
        height_field.flush()
    return topo_phase, height_field


def stage_coregister(
    state: ProductionPairState,
    *,
    control_spacing: int | None = None,
    esd_enabled: bool = False,
    amplitude_refinement_enabled: bool = False,
    executor: str = "torch",
    device: str = "auto",
    coregistration_grid: CoregistrationGrid = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_height_m: float = 0.0,
    geo_chunk_size: int = 128,
    geo_work_dir: str | Path | None = None,
    memory_watchdog: MemoryWatchdog | None = None,
) -> ProductionPairState:
    """Estimate dense offsets and coregister on a radar or geographic grid.

    Memory notes
    ------------
    Full-burst Lanczos resampling is chunked inside :func:`resample_complex`.
    ESD uses a cheaper bilinear pre-align (``order=1``) so the expensive
    phase-preserving Lanczos path runs only once for the final secondary.
    Offset fields and the secondary deramped buffer are released after the
    final resample so peak RSS does not stack geometry + pre + final copies.

    Parameters
    ----------
    state : ProductionPairState
        Mutable pair workflow state (requires deramp outputs).
    control_spacing : int, optional
        Stride between geometry control points. Defaults to 8 pixels on the
        radar grid for phase-accurate flattening and 64 on the geographic
        grid.
    esd_enabled : bool, optional
        Run ESD azimuth residual estimation. Default False.
    amplitude_refinement_enabled : bool, optional
        Refine the geometry offsets with a global amplitude-correlation shift.
        Disabled by default because a single TOPS burst does not provide the
        overlap constraints needed to distinguish a true residual from burst
        envelope structure.
    executor : {"torch"}, optional
        Unified Torch Lanczos path for the final resample.
    device : {"auto","cpu","cuda"}, optional
        Torch device. ``"auto"`` selects CUDA, then MPS, then CPU.
    coregistration_grid : {"radar", "geo"}, optional
        Grid on which both aligned SLCs are produced.
    geo_grid : GeoGridSpec, optional
        Native-resolution SLC grid required for geographic coregistration.
    geo_height_m : float, optional
        Fallback ellipsoidal height for geographic coregistration.
    geo_chunk_size : int, optional
        Geo2rdr row chunk size.
    geo_work_dir : str or pathlib.Path, optional
        Directory for disk-backed Geo intermediate arrays.
    memory_watchdog : MemoryWatchdog, optional
        Memory guard sampled after each geographic tile.

    """
    if state.reference_deramped is None or state.secondary_deramped is None:
        reject_invalid_state("coregister requires deramp")

    # Align secondary crop to reference shape if needed (same swath expected)
    ref = state.reference_deramped
    sec = state.secondary_deramped
    if ref.shape != sec.shape:
        # crop to common shape (min)
        h = min(ref.shape[0], sec.shape[0])
        w = min(ref.shape[1], sec.shape[1])
        ref = ref[:h, :w]
        sec = sec[:h, :w]
        state.reference_deramped = ref
        state.secondary_deramped = sec
        state.note(f"COREG cropped both scenes to common shape {(h, w)}")

    dem = state.dem
    resolved_control_spacing = (
        8
        if control_spacing is None and coregistration_grid == "radar"
        else 64
        if control_spacing is None
        else control_spacing
    )
    if resolved_control_spacing < 1:
        reject_invalid_state("control_spacing must be >= 1")
    substage_started = time.perf_counter()
    geometry_field = dense_geometry_offsets(
        shape=ref.shape,
        reference_model=state.reference.geometry,
        secondary_model=state.secondary.geometry,
        dem=dem,
        stride=resolved_control_spacing,
    )
    state.coregistration_timings_s["dense_geometry_offsets"] = (
        time.perf_counter() - substage_started
    )

    prior_rg = float(
        np.nanmedian(geometry_field.range_offset_px[geometry_field.coverage])
    )
    prior_az = float(
        np.nanmedian(geometry_field.azimuth_offset_px[geometry_field.coverage])
    )
    amp_res_rg = 0.0
    amp_res_az = 0.0
    esd_az = 0.0
    if coregistration_grid == "radar" and amplitude_refinement_enabled:
        amp_rg, amp_az = refine_shift_with_correlation(
            ref,
            sec,
            prior_rg=prior_rg,
            prior_az=prior_az,
            search_radius=32,
        )
        amp_res_rg = amp_rg - prior_rg
        amp_res_az = amp_az - prior_az
    if coregistration_grid == "radar" and esd_enabled:
        # Bilinear pre-align is sufficient for ESD spectral estimation and
        # avoids a second full-burst Lanczos pass (~minutes and peak RSS).
        pre = resample_complex(
            sec,
            range_offset_px=geometry_field.range_offset_px + amp_res_rg,
            azimuth_offset_px=geometry_field.azimuth_offset_px + amp_res_az,
            order=1,
        )
        esd = estimate_azimuth_shift_esd(ref, pre)
        esd_az = float(esd.azimuth_shift_px)
        state.esd_azimuth_shift_px = esd_az
        state.note(f"ESD az={esd_az:.4f} px coherence={esd.coherence:.3f}")
        del pre
        gc.collect()

    offsets = combine_offset_fields(
        geometry_field,
        esd_azimuth_shift_px=esd_az,
        amplitude_residual_rg=amp_res_rg,
        amplitude_residual_az=amp_res_az,
    )
    # Drop geometry-only fields once combined; offsets retains the dense maps.
    del geometry_field
    gc.collect()

    state.range_shift_px = float(
        np.nanmedian(offsets.range_offset_px[offsets.coverage])
    )
    state.azimuth_shift_px = float(
        np.nanmedian(offsets.azimuth_offset_px[offsets.coverage])
    )
    coverage = float(np.mean(offsets.coverage))
    if coregistration_grid == "geo":
        if geo_grid is None:
            reject_invalid_state("geo coregistration requires geo_grid")
        from faninsar.processing.pipeline.geo_lut import build_geo2rdr_lut
        from faninsar.processing.pipeline.geo_modes import (
            coregister_geocoded_slcs_chunked,
        )

        if geo_work_dir is None:
            reject_invalid_state("geo coregistration requires geo_work_dir")
        work_directory = Path(geo_work_dir)
        work_directory.mkdir(parents=True, exist_ok=True)
        substage_started = time.perf_counter()
        lut = build_geo2rdr_lut(
            geometry=state.reference.geometry,
            grid=geo_grid,
            full_radar_shape=ref.shape,
            height_m=geo_height_m,
            dem=state.dem,
            chunk_size=geo_chunk_size,
            storage_dir=work_directory / "lut",
        )
        state.coregistration_timings_s["geo2rdr_lut"] = (
            time.perf_counter() - substage_started
        )
        substage_started = time.perf_counter()
        reference_geo, secondary_geo, valid = coregister_geocoded_slcs_chunked(
            ref,
            sec,
            reference_carrier=state.reference.carrier,
            secondary_carrier=state.secondary.carrier,
            reference_lut=lut,
            offsets=offsets,
            output_dir=work_directory / "slc",
            row_chunk=geo_chunk_size,
            executor=executor,
            device=device,
            watchdog=memory_watchdog,
        )
        state.coregistration_timings_s["geo_slc_resample"] = (
            time.perf_counter() - substage_started
        )
        substage_started = time.perf_counter()
        topo_phase, height_field = _apply_geo_topographic_phase_chunked(
            state,
            lut,
            offsets,
            reference_geo,
            secondary_geo,
            valid,
            grid=geo_grid,
            output_dir=work_directory,
            chunk_size=geo_chunk_size,
            watchdog=memory_watchdog,
        )
        state.coregistration_timings_s["geo_topographic_phase"] = (
            time.perf_counter() - substage_started
        )
        state.reference_geocoded_slc = reference_geo
        state.secondary_geocoded_slc = secondary_geo
        state.geocoded_slc_valid = valid
        state.reference_deramped = reference_geo
        state.secondary_deramped = None
        state.secondary_aligned = secondary_geo
        state.geo2rdr_lut = lut
        state.geo_grid = geo_grid
        state.topo_phase = topo_phase
        state.geo_height_field = height_field
        state.geo_work_dir = work_directory
        state.memory_watchdog = memory_watchdog
        del ref, sec, offsets
        gc.collect()
        state.note(
            f"COREG geo grid={geo_grid.shape} median_rg={state.range_shift_px:.3f} "
            f"median_az={state.azimuth_shift_px:.3f} "
            f"coverage={float(valid.mean()):.3f} "
            "(two deramped single-remaps + fractional reramp + SLC flatten)"
        )
        state.note(f"COREG timings={state.coregistration_timings_s}")
        return state

    substage_started = time.perf_counter()
    sec_resamp = resample_complex_deramped_reramp(
        sec,
        secondary_carrier=state.secondary.carrier,
        range_offset_px=offsets.range_offset_px,
        azimuth_offset_px=offsets.azimuth_offset_px,
        executor=executor,
        device=device,
    )
    secondary_geometry = state.secondary.geometry
    phase_per_range_pixel = (
        4.0
        * np.pi
        * secondary_geometry.range_spacing_m
        / secondary_geometry.wavelength_m
    )
    for row_start in range(0, sec_resamp.shape[0], 64):
        rows = slice(row_start, min(row_start + 64, sec_resamp.shape[0]))
        range_carrier_phase = phase_per_range_pixel * offsets.range_offset_px[rows]
        sec_resamp[rows] = remove_topographic_phase(
            sec_resamp[rows],
            range_carrier_phase,
        ).astype(np.complex64, copy=False)
    state.coregistration_timings_s["radar_resample_and_flatten"] = (
        time.perf_counter() - substage_started
    )
    state.secondary_aligned_is_flattened = True
    state.range_offset_flatten_phase = (
        phase_per_range_pixel * offsets.range_offset_px
    ).astype(np.float32)
    state.secondary_deramped = None
    state.reference_deramped = reramp(ref, state.reference.carrier)
    state.secondary_aligned = sec_resamp
    del sec_resamp, sec, ref, offsets
    gc.collect()
    state.note(
        f"COREG geometry median_rg={state.range_shift_px:.3f} "
        f"median_az={state.azimuth_shift_px:.3f} "
        f"esd={esd_enabled} amplitude_refinement={amplitude_refinement_enabled} "
        f"control_spacing={resolved_control_spacing} "
        f"coverage={coverage:.3f} "
        "(deramped single-remap + fractional reramp + range-offset flatten)"
    )
    state.note(f"COREG timings={state.coregistration_timings_s}")
    return state


def stage_interferogram(
    state: ProductionPairState,
    *,
    multilook: tuple[int, int] = (4, 20),
    goldstein_alpha: float = 0.5,
    dead_pixel_amp_threshold: float = 0.0,
) -> ProductionPairState:
    """Form multilooked interferogram and apply Goldstein filter.

    Parameters
    ----------
    state : ProductionPairState
        Pair state after coregistration.
    multilook : tuple[int, int], optional
        Non-overlapping azimuth and range look factors.
    goldstein_alpha : float, optional
        Goldstein filter exponent.
    dead_pixel_amp_threshold : float, optional
        SLC amplitude below which a pixel is excluded from the multilook
        average (dead-pixel masking). Set to 0 to disable. The production
        entry point :func:`run_production_pair` passes 3.0 for real S1
        data; synthetic tests use the default 0.

    """
    if state.reference_deramped is None or state.secondary_aligned is None:
        reject_invalid_state("interferogram requires coregister")
    ifg = form_interferogram(
        state.reference_deramped,
        state.secondary_aligned,
        multilook=multilook,
        dead_pixel_amp_threshold=dead_pixel_amp_threshold,
    )
    state.reference_deramped = None
    state.secondary_aligned = None
    if state.coregistration_grid == "geo" and state.geo_work_dir is not None:
        for array in (
            state.reference_geocoded_slc,
            state.secondary_geocoded_slc,
            state.geocoded_slc_valid,
        ):
            if isinstance(array, np.memmap):
                close_memmap(array)
        state.reference_geocoded_slc = None
        state.secondary_geocoded_slc = None
        state.geocoded_slc_valid = None
        if state.geo2rdr_lut is not None:
            for array in (
                state.geo2rdr_lut.az_full,
                state.geo2rdr_lut.rg_full,
                state.geo2rdr_lut.valid,
            ):
                if isinstance(array, np.memmap):
                    close_memmap(array)
            state.geo2rdr_lut = None
        if isinstance(state.topo_phase, np.memmap):
            close_memmap(state.topo_phase)
            state.topo_phase = None
    gc.collect()
    # alpha=0 is not a no-op in the windowed Goldstein implementation (Hann
    # taper still smooths). Skip the filter entirely so high-rate geometric
    # fringes stay intact for flattening.
    if goldstein_alpha > 0.0:
        complex_ifg = goldstein_filter(ifg.complex_ifg, alpha=goldstein_alpha)
    else:
        complex_ifg = ifg.complex_ifg
    complex_ifg, coherence, wrapped = mask_invalid_looks(complex_ifg, ifg.coherence)
    state.complex_ifg = complex_ifg
    state.coherence = coherence if coherence is not None else ifg.coherence
    state.wrapped_phase = wrapped
    n_invalid = int(np.isnan(state.wrapped_phase).sum())
    state.note(
        f"IFG multilook={multilook} goldstein={goldstein_alpha} "
        f"mean_coh={float(np.nanmean(state.coherence)):.3f} "
        f"invalid_looks={n_invalid}"
    )
    return state


def stage_flatten(state: ProductionPairState) -> ProductionPairState:
    """Remove topographic phase using DEM + dual-orbit geometry.

    When radar coreg already applied a *range-offset* phase screen on the
    secondary SLC (``secondary_aligned_is_flattened=True``), that screen is only
    a first-order approximation of the path-length topographic phase.  Residual
    DEM topography — especially where dense geometry offsets are incomplete
    (DEM edge, outer subswath) — is still removed here with the full dual-orbit
    geometric model.  If the range-offset screen already matched the DEM model,
    the residual is near zero and this step is a no-op in practice.
    """
    if state.complex_ifg is None:
        reject_invalid_state("flatten requires interferogram")
    height, width = state.complex_ifg.shape
    # Multilooked grid → full-res radar indices on the geometry model.
    # Geometry (0,0) is the array origin (already cropped). Use look-window
    # centres rather than leading edges so path-length phase matches the
    # multilooked ifg sampling.
    full_h, full_w = state.reference.array.samples.shape
    az_looks = max(full_h // max(height, 1), 1)
    rg_looks = max(full_w // max(width, 1), 1)
    az_full = (np.arange(height, dtype=np.float64) + 0.5) * az_looks - 0.5
    rg_full = (np.arange(width, dtype=np.float64) + 0.5) * rg_looks - 0.5
    az_grid, rg_grid = np.meshgrid(az_full, rg_full, indexing="ij")

    def _topographic_phase() -> tuple[np.ndarray, float, float, float]:
        """Compute the full DEM topo phase and its spread (lazy, costly)."""
        topo = compute_topographic_phase(
            state.reference.geometry,
            state.secondary.geometry,
            az_grid,
            rg_grid,
            state.dem,
        )
        topo_finite = topo[np.isfinite(topo)]
        topo_valid_frac = float(np.isfinite(topo).mean())
        if topo_finite.size:
            rms = float(np.std(topo_finite))
            span = float(np.max(topo_finite) - np.min(topo_finite))
        else:
            rms, span = float("nan"), float("nan")
        return topo, topo_valid_frac, rms, span

    if state.secondary_aligned_is_flattened:
        # ISCE2's burstifg flattens the interferogram with the full-resolution
        # range-offset screen only (fact * range_offset); no second DEM phase
        # removal is applied.  The residual ``topo + range_offset_phase``
        # computed on the multilooked grid is nonzero only because the
        # geometric topo model evaluates the secondary orbit at the reference
        # azimuth, which does not match the resampled source azimuth; removing
        # it corrupts the flattened phase (observed ~0.6 rad on real S1 pairs).
        if state.range_offset_flatten_phase is not None:
            residual_span = 0.0
            screen = state.range_offset_flatten_phase
            screen_std = (
                float(np.std(screen)) if np.isfinite(screen).all() else float("nan")
            )
            state.note(
                f"FLATTEN range-offset screen only (ISCE2 parity); "
                f"screen_std={screen_std:.3f} residual_span={residual_span:.3f} rad"
            )
            flat = state.complex_ifg
            state.topo_phase = np.zeros((height, width), dtype=np.float32)
            # Do NOT remove a residual azimuth ramp here: ISCE2 does not apply
            # one in its flatten step, and the estimated ramp was found to
            # introduce ~2 rad of spurious phase on IW3_b0.
        else:
            topo, topo_valid_frac, rms, span = _topographic_phase()
            # Fallback: no stored range-offset phase, use scale estimation.
            scale, residual_rms = estimate_residual_topographic_scale(
                state.complex_ifg,
                topo,
                coherence=state.coherence,
            )
            if abs(scale) > 1e-3 and topo_valid_frac > 0.05:
                residual_topo = (scale * topo).astype(np.float64)
                flat = remove_topographic_phase(state.complex_ifg, residual_topo)
                state.note(
                    f"FLATTEN residual DEM topo after range-offset "
                    f"scale={scale:.3f} model_rms={rms:.3f} "
                    f"residual_rms={residual_rms:.3f} "
                    f"topo_valid_frac={topo_valid_frac:.3f}"
                )
            else:
                flat = state.complex_ifg
                state.note(
                    f"FLATTEN range-offset only (no residual DEM; "
                    f"scale={scale:.3f} residual_rms={residual_rms:.3f} "
                    f"topo_valid_frac={topo_valid_frac:.3f})"
                )
            state.topo_phase = (
                np.asarray(scale * topo, dtype=np.float32)
                if abs(scale) > 1e-3
                else np.zeros_like(topo, dtype=np.float32)
            )
            if topo_valid_frac > 0.05:
                az_ramp = estimate_residual_azimuth_ramp(
                    flat,
                    topo,
                    coherence=state.coherence,
                )
                if abs(az_ramp) > 1e-6:
                    flat = remove_azimuth_phase_ramp(flat, az_ramp)
                    state.note(f"FLATTEN residual_az_ramp={az_ramp:.5f} rad/az_sample")
        flat_masked, coh_flat, wrapped_flat = mask_invalid_looks(
            np.asarray(flat, dtype=np.complex64), state.coherence
        )
        state.complex_ifg_flat = flat_masked
        if coh_flat is not None:
            state.coherence = coh_flat
        state.wrapped_phase = wrapped_flat
        return state

    # Residual Doppler / differential TOPS carrier leaves a near-linear
    # azimuth phase ramp on the original-domain ifg. Estimate it against the
    # path-length geometric model and remove it from the unflattened product.
    topo, topo_valid_frac, rms, span = _topographic_phase()
    az_ramp = estimate_residual_azimuth_ramp(
        state.complex_ifg,
        topo,
        coherence=state.coherence,
    )
    if abs(az_ramp) > 1e-6:
        state.complex_ifg = remove_azimuth_phase_ramp(state.complex_ifg, az_ramp)
        state.note(f"FLATTEN residual_az_ramp={az_ramp:.5f} rad/az_sample")

    flat = remove_topographic_phase(state.complex_ifg, topo)
    state.topo_phase = topo.astype(np.float32)
    # Keep invalid looks as NaN through ramp/topo multiply (0·e^{iφ}=0 would
    # otherwise repaint a solid phase=0 black edge on the burst margin).
    # Remask the ramp-corrected unflattened ifg for intermediate products.
    state.complex_ifg, state.coherence, _ = mask_invalid_looks(
        state.complex_ifg, state.coherence
    )
    # wrapped_phase must track the flattened product that stage_write archives
    # as complex_ifg (complex_ifg_flat), not the unflattened angle.
    flat_masked, coh_flat, wrapped_flat = mask_invalid_looks(
        flat.astype(np.complex64, copy=False), state.coherence
    )
    state.complex_ifg_flat = flat_masked
    if coh_flat is not None:
        state.coherence = coh_flat
    state.wrapped_phase = wrapped_flat
    state.note(
        f"FLATTEN geometric_phase rms={rms:.3f} rad span={span:.1f} rad "
        f"topo_valid_frac={topo_valid_frac:.3f}"
    )
    return state


def stage_unwrap(
    state: ProductionPairState,
    *,
    method: UnwrapBackend = "snaphu",
    config: SnaphuConfig | None = None,
    irls_kwargs: dict[str, Any] | None = None,
) -> ProductionPairState:
    """Unwrap the flattened interferogram with the selected backend.

    Parameters
    ----------
    state : ProductionPairState
        Pair state containing a flattened interferogram and coherence.
    method : {"irls", "snaphu"}, optional
        Spatial unwrapping backend.
    config : SnaphuConfig, optional
        SNAPHU configuration when ``method="snaphu"``.
    irls_kwargs : dict, optional
        Arguments forwarded to the IRLS backend.

    Returns
    -------
    ProductionPairState
        Updated state containing unwrapped phase and component labels.

    """
    ifg = (
        state.complex_ifg_flat
        if state.complex_ifg_flat is not None
        else state.complex_ifg
    )
    if ifg is None or state.coherence is None:
        reject_invalid_state("unwrap requires interferogram")
    result: CommonUnwrapResult = unwrap_dispatch(
        ifg,
        state.coherence,
        method=method,
        snaphu_config=config,
        irls_kwargs=irls_kwargs,
    )
    state.unwrapped_phase = result.unwrapped_phase
    state.connected_components = result.connected_components
    state.unwrap_method = result.method
    state.note(f"UNWRAP method={result.method} metrics={result.metrics}")

    # Non-DEM residual range/azimuth poly (orbit residual / APS / far-range).
    # Non-DEM residual range/azimuth poly (orbit residual / APS / far-range).
    # Fit on unwrapped phase (stable for multi-fringe), apply to both unwrap
    # and the flattened complex product used by merge.
    from faninsar.processing.interferometry.flatten import (
        apply_residual_phase_screen_to_products,
    )

    flat_src = (
        state.complex_ifg_flat
        if state.complex_ifg_flat is not None
        else state.complex_ifg
    )
    unw_corr, z_corr, _screen, span, applied = apply_residual_phase_screen_to_products(
        unwrapped_phase=state.unwrapped_phase,
        complex_ifg=flat_src,
        coherence=state.coherence,
        range_degree=2,
        azimuth_degree=1,
        min_span_rad=1.0,
        max_span_rad=40.0,
        connected_components=state.connected_components,
    )
    if applied:
        state.unwrapped_phase = np.asarray(unw_corr, dtype=np.float32)
        if z_corr is not None:
            state.complex_ifg_flat = np.asarray(z_corr, dtype=np.complex64)
            state.wrapped_phase = np.angle(state.complex_ifg_flat).astype(np.float32)
        state.note(
            f"UNWRAP residual phase screen span={span:.2f} rad rg_deg=2 az_deg=1"
        )
    elif span >= 1.0:
        state.note(f"UNWRAP residual phase screen skipped span={span:.2f} rad")
    return state


def stage_baseline(state: ProductionPairState) -> ProductionPairState:
    """Compute geometric baseline at burst mid-time for metadata."""
    mid_az = 0.5 * (state.reference.array.samples.shape[0] - 1)
    ref_time = state.reference.geometry.azimuth_time(mid_az)
    sec_time = state.secondary.geometry.azimuth_time(mid_az)
    from faninsar.processing.geometry.orbit import OrbitInterpolator

    ref_interp = OrbitInterpolator.from_orbit(state.reference.swath.orbit)
    sec_interp = OrbitInterpolator.from_orbit(state.secondary.swath.orbit)
    state_ref = ref_interp.evaluate(ref_time)
    state_sec = sec_interp.evaluate(sec_time)
    sat = np.asarray(state_ref.position_m, dtype=np.float64)
    look = -sat / np.linalg.norm(sat)
    baseline_vec = np.asarray(state_sec.position_m, dtype=np.float64) - sat
    parallel = float(np.dot(baseline_vec, look))
    perp_vec = baseline_vec - parallel * look
    state.baseline = BaselineComponents(
        parallel_m=parallel,
        perpendicular_m=float(np.linalg.norm(perp_vec)),
        magnitude_m=float(np.linalg.norm(baseline_vec)),
    )
    state.note(
        f"BASELINE parallel={state.baseline.parallel_m:.1f} m "
        f"perp={state.baseline.perpendicular_m:.1f} m"
    )
    return state


def stage_geocode(
    state: ProductionPairState,
    *,
    chunk_size: tuple[int, int] = (256, 256),
) -> ProductionPairState:
    """Geocode unwrapped phase and coherence with chunked vectorized rdr2geo."""
    if state.unwrapped_phase is None or state.coherence is None:
        reject_invalid_state("geocode requires unwrap")
    height, width = state.unwrapped_phase.shape
    full_h, full_w = state.reference.array.samples.shape
    az_scale = full_h / max(height, 1)
    rg_scale = full_w / max(width, 1)
    az = np.arange(height, dtype=np.float64) * az_scale
    rg = np.arange(width, dtype=np.float64) * rg_scale
    az_grid, rg_grid = np.meshgrid(az, rg, indexing="ij")
    transform = rdr2geo_with_dem_chunked(
        state.reference.geometry,
        az_grid,
        rg_grid,
        state.dem,
        chunk_size=chunk_size,
    )
    unw = np.where(transform.converged, state.unwrapped_phase, np.nan)
    coh = np.where(transform.converged, state.coherence, np.nan)
    state.geocoded = {
        "unwrapped_phase": unw.astype(np.float32),
        "coherence": coh.astype(np.float32),
        "latitude_deg": transform.latitude_deg.astype(np.float64),
        "longitude_deg": transform.longitude_deg.astype(np.float64),
        "height_m": transform.height_m.astype(np.float64),
        "converged": transform.converged.astype(np.uint8),
    }
    n_ok = int(np.count_nonzero(transform.converged))
    state.note(f"GEOCODE converged={n_ok}/{transform.converged.size}")
    return state


def stage_write(
    state: ProductionPairState, output_dir: str | Path
) -> ProductionPairState:
    """Write radar products, geocoded layers, baselines, and STAC with bbox."""
    if (
        state.complex_ifg is None
        or state.coherence is None
        or state.wrapped_phase is None
        or state.unwrapped_phase is None
        or state.connected_components is None
    ):
        reject_invalid_state("write requires completed products")

    try:
        temporal_days = float(
            (
                state.secondary.burst.azimuth_time - state.reference.burst.azimuth_time
            ).total_seconds()
            / 86400.0
        )
    except (TypeError, AttributeError):
        temporal_days = float("nan")
    meta: dict[str, Any] = {
        "unwrap_method": state.unwrap_method,
        "coregistration_grid": state.coregistration_grid,
        "dem_id": state.dem_id,
        "multilook": list(state.multilook),
        "goldstein_alpha": state.goldstein_alpha,
        "coreg_executor": state.coreg_executor,
        "coreg_device": state.coreg_device,
        "lanczos_a": 4,
        "geo_grid": state.geo_grid_meta,
        "range_shift_px": state.range_shift_px,
        "azimuth_shift_px": state.azimuth_shift_px,
        "esd_azimuth_shift_px": state.esd_azimuth_shift_px,
        "reference_scene": state.reference.scene_id,
        "secondary_scene": state.secondary.scene_id,
        "swath": state.reference.swath.swath,
        "burst_index": state.reference.burst.index,
        "shape": list(state.unwrapped_phase.shape),
        "full_burst_shape": list(state.reference.array.samples.shape),
        "temporal_baseline_days": temporal_days,
        "stages": list(state.log),
        "stage_timings_s": dict(state.stage_timings_s),
        "coregistration_timings_s": dict(state.coregistration_timings_s),
    }
    if state.baseline is not None:
        meta["baseline_parallel_m"] = state.baseline.parallel_m
        meta["baseline_perpendicular_m"] = state.baseline.perpendicular_m
        meta["baseline_magnitude_m"] = state.baseline.magnitude_m

    product = PairProductArrays(
        pair_id=state.pair_id,
        complex_ifg=(
            state.complex_ifg_flat
            if state.complex_ifg_flat is not None
            else state.complex_ifg
        ),
        coherence=state.coherence,
        wrapped_phase=state.wrapped_phase,
        unwrapped_phase=state.unwrapped_phase,
        connected_components=state.connected_components,
        metadata=meta,
    )
    out = Path(output_dir)
    zarr_path = write_pair_zarr(product, out / f"{state.pair_id}.zarr")
    reference_slc = state.reference_geocoded_slc
    secondary_slc = state.secondary_geocoded_slc
    slc_valid = state.geocoded_slc_valid
    transform_azimuth = None if state.geo2rdr_lut is None else state.geo2rdr_lut.az_full
    transform_range = None if state.geo2rdr_lut is None else state.geo2rdr_lut.rg_full
    transform_valid = None if state.geo2rdr_lut is None else state.geo2rdr_lut.valid
    topo_phase = state.topo_phase
    reopened_from_work = state.geo_work_dir is not None and state.geo_grid is not None
    if reopened_from_work:
        assert state.geo_work_dir is not None
        assert state.geo_grid is not None
        shape = state.geo_grid.shape
        reference_slc = np.memmap(
            state.geo_work_dir / "slc" / "reference_geo.complex64",
            mode="r",
            dtype=np.complex64,
            shape=shape,
        )
        secondary_slc = np.memmap(
            state.geo_work_dir / "slc" / "secondary_geo.complex64",
            mode="r",
            dtype=np.complex64,
            shape=shape,
        )
        slc_valid = np.memmap(
            state.geo_work_dir / "slc" / "geo_valid.bool",
            mode="r",
            dtype=np.bool_,
            shape=shape,
        )
        transform_azimuth = np.memmap(
            state.geo_work_dir / "lut" / "reference_azimuth.float64",
            mode="r",
            dtype=np.float64,
            shape=shape,
        )
        transform_range = np.memmap(
            state.geo_work_dir / "lut" / "reference_range.float64",
            mode="r",
            dtype=np.float64,
            shape=shape,
        )
        transform_valid = np.memmap(
            state.geo_work_dir / "lut" / "reference_valid.bool",
            mode="r",
            dtype=np.bool_,
            shape=shape,
        )
        topo_phase = np.memmap(
            state.geo_work_dir / "topographic_phase.float32",
            mode="r",
            dtype=np.float32,
            shape=shape,
        )
    if (
        reference_slc is not None
        and secondary_slc is not None
        and state.geo_grid is not None
    ):
        import zarr

        root = zarr.open_group(str(zarr_path), mode="a")
        slc = root.require_group("slc")
        slc.create_array(
            "reference",
            data=reference_slc,
            overwrite=True,
        )
        if reopened_from_work and isinstance(reference_slc, np.memmap):
            close_memmap(reference_slc)
        slc.create_array(
            "secondary",
            data=secondary_slc,
            overwrite=True,
        )
        if reopened_from_work and isinstance(secondary_slc, np.memmap):
            close_memmap(secondary_slc)
        if slc_valid is not None:
            slc.create_array(
                "valid",
                data=slc_valid.astype(np.uint8),
                overwrite=True,
            )
            if reopened_from_work and isinstance(slc_valid, np.memmap):
                close_memmap(slc_valid)
        x0, dx, _, y0, _, dy = state.geo_grid.transform
        x = x0 + dx * (0.5 + np.arange(state.geo_grid.width, dtype=np.float64))
        y = y0 + dy * (0.5 + np.arange(state.geo_grid.height, dtype=np.float64))
        slc.create_array("x", data=x, overwrite=True)
        slc.create_array("y", data=y, overwrite=True)
        slc.attrs.update(
            {
                "crs": state.geo_grid.crs,
                "transform": list(state.geo_grid.transform),
                "phase_domain": "reramped_flattened",
            }
        )
        if (
            transform_azimuth is not None
            and transform_range is not None
            and transform_valid is not None
        ):
            transform = root.require_group("transform")
            transform.create_array(
                "reference_azimuth_index",
                data=transform_azimuth,
                overwrite=True,
            )
            if reopened_from_work and isinstance(transform_azimuth, np.memmap):
                close_memmap(transform_azimuth)
            transform.create_array(
                "reference_range_index",
                data=transform_range,
                overwrite=True,
            )
            if reopened_from_work and isinstance(transform_range, np.memmap):
                close_memmap(transform_range)
            transform.create_array(
                "valid",
                data=transform_valid.astype(np.uint8),
                overwrite=True,
            )
            if reopened_from_work and isinstance(transform_valid, np.memmap):
                close_memmap(transform_valid)
    if state.geocoded is not None:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="a")
        geo = root.require_group("geocoded")
        for name, array in state.geocoded.items():
            geo.create_array(name, data=array, overwrite=True)
        if topo_phase is not None:
            root.create_array("topo_phase", data=topo_phase, overwrite=True)
            if reopened_from_work and isinstance(topo_phase, np.memmap):
                close_memmap(topo_phase)

    # STAC with bbox from geocoded lon/lat when available
    stac_path = out / f"{state.pair_id}.json"
    if state.geocoded is not None:
        conv = state.geocoded["converged"].astype(bool)
        if np.any(conv):
            lats = state.geocoded["latitude_deg"][conv]
            lons = state.geocoded["longitude_deg"][conv]
            bbox = [
                float(np.min(lons)),
                float(np.min(lats)),
                float(np.max(lons)),
                float(np.max(lats)),
            ]
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ],
            }
            # write custom STAC with geometry
            import json
            from datetime import UTC, datetime

            item = {
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": state.pair_id,
                "geometry": geometry,
                "bbox": bbox,
                "properties": {
                    "datetime": datetime.now(UTC).isoformat(),
                    "faninsar:pair_id": state.pair_id,
                    **{
                        f"faninsar:{k}": (
                            v
                            if isinstance(v, (str, int, float, bool, list, type(None)))
                            else str(v)
                        )
                        for k, v in meta.items()
                        if k != "stages"
                    },
                },
                "assets": {
                    "zarr": {
                        "href": str(zarr_path),
                        "type": "application/vnd+zarr",
                        "roles": ["data"],
                    }
                },
                "links": [],
                "stac_extensions": [],
            }
            stac_path.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
        else:
            write_pair_stac_item(product, zarr_path, stac_path)
    else:
        write_pair_stac_item(product, zarr_path, stac_path)

    state.zarr_path = zarr_path
    state.stac_path = stac_path
    if state.memory_watchdog is not None:
        state.memory_watchdog.sample("write:complete")
    if state.geo_work_dir is not None:
        work_directory = state.geo_work_dir
        state.reference_geocoded_slc = None
        state.secondary_geocoded_slc = None
        state.geocoded_slc_valid = None
        state.geo2rdr_lut = None
        state.geo_height_field = None
        state.topo_phase = None
        state.geo_work_dir = None
        gc.collect()
        shutil.rmtree(work_directory)
    state.note(f"WRITE {zarr_path}")
    return state


def _dem_id(dem: DEMSampler) -> str:
    """Stable DEM identity string for product provenance."""
    name = type(dem).__name__
    path = getattr(dem, "path", None)
    if path is not None:
        return f"{name}:{Path(path)}"
    height = getattr(dem, "height_m", None)
    if height is not None:
        return f"{name}:{float(height)}"
    return name


def _geo_grid_meta(geo_grid: GeoGridSpec | None) -> dict[str, Any] | None:
    """Serialize GeoGridSpec fields for Zarr/STAC metadata."""
    if geo_grid is None:
        return None
    return {
        "crs": geo_grid.crs,
        "transform": list(geo_grid.transform),
        "width": geo_grid.width,
        "height": geo_grid.height,
        "resolution_m": list(geo_grid.resolution_m),
    }


def _multilooked_geo_grid(
    grid: GeoGridSpec,
    multilook: tuple[int, int],
) -> GeoGridSpec:
    """Return the pixel-centre-aligned product grid after block multilooking."""
    from faninsar.processing.merge.grid import GeoGridSpec

    azimuth_looks, range_looks = multilook
    x0, dx, x_skew, y0, y_skew, dy = grid.transform
    return GeoGridSpec(
        crs=grid.crs,
        transform=(x0, dx * range_looks, x_skew, y0, y_skew, dy * azimuth_looks),
        width=grid.width // range_looks,
        height=grid.height // azimuth_looks,
        resolution_m=(
            grid.resolution_m[0] * range_looks,
            grid.resolution_m[1] * azimuth_looks,
        ),
    )


def _multilook_real_field(
    field: np.ndarray,
    multilook: tuple[int, int],
) -> np.ndarray:
    """Average a real geographic field over the SLC multilook blocks."""
    azimuth_looks, range_looks = multilook
    height = field.shape[0] // azimuth_looks * azimuth_looks
    width = field.shape[1] // range_looks * range_looks
    return np.nanmean(
        field[:height, :width].reshape(
            height // azimuth_looks,
            azimuth_looks,
            width // range_looks,
            range_looks,
        ),
        axis=(1, 3),
    )


def run_production_pair(
    reference_path: str | Path,
    secondary_path: str | Path,
    *,
    output_dir: str | Path,
    swath: str = "IW1",
    scope: ScopeMode = "burst",
    burst_index: int = 0,
    dem: DEMSampler | None = None,
    multilook: tuple[int, int] = (4, 20),
    goldstein_alpha: float = 0.5,
    snaphu_config: SnaphuConfig | None = None,
    unwrap_method: UnwrapBackend | None = None,
    irls_kwargs: dict[str, Any] | None = None,
    esd_enabled: bool = False,
    amplitude_refinement_enabled: bool = False,
    control_spacing: int | None = None,
    executor: str = "torch",
    device: str = "auto",
    coregistration_grid: CoregistrationGrid = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_height_m: float = 0.0,
    geo_chunk_size: int = 128,
    reference_orbit_path: str | Path | None = None,
    secondary_orbit_path: str | Path | None = None,
    geoid_correction: bool = True,
) -> ProductionPairState:
    """Run the full production pair chain on full burst or full swath.

    ``coregistration_grid="radar"`` aligns the secondary SLC to the reference
    radar grid and defers geocoding. ``coregistration_grid="geo"`` remaps both
    deramped SLCs once onto ``geo_grid`` and analytically reramps them there.

    Parameters
    ----------
    reference_path, secondary_path : path
        SAFE products.
    output_dir : path
        Output directory.
    swath : str, optional
        Sub-swath.
    scope : {"burst", "swath"}, optional
        Process one full burst or stitch entire sub-swath.
    burst_index : int, optional
        Burst index when ``scope="burst"``.
    dem : DEMSampler, optional
        DEM for coreg/flatten/geocode. Defaults to zero ellipsoid.
    multilook : tuple[int, int], optional
        ``(az, rg)`` looks (radar path).
    goldstein_alpha : float, optional
        Goldstein filter exponent (default 0.5).
    snaphu_config : SnaphuConfig, optional
        snaphu-py configuration. By default, ``nlooks`` is the product of the
        azimuth and range multilook factors.
    unwrap_method : {"irls", "snaphu"}, optional
        Spatial unwrap backend. Defaults to ``"irls"`` for geo coregistration
        and ``"snaphu"`` for radar coregistration.
    irls_kwargs : dict, optional
        Arguments forwarded to IRLS. The pipeline device is used by default.
    esd_enabled : bool, optional
        Enable spectral diversity azimuth residual.
    amplitude_refinement_enabled : bool, optional
        Enable a global amplitude-correlation residual shift after geometric
        coregistration. Disabled by default.
    control_spacing : int, optional
        Geometry control-point spacing. Defaults to 8 pixels for radar mode
        and 64 pixels for geo mode.
    executor : {"torch"}, optional
        Unified Torch Lanczos path for coregistration and geo-grid interpolation.
    device : {"auto","cpu","cuda","mps"}, optional
        Torch device. ``"auto"`` selects CUDA, then MPS, then CPU.
    coregistration_grid : {"radar", "geo"}, optional
        Coordinate grid on which the two SLCs are coregistered.
    geo_grid : GeoGridSpec, optional
        Native-resolution geocoded SLC grid required for ``"geo"``.
    geo_height_m : float, optional
        Fallback constant height (m) when DEM is unavailable for geo2rdr.
        A real DEM should be supplied for production. Default 0.
    geo_chunk_size : int, optional
        Row chunk shared by geo2rdr, SLC remapping, and geographic flattening.
        Default 128 to amortize Dask graph and disk-backed array overhead while
        retaining bounded memory use. Larger values can improve throughput on
        hosts with more available memory.
    reference_orbit_path, secondary_orbit_path : path, optional
        Precise ESA EOF orbits for the two acquisitions.
    geoid_correction : bool, optional
        Convert orthometric raster DEM heights to ellipsoidal heights with
        the EGM96 geoid (matching ISCE2) before geometry processing.
        Default True. Ignored for constant-height or already-corrected DEMs.

    Returns
    -------
    ProductionPairState
        Full state with products and stage log.

    """
    from faninsar.processing.pipeline.geo_lut import grid_lonlat

    dem_sampler: DEMSampler = dem if dem is not None else ConstantHeightDEM(0.0)
    if geoid_correction and isinstance(dem_sampler, RasterDEM):
        from faninsar.processing.geometry.egm96 import EGM96Geoid

        dem_sampler = GeoidAdjustedDEM(dem_sampler, EGM96Geoid())
    if coregistration_grid == "geo" and geo_grid is None:
        reject_invalid_state("coregistration_grid='geo' requires geo_grid")
    if coregistration_grid == "geo" and scope != "burst":
        reject_invalid_state(
            f"coregistration_grid='geo' requires scope='burst'; got scope={scope!r}"
        )

    total_started = time.perf_counter()
    stage_started = time.perf_counter()
    reference = load_production_scene(
        reference_path,
        swath=swath,
        scope=scope,
        burst_index=burst_index,
        dem=dem_sampler,
        orbit_path=reference_orbit_path,
        coregistration_grid=coregistration_grid,
    )
    load_reference_s = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    secondary = load_production_scene(
        secondary_path,
        swath=swath,
        scope=scope,
        burst_index=burst_index,
        dem=dem_sampler,
        orbit_path=secondary_orbit_path,
        coregistration_grid=coregistration_grid,
    )
    load_secondary_s = time.perf_counter() - stage_started
    resolved_unwrap_method: UnwrapBackend = (
        unwrap_method
        if unwrap_method is not None
        else ("irls" if coregistration_grid == "geo" else "snaphu")
    )
    state = ProductionPairState(
        pair_id=f"{reference.scene_id}_{secondary.scene_id}",
        reference=reference,
        secondary=secondary,
        dem=dem_sampler,
        coregistration_grid=coregistration_grid,
        dem_id=_dem_id(dem_sampler),
        coreg_executor=str(executor),
        coreg_device=str(device),
        multilook=tuple(int(x) for x in multilook),
        goldstein_alpha=float(goldstein_alpha),
        unwrap_method=resolved_unwrap_method,
        geo_grid_meta=_geo_grid_meta(geo_grid),
        stage_timings_s={
            "load_reference": load_reference_s,
            "load_secondary": load_secondary_s,
        },
    )
    _geo_temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    geo_work_directory: Path | None = None
    memory_watchdog: MemoryWatchdog | None = None
    if coregistration_grid == "geo":
        from faninsar.processing.memory import MemoryWatchdog

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _geo_temporary_directory = tempfile.TemporaryDirectory(
            prefix=f".{state.pair_id}-geo-",
            dir=output_path,
        )
        geo_work_directory = Path(_geo_temporary_directory.name)
        memory_watchdog = MemoryWatchdog.for_current_system(
            record_path=output_path / f"{state.pair_id}.memory.jsonl"
        )
        state.geo_work_dir = geo_work_directory
        state.memory_watchdog = memory_watchdog
        memory_watchdog.sample("geo_pipeline:start")
    state.note(f"START scope={scope} coregistration_grid={coregistration_grid}")
    stage_started = time.perf_counter()
    state = stage_deramp(state)
    state.stage_timings_s["deramp"] = time.perf_counter() - stage_started
    if memory_watchdog is not None:
        memory_watchdog.sample(
            "deramp:complete",
            wall_s=state.stage_timings_s["deramp"],
        )
    stage_started = time.perf_counter()
    state = stage_coregister(
        state,
        control_spacing=control_spacing,
        esd_enabled=esd_enabled,
        amplitude_refinement_enabled=amplitude_refinement_enabled,
        executor=executor,
        device=device,
        coregistration_grid=coregistration_grid,
        geo_grid=geo_grid,
        geo_height_m=geo_height_m,
        geo_chunk_size=geo_chunk_size,
        geo_work_dir=geo_work_directory,
        memory_watchdog=memory_watchdog,
    )
    state.stage_timings_s["coregister"] = time.perf_counter() - stage_started
    if memory_watchdog is not None:
        memory_watchdog.sample(
            "coregister:complete",
            wall_s=state.stage_timings_s["coregister"],
        )
    stage_started = time.perf_counter()
    state = stage_interferogram(
        state,
        multilook=multilook,
        goldstein_alpha=goldstein_alpha,
        dead_pixel_amp_threshold=3.0,
    )
    state.stage_timings_s["interferogram"] = time.perf_counter() - stage_started
    if memory_watchdog is not None:
        memory_watchdog.sample(
            "interferogram:complete",
            wall_s=state.stage_timings_s["interferogram"],
        )
    stage_started = time.perf_counter()
    if coregistration_grid == "radar":
        state = stage_flatten(state)
    else:
        state.complex_ifg_flat = state.complex_ifg
        state.note("FLATTEN applied to secondary geocoded SLC before IFG formation")
    state.stage_timings_s["flatten"] = time.perf_counter() - stage_started
    if resolved_unwrap_method == "snaphu" and snaphu_config is None:
        snaphu_config = SnaphuConfig(nlooks=float(multilook[0] * multilook[1]))
    resolved_irls_kwargs = dict(irls_kwargs or {})
    if resolved_unwrap_method == "irls":
        resolved_irls_kwargs.setdefault("device", device)
    stage_started = time.perf_counter()
    state = stage_unwrap(
        state,
        method=resolved_unwrap_method,
        config=snaphu_config,
        irls_kwargs=resolved_irls_kwargs,
    )
    state.stage_timings_s["unwrap"] = time.perf_counter() - stage_started
    if memory_watchdog is not None:
        memory_watchdog.sample(
            "unwrap:complete",
            wall_s=state.stage_timings_s["unwrap"],
        )
    stage_started = time.perf_counter()
    state = stage_baseline(state)
    state.stage_timings_s["baseline"] = time.perf_counter() - stage_started
    if coregistration_grid == "geo":
        assert geo_grid is not None
        assert state.unwrapped_phase is not None
        assert state.coherence is not None
        assert state.wrapped_phase is not None
        product_grid = _multilooked_geo_grid(geo_grid, multilook)
        lat, lon = grid_lonlat(product_grid)
        converged = np.isfinite(state.unwrapped_phase).astype(np.uint8)
        height_field = state.geo_height_field
        if height_field is None:
            height_field = np.full(
                product_grid.shape,
                float(geo_height_m),
                dtype=np.float64,
            )
        elif height_field.shape != product_grid.shape:
            height_field = _multilook_real_field(height_field, multilook)
        state.geocoded = {
            "unwrapped_phase": np.asarray(state.unwrapped_phase, dtype=np.float32),
            "coherence": np.asarray(state.coherence, dtype=np.float32),
            "wrapped_phase": np.asarray(state.wrapped_phase, dtype=np.float32),
            "latitude_deg": lat.astype(np.float64),
            "longitude_deg": lon.astype(np.float64),
            "height_m": np.asarray(height_field, dtype=np.float64),
            "converged": converged,
        }
        if isinstance(state.geo_height_field, np.memmap):
            close_memmap(state.geo_height_field)
            state.geo_height_field = None
        state.note("GEO products complete on multilooked geographic grid")
    stage_started = time.perf_counter()
    state = stage_write(state, output_dir)
    state.stage_timings_s["write"] = time.perf_counter() - stage_started
    state.stage_timings_s["total"] = time.perf_counter() - total_started
    state.note("DONE")
    return state


def run_production_swath(
    reference_path: str | Path,
    secondary_path: str | Path,
    *,
    output_dir: str | Path,
    swath: str = "IW1",
    dem: DEMSampler | None = None,
    multilook: tuple[int, int] = (2, 10),
    goldstein_alpha: float = 0.5,
    dead_pixel_amp_threshold: float = 3.0,
    esd_enabled: bool = False,
    amplitude_refinement_enabled: bool = False,
    control_spacing: int | None = None,
    executor: str = "torch",
    device: str = "auto",
    reference_orbit_path: str | Path | None = None,
    secondary_orbit_path: str | Path | None = None,
    geoid_correction: bool = True,
    burst_indices: list[int] | None = None,
    frame_azimuth_origin: datetime | None = None,
    frame_range_offset_fullres: int = 0,
) -> ProductionPairState:
    """Process a full swath burst-by-burst and merge into one radar product.

    Sentinel-1 TOPS bursts each carry their own Doppler/FM-rate carrier, so
    a swath must be processed per burst (deramp -> coregister -> full-res
    interferogram -> flatten) and the full-resolution interferograms merged
    afterwards.  The merge places each burst on the absolute azimuth grid
    (from its sensing time against the frame origin) and keeps the later
    burst in the overlap zones, matching ISCE2's multi-swath VRT merge.

    Parameters
    ----------
    reference_path, secondary_path : path
        SAFE products.
    output_dir : path
        Output directory.
    swath : str, optional
        Sub-swath name.
    dem : DEMSampler, optional
        DEM sampler.  Defaults to a zero ellipsoid.
    multilook : tuple[int, int], optional
        ``(az, rg)`` looks applied after the full-res merge.
    goldstein_alpha : float, optional
        Goldstein filter exponent applied to the merged product.
    dead_pixel_amp_threshold : float, optional
        Dead-pixel amplitude mask threshold for the interferogram.
    esd_enabled : bool, optional
        Enable ESD azimuth residual.
    amplitude_refinement_enabled : bool, optional
        Enable amplitude-correlation residual refinement.
    control_spacing : int, optional
        Geometry control-point spacing.
    executor : {"torch"}, optional
        Resample executor.
    device : {"auto","cpu","cuda","mps"}, optional
        Torch device.
    reference_orbit_path, secondary_orbit_path : path, optional
        Precise ESA EOF orbits.
    geoid_correction : bool, optional
        Convert orthometric raster DEM heights to ellipsoidal with EGM96.
    burst_indices : list of int, optional
        Bursts to process. Defaults to all bursts of the swath.
    frame_azimuth_origin : datetime, optional
        Absolute azimuth time of frame row 0.  Defaults to the earliest
        burst start across all sub-swaths of the reference product, so the
        swath shares one absolute azimuth grid with the other swaths
        (ISCE2 merged-frame convention).
    frame_range_offset_fullres : int, optional
        Full-resolution frame column of this swath's range sample 0 in the
        concatenated multi-swath frame (0 for the near swath).  Matches
        ISCE2's ``dst.xOff - src.xOff`` per-swath placement.

    Returns
    -------
    ProductionPairState
        State with the merged radar product (wrapped phase, coherence,
        filtered complex interferogram) and stage timings.

    """
    import time

    from faninsar.missions.sentinel1.io.read import read_full_burst
    from faninsar.processing.tops.carrier import carrier_from_swath

    dem_sampler: DEMSampler = dem if dem is not None else ConstantHeightDEM(0.0)
    if geoid_correction and isinstance(dem_sampler, RasterDEM):
        from faninsar.processing.geometry.egm96 import EGM96Geoid

        dem_sampler = GeoidAdjustedDEM(dem_sampler, EGM96Geoid())

    def load_burst(path: Path, orbit_path: Path | None, bi: int) -> ProductionScene:
        product = open_safe_product(path)
        s1_swath = product.swath(swath)
        if orbit_path is not None:
            s1_swath = replace(s1_swath, orbit=read_eof_orbit(orbit_path))
        array = read_full_burst(s1_swath, burst_index=bi, full_range=True)
        burst = s1_swath.bursts[bi]
        carrier = carrier_from_swath(
            s1_swath, burst, first_range_sample=array.col0
        )
        geometry = _radar_model(
            s1_swath,
            burst,
            shape=array.samples.shape,
            row0=array.row0,
            col0=array.col0,
        )
        return ProductionScene(
            scene_id=_scene_id(path),
            path=path,
            product=product,
            swath=s1_swath,
            burst=burst,
            array=array,
            carrier=carrier,
            geometry=geometry,
        )

    total_started = time.perf_counter()
    reference = load_burst(Path(reference_path), reference_orbit_path, 0)
    swath_obj = reference.swath
    dt = swath_obj.azimuth_time_interval_s
    azimuth_origin = frame_azimuth_origin
    if azimuth_origin is None:
        product = open_safe_product(reference_path)
        starts = [item.bursts[0].azimuth_time for item in product.swaths if item.bursts]
        if not starts:
            reject_invalid_state("reference product has no bursts")
        azimuth_origin = min(starts)
    all_bursts = list(range(len(swath_obj.bursts)))
    bursts = burst_indices if burst_indices is not None else all_bursts
    if not bursts:
        reject_invalid_state("burst_indices must not be empty")
    for bi in bursts:
        if bi < 0 or bi >= len(all_bursts):
            reject_invalid_state(f"burst index {bi} out of range")

    az_looks, rg_looks = int(multilook[0]), int(multilook[1])
    az_offsets = {}
    for bi in bursts:
        az_offsets[bi] = round(
            (swath_obj.bursts[bi].azimuth_time - azimuth_origin).total_seconds() / dt
        )
    burst_width = reference.array.samples.shape[1]
    burst_lines = reference.array.samples.shape[0]
    frame_rows = az_offsets[bursts[-1]] + burst_lines
    frame_cols = frame_range_offset_fullres + burst_width
    out_rows = frame_rows // az_looks
    out_cols = frame_cols // rg_looks
    ifc_acc = np.zeros((out_rows, out_cols), dtype=np.complex128)
    pri_pow_acc = np.zeros((out_rows, out_cols), dtype=np.float64)
    sec_pow_acc = np.zeros((out_rows, out_cols), dtype=np.float64)
    claimed = np.zeros((out_rows, out_cols), dtype=np.int32)
    looks_per_window = az_looks * rg_looks

    per_burst_timings: dict[str, dict[str, float]] = {}
    for bi in reversed(bursts):
        tag = f"{swath}_b{bi}"
        ref = load_burst(Path(reference_path), reference_orbit_path, bi)
        sec = load_burst(Path(secondary_path), secondary_orbit_path, bi)
        state = ProductionPairState(
            pair_id=f"{ref.scene_id}_{sec.scene_id}_{tag}",
            reference=ref,
            secondary=sec,
            dem=dem_sampler,
            coregistration_grid="radar",
            multilook=multilook,
            goldstein_alpha=goldstein_alpha,
            unwrap_method="snaphu",
        )
        stage_times: dict[str, float] = {}
        t0 = time.perf_counter()
        state = stage_deramp(state)
        stage_times["deramp"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        state = stage_coregister(
            state,
            control_spacing=control_spacing,
            esd_enabled=esd_enabled,
            amplitude_refinement_enabled=amplitude_refinement_enabled,
            executor=executor,
            device=device,
        )
        stage_times["coregister"] = time.perf_counter() - t0
        # Accumulate full-res primary/secondary powers for the merged coherence.
        assert state.reference_deramped is not None
        assert state.secondary_aligned is not None
        pri_power = state.reference_deramped.real**2 + state.reference_deramped.imag**2
        sec_power = state.secondary_aligned.real**2 + state.secondary_aligned.imag**2
        t0 = time.perf_counter()
        state = stage_interferogram(
            state,
            multilook=(1, 1),
            goldstein_alpha=0.0,
            dead_pixel_amp_threshold=dead_pixel_amp_threshold,
        )
        stage_times["interferogram"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        state = stage_flatten(state)
        stage_times["flatten"] = time.perf_counter() - t0
        ifg_full = (
            state.complex_ifg_flat
            if state.complex_ifg_flat is not None
            else state.complex_ifg
        )
        if ifg_full is None:
            reject_invalid_state(f"{tag}: no interferogram produced")
        per_burst_timings[tag] = stage_times

        # Scatter into the multilooked frame in reverse order so the later
        # burst claims overlap windows first (ISCE2 VRT last-writer-wins).
        valid = np.abs(ifg_full) > 0
        rows = az_offsets[bi] + np.arange(ifg_full.shape[0])
        cols = frame_range_offset_fullres + np.arange(ifg_full.shape[1])
        orow = rows[:, None] // az_looks
        ocol = cols[None, :] // rg_looks
        inb = (orow < out_rows) & (ocol < out_cols) & valid
        r_i, c_i = np.broadcast_arrays(orow, ocol)
        r_v, c_v = r_i[inb], c_i[inb]
        free = claimed[r_v, c_v] < looks_per_window
        r_f, c_f = r_v[free], c_v[free]
        np.add.at(ifc_acc, (r_f, c_f), ifg_full[inb][free].astype(np.complex128))
        np.add.at(pri_pow_acc, (r_f, c_f), pri_power[inb][free])
        np.add.at(sec_pow_acc, (r_f, c_f), sec_power[inb][free])
        np.add.at(claimed, (r_f, c_f), 1)
        del pri_power, sec_power, ifg_full, state
        logger.info("Merged %s into frame", tag)

    has = claimed > 0
    merged_ifg = np.where(
        has, ifc_acc / np.where(has, claimed, 1), 0
    ).astype(np.complex64)
    pri_ml = pri_pow_acc / np.where(has, claimed, 1)
    sec_ml = sec_pow_acc / np.where(has, claimed, 1)
    denom = np.sqrt(np.maximum(pri_ml * sec_ml, 1e-30))
    coherence = np.clip(np.abs(merged_ifg) / denom, 0.0, 1.0).astype(np.float32)
    invalid = claimed == 0
    merged_ifg = np.asarray(merged_ifg, dtype=np.complex64).copy()
    merged_ifg[invalid] = np.nan + 1j * np.nan
    coherence[invalid] = np.nan
    wrapped = np.angle(merged_ifg).astype(np.float32)
    wrapped = np.where(
        np.isfinite(merged_ifg.real) & np.isfinite(merged_ifg.imag),
        wrapped,
        np.nan,
    )

    filtered = merged_ifg
    if goldstein_alpha > 0.0:
        from faninsar.processing.interferometry.pair import goldstein_filter

        filtered = goldstein_filter(merged_ifg, alpha=goldstein_alpha)

    result = ProductionPairState(
        pair_id=f"{reference.scene_id}_{_scene_id(Path(secondary_path))}_swath",
        reference=reference,
        secondary=load_burst(Path(secondary_path), secondary_orbit_path, 0),
        dem=dem_sampler,
        coregistration_grid="radar",
        dem_id=_dem_id(dem_sampler),
        coreg_executor=str(executor),
        coreg_device=str(device),
        multilook=multilook,
        goldstein_alpha=float(goldstein_alpha),
        unwrap_method="snaphu",
        complex_ifg=merged_ifg,
        complex_ifg_flat=filtered,
        coherence=coherence,
        wrapped_phase=wrapped,
        stage_timings_s={
            "total": time.perf_counter() - total_started,
            "per_burst": per_burst_timings,
        },
    )
    result.note(
        f"SWATH {swath} bursts={bursts} merged={merged_ifg.shape} "
        f"multilook={multilook} valid={float((~invalid).mean()):.3f} "
        f"mean_coh={float(np.nanmean(coherence)):.3f}"
    )
    result.unwrapped_phase = np.zeros_like(wrapped, dtype=np.float32)
    result.connected_components = np.zeros_like(wrapped, dtype=np.uint8)
    result.note("UNWRAP skipped: swath mode emits wrapped radar products only")
    stage_write(result, output_dir)
    return result


def run_full_frame(
    reference_path: str | Path,
    secondary_path: str | Path,
    *,
    output_dir: str | Path,
    dem: DEMSampler | None = None,
    swaths: tuple[str, ...] = ("IW1", "IW2", "IW3"),
    multilook: tuple[int, int] = (2, 10),
    goldstein_alpha: float = 0.5,
    dead_pixel_amp_threshold: float = 3.0,
    executor: str = "torch",
    device: str = "auto",
    reference_orbit_path: str | Path | None = None,
    secondary_orbit_path: str | Path | None = None,
    geoid_correction: bool = True,
    burst_indices: dict[str, list[int]] | None = None,
) -> ProductionPairState:
    """Process a full multi-swath frame and merge it into one radar product.

    Convenience wrapper over :func:`run_production_swath`: derives the
    per-swath range offset in the concatenated frame from the annotation
    slant-range times, processes each sub-swath burst-by-burst, overlays the
    swath canvases on the absolute frame grid (valid pixels win), applies the
    Goldstein filter after the merge, and writes the merged Zarr/STAC.

    Parameters
    ----------
    reference_path, secondary_path : path
        SAFE products.
    output_dir : path
        Output directory; one subdirectory per swath plus the merged frame.
    dem : DEMSampler, optional
        DEM sampler.  Defaults to a zero ellipsoid.
    swaths : list of str, optional
        Sub-swaths to process, in range order.  Frame column 0 is the first
        swath's range sample 0.
    multilook : tuple[int, int], optional
        ``(az, rg)`` looks applied after the full-res merge.
    goldstein_alpha : float, optional
        Goldstein filter exponent applied to the merged product.
    dead_pixel_amp_threshold : float, optional
        Dead-pixel amplitude mask threshold for the interferogram.
    executor : {"torch"}, optional
        Resample executor.
    device : {"auto","cpu","cuda","mps"}, optional
        Torch device.
    reference_orbit_path, secondary_orbit_path : path, optional
        Precise ESA EOF orbits.
    geoid_correction : bool, optional
        Convert orthometric raster DEM heights to ellipsoidal with EGM96.
    burst_indices : dict of str to list of int, optional
        Bursts to process per swath.  Defaults to all bursts of each swath.

    Returns
    -------
    ProductionPairState
        State with the merged frame (wrapped phase, coherence, filtered
        complex interferogram) and per-swath stage timings.

    """
    import time

    from faninsar.missions.sentinel1.safe import open_safe_product

    if not swaths:
        reject_invalid_state("swaths must not be empty")
    product = open_safe_product(reference_path)
    swath_map = {item.swath: item for item in product.swaths}
    missing = [name for name in swaths if name not in swath_map]
    if missing:
        reject_invalid_state(
            f"swaths {missing} not found in {Path(reference_path).name}; "
            f"available={sorted(swath_map)}"
        )

    ref_swath = swath_map[swaths[0]]
    rg_offsets: dict[str, int] = {}
    for name in swaths:
        item = swath_map[name]
        if name == swaths[0]:
            rg_offsets[name] = 0
        else:
            offset = round(
                (item.slant_range_time_s - ref_swath.slant_range_time_s)
                * item.range_sampling_rate_hz
            )
            if offset < 0:
                reject_invalid_state(
                    f"swath {name} starts before frame swath {swaths[0]}"
                )
            rg_offsets[name] = offset

    total_started = time.perf_counter()
    canvases: dict[str, np.ndarray] = {}
    coherences: dict[str, np.ndarray] = {}
    per_swath_timings: dict[str, float] = {}
    first_state: ProductionPairState | None = None
    for name in swaths:
        state = run_production_swath(
            reference_path,
            secondary_path,
            output_dir=Path(output_dir) / name,
            swath=name,
            dem=dem,
            multilook=multilook,
            goldstein_alpha=0.0,
            dead_pixel_amp_threshold=dead_pixel_amp_threshold,
            executor=executor,
            device=device,
            reference_orbit_path=reference_orbit_path,
            secondary_orbit_path=secondary_orbit_path,
            geoid_correction=geoid_correction,
            burst_indices=None if burst_indices is None else burst_indices.get(name),
            frame_range_offset_fullres=rg_offsets[name],
        )
        if first_state is None:
            first_state = state
        canvases[name] = np.asarray(state.complex_ifg)
        coherences[name] = np.asarray(state.coherence)
        per_swath_timings[name] = state.stage_timings_s["total"]
        logger.info("Full-frame %s canvas %s", name, canvases[name].shape)

    rows = max(canvas.shape[0] for canvas in canvases.values())
    cols = max(canvas.shape[1] for canvas in canvases.values())
    merged = np.zeros((rows, cols), dtype=np.complex64)
    merged_coh = np.full((rows, cols), np.nan, dtype=np.float32)
    for name in swaths:
        canvas = canvases[name]
        valid = np.abs(canvas) > 0
        merged[: canvas.shape[0], : canvas.shape[1]][valid] = canvas[valid]
        merged_coh[: canvas.shape[0], : canvas.shape[1]][valid] = (
            coherences[name][valid]
        )
    invalid = np.abs(merged) <= 0
    wrapped = np.where(invalid, np.nan, np.angle(merged).astype(np.float32))

    filtered = merged
    if goldstein_alpha > 0.0:
        from faninsar.processing.interferometry.pair import goldstein_filter

        filtered = goldstein_filter(merged, alpha=goldstein_alpha)

    assert first_state is not None
    result = ProductionPairState(
        pair_id=f"{first_state.pair_id.split('_swath')[0]}_frame",
        reference=first_state.reference,
        secondary=first_state.secondary,
        dem=first_state.dem,
        coregistration_grid="radar",
        dem_id=first_state.dem_id,
        coreg_executor=str(executor),
        coreg_device=str(device),
        multilook=multilook,
        goldstein_alpha=float(goldstein_alpha),
        unwrap_method="snaphu",
        complex_ifg=merged,
        complex_ifg_flat=filtered,
        coherence=merged_coh,
        wrapped_phase=wrapped,
        stage_timings_s={
            "total": time.perf_counter() - total_started,
            "per_swath": per_swath_timings,
        },
    )
    result.note(
        f"FULL FRAME swaths={swaths} merged={merged.shape} "
        f"multilook={multilook} valid={float((~invalid).mean()):.3f}"
    )
    result.unwrapped_phase = np.zeros_like(wrapped, dtype=np.float32)
    result.connected_components = np.zeros_like(wrapped, dtype=np.uint8)
    result.note("UNWRAP skipped: full-frame mode emits wrapped radar products only")
    stage_write(result, output_dir)
    return result
