"""Production S1 pair workflow.

Stages: full-burst read → deramp → dense coreg+ESD → ifg → flatten →
unwrap → geocode → write.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.coreg import (
    combine_offset_fields,
    dense_geometry_offsets,
    estimate_azimuth_shift_esd,
    refine_shift_with_correlation,
    resample_complex,
)
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry import (
    ConstantHeightDEM,
    RadarGeometryModel,
    rdr2geo_with_dem_chunked,
)
from faninsar.processing.geometry.baseline import BaselineComponents
from faninsar.processing.interferometry.flatten import (
    compute_topographic_phase,
    remove_topographic_phase,
)
from faninsar.processing.interferometry.pair import form_interferogram, goldstein_filter
from faninsar.processing.pipeline.products import (
    PairProductArrays,
    write_pair_stac_item,
    write_pair_zarr,
)
from faninsar.processing.tops.carrier import carrier_from_swath
from faninsar.processing.tops.deramp import TOPSCarrierModel, deramp, reramp
from faninsar.processing.unwrap import unwrap as unwrap_dispatch
from faninsar.sentinel1 import (
    open_safe_product,
    read_full_burst,
    read_swath_bursts,
    stitch_bursts,
)

if TYPE_CHECKING:
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.unwrap.common import CommonUnwrapResult
    from faninsar.sentinel1.io import BurstArray
    from faninsar.sentinel1.types import S1Burst, S1Product, S1Swath

logger = setup_logger(__name__)

SPEED_OF_LIGHT_M_S = 299_792_458.0
ScopeMode = Literal["burst", "swath"]


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
    complex_ifg: np.ndarray | None = None
    complex_ifg_flat: np.ndarray | None = None
    topo_phase: np.ndarray | None = None
    coherence: np.ndarray | None = None
    wrapped_phase: np.ndarray | None = None
    unwrapped_phase: np.ndarray | None = None
    connected_components: np.ndarray | None = None
    baseline: BaselineComponents | None = None
    geocoded: dict[str, np.ndarray] | None = None
    zarr_path: Path | None = None
    stac_path: Path | None = None
    log: list[str] = field(default_factory=list)

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

    Returns
    -------
    ProductionScene
        Full complex array with carrier and geometry model.

    """
    _ = dem
    product = open_safe_product(path)
    s1_swath = product.swath(swath)
    if scope == "burst":
        array = read_full_burst(s1_swath, burst_index=burst_index)
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


def stage_coregister(
    state: ProductionPairState,
    *,
    control_spacing: int = 64,
    esd_enabled: bool = True,
    executor: str = "serial",
    device: str = "auto",
) -> ProductionPairState:
    """Dense geometry offsets + ESD az residual + amplitude residual + resample.

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
        Stride (pixels) between geometry control points. Default 64.
    esd_enabled : bool, optional
        Run ESD azimuth residual estimation. Default True.
    executor : {"serial", "dask-torch"}, optional
        Lanczos compute path for the final resample. ``"dask-torch"`` runs the
        dask + torch + numpy Lanczos block_fn
        (:func:`lanczos_resample_dask_torch`); ``"serial"`` is the NumPy
        reference. Ignored for the ESD bilinear pre-align. Default ``"serial"``.
    device : {"auto","cpu","cuda"}, optional
        Torch device when ``executor="dask-torch"``. ``"auto"`` resolves to
        CUDA if available else CPU (never MPS). Default ``"auto"``.

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
    geometry_field = dense_geometry_offsets(
        shape=ref.shape,
        reference_model=state.reference.geometry,
        secondary_model=state.secondary.geometry,
        dem=dem,
        stride=control_spacing,
    )

    # Amplitude residual around geometry mean
    prior_rg = float(
        np.nanmedian(geometry_field.range_offset_px[geometry_field.coverage])
    )
    prior_az = float(
        np.nanmedian(geometry_field.azimuth_offset_px[geometry_field.coverage])
    )
    amp_rg, amp_az = refine_shift_with_correlation(
        ref,
        sec,
        prior_rg=prior_rg,
        prior_az=prior_az,
        search_radius=32,
    )
    amp_res_rg = amp_rg - prior_rg
    amp_res_az = amp_az - prior_az

    esd_az = 0.0
    if esd_enabled:
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

    sec_resamp = resample_complex(
        sec,
        range_offset_px=offsets.range_offset_px,
        azimuth_offset_px=offsets.azimuth_offset_px,
        executor=executor,
        device=device,
    )
    # Secondary deramped no longer needed after final resample.
    state.secondary_deramped = None
    del sec
    gc.collect()

    # Reramp onto reference carrier
    state.reference_deramped = reramp(ref, state.reference.carrier)
    state.secondary_aligned = reramp(sec_resamp, state.reference.carrier)
    del sec_resamp, ref
    state.range_shift_px = float(
        np.nanmedian(offsets.range_offset_px[offsets.coverage])
    )
    state.azimuth_shift_px = float(
        np.nanmedian(offsets.azimuth_offset_px[offsets.coverage])
    )
    coverage = float(np.mean(offsets.coverage))
    del offsets
    gc.collect()
    state.note(
        f"COREG dense+ESD+amp median_rg={state.range_shift_px:.3f} "
        f"median_az={state.azimuth_shift_px:.3f} "
        f"coverage={coverage:.3f}"
    )
    return state


def stage_interferogram(
    state: ProductionPairState,
    *,
    multilook: tuple[int, int] = (4, 20),
    goldstein_alpha: float = 0.5,
) -> ProductionPairState:
    """Form multilooked interferogram and apply Goldstein filter."""
    if state.reference_deramped is None or state.secondary_aligned is None:
        reject_invalid_state("interferogram requires coregister")
    ifg = form_interferogram(
        state.reference_deramped,
        state.secondary_aligned,
        multilook=multilook,
    )
    # Full-res SLCs no longer needed once multilooked ifg/coherence exist.
    state.reference_deramped = None
    state.secondary_aligned = None
    gc.collect()
    filtered = goldstein_filter(ifg.complex_ifg, alpha=goldstein_alpha)
    state.complex_ifg = filtered.astype(np.complex64, copy=False)
    state.coherence = ifg.coherence
    state.wrapped_phase = np.angle(filtered).astype(np.float32)
    state.note(
        f"IFG multilook={multilook} mean_coh={float(np.nanmean(ifg.coherence)):.3f}"
    )
    return state


def stage_flatten(state: ProductionPairState) -> ProductionPairState:
    """Remove topographic phase using DEM + dual-orbit geometry."""
    if state.complex_ifg is None:
        reject_invalid_state("flatten requires interferogram")
    height, width = state.complex_ifg.shape
    # Multilooked grid: approximate geometry by scaling indices
    # Use reference geometry on the multilooked sampling lattice
    az = np.arange(height, dtype=np.float64)
    rg = np.arange(width, dtype=np.float64)
    # Scale factors from full-res to multilooked (approximate uniform)
    full_h, full_w = state.reference.array.samples.shape
    az_scale = full_h / max(height, 1)
    rg_scale = full_w / max(width, 1)
    az_full = az * az_scale
    rg_full = rg * rg_scale
    az_grid, rg_grid = np.meshgrid(az_full, rg_full, indexing="ij")

    topo = compute_topographic_phase(
        state.reference.geometry,
        state.secondary.geometry,
        az_grid,
        rg_grid,
        state.dem,
    )
    flat = remove_topographic_phase(state.complex_ifg, topo)
    state.topo_phase = topo.astype(np.float32)
    state.complex_ifg_flat = flat.astype(np.complex64, copy=False)
    state.wrapped_phase = np.angle(flat).astype(np.float32)
    topo_finite = topo[np.isfinite(topo)]
    if topo_finite.size:
        rms = float(np.std(topo_finite))
        span = float(np.max(topo_finite) - np.min(topo_finite))
    else:
        rms, span = float("nan"), float("nan")
    state.note(f"FLATTEN geometric_phase rms={rms:.3f} rad span={span:.1f} rad")
    return state


def stage_unwrap(
    state: ProductionPairState,
    *,
    method: Literal["irls", "dct_irls", "snaphu"] = "irls",
) -> ProductionPairState:
    """Unwrap flattened interferogram with explicit backend selection."""
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
    )
    state.unwrapped_phase = result.unwrapped_phase
    state.connected_components = result.connected_components
    state.note(f"UNWRAP method={result.method} metrics={result.metrics}")
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
        "unwrap_method": "irls",
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
    if state.geocoded is not None:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="a")
        geo = root.require_group("geocoded")
        for name, array in state.geocoded.items():
            geo.create_array(name, data=array, overwrite=True)
        if state.topo_phase is not None:
            root.create_array("topo_phase", data=state.topo_phase, overwrite=True)

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
    state.note(f"WRITE {zarr_path}")
    return state


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
    unwrap_method: Literal["irls", "dct_irls", "snaphu"] = "irls",
    esd_enabled: bool = True,
    control_spacing: int = 64,
    executor: str = "serial",
    device: str = "auto",
) -> ProductionPairState:
    """Run the full production pair chain on full burst or full swath.

    Stages
    ------
    1. Load full burst/swath + carrier + geometry
    2. Deramp
    3. Dense geometry coreg + ESD + amplitude residual + resample + reramp
    4. Interferogram + multilook + Goldstein
    5. Topographic flattening (DEM)
    6. Unwrap (IRLS or snaphu)
    7. Baseline metadata
    8. Chunked vectorized geocode
    9. Write Zarr/STAC with bbox

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
        ``(az, rg)`` looks.
    goldstein_alpha : float, optional
        Goldstein filter exponent (default 0.5).
    unwrap_method : {"irls", "dct_irls", "snaphu"}, optional
        Explicit unwrap backend passed to :func:`stage_unwrap`. Default
        ``"irls"``.
    esd_enabled : bool, optional
        Enable spectral diversity azimuth residual.
    control_spacing : int, optional
        Geometry control-point spacing in pixels.
    executor : {"serial", "dask-torch"}, optional
        Lanczos compute path for the final coreg resample
        (:func:`stage_coregister`). Default ``"serial"`` preserves prior
        behaviour.
    device : {"auto","cpu","cuda","mps"}, optional
        Torch device when ``executor="dask-torch"``. ``"auto"`` selects CUDA
        when available, otherwise CPU/NumPy fallback. Default ``"auto"``.

    Returns
    -------
    ProductionPairState
        Full state with products and stage log.

    """
    dem_sampler: DEMSampler = dem if dem is not None else ConstantHeightDEM(0.0)
    reference = load_production_scene(
        reference_path,
        swath=swath,
        scope=scope,
        burst_index=burst_index,
        dem=dem_sampler,
    )
    secondary = load_production_scene(
        secondary_path,
        swath=swath,
        scope=scope,
        burst_index=burst_index,
        dem=dem_sampler,
    )
    state = ProductionPairState(
        pair_id=f"{reference.scene_id}_{secondary.scene_id}",
        reference=reference,
        secondary=secondary,
        dem=dem_sampler,
    )
    state.note(f"START production pair scope={scope}")
    state = stage_deramp(state)
    state = stage_coregister(
        state,
        control_spacing=control_spacing,
        esd_enabled=esd_enabled,
        executor=executor,
        device=device,
    )
    state = stage_interferogram(
        state,
        multilook=multilook,
        goldstein_alpha=goldstein_alpha,
    )
    state = stage_flatten(state)
    state = stage_unwrap(state, method=unwrap_method)
    state = stage_baseline(state)
    state = stage_geocode(state)
    state = stage_write(state, output_dir)
    state.note("DONE")
    return state
