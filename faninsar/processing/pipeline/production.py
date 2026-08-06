"""Production Sentinel-1 pair workflows on radar or geographic grids."""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

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
from faninsar.query import BoundingBox, Polygons

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def looks_dir(azimuth_looks: int, range_looks: int) -> str:
    """Return the deterministic per-config output subtree name."""
    return f"looks_{azimuth_looks}x{range_looks}"


def _sweep_list_metadata(
    all_configs: list[tuple[int, int]] | None,
    config: tuple[int, int],
) -> list[list[int]]:
    """Serialize the full multilook sweep list for product metadata."""
    return [
        list(item) for item in (all_configs if all_configs is not None else [config])
    ]


def _is_multilook_pair(value: object) -> bool:
    """Return whether the multilook argument is one (az, rg) pair."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return all(isinstance(part, int) and not isinstance(part, bool) for part in value)


def normalize_multilook_sweep(value: object) -> list[tuple[int, int]]:
    """Normalize a multilook sweep specification into validated configs."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        message = "multilook must be an iterable of [azimuth_looks, range_looks] pairs"
        raise TypeError(message)
    configs: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, (str, bytes)) or not isinstance(item, (list, tuple)):
            message = f"each multilook config must be a [az, rg] pair; got {item!r}"
            raise TypeError(message)
        if len(item) != 2:
            message = f"each multilook config must be a [az, rg] pair; got {item!r}"
            raise ValueError(message)
        azimuth_looks, range_looks = item
        if (
            isinstance(azimuth_looks, bool)
            or isinstance(range_looks, bool)
            or not isinstance(azimuth_looks, int)
            or not isinstance(range_looks, int)
        ):
            message = f"multilook factors must be integers; got {item!r}"
            raise TypeError(message)
        if azimuth_looks < 1 or range_looks < 1:
            message = f"multilook factors must be >= 1; got {item!r}"
            raise ValueError(message)
        configs.append((azimuth_looks, range_looks))
    if not configs:
        message = "multilook must contain at least one config"
        raise ValueError(message)
    deduplicated = list(dict.fromkeys(configs))
    if len(deduplicated) != len(configs):
        message = f"duplicate multilook configs are not allowed: {configs!r}"
        raise ValueError(message)
    return deduplicated


@dataclass(frozen=True, slots=True)
class PairSweepOutcome:
    """Lightweight per-config outputs of a multilook pair sweep."""

    config: tuple[int, int]
    zarr_path: Path
    stac_path: Path
    shape: tuple[int, int]
    metadata: dict[str, Any]
    stage_timings_s: dict[str, float]
    log: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProductionPairSweepResult:
    """Outputs of a multilook sweep over one reference/secondary pair."""

    pair_id: str
    per_config: dict[tuple[int, int], PairSweepOutcome]


@dataclass
class SharedPairResources:
    """Ownership bundle for the shared pair prefix used by sweeps."""

    prefix_state: ProductionPairState | None = None
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    ifg_archive: dict[str, Any] | None = None

    def cleanup(self) -> None:
        """Release shared arrays and remove the geo work dir exactly once."""
        candidates: list[np.ndarray | None] = []
        state = self.prefix_state
        if state is not None and state.geo2rdr_lut is not None:
            candidates.extend(
                (
                    state.geo2rdr_lut.az_full,
                    state.geo2rdr_lut.rg_full,
                    state.geo2rdr_lut.valid,
                    state.geo2rdr_lut.height_full,
                )
            )
            state.geo2rdr_lut = None
        if state is not None:
            for name in (
                "reference_geocoded_slc",
                "secondary_geocoded_slc",
                "geocoded_slc_valid",
                "topo_phase",
                "geo_height_field",
                "reference_deramped",
                "secondary_aligned",
            ):
                candidates.append(getattr(state, name))
                setattr(state, name, None)
        closed: set[int] = set()
        for array in candidates:
            if not isinstance(array, np.memmap) or id(array) in closed:
                continue
            mapping = getattr(array, "_mmap", None)
            if mapping is None or mapping.closed:
                continue
            closed.add(id(array))
            close_memmap(array)
        temporary_directory = self.temporary_directory
        self.temporary_directory = None
        gc.collect()
        if temporary_directory is not None:
            temporary_directory.cleanup()


def _output_bytes(path: Path) -> int:
    """Return total bytes of a Zarr store on disk."""
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def _process_burst_worker(task: dict[str, object]) -> dict[str, object]:
    """Process one burst unit in a worker process.

    Parameters
    ----------
    task : dict
        Picklable task arguments produced by ``_archive_burst_ifgs``.

    Returns
    -------
    dict
        ``{"unit": unit_or_None, "stage_times": dict}``.

    """
    from dataclasses import replace as _replace

    from faninsar.missions.sentinel1 import read_eof_orbit, read_full_burst
    from faninsar.missions.sentinel1.safe import open_safe_product
    from faninsar.processing.tops.carrier import carrier_from_swath

    tag = str(task["tag"])
    swath = str(task["swath"])
    frame_index = int(task["frame_index"])
    burst_index = int(task["burst_index"])
    azimuth_offset = int(task["azimuth_offset"])
    ref_path = Path(task["ref_path"])
    sec_path = Path(task["sec_path"])
    ref_orbit = task.get("ref_orbit")
    sec_orbit = task.get("sec_orbit")
    roi = task.get("roi")
    control_spacing = task.get("control_spacing")
    esd_enabled = bool(task["esd_enabled"])
    amplitude_refinement_enabled = bool(task["amplitude_refinement_enabled"])
    executor = str(task["executor"])
    device = str(task["device"])
    dead_pixel_amp_threshold = float(task["dead_pixel_amp_threshold"])
    coregistration_grid = task["coregistration_grid"]
    geo_grid = task["geo_grid"]
    geo_height_m = float(task["geo_height_m"])
    geo_chunk_size = int(task["geo_chunk_size"])
    roi_buffer_m = float(task.get("roi_buffer_m", 320.0))
    ifg_dir = Path(task["ifg_dir"])
    dem = task["dem"]
    geo_work_dir = task["geo_work_dir"]

    ref_product = open_safe_product(ref_path)
    sec_product = open_safe_product(sec_path)

    def load_burst_worker(
        path: Path,
        orbit_path: Path | None,
        product: object,
    ) -> ProductionScene:
        s1_swath = product.swath(swath)
        if orbit_path is not None:
            s1_swath = _replace(s1_swath, orbit=read_eof_orbit(orbit_path))
        array = read_full_burst(s1_swath, burst_index=burst_index, full_range=True)
        burst = s1_swath.bursts[burst_index]
        carrier = carrier_from_swath(s1_swath, burst, first_range_sample=array.col0)
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

    ref = load_burst_worker(ref_path, ref_orbit, ref_product)
    sec = load_burst_worker(sec_path, sec_orbit, sec_product)
    state = ProductionPairState(
        pair_id=ref.scene_id + "_" + sec.scene_id + "_" + tag,
        reference=ref,
        secondary=sec,
        dem=dem,
        coregistration_grid="radar",
        multilook=(1, 1),
        goldstein_alpha=0.0,
        unwrap_method="snaphu",
    )
    stage_times: dict[str, float] = {}
    t0 = time.perf_counter()
    state = stage_deramp(state)
    stage_times["deramp"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    burst_work_dir: Path | None = None
    if coregistration_grid == "geo" and geo_work_dir is not None:
        burst_work_dir = Path(geo_work_dir) / tag
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
        geo_work_dir=burst_work_dir,
        roi=roi,
        roi_buffer_m=roi_buffer_m,
    )
    stage_times["coregister"] = time.perf_counter() - t0
    burst_row0 = 0
    burst_col0 = 0
    geo_valid_mask: np.ndarray | None = None
    if coregistration_grid == "geo":
        if state.geocoded_slc_valid is not None:
            geo_valid_mask = np.asarray(state.geocoded_slc_valid)
    elif roi is not None:
        assert state.reference_deramped is not None
        window = _roi_burst_window(
            roi,
            ref.geometry,
            dem,
            state.reference_deramped.shape,
        )
        if window is not None:
            burst_row0, burst_row1, burst_col0, burst_col1 = window
            state.reference_deramped = state.reference_deramped[
                burst_row0:burst_row1, burst_col0:burst_col1
            ]
            state.secondary_aligned = state.secondary_aligned[
                burst_row0:burst_row1, burst_col0:burst_col1
            ]
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
    if coregistration_grid == "geo":
        state.complex_ifg_flat = state.complex_ifg
        state.note(
            "FLATTEN applied to secondary geocoded SLC before IFG formation"
        )
        stage_times["flatten"] = 0.0
    else:
        state = stage_flatten(state)
        stage_times["flatten"] = time.perf_counter() - t0
    ifg_full = (
        state.complex_ifg_flat
        if state.complex_ifg_flat is not None
        else state.complex_ifg
    )
    if ifg_full is None:
        reject_invalid_state(tag + ": no interferogram produced")

    base = ifg_dir / tag
    ifg_path = str(base) + ".complex64"
    pri_path = str(base) + ".pri.f64"
    sec_path_out = str(base) + ".sec.f64"
    if coregistration_grid == "geo":
        geo_valid = np.zeros(ifg_full.shape, dtype=bool)
        if geo_valid_mask is not None:
            geo_valid = geo_valid_mask
        geo_valid &= np.abs(ifg_full) > 0
        if not geo_valid.any():
            logger.warning("%s: no valid geocoded footprint; skipped", tag)
            return {"unit": None, "stage_times": stage_times}
        rows_any = geo_valid.any(axis=1)
        cols_any = geo_valid.any(axis=0)
        row0 = int(np.argmax(rows_any))
        row1 = int(rows_any.size - np.argmax(rows_any[::-1]))
        col0 = int(np.argmax(cols_any))
        col1 = int(cols_any.size - np.argmax(cols_any[::-1]))
        geo_bbox = state.geo_bbox
        if geo_bbox is not None:
            row0 += geo_bbox[0]
            row1 += geo_bbox[0]
            col0 += geo_bbox[2]
            col1 += geo_bbox[2]
            bbox_local = (
                row0 - geo_bbox[0],
                row1 - geo_bbox[0],
                col0 - geo_bbox[2],
                col1 - geo_bbox[2],
            )
        else:
            bbox_local = (row0, row1, col0, col1)
        height_path = str(base) + ".height.f64"
        height_full = state.geo_height_field
        if height_full is None:
            height_full = np.full(
                ifg_full.shape, float(geo_height_m), dtype=np.float64
            )
        else:
            height_full = np.asarray(height_full, dtype=np.float64)
        lr0, lr1, lc0, lc1 = bbox_local
        bbox_ifg = ifg_full[lr0:lr1, lc0:lc1]
        bbox_pri = pri_power[lr0:lr1, lc0:lc1]
        bbox_sec = sec_power[lr0:lr1, lc0:lc1]
        bbox_height = height_full[lr0:lr1, lc0:lc1]
        bbox_ifg.astype(np.complex64, copy=False).tofile(ifg_path)
        bbox_pri.astype(np.float64, copy=False).tofile(pri_path)
        bbox_sec.astype(np.float64, copy=False).tofile(sec_path_out)
        bbox_height.astype(np.float64, copy=False).tofile(height_path)
        unit: dict[str, object] = {
            "tag": tag,
            "swath": swath,
            "frame_index": frame_index,
            "ifg_path": ifg_path,
            "pri_path": pri_path,
            "sec_path": sec_path_out,
            "height_path": height_path,
            "rows": int(bbox_ifg.shape[0]),
            "cols": int(bbox_ifg.shape[1]),
            "row0": row0,
            "col0": col0,
            "mode": "geo",
        }
    else:
        ifg_full.astype(np.complex64, copy=False).tofile(ifg_path)
        pri_power.astype(np.float64, copy=False).tofile(pri_path)
        sec_power.astype(np.float64, copy=False).tofile(sec_path_out)
        unit = {
            "tag": tag,
            "swath": swath,
            "frame_index": frame_index,
            "ifg_path": ifg_path,
            "pri_path": pri_path,
            "sec_path": sec_path_out,
            "rows": int(ifg_full.shape[0]),
            "cols": int(ifg_full.shape[1]),
            "azimuth_offset": azimuth_offset,
            "burst_row0": burst_row0,
            "burst_col0": burst_col0,
        }
    logger.info("Archived %s into sweep prefix", tag)
    return {"unit": unit, "stage_times": stage_times}


def _write_run_manifest(
    output_dir: str | Path,
    *,
    looks: tuple[int, int],
    pair_id: str,
    products: list[str],
    grid: dict[str, object],
) -> Path:
    """Atomically write a run.json completion manifest for one config."""
    import json
    from datetime import UTC

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1",
        "status": "complete",
        "looks": [looks[0], looks[1]],
        "pair_id": pair_id,
        "products": products,
        "product_bytes": {name: _output_bytes(out / name) for name in products},
        "grid": grid,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    temporary = out / "run.json.tmp"
    target = out / "run.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


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
    geo_bbox: tuple[int, int, int, int] | None = None

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
    row0_offset: int = 0,
    col0_offset: int = 0,
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
    row0_offset, col0_offset : int
        Offset of the cropped LUT inside the full geographic grid, used to
        map cropped rows/cols back to global grid coordinates.

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
        shape=lut.shape,
    )
    height_field = lut.height_full
    if height_field is None:
        height_field = np.memmap(
            output_dir / "height.float64",
            mode="w+",
            dtype=np.float64,
            shape=lut.shape,
        )
    invalid = np.complex64(np.nan + 1j * np.nan)
    for row_start in range(0, lut.shape[0], chunk_size):
        row_stop = min(row_start + chunk_size, lut.shape[0])
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
        latitude, longitude = grid_lonlat_rows(
            grid,
            row_start + row0_offset,
            row_stop + row0_offset,
        )
        col_stop = col0_offset + lut.shape[1]
        if col0_offset > 0 or col_stop < grid.width:
            latitude = latitude[:, col0_offset:col_stop]
            longitude = longitude[:, col0_offset:col_stop]
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
    roi: BoundingBox | Polygons | None = None,
    roi_buffer_m: float = 320.0,
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
    roi : BoundingBox or Polygons, optional
        Restricts geo processing to the ROI-burst quad intersection
        (buffered by ``roi_buffer_m``) instead of the full burst bbox.
    roi_buffer_m : float, optional
        Physical margin in meters kept around the ROI in geo mode, applied in
        the grid CRS so the extension is isotropic on the ground.
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
        from faninsar.processing.pipeline.geo_lut import (
            build_geo2rdr_lut,
            burst_geo_quad_lonlat,
            derive_burst_geo_bbox,
            roi_geo_bbox,
        )
        from faninsar.processing.pipeline.geo_modes import (
            coregister_geocoded_slcs_chunked,
        )

        if geo_work_dir is None:
            reject_invalid_state("geo coregistration requires geo_work_dir")
        work_directory = Path(geo_work_dir)
        work_directory.mkdir(parents=True, exist_ok=True)
        burst_row0, burst_row1, burst_col0, burst_col1 = derive_burst_geo_bbox(
            geometry=state.reference.geometry,
            radar_shape=ref.shape,
            grid=geo_grid,
            dem=state.dem,
        )
        footprint_lonlat: np.ndarray | None = None
        roi_geometry: object | None = None
        if roi is not None:
            from shapely.geometry import MultiPolygon
            from shapely.geometry import Polygon as ShapelyPolygon

            from faninsar.processing.pipeline.geo_lut import polygon_parts

            burst_quad = burst_geo_quad_lonlat(
                geometry=state.reference.geometry,
                radar_shape=ref.shape,
                dem=state.dem,
            )
            if burst_quad is not None:
                intersection = _roi_geometry(roi).intersection(
                    ShapelyPolygon(burst_quad).buffer(0)
                )
                parts = [
                    part
                    for part in polygon_parts(intersection)
                    if not part.is_empty
                ]
                if parts:
                    roi_polygon = (
                        parts[0] if len(parts) == 1 else MultiPolygon(parts)
                    )
                    roi_geometry = _buffer_geometry_meters(
                        roi_polygon,
                        geo_grid.crs,
                        roi_buffer_m,
                    )
                    burst_row0, burst_row1, burst_col0, burst_col1 = roi_geo_bbox(
                        roi_geometry,
                        geo_grid,
                        margin_px=2,
                    )
        state.geo_bbox = (burst_row0, burst_row1, burst_col0, burst_col1)
        substage_started = time.perf_counter()
        lut = build_geo2rdr_lut(
            geometry=state.reference.geometry,
            grid=geo_grid,
            full_radar_shape=ref.shape,
            height_m=geo_height_m,
            dem=state.dem,
            chunk_size=geo_chunk_size,
            storage_dir=work_directory / "lut",
            row_range=(burst_row0, burst_row1),
            col_range=(burst_col0, burst_col1),
            footprint_lonlat=footprint_lonlat,
            roi_geometry=roi_geometry,
            polygon_dilate_px=2,
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
            row0_offset=burst_row0,
            col0_offset=burst_col0,
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
        entry point :func:`run_pair` passes 3.0 for real S1
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
    state: ProductionPairState,
    output_dir: str | Path,
    *,
    preserve_geo_work_dir: bool = False,
    stac_item_id: str | None = None,
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
                "id": stac_item_id if stac_item_id is not None else state.pair_id,
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
            write_pair_stac_item(
                product, zarr_path, stac_path, stac_item_id=stac_item_id
            )
    else:
        write_pair_stac_item(product, zarr_path, stac_path, stac_item_id=stac_item_id)

    state.zarr_path = zarr_path
    state.stac_path = stac_path
    if state.memory_watchdog is not None:
        state.memory_watchdog.sample("write:complete")
    if state.geo_work_dir is not None and not preserve_geo_work_dir:
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


def _finalize_geo_products(
    state: ProductionPairState,
    *,
    geo_grid: GeoGridSpec,
    multilook: tuple[int, int],
    geo_height_m: float,
) -> ProductionPairState:
    """Assemble the multilooked geographic product grid from unwrapped phase."""
    from faninsar.processing.pipeline.geo_lut import grid_lonlat

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
    state.geo_grid_meta = _geo_grid_meta(product_grid)
    state.note("GEO products complete on multilooked geographic grid")
    return state


BurstSelection = (
    dict[str, list[int] | range | str] | list[dict[str, list[int] | range | str]]
)


def _as_frame_paths(
    value: str | Path | Sequence[str | Path],
    name: str,
) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    paths = [Path(item) for item in value]
    if not paths:
        reject_invalid_state(name + " must contain at least one SAFE product")
    return paths


def _as_optional_frame_sequence(
    value: str | Path | Sequence[str | Path] | None,
    frame_count: int,
    name: str,
) -> list[Path | None]:
    if value is None:
        return [None] * frame_count
    paths = _as_frame_paths(value, name)
    if len(paths) == 1 and frame_count > 1:
        return [paths[0]] * frame_count
    if len(paths) != frame_count:
        reject_invalid_state(
            name
            + " has "
            + str(len(paths))
            + " entries for "
            + str(frame_count)
            + " frames"
        )
    return list(paths)


def _burst_index_list(
    value: list[int] | range | str,
    swath: str,
    count: int,
) -> list[int]:
    if isinstance(value, str):
        if value.strip() == "all":
            return list(range(count))
        parts = value.split(":")
        if len(parts) == 2 and all(part.strip().isdigit() for part in parts):
            value = range(int(parts[0]), int(parts[1]))
        else:
            reject_invalid_state(
                "invalid burst selection "
                + repr(value)
                + " for "
                + swath
                + "; expected 'all' or 'start:stop'"
            )
    if isinstance(value, range):
        indices = list(value)
    else:
        indices = [int(index) for index in value]
    if not indices:
        reject_invalid_state("empty burst selection for " + swath)
    for index in indices:
        if index < 0 or index >= count:
            reject_invalid_state(
                "burst index "
                + str(index)
                + " out of range 0.."
                + str(count - 1)
                + " for "
                + swath
            )
    return sorted(set(indices))


def _normalize_burst_selection(
    bursts: BurstSelection | None,
    frame_count: int,
    swaths: tuple[str, ...],
    burst_counts: dict[tuple[int, str], int],
) -> dict[tuple[int, str], list[int]]:
    resolved: dict[tuple[int, str], list[int]] = {}
    if bursts is None:
        for frame_index in range(frame_count):
            for swath in swaths:
                resolved[(frame_index, swath)] = list(
                    range(burst_counts[(frame_index, swath)])
                )
        return resolved
    if isinstance(bursts, list):
        if len(bursts) != frame_count:
            reject_invalid_state(
                "per-frame burst selection has "
                + str(len(bursts))
                + " entries for "
                + str(frame_count)
                + " frames"
            )
        per_frame: list[dict[str, list[int] | range | str] | None] = list(bursts)
    else:
        per_frame = [bursts] * frame_count
    for frame_index, selection in enumerate(per_frame):
        if selection is None:
            for swath in swaths:
                resolved[(frame_index, swath)] = list(
                    range(burst_counts[(frame_index, swath)])
                )
            continue
        for swath in swaths:
            resolved[(frame_index, swath)] = _burst_index_list(
                selection.get(swath, "all"),
                swath,
                burst_counts[(frame_index, swath)],
            )
    return resolved


def _roi_geometry(roi: BoundingBox | Polygons) -> Any:
    from shapely.geometry import box

    if isinstance(roi, BoundingBox):
        return box(roi.left, roi.bottom, roi.right, roi.top)
    series = roi.geometry
    union = (
        series.union_all() if hasattr(series, "union_all") else series.unary_union
    )
    crs = getattr(roi, "crs", None)
    if crs is not None and str(crs) != "EPSG:4326":
        import geopandas as gpd

        return gpd.GeoSeries([union], crs=crs).to_crs("EPSG:4326").iloc[0]
    return union


def _auto_dem_bounds(
    roi: BoundingBox | Polygons | None,
    resolved: dict[tuple[int, str], list[int]],
    reference_products: list,
) -> tuple[float, float, float, float]:
    """Return EPSG:4326 bounds covering the ROI or the selected bursts.

    Parameters
    ----------
    roi : BoundingBox, Polygons, or None
        ROI passed to run_pair.
    resolved : dict
        Selected burst indices per (frame_index, swath).
    reference_products : list
        Opened reference SAFE products, one per frame.

    Returns
    -------
    tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat) with 0.01 deg padding when the
        bounds come from burst footprints.

    """
    if isinstance(roi, BoundingBox):
        return (
            float(roi.left),
            float(roi.bottom),
            float(roi.right),
            float(roi.top),
        )
    if isinstance(roi, Polygons):
        total = roi.to_geodataframe().total_bounds
        return (
            float(total[0]),
            float(total[1]),
            float(total[2]),
            float(total[3]),
        )
    lons: list[float] = []
    lats: list[float] = []
    for (frame_index, swath), indices in resolved.items():
        swath_obj = reference_products[frame_index].swath(swath)
        for burst_index in indices:
            footprint = swath_obj.bursts[burst_index].footprint
            if footprint is None:
                continue
            for lon, lat in footprint:
                lons.append(float(lon))
                lats.append(float(lat))
    if not lons:
        for product in reference_products:
            for swath_item in product.swaths:
                for burst in swath_item.bursts:
                    if burst.footprint is None:
                        continue
                    for lon, lat in burst.footprint:
                        lons.append(float(lon))
                        lats.append(float(lat))
    if not lons:
        reject_invalid_state("cannot derive DEM bounds: no burst footprints available")
    pad = 0.01
    return (
        min(lons) - pad,
        min(lats) - pad,
        max(lons) + pad,
        max(lats) + pad,
    )


def _select_bursts_by_roi(
    roi: BoundingBox | Polygons,
    frame_paths: list[Path],
    swaths: tuple[str, ...],
    orbits: Sequence[str | Path | None] | None = None,
    dem: DEMSampler | None = None,
) -> dict[tuple[int, str], list[int]]:
    """Select bursts whose radar-frame ground quad intersects the ROI."""
    from dataclasses import replace as _replace

    from shapely.geometry import Polygon as ShapelyPolygon

    from faninsar.missions.sentinel1 import read_eof_orbit
    from faninsar.missions.sentinel1.safe import open_safe_product
    from faninsar.processing.pipeline.geo_lut import burst_geo_quad_lonlat

    region = _roi_geometry(roi)
    resolved: dict[tuple[int, str], list[int]] = {}
    any_quad = False
    for frame_index, path in enumerate(frame_paths):
        product = open_safe_product(path)
        orbit_path = None if orbits is None else orbits[frame_index]
        for swath in swaths:
            s1_swath = product.swath(swath)
            if orbit_path is not None:
                s1_swath = _replace(s1_swath, orbit=read_eof_orbit(orbit_path))
            indices: list[int] = []
            shape = (s1_swath.lines_per_burst, s1_swath.samples_per_burst)
            for burst in s1_swath.bursts:
                geometry = _radar_model(
                    s1_swath,
                    burst,
                    shape=shape,
                    row0=burst.index * s1_swath.lines_per_burst,
                    col0=0,
                )
                quad = burst_geo_quad_lonlat(
                    geometry=geometry,
                    radar_shape=shape,
                    dem=dem,
                )
                if quad is None:
                    continue
                any_quad = True
                if region.intersects(ShapelyPolygon(quad)):
                    indices.append(burst.index)
            resolved[(frame_index, swath)] = indices
    if not any_quad:
        reject_invalid_state(
            "ROI selection requires radar geometry; burst rdr2geo quads "
            "could not be built for any burst"
        )
    return resolved


def _buffer_geometry_meters(
    geometry: object,
    target_crs: object,
    buffer_m: float,
) -> object:
    """Buffer a WGS84 shapely geometry by meters in a projected CRS."""
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    inverse = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    projected = shp_transform(
        lambda x, y: transformer.transform(x, y),
        geometry,
    )
    buffered = projected.buffer(buffer_m)
    return shp_transform(
        lambda x, y: inverse.transform(x, y),
        buffered,
    )


def _roi_burst_window(
    roi: BoundingBox | Polygons,
    geometry: RadarGeometryModel,
    dem: DEMSampler,
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    from faninsar.processing.geometry import geo2rdr

    region = _roi_geometry(roi)
    min_lon, min_lat, max_lon, max_lat = region.bounds
    lon = np.asarray([min_lon, max_lon, max_lon, min_lon], dtype=np.float64)
    lat = np.asarray([max_lat, max_lat, min_lat, min_lat], dtype=np.float64)
    height = float(np.mean(dem.sample(lat, lon)))
    transform = geo2rdr(geometry, lat, lon, height)
    azimuth = transform.azimuth_index
    range_index = transform.range_index
    ok = transform.converged & np.isfinite(azimuth) & np.isfinite(range_index)
    if not np.any(ok):
        return None
    row0 = max(0, int(np.floor(np.min(azimuth[ok]))))
    row1 = min(shape[0], int(np.ceil(np.max(azimuth[ok]))) + 1)
    col0 = max(0, int(np.floor(np.min(range_index[ok]))))
    col1 = min(shape[1], int(np.ceil(np.max(range_index[ok]))) + 1)
    if row1 <= row0 or col1 <= col0:
        return None
    return row0, row1, col0, col1


def _seconds_of_day(value: datetime) -> float:
    midnight = value.replace(hour=0, minute=0, second=0, microsecond=0)
    return (value - midnight).total_seconds()


def _common_burst_indices(
    reference_swath: S1Swath,
    secondary_swath: S1Swath,
) -> list[int]:
    common: list[int] = []
    max_index = min(len(reference_swath.bursts), len(secondary_swath.bursts))
    window_s = reference_swath.lines_per_burst * reference_swath.azimuth_time_interval_s
    for index in range(max_index):
        ref_burst = reference_swath.bursts[index]
        sec_burst = secondary_swath.bursts[index]
        if ref_burst.azimuth_time is None or sec_burst.azimuth_time is None:
            common.append(index)
            continue
        gap = abs(
            _seconds_of_day(ref_burst.azimuth_time)
            - _seconds_of_day(sec_burst.azimuth_time)
        )
        if gap > 12 * 3600:
            gap = 24 * 3600 - gap
        if gap <= window_s:
            common.append(index)
    return common


def _swath_range_offsets(
    swaths: tuple[str, ...],
    reference_products: Sequence[S1Product],
) -> dict[str, int]:
    first = reference_products[0].swath(swaths[0])
    offsets: dict[str, int] = {}
    for name in swaths:
        item = reference_products[0].swath(name)
        if name == swaths[0]:
            offsets[name] = 0
        else:
            offset = round(
                (item.slant_range_time_s - first.slant_range_time_s)
                * item.range_sampling_rate_hz
            )
            if offset < 0:
                reject_invalid_state(
                    "swath " + name + " starts before frame swath " + swaths[0]
                )
            offsets[name] = offset
    return offsets


@overload
def run_pair(
    reference_path: str | Path | Sequence[str | Path],
    secondary_path: str | Path | Sequence[str | Path],
    *,
    output_dir: str | Path,
    roi: BoundingBox | Polygons | None = None,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    dem: DEMSampler | None = None,
    multilook: tuple[int, int] | list[int] = (2, 10),
    overwrite: bool = False,
    goldstein_alpha: float = 0.5,
    dead_pixel_amp_threshold: float = 3.0,
    esd_enabled: bool = False,
    amplitude_refinement_enabled: bool = False,
    control_spacing: int | None = None,
    executor: str = "torch",
    device: str = "auto",
    coregistration_grid: CoregistrationGrid = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_height_m: float = 0.0,
    geo_chunk_size: int = 128,
    geo_work_dir: str | Path | None = None,
    n_jobs: int = 1,
    roi_buffer_m: float = 320.0,
    snaphu_config: SnaphuConfig | None = None,
    unwrap_method: UnwrapBackend | None = None,
    irls_kwargs: dict[str, Any] | None = None,
    reference_orbit_path: str | Path | Sequence[str | Path] | None = None,
    secondary_orbit_path: str | Path | Sequence[str | Path] | None = None,
    unwrap: bool = False,
    geoid_correction: bool = True,
) -> ProductionPairState: ...


@overload
def run_pair(
    reference_path: str | Path | Sequence[str | Path],
    secondary_path: str | Path | Sequence[str | Path],
    *,
    output_dir: str | Path,
    roi: BoundingBox | Polygons | None = None,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    dem: DEMSampler | None = None,
    multilook: Iterable[tuple[int, int]],
    overwrite: bool = False,
    goldstein_alpha: float = 0.5,
    dead_pixel_amp_threshold: float = 3.0,
    esd_enabled: bool = False,
    amplitude_refinement_enabled: bool = False,
    control_spacing: int | None = None,
    executor: str = "torch",
    device: str = "auto",
    coregistration_grid: CoregistrationGrid = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_height_m: float = 0.0,
    geo_chunk_size: int = 128,
    geo_work_dir: str | Path | None = None,
    n_jobs: int = 1,
    roi_buffer_m: float = 320.0,
    snaphu_config: SnaphuConfig | None = None,
    unwrap_method: UnwrapBackend | None = None,
    irls_kwargs: dict[str, Any] | None = None,
    reference_orbit_path: str | Path | Sequence[str | Path] | None = None,
    secondary_orbit_path: str | Path | Sequence[str | Path] | None = None,
    unwrap: bool = False,
    geoid_correction: bool = True,
) -> ProductionPairSweepResult: ...


def run_pair(
    reference_path: str | Path | Sequence[str | Path],
    secondary_path: str | Path | Sequence[str | Path],
    *,
    output_dir: str | Path,
    roi: BoundingBox | Polygons | None = None,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    dem: DEMSampler | None = None,
    multilook: tuple[int, int] | list[int] | Iterable[tuple[int, int]] = (2, 10),
    overwrite: bool = False,
    goldstein_alpha: float = 0.5,
    dead_pixel_amp_threshold: float = 3.0,
    esd_enabled: bool = False,
    amplitude_refinement_enabled: bool = False,
    control_spacing: int | None = None,
    executor: str = "torch",
    device: str = "auto",
    coregistration_grid: CoregistrationGrid = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_height_m: float = 0.0,
    geo_chunk_size: int = 128,
    geo_work_dir: str | Path | None = None,
    n_jobs: int = 1,
    roi_buffer_m: float = 320.0,
    snaphu_config: SnaphuConfig | None = None,
    unwrap_method: UnwrapBackend | None = None,
    irls_kwargs: dict[str, Any] | None = None,
    reference_orbit_path: str | Path | Sequence[str | Path] | None = None,
    secondary_orbit_path: str | Path | Sequence[str | Path] | None = None,
    unwrap: bool = False,
    geoid_correction: bool = True,
) -> ProductionPairState | ProductionPairSweepResult:
    """Process any burst selection across frames and swaths into one product.

    Parameters
    ----------
    reference_path, secondary_path : str, Path, or sequence
        One or more SAFE products per date. Sequences represent consecutive
        frames along the same pass; all entries are placed on one shared
        absolute azimuth/range grid.
    output_dir : path
        Output directory for the merged Zarr/STAC products.
    roi : BoundingBox or Polygons, optional
        Region of interest in EPSG:4326. When provided, swaths and bursts
        are ignored and the burst set is derived from footprint intersection
        (edge-aware; neighboring bursts are included when the ROI reaches a
        burst boundary because footprints overlap).
    swaths : tuple of str, optional
        Sub-swaths to process in range order. Defaults to all sub-swaths of
        the reference product ordered by slant-range time.
    bursts : BurstSelection, optional
        Explicit per-swath burst selection: {"IW1": [0, 2], "IW2": "1:4",
        "IW3": "all"} applied to every frame, or a list with one such
        mapping per frame. None selects every burst.
    dem : DEMSampler, optional
        DEM for coregistration, flattening, and geometry. Defaults to a zero
        ellipsoid.
    multilook : tuple[int, int] or iterable of pairs, optional
        One (az, rg) pair for a single product, or an iterable of pairs to
        emit one product per configuration (sweep mode).
    overwrite : bool, optional
        Sweep mode only: replace existing looks_{az}x{rg} subtrees.
    goldstein_alpha : float, optional
        Goldstein filter exponent applied to the merged product.
    dead_pixel_amp_threshold : float, optional
        Dead-pixel amplitude mask threshold for the interferogram.
    esd_enabled : bool, optional
        Enable spectral-diversity azimuth residual estimation.
    amplitude_refinement_enabled : bool, optional
        Enable amplitude-correlation residual refinement.
    control_spacing : int, optional
        Geometry control-point spacing.
    executor : {"torch"}, optional
        Resample executor.
    device : {"auto","cpu","cuda","mps"}, optional
        Torch device.
    coregistration_grid : {"radar", "geo"}, optional
        Coordinate grid on which the pair is coregistered. Geo requires
        geo_grid and is available for single-config runs.
    geo_grid : GeoGridSpec, optional
        Geographic product grid required for coregistration_grid="geo".
    geo_height_m : float, optional
        Fallback constant height (m) when the DEM is unavailable for geo2rdr.
    geo_chunk_size : int, optional
        Row chunk shared by geo2rdr and SLC remapping in geo mode.
    geo_work_dir : path, optional
        Working directory for geo memmaps; a temporary directory is used
        when omitted.
    n_jobs : int, optional
        Number of parallel burst workers in geo mode (default 1).
    roi_buffer_m : float, optional
        Physical margin in meters kept around the ROI in geo mode
        (default 320).
    snaphu_config : SnaphuConfig, optional
        SNAPHU configuration; nlooks defaults to az * rg per config.
    unwrap_method : {"irls", "snaphu"}, optional
        Unwrapping backend when unwrap=True.
    irls_kwargs : dict, optional
        Arguments forwarded to the IRLS unwrap backend.
    reference_orbit_path, secondary_orbit_path : path or sequence, optional
        Precise ESA EOF orbits; one per frame or a single orbit reused for
        every frame.
    unwrap : bool, optional
        Run unwrapping on the merged wrapped phase when True.
    geoid_correction : bool, optional
        Convert orthometric raster DEM heights to ellipsoidal with EGM96.
        Default True.

    Returns
    -------
    ProductionPairState or ProductionPairSweepResult
        Single-config state, or per-config outcomes for a sweep.

    """
    if not _is_multilook_pair(multilook) or coregistration_grid == "geo":
        return _run_pair_sweep(
            reference_path,
            secondary_path,
            output_dir=output_dir,
            roi=roi,
            swaths=swaths,
            bursts=bursts,
            dem=dem,
            multilook=multilook,
            overwrite=overwrite,
            goldstein_alpha=goldstein_alpha,
            dead_pixel_amp_threshold=dead_pixel_amp_threshold,
            esd_enabled=esd_enabled,
            amplitude_refinement_enabled=amplitude_refinement_enabled,
            control_spacing=control_spacing,
            executor=executor,
            device=device,
            coregistration_grid=coregistration_grid,
            geo_grid=geo_grid,
            geo_height_m=geo_height_m,
            geo_chunk_size=geo_chunk_size,
            geo_work_dir=geo_work_dir,
            n_jobs=n_jobs,
            roi_buffer_m=roi_buffer_m,
            snaphu_config=snaphu_config,
            unwrap_method=unwrap_method,
            irls_kwargs=irls_kwargs,
            reference_orbit_path=reference_orbit_path,
            secondary_orbit_path=secondary_orbit_path,
            unwrap=unwrap,
            geoid_correction=geoid_correction,
        )
    from faninsar.missions.sentinel1.safe import open_safe_product
    from faninsar.processing.geometry.egm96 import EGM96Geoid

    dem_sampler: DEMSampler = dem if dem is not None else ConstantHeightDEM(0.0)
    if geoid_correction and isinstance(dem_sampler, RasterDEM):
        dem_sampler = GeoidAdjustedDEM(dem_sampler, EGM96Geoid())

    ref_paths = _as_frame_paths(reference_path, "reference_path")
    sec_paths = _as_frame_paths(secondary_path, "secondary_path")
    if len(ref_paths) != len(sec_paths):
        reject_invalid_state(
            "reference/secondary frame counts differ: "
            + str(len(ref_paths))
            + " vs "
            + str(len(sec_paths))
        )
    frame_count = len(ref_paths)
    ref_orbits = _as_optional_frame_sequence(
        reference_orbit_path, frame_count, "reference_orbit_path"
    )
    sec_orbits = _as_optional_frame_sequence(
        secondary_orbit_path, frame_count, "secondary_orbit_path"
    )

    reference_products = [open_safe_product(path) for path in ref_paths]
    secondary_products = [open_safe_product(path) for path in sec_paths]
    if swaths is None:
        ordered = sorted(
            reference_products[0].swaths,
            key=lambda item: item.slant_range_time_s,
        )
        swath_tuple = tuple(item.swath for item in ordered)
    else:
        swath_tuple = tuple(swaths)
    if not swath_tuple:
        reject_invalid_state("swaths must not be empty")
    for frame_index, product in enumerate(reference_products):
        present = {item.swath for item in product.swaths}
        missing = [name for name in swath_tuple if name not in present]
        if missing:
            reject_invalid_state(
                "reference frame "
                + str(frame_index)
                + " ("
                + ref_paths[frame_index].name
                + ") missing swaths "
                + str(missing)
                + "; available="
                + str(sorted(present))
            )
    for frame_index, product in enumerate(secondary_products):
        present = {item.swath for item in product.swaths}
        missing = [name for name in swath_tuple if name not in present]
        if missing:
            reject_invalid_state(
                "secondary frame "
                + str(frame_index)
                + " ("
                + sec_paths[frame_index].name
                + ") missing swaths "
                + str(missing)
                + "; available="
                + str(sorted(present))
            )

    if roi is not None:
        logger.info("run_pair: ROI provided; explicit swaths/bursts selection ignored")
        ordered = sorted(
            reference_products[0].swaths,
            key=lambda item: item.slant_range_time_s,
        )
        swath_tuple = tuple(item.swath for item in ordered)
        resolved = _select_bursts_by_roi(
            roi,
            ref_paths,
            swath_tuple,
            orbits=ref_orbits,
            dem=dem_sampler,
        )
    else:
        burst_counts = {
            (frame_index, swath): len(
                reference_products[frame_index].swath(swath).bursts
            )
            for frame_index in range(frame_count)
            for swath in swath_tuple
        }
        resolved = _normalize_burst_selection(
            bursts, frame_count, swath_tuple, burst_counts
        )

    common_aligned: dict[tuple[int, str], list[int]] = {}
    for frame_index in range(frame_count):
        for swath in swath_tuple:
            reference_swath = reference_products[frame_index].swath(swath)
            secondary_swath = secondary_products[frame_index].swath(swath)
            common = _common_burst_indices(reference_swath, secondary_swath)
            selected = [
                index for index in resolved[(frame_index, swath)] if index in common
            ]
            if not selected:
                if roi is not None:
                    continue
                reject_invalid_state(
                    "no common bursts between reference/secondary frame "
                    + str(frame_index)
                    + " swath "
                    + swath
                )
            common_aligned[(frame_index, swath)] = selected
    resolved = common_aligned
    if not any(indices for indices in resolved.values()):
        reject_invalid_state("selection contains no bursts")

    if dem is None and os.environ.get("FANINSAR_DEM_CACHE_DIR"):
        from faninsar.processing.geometry.dem_manager import (
            default_dem_name,
            get_dem_manager,
        )

        bounds = _auto_dem_bounds(roi, resolved, reference_products)
        dem_path = get_dem_manager().fetch_dem(
            bounds, Path(output_dir) / "dem" / default_dem_name()
        )
        logger.info("Automatic DEM built for %s: %s", bounds, dem_path)
        dem_sampler = RasterDEM(dem_path, interpolation="biquintic")
        if geoid_correction:
            dem_sampler = GeoidAdjustedDEM(dem_sampler, EGM96Geoid())

    range_offsets = _swath_range_offsets(swath_tuple, reference_products)
    reference_swath0 = reference_products[0].swath(swath_tuple[0])
    dt = reference_swath0.azimuth_time_interval_s
    burst_lines = {
        swath: reference_products[0].swath(swath).lines_per_burst
        for swath in swath_tuple
    }
    burst_width = {
        swath: reference_products[0].swath(swath).samples_per_burst
        for swath in swath_tuple
    }

    azimuth_origin = min(
        reference_products[frame_index].swath(swath).bursts[burst_index].azimuth_time
        for (frame_index, swath), indices in resolved.items()
        for burst_index in indices
    )
    units_by_swath: dict[str, list[tuple[int, int, int]]] = {}
    for swath in swath_tuple:
        units: list[tuple[int, int, int]] = []
        for (frame_index, swath_key), indices in resolved.items():
            if swath_key != swath:
                continue
            swath_obj = reference_products[frame_index].swath(swath)
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

    selected_unit_count = sum(len(units) for units in units_by_swath.values())
    total_started = time.perf_counter()
    frame_rows = max(
        azimuth_offset + burst_lines[swath]
        for swath, units in units_by_swath.items()
        for _, _, azimuth_offset in units
    )
    frame_cols = max(range_offsets[swath] + burst_width[swath] for swath in swath_tuple)
    if not _is_multilook_pair(multilook):
        reject_invalid_state("internal error: single-config multilook must be a pair")
    az_looks, rg_looks = int(multilook[0]), int(multilook[1])
    out_rows = frame_rows // az_looks
    out_cols = frame_cols // rg_looks
    swath_rows = {
        swath: (
            max(
                azimuth_offset + burst_lines[swath]
                for _, _, azimuth_offset in units_by_swath[swath]
            )
            // az_looks
        )
        for swath in swath_tuple
        if units_by_swath[swath]
    }
    swath_cols = {
        swath: (range_offsets[swath] + burst_width[swath]) // rg_looks
        for swath in swath_tuple
    }
    ifc_acc = {
        swath: np.zeros((swath_rows[swath], swath_cols[swath]), dtype=np.complex128)
        for swath in swath_tuple
        if swath in swath_rows
    }
    pri_pow_acc = {
        swath: np.zeros((swath_rows[swath], swath_cols[swath]), dtype=np.float64)
        for swath in swath_tuple
        if swath in swath_rows
    }
    sec_pow_acc = {
        swath: np.zeros((swath_rows[swath], swath_cols[swath]), dtype=np.float64)
        for swath in swath_tuple
        if swath in swath_rows
    }
    claimed = {
        swath: np.zeros((swath_rows[swath], swath_cols[swath]), dtype=np.int32)
        for swath in swath_tuple
        if swath in swath_rows
    }
    looks_per_window = az_looks * rg_looks

    def load_burst(
        swath: str,
        burst_index: int,
        path: Path,
        orbit_path: Path | None,
        product: S1Product,
    ) -> ProductionScene:
        s1_swath = product.swath(swath)
        if orbit_path is not None:
            s1_swath = replace(s1_swath, orbit=read_eof_orbit(orbit_path))
        array = read_full_burst(s1_swath, burst_index=burst_index, full_range=True)
        burst = s1_swath.bursts[burst_index]
        carrier = carrier_from_swath(s1_swath, burst, first_range_sample=array.col0)
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

    per_burst_timings: dict[str, dict[str, float]] = {}
    first_state: ProductionPairState | None = None
    origin_state: ProductionPairState | None = None
    for swath in reversed(swath_tuple):
        for frame_index, burst_index, azimuth_offset in units_by_swath[swath]:
            tag = "f" + str(frame_index) + "_" + swath + "_b" + str(burst_index)
            ref = load_burst(
                swath,
                burst_index,
                ref_paths[frame_index],
                ref_orbits[frame_index],
                reference_products[frame_index],
            )
            sec = load_burst(
                swath,
                burst_index,
                sec_paths[frame_index],
                sec_orbits[frame_index],
                secondary_products[frame_index],
            )
            state = ProductionPairState(
                pair_id=ref.scene_id + "_" + sec.scene_id + "_" + tag,
                reference=ref,
                secondary=sec,
                dem=dem_sampler,
                coregistration_grid="radar",
                multilook=multilook,
                goldstein_alpha=0.0,
                unwrap_method="snaphu",
            )
            if first_state is None:
                first_state = state
            if origin_state is None:
                origin_state = state
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
            burst_row0 = 0
            burst_col0 = 0
            if roi is not None:
                assert state.reference_deramped is not None
                window = _roi_burst_window(
                    roi,
                    ref.geometry,
                    dem_sampler,
                    state.reference_deramped.shape,
                )
                if window is not None:
                    burst_row0, burst_row1, burst_col0, burst_col1 = window
                    state.reference_deramped = state.reference_deramped[
                        burst_row0:burst_row1, burst_col0:burst_col1
                    ]
                    state.secondary_aligned = state.secondary_aligned[
                        burst_row0:burst_row1, burst_col0:burst_col1
                    ]
            assert state.reference_deramped is not None
            assert state.secondary_aligned is not None
            pri_power = (
                state.reference_deramped.real**2 + state.reference_deramped.imag**2
            )
            sec_power = (
                state.secondary_aligned.real**2 + state.secondary_aligned.imag**2
            )
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
                reject_invalid_state(tag + ": no interferogram produced")
            per_burst_timings[tag] = stage_times

            valid = np.abs(ifg_full) > 0
            rows = azimuth_offset + burst_row0 + np.arange(ifg_full.shape[0])
            cols = range_offsets[swath] + burst_col0 + np.arange(ifg_full.shape[1])
            orow = rows[:, None] // az_looks
            ocol = cols[None, :] // rg_looks
            ocol_local = ocol - range_offsets[swath] // rg_looks
            inb = (
                (orow < swath_rows[swath])
                & (ocol_local >= 0)
                & (ocol_local < swath_cols[swath])
                & valid
            )
            r_i, c_i = np.broadcast_arrays(orow, ocol_local)
            r_v, c_v = r_i[inb], c_i[inb]
            free = claimed[swath][r_v, c_v] < looks_per_window
            r_f, c_f = r_v[free], c_v[free]
            np.add.at(
                ifc_acc[swath],
                (r_f, c_f),
                ifg_full[inb][free].astype(np.complex128),
            )
            np.add.at(pri_pow_acc[swath], (r_f, c_f), pri_power[inb][free])
            np.add.at(sec_pow_acc[swath], (r_f, c_f), sec_power[inb][free])
            np.add.at(claimed[swath], (r_f, c_f), 1)
            del pri_power, sec_power, ifg_full, state
            logger.info("Merged %s into frame", tag)

    if first_state is None:
        reject_invalid_state("no burst units selected for processing")

    merged_ifg = np.zeros((out_rows, out_cols), dtype=np.complex64)
    coherence = np.full((out_rows, out_cols), np.nan, dtype=np.float32)
    for swath, swath_acc in ifc_acc.items():
        has = claimed[swath] > 0
        swath_ifg = np.where(
            has, swath_acc / np.where(has, claimed[swath], 1), 0
        ).astype(np.complex64)
        pri_ml = pri_pow_acc[swath] / np.where(has, claimed[swath], 1)
        sec_ml = sec_pow_acc[swath] / np.where(has, claimed[swath], 1)
        denom = np.sqrt(np.maximum(pri_ml * sec_ml, 1e-30))
        swath_coh = np.clip(np.abs(swath_ifg) / denom, 0.0, 1.0).astype(np.float32)
        col0 = range_offsets[swath] // rg_looks
        rows = swath_ifg.shape[0]
        cols = swath_ifg.shape[1]
        col1 = min(col0 + cols, out_cols)
        cols = col1 - col0
        band = np.abs(swath_ifg[:rows, :cols]) > 0
        merged_ifg[:rows, col0:col1][band] = swath_ifg[:rows, :cols][band]
        coherence[:rows, col0:col1][band] = swath_coh[:rows, :cols][band]
        del swath_ifg, swath_coh
    invalid = np.abs(merged_ifg) <= 0
    wrapped = np.where(invalid, np.nan, np.angle(merged_ifg).astype(np.float32))

    filtered = merged_ifg
    if goldstein_alpha > 0.0:
        from faninsar.processing.interferometry.pair import goldstein_filter

        filtered = goldstein_filter(merged_ifg, alpha=goldstein_alpha)

    result = ProductionPairState(
        pair_id=_scene_id(ref_paths[0]) + "_" + _scene_id(sec_paths[0]) + "_pair",
        reference=(
            origin_state.reference
            if origin_state is not None
            else first_state.reference
        ),
        secondary=(
            origin_state.secondary
            if origin_state is not None
            else first_state.secondary
        ),
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
        "PAIR frames="
        + str(frame_count)
        + " swaths="
        + str(swath_tuple)
        + " units="
        + str(selected_unit_count)
        + " merged="
        + str(merged_ifg.shape)
        + " multilook="
        + str(multilook)
        + " valid="
        + format(float((~invalid).mean()), ".3f")
        + " mean_coh="
        + format(float(np.nanmean(coherence)), ".3f")
    )
    if unwrap:
        from faninsar.processing.unwrap import SnaphuConfig

        method: UnwrapBackend = unwrap_method if unwrap_method is not None else "snaphu"
        config_for_unwrap = snaphu_config
        if config_for_unwrap is None:
            config_for_unwrap = SnaphuConfig(nlooks=float(az_looks * rg_looks))
        result = stage_unwrap(
            result,
            method=method,
            config=config_for_unwrap,
            irls_kwargs=irls_kwargs,
        )
        result.note("UNWRAP complete on merged product")
    else:
        result.unwrapped_phase = np.zeros_like(wrapped, dtype=np.float32)
        result.connected_components = np.zeros_like(wrapped, dtype=np.uint8)
        result.note("UNWRAP skipped (unwrap=False)")
    stage_write(result, output_dir)
    return result


def _run_pair_sweep(
    reference_path: str | Path | Sequence[str | Path],
    secondary_path: str | Path | Sequence[str | Path],
    *,
    output_dir: str | Path,
    roi: BoundingBox | Polygons | None,
    swaths: tuple[str, ...] | None,
    bursts: BurstSelection | None,
    dem: DEMSampler | None,
    multilook: object,
    overwrite: bool,
    goldstein_alpha: float,
    dead_pixel_amp_threshold: float,
    esd_enabled: bool,
    amplitude_refinement_enabled: bool,
    control_spacing: int | None,
    executor: str,
    device: str,
    coregistration_grid: CoregistrationGrid,
    geo_grid: GeoGridSpec | None,
    geo_height_m: float,
    geo_chunk_size: int,
    geo_work_dir: str | Path | None,
    n_jobs: int = 1,
    roi_buffer_m: float = 320.0,
    snaphu_config: SnaphuConfig | None,
    unwrap_method: UnwrapBackend | None,
    irls_kwargs: dict[str, Any] | None,
    reference_orbit_path: str | Path | Sequence[str | Path] | None,
    secondary_orbit_path: str | Path | Sequence[str | Path] | None,
    unwrap: bool,
    geoid_correction: bool,
) -> ProductionPairState | ProductionPairSweepResult:
    """Run one shared prefix and emit every look configuration."""
    from faninsar.missions.sentinel1.safe import open_safe_product
    from faninsar.processing.geometry.egm96 import EGM96Geoid

    single_config = _is_multilook_pair(multilook)
    if single_config:
        configs = [tuple(int(part) for part in multilook)]
    else:
        configs = normalize_multilook_sweep(multilook)
    if coregistration_grid == "geo" and geo_grid is None:
        reject_invalid_state("coregistration_grid='geo' requires geo_grid")
    output_root = Path(output_dir)
    existing = [
        looks_dir(az, rg)
        for az, rg in configs
        if (output_root / looks_dir(az, rg)).exists()
    ]
    if existing and not overwrite:
        reject_invalid_state(
            "multilook sweep targets already exist; pass overwrite=True: "
            + ", ".join(existing)
        )
    if overwrite:
        for az_looks, rg_looks in configs:
            stale = output_root / looks_dir(az_looks, rg_looks)
            if stale.exists():
                shutil.rmtree(stale)

    dem_sampler: DEMSampler = dem if dem is not None else ConstantHeightDEM(0.0)
    if geoid_correction and isinstance(dem_sampler, RasterDEM):
        dem_sampler = GeoidAdjustedDEM(dem_sampler, EGM96Geoid())

    ref_paths = _as_frame_paths(reference_path, "reference_path")
    sec_paths = _as_frame_paths(secondary_path, "secondary_path")
    if len(ref_paths) != len(sec_paths):
        reject_invalid_state(
            "reference/secondary frame counts differ: "
            + str(len(ref_paths))
            + " vs "
            + str(len(sec_paths))
        )
    frame_count = len(ref_paths)
    ref_orbits = _as_optional_frame_sequence(
        reference_orbit_path, frame_count, "reference_orbit_path"
    )
    sec_orbits = _as_optional_frame_sequence(
        secondary_orbit_path, frame_count, "secondary_orbit_path"
    )
    reference_products = [open_safe_product(path) for path in ref_paths]
    secondary_products = [open_safe_product(path) for path in sec_paths]
    if swaths is None:
        ordered = sorted(
            reference_products[0].swaths,
            key=lambda item: item.slant_range_time_s,
        )
        swath_tuple = tuple(item.swath for item in ordered)
    else:
        swath_tuple = tuple(swaths)
    if not swath_tuple:
        reject_invalid_state("swaths must not be empty")
    for frame_index, product in enumerate(reference_products):
        present = {item.swath for item in product.swaths}
        missing = [name for name in swath_tuple if name not in present]
        if missing:
            reject_invalid_state(
                "reference frame "
                + str(frame_index)
                + " ("
                + ref_paths[frame_index].name
                + ") missing swaths "
                + str(missing)
                + "; available="
                + str(sorted(present))
            )
    for frame_index, product in enumerate(secondary_products):
        present = {item.swath for item in product.swaths}
        missing = [name for name in swath_tuple if name not in present]
        if missing:
            reject_invalid_state(
                "secondary frame "
                + str(frame_index)
                + " ("
                + sec_paths[frame_index].name
                + ") missing swaths "
                + str(missing)
                + "; available="
                + str(sorted(present))
            )

    if roi is not None:
        logger.info("run_pair: ROI provided; explicit swaths/bursts selection ignored")
        ordered = sorted(
            reference_products[0].swaths,
            key=lambda item: item.slant_range_time_s,
        )
        swath_tuple = tuple(item.swath for item in ordered)
        resolved = _select_bursts_by_roi(
            roi,
            ref_paths,
            swath_tuple,
            orbits=ref_orbits,
            dem=dem_sampler,
        )
    else:
        burst_counts = {
            (frame_index, swath): len(
                reference_products[frame_index].swath(swath).bursts
            )
            for frame_index in range(frame_count)
            for swath in swath_tuple
        }
        resolved = _normalize_burst_selection(
            bursts, frame_count, swath_tuple, burst_counts
        )

    common_aligned: dict[tuple[int, str], list[int]] = {}
    for frame_index in range(frame_count):
        for swath in swath_tuple:
            reference_swath = reference_products[frame_index].swath(swath)
            secondary_swath = secondary_products[frame_index].swath(swath)
            common = _common_burst_indices(reference_swath, secondary_swath)
            selected = [
                index for index in resolved[(frame_index, swath)] if index in common
            ]
            if not selected:
                if roi is not None:
                    continue
                reject_invalid_state(
                    "no common bursts between reference/secondary frame "
                    + str(frame_index)
                    + " swath "
                    + swath
                )
            common_aligned[(frame_index, swath)] = selected
    resolved = common_aligned
    if not any(indices for indices in resolved.values()):
        reject_invalid_state("selection contains no bursts")

    if dem is None and os.environ.get("FANINSAR_DEM_CACHE_DIR"):
        from faninsar.processing.geometry.dem_manager import (
            default_dem_name,
            get_dem_manager,
        )

        bounds = _auto_dem_bounds(roi, resolved, reference_products)
        dem_path = get_dem_manager().fetch_dem(
            bounds, output_root / "dem" / default_dem_name()
        )
        logger.info("Automatic DEM built for %s: %s", bounds, dem_path)
        dem_sampler = RasterDEM(dem_path, interpolation="biquintic")
        if geoid_correction:
            dem_sampler = GeoidAdjustedDEM(dem_sampler, EGM96Geoid())

    range_offsets = _swath_range_offsets(swath_tuple, reference_products)
    reference_swath0 = reference_products[0].swath(swath_tuple[0])
    dt = reference_swath0.azimuth_time_interval_s
    burst_lines = {
        swath: reference_products[0].swath(swath).lines_per_burst
        for swath in swath_tuple
    }
    burst_width = {
        swath: reference_products[0].swath(swath).samples_per_burst
        for swath in swath_tuple
    }
    azimuth_origin = min(
        reference_products[frame_index].swath(swath).bursts[burst_index].azimuth_time
        for (frame_index, swath), indices in resolved.items()
        for burst_index in indices
    )
    units_by_swath: dict[str, list[tuple[int, int, int]]] = {}
    for swath in swath_tuple:
        units: list[tuple[int, int, int]] = []
        for (frame_index, swath_key), indices in resolved.items():
            if swath_key != swath:
                continue
            swath_obj = reference_products[frame_index].swath(swath)
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

    frame_rows = max(
        azimuth_offset + burst_lines[swath]
        for swath, units in units_by_swath.items()
        for _, _, azimuth_offset in units
    )
    frame_cols = max(range_offsets[swath] + burst_width[swath] for swath in swath_tuple)

    temporary = tempfile.TemporaryDirectory(prefix="faninsar-sweep-")
    resolved_geo_work_dir: Path | None = None
    if coregistration_grid == "geo":
        if geo_work_dir is None:
            resolved_geo_work_dir = Path(temporary.name)
        else:
            resolved_geo_work_dir = Path(geo_work_dir)
            resolved_geo_work_dir.mkdir(parents=True, exist_ok=True)
    resources = SharedPairResources(temporary_directory=temporary)
    try:
        archive = _archive_burst_ifgs(
            Path(temporary.name),
            ref_paths=ref_paths,
            sec_paths=sec_paths,
            ref_orbits=ref_orbits,
            sec_orbits=sec_orbits,
            reference_products=reference_products,
            secondary_products=secondary_products,
            swath_tuple=swath_tuple,
            units_by_swath=units_by_swath,
            roi=roi,
            dem_sampler=dem_sampler,
            control_spacing=control_spacing,
            esd_enabled=esd_enabled,
            amplitude_refinement_enabled=amplitude_refinement_enabled,
            executor=executor,
            device=device,
            dead_pixel_amp_threshold=dead_pixel_amp_threshold,
            coregistration_grid=coregistration_grid,
            geo_grid=geo_grid,
            geo_height_m=geo_height_m,
            geo_chunk_size=geo_chunk_size,
            geo_work_dir=resolved_geo_work_dir,
            n_jobs=n_jobs,
            roi_buffer_m=roi_buffer_m,
        )
        resources.ifg_archive = archive
        if coregistration_grid == "geo":
            resources.prefix_state = archive.get("geo_prefix_state")
        pair_id = _scene_id(ref_paths[0]) + "_" + _scene_id(sec_paths[0]) + "_pair"
        per_config: dict[tuple[int, int], PairSweepOutcome] = {}
        single_state: ProductionPairState | None = None
        for config in configs:
            az_looks, rg_looks = config
            merged = _merge_burst_ifgs(
                archive,
                swath_tuple=swath_tuple,
                range_offsets=range_offsets,
                burst_lines=burst_lines,
                burst_width=burst_width,
                frame_rows=frame_rows,
                frame_cols=frame_cols,
                az_looks=az_looks,
                rg_looks=rg_looks,
                geo_grid=geo_grid,
            )
            outcome, state = _finalize_sweep_config(
                merged,
                config=config,
                pair_id=pair_id,
                output_root=output_root,
                goldstein_alpha=goldstein_alpha,
                snaphu_config=snaphu_config,
                unwrap_method=unwrap_method,
                irls_kwargs=irls_kwargs,
                unwrap=unwrap,
                geo_height_m=geo_height_m,
                all_configs=configs,
            )
            per_config[config] = outcome
            if single_config:
                single_state = state
        if single_config:
            assert single_state is not None
            return single_state
        return ProductionPairSweepResult(pair_id=pair_id, per_config=per_config)
    finally:
        resources.cleanup()


def _archive_burst_ifgs(
    work_dir: Path,
    *,
    ref_paths: Sequence[str | Path],
    sec_paths: Sequence[str | Path],
    ref_orbits: Sequence[Path | None],
    sec_orbits: Sequence[Path | None],
    reference_products: Sequence[object],
    secondary_products: Sequence[object],
    swath_tuple: tuple[str, ...],
    units_by_swath: dict[str, list[tuple[int, int, int]]],
    roi: BoundingBox | Polygons | None,
    dem_sampler: DEMSampler,
    control_spacing: int | None,
    esd_enabled: bool,
    amplitude_refinement_enabled: bool,
    executor: str,
    device: str,
    dead_pixel_amp_threshold: float,
    coregistration_grid: CoregistrationGrid,
    geo_grid: GeoGridSpec | None,
    geo_height_m: float,
    geo_chunk_size: int,
    geo_work_dir: Path | None,
    n_jobs: int = 1,
    roi_buffer_m: float = 320.0,
) -> dict[str, Any]:
    """Process every burst unit once and store flat IFGs on disk.

    Radar units are archived in radar geometry with burst placement offsets.
    Geographic units (``coregistration_grid="geo"``) are coregistered onto
    the shared ``geo_grid`` and archived as valid-bounding-box crops of the
    geocoded interferogram, per-unit power, and DEM height field.
    """
    from faninsar.missions.sentinel1 import read_eof_orbit, read_full_burst
    from faninsar.processing.tops.carrier import carrier_from_swath

    if coregistration_grid == "geo" and (geo_grid is None or geo_work_dir is None):
        reject_invalid_state("geo coregistration requires geo_grid and geo_work_dir")

    ifg_dir = work_dir / "ifgs"
    ifg_dir.mkdir(parents=True, exist_ok=True)
    units: list[dict[str, Any]] = []
    per_burst_timings: dict[str, dict[str, float]] = {}
    origin_state: ProductionPairState | None = None
    geo_prefix_state: ProductionPairState | None = None
    prefix_started = time.perf_counter()

    def load_burst(
        swath: str,
        burst_index: int,
        path: Path,
        orbit_path: Path | None,
        product: object,
    ) -> ProductionScene:
        s1_swath = product.swath(swath)
        if orbit_path is not None:
            s1_swath = replace(s1_swath, orbit=read_eof_orbit(orbit_path))
        array = read_full_burst(s1_swath, burst_index=burst_index, full_range=True)
        burst = s1_swath.bursts[burst_index]
        carrier = carrier_from_swath(s1_swath, burst, first_range_sample=array.col0)
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

    def scene_rebuild_args(
        frame_index: int,
        swath: str,
        burst_index: int,
    ) -> dict[str, object]:
        return {
            "ref_path": ref_paths[frame_index],
            "sec_path": sec_paths[frame_index],
            "ref_orbit": ref_orbits[frame_index],
            "sec_orbit": sec_orbits[frame_index],
            "swath": swath,
            "burst_index": burst_index,
            "frame_index": frame_index,
        }

    def rebuild_origin_state(
        scene_args: dict[str, object],
        dem: DEMSampler,
    ) -> ProductionPairState:
        ref = load_burst(
            str(scene_args["swath"]),
            int(scene_args["burst_index"]),
            Path(scene_args["ref_path"]),
            scene_args["ref_orbit"],
            reference_products[int(scene_args["frame_index"])],
        )
        sec = load_burst(
            str(scene_args["swath"]),
            int(scene_args["burst_index"]),
            Path(scene_args["sec_path"]),
            scene_args["sec_orbit"],
            secondary_products[int(scene_args["frame_index"])],
        )
        return ProductionPairState(
            pair_id=ref.scene_id + "_" + sec.scene_id,
            reference=ref,
            secondary=sec,
            dem=dem,
            coregistration_grid=coregistration_grid,
            multilook=(1, 1),
            goldstein_alpha=0.0,
            unwrap_method="snaphu",
        )

    task_args: list[dict[str, object]] = []
    for swath in reversed(swath_tuple):
        for frame_index, burst_index, azimuth_offset in units_by_swath[swath]:
            tag = "f" + str(frame_index) + "_" + swath + "_b" + str(burst_index)
            task_args.append(
                {
                    "tag": tag,
                    "swath": swath,
                    "frame_index": frame_index,
                    "burst_index": burst_index,
                    "azimuth_offset": azimuth_offset,
                    "ref_path": ref_paths[frame_index],
                    "sec_path": sec_paths[frame_index],
                    "ref_orbit": ref_orbits[frame_index],
                    "sec_orbit": sec_orbits[frame_index],
                    "roi": roi,
                    "control_spacing": control_spacing,
                    "esd_enabled": esd_enabled,
                    "amplitude_refinement_enabled": amplitude_refinement_enabled,
                    "executor": executor,
                    "device": device,
                    "dead_pixel_amp_threshold": dead_pixel_amp_threshold,
                    "coregistration_grid": coregistration_grid,
                    "geo_grid": geo_grid,
                    "geo_height_m": geo_height_m,
                    "geo_chunk_size": geo_chunk_size,
                    "roi_buffer_m": roi_buffer_m,
                    "ifg_dir": ifg_dir,
                    "dem": dem_sampler,
                    "geo_work_dir": geo_work_dir,
                }
            )

    if n_jobs > 1 and len(task_args) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_process_burst_worker, task_args))
    else:
        results = [_process_burst_worker(task) for task in task_args]

    ordered_results: list[tuple[dict[str, object], dict[str, float]]] = []
    origin_scene_args: dict[str, object] | None = None
    for task, result in zip(task_args, results, strict=True):
        unit = result["unit"]
        if unit is None:
            continue
        ordered_results.append((unit, result["stage_times"]))
        if origin_scene_args is None:
            origin_scene_args = scene_rebuild_args(
                int(task["frame_index"]),
                str(task["swath"]),
                int(task["burst_index"]),
            )
    for unit, stage_times in ordered_results:
        units.append(unit)
        per_burst_timings[unit["tag"]] = stage_times
    if origin_scene_args is not None:
        origin_state = rebuild_origin_state(origin_scene_args, dem_sampler)

    if origin_state is None:
        reject_invalid_state("no burst units selected for processing")
    return {
        "units": units,
        "origin_state": origin_state,
        "per_burst_timings": per_burst_timings,
        "prefix_started": prefix_started,
        "grid_mode": "geo" if coregistration_grid == "geo" else "radar",
        "geo_grid": geo_grid,
        "geo_prefix_state": geo_prefix_state,
    }


def _merge_burst_ifgs(
    archive: dict[str, Any],
    *,
    swath_tuple: tuple[str, ...],
    range_offsets: dict[str, int],
    burst_lines: dict[str, int],
    burst_width: dict[str, int],
    frame_rows: int,
    frame_cols: int,
    az_looks: int,
    rg_looks: int,
    geo_grid: GeoGridSpec | None = None,
) -> dict[str, Any]:
    """Replay archived flat IFGs into one merged product for a look config."""
    units = archive["units"]
    if archive.get("grid_mode") == "geo":
        if geo_grid is None:
            reject_invalid_state("geo merge requires geo_grid")
        return _merge_geo_ifgs(
            archive,
            geo_grid=geo_grid,
            az_looks=az_looks,
            rg_looks=rg_looks,
        )
    out_rows = frame_rows // az_looks
    out_cols = frame_cols // rg_looks
    swath_rows = {
        swath: (
            max(
                unit["azimuth_offset"] + burst_lines[swath]
                for unit in units
                if unit["swath"] == swath
            )
            // az_looks
        )
        for swath in swath_tuple
    }
    swath_cols = {
        swath: (range_offsets[swath] + burst_width[swath]) // rg_looks
        for swath in swath_tuple
    }
    ifc_acc = {
        swath: np.zeros((swath_rows[swath], swath_cols[swath]), dtype=np.complex128)
        for swath in swath_tuple
    }
    pri_pow_acc = {
        swath: np.zeros((swath_rows[swath], swath_cols[swath]), dtype=np.float64)
        for swath in swath_tuple
    }
    sec_pow_acc = {
        swath: np.zeros((swath_rows[swath], swath_cols[swath]), dtype=np.float64)
        for swath in swath_tuple
    }
    claimed = {
        swath: np.zeros((swath_rows[swath], swath_cols[swath]), dtype=np.int32)
        for swath in swath_tuple
    }
    looks_per_window = az_looks * rg_looks
    for unit in units:
        swath = unit["swath"]
        rows = unit["rows"]
        cols = unit["cols"]
        ifg_full = np.fromfile(unit["ifg_path"], dtype=np.complex64).reshape(rows, cols)
        pri_power = np.fromfile(unit["pri_path"], dtype=np.float64).reshape(rows, cols)
        sec_power = np.fromfile(unit["sec_path"], dtype=np.float64).reshape(rows, cols)
        valid = np.abs(ifg_full) > 0
        r0 = unit["azimuth_offset"] + unit["burst_row0"]
        c0 = range_offsets[swath] + unit["burst_col0"]
        orow = (r0 + np.arange(rows))[:, None] // az_looks
        ocol = (c0 + np.arange(cols))[None, :] // rg_looks
        ocol_local = ocol - range_offsets[swath] // rg_looks
        inb = (
            (orow < swath_rows[swath])
            & (ocol_local >= 0)
            & (ocol_local < swath_cols[swath])
            & valid
        )
        r_i, c_i = np.broadcast_arrays(orow, ocol_local)
        r_v, c_v = r_i[inb], c_i[inb]
        free = claimed[swath][r_v, c_v] < looks_per_window
        r_f, c_f = r_v[free], c_v[free]
        np.add.at(
            ifc_acc[swath],
            (r_f, c_f),
            ifg_full[inb][free].astype(np.complex128),
        )
        np.add.at(pri_pow_acc[swath], (r_f, c_f), pri_power[inb][free])
        np.add.at(sec_pow_acc[swath], (r_f, c_f), sec_power[inb][free])
        np.add.at(claimed[swath], (r_f, c_f), 1)
        del ifg_full, pri_power, sec_power

    merged_ifg = np.zeros((out_rows, out_cols), dtype=np.complex64)
    coherence = np.full((out_rows, out_cols), np.nan, dtype=np.float32)
    for swath in swath_tuple:
        has = claimed[swath] > 0
        swath_ifg = np.where(
            has, ifc_acc[swath] / np.where(has, claimed[swath], 1), 0
        ).astype(np.complex64)
        pri_ml = pri_pow_acc[swath] / np.where(has, claimed[swath], 1)
        sec_ml = sec_pow_acc[swath] / np.where(has, claimed[swath], 1)
        denom = np.sqrt(np.maximum(pri_ml * sec_ml, 1e-30))
        swath_coh = np.clip(np.abs(swath_ifg) / denom, 0.0, 1.0).astype(np.float32)
        col0 = range_offsets[swath] // rg_looks
        rows = swath_ifg.shape[0]
        cols = swath_ifg.shape[1]
        col1 = min(col0 + cols, out_cols)
        cols = col1 - col0
        band = np.abs(swath_ifg[:rows, :cols]) > 0
        merged_ifg[:rows, col0:col1][band] = swath_ifg[:rows, :cols][band]
        coherence[:rows, col0:col1][band] = swath_coh[:rows, :cols][band]
        del swath_ifg, swath_coh
    invalid = np.abs(merged_ifg) <= 0
    wrapped = np.where(invalid, np.nan, np.angle(merged_ifg).astype(np.float32))
    frame_indices = {unit["frame_index"] for unit in units}
    return {
        "merged_ifg": merged_ifg,
        "coherence": coherence,
        "wrapped": wrapped,
        "invalid": invalid,
        "out_rows": out_rows,
        "out_cols": out_cols,
        "origin_state": archive["origin_state"],
        "per_burst_timings": archive["per_burst_timings"],
        "units": units,
        "frame_count": max(frame_indices) + 1 if frame_indices else 0,
        "total_seconds": time.perf_counter() - archive["prefix_started"],
    }


def _finalize_sweep_config(
    merged: dict[str, Any],
    *,
    config: tuple[int, int],
    pair_id: str,
    output_root: Path,
    goldstein_alpha: float,
    snaphu_config: SnaphuConfig | None,
    unwrap_method: UnwrapBackend | None,
    irls_kwargs: dict[str, Any] | None,
    unwrap: bool,
    geo_height_m: float = 0.0,
    all_configs: list[tuple[int, int]] | None = None,
) -> tuple[PairSweepOutcome, ProductionPairState]:
    """Build the config product, write it, and record the completion manifest."""
    az_looks, rg_looks = config
    merged_ifg = merged["merged_ifg"]
    coherence = merged["coherence"]
    wrapped = merged["wrapped"]
    invalid = merged["invalid"]
    filtered = merged_ifg
    if goldstein_alpha > 0.0:
        from faninsar.processing.interferometry.pair import goldstein_filter

        filtered = goldstein_filter(merged_ifg, alpha=goldstein_alpha)

    origin = merged["origin_state"]
    if merged.get("grid_mode") == "geo":
        return _finalize_geo_config(
            merged,
            config=config,
            pair_id=pair_id,
            output_root=output_root,
            goldstein_alpha=goldstein_alpha,
            snaphu_config=snaphu_config,
            unwrap_method=unwrap_method,
            irls_kwargs=irls_kwargs,
            unwrap=unwrap,
            geo_height_m=geo_height_m,
            all_configs=all_configs,
        )
    result = ProductionPairState(
        pair_id=pair_id,
        reference=origin.reference,
        secondary=origin.secondary,
        dem=origin.dem,
        coregistration_grid="radar",
        dem_id=origin.dem_id,
        coreg_executor=origin.coreg_executor,
        coreg_device=origin.coreg_device,
        multilook=config,
        goldstein_alpha=float(goldstein_alpha),
        unwrap_method="snaphu",
        complex_ifg=merged_ifg,
        complex_ifg_flat=filtered,
        coherence=coherence,
        wrapped_phase=wrapped,
        stage_timings_s={
            "total": merged["total_seconds"],
            "per_burst": merged["per_burst_timings"],
        },
    )
    result.note(
        "PAIR frames="
        + str(merged["frame_count"])
        + " units="
        + str(len(merged["units"]))
        + " merged="
        + str(merged_ifg.shape)
        + " multilook="
        + str(config)
        + " valid="
        + format(float((~invalid).mean()), ".3f")
        + " mean_coh="
        + format(float(np.nanmean(coherence)), ".3f")
    )
    if unwrap:
        method: UnwrapBackend = unwrap_method if unwrap_method is not None else "snaphu"
        config_for_unwrap = snaphu_config
        if config_for_unwrap is None:
            config_for_unwrap = SnaphuConfig(nlooks=float(az_looks * rg_looks))
        else:
            config_for_unwrap = replace(
                config_for_unwrap, nlooks=float(az_looks * rg_looks)
            )
        result = stage_unwrap(
            result,
            method=method,
            config=config_for_unwrap,
            irls_kwargs=irls_kwargs,
        )
        result.note("UNWRAP complete on merged product")
    else:
        result.unwrapped_phase = np.zeros_like(wrapped, dtype=np.float32)
        result.connected_components = np.zeros_like(wrapped, dtype=np.uint8)
        result.note("UNWRAP skipped (unwrap=False)")

    config_dir = output_root / looks_dir(az_looks, rg_looks)
    stac_item_id = pair_id + "__l" + str(az_looks) + "x" + str(rg_looks)
    result = stage_write(
        result,
        config_dir,
        stac_item_id=stac_item_id,
    )
    assert result.zarr_path is not None
    assert result.stac_path is not None
    grid: dict[str, object] = {
        "crs": None,
        "transform": None,
        "shape": [merged_ifg.shape[0], merged_ifg.shape[1]],
        "looks": [az_looks, rg_looks],
    }
    _write_run_manifest(
        config_dir,
        looks=config,
        pair_id=pair_id,
        products=[result.zarr_path.name, result.stac_path.name],
        grid=grid,
    )
    metadata = {
        "multilook": [az_looks, rg_looks],
        "multilook_sweep": _sweep_list_metadata(all_configs, config),
        "wavelength_m": float(origin.reference.geometry.wavelength_m),
        "pair_id": pair_id,
        "product_grid": grid,
    }
    return PairSweepOutcome(
        config=config,
        zarr_path=result.zarr_path,
        stac_path=result.stac_path,
        shape=(merged_ifg.shape[0], merged_ifg.shape[1]),
        metadata=metadata,
        stage_timings_s=result.stage_timings_s,
        log=tuple(result.log),
    ), result


def _merge_geo_ifgs(
    archive: dict[str, Any],
    *,
    geo_grid: GeoGridSpec,
    az_looks: int,
    rg_looks: int,
) -> dict[str, Any]:
    """Accumulate archived geocoded IFGs onto the multilooked product grid."""
    units = archive["units"]
    out_rows = geo_grid.height // az_looks
    out_cols = geo_grid.width // rg_looks
    ifc_acc = np.zeros((out_rows, out_cols), dtype=np.complex128)
    pri_pow_acc = np.zeros((out_rows, out_cols), dtype=np.float64)
    sec_pow_acc = np.zeros((out_rows, out_cols), dtype=np.float64)
    height_acc = np.zeros((out_rows, out_cols), dtype=np.float64)
    claimed = np.zeros((out_rows, out_cols), dtype=np.int32)
    looks_per_window = az_looks * rg_looks
    for unit in units:
        rows = unit["rows"]
        cols = unit["cols"]
        ifg_full = np.fromfile(unit["ifg_path"], dtype=np.complex64).reshape(rows, cols)
        pri_power = np.fromfile(unit["pri_path"], dtype=np.float64).reshape(rows, cols)
        sec_power = np.fromfile(unit["sec_path"], dtype=np.float64).reshape(rows, cols)
        height = np.fromfile(unit["height_path"], dtype=np.float64).reshape(rows, cols)
        valid = np.abs(ifg_full) > 0
        r0 = unit["row0"]
        c0 = unit["col0"]
        orow = (r0 + np.arange(rows))[:, None] // az_looks
        ocol = (c0 + np.arange(cols))[None, :] // rg_looks
        inb = (orow < out_rows) & (ocol >= 0) & (ocol < out_cols) & valid
        r_i, c_i = np.broadcast_arrays(orow, ocol)
        r_v, c_v = r_i[inb], c_i[inb]
        free = claimed[r_v, c_v] < looks_per_window
        r_f, c_f = r_v[free], c_v[free]
        np.add.at(ifc_acc, (r_f, c_f), ifg_full[inb][free].astype(np.complex128))
        np.add.at(pri_pow_acc, (r_f, c_f), pri_power[inb][free])
        np.add.at(sec_pow_acc, (r_f, c_f), sec_power[inb][free])
        np.add.at(height_acc, (r_f, c_f), height[inb][free])
        np.add.at(claimed, (r_f, c_f), 1)
        del ifg_full, pri_power, sec_power, height
    has = claimed > 0
    merged_ifg = np.where(has, ifc_acc / np.where(has, claimed, 1), 0).astype(
        np.complex64
    )
    pri_ml = pri_pow_acc / np.where(has, claimed, 1)
    sec_ml = sec_pow_acc / np.where(has, claimed, 1)
    denom = np.sqrt(np.maximum(pri_ml * sec_ml, 1e-30))
    coherence = np.clip(np.abs(merged_ifg) / denom, 0.0, 1.0).astype(np.float32)
    coherence[~has] = np.nan
    height_field = np.where(has, height_acc / np.where(has, claimed, 1), np.nan).astype(
        np.float64
    )
    invalid = np.abs(merged_ifg) <= 0
    wrapped = np.where(invalid, np.nan, np.angle(merged_ifg).astype(np.float32))
    frame_indices = {unit["frame_index"] for unit in units}
    return {
        "merged_ifg": merged_ifg,
        "coherence": coherence,
        "wrapped": wrapped,
        "invalid": invalid,
        "out_rows": out_rows,
        "out_cols": out_cols,
        "origin_state": archive["origin_state"],
        "per_burst_timings": archive["per_burst_timings"],
        "units": units,
        "frame_count": max(frame_indices) + 1 if frame_indices else 0,
        "total_seconds": time.perf_counter() - archive["prefix_started"],
        "grid_mode": "geo",
        "geo_grid": geo_grid,
        "height_field": height_field,
    }


def _finalize_geo_config(
    merged: dict[str, Any],
    *,
    config: tuple[int, int],
    pair_id: str,
    output_root: Path,
    goldstein_alpha: float,
    snaphu_config: SnaphuConfig | None,
    unwrap_method: UnwrapBackend | None,
    irls_kwargs: dict[str, Any] | None,
    unwrap: bool,
    geo_height_m: float,
    all_configs: list[tuple[int, int]] | None = None,
) -> tuple[PairSweepOutcome, ProductionPairState]:
    """Build and write the geocoded product for one look configuration."""
    az_looks, rg_looks = config
    merged_ifg = merged["merged_ifg"]
    coherence = merged["coherence"]
    wrapped = merged["wrapped"]
    invalid = merged["invalid"]
    geo_grid: GeoGridSpec = merged["geo_grid"]
    filtered = merged_ifg
    if goldstein_alpha > 0.0:
        from faninsar.processing.interferometry.pair import goldstein_filter

        filtered = goldstein_filter(merged_ifg, alpha=goldstein_alpha)
    origin = merged["origin_state"]
    resolved_method: UnwrapBackend = (
        unwrap_method if unwrap_method is not None else "irls"
    )
    resolved_irls_kwargs = dict(irls_kwargs or {})
    if resolved_method == "irls":
        resolved_irls_kwargs.setdefault("device", origin.coreg_device)
    result = ProductionPairState(
        pair_id=pair_id,
        reference=origin.reference,
        secondary=origin.secondary,
        dem=origin.dem,
        coregistration_grid="geo",
        dem_id=origin.dem_id,
        coreg_executor=origin.coreg_executor,
        coreg_device=origin.coreg_device,
        multilook=config,
        goldstein_alpha=float(goldstein_alpha),
        unwrap_method=resolved_method,
        complex_ifg=merged_ifg,
        complex_ifg_flat=filtered,
        coherence=coherence,
        wrapped_phase=wrapped,
        geo_height_field=merged["height_field"],
        stage_timings_s={
            "total": merged["total_seconds"],
            "per_burst": merged["per_burst_timings"],
        },
    )
    result.note(
        "PAIR geo frames="
        + str(merged["frame_count"])
        + " units="
        + str(len(merged["units"]))
        + " merged="
        + str(merged_ifg.shape)
        + " multilook="
        + str(config)
        + " valid="
        + format(float((~invalid).mean()), ".3f")
        + " mean_coh="
        + format(float(np.nanmean(coherence)), ".3f")
    )
    if unwrap:
        config_for_unwrap = snaphu_config
        if config_for_unwrap is None:
            config_for_unwrap = SnaphuConfig(nlooks=float(az_looks * rg_looks))
        else:
            config_for_unwrap = replace(
                config_for_unwrap, nlooks=float(az_looks * rg_looks)
            )
        result = stage_unwrap(
            result,
            method=resolved_method,
            config=config_for_unwrap,
            irls_kwargs=resolved_irls_kwargs,
        )
        result.note("UNWRAP complete on merged geo product")
    else:
        result.unwrapped_phase = np.zeros_like(wrapped, dtype=np.float32)
        result.connected_components = np.zeros_like(wrapped, dtype=np.uint8)
        result.note("UNWRAP skipped (unwrap=False)")
    result = _finalize_geo_products(
        result,
        geo_grid=geo_grid,
        multilook=config,
        geo_height_m=geo_height_m,
    )
    config_dir = output_root / looks_dir(az_looks, rg_looks)
    stac_item_id = pair_id + "__l" + str(az_looks) + "x" + str(rg_looks)
    result = stage_write(
        result,
        config_dir,
        stac_item_id=stac_item_id,
    )
    assert result.zarr_path is not None
    assert result.stac_path is not None
    grid_meta = result.geo_grid_meta
    grid: dict[str, object] = {
        "crs": None if grid_meta is None else grid_meta["crs"],
        "transform": None if grid_meta is None else grid_meta["transform"],
        "shape": [merged_ifg.shape[0], merged_ifg.shape[1]],
        "looks": [az_looks, rg_looks],
    }
    _write_run_manifest(
        config_dir,
        looks=config,
        pair_id=pair_id,
        products=[result.zarr_path.name, result.stac_path.name],
        grid=grid,
    )
    metadata = {
        "multilook": [az_looks, rg_looks],
        "multilook_sweep": _sweep_list_metadata(all_configs, config),
        "wavelength_m": float(origin.reference.geometry.wavelength_m),
        "pair_id": pair_id,
        "product_grid": grid,
    }
    return PairSweepOutcome(
        config=config,
        zarr_path=result.zarr_path,
        stac_path=result.stac_path,
        shape=(merged_ifg.shape[0], merged_ifg.shape[1]),
        metadata=metadata,
        stage_timings_s=result.stage_timings_s,
        log=tuple(result.log),
    ), result
