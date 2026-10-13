"""Test-only one-shot Sentinel-1 pair runner.

Production code uses stage functions, coregister_s1_scene, and
Stack.form_interferograms. This module composes those stages for
legacy pair-engine tests only.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from faninsar.data.query import BoundingBox, Polygons
from faninsar.logging import setup_logger
from faninsar.missions.s1 import open_safe_product, read_eof_orbit
from faninsar.missions.s1.processing import (
    BurstSelection,
    PreparedGeometryField,
    ProductionPairState,
    ProductionScene,
    _as_frame_paths,
    _as_optional_frame_sequence,
    _auto_dem_bounds,
    _clock_start,
    _clock_stop,
    _common_burst_indices,
    _dem_id,
    _finalize_geo_products,
    _merge_coreg_stage_times,
    _normalize_burst_selection,
    _own_geo_valid_mask,
    _phase_state_manifest,
    _radar_model,
    _roi_burst_window,
    _scene_id,
    _select_bursts_by_roi,
    _snapshot_optional_paths,
    _snapshot_paths,
    _swath_range_offsets,
    _worker_prepared_provider,
    read_prepared_lut,
    resolve_auto_dem,
    stage_coregister,
    stage_deramp,
    stage_flatten,
    stage_interferogram,
    stage_unwrap,
    stage_write,
)
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.coregistration import resolve_ampcor_policy
from faninsar.processing.geometry import ConstantDEM, DEM
from faninsar.processing.geometry import RasterDEM as PublicRasterDEM
from faninsar.processing.interferometry.pair import goldstein_filter, mask_invalid_looks
from faninsar.processing.runtime.memory import close_memmap, release_memmap_pages
from faninsar.processing.unwrapping import SnaphuConfig, UnwrapBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.missions.s1.types import S1Product
    from faninsar.processing.geometry import DEM
    from faninsar.processing.geometry.prepared import (
        PreparedLutHandle,
        ProviderLeaseToken,
        ResourceLimits,
    )
    from faninsar.processing.mosaicking.grid import GeoGridSpec

logger = setup_logger(__name__)

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
                "primary_geocoded_slc",
                "secondary_geocoded_slc",
                "geocoded_slc_valid",
                "topo_phase",
                "geo_height_field",
                "primary_deramped",
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


_ARCHIVE_WRITE_CHUNK_BYTES = 256 * 1024 * 1024
_ARCHIVE_WRITE_RETRIES = 120
_ARCHIVE_RETRY_SECONDS = 60.0


def _archive_write_chunks(payload: np.ndarray, path: str, tag: str) -> None:
    """Write one contiguous payload, restarting after a transient ENOSPC.

    A write that fails mid-transfer leaves an indeterminate number of bytes
    in the file, so each ENOSPC attempt reopens the path with ``O_TRUNC``
    and restarts the payload from byte zero.
    """
    stream = payload.view(np.uint8).reshape(-1)
    for attempt in range(_ARCHIVE_WRITE_RETRIES):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            offset = 0
            while offset < stream.size:
                written = os.write(
                    fd, stream[offset : offset + _ARCHIVE_WRITE_CHUNK_BYTES].tobytes()
                )
                offset += written
            return
        except OSError as error:
            if error.errno != errno.ENOSPC or attempt + 1 == _ARCHIVE_WRITE_RETRIES:
                raise
            logger.warning(
                "%s: archive write ENOSPC at offset %d (attempt %d/%d); "
                "syncing and restarting from byte 0",
                tag,
                offset,
                attempt + 1,
                _ARCHIVE_WRITE_RETRIES,
            )
            os.sync()
            time.sleep(_ARCHIVE_RETRY_SECONDS)
        finally:
            os.close(fd)


def _archive_tofile(array: np.ndarray, path: str, tag: str) -> None:
    """Write one burst-archive array, recovering from a short ``tofile`` write.

    Large archive writes have been observed to fail with numpy's generic
    short-write ``OSError`` while the filesystem still reports free space
    (a transient short ``fwrite`` of the whole payload).  Retry through
    explicit chunked ``os.write`` calls with an ENOSPC backoff; a still
    stuck file falls back to a sibling payload followed by an atomic
    rename so one wedged inode cannot fail the run.
    """
    try:
        array.tofile(path)
    except OSError as error:
        logger.warning(
            "%s: archive tofile short write (errno=%s); retrying with chunked os.write",
            tag,
            error.errno,
        )
    else:
        return
    payload = np.ascontiguousarray(array)
    try:
        _archive_write_chunks(payload, path, tag)
    except OSError as error:
        if error.errno != errno.ENOSPC:
            raise
        logger.warning(
            "%s: archive write persistently ENOSPC; retrying via sibling file",
            tag,
        )
        sibling = path + ".archive_retry"
        _archive_write_chunks(payload, sibling, tag)
        Path(sibling).replace(path)


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
    if (
        bool(task.get("require_worker_bootstrap"))
        and os.environ.get("FANINSAR_WORKER_BOOTSTRAPPED") != "1"
    ):
        reject_invalid_state(
            "resource-gated burst worker was started without numerical "
            "runtime bootstrap"
        )

    from dataclasses import replace as _replace

    from faninsar.missions.s1 import read_eof_orbit, read_full_burst
    from faninsar.missions.s1.safe import open_safe_product
    from faninsar.processing.coregistration.tops.carrier import carrier_from_swath

    tag = str(task["tag"])
    swath = str(task["swath"])
    frame_index = int(task["frame_index"])
    burst_index = int(task["burst_index"])
    azimuth_offset = int(task["azimuth_offset"])
    range_offset = int(task.get("range_offset", 0))
    ref_path = Path(task["ref_path"])
    sec_path = Path(task["sec_path"])
    ref_orbit = task.get("ref_orbit")
    sec_orbit = task.get("sec_orbit")
    roi = task.get("roi")
    control_spacing = task.get("control_spacing")
    esd_enabled = bool(task["esd_enabled"])
    amplitude_refinement_enabled = bool(task["amplitude_refinement_enabled"])
    misreg_az_px = float(task.get("misreg_az_px", 0.0))
    misreg_rg_px = float(task.get("misreg_rg_px", 0.0))
    force_amp_rg = task.get("force_amplitude_residual_rg")
    force_esd_az = task.get("force_esd_azimuth_shift_px")
    force_amplitude_residual_rg = (
        float(force_amp_rg) if force_amp_rg is not None else None
    )
    force_esd_azimuth_shift_px = (
        float(force_esd_az) if force_esd_az is not None else None
    )
    executor = str(task["executor"])
    device = str(task["device"])
    dask_client = task.get("dask_client")
    dead_pixel_amp_threshold = float(task["dead_pixel_amp_threshold"])
    coregistration_grid = task["coregistration_grid"]
    geo_grid = task["geo_grid"]
    geo_height_m = float(task["geo_height_m"])
    geo_chunk_size = int(task["geo_chunk_size"])
    roi_buffer_m = float(task.get("roi_buffer_m", 320.0))
    record_scientific_lineage = bool(task.get("record_scientific_lineage", False))
    ifg_dir = Path(task["ifg_dir"])
    dem = task["dem"]
    geo_work_dir = task["geo_work_dir"]
    geo_lut_cache_dir = task.get("geo_lut_cache_dir")
    geo_footprint_mask_enabled = bool(task.get("geo_footprint_mask_enabled", True))
    scene_store_dir = task.get("scene_store_dir")
    scene_grid_shape = task.get("scene_grid_shape")
    prepared_lut_handle = task.get("prepared_geo_lut_handle")
    prepared_provider_token = task.get("prepared_provider_token")
    prepared_provider_root = task.get("prepared_provider_root")
    prepared_resource_limits = task.get("resource_limits")
    prepared_values = (
        prepared_lut_handle,
        prepared_provider_token,
        prepared_provider_root,
    )
    if any(value is not None for value in prepared_values) and not all(
        value is not None for value in prepared_values
    ):
        reject_invalid_state(
            "prepared Geo LUT worker inputs must include handle, lease token, "
            "and provider root"
        )
    prepared_geo_lut: Geo2RdrLUT | None = None
    if prepared_lut_handle is not None:
        from faninsar.processing.geometry.prepared import (
            ProviderLeaseToken as _ProviderLeaseToken,
        )

        if coregistration_grid != "geo":
            reject_invalid_state("prepared Geo LUT reuse requires geo coregistration")
        if not isinstance(prepared_provider_token, _ProviderLeaseToken):
            reject_invalid_state("prepared Geo LUT worker lease token is invalid")
        provider = _worker_prepared_provider(
            prepared_provider_root,
            prepared_provider_token,
            prepared_resource_limits,
        )
        prepared_geo_lut = read_prepared_lut(
            provider,
            prepared_lut_handle,
            prepared_provider_token,
        )

    load_started = time.perf_counter()
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
    load_s = time.perf_counter() - load_started
    state = ProductionPairState(
        pair_id=ref.scene_id + "_" + sec.scene_id + "_" + tag,
        primary=ref,
        secondary=sec,
        dem=dem,
        coregistration_grid=coregistration_grid,
        multilook=(1, 1),
        goldstein_alpha=0.0,
        unwrap_method="snaphu",
        record_scientific_lineage=record_scientific_lineage,
    )
    stage_times: dict[str, float] = {"load": load_s}
    t0 = _clock_start(device)
    state = stage_deramp(state, device=device, dask_client=dask_client)
    stage_times["deramp"] = _clock_stop(device, t0)
    t0 = _clock_start(device)
    burst_work_dir: Path | None = None
    if coregistration_grid == "geo" and geo_work_dir is not None:
        burst_work_dir = Path(geo_work_dir) / tag
    roi_window: tuple[int, int, int, int] | None = None
    if coregistration_grid == "radar" and roi is not None:
        assert state.primary_deramped is not None
        roi_window = _roi_burst_window(
            roi,
            ref.geometry,
            dem,
            state.primary_deramped.shape,
            device=device,
            buffer_m=roi_buffer_m,
        )
    state = stage_coregister(
        state,
        control_spacing=control_spacing,
        esd_enabled=esd_enabled,
        amplitude_refinement_enabled=amplitude_refinement_enabled,
        misreg_az_px=misreg_az_px,
        misreg_rg_px=misreg_rg_px,
        force_amplitude_residual_rg=force_amplitude_residual_rg,
        force_esd_azimuth_shift_px=force_esd_azimuth_shift_px,
        executor=executor,
        device=device,
        dask_client=dask_client,
        coregistration_grid=coregistration_grid,
        geo_grid=geo_grid,
        geo_height_m=geo_height_m,
        geo_chunk_size=geo_chunk_size,
        geo_work_dir=burst_work_dir,
        geo_lut_cache_dir=geo_lut_cache_dir,
        geo_footprint_mask_enabled=geo_footprint_mask_enabled,
        roi=roi,
        roi_buffer_m=roi_buffer_m,
        roi_window=roi_window,
        prepared_geo_lut=prepared_geo_lut,
    )
    stage_times["coregister"] = _clock_stop(device, t0)
    _merge_coreg_stage_times(stage_times, state.coregistration_timings_s)
    burst_row0 = 0
    burst_col0 = 0
    geo_valid_mask: np.ndarray | None = None
    if coregistration_grid == "geo":
        # ``stage_interferogram`` releases the disk-backed geographic
        # validity map after forming the IFG.  Keep an owning copy here;
        # retaining a memmap view would leave this worker writing through an
        # unmapped address during the archive crop.
        geo_valid_mask = _own_geo_valid_mask(state.geocoded_slc_valid)
    elif state.radar_roi_origin is not None:
        burst_row0, burst_col0 = state.radar_roi_origin
    assert state.primary_deramped is not None
    assert state.secondary_aligned is not None
    if coregistration_grid == "geo" and (
        np.asarray(state.primary_deramped).size == 0
        or np.asarray(state.secondary_aligned).size == 0
    ):
        logger.warning("%s: empty geocoded footprint; skipped", tag)
        return {"unit": None, "stage_times": stage_times}
    if scene_store_dir is not None:
        from faninsar.io.storage.scene_store import (
            scene_grid_identity,
            write_scene_unit,
        )

        if coregistration_grid == "geo":
            row_origin = state.geo_bbox[0] if state.geo_bbox is not None else 0
            col_origin = state.geo_bbox[2] if state.geo_bbox is not None else 0
        else:
            row_origin = azimuth_offset + burst_row0
            col_origin = range_offset + burst_col0
        resolved_scene_grid_shape = (
            (int(scene_grid_shape[0]), int(scene_grid_shape[1]))
            if scene_grid_shape is not None
            else None
        )
        grid_identity = None
        if coregistration_grid == "geo" and geo_grid is not None:
            grid_identity = scene_grid_identity(
                "geo",
                geo_grid.shape,
                {
                    "crs": geo_grid.crs,
                    "transform": list(geo_grid.transform),
                    "resolution_m": list(geo_grid.resolution_m),
                },
            )
        write_scene_unit(
            scene_store_dir,
            date_id=_scene_id(sec_path),
            reference_id=_scene_id(ref_path),
            domain=str(coregistration_grid),
            tag=tag,
            primary=np.asarray(state.primary_deramped, dtype=np.complex64),
            secondary=np.asarray(state.secondary_aligned, dtype=np.complex64),
            row_origin=row_origin,
            col_origin=col_origin,
            grid_shape=resolved_scene_grid_shape,
            wavelength_m=float(state.primary.geometry.wavelength_m),
            grid_identity=grid_identity,
            crs=(
                geo_grid.crs
                if coregistration_grid == "geo" and geo_grid is not None
                else None
            ),
            transform=(
                tuple(float(value) for value in tuple(geo_grid.transform)[:6])
                if coregistration_grid == "geo" and geo_grid is not None
                else None
            ),
            scientific_lineage=state.scientific_lineage,
            phase_state=_phase_state_manifest(state),
        )
    pri_power = state.primary_deramped.real**2 + state.primary_deramped.imag**2
    sec_power = state.secondary_aligned.real**2 + state.secondary_aligned.imag**2
    t0 = _clock_start(device)
    state = stage_interferogram(
        state,
        multilook=(1, 1),
        goldstein_alpha=0.0,
        dead_pixel_amp_threshold=dead_pixel_amp_threshold,
        device=device,
        dask_client=dask_client,
    )
    stage_times["interferogram"] = _clock_stop(device, t0)
    t0 = _clock_start(device)
    if coregistration_grid == "geo":
        state.complex_ifg_flat = state.complex_ifg
        state.note("FLATTEN applied to secondary geocoded SLC before IFG formation")
        stage_times["flatten"] = 0.0
    else:
        state = stage_flatten(
            state,
            device=device,
            dask_client=dask_client,
        )
        stage_times["flatten"] = _clock_stop(device, t0)
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
            height_full = np.full(ifg_full.shape, float(geo_height_m), dtype=np.float64)
        else:
            height_full = np.asarray(height_full, dtype=np.float64)
        lr0, lr1, lc0, lc1 = bbox_local
        bbox_ifg = ifg_full[lr0:lr1, lc0:lc1]
        bbox_pri = pri_power[lr0:lr1, lc0:lc1]
        bbox_sec = sec_power[lr0:lr1, lc0:lc1]
        bbox_height = height_full[lr0:lr1, lc0:lc1]
        _archive_tofile(bbox_ifg.astype(np.complex64, copy=False), ifg_path, tag)
        _archive_tofile(bbox_pri.astype(np.float64, copy=False), pri_path, tag)
        _archive_tofile(bbox_sec.astype(np.float64, copy=False), sec_path_out, tag)
        _archive_tofile(bbox_height.astype(np.float64, copy=False), height_path, tag)
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
        _archive_tofile(ifg_full.astype(np.complex64, copy=False), ifg_path, tag)
        _archive_tofile(pri_power.astype(np.float64, copy=False), pri_path, tag)
        _archive_tofile(sec_power.astype(np.float64, copy=False), sec_path_out, tag)
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
    # Carry the observation across the worker/archive boundary.  Geo sweep
    # results are rebuilt from flat IFG units, so dropping these fields here
    # would make network arcs appear to have zero Ampcor/ESD correction.
    for name in (
        "range_shift_px",
        "azimuth_shift_px",
        "esd_azimuth_shift_px",
        "amplitude_residual_rg_px",
    ):
        unit[name] = getattr(state, name)
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


@overload
def _produce_interferogram_pair(
    primary_path: str | Path | Sequence[str | Path],
    secondary_path: str | Path | Sequence[str | Path],
    *,
    output_dir: str | Path,
    roi: BoundingBox | Polygons | None = None,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    dem: DEM | None = None,
    multilook: tuple[int, int] | list[int] = (2, 10),
    overwrite: bool = False,
    goldstein_alpha: float = 0.5,
    dead_pixel_amp_threshold: float = 3.0,
    esd_enabled: bool = False,
    amplitude_refinement_enabled: bool = False,
    control_spacing: int | None = None,
    executor: str = "torch",
    device: str = "auto",
    dask_client: Any | None = None,
    coregistration_grid: CoregistrationGrid = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_height_m: float = 0.0,
    geo_chunk_size: int = 128,
    geo_work_dir: str | Path | None = None,
    geo_lut_cache_dir: str | Path | None = None,
    geo_footprint_mask_enabled: bool = True,
    scene_store_dir: str | Path | None = None,
    n_jobs: int = 1,
    resource_limits: ResourceLimits | None = None,
    prepared_geo_lut_handles: Mapping[str, tuple[PreparedLutHandle, ProviderLeaseToken]]
    | None = None,
    prepared_provider_root: str | Path | None = None,
    source_snapshot_root: str | Path | None = None,
    roi_buffer_m: float = 320.0,
    snaphu_config: SnaphuConfig | None = None,
    unwrap_method: UnwrapBackend | None = None,
    irls_kwargs: dict[str, Any] | None = None,
    primary_orbit_path: str | Path | Sequence[str | Path] | None = None,
    secondary_orbit_path: str | Path | Sequence[str | Path] | None = None,
    unwrap: bool = False,
    geoid_correction: bool = True,
    dem_source: str | None = None,
    record_scientific_lineage: bool = False,
) -> ProductionPairState: ...


@overload
def _produce_interferogram_pair(
    primary_path: str | Path | Sequence[str | Path],
    secondary_path: str | Path | Sequence[str | Path],
    *,
    output_dir: str | Path,
    roi: BoundingBox | Polygons | None = None,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    dem: DEM | None = None,
    multilook: Iterable[tuple[int, int]],
    overwrite: bool = False,
    goldstein_alpha: float = 0.5,
    dead_pixel_amp_threshold: float = 3.0,
    esd_enabled: bool = False,
    amplitude_refinement_enabled: bool = False,
    control_spacing: int | None = None,
    executor: str = "torch",
    device: str = "auto",
    dask_client: Any | None = None,
    coregistration_grid: CoregistrationGrid = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_height_m: float = 0.0,
    geo_chunk_size: int = 128,
    geo_work_dir: str | Path | None = None,
    geo_lut_cache_dir: str | Path | None = None,
    geo_footprint_mask_enabled: bool = True,
    scene_store_dir: str | Path | None = None,
    n_jobs: int = 1,
    resource_limits: ResourceLimits | None = None,
    prepared_geo_lut_handles: Mapping[str, tuple[PreparedLutHandle, ProviderLeaseToken]]
    | None = None,
    prepared_provider_root: str | Path | None = None,
    source_snapshot_root: str | Path | None = None,
    roi_buffer_m: float = 320.0,
    snaphu_config: SnaphuConfig | None = None,
    unwrap_method: UnwrapBackend | None = None,
    irls_kwargs: dict[str, Any] | None = None,
    primary_orbit_path: str | Path | Sequence[str | Path] | None = None,
    secondary_orbit_path: str | Path | Sequence[str | Path] | None = None,
    unwrap: bool = False,
    geoid_correction: bool = True,
    dem_source: str | None = None,
    record_scientific_lineage: bool = False,
) -> ProductionPairSweepResult: ...


def _produce_interferogram_pair(
    primary_path: str | Path | Sequence[str | Path],
    secondary_path: str | Path | Sequence[str | Path],
    *,
    output_dir: str | Path,
    roi: BoundingBox | Polygons | None = None,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    dem: DEM | None = None,
    multilook: tuple[int, int] | list[int] | Iterable[tuple[int, int]] = (2, 10),
    overwrite: bool = False,
    goldstein_alpha: float = 0.5,
    dead_pixel_amp_threshold: float = 3.0,
    esd_enabled: bool = False,
    amplitude_refinement_enabled: bool = False,
    misreg_az_px: float = 0.0,
    misreg_rg_px: float = 0.0,
    control_spacing: int | None = None,
    executor: str = "torch",
    device: str = "auto",
    dask_client: Any | None = None,
    coregistration_grid: CoregistrationGrid = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_height_m: float = 0.0,
    geo_chunk_size: int = 128,
    geo_work_dir: str | Path | None = None,
    geo_lut_cache_dir: str | Path | None = None,
    geo_footprint_mask_enabled: bool = True,
    scene_store_dir: str | Path | None = None,
    n_jobs: int = 1,
    resource_limits: ResourceLimits | None = None,
    prepared_geo_lut_handles: Mapping[str, tuple[PreparedLutHandle, ProviderLeaseToken]]
    | None = None,
    prepared_provider_root: str | Path | None = None,
    source_snapshot_root: str | Path | None = None,
    roi_buffer_m: float = 320.0,
    snaphu_config: SnaphuConfig | None = None,
    unwrap_method: UnwrapBackend | None = None,
    irls_kwargs: dict[str, Any] | None = None,
    primary_orbit_path: str | Path | Sequence[str | Path] | None = None,
    secondary_orbit_path: str | Path | Sequence[str | Path] | None = None,
    unwrap: bool = False,
    geoid_correction: bool = True,
    dem_source: str | None = None,
    record_scientific_lineage: bool = False,
) -> ProductionPairState | ProductionPairSweepResult:
    """Process any burst selection across frames and swaths into one product.

    Parameters
    ----------
    primary_path, secondary_path : str, Path, or sequence
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
    dem : DEM, optional
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
    misreg_az_px, misreg_rg_px : float, optional
        Network (or external) rigid misregistration constants added to the
        dense offset field (PROPOSAL-0017). Default 0.
    control_spacing : int, optional
        Geometry control-point spacing.
    executor : {"torch"}, optional
        Resample executor.
    device : {"auto","cpu","cuda","mps"}, optional
        Torch device.
    dask_client : object or None, optional
        Explicitly trusted Dask client. Distributed work is submitted only
        through this client; ambient clients are ignored.
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
        Directory for geo-mode burst work products (LUT, geocoded SLC memmaps).
    geo_lut_cache_dir : path, optional
        Shared cache directory for reference-scene geo2rdr LUTs. Pairs that
        share the same reference date and grid reuse the cached solve instead
        of recomputing it.
        Working directory for geo memmaps; a temporary directory is used
        when omitted.
    geo_footprint_mask_enabled : bool, optional
        When True (default), apply the burst-footprint hull mask in geo mode
        when no ROI is supplied.  ROI burst-quad intersection is unchanged.
    scene_store_dir : path, optional
        Caller-owned directory for immutable Reference-aligned scene units. Radar
        and Geo sweep paths support this seam when ``n_jobs=1``; parallel
        publication is rejected until generation-level locking is enabled.
    n_jobs : int, optional
        Number of parallel burst workers (default 1; applies to radar mode
        with an ROI and to geo mode).
    resource_limits : ResourceLimits, optional
        Authenticated finite resource profile.  When supplied, sweep/geo
        burst execution reserves worker slots and output files before
        submission and monitors the complete parent-plus-descendant RSS tree
        at 100 ms.  A missing profile keeps compatibility but is not a
        qualified resource-gate run.
    prepared_geo_lut_handles : mapping, optional
        Per-unit prepared LUT handles and lease tokens keyed by burst task
        tag.  These are consumed by geographic workers without rebuilding
        geo2rdr LUTs.
    prepared_provider_root : path, optional
        Provider-owned prepared-generation store root used by workers to
        attach the lease token and read ``prepared_geo_lut_handles``.
    source_snapshot_root : path, optional
        Caller-owned private root for immutable local SAFE, orbit, and DEM
        snapshots.  When supplied, all local source paths are copied and the
        copied paths—not the original aliases—are used for parsing and reads.
    roi_buffer_m : float, optional
        Physical margin in meters kept around the ROI (default 320).  In geo
        mode it is applied in the grid CRS; in radar mode it is converted to
        radar pixels to expand the burst window.
    snaphu_config : SnaphuConfig, optional
        SNAPHU configuration; nlooks defaults to az * rg per config.
    unwrap_method : {"irls", "snaphu"}, optional
        Unwrapping backend when unwrap=True.
    irls_kwargs : dict, optional
        Arguments forwarded to the IRLS unwrap backend.
    primary_orbit_path, secondary_orbit_path : path or sequence, optional
        Precise ESA EOF orbits; one per frame or a single orbit reused for
        every frame.
    unwrap : bool, optional
        Run unwrapping on the merged wrapped phase when True.
    geoid_correction : bool, optional
        Convert orthometric raster DEM heights to ellipsoidal with EGM2008.
        Default True.
    dem_source : str, optional
        DEM selection for the automatic bare-name build (``<product>`` or
        ``<product>:<provider>``); ``None`` defers to ``FANINSAR_DEM_SOURCE``
        and then the ``glo30`` default.
    record_scientific_lineage : bool, optional
        Persist ordered residual/carrier/phase operation records with payload
        hashes. Disabled by default to keep the ordinary production path
        free of hashing overhead.

    Returns
    -------
    ProductionPairState or ProductionPairSweepResult
        Single-config state, or per-config outcomes for a sweep.

    """
    if (
        prepared_geo_lut_handles is not None or prepared_provider_root is not None
    ) and coregistration_grid != "geo":
        reject_invalid_state("prepared Geo LUT reuse requires geo coregistration")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        reject_invalid_state("n_jobs must be a positive integer")
    _, resolved_ampcor_device = resolve_ampcor_policy(executor, device)
    if n_jobs > 1 and resolved_ampcor_device.startswith("cuda"):
        reject_invalid_state(
            "explicit CUDA Ampcor is incompatible with ProcessPoolExecutor "
            "n_jobs > 1; use n_jobs=1 for device-local admission"
        )
    if not _is_multilook_pair(multilook) or coregistration_grid == "geo":
        return _produce_interferogram_sweep(
            primary_path,
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
            misreg_az_px=misreg_az_px,
            misreg_rg_px=misreg_rg_px,
            control_spacing=control_spacing,
            executor=executor,
            device=device,
            dask_client=dask_client,
            coregistration_grid=coregistration_grid,
            geo_grid=geo_grid,
            geo_height_m=geo_height_m,
            geo_chunk_size=geo_chunk_size,
            geo_work_dir=geo_work_dir,
            geo_lut_cache_dir=geo_lut_cache_dir,
            geo_footprint_mask_enabled=geo_footprint_mask_enabled,
            scene_store_dir=scene_store_dir,
            n_jobs=n_jobs,
            resource_limits=resource_limits,
            prepared_geo_lut_handles=prepared_geo_lut_handles,
            prepared_provider_root=prepared_provider_root,
            source_snapshot_root=source_snapshot_root,
            roi_buffer_m=roi_buffer_m,
            snaphu_config=snaphu_config,
            unwrap_method=unwrap_method,
            irls_kwargs=irls_kwargs,
            primary_orbit_path=primary_orbit_path,
            secondary_orbit_path=secondary_orbit_path,
            unwrap=unwrap,
            geoid_correction=geoid_correction,
            dem_source=dem_source,
            record_scientific_lineage=record_scientific_lineage,
        )
    from faninsar.missions.s1.safe import open_safe_product

    dem_sampler: DEM = dem if dem is not None else ConstantDEM(0.0)
    snapshot_root = Path(source_snapshot_root) if source_snapshot_root else None
    if snapshot_root is not None and isinstance(dem_sampler, PublicRasterDEM):
        from faninsar.processing.runtime.source_snapshots import snapshot_local_source

        if dem_sampler.path is not None:
            dem_snapshot = snapshot_local_source(
                dem_sampler.path,
                snapshot_root / "dem",
            )
            dem_sampler = DEM.from_raster(
                dem_snapshot.path,
                vertical_datum=dem_sampler.vertical_datum,
            )
    if geoid_correction and isinstance(dem_sampler, PublicRasterDEM):
        dem_sampler = dem_sampler.to_raster(
            dem_sampler.grid,
            vertical_datum="ellipsoidal",
        )

    primary_paths = _as_frame_paths(primary_path, "primary_path")
    sec_paths = _as_frame_paths(secondary_path, "secondary_path")
    if len(primary_paths) != len(sec_paths):
        reject_invalid_state(
            "reference/secondary frame counts differ: "
            + str(len(primary_paths))
            + " vs "
            + str(len(sec_paths))
        )
    frame_count = len(primary_paths)
    ref_orbits = _as_optional_frame_sequence(
        primary_orbit_path, frame_count, "primary_orbit_path"
    )
    sec_orbits = _as_optional_frame_sequence(
        secondary_orbit_path, frame_count, "secondary_orbit_path"
    )
    if snapshot_root is not None:
        primary_paths = _snapshot_paths(primary_paths, snapshot_root / "safe")
        sec_paths = _snapshot_paths(sec_paths, snapshot_root / "safe")
        ref_orbits = _snapshot_optional_paths(
            ref_orbits,
            snapshot_root / "orbit",
        )
        sec_orbits = _snapshot_optional_paths(
            sec_orbits,
            snapshot_root / "orbit",
        )

    primary_products = [open_safe_product(path) for path in primary_paths]
    secondary_products = [open_safe_product(path) for path in sec_paths]
    if swaths is None:
        ordered = sorted(
            primary_products[0].swaths,
            key=lambda item: item.slant_range_time_s,
        )
        swath_tuple = tuple(item.swath for item in ordered)
    else:
        swath_tuple = tuple(swaths)
    if not swath_tuple:
        reject_invalid_state("swaths must not be empty")
    for frame_index, product in enumerate(primary_products):
        present = {item.swath for item in product.swaths}
        missing = [name for name in swath_tuple if name not in present]
        if missing:
            reject_invalid_state(
                "reference frame "
                + str(frame_index)
                + " ("
                + primary_paths[frame_index].name
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
        logger.info(
            "Stack provider: ROI provided; explicit swaths/bursts selection ignored"
        )
        ordered = sorted(
            primary_products[0].swaths,
            key=lambda item: item.slant_range_time_s,
        )
        swath_tuple = tuple(item.swath for item in ordered)
        resolved = _select_bursts_by_roi(
            roi,
            primary_paths,
            swath_tuple,
            orbits=ref_orbits,
            dem=dem_sampler,
        )
    else:
        burst_counts = {
            (frame_index, swath): len(primary_products[frame_index].swath(swath).bursts)
            for frame_index in range(frame_count)
            for swath in swath_tuple
        }
        resolved = _normalize_burst_selection(
            bursts, frame_count, swath_tuple, burst_counts
        )

    common_aligned: dict[tuple[int, str], list[int]] = {}
    for frame_index in range(frame_count):
        for swath in swath_tuple:
            primary_swath = primary_products[frame_index].swath(swath)
            secondary_swath = secondary_products[frame_index].swath(swath)
            common = _common_burst_indices(primary_swath, secondary_swath)
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
        bounds = _auto_dem_bounds(
            roi,
            resolved,
            primary_products,
            orbits=ref_orbits,
        )
        dem_sampler = resolve_auto_dem(
            bounds,
            output_dir=Path(output_dir),
            geoid_correction=geoid_correction,
            dem_source=dem_source,
        )

    range_offsets = _swath_range_offsets(swath_tuple, primary_products)
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
    # ISCE2 applies one secondary range correction and one ESD azimuth
    # correction to the whole pair. Per-burst Ampcor/ESD residuals create
    # constant phase jumps at burst seams (visible N-S boundaries).
    force_amp_rg: float | None = None
    force_esd_az: float | None = None
    prepared_geometry_by_tag: dict[str, PreparedGeometryField] = {}
    if selected_unit_count > 1 and (esd_enabled or amplitude_refinement_enabled):
        measured_amp: list[float] = []
        measured_esd: list[float] = []
        for swath in reversed(swath_tuple):
            for frame_index, burst_index, _azimuth_offset in units_by_swath[swath]:
                tag = "f" + str(frame_index) + "_" + swath + "_b" + str(burst_index)
                ref = load_burst(
                    swath,
                    burst_index,
                    primary_paths[frame_index],
                    ref_orbits[frame_index],
                    primary_products[frame_index],
                )
                sec = load_burst(
                    swath,
                    burst_index,
                    sec_paths[frame_index],
                    sec_orbits[frame_index],
                    secondary_products[frame_index],
                )
                measure_state = ProductionPairState(
                    pair_id=ref.scene_id + "_" + sec.scene_id + "_" + tag,
                    primary=ref,
                    secondary=sec,
                    dem=dem_sampler,
                    coregistration_grid="radar",
                    multilook=multilook,
                    goldstein_alpha=0.0,
                    unwrap_method="snaphu",
                    record_scientific_lineage=record_scientific_lineage,
                )
                measure_started = _clock_start(device)
                measure_state = stage_deramp(
                    measure_state,
                    device=device,
                    dask_client=dask_client,
                )
                measure_deramp_s = _clock_stop(device, measure_started)
                roi_window_m: tuple[int, int, int, int] | None = None
                if roi is not None:
                    assert measure_state.primary_deramped is not None
                    roi_window_m = _roi_burst_window(
                        roi,
                        ref.geometry,
                        dem_sampler,
                        measure_state.primary_deramped.shape,
                        device=device,
                        buffer_m=roi_buffer_m,
                    )
                measure_coreg_started = _clock_start(device)
                measure_state = stage_coregister(
                    measure_state,
                    control_spacing=control_spacing,
                    esd_enabled=esd_enabled,
                    amplitude_refinement_enabled=amplitude_refinement_enabled,
                    misreg_az_px=misreg_az_px,
                    misreg_rg_px=misreg_rg_px,
                    residuals_only=True,
                    executor=executor,
                    device=device,
                    dask_client=dask_client,
                    geo_footprint_mask_enabled=geo_footprint_mask_enabled,
                    roi_window=roi_window_m,
                )
                measure_coreg_s = _clock_stop(device, measure_coreg_started)
                measure_times = {
                    "deramp": measure_deramp_s,
                    "coregister": measure_coreg_s,
                }
                _merge_coreg_stage_times(
                    measure_times, measure_state.coregistration_timings_s
                )
                per_burst_timings[f"measure.{tag}"] = measure_times
                if measure_state.amplitude_residual_rg_px is not None:
                    measured_amp.append(float(measure_state.amplitude_residual_rg_px))
                if measure_state.esd_azimuth_shift_px is not None:
                    measured_esd.append(float(measure_state.esd_azimuth_shift_px))
                if measure_state.prepared_geometry_field is None:
                    reject_invalid_state(
                        f"missing prepared geometry field for measured burst {tag}"
                    )
                prepared_geometry_by_tag[tag] = measure_state.prepared_geometry_field
                del measure_state, ref, sec
                gc.collect()
        if amplitude_refinement_enabled and measured_amp:
            force_amp_rg = float(
                np.nanmedian(np.asarray(measured_amp, dtype=np.float64))
            )
        if esd_enabled and measured_esd:
            force_esd_az = float(
                np.nanmedian(np.asarray(measured_esd, dtype=np.float64))
            )
        logger.info(
            "Multi-burst common residual (ISCE2-style) amp_rg=%s esd_az=%s "
            "from n_amp=%d n_esd=%d burst measures",
            f"{force_amp_rg:.4f}" if force_amp_rg is not None else "None",
            f"{force_esd_az:.4f}" if force_esd_az is not None else "None",
            len(measured_amp),
            len(measured_esd),
        )

    for swath in reversed(swath_tuple):
        for frame_index, burst_index, azimuth_offset in units_by_swath[swath]:
            tag = "f" + str(frame_index) + "_" + swath + "_b" + str(burst_index)
            ref = load_burst(
                swath,
                burst_index,
                primary_paths[frame_index],
                ref_orbits[frame_index],
                primary_products[frame_index],
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
                primary=ref,
                secondary=sec,
                dem=dem_sampler,
                coregistration_grid=coregistration_grid,
                multilook=multilook,
                goldstein_alpha=0.0,
                unwrap_method="snaphu",
                record_scientific_lineage=record_scientific_lineage,
            )
            if first_state is None:
                first_state = state
            if origin_state is None:
                origin_state = state
            stage_times: dict[str, float] = {}
            t0 = _clock_start(device)
            state = stage_deramp(
                state,
                device=device,
                dask_client=dask_client,
            )
            stage_times["deramp"] = _clock_stop(device, t0)
            t0 = _clock_start(device)
            roi_window: tuple[int, int, int, int] | None = None
            if roi is not None:
                assert state.primary_deramped is not None
                roi_window = _roi_burst_window(
                    roi,
                    ref.geometry,
                    dem_sampler,
                    state.primary_deramped.shape,
                    device=device,
                    buffer_m=roi_buffer_m,
                )
            state = stage_coregister(
                state,
                control_spacing=control_spacing,
                esd_enabled=esd_enabled,
                amplitude_refinement_enabled=amplitude_refinement_enabled,
                misreg_az_px=misreg_az_px,
                misreg_rg_px=misreg_rg_px,
                force_amplitude_residual_rg=force_amp_rg,
                force_esd_azimuth_shift_px=force_esd_az,
                prepared_geometry_field=prepared_geometry_by_tag.get(tag),
                executor=executor,
                device=device,
                dask_client=dask_client,
                geo_footprint_mask_enabled=geo_footprint_mask_enabled,
                roi_window=roi_window,
            )
            stage_times["coregister"] = _clock_stop(device, t0)
            _merge_coreg_stage_times(stage_times, state.coregistration_timings_s)
            burst_row0 = 0
            burst_col0 = 0
            if state.radar_roi_origin is not None:
                burst_row0, burst_col0 = state.radar_roi_origin
            assert state.primary_deramped is not None
            assert state.secondary_aligned is not None
            if scene_store_dir is not None:
                from faninsar.io.storage.scene_store import write_scene_unit

                write_scene_unit(
                    scene_store_dir,
                    date_id=_scene_id(sec_paths[0]),
                    reference_id=_scene_id(primary_paths[0]),
                    domain="radar",
                    tag=tag,
                    primary=np.asarray(state.primary_deramped, dtype=np.complex64),
                    secondary=np.asarray(state.secondary_aligned, dtype=np.complex64),
                    row_origin=azimuth_offset + burst_row0,
                    col_origin=range_offsets[swath] + burst_col0,
                    grid_shape=(frame_rows, frame_cols),
                    wavelength_m=float(state.primary.geometry.wavelength_m),
                    crs=None,
                    transform=None,
                    scientific_lineage=state.scientific_lineage,
                    phase_state=_phase_state_manifest(state),
                )
            pri_power = state.primary_deramped.real**2 + state.primary_deramped.imag**2
            sec_power = (
                state.secondary_aligned.real**2 + state.secondary_aligned.imag**2
            )
            t0 = _clock_start(device)
            state = stage_interferogram(
                state,
                multilook=(1, 1),
                goldstein_alpha=0.0,
                dead_pixel_amp_threshold=dead_pixel_amp_threshold,
                device=device,
                dask_client=dask_client,
            )
            stage_times["interferogram"] = _clock_stop(device, t0)
            t0 = _clock_start(device)
            state = stage_flatten(
                state,
                device=device,
                dask_client=dask_client,
            )
            stage_times["flatten"] = _clock_stop(device, t0)
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

    merge_started = time.perf_counter()
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
    merge_s = time.perf_counter() - merge_started

    filtered = merged_ifg
    goldstein_started = time.perf_counter()
    if goldstein_alpha > 0.0:
        from faninsar.processing.interferometry.pair import goldstein_filter

        filtered = goldstein_filter(merged_ifg, alpha=goldstein_alpha)
        # Keep the saved wrapped phase consistent with the filtered product;
        # unwrapping also consumes the filtered phase (standard practice).
        wrapped = np.where(invalid, np.nan, np.angle(filtered).astype(np.float32))
    goldstein_s = time.perf_counter() - goldstein_started

    result = ProductionPairState(
        pair_id=_scene_id(primary_paths[0]) + "_" + _scene_id(sec_paths[0]) + "_pair",
        primary=(
            origin_state.primary if origin_state is not None else first_state.primary
        ),
        secondary=(
            origin_state.secondary
            if origin_state is not None
            else first_state.secondary
        ),
        dem=dem_sampler,
        coregistration_grid=coregistration_grid,
        dem_id=_dem_id(dem_sampler),
        coreg_executor=str(executor),
        coreg_device=str(device),
        multilook=multilook,
        goldstein_alpha=float(goldstein_alpha),
        unwrap_method="snaphu",
        record_scientific_lineage=record_scientific_lineage,
        scientific_lineage=(
            list(origin_state.scientific_lineage) if origin_state is not None else []
        ),
        phase_state=(origin_state.phase_state if origin_state is not None else None),
        complex_ifg=merged_ifg,
        complex_ifg_flat=filtered,
        coherence=coherence,
        wrapped_phase=wrapped,
        stage_timings_s={
            "total": time.perf_counter() - total_started,
            "per_burst": per_burst_timings,
            "merge": merge_s,
            "goldstein": goldstein_s,
        },
    )
    if origin_state is not None:
        _inherit_coregistration_residuals(result, origin_state)
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
        from faninsar.processing.unwrapping import SnaphuConfig

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


def _produce_interferogram_sweep(
    primary_path: str | Path | Sequence[str | Path],
    secondary_path: str | Path | Sequence[str | Path],
    *,
    output_dir: str | Path,
    roi: BoundingBox | Polygons | None,
    swaths: tuple[str, ...] | None,
    bursts: BurstSelection | None,
    dem: DEM | None,
    multilook: object,
    overwrite: bool,
    goldstein_alpha: float,
    dead_pixel_amp_threshold: float,
    esd_enabled: bool,
    amplitude_refinement_enabled: bool,
    control_spacing: int | None,
    executor: str,
    device: str,
    dask_client: Any | None,
    coregistration_grid: CoregistrationGrid,
    geo_grid: GeoGridSpec | None,
    geo_height_m: float,
    geo_chunk_size: int,
    geo_work_dir: str | Path | None,
    geo_lut_cache_dir: str | Path | None = None,
    geo_footprint_mask_enabled: bool = True,
    scene_store_dir: str | Path | None = None,
    snaphu_config: SnaphuConfig | None,
    unwrap_method: UnwrapBackend | None,
    irls_kwargs: dict[str, Any] | None,
    primary_orbit_path: str | Path | Sequence[str | Path] | None,
    secondary_orbit_path: str | Path | Sequence[str | Path] | None,
    unwrap: bool,
    geoid_correction: bool,
    dem_source: str | None = None,
    record_scientific_lineage: bool = False,
    n_jobs: int = 1,
    resource_limits: ResourceLimits | None = None,
    prepared_geo_lut_handles: Mapping[str, tuple[PreparedLutHandle, ProviderLeaseToken]]
    | None = None,
    prepared_provider_root: str | Path | None = None,
    source_snapshot_root: str | Path | None = None,
    roi_buffer_m: float = 320.0,
    misreg_az_px: float = 0.0,
    misreg_rg_px: float = 0.0,
) -> ProductionPairState | ProductionPairSweepResult:
    """Run one shared prefix and emit every look configuration."""
    from faninsar.missions.s1.safe import open_safe_product

    if (prepared_geo_lut_handles is None) != (prepared_provider_root is None):
        reject_invalid_state(
            "prepared Geo LUT reuse requires both handles and provider root"
        )
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        reject_invalid_state("n_jobs must be a positive integer")
    _, resolved_ampcor_device = resolve_ampcor_policy(executor, device)
    if n_jobs > 1 and resolved_ampcor_device.startswith("cuda"):
        reject_invalid_state(
            "explicit CUDA Ampcor is incompatible with ProcessPoolExecutor "
            "n_jobs > 1; use n_jobs=1 for device-local admission"
        )
    if prepared_geo_lut_handles is not None and coregistration_grid != "geo":
        reject_invalid_state("prepared Geo LUT reuse requires geo coregistration")
    if prepared_geo_lut_handles is not None and not isinstance(
        prepared_geo_lut_handles, Mapping
    ):
        reject_invalid_state("prepared Geo LUT handles must be a mapping")

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

    dem_sampler: DEM = dem if dem is not None else ConstantDEM(0.0)
    snapshot_root = Path(source_snapshot_root) if source_snapshot_root else None
    if snapshot_root is not None and isinstance(dem_sampler, PublicRasterDEM):
        from faninsar.processing.runtime.source_snapshots import snapshot_local_source

        if dem_sampler.path is not None:
            dem_snapshot = snapshot_local_source(
                dem_sampler.path,
                snapshot_root / "dem",
            )
            dem_sampler = DEM.from_raster(
                dem_snapshot.path,
                vertical_datum=dem_sampler.vertical_datum,
            )
    if geoid_correction and isinstance(dem_sampler, PublicRasterDEM):
        dem_sampler = dem_sampler.to_raster(
            dem_sampler.grid,
            vertical_datum="ellipsoidal",
        )

    primary_paths = _as_frame_paths(primary_path, "primary_path")
    sec_paths = _as_frame_paths(secondary_path, "secondary_path")
    if len(primary_paths) != len(sec_paths):
        reject_invalid_state(
            "reference/secondary frame counts differ: "
            + str(len(primary_paths))
            + " vs "
            + str(len(sec_paths))
        )
    frame_count = len(primary_paths)
    ref_orbits = _as_optional_frame_sequence(
        primary_orbit_path, frame_count, "primary_orbit_path"
    )
    sec_orbits = _as_optional_frame_sequence(
        secondary_orbit_path, frame_count, "secondary_orbit_path"
    )
    if snapshot_root is not None:
        primary_paths = _snapshot_paths(primary_paths, snapshot_root / "safe")
        sec_paths = _snapshot_paths(sec_paths, snapshot_root / "safe")
        ref_orbits = _snapshot_optional_paths(
            ref_orbits,
            snapshot_root / "orbit",
        )
        sec_orbits = _snapshot_optional_paths(
            sec_orbits,
            snapshot_root / "orbit",
        )
    primary_products = [open_safe_product(path) for path in primary_paths]
    secondary_products = [open_safe_product(path) for path in sec_paths]
    if swaths is None:
        ordered = sorted(
            primary_products[0].swaths,
            key=lambda item: item.slant_range_time_s,
        )
        swath_tuple = tuple(item.swath for item in ordered)
    else:
        swath_tuple = tuple(swaths)
    if not swath_tuple:
        reject_invalid_state("swaths must not be empty")
    for frame_index, product in enumerate(primary_products):
        present = {item.swath for item in product.swaths}
        missing = [name for name in swath_tuple if name not in present]
        if missing:
            reject_invalid_state(
                "reference frame "
                + str(frame_index)
                + " ("
                + primary_paths[frame_index].name
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
        logger.info(
            "Stack provider: ROI provided; explicit swaths/bursts selection ignored"
        )
        ordered = sorted(
            primary_products[0].swaths,
            key=lambda item: item.slant_range_time_s,
        )
        swath_tuple = tuple(item.swath for item in ordered)
        resolved = _select_bursts_by_roi(
            roi,
            primary_paths,
            swath_tuple,
            orbits=ref_orbits,
            dem=dem_sampler,
        )
    else:
        burst_counts = {
            (frame_index, swath): len(primary_products[frame_index].swath(swath).bursts)
            for frame_index in range(frame_count)
            for swath in swath_tuple
        }
        resolved = _normalize_burst_selection(
            bursts, frame_count, swath_tuple, burst_counts
        )

    common_aligned: dict[tuple[int, str], list[int]] = {}
    for frame_index in range(frame_count):
        for swath in swath_tuple:
            primary_swath = primary_products[frame_index].swath(swath)
            secondary_swath = secondary_products[frame_index].swath(swath)
            common = _common_burst_indices(primary_swath, secondary_swath)
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
        bounds = _auto_dem_bounds(
            roi,
            resolved,
            primary_products,
            orbits=ref_orbits,
        )
        dem_sampler = resolve_auto_dem(
            bounds,
            output_dir=output_root,
            geoid_correction=geoid_correction,
            dem_source=dem_source,
        )

    range_offsets = _swath_range_offsets(swath_tuple, primary_products)
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

    frame_rows = max(
        azimuth_offset + burst_lines[swath]
        for swath, units in units_by_swath.items()
        for _, _, azimuth_offset in units
    )
    frame_cols = max(range_offsets[swath] + burst_width[swath] for swath in swath_tuple)

    if scene_store_dir is not None and n_jobs > 1:
        reject_invalid_state(
            "scene_store_dir requires n_jobs=1 until generation publication "
            "is coordinated across workers"
        )
    if dask_client is not None and n_jobs > 1:
        reject_invalid_state(
            "an explicit dask_client requires n_jobs=1; nested process workers "
            "cannot safely share a distributed client"
        )

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
            primary_paths=primary_paths,
            sec_paths=sec_paths,
            ref_orbits=ref_orbits,
            sec_orbits=sec_orbits,
            primary_products=primary_products,
            secondary_products=secondary_products,
            swath_tuple=swath_tuple,
            units_by_swath=units_by_swath,
            range_offsets=range_offsets,
            frame_shape=(frame_rows, frame_cols),
            roi=roi,
            dem_sampler=dem_sampler,
            control_spacing=control_spacing,
            esd_enabled=esd_enabled,
            amplitude_refinement_enabled=amplitude_refinement_enabled,
            misreg_az_px=misreg_az_px,
            misreg_rg_px=misreg_rg_px,
            executor=executor,
            device=device,
            dask_client=dask_client,
            dead_pixel_amp_threshold=dead_pixel_amp_threshold,
            coregistration_grid=coregistration_grid,
            geo_grid=geo_grid,
            geo_height_m=geo_height_m,
            geo_chunk_size=geo_chunk_size,
            geo_work_dir=resolved_geo_work_dir,
            geo_lut_cache_dir=geo_lut_cache_dir,
            geo_footprint_mask_enabled=geo_footprint_mask_enabled,
            scene_store_dir=scene_store_dir,
            n_jobs=n_jobs,
            resource_limits=resource_limits,
            prepared_geo_lut_handles=prepared_geo_lut_handles,
            prepared_provider_root=prepared_provider_root,
            roi_buffer_m=roi_buffer_m,
            record_scientific_lineage=record_scientific_lineage,
        )
        resources.ifg_archive = archive
        if coregistration_grid == "geo":
            resources.prefix_state = archive.get("geo_prefix_state")
        pair_id = _scene_id(primary_paths[0]) + "_" + _scene_id(sec_paths[0]) + "_pair"
        per_config: dict[tuple[int, int], PairSweepOutcome] = {}
        single_state: ProductionPairState | None = None
        for config in configs:
            az_looks, rg_looks = config
            merge_started = time.perf_counter()
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
            merged["merge_s"] = time.perf_counter() - merge_started
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
                record_scientific_lineage=record_scientific_lineage,
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
    primary_paths: Sequence[str | Path],
    sec_paths: Sequence[str | Path],
    ref_orbits: Sequence[Path | None],
    sec_orbits: Sequence[Path | None],
    primary_products: Sequence[object],
    secondary_products: Sequence[object],
    swath_tuple: tuple[str, ...],
    units_by_swath: dict[str, list[tuple[int, int, int]]],
    range_offsets: dict[str, int],
    frame_shape: tuple[int, int],
    roi: BoundingBox | Polygons | None,
    dem_sampler: DEM,
    control_spacing: int | None,
    esd_enabled: bool,
    amplitude_refinement_enabled: bool,
    executor: str,
    device: str,
    dask_client: Any | None,
    dead_pixel_amp_threshold: float,
    coregistration_grid: CoregistrationGrid,
    geo_grid: GeoGridSpec | None,
    geo_height_m: float,
    geo_chunk_size: int,
    geo_work_dir: Path | None,
    geo_lut_cache_dir: str | Path | None = None,
    geo_footprint_mask_enabled: bool = True,
    scene_store_dir: str | Path | None = None,
    n_jobs: int = 1,
    resource_limits: ResourceLimits | None = None,
    prepared_geo_lut_handles: Mapping[str, tuple[PreparedLutHandle, ProviderLeaseToken]]
    | None = None,
    prepared_provider_root: str | Path | None = None,
    roi_buffer_m: float = 320.0,
    misreg_az_px: float = 0.0,
    misreg_rg_px: float = 0.0,
    record_scientific_lineage: bool = False,
) -> dict[str, Any]:
    """Process every burst unit once and store flat IFGs on disk.

    Radar units are archived in radar geometry with burst placement offsets.
    Geographic units (``coregistration_grid="geo"``) are coregistered onto
    the shared ``geo_grid`` and archived as valid-bounding-box crops of the
    geocoded interferogram, per-unit power, and DEM height field.
    """
    from faninsar.missions.s1 import read_eof_orbit, read_full_burst
    from faninsar.processing.coregistration.tops.carrier import carrier_from_swath

    if coregistration_grid == "geo" and (geo_grid is None or geo_work_dir is None):
        reject_invalid_state("geo coregistration requires geo_grid and geo_work_dir")
    if (prepared_geo_lut_handles is None) != (prepared_provider_root is None):
        reject_invalid_state(
            "prepared Geo LUT reuse requires both handles and provider root"
        )
    if prepared_geo_lut_handles is not None:
        if coregistration_grid != "geo":
            reject_invalid_state("prepared Geo LUT reuse requires geo coregistration")
        if not isinstance(prepared_geo_lut_handles, Mapping):
            reject_invalid_state("prepared Geo LUT handles must be a mapping")

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
            "ref_path": primary_paths[frame_index],
            "sec_path": sec_paths[frame_index],
            "ref_orbit": ref_orbits[frame_index],
            "sec_orbit": sec_orbits[frame_index],
            "swath": swath,
            "burst_index": burst_index,
            "frame_index": frame_index,
        }

    def rebuild_origin_state(
        scene_args: dict[str, object],
        dem: DEM,
    ) -> ProductionPairState:
        ref = load_burst(
            str(scene_args["swath"]),
            int(scene_args["burst_index"]),
            Path(scene_args["ref_path"]),
            scene_args["ref_orbit"],
            primary_products[int(scene_args["frame_index"])],
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
            primary=ref,
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
            prepared_lut_handle: object | None = None
            prepared_provider_token: object | None = None
            if prepared_geo_lut_handles is not None:
                if tag not in prepared_geo_lut_handles:
                    reject_invalid_state(
                        "prepared Geo LUT handles are missing burst tag " + tag
                    )
                entry = prepared_geo_lut_handles[tag]
                if not isinstance(entry, tuple) or len(entry) != 2:
                    reject_invalid_state(
                        "prepared Geo LUT handle entry must be a handle/token pair"
                    )
                prepared_lut_handle, prepared_provider_token = entry
            task_args.append(
                {
                    "tag": tag,
                    "swath": swath,
                    "frame_index": frame_index,
                    "burst_index": burst_index,
                    "azimuth_offset": azimuth_offset,
                    "range_offset": range_offsets[swath],
                    "ref_path": primary_paths[frame_index],
                    "sec_path": sec_paths[frame_index],
                    "ref_orbit": ref_orbits[frame_index],
                    "sec_orbit": sec_orbits[frame_index],
                    "roi": roi,
                    "control_spacing": control_spacing,
                    "esd_enabled": esd_enabled,
                    "amplitude_refinement_enabled": amplitude_refinement_enabled,
                    "misreg_az_px": float(misreg_az_px),
                    "misreg_rg_px": float(misreg_rg_px),
                    "executor": executor,
                    "device": device,
                    "dask_client": dask_client,
                    "dead_pixel_amp_threshold": dead_pixel_amp_threshold,
                    "coregistration_grid": coregistration_grid,
                    "geo_grid": geo_grid,
                    "geo_height_m": geo_height_m,
                    "geo_chunk_size": geo_chunk_size,
                    "roi_buffer_m": roi_buffer_m,
                    "ifg_dir": ifg_dir,
                    "dem": dem_sampler,
                    "geo_work_dir": geo_work_dir,
                    "geo_lut_cache_dir": geo_lut_cache_dir,
                    "geo_footprint_mask_enabled": geo_footprint_mask_enabled,
                    "scene_store_dir": scene_store_dir,
                    "scene_grid_shape": (
                        geo_grid.shape
                        if coregistration_grid == "geo" and geo_grid is not None
                        else frame_shape
                    ),
                    "prepared_geo_lut_handle": prepared_lut_handle,
                    "prepared_provider_token": prepared_provider_token,
                    "prepared_provider_root": prepared_provider_root,
                    "resource_limits": resource_limits,
                    "record_scientific_lineage": record_scientific_lineage,
                    "require_worker_bootstrap": (
                        resource_limits is not None and n_jobs > 1
                    ),
                }
            )

    expected_tags = tuple(str(task["tag"]) for task in task_args)
    if len(expected_tags) != len(set(expected_tags)):
        reject_invalid_state("burst selection contains duplicate unit tags")
    if prepared_geo_lut_handles is not None and set(prepared_geo_lut_handles) != set(
        expected_tags
    ):
        reject_invalid_state(
            "prepared Geo LUT handles must cover exactly the selected burst tags"
        )

    from contextlib import nullcontext

    from faninsar.processing.runtime.resources import ProcessTreeAdmission

    worker_count = min(max(1, n_jobs), max(1, len(task_args)))
    output_file_count = len(task_args) * (4 if coregistration_grid == "geo" else 3)
    if resource_limits is None:
        logger.info(
            "burst archive running without an authenticated resource profile; "
            "resource-gate evidence is not claimed"
        )
        admission_context = nullcontext()
    else:
        admission_context = ProcessTreeAdmission.from_limits(
            resource_limits,
            work_dir,
            workers=worker_count,
            files=output_file_count,
        )

    with admission_context:
        if n_jobs > 1 and len(task_args) > 1:
            from concurrent.futures import ProcessPoolExecutor

            pool_kwargs: dict[str, object] = {"max_workers": n_jobs}
            if resource_limits is not None:
                from multiprocessing import get_context

                from faninsar.processing.runtime.resources import (
                    bootstrap_worker_runtime,
                )

                pool_kwargs.update(
                    {
                        "mp_context": get_context("spawn"),
                        "initializer": bootstrap_worker_runtime,
                        "initargs": (1,),
                    }
                )
            with ProcessPoolExecutor(**pool_kwargs) as pool:
                results = list(pool.map(_process_burst_worker, task_args))
        else:
            results = [_process_burst_worker(task) for task in task_args]

    resource_watchdog = getattr(admission_context, "watchdog", None)
    resource_peak_rss_bytes = 0
    resource_sample_count = 0
    if resource_watchdog is not None:
        resource_sample_count = len(resource_watchdog.samples)
        resource_peak_rss_bytes = max(
            (snapshot.rss_bytes for snapshot in resource_watchdog.samples),
            default=0,
        )

    ordered_results: list[tuple[dict[str, object], dict[str, float]]] = []
    produced_tags: set[str] = set()
    origin_scene_args: dict[str, object] | None = None
    for task, result in zip(task_args, results, strict=True):
        unit = result["unit"]
        if unit is None:
            reject_invalid_state(
                f"burst worker produced no unit for expected tag {task['tag']!r}"
            )
        if not isinstance(unit, dict):
            reject_invalid_state("burst worker returned an invalid unit record")
        tag = str(unit.get("tag", ""))
        if tag not in expected_tags:
            reject_invalid_state(f"burst worker produced unexpected unit tag {tag!r}")
        if tag in produced_tags:
            reject_invalid_state(f"burst worker produced duplicate unit tag {tag!r}")
        produced_tags.add(tag)
        ordered_results.append((unit, result["stage_times"]))
        if origin_scene_args is None:
            origin_scene_args = scene_rebuild_args(
                int(task["frame_index"]),
                str(task["swath"]),
                int(task["burst_index"]),
            )
    if produced_tags != set(expected_tags):
        reject_invalid_state(
            "burst worker output manifest differs from the requested set: "
            f"expected={sorted(expected_tags)} produced={sorted(produced_tags)}"
        )
    for unit, stage_times in ordered_results:
        units.append(unit)
        per_burst_timings[unit["tag"]] = stage_times
    if origin_scene_args is not None:
        origin_state = rebuild_origin_state(origin_scene_args, dem_sampler)
        for name in (
            "range_shift_px",
            "azimuth_shift_px",
            "esd_azimuth_shift_px",
            "amplitude_residual_rg_px",
        ):
            observations = [
                float(unit[name])
                for unit, _ in ordered_results
                if unit.get(name) is not None and np.isfinite(float(unit[name]))
            ]
            if observations:
                setattr(
                    origin_state,
                    name,
                    float(np.nanmedian(np.asarray(observations, dtype=np.float64))),
                )

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
        "resource_peak_rss_bytes": resource_peak_rss_bytes,
        "resource_sample_count": resource_sample_count,
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
        "resource_peak_rss_bytes": archive.get("resource_peak_rss_bytes", 0),
        "resource_sample_count": archive.get("resource_sample_count", 0),
    }


def _inherit_coregistration_residuals(
    target: ProductionPairState,
    source: ProductionPairState,
) -> None:
    """Carry measured residuals into the merged sweep result.

    The archive owns the full-burst arrays, while the final multilook result
    owns the merged product.  Residual metadata must cross that boundary so
    Stack network arcs and activation manifests cannot silently turn a valid
    Ampcor/ESD measurement into a zero correction.
    """
    for name in (
        "range_shift_px",
        "azimuth_shift_px",
        "esd_azimuth_shift_px",
        "amplitude_residual_rg_px",
    ):
        setattr(target, name, getattr(source, name))


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
    record_scientific_lineage: bool = False,
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
        # Keep the saved wrapped phase consistent with the filtered product;
        # unwrapping also consumes the filtered phase (standard practice).
        wrapped = np.where(invalid, np.nan, np.angle(filtered).astype(np.float32))

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
            record_scientific_lineage=record_scientific_lineage,
        )
    result = ProductionPairState(
        pair_id=pair_id,
        primary=origin.primary,
        secondary=origin.secondary,
        dem=origin.dem,
        coregistration_grid="radar",
        dem_id=origin.dem_id,
        coreg_executor=origin.coreg_executor,
        coreg_device=origin.coreg_device,
        multilook=config,
        goldstein_alpha=float(goldstein_alpha),
        unwrap_method="snaphu",
        record_scientific_lineage=origin.record_scientific_lineage,
        scientific_lineage=list(origin.scientific_lineage),
        phase_state=origin.phase_state,
        complex_ifg=merged_ifg,
        complex_ifg_flat=filtered,
        coherence=coherence,
        wrapped_phase=wrapped,
        stage_timings_s={
            "total": merged["total_seconds"],
            "per_burst": merged["per_burst_timings"],
            "merge": float(merged.get("merge_s", 0.0)),
            "resource_peak_rss_bytes": merged.get("resource_peak_rss_bytes", 0),
            "resource_sample_count": merged.get("resource_sample_count", 0),
        },
    )
    _inherit_coregistration_residuals(result, origin)
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
        "wavelength_m": float(origin.primary.geometry.wavelength_m),
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
        "resource_peak_rss_bytes": archive.get("resource_peak_rss_bytes", 0),
        "resource_sample_count": archive.get("resource_sample_count", 0),
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
    record_scientific_lineage: bool = False,
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
        # Keep the saved wrapped phase consistent with the filtered product;
        # unwrapping also consumes the filtered phase (standard practice).
        wrapped = np.where(invalid, np.nan, np.angle(filtered).astype(np.float32))
    origin = merged["origin_state"]
    resolved_method: UnwrapBackend = (
        unwrap_method if unwrap_method is not None else "irls"
    )
    resolved_irls_kwargs = dict(irls_kwargs or {})
    if resolved_method == "irls":
        resolved_irls_kwargs.setdefault("device", origin.coreg_device)
    result = ProductionPairState(
        pair_id=pair_id,
        primary=origin.primary,
        secondary=origin.secondary,
        dem=origin.dem,
        coregistration_grid="geo",
        dem_id=origin.dem_id,
        coreg_executor=origin.coreg_executor,
        coreg_device=origin.coreg_device,
        multilook=config,
        goldstein_alpha=float(goldstein_alpha),
        unwrap_method=resolved_method,
        record_scientific_lineage=record_scientific_lineage,
        scientific_lineage=list(origin.scientific_lineage),
        phase_state=origin.phase_state,
        complex_ifg=merged_ifg,
        complex_ifg_flat=filtered,
        coherence=coherence,
        wrapped_phase=wrapped,
        geo_height_field=merged["height_field"],
        stage_timings_s={
            "total": merged["total_seconds"],
            "per_burst": merged["per_burst_timings"],
            "merge": float(merged.get("merge_s", 0.0)),
            "resource_peak_rss_bytes": merged.get("resource_peak_rss_bytes", 0),
            "resource_sample_count": merged.get("resource_sample_count", 0),
        },
    )
    _inherit_coregistration_residuals(result, origin)
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
        "wavelength_m": float(origin.primary.geometry.wavelength_m),
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
