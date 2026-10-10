"""Radar-domain Sentinel-1 pair misregistration measurement (developer stage)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.missions.s1.io import BurstArray, read_full_burst
from faninsar.missions.s1.orbit import read_eof_orbit
from faninsar.missions.s1.overlap import BurstOverlapWindow, burst_overlap_windows
from faninsar.missions.s1.processing import (
    BurstSelection,
    ProductionScene,
    _as_frame_paths,
    _common_burst_indices,
    _normalize_burst_selection,
    _radar_model,
    _scene_id,
)
from faninsar.missions.s1.safe import open_safe_product
from faninsar.processing.coregistration.ampcor_backend import AmpcorCandidateError
from faninsar.processing.coregistration.dense_geometry import dense_geometry_offsets
from faninsar.processing.coregistration.esd_overlap import (
    OverlapESDInput,
    OverlapESDRecord,
    aggregate_overlap_esd,
    estimate_overlap_esd,
)
from faninsar.processing.coregistration.misreg_network import MisregArc
from faninsar.processing.coregistration.offsets import (
    PatchAmplitudeShiftResult,
    estimate_patch_amplitude_shift,
    resample_complex_deramped_reramp,
)
from faninsar.processing.coregistration.tops.carrier import carrier_from_swath
from faninsar.processing.coregistration.tops.deramp import deramp
from faninsar.processing.errors import InvalidProcessingStateError, reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from faninsar.missions.s1.types import S1Product, S1Swath
    from faninsar.processing.coregistration.offsets import OffsetFieldResult
    from faninsar.processing.geometry import DEM

logger = setup_logger(__name__)

__all__ = ["measure_s1_pair_misreg"]

_ESD_MULTILOOK = (5, 15)
_LANCZOS_A = 4
_AMPCOR_MIN_CHIP = 8
_SIGMA_FLOOR_PX = 0.01


def measure_s1_pair_misreg(
    primary_path: Path | Sequence[Path],
    secondary_path: Path | Sequence[Path],
    *,
    dem: DEM,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    coherence_threshold: float = 0.85,
    ampcor_snr_threshold: float = 6.0,
    primary_orbit_path: Path | None = None,
    secondary_orbit_path: Path | None = None,
    device: str = "auto",
) -> MisregArc:
    """Measure one pair's overlap-ESD azimuth and overlap-Ampcor range residuals.

    Radar-domain only. For each
    :class:`~faninsar.missions.s1.overlap.BurstOverlapWindow` of the selected
    bursts, geometry-resample the full overlap strip, estimate overlap ESD
    (azimuth) and overlap Ampcor ``dx`` (range) independently, then aggregate
    to one
    :class:`~faninsar.processing.coregistration.misreg_network.MisregArc`.
    A finite float is returned only for a valid observation (F17); otherwise
    that component is ``None``. No overlap windows yields both ``None``.

    Parameters
    ----------
    primary_path, secondary_path : path or sequence of path
        One or more SAFE products per date. Sequences are consecutive frames
        along the same pass.
    dem : DEM
        Height sampler used for geometry-only overlap resampling.
    swaths : tuple of str, optional
        Sub-swaths to measure. ``None`` selects every sub-swath of the
        reference product, ordered by slant-range time.
    bursts : BurstSelection, optional
        Per-swath burst selection. ``None`` selects every burst of each
        selected swath.
    coherence_threshold : float, optional
        Overlap-ESD coherence gate. Default 0.85.
    ampcor_snr_threshold : float, optional
        Overlap-Ampcor SNR cull. Default 6.0.
    primary_orbit_path, secondary_orbit_path : path, optional
        Precise ESA EOF orbits. Annotation orbits are used when omitted.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Numerical device for geometry resampling.

    Returns
    -------
    MisregArc
        Pair observation with ``float | None`` components. Date ids are
        ``YYYYMMDD`` parsed from the product names. ``method`` is ``overlap``.

    Raises
    ------
    InvalidProcessingStateError
        If the pair has no frames, thresholds are invalid, or selected
        swaths are missing from a product.

    """
    if not 0.0 < float(coherence_threshold) <= 1.0:
        reject_invalid_state("coherence_threshold must lie in (0, 1]")
    if not np.isfinite(ampcor_snr_threshold) or float(ampcor_snr_threshold) < 0.0:
        reject_invalid_state("ampcor_snr_threshold must be finite and non-negative")

    primary_paths = _as_frame_paths(primary_path, "primary_path")
    secondary_paths = _as_frame_paths(secondary_path, "secondary_path")
    if len(primary_paths) != len(secondary_paths):
        reject_invalid_state(
            "reference/secondary frame counts differ: "
            f"{len(primary_paths)} vs {len(secondary_paths)}"
        )
    primary_id = _scene_id(primary_paths[0])
    secondary_id = _scene_id(secondary_paths[0])
    primary_products = [open_safe_product(path) for path in primary_paths]
    secondary_products = [open_safe_product(path) for path in secondary_paths]
    swath_tuple = _resolve_swaths(primary_products, secondary_products, swaths)
    resolved = _resolve_common_bursts(
        primary_products,
        secondary_products,
        swath_tuple,
        bursts,
    )
    windows = _collect_overlap_windows(
        primary_products,
        primary_orbit_path=primary_orbit_path,
        resolved=resolved,
        swath_tuple=swath_tuple,
    )
    if not windows:
        logger.warning(
            "No burst-overlap windows for pair %s_%s; omitting azimuth and range",
            primary_id,
            secondary_id,
        )
        return MisregArc(
            primary=primary_id,
            secondary=secondary_id,
            azimuth_shift_px=None,
            range_shift_px=None,
            method="overlap",
            n_valid=0,
        )

    esd_records: list[OverlapESDRecord] = []
    ampcor_results: list[PatchAmplitudeShiftResult] = []
    for frame_index, swath_name, window in windows:
        esd, ampcor = _observe_overlap_window(
            window,
            primary_path=primary_paths[frame_index],
            secondary_path=secondary_paths[frame_index],
            primary_product=primary_products[frame_index],
            secondary_product=secondary_products[frame_index],
            swath_name=swath_name,
            dem=dem,
            coherence_threshold=coherence_threshold,
            ampcor_snr_threshold=ampcor_snr_threshold,
            primary_orbit_path=primary_orbit_path,
            secondary_orbit_path=secondary_orbit_path,
            device=device,
        )
        if esd is not None:
            esd_records.append(esd)
        if ampcor is not None:
            ampcor_results.append(ampcor)
    return _aggregate_pair_arc(primary_id, secondary_id, esd_records, ampcor_results)


def _resolve_swaths(
    primary_products: Sequence[S1Product],
    secondary_products: Sequence[S1Product],
    swaths: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Return the ordered swath names present on every frame of both dates."""
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
    for role, products in (
        ("reference", primary_products),
        ("secondary", secondary_products),
    ):
        for frame_index, product in enumerate(products):
            present = {item.swath for item in product.swaths}
            missing = [name for name in swath_tuple if name not in present]
            if missing:
                reject_invalid_state(
                    f"{role} frame {frame_index} missing swaths {missing}; "
                    f"available={sorted(present)}"
                )
    return swath_tuple


def _resolve_common_bursts(
    primary_products: Sequence[S1Product],
    secondary_products: Sequence[S1Product],
    swath_tuple: tuple[str, ...],
    bursts: BurstSelection | None,
) -> dict[tuple[int, str], list[int]]:
    """Intersect the requested burst selection with common primary/secondary indices."""
    frame_count = len(primary_products)
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
            common = set(_common_burst_indices(primary_swath, secondary_swath))
            selected = [
                index for index in resolved[(frame_index, swath)] if index in common
            ]
            if not selected:
                reject_invalid_state(
                    "no common bursts between reference/secondary frame "
                    f"{frame_index} swath {swath}"
                )
            common_aligned[(frame_index, swath)] = selected
    return common_aligned


def _collect_overlap_windows(
    primary_products: Sequence[S1Product],
    *,
    primary_orbit_path: Path | None,
    resolved: dict[tuple[int, str], list[int]],
    swath_tuple: tuple[str, ...],
) -> tuple[tuple[int, str, BurstOverlapWindow], ...]:
    """Return overlap windows of the selected bursts, tagged by frame and swath."""
    windows: list[tuple[int, str, BurstOverlapWindow]] = []
    for frame_index, product in enumerate(primary_products):
        for swath_name in swath_tuple:
            selected = resolved.get((frame_index, swath_name), [])
            if len(selected) < 2:
                continue
            swath = _swath_with_orbit(
                product.swath(swath_name),
                primary_orbit_path,
            )
            windows.extend(
                (frame_index, swath_name, window)
                for window in burst_overlap_windows(swath, burst_indices=selected)
            )
    return tuple(windows)


def _swath_with_orbit(swath: S1Swath, orbit_path: Path | None) -> S1Swath:
    """Attach a precise orbit when the caller supplied one."""
    if orbit_path is None:
        return swath
    return replace(swath, orbit=read_eof_orbit(orbit_path))


def _observe_overlap_window(
    window: BurstOverlapWindow,
    *,
    primary_path: Path,
    secondary_path: Path,
    primary_product: S1Product,
    secondary_product: S1Product,
    swath_name: str,
    dem: DEM,
    coherence_threshold: float,
    ampcor_snr_threshold: float,
    primary_orbit_path: Path | None,
    secondary_orbit_path: Path | None,
    device: str,
) -> tuple[OverlapESDRecord | None, PatchAmplitudeShiftResult | None]:
    """Measure ESD and Ampcor independently on one full overlap strip.

    Ampcor failure or an empty chip never aborts the ESD observation (F15/F17).
    """
    data = _geometry_resampled_overlap(
        window,
        primary_path=primary_path,
        secondary_path=secondary_path,
        primary_product=primary_product,
        secondary_product=secondary_product,
        swath_name=swath_name,
        dem=dem,
        primary_orbit_path=primary_orbit_path,
        secondary_orbit_path=secondary_orbit_path,
        device=device,
    )
    if data is None:
        return None, None
    esd = estimate_overlap_esd(data, coherence_threshold=coherence_threshold)
    ampcor = _overlap_ampcor_range(
        data,
        snr_threshold=ampcor_snr_threshold,
        device=device,
    )
    return esd, ampcor


def _geometry_resampled_overlap(
    window: BurstOverlapWindow,
    *,
    primary_path: Path,
    secondary_path: Path,
    primary_product: S1Product,
    secondary_product: S1Product,
    swath_name: str,
    dem: DEM,
    primary_orbit_path: Path | None,
    secondary_orbit_path: Path | None,
    device: str,
) -> OverlapESDInput | None:
    """Geometry-resample full overlap strips in the original focused domain."""
    try:
        ref_top = _load_burst_scene(
            primary_path,
            primary_product,
            swath_name,
            window.top_index,
            primary_orbit_path,
        )
        ref_bot = _load_burst_scene(
            primary_path,
            primary_product,
            swath_name,
            window.bottom_index,
            primary_orbit_path,
        )
        sec_top = _load_burst_scene(
            secondary_path,
            secondary_product,
            swath_name,
            window.top_index,
            secondary_orbit_path,
        )
        sec_bot = _load_burst_scene(
            secondary_path,
            secondary_product,
            swath_name,
            window.bottom_index,
            secondary_orbit_path,
        )
    except (OSError, InvalidProcessingStateError) as error:
        logger.warning(
            "Skipping overlap %s/%s on %s: failed to load bursts (%s)",
            window.top_index,
            window.bottom_index,
            swath_name,
            error,
        )
        return None

    sec_top_deramped = deramp(sec_top.array.samples, sec_top.carrier)
    sec_bot_deramped = deramp(sec_bot.array.samples, sec_bot.carrier)
    top_field = dense_geometry_offsets(
        sec_top.array.samples.shape,
        reference_model=ref_top.geometry,
        secondary_model=sec_top.geometry,
        device=device,
        dem=dem,
        stride=8,
        row0=0,
        col0=0,
    )
    bot_field = dense_geometry_offsets(
        sec_bot.array.samples.shape,
        reference_model=ref_bot.geometry,
        secondary_model=sec_bot.geometry,
        device=device,
        dem=dem,
        stride=8,
        row0=0,
        col0=0,
    )
    top_pack = _resample_window_strip(
        window_row0=window.top_row0,
        window_col0=window.col0,
        n_lines=window.n_lines,
        n_samples=window.n_samples,
        reference=ref_top.array,
        secondary_deramped=sec_top_deramped,
        secondary_carrier=sec_top.carrier,
        offsets=top_field,
        device=device,
    )
    bot_pack = _resample_window_strip(
        window_row0=window.bottom_row0,
        window_col0=window.col0,
        n_lines=window.n_lines,
        n_samples=window.n_samples,
        reference=ref_bot.array,
        secondary_deramped=sec_bot_deramped,
        secondary_carrier=sec_bot.carrier,
        offsets=bot_field,
        device=device,
    )
    if top_pack is None or bot_pack is None:
        logger.warning(
            "Skipping overlap %s/%s on %s: window crop is outside a burst",
            window.top_index,
            window.bottom_index,
            swath_name,
        )
        return None
    ref_top_crop, sec_top_crop, top_az, top_rg = top_pack
    ref_bot_crop, sec_bot_crop, bot_az, bot_rg = bot_pack
    return OverlapESDInput(
        ref_top=ref_top_crop,
        ref_bot=ref_bot_crop,
        sec_top=sec_top_crop,
        sec_bot=sec_bot_crop,
        sec_top_carrier=sec_top.carrier,
        sec_bot_carrier=sec_bot.carrier,
        top_row0=window.top_row0,
        bot_row0=window.bottom_row0,
        top_col0=window.col0,
        bot_col0=window.col0,
        top_native_height=int(sec_top.array.samples.shape[0]),
        bot_native_height=int(sec_bot.array.samples.shape[0]),
        top_offsets=(top_az, top_rg),
        bot_offsets=(bot_az, bot_rg),
        ref_top_carrier=ref_top.carrier,
        ref_bot_carrier=ref_bot.carrier,
        ref_top_row0=window.top_row0,
        ref_bot_row0=window.bottom_row0,
        ref_top_col0=window.col0,
        ref_bot_col0=window.col0,
        ref_top_native_height=int(ref_top.array.samples.shape[0]),
        ref_bot_native_height=int(ref_bot.array.samples.shape[0]),
        top_index=window.top_index,
        bottom_index=window.bottom_index,
    )


def _load_burst_scene(
    path: Path,
    product: S1Product,
    swath_name: str,
    burst_index: int,
    orbit_path: Path | None,
) -> ProductionScene:
    """Load one full-range burst with carrier and radar geometry."""
    swath = _swath_with_orbit(product.swath(swath_name), orbit_path)
    array = read_full_burst(swath, burst_index=burst_index, full_range=True)
    burst = swath.bursts[burst_index]
    carrier = carrier_from_swath(swath, burst, first_range_sample=array.col0)
    geometry = _radar_model(
        swath,
        burst,
        shape=array.samples.shape,
        row0=array.row0,
        col0=array.col0,
    )
    return ProductionScene(
        scene_id=_scene_id(path),
        path=path,
        product=product,
        swath=swath,
        burst=burst,
        array=array,
        carrier=carrier,
        geometry=geometry,
    )


def _resample_window_strip(
    *,
    window_row0: int,
    window_col0: int,
    n_lines: int,
    n_samples: int,
    reference: BurstArray,
    secondary_deramped: np.ndarray,
    secondary_carrier: object,
    offsets: OffsetFieldResult,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Crop, halo-pad, and geometry-resample one overlap strip.

    The secondary is deramped on the **full burst** before this crop; the
    output secondary strip is original-domain (deramp-resample-reramp).
    """
    height, width = reference.samples.shape
    # Full-range radar reads place burst line 0 at samples[0]; window coordinates
    # are burst-local. Range origin follows BurstArray.col0 when the array is
    # a valid-sample envelope rather than a full-width read.
    row0 = int(window_row0)
    col0 = int(window_col0) - int(reference.col0)
    if (
        row0 < 0
        or col0 < 0
        or row0 + n_lines > height
        or col0 + n_samples > width
        or secondary_deramped.shape != reference.samples.shape
        or offsets.azimuth_offset_px.shape != reference.samples.shape
    ):
        return None

    az_window = np.asarray(
        offsets.azimuth_offset_px[row0 : row0 + n_lines, col0 : col0 + n_samples],
        dtype=np.float64,
    )
    rg_window = np.asarray(
        offsets.range_offset_px[row0 : row0 + n_lines, col0 : col0 + n_samples],
        dtype=np.float64,
    )
    max_az = float(np.nanmax(np.abs(az_window))) if az_window.size else 0.0
    max_rg = float(np.nanmax(np.abs(rg_window))) if rg_window.size else 0.0
    if not np.isfinite(max_az):
        max_az = 0.0
    if not np.isfinite(max_rg):
        max_rg = 0.0
    halo_az = int(np.ceil(_LANCZOS_A + max_az + _ESD_MULTILOOK[0]))
    halo_rg = int(np.ceil(_LANCZOS_A + max_rg + _ESD_MULTILOOK[1]))
    pad_r0 = max(0, row0 - halo_az)
    pad_r1 = min(height, row0 + n_lines + halo_az)
    pad_c0 = max(0, col0 - halo_rg)
    pad_c1 = min(width, col0 + n_samples + halo_rg)
    sec_pad = secondary_deramped[pad_r0:pad_r1, pad_c0:pad_c1]
    az_pad = offsets.azimuth_offset_px[pad_r0:pad_r1, pad_c0:pad_c1]
    rg_pad = offsets.range_offset_px[pad_r0:pad_r1, pad_c0:pad_c1]
    sec_resamp = resample_complex_deramped_reramp(
        sec_pad,
        secondary_carrier=secondary_carrier,
        range_offset_px=rg_pad,
        azimuth_offset_px=az_pad,
        lanczos_a=_LANCZOS_A,
        device=device,
        row0=pad_r0,
        col0=pad_c0 + int(reference.col0),
        native_height=height,
    )
    inner_r0 = row0 - pad_r0
    inner_c0 = col0 - pad_c0
    sec_crop = np.asarray(
        sec_resamp[inner_r0 : inner_r0 + n_lines, inner_c0 : inner_c0 + n_samples]
    )
    ref_crop = np.asarray(
        reference.samples[row0 : row0 + n_lines, col0 : col0 + n_samples]
    )
    return ref_crop, sec_crop, az_window, rg_window


def _overlap_ampcor_range(
    data: OverlapESDInput,
    *,
    snr_threshold: float,
    device: str,
) -> PatchAmplitudeShiftResult | None:
    """Estimate range Ampcor on the same overlap strips; never raise into ESD."""
    results: list[PatchAmplitudeShiftResult] = []
    for reference, secondary in (
        (data.ref_top, data.sec_top),
        (data.ref_bot, data.sec_bot),
    ):
        result = _ampcor_one_strip(
            reference,
            secondary,
            snr_threshold=snr_threshold,
            device=device,
        )
        if result is not None:
            results.append(result)
    if not results:
        return PatchAmplitudeShiftResult(0.0, 0.0, 0, 0.0, 0)
    valid = [item for item in results if _is_valid_ampcor(item)]
    if not valid:
        attempted = int(sum(item.n_attempted for item in results))
        return PatchAmplitudeShiftResult(0.0, 0.0, 0, 0.0, attempted)
    dx = float(np.median([item.range_shift_px for item in valid]))
    return PatchAmplitudeShiftResult(
        range_shift_px=dx,
        azimuth_shift_px=0.0,
        n_valid=int(sum(item.n_valid for item in valid)),
        snr_median=float(np.median([item.snr_median for item in valid])),
        n_attempted=int(sum(item.n_attempted for item in results)),
    )


def _ampcor_one_strip(
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    snr_threshold: float,
    device: str,
) -> PatchAmplitudeShiftResult | None:
    """Run Ampcor on one strip, shrinking the chip to fit or skipping it."""
    params = _fit_ampcor_chip(reference.shape)
    if params is None:
        logger.warning(
            "Overlap Ampcor skipped: strip cannot host a chip shape=%s",
            reference.shape,
        )
        return None
    try:
        return estimate_patch_amplitude_shift(
            reference,
            secondary,
            snr_threshold=snr_threshold,
            executor="numpy",
            device=(
                "cpu"
                if str(device).strip().lower() in {"auto", "cpu", "mps"}
                else device
            ),
            **params,
        )
    except (InvalidProcessingStateError, AmpcorCandidateError, RuntimeError) as error:
        logger.warning(
            "Overlap Ampcor failed on strip %s: %s",
            reference.shape,
            error,
        )
        return None


def _fit_ampcor_chip(shape: tuple[int, int]) -> dict[str, int] | None:
    """Shrink the ISCE2 Ampcor chip until it fits the overlap strip, or skip."""
    n_lines, n_samples = (int(shape[0]), int(shape[1]))
    if n_lines < _AMPCOR_MIN_CHIP or n_samples < _AMPCOR_MIN_CHIP:
        return None
    window_az = min(32, n_lines)
    window_rg = min(64, n_samples)
    search_az = min(16, max(1, (n_lines - window_az) // 2))
    search_rg = min(16, max(1, (n_samples - window_rg) // 2))
    window_az, search_az = _shrink_axis(n_lines, window_az, search_az)
    window_rg, search_rg = _shrink_axis(n_samples, window_rg, search_rg)
    if window_az is None or window_rg is None:
        return None
    n_az = max(1, min(20, 1 + n_lines - window_az - 2 * search_az))
    n_rg = max(1, min(40, 1 + n_samples - window_rg - 2 * search_rg))
    return {
        "window_az": window_az,
        "window_rg": window_rg,
        "search_az": search_az,
        "search_rg": search_rg,
        "n_az": n_az,
        "n_rg": n_rg,
        "margin_rg": min(1000, max(0, n_samples // 8)),
        "margin_az": max(0, window_az // 2),
    }


def _shrink_axis(
    length: int,
    window: int,
    search: int,
) -> tuple[int, int] | tuple[None, None]:
    """Reduce window/search until ``window + 2 * search`` fits ``length``."""
    while window + 2 * search > length and window > _AMPCOR_MIN_CHIP:
        window -= 1
    while window + 2 * search > length and search > 1:
        search -= 1
    if window + 2 * search > length:
        return None, None
    return window, search


def _is_valid_ampcor(result: PatchAmplitudeShiftResult) -> bool:
    """Return True iff Ampcor produced a finite range observation (F17)."""
    return (
        int(result.n_valid) > 0
        and result.range_shift_px is not None
        and bool(np.isfinite(result.range_shift_px))
    )


def _aggregate_pair_arc(
    primary_id: str,
    secondary_id: str,
    esd_records: Sequence[OverlapESDRecord],
    ampcor_results: Sequence[PatchAmplitudeShiftResult],
) -> MisregArc:
    """Pool per-window observations into one pair arc with F17 omit semantics."""
    esd = aggregate_overlap_esd(esd_records)
    if esd is None:
        azimuth_shift_px: float | None = None
        azimuth_sigma_px = 1.0
        n_valid = 0
    else:
        azimuth_shift_px = float(esd.azimuth_shift_px)
        azimuth_sigma_px = float(esd.sigma_px)
        n_valid = int(esd.n_valid)

    valid_ampcor = [item for item in ampcor_results if _is_valid_ampcor(item)]
    if not valid_ampcor:
        range_shift_px: float | None = None
        range_sigma_px = 1.0
    else:
        dx = np.asarray(
            [item.range_shift_px for item in valid_ampcor],
            dtype=np.float64,
        )
        range_shift_px = float(np.median(dx))
        scatter = float(np.std(dx)) if dx.size > 1 else 0.0
        range_sigma_px = max(scatter, _SIGMA_FLOOR_PX)
        if n_valid == 0:
            n_valid = int(sum(item.n_valid for item in valid_ampcor))

    return MisregArc(
        primary=primary_id,
        secondary=secondary_id,
        azimuth_shift_px=azimuth_shift_px,
        range_shift_px=range_shift_px,
        azimuth_sigma_px=azimuth_sigma_px,
        range_sigma_px=range_sigma_px,
        method="overlap",
        n_valid=n_valid,
    )
