"""Build TOPS carrier models from Sentinel-1 annotation metadata."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.tops.deramp import SPEED_OF_LIGHT_M_S, TOPSCarrierModel

if TYPE_CHECKING:
    from faninsar.sentinel1.types import S1Burst, S1Swath

logger = setup_logger(__name__)


def _nearest_index(times: list[datetime], target: datetime) -> int:
    """Return index of the timestamp nearest to ``target``."""
    if not times:
        reject_invalid_state("no timestamps available for carrier selection")
    deltas = [abs((t - target).total_seconds()) for t in times]
    return int(min(range(len(deltas)), key=lambda i: deltas[i]))


def carrier_from_swath(
    swath: S1Swath,
    burst: S1Burst,
    *,
    first_range_sample: int = 0,
) -> TOPSCarrierModel:
    """Construct a TOPS carrier model from parsed SAFE annotation.

    Parameters
    ----------
    swath : S1Swath
        Parsed sub-swath with Doppler and FM-rate polynomials.
    burst : S1Burst
        Burst whose sensing time selects the nearest polynomials.
    first_range_sample : int, optional
        Absolute range sample index of column 0 in the complex array
        (``BurstArray.col0``).  The carrier range-time axis is shifted so
        local sample 0 maps to the correct slant-range time.

    Returns
    -------
    TOPSCarrierModel
        Carrier matching the burst window geometry.

    """
    if not swath.doppler_centroid:
        reject_invalid_state(f"swath {swath.swath} has no Doppler centroid estimates")
    if not swath.azimuth_fm_rate:
        reject_invalid_state(f"swath {swath.swath} has no azimuth FM-rate estimates")
    if first_range_sample < 0:
        reject_invalid_state("first_range_sample must be >= 0")

    burst_time = burst.sensing_time or burst.azimuth_time

    # Prefer FM-rate list (timed) to pick the estimate nearest the burst.
    fm_times = [fm[0] for fm in swath.azimuth_fm_rate]
    fm_idx = _nearest_index(fm_times, burst_time)
    _fm_time, fm_t0_s, fm_coeffs = swath.azimuth_fm_rate[fm_idx]

    # Doppler estimates are typically 1:1 with FM entries; fall back to clamp.
    dc_idx = min(fm_idx, len(swath.doppler_centroid) - 1)
    dc = swath.doppler_centroid[dc_idx]
    doppler_t0_s = dc.reference_range_m * 2.0 / SPEED_OF_LIGHT_M_S
    if dc.reference_time_s != 0.0:
        doppler_t0_s = float(dc.reference_time_s)

    # Shift range-time origin so local sample 0 == absolute sample first_range_sample.
    slant_range_time0_s = (
        swath.slant_range_time_s
        + float(first_range_sample) / swath.range_sampling_rate_hz
    )

    # Store burst mid-time for provenance / future absolute-time models.
    n_lines = max(int(burst.lines), 1)
    burst_centre_offset_s = 0.5 * (n_lines - 1) * swath.azimuth_time_interval_s
    burst_sensing_time_s = float(burst.azimuth_anx_time_s) + burst_centre_offset_s

    model = TOPSCarrierModel(
        radar_frequency_hz=swath.radar_frequency_hz,
        slant_range_time0_s=float(slant_range_time0_s),
        range_sampling_rate_hz=swath.range_sampling_rate_hz,
        azimuth_time_interval_s=swath.azimuth_time_interval_s,
        doppler_centroid_hz=tuple(float(v) for v in dc.coefficients_hz),
        doppler_t0_s=float(doppler_t0_s),
        fm_rate_hz_s=tuple(float(v) for v in fm_coeffs),
        fm_t0_s=float(fm_t0_s),
        burst_sensing_time_s=burst_sensing_time_s,
    )
    logger.info(
        "Built TOPS carrier for %s burst %s (dc_idx=%s fm_idx=%s first_rg=%s "
        "dc_order=%s fm_order=%s)",
        swath.swath,
        burst.index,
        dc_idx,
        fm_idx,
        first_range_sample,
        len(model.doppler_centroid_hz) - 1,
        len(model.fm_rate_hz_s) - 1,
    )
    return model
