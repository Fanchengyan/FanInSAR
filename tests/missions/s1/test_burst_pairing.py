"""Cross-satellite common-burst pairing by along-track position.

Sentinel-1 products clip the burst grid to different start lines: a given
ground strip may sit at burst 7 of a Sentinel-1C swath but burst 5 of the
matching Sentinel-1D swath.  ``_common_burst_pairs`` must recover that whole
index offset (ISCE2 ``getBurstOffset`` model) so the same ground is paired.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.missions.protocols import DopplerCentroidPolynomial
from faninsar.missions.s1.processing import (
    _common_burst_indices,
    _common_burst_pairs,
)
from faninsar.missions.s1.types import S1Burst, S1Swath

from .test_measure import _LINES, _SAMPLES

_BURST_STEP_S = 2.756
# One burst's azimuth extent equals the burst step, matching real TOPS
# products where consecutive bursts are separated by ~1 lines_per_burst.
_ATI_S = _BURST_STEP_S / _LINES


def _offset_swath(
    count: int,
    *,
    first_burst: int,
    azimuth_origin: datetime,
    anx_origin: float,
) -> S1Swath:
    """Build a swath whose bursts step ``_BURST_STEP_S`` apart on the orbit.

    ``first_burst`` shifts where burst index 0 lands on the shared orbit grid
    (in burst steps); ``anx_origin`` supplies distinct along-track times so
    the ANX axis is exercised (real products carry annotation-derived ANX
    times).
    """
    bursts = []
    for index in range(count):
        orbit_burst = first_burst + index
        start_line = orbit_burst * _LINES
        azimuth_time = azimuth_origin + timedelta(seconds=start_line * _ATI_S)
        bursts.append(
            S1Burst(
                index=index,
                azimuth_time=azimuth_time,
                sensing_time=azimuth_time,
                azimuth_anx_time_s=anx_origin + _BURST_STEP_S * index,
                byte_offset=index * _LINES * _SAMPLES * 4,
                lines=_LINES,
                samples=_SAMPLES,
                first_valid_sample=tuple([0] * _LINES),
                last_valid_sample=tuple([_SAMPLES - 1] * _LINES),
            )
        )
    start = bursts[0].azimuth_time
    stop = bursts[-1].azimuth_time + timedelta(seconds=(_LINES - 1) * _ATI_S)
    return S1Swath(
        swath="IW2",
        polarization="VV",
        annotation_path="memory://annotation.xml",
        measurement_path="memory://measurement.tiff",
        lines=_LINES * count,
        samples=_SAMPLES,
        lines_per_burst=_LINES,
        samples_per_burst=_SAMPLES,
        range_pixel_spacing_m=2.3,
        azimuth_pixel_spacing_m=14.0,
        azimuth_time_interval_s=_ATI_S,
        slant_range_time_s=0.005,
        range_sampling_rate_hz=64_000_000.0,
        radar_frequency_hz=5.405e9,
        sensing_start=start,
        sensing_stop=stop,
        bursts=tuple(bursts),
        orbit=OrbitMetadata(
            reference_frame="ITRF",
            source="test",
            vectors=(
                OrbitStateVector(
                    time=start,
                    position_m=(7_000_000.0, 0.0, 0.0),
                    velocity_m_s=(0.0, 7_500.0, 0.0),
                ),
            ),
        ),
        doppler_centroid=(
            DopplerCentroidPolynomial(
                coefficients_hz=(100.0, -0.01),
                reference_range_m=800_000.0,
                reference_time_s=0.0,
            ),
        ),
        azimuth_fm_rate=(),
        azimuth_steering_rate_rad_s=0.0,
    )


def test_common_burst_pairs_recovers_whole_index_offset() -> None:
    """S1C(9 bursts) vs S1D(10 bursts) offset by -2 pairs the same ground."""
    s1c = _offset_swath(
        9,
        first_burst=0,
        azimuth_origin=datetime(2026, 7, 1, 23, 26, 6, tzinfo=UTC),
        anx_origin=1_000.0,
    )
    # S1D starts two orbit-grid burst steps later: its burst *i* covers the
    # ground of S1C burst *i + 2*.
    s1d = _offset_swath(
        10,
        first_burst=2,
        azimuth_origin=datetime(2026, 7, 15, 23, 26, 27, tzinfo=UTC),
        anx_origin=1_000.0 + 2 * _BURST_STEP_S,
    )
    pairs = _common_burst_pairs(s1c, s1d)
    assert pairs[7] == 5
    assert pairs[2] == 0
    assert _common_burst_indices(s1c, s1d) == list(range(2, 9))


def test_common_burst_pairs_same_grid_is_identity() -> None:
    """Same-satellite products keep the index-for-index mapping."""
    ref = _offset_swath(
        6,
        first_burst=0,
        azimuth_origin=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
        anx_origin=500.0,
    )
    sec = _offset_swath(
        6,
        first_burst=0,
        azimuth_origin=datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC),
        anx_origin=500.0,
    )
    pairs = _common_burst_pairs(ref, sec)
    assert pairs == {index: index for index in range(6)}


def test_common_burst_pairs_drops_unmatched_ground() -> None:
    """Bursts with no secondary counterpart on the same strip are omitted."""
    ref = _offset_swath(
        8,
        first_burst=0,
        azimuth_origin=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
        anx_origin=200.0,
    )
    # Secondary only covers the last three bursts' worth of ground.
    sec = _offset_swath(
        3,
        first_burst=5,
        azimuth_origin=datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC),
        anx_origin=200.0 + 5 * _BURST_STEP_S,
    )
    pairs = _common_burst_pairs(ref, sec)
    assert set(pairs.values()) == {0, 1, 2}
    assert set(pairs) == {5, 6, 7}
