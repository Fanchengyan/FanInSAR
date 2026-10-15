"""Sensing-time burst-overlap windows for Sentinel-1 TOPS swaths."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.missions.protocols import DopplerCentroidPolynomial
from faninsar.missions.s1.overlap import (
    OverlapArrayEnvelope,
    burst_overlap_windows,
    intersect_overlap_window,
)
from faninsar.missions.s1.types import S1Burst, S1Swath

if TYPE_CHECKING:
    from collections.abc import Sequence


_ORIGIN = datetime(2020, 1, 1, tzinfo=UTC)
_DT_S = 1.0
_LINES = 100
_SAMPLES = 80


def _valid_window(
    lines: int,
    *,
    first: int = 0,
    last: int | None = None,
    invalid: Sequence[int] = (),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return per-line first/last valid samples, with optional invalid lines."""
    last_sample = _SAMPLES - 1 if last is None else last
    first_vals: list[int] = []
    last_vals: list[int] = []
    invalid_set = set(invalid)
    for line in range(lines):
        if line in invalid_set:
            first_vals.append(-1)
            last_vals.append(-1)
            continue
        first_vals.append(first)
        last_vals.append(last_sample)
    return tuple(first_vals), tuple(last_vals)


def _burst(
    index: int,
    *,
    start_line: int,
    first: int = 0,
    last: int | None = None,
    invalid: Sequence[int] = (),
) -> S1Burst:
    """Build one burst whose line 0 is ``start_line`` intervals after origin."""
    first_valid, last_valid = _valid_window(
        _LINES,
        first=first,
        last=last,
        invalid=invalid,
    )
    azimuth_time = _ORIGIN + timedelta(seconds=start_line * _DT_S)
    return S1Burst(
        index=index,
        azimuth_time=azimuth_time,
        sensing_time=azimuth_time,
        azimuth_anx_time_s=0.0,
        byte_offset=index * _LINES * _SAMPLES * 4,
        lines=_LINES,
        samples=_SAMPLES,
        first_valid_sample=first_valid,
        last_valid_sample=last_valid,
    )


def _swath(*bursts: S1Burst) -> S1Swath:
    """Build a minimal IW1 swath around the given bursts."""
    start = bursts[0].azimuth_time
    stop = bursts[-1].azimuth_time + timedelta(seconds=(_LINES - 1) * _DT_S)
    return S1Swath(
        swath="IW1",
        polarization="VV",
        annotation_path="memory://annotation.xml",
        measurement_path="memory://measurement.tiff",
        lines=_LINES * len(bursts),
        samples=_SAMPLES,
        lines_per_burst=_LINES,
        samples_per_burst=_SAMPLES,
        range_pixel_spacing_m=2.3,
        azimuth_pixel_spacing_m=14.0,
        azimuth_time_interval_s=_DT_S,
        slant_range_time_s=0.005,
        range_sampling_rate_hz=64_000_000.0,
        radar_frequency_hz=5.405e9,
        sensing_start=start,
        sensing_stop=stop,
        bursts=bursts,
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


def test_one_burst_yields_no_overlap_windows() -> None:
    """A single-burst swath has no adjacent pair, so NESD has no windows."""
    swath = _swath(_burst(0, start_line=0))

    windows = burst_overlap_windows(swath)

    assert windows == ()


def test_two_overlapping_bursts_yield_sensing_time_window() -> None:
    """Adjacent bursts overlapping by 20 lines form one burst-local window."""
    # Burst 1 starts 80 intervals after burst 0, so lines 80-99 overlap 0-19.
    swath = _swath(_burst(0, start_line=0), _burst(1, start_line=80))

    windows = burst_overlap_windows(swath)

    assert len(windows) == 1
    window = windows[0]
    assert window.swath == "IW1"
    assert window.top_index == 0
    assert window.bottom_index == 1
    assert window.top_row0 == 80
    assert window.bottom_row0 == 0
    assert window.col0 == 0
    assert window.n_lines == 20
    assert window.n_samples == 80


def test_min_lines_drops_short_sensing_overlap() -> None:
    """Overlaps shorter than min_lines are omitted rather than shortened."""
    swath = _swath(_burst(0, start_line=0), _burst(1, start_line=80))

    windows = burst_overlap_windows(swath, min_lines=21)

    assert windows == ()


def test_valid_sample_intersection_sets_column_window() -> None:
    """col0/n_samples are the intersection of both bursts' valid-sample ranges."""
    top = _burst(0, start_line=0, first=10, last=60)
    bottom = _burst(1, start_line=80, first=25, last=70)
    swath = _swath(top, bottom)

    windows = burst_overlap_windows(swath)

    assert len(windows) == 1
    assert windows[0].col0 == 25
    assert windows[0].n_samples == 36
    assert windows[0].n_lines == 20


def test_empty_valid_sample_intersection_drops_window() -> None:
    """A sensing-time overlap with no common valid columns is not a window."""
    top = _burst(0, start_line=0, first=0, last=20)
    bottom = _burst(1, start_line=80, first=40, last=70)
    swath = _swath(top, bottom)

    windows = burst_overlap_windows(swath)

    assert windows == ()


def test_min_samples_drops_narrow_valid_intersection() -> None:
    """Overlaps narrower than the range-multilook floor are omitted."""
    top = _burst(0, start_line=0, first=0, last=20)
    bottom = _burst(1, start_line=80, first=10, last=19)
    swath = _swath(top, bottom)

    dropped = burst_overlap_windows(swath)
    kept = burst_overlap_windows(swath, min_samples=10)

    assert dropped == ()
    assert len(kept) == 1
    assert kept[0].col0 == 10
    assert kept[0].n_samples == 10


def test_invalid_azimuth_lines_are_trimmed_from_overlap() -> None:
    """Invalid first/last valid-sample lines shrink the azimuth overlap."""
    top = _burst(0, start_line=0, invalid=range(95, 100))
    bottom = _burst(1, start_line=80)
    swath = _swath(top, bottom)

    windows = burst_overlap_windows(swath, min_lines=10)

    assert len(windows) == 1
    assert windows[0].top_row0 == 80
    assert windows[0].bottom_row0 == 0
    assert windows[0].n_lines == 15


def test_non_adjacent_burst_indices_are_not_paired() -> None:
    """Only physically adjacent indices (i, i+1) form a window."""
    swath = _swath(
        _burst(0, start_line=0),
        _burst(1, start_line=80),
        _burst(2, start_line=160),
    )

    skipped = burst_overlap_windows(swath, burst_indices=(0, 2))
    selected = burst_overlap_windows(swath, burst_indices=(1, 2))

    assert skipped == ()
    assert len(selected) == 1
    assert selected[0].top_index == 1
    assert selected[0].bottom_index == 2


def test_intersect_overlap_window_is_identity_without_extras() -> None:
    """A window with no extra dates or envelopes is unchanged."""
    swath = _swath(_burst(0, start_line=0), _burst(1, start_line=80))
    window = burst_overlap_windows(swath)[0]

    assert intersect_overlap_window(window) == window


def test_intersect_overlap_window_clips_to_secondary_valid_samples() -> None:
    """Secondary valid-sample masks shrink the reference overlap (ISCE2-style)."""
    swath = _swath(_burst(0, start_line=0), _burst(1, start_line=80))
    window = burst_overlap_windows(swath)[0]
    sec_top = _burst(0, start_line=0, first=10, last=40)
    sec_bot = _burst(1, start_line=80, first=10, last=40)

    clipped = intersect_overlap_window(
        window,
        extra_bursts=((sec_top, "top"), (sec_bot, "bottom")),
    )

    assert clipped is not None
    assert clipped.col0 == 10
    assert clipped.n_samples == 31
    assert clipped.n_lines == window.n_lines


def test_intersect_overlap_window_clips_unequal_array_widths() -> None:
    """Loaded arrays with unequal widths keep the common columns.

    Production analogue: a pair whose reference burst is wider than the
    secondary (A100 22935 vs 22794) used to skip the window because crop
    required identical full-burst shapes.
    """
    swath = _swath(_burst(0, start_line=0), _burst(1, start_line=80))
    window = burst_overlap_windows(swath)[0]
    envelopes = (
        OverlapArrayEnvelope("top", col0=0, n_samples=40, n_lines=_LINES),
        OverlapArrayEnvelope("bottom", col0=0, n_samples=40, n_lines=_LINES),
        OverlapArrayEnvelope("top", col0=0, n_samples=30, n_lines=_LINES),
        OverlapArrayEnvelope("bottom", col0=0, n_samples=30, n_lines=_LINES),
    )

    clipped = intersect_overlap_window(window, envelopes=envelopes)

    assert clipped is not None
    assert clipped.col0 == 0
    assert clipped.n_samples == 30
    assert clipped.n_lines == window.n_lines


def test_intersect_overlap_window_none_when_envelopes_miss() -> None:
    """A secondary envelope that misses the window omits that overlap."""
    swath = _swath(_burst(0, start_line=0), _burst(1, start_line=80))
    window = burst_overlap_windows(swath)[0]
    envelopes = (
        OverlapArrayEnvelope("top", col0=0, n_samples=80, n_lines=_LINES),
        OverlapArrayEnvelope("bottom", col0=0, n_samples=80, n_lines=_LINES),
        OverlapArrayEnvelope("top", col0=200, n_samples=10, n_lines=_LINES),
        OverlapArrayEnvelope("bottom", col0=200, n_samples=10, n_lines=_LINES),
    )

    assert intersect_overlap_window(window, envelopes=envelopes) is None


def test_intersect_overlap_window_trims_invalid_secondary_azimuth_lines() -> None:
    """Trailing invalid lines on a later date shrink n_lines, not columns.

    Production analogue: 20240705 IW1 burst 0 ends the overlap with three
    ``first_valid_sample = -1`` lines.  Using ``min(last)`` over those
    ``-1`` values used to empty the column intersection.
    """
    swath = _swath(_burst(0, start_line=0), _burst(1, start_line=80))
    window = burst_overlap_windows(swath)[0]
    invalid_top = range(window.top_row0 + window.n_lines - 3, _LINES)
    sec_top = _burst(0, start_line=0, first=5, last=50, invalid=invalid_top)
    sec_bot = _burst(1, start_line=80, first=5, last=50)

    clipped = intersect_overlap_window(
        window,
        extra_bursts=((sec_top, "top"), (sec_bot, "bottom")),
        min_lines=10,
    )

    assert clipped is not None
    assert clipped.n_lines == window.n_lines - 3
    assert clipped.top_row0 == window.top_row0
    assert clipped.bottom_row0 == window.bottom_row0
    assert clipped.col0 == 5
    assert clipped.n_samples == 46


def test_intersect_overlap_window_advances_past_leading_invalid_lines() -> None:
    """Invalid lines at the start of the overlap keep the strips aligned."""
    swath = _swath(_burst(0, start_line=0), _burst(1, start_line=80))
    window = burst_overlap_windows(swath)[0]
    invalid_top = range(window.top_row0, window.top_row0 + 4)
    sec_top = _burst(0, start_line=0, invalid=invalid_top)
    sec_bot = _burst(1, start_line=80)

    clipped = intersect_overlap_window(
        window,
        extra_bursts=((sec_top, "top"), (sec_bot, "bottom")),
        min_lines=10,
    )

    assert clipped is not None
    assert clipped.n_lines == window.n_lines - 4
    assert clipped.top_row0 == window.top_row0 + 4
    assert clipped.bottom_row0 == window.bottom_row0 + 4


def test_intersect_overlap_window_none_when_secondary_overlap_is_all_invalid() -> None:
    """A secondary burst with no valid overlap lines is omitted."""
    swath = _swath(_burst(0, start_line=0), _burst(1, start_line=80))
    window = burst_overlap_windows(swath)[0]
    invalid_top = range(window.top_row0, window.top_row0 + window.n_lines)
    sec_top = _burst(0, start_line=0, invalid=invalid_top)
    sec_bot = _burst(1, start_line=80)

    clipped = intersect_overlap_window(
        window,
        extra_bursts=((sec_top, "top"), (sec_bot, "bottom")),
        min_lines=10,
    )

    assert clipped is None
