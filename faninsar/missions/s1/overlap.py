"""Sensing-time burst-overlap windows for Sentinel-1 TOPS swaths.

Overlap geometry is owned by the mission: adjacent bursts are paired from
annotation sensing times and valid-sample masks, never from blended stitch
row ranges that would destroy the TOPS carrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger
from faninsar.missions.s1.errors import reject_product

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.missions.s1.types import S1Burst, S1Swath

logger = setup_logger(__name__)

_DEFAULT_MIN_SAMPLES = 15


@dataclass(frozen=True, slots=True)
class BurstOverlapWindow:
    """Rectangular overlap between two adjacent bursts of one sub-swath.

    Attributes
    ----------
    swath : str
        Sub-swath name (``IW1``, ``IW2``, ...).
    top_index : int
        Burst index *i* (earlier burst in the pair).
    bottom_index : int
        Burst index *i + 1*.
    top_row0 : int
        First overlap line inside burst *i* (burst-local).
    bottom_row0 : int
        First overlap line inside burst *i + 1* (burst-local).
    col0 : int
        Start column of the valid-sample intersection (inclusive).
    n_lines : int
        Overlap height in azimuth lines.
    n_samples : int
        Overlap width in range samples.

    """

    swath: str
    top_index: int
    bottom_index: int
    top_row0: int
    bottom_row0: int
    col0: int
    n_lines: int
    n_samples: int


def burst_overlap_windows(
    swath: S1Swath,
    *,
    burst_indices: Sequence[int] | None = None,
    min_lines: int = 20,
    min_samples: int | None = None,
) -> tuple[BurstOverlapWindow, ...]:
    """Return sensing-time overlap windows for adjacent bursts of a swath.

    Each window is the intersection of (1) the two bursts' sensing-time
    coverage, converted through ``azimuth_time / azimuth_time_interval_s``,
    and (2) their ``first_valid_sample`` / ``last_valid_sample`` masks.
    Pairs that are not adjacent indices ``(i, i+1)``, have no sensing-time
    overlap, have an empty valid-sample intersection, or fall below
    ``min_lines`` / ``min_samples`` are omitted.  Physically adjacent
    bursts in different SAFE frames are not represented.

    Parameters
    ----------
    swath : S1Swath
        Parsed sub-swath whose bursts supply sensing times and valid samples.
    burst_indices : sequence of int, optional
        Burst indices to consider.  Defaults to every burst in ``swath``.
        Only selected pairs whose indices differ by one are eligible.
    min_lines : int, optional
        Minimum overlap height in azimuth lines.  Shorter overlaps are
        dropped.  Default is 20.
    min_samples : int, optional
        Minimum overlap width in range samples.  Defaults to 15 (the
        conventional range multilook).  An empty valid-sample intersection
        always drops the window.

    Returns
    -------
    tuple of BurstOverlapWindow
        One window per retained adjacent pair, in increasing top index.
        A swath with fewer than two selected bursts yields an empty tuple.

    Raises
    ------
    Sentinel1ProductError
        If ``burst_indices`` names an unknown burst or the azimuth time
        interval is not positive.
    ValueError
        If ``min_lines`` or ``min_samples`` is negative.

    Examples
    --------
    >>> windows = burst_overlap_windows(swath)
    >>> all(window.n_lines >= 20 for window in windows)
    True

    """
    sample_floor = _DEFAULT_MIN_SAMPLES if min_samples is None else min_samples
    _reject_negative_threshold("min_lines", min_lines)
    _reject_negative_threshold("min_samples", sample_floor)

    dt = float(swath.azimuth_time_interval_s)
    if dt <= 0.0:
        reject_product(
            f"swath {swath.swath} azimuth_time_interval_s must be positive, got {dt}"
        )

    by_index = {burst.index: burst for burst in swath.bursts}
    selected = _selected_indices(swath, burst_indices, known=by_index)
    windows: list[BurstOverlapWindow] = []
    for top_index in sorted(selected):
        bottom_index = top_index + 1
        if bottom_index not in selected:
            continue
        window = _adjacent_window(
            swath.swath,
            by_index[top_index],
            by_index[bottom_index],
            dt=dt,
            min_lines=min_lines,
            min_samples=sample_floor,
        )
        if window is not None:
            windows.append(window)
    return tuple(windows)


def _reject_negative_threshold(name: str, value: int) -> None:
    """Log and raise when a size threshold is negative."""
    if value < 0:
        message = f"{name} must be >= 0, got {value}"
        logger.error(message)
        raise ValueError(message)


def _selected_indices(
    swath: S1Swath,
    burst_indices: Sequence[int] | None,
    *,
    known: dict[int, S1Burst],
) -> set[int]:
    """Return the burst indices to pair, rejecting unknown values."""
    if burst_indices is None:
        return set(known)
    selected = set(burst_indices)
    unknown = selected - set(known)
    if unknown:
        reject_product(
            f"burst indices {sorted(unknown)} are not in swath {swath.swath}; "
            f"available={sorted(known)}"
        )
    return selected


def _adjacent_window(
    swath_name: str,
    top: S1Burst,
    bottom: S1Burst,
    *,
    dt: float,
    min_lines: int,
    min_samples: int,
) -> BurstOverlapWindow | None:
    """Build one window for bursts ``(i, i+1)``, or ``None`` if it is unusable."""
    offset = round((bottom.azimuth_time - top.azimuth_time).total_seconds() / dt)
    az0 = max(0, offset)
    az1 = min(top.lines, offset + bottom.lines)
    if az1 <= az0:
        logger.debug(
            "No sensing-time overlap for swath %s bursts %s/%s",
            swath_name,
            top.index,
            bottom.index,
        )
        return None

    span = _longest_valid_run(top, bottom, offset=offset, az0=az0, az1=az1)
    if span is None:
        logger.debug(
            "Valid-sample azimuth intersection is empty for swath %s bursts %s/%s",
            swath_name,
            top.index,
            bottom.index,
        )
        return None
    top_row0, top_row1 = span
    n_lines = top_row1 - top_row0 + 1
    bottom_row0 = top_row0 - offset
    if n_lines < min_lines:
        logger.debug(
            "Dropping swath %s overlap %s/%s: n_lines=%s < min_lines=%s",
            swath_name,
            top.index,
            bottom.index,
            n_lines,
            min_lines,
        )
        return None

    columns = _column_intersection(
        top,
        bottom,
        top_row0=top_row0,
        bottom_row0=bottom_row0,
        n_lines=n_lines,
    )
    if columns is None:
        logger.debug(
            "Valid-sample column intersection is empty for swath %s bursts %s/%s",
            swath_name,
            top.index,
            bottom.index,
        )
        return None
    col0, n_samples = columns
    if n_samples < min_samples:
        logger.debug(
            "Dropping swath %s overlap %s/%s: n_samples=%s < min_samples=%s",
            swath_name,
            top.index,
            bottom.index,
            n_samples,
            min_samples,
        )
        return None

    return BurstOverlapWindow(
        swath=swath_name,
        top_index=top.index,
        bottom_index=bottom.index,
        top_row0=top_row0,
        bottom_row0=bottom_row0,
        col0=col0,
        n_lines=n_lines,
        n_samples=n_samples,
    )


def _line_is_valid(burst: S1Burst, row: int) -> bool:
    """Return whether *row* has a non-negative first valid sample."""
    if row < 0 or row >= burst.lines or row >= len(burst.first_valid_sample):
        return False
    return burst.first_valid_sample[row] >= 0


def _longest_valid_run(
    top: S1Burst,
    bottom: S1Burst,
    *,
    offset: int,
    az0: int,
    az1: int,
) -> tuple[int, int] | None:
    """Return inclusive top-frame ``(start, end)`` of the longest valid run."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for row in range(az0, az1):
        if _line_is_valid(top, row) and _line_is_valid(bottom, row - offset):
            if run_start is None:
                run_start = row
            continue
        if run_start is not None:
            runs.append((run_start, row - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, az1 - 1))
    if not runs:
        return None
    return max(runs, key=lambda span: span[1] - span[0])


def _column_intersection(
    top: S1Burst,
    bottom: S1Burst,
    *,
    top_row0: int,
    bottom_row0: int,
    n_lines: int,
) -> tuple[int, int] | None:
    """Return ``(col0, n_samples)`` for the overlap rectangle, if non-empty."""
    firsts: list[int] = []
    lasts: list[int] = []
    for line in range(n_lines):
        top_row = top_row0 + line
        bottom_row = bottom_row0 + line
        firsts.append(top.first_valid_sample[top_row])
        firsts.append(bottom.first_valid_sample[bottom_row])
        lasts.append(top.last_valid_sample[top_row])
        lasts.append(bottom.last_valid_sample[bottom_row])
    col0 = max(firsts)
    col1 = min(lasts) + 1
    col0 = max(col0, 0)
    col1 = min(col1, top.samples, bottom.samples)
    n_samples = col1 - col0
    if n_samples <= 0:
        return None
    return col0, n_samples


__all__ = ["BurstOverlapWindow", "burst_overlap_windows"]
