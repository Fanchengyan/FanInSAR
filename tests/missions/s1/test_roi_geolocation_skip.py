"""Per-burst geolocation pre-filter for CDSE ROI canvas preparation.

The whole-strip geolocation bbox over-covers diagonal Sentinel-1 strips
(the ascending track drifts sideways along track), so an ROI outside every
burst can still overlap the strip bbox and be prepared as an all-zero
canvas. The pre-filter therefore tests the ROI envelope against each
burst's own geolocation bbox and only falls back to the whole-strip bbox
when the annotation carries no ``linesPerBurst``.
"""

from __future__ import annotations

import pytest

from faninsar.data.query import BoundingBox
from faninsar.missions.s1.acquisition import _roi_overlaps_geolocation


def _strip_annotation(lines_per_burst: int | None) -> str:
    """Return a minimal annotation for a west-drifting three-burst strip."""
    points: list[str] = []
    base_lon = (100.0, 99.0, 98.5)
    for line in (0, 5, 10, 15, 20, 25):
        base = base_lon[line // 10]
        points.extend(
            "<geolocationGridPoint>"
            f"<line>{line}</line><pixel>{pixel}</pixel>"
            f"<latitude>{38.3 - pixel * 0.02}</latitude>"
            f"<longitude>{base + pixel * 0.01}</longitude>"
            "</geolocationGridPoint>"
            for pixel in (0, 10, 20)
        )
    swath_timing = ""
    if lines_per_burst is not None:
        swath_timing = f"<linesPerBurst>{lines_per_burst}</linesPerBurst>"
    return (
        "<product>"
        f"<swathTiming>{swath_timing}</swathTiming>"
        f"<geolocationGrid>{''.join(points)}</geolocationGrid>"
        "</product>"
    )


def test_offswath_roi_rejected_when_only_whole_strip_bbox_overlaps() -> None:
    """A strip whose whole bbox covers the ROI but no burst does is skipped."""
    roi = BoundingBox(98.8, 38.0, 98.9, 38.1, crs="EPSG:4326")
    assert _roi_overlaps_geolocation(_strip_annotation(10), roi) is False


def test_intersecting_burst_is_accepted() -> None:
    """An ROI inside one burst's geolocation bbox still passes the filter."""
    roi = BoundingBox(99.05, 38.0, 99.15, 38.1, crs="EPSG:4326")
    assert _roi_overlaps_geolocation(_strip_annotation(10), roi) is True


def test_missing_lines_per_burst_falls_back_to_whole_grid_bbox() -> None:
    """Without burst lengths the conservative whole-grid bbox is kept."""
    roi = BoundingBox(98.8, 38.0, 98.9, 38.1, crs="EPSG:4326")
    assert _roi_overlaps_geolocation(_strip_annotation(None), roi) is True
