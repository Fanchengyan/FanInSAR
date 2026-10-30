"""Tests for per-polygon ROI window planning (multi-polygon CDSE acquisition)."""

from __future__ import annotations

import itertools
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from faninsar.data.query.polygons import Polygons
from faninsar.missions.s1.acquisition import (
    _geo2rdr_grid_size,
    _merge_burst_windows,
    _merge_line_segments,
    _polygon_envelopes,
    _windows_for_polygons,
)

# ---------------------------------------------------------------------------
# _geo2rdr_grid_size
# ---------------------------------------------------------------------------


class TestGeo2rdrGridSize:
    """Adaptive geo2rdr sampling grid scales with polygon size."""

    def test_tiny_polygon_gets_4x4(self) -> None:
        """Polygons under 0.1 km use the 4x4 floor."""
        assert _geo2rdr_grid_size(0.001, 0.001) == 4

    def test_small_polygon_gets_4x4_to_6x6(self) -> None:
        """~0.5 km at 46 deg N falls in the 5x5/6x6 band."""
        n = _geo2rdr_grid_size(0.006, 0.005)
        assert 4 <= n <= 6

    def test_medium_polygon_scales_up(self) -> None:
        """~2 km polygon scales past the 4x4 floor."""
        n = _geo2rdr_grid_size(0.026, 0.018)
        assert 4 < n <= 10

    def test_large_polygon_gets_10x10(self) -> None:
        """Polygons at/above 20 km use the 10x10 cap."""
        assert _geo2rdr_grid_size(0.3, 0.2) == 10

    def test_huge_polygon_caps_at_10x10(self) -> None:
        """Very large polygons stay at the 10x10 cap."""
        assert _geo2rdr_grid_size(1.0, 1.0) == 10

    def test_monotonic_increase(self) -> None:
        """Grid size never shrinks as polygon size grows."""
        sizes = [
            _geo2rdr_grid_size(d, d) for d in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        ]
        assert all(a <= b for a, b in itertools.pairwise(sizes))


# ---------------------------------------------------------------------------
# _merge_line_segments
# ---------------------------------------------------------------------------


class TestMergeLineSegments:
    """Overlapping/adjacent line segments merge into disjoint intervals."""

    def test_empty(self) -> None:
        """Empty input yields no segments."""
        assert _merge_line_segments([]) == []

    def test_single(self) -> None:
        """One segment returns unchanged."""
        assert _merge_line_segments([(10, 20)]) == [(10, 20)]

    def test_non_overlapping(self) -> None:
        """Disjoint segments are preserved."""
        assert _merge_line_segments([(10, 20), (30, 40)]) == [(10, 20), (30, 40)]

    def test_overlapping(self) -> None:
        """Overlapping segments merge to the union."""
        assert _merge_line_segments([(10, 25), (20, 40)]) == [(10, 40)]

    def test_nested(self) -> None:
        """A contained segment merges into its parent."""
        assert _merge_line_segments([(10, 50), (20, 30)]) == [(10, 50)]

    def test_adjacent(self) -> None:
        """Abutting segments (end == next start) merge."""
        assert _merge_line_segments([(10, 20), (20, 30)]) == [(10, 30)]

    def test_unsorted_input(self) -> None:
        """Unsorted input is sorted before merging."""
        assert _merge_line_segments([(30, 40), (10, 20)]) == [(10, 20), (30, 40)]

    def test_many_merges(self) -> None:
        """Multiple clusters each merge to their own interval."""
        segs = [(100, 200), (150, 250), (300, 400), (350, 500), (600, 700)]
        assert _merge_line_segments(segs) == [(100, 250), (300, 500), (600, 700)]


# ---------------------------------------------------------------------------
# _polygon_envelopes  (requires a real S1Swath + DEM; skip if unavailable)
# ---------------------------------------------------------------------------


def _qinghai_safe() -> Any:
    """Return path to the Qinghai test SAFE if it exists."""
    from pathlib import Path

    safe = Path(
        "/Volumes/DATA2/FanInSAR/test_full/roi/inputs/"
        "S1D_IW_SLC__1SDV_20260724T232625_20260724T232655_003822_006E1D_F14C-IW2-VV.SAFE"
    )
    return safe if safe.exists() else None


def _qinghai_dem() -> Any:
    """Return path to the Qinghai stack DEM if it exists."""
    from pathlib import Path

    dem = Path("/Volumes/DATA2/FanInSAR/test_full/roi/dem/stack.tif")
    return dem if dem.exists() else None


@pytest.mark.skipif(
    _qinghai_safe() is None or _qinghai_dem() is None,
    reason="Qinghai test SAFE/DEM not available",
)
class TestPolygonEnvelopes:
    """geo2rdr inversion of desired polygons to radar envelopes."""

    def test_synthetic_polygons_invert(self) -> None:
        """Two small ROI polygons invert to two radar envelopes."""
        import geopandas as gpd
        from shapely.geometry import box

        from faninsar.missions.s1.annotation import load_annotation_file
        from faninsar.processing.geometry.dem import RasterDEM

        safe = _qinghai_safe()
        ann = next(safe.glob("annotation/*.xml"))
        swath = load_annotation_file(ann, measurement_path="x")
        dem = RasterDEM(path=_qinghai_dem())

        # Two small polygons inside the ROI footprint
        gdf = gpd.GeoDataFrame(
            geometry=[
                box(98.86, 38.79, 98.87, 38.80),
                box(98.89, 38.80, 98.91, 38.82),
            ],
            crs=4326,
        )
        polys = Polygons(gdf, types="desired")
        envelopes = _polygon_envelopes(polys, swath, dem)
        assert len(envelopes) == 2
        for line0, line1, col0, col1 in envelopes:
            assert 0 <= line0 < line1 <= swath.lines
            assert 0 <= col0 < col1 <= swath.samples_per_burst

    def test_outside_polygon_skipped(self) -> None:
        """A polygon outside the swath produces no envelope."""
        import geopandas as gpd
        from shapely.geometry import box

        from faninsar.missions.s1.annotation import load_annotation_file
        from faninsar.processing.geometry.dem import RasterDEM

        safe = _qinghai_safe()
        ann = next(safe.glob("annotation/*.xml"))
        swath = load_annotation_file(ann, measurement_path="x")
        dem = RasterDEM(path=_qinghai_dem())

        # Polygon far outside the swath (Alps)
        gdf = gpd.GeoDataFrame(
            geometry=[box(7.2, 46.0, 7.3, 46.1)],
            crs=4326,
        )
        polys = Polygons(gdf, types="desired")
        envelopes = _polygon_envelopes(polys, swath, dem)
        assert envelopes == []


# ---------------------------------------------------------------------------
# _windows_for_polygons
# ---------------------------------------------------------------------------


def _fake_swath(n_bursts: int = 10, lpb: int = 1507, spb: int = 25266) -> Any:
    """Build a minimal S1Swath mock."""
    swath = MagicMock()
    swath.lines_per_burst = lpb
    swath.samples_per_burst = spb
    swath.lines = n_bursts * lpb
    swath.samples = spb
    bursts = []
    for i in range(n_bursts):
        b = MagicMock()
        b.index = i
        bursts.append(b)
    swath.bursts = bursts
    return swath


def _fake_bodies(
    url: str,
    line0: int,
    line1: int,
    col0: int,
    col1: int,
    samples: int,
    **kwargs: Any,
) -> dict[str, bytes]:
    """Return fake range bodies for a (line0, line1) whole-row fetch."""
    del url, col0, col1, kwargs
    n_lines = line1 - line0
    n_bytes = n_lines * samples * 4
    # Emit one span covering all requested rows
    return {f"row-{line0}-{line1}": b"\x00" * n_bytes}


class TestMergeBurstWindows:
    """Envelopes on one burst merge into a single union window."""

    def test_empty(self) -> None:
        """Empty input yields no window."""
        assert _merge_burst_windows([]) == []

    def test_single(self) -> None:
        """One envelope returns unchanged."""
        assert _merge_burst_windows([(10, 20, 50, 100)]) == [(10, 20, 50, 100)]

    def test_merges_all_to_union(self) -> None:
        """Adjacent envelopes merge to their union."""
        envs = [(10, 20, 50, 100), (30, 40, 50, 100)]
        assert _merge_burst_windows(envs) == [(10, 40, 50, 100)]

    def test_disjoint_cols_merge(self) -> None:
        """Disjoint column ranges on the same rows merge to a union window."""
        envs = [(10, 20, 50, 80), (10, 20, 120, 150)]
        assert _merge_burst_windows(envs) == [(10, 20, 50, 150)]

    def test_overlapping_both(self) -> None:
        """Envelopes overlapping in both axes merge."""
        envs = [(10, 30, 50, 120), (20, 40, 80, 150)]
        assert _merge_burst_windows(envs) == [(10, 40, 50, 150)]

    def test_nested(self) -> None:
        """A contained envelope merges into its parent."""
        envs = [(10, 50, 0, 200), (20, 30, 50, 100)]
        assert _merge_burst_windows(envs) == [(10, 50, 0, 200)]


class TestWindowsForPolygons:
    """Per-polygon envelopes fetch and assemble into burst windows."""

    def test_two_polygons_same_burst_overlapping_cols(self) -> None:
        """Overlapping envelopes on one burst fetch once and merge."""
        swath = _fake_swath(n_bursts=3, lpb=100, spb=200)
        envelopes = [
            (10, 30, 50, 100),  # lines 10-30, cols 50-100
            (20, 40, 80, 150),  # lines 20-40, cols 80-150 (overlapping both)
        ]
        with patch(
            "faninsar.missions.s1.acquisition._fetch_measurement_bodies",
            side_effect=_fake_bodies,
        ) as mock_fetch:
            windows = _windows_for_polygons(swath, envelopes, url="test", file_size=0)
        # Should have fetched merged lines (10,40) once, not twice
        assert mock_fetch.call_count == 1
        call_args = mock_fetch.call_args
        assert call_args[1]["line0"] == 10
        assert call_args[1]["line1"] == 40
        # Overlapping cols on same burst merge into one window
        assert len(windows) == 1
        w = windows[0]
        assert w.burst_index == 0
        assert w.col0 == 50
        assert w.samples.shape == (30, 100)  # union lines 10-40, union cols

    def test_two_polygons_same_burst_disjoint_cols(self) -> None:
        """Line-overlapping/col-disjoint envelopes on one burst merge."""
        swath = _fake_swath(n_bursts=3, lpb=100, spb=200)
        envelopes = [
            (10, 30, 50, 80),  # lines 10-30, cols 50-80
            (20, 40, 100, 150),  # lines 20-40, cols 100-150
        ]
        with patch(
            "faninsar.missions.s1.acquisition._fetch_measurement_bodies",
            side_effect=_fake_bodies,
        ) as mock_fetch:
            windows = _windows_for_polygons(swath, envelopes, url="test", file_size=0)
        # Merged lines (10,40) fetched once; same burst merges to one window
        assert mock_fetch.call_count == 1
        assert len(windows) == 1
        w = windows[0]
        assert w.burst_index == 0
        assert w.col0 == 50
        assert w.samples.shape == (30, 100)  # union lines, union cols

    def test_two_polygons_different_bursts(self) -> None:
        """Envelopes on different bursts fetch separately and stay apart."""
        swath = _fake_swath(n_bursts=3, lpb=100, spb=200)
        envelopes = [
            (10, 30, 50, 100),  # burst 0
            (110, 130, 60, 120),  # burst 1
        ]
        with patch(
            "faninsar.missions.s1.acquisition._fetch_measurement_bodies",
            side_effect=_fake_bodies,
        ) as mock_fetch:
            windows = _windows_for_polygons(swath, envelopes, url="test", file_size=0)
        # Two separate line ranges -> two fetches
        assert mock_fetch.call_count == 2
        assert len(windows) == 2
        assert windows[0].burst_index == 0
        assert windows[1].burst_index == 1

    def test_polygon_column_crop(self) -> None:
        """A single envelope crops columns from the fetched rows."""
        swath = _fake_swath(n_bursts=1, lpb=100, spb=200)
        envelopes = [(10, 20, 50, 100)]
        with patch(
            "faninsar.missions.s1.acquisition._fetch_measurement_bodies",
            side_effect=_fake_bodies,
        ):
            windows = _windows_for_polygons(swath, envelopes, url="test", file_size=0)
        assert len(windows) == 1
        w = windows[0]
        assert w.col0 == 50
        assert w.samples.shape == (10, 50)  # n_lines x n_cols

    def test_empty_envelopes(self) -> None:
        """Empty envelopes yield no windows."""
        swath = _fake_swath()
        assert _windows_for_polygons(swath, [], url="test", file_size=0) == []

    def test_no_intersecting_bursts(self) -> None:
        """Envelopes intersecting no burst raise a product error."""
        swath = _fake_swath(n_bursts=2, lpb=100, spb=200)
        envelopes = [(500, 600, 0, 100)]  # beyond burst 1
        with pytest.raises(Exception, match="no polygon envelope"):
            _windows_for_polygons(swath, envelopes, url="test", file_size=0)
