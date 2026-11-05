"""Tests for per-polygon ROI window planning (multi-polygon CDSE acquisition)."""

from __future__ import annotations

import itertools
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from faninsar.data.query.polygons import Polygons
from faninsar.missions.s1.acquisition import (
    _azimuth_to_physical_lines,
    _burst_azimuth_origin_lines,
    _geo2rdr_grid_size,
    _merge_burst_windows,
    _merge_line_segments,
    _polygon_envelopes,
    _select_orbit_group,
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


def _fake_swath_overlapping(
    n_bursts: int = 4,
    lpb: int = 100,
    spb: int = 200,
    step_lines: int = 80,
) -> Any:
    """Build a swath whose bursts overlap in azimuth.

    ``step_lines < lpb`` mimics real Sentinel-1 TOPS spacing: each burst's
    ``azimuth_time`` advances by ``step_lines * ati`` so consecutive bursts
    share ``lpb - step_lines`` azimuth lines.
    """
    from datetime import UTC, datetime, timedelta

    swath = MagicMock()
    swath.lines_per_burst = lpb
    swath.samples_per_burst = spb
    swath.lines = n_bursts * lpb
    swath.samples = spb
    ati = 0.002
    swath.azimuth_time_interval_s = ati
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    bursts = []
    for i in range(n_bursts):
        b = MagicMock()
        b.index = i
        b.azimuth_time = t0 + timedelta(seconds=i * step_lines * ati)
        bursts.append(b)
    swath.bursts = bursts
    return swath


def _fake_strip_tables(
    url: str,
    *,
    file_size: int,
    samples: int,
    fetch_context: Any = None,
) -> tuple[list[int], list[int], int]:
    """Fake IFD strip tables so span planning does not hit the network."""
    del url, file_size, fetch_context
    rows_per_strip = 1
    offsets = [i * samples * 4 for i in range(200)]
    counts = [samples * 4] * 200
    return offsets, counts, rows_per_strip


def _fake_bodies(spans: Any, fetch_context: Any = None) -> dict[str, bytes]:
    """Return fake range bodies for a batch of planned spans.

    Stands in for ``_fetch_deduped``, the batch fetch boundary.  Each span
    is a ``RangeSpan`` whose label is ``row-{line0}-{line1}``; the body is
    sized from the span's byte length (matching the planner's convention of
    4 bytes per sample per row).
    """
    del fetch_context
    out: dict[str, bytes] = {}
    for span in spans:
        out[span.label] = b"\x00" * (span.end - span.start + 1)
    return out


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
        with (
            patch(
                "faninsar.missions.s1.acquisition._strip_tables_for",
                side_effect=_fake_strip_tables,
            ),
            patch(
                "faninsar.missions.s1.acquisition._fetch_deduped",
                side_effect=_fake_bodies,
            ) as mock_fetch,
        ):
            windows = _windows_for_polygons(swath, envelopes, url="test", file_size=0)
        # All burst spans are planned first and fetched in one parallel batch
        assert mock_fetch.call_count == 1
        spans = mock_fetch.call_args[0][0]
        assert any(s.label == "row-10-40" for s in spans)
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
        with (
            patch(
                "faninsar.missions.s1.acquisition._strip_tables_for",
                side_effect=_fake_strip_tables,
            ),
            patch(
                "faninsar.missions.s1.acquisition._fetch_deduped",
                side_effect=_fake_bodies,
            ) as mock_fetch,
        ):
            windows = _windows_for_polygons(swath, envelopes, url="test", file_size=0)
        # Merged lines (10,40) fetched once in the batch; one window
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
        with (
            patch(
                "faninsar.missions.s1.acquisition._strip_tables_for",
                side_effect=_fake_strip_tables,
            ),
            patch(
                "faninsar.missions.s1.acquisition._fetch_deduped",
                side_effect=_fake_bodies,
            ) as mock_fetch,
        ):
            windows = _windows_for_polygons(swath, envelopes, url="test", file_size=0)
        # Two bursts' spans batched into ONE parallel fetch
        assert mock_fetch.call_count == 1
        spans = mock_fetch.call_args[0][0]
        assert len(spans) == 2  # one span per burst
        assert len(windows) == 2
        assert windows[0].burst_index == 0
        assert windows[1].burst_index == 1

    def test_polygon_column_crop(self) -> None:
        """A single envelope crops columns from the fetched rows."""
        swath = _fake_swath(n_bursts=1, lpb=100, spb=200)
        envelopes = [(10, 20, 50, 100)]
        with (
            patch(
                "faninsar.missions.s1.acquisition._strip_tables_for",
                side_effect=_fake_strip_tables,
            ),
            patch(
                "faninsar.missions.s1.acquisition._fetch_deduped",
                side_effect=_fake_bodies,
            ),
        ):
            windows = _windows_for_polygons(swath, envelopes, url="test", file_size=0)
        assert len(windows) == 1
        w = windows[0]
        assert w.col0 == 50
        assert w.samples.shape == (10, 50)  # n_lines x n_cols

    def test_disjoint_rows_do_not_fetch_the_gap(self) -> None:
        """Disjoint ROI rows keep separate requests but one masked burst canvas."""
        swath = _fake_swath(n_bursts=1, lpb=100, spb=200)
        envelopes = [(10, 20, 50, 80), (30, 40, 100, 150)]
        with (
            patch(
                "faninsar.missions.s1.acquisition._strip_tables_for",
                side_effect=_fake_strip_tables,
            ),
            patch(
                "faninsar.missions.s1.acquisition._fetch_deduped",
                side_effect=_fake_bodies,
            ) as mock_fetch,
        ):
            windows = _windows_for_polygons(swath, envelopes, url="test", file_size=0)
        spans = mock_fetch.call_args[0][0]
        assert [span.label for span in spans] == ["row-10-20", "row-30-40"]
        assert len(windows) == 1
        window = windows[0]
        assert (window.row0, window.col0, window.samples.shape) == (10, 50, (30, 100))
        assert window.line_valid is not None
        assert window.line_valid[10:20] == ((50, 79),) * 10
        assert window.line_valid[20:30] == ((-1, -1),) * 10
        assert window.line_valid[30:40] == ((100, 149),) * 10

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


class TestSpatialWindowPlans:
    """Spatial plans are payload-free and assemble one burst at a time."""

    def test_public_planning_keeps_physical_segments_without_payload_io(self) -> None:
        from faninsar.missions.s1 import acquisition

        swath = _fake_swath(n_bursts=2, lpb=100, spb=200)
        envelopes = [(110, 120, 50, 80), (130, 140, 100, 150)]
        with (
            patch.object(acquisition, "parse_annotation_xml", return_value=swath),
            patch.object(acquisition, "_roi_envelopes", return_value=envelopes),
            patch.object(acquisition, "_strip_tables_for") as probe,
            patch.object(acquisition, "_fetch_deduped") as fetch,
        ):
            parent, plans, xml = acquisition._plan_cdse_windows(
                {"measurement_url": "test"},
                MagicMock(),
                xml_text="annotation",
                dem=MagicMock(),
            )
        assert parent is swath and xml == "annotation"
        assert [(plan.parent_line0, plan.parent_line1) for plan in plans] == [
            (110, 120),
            (130, 140),
        ]
        assert [(plan.line0, plan.col0, plan.n_lines, plan.n_samples) for plan in plans] == [
            (10, 50, 10, 30),
            (30, 100, 10, 50),
        ]
        probe.assert_not_called()
        fetch.assert_not_called()

    def test_assembly_accepts_one_burst_subset_and_keeps_gap_invalid(self) -> None:
        from faninsar.missions.s1 import acquisition

        swath = _fake_swath(n_bursts=2, lpb=100, spb=200)
        plans = acquisition._plan_windows_for_polygons(
            swath,
            [(10, 20, 5, 10), (30, 40, 15, 20), (110, 120, 50, 80)],
        )
        pairs = np.zeros((10, 200, 2), dtype="<i2")
        pairs[..., 0] = np.arange(200)
        pairs[..., 1] = 3
        bodies = {"row-10-20": pairs.tobytes(), "row-30-40": pairs.tobytes()}
        windows = acquisition._assemble_planned_windows(
            [plan for plan in plans if plan.burst_index == 0],
            bodies,
            samples_per_row=200,
        )
        assert len(windows) == 1
        window = windows[0]
        assert (window.burst_index, window.row0, window.col0) == (0, 10, 5)
        assert window.samples.shape == (30, 15)
        np.testing.assert_array_equal(window.samples[:10, :5], pairs[..., 0][:, 5:10] + 3j)
        np.testing.assert_array_equal(window.samples[20:, 10:], pairs[..., 0][:, 15:20] + 3j)
        np.testing.assert_array_equal(window.samples[10:20], 0)
        assert window.line_valid is not None
        assert window.line_valid[20:30] == ((-1, -1),) * 10


# ---------------------------------------------------------------------------
# azimuth-line -> physical-TIFF-row mapping (burst azimuth overlap)
# ---------------------------------------------------------------------------


class TestAzimuthToPhysicalLines:
    """Geo2rdr azimuth rows map to physical TIFF rows under burst overlap."""

    def test_origin_lines_use_real_azimuth_spacing(self) -> None:
        """Burst origins track azimuth_time, not index*lpb."""
        swath = _fake_swath_overlapping(n_bursts=4, lpb=100, step_lines=80)
        origins = _burst_azimuth_origin_lines(swath)
        np.testing.assert_allclose(origins, [0.0, 80.0, 160.0, 240.0])

    def test_inner_burst_rows_map_to_that_burst_slot(self) -> None:
        """An az line deep inside burst k lands at k*lpb+local."""
        swath = _fake_swath_overlapping(n_bursts=4, lpb=100, step_lines=80)
        origins = _burst_azimuth_origin_lines(swath)
        # az=200 sits in burst2 (origin 160, local 40).  burst1 covers
        # az in [80,180); burst2 [160,260).  Only burst2 claims az=200.
        mapped = _azimuth_to_physical_lines(swath, np.array([200.0]), origins)
        finite = mapped[np.isfinite(mapped)]
        np.testing.assert_allclose(finite, [2 * 100 + 40.0])

    def test_overlap_row_covers_both_burst_copies(self) -> None:
        """An az in the k/k+1 overlap maps to *both* physical rows."""
        swath = _fake_swath_overlapping(n_bursts=4, lpb=100, step_lines=80)
        origins = _burst_azimuth_origin_lines(swath)
        # az=170: burst1 local 90 (physical 190) and burst2 local 10
        # (physical 210).  Both burst copies are returned so the envelope
        # encloses the polygon's echo in every burst it intersects.
        mapped = _azimuth_to_physical_lines(swath, np.array([170.0]), origins)
        finite = np.sort(mapped[np.isfinite(mapped)])
        np.testing.assert_allclose(finite, [100.0 + 90.0, 2 * 100.0 + 10.0])

    def test_azimuth_beyond_swath_is_nan(self) -> None:
        """An az outside every burst azimuth window maps to NaN."""
        swath = _fake_swath_overlapping(n_bursts=2, lpb=100, step_lines=80)
        origins = _burst_azimuth_origin_lines(swath)
        mapped = _azimuth_to_physical_lines(swath, np.array([999.0]), origins)
        assert not np.isfinite(mapped).any()


# ---------------------------------------------------------------------------
# _select_orbit_group — orbit-group filtering of discovered bindings
# ---------------------------------------------------------------------------


class TestSelectOrbitGroup:
    """``materialize_cdse_roi_safes`` keeps items in one orbit group."""

    @staticmethod
    def _item(
        mission: str,
        rel_orbit: int,
        direction: str,
        product_suffix: str = "",
    ) -> dict[str, Any]:
        return {
            "product_id": f"S1{mission}_{product_suffix or 'X'}",
            "mission_id": mission,
            "relative_orbit": rel_orbit,
            "orbit_direction": direction,
        }

    def test_passthrough_when_group_is_none(self) -> None:
        items = [
            self._item("C", 88, "ASCENDING"),
            self._item("D", 15, "ASCENDING"),
        ]
        out = _select_orbit_group(items, None)
        assert out == items

    def test_passthrough_when_single_group(self) -> None:
        items = [
            self._item("C", 88, "ASCENDING", "a"),
            self._item("C", 88, "ASCENDING", "b"),
        ]
        out = _select_orbit_group(items, "auto")
        assert out == items

    def test_auto_picks_largest_group(self) -> None:
        # Orbit groups are mission-agnostic: S1C and S1D sharing orbit 88
        # form one track group, so "auto" keeps all three orbit-88 items and
        # drops only the orbit-15 straggler.
        items = [
            self._item("C", 88, "ASCENDING", "a"),
            self._item("D", 88, "ASCENDING", "b"),
            self._item("C", 88, "ASCENDING", "c"),
            self._item("C", 15, "ASCENDING", "d"),
        ]
        out = _select_orbit_group(items, "auto")
        assert len(out) == 3
        assert all(it["relative_orbit"] == 88 for it in out)
        assert {it["mission_id"] for it in out} == {"C", "D"}

    def test_require_raises_on_mixed_groups(self) -> None:
        items = [
            self._item("C", 88, "ASCENDING"),
            self._item("D", 15, "ASCENDING"),
        ]
        with pytest.raises(Exception, match="multiple orbit groups"):
            _select_orbit_group(items, "require")

    def test_explicit_selector_matches(self) -> None:
        # A mission-prefixed selector keeps the whole orbit-88 track,
        # satellites included — the mission token is ignored on grouping.
        items = [
            self._item("C", 88, "ASCENDING", "a"),
            self._item("D", 15, "ASCENDING", "b"),
            self._item("C", 88, "ASCENDING", "c"),
        ]
        out = _select_orbit_group(items, "S1C/88/ASCENDING")
        assert len(out) == 2
        assert all(it["relative_orbit"] == 88 for it in out)

    def test_explicit_selector_bare_form(self) -> None:
        # The canonical "<REL>/<DIR>" form selects the same group.
        items = [
            self._item("C", 88, "ASCENDING", "a"),
            self._item("D", 15, "ASCENDING", "b"),
        ]
        out = _select_orbit_group(items, "15/ASCENDING")
        assert [it["relative_orbit"] for it in out] == [15]

    def test_explicit_selector_no_match(self) -> None:
        items = [
            self._item("C", 88, "ASCENDING"),
            self._item("D", 15, "ASCENDING"),
        ]
        with pytest.raises(Exception, match="matches no item"):
            _select_orbit_group(items, "15/DESCENDING")

    def test_direction_distinguishes_groups(self) -> None:
        items = [
            self._item("C", 88, "ASCENDING", "a"),
            self._item("C", 88, "DESCENDING", "b"),
        ]
        out = _select_orbit_group(items, "auto")
        assert len(out) == 1
        assert out[0]["orbit_direction"] in {"ASCENDING", "DESCENDING"}
