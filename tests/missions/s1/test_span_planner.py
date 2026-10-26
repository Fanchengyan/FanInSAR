"""Tests for the mission-owned SLC range span planner (PROPOSAL-0061).

The planner is pure: no HTTP, no ``RangeFetcher``.  Fixtures invent literal
strip tables (TIFF ``StripOffsets`` / ``StripByteCounts`` / ``RowsPerStrip``)
and assert exact spans.
"""

from __future__ import annotations

import itertools

import pytest

from faninsar.missions.s1.acquisition import (
    MAX_SPAN_BYTES,
    MERGE_FLOOR_BYTES,
    plan_ifd_probe_spans,
    plan_measurement_spans,
)
from faninsar.missions.s1.errors import Sentinel1ProductError
from faninsar.remote.fetch import RangeSpan

URL = "https://example.test/S1A_IW_SLC/measurement.tiff"

MIB = 1024 * 1024

# 8 contiguous single-row strips of 4000 B whose first strip starts at 1000,
# so a naive ``line * width * 4`` mapping (which would start at 0) is
# distinguishable from table-driven mapping.
_OFFSETS_8 = [1000 + i * 4000 for i in range(8)]
_COUNTS_8 = [4000] * 8


def _tables_8x1000x4() -> tuple[list[int], list[int]]:
    """Return strip tables for 8 rows, 1000 samples, 4 bytes per sample."""
    return list(_OFFSETS_8), list(_COUNTS_8)


def _plan(
    strip_offsets: list[int],
    strip_byte_counts: list[int],
    *,
    rows_per_strip: int = 1,
    samples: int = 1000,
    bytes_per_sample: int = 4,
    line0: int = 0,
    line1: int = 8,
    col0: int = 0,
    col1: int = 1000,
    **kwargs: object,
) -> tuple[RangeSpan, ...]:
    """Call ``plan_measurement_spans`` with the shared URL and defaults."""
    return plan_measurement_spans(
        url=URL,
        strip_offsets=strip_offsets,
        strip_byte_counts=strip_byte_counts,
        rows_per_strip=rows_per_strip,
        samples=samples,
        bytes_per_sample=bytes_per_sample,
        line0=line0,
        line1=line1,
        col0=col0,
        col1=col1,
        **kwargs,  # type: ignore[arg-type]
    )


class TestWideColumn:
    """Wide columns fetch whole-row blocks from the strip tables."""

    def test_full_roi_is_one_contiguous_span(self) -> None:
        """A full-width ROI over contiguous strips coalesces to one span."""
        offsets, counts = _tables_8x1000x4()
        spans = _plan(offsets, counts)
        assert spans == (RangeSpan(URL, 1000, 1000 + 8 * 4000 - 1, "row-0-8"),)

    def test_partial_roi_maps_rows_through_strip_table(self) -> None:
        """Row bounds come from the strip tables, not ``line * width * 4``."""
        spans = _plan(list(_OFFSETS_8), list(_COUNTS_8), line0=3, line1=7)
        # Row 3 lives in strip 3 at offset 13000 (table-driven); a naive
        # line*width*4 planner would have produced 12000.
        assert spans == (RangeSpan(URL, 13000, 29000 - 1, "row-3-7"),)

    def test_adjacent_sub_floor_row_blocks_merge_into_one_span(self) -> None:
        """Adjacent sub-floor row blocks merge and the floor never splits."""
        # 8 strips of 4000 B and 70 strips of 4000 B must each coalesce into a
        # single span: adjacent ranges below the 256 KiB floor merge, and the
        # floor is never a split threshold for contiguous data.
        offsets, counts = _tables_8x1000x4()
        assert len(_plan(offsets, counts)) == 1

        stride = 4000
        offsets70 = [i * stride for i in range(70)]
        counts70 = [stride] * 70
        spans = _plan(offsets70, counts70, line1=70)
        assert spans == (RangeSpan(URL, 0, 70 * stride - 1, "row-0-70"),)
        assert spans[0].end - spans[0].start + 1 > MERGE_FLOOR_BYTES
        assert spans[0].end - spans[0].start + 1 < MAX_SPAN_BYTES

    def test_large_roi_splits_at_4mib_on_row_boundaries(self) -> None:
        """A 10 MiB ROI splits into spans capped at 4 MiB, row-aligned."""
        stride = 262144 * 4  # 1 MiB per row
        offsets = [k * 2 * stride for k in range(5)]  # RowsPerStrip = 2
        counts = [2 * stride] * 5
        spans = _plan(
            offsets,
            counts,
            rows_per_strip=2,
            samples=262144,
            line1=10,
            col1=262144,
        )
        assert spans == (
            RangeSpan(URL, 0, 4 * 262144 * 4 - 1, "row-0-4"),
            RangeSpan(URL, 4 * 262144 * 4, 8 * 262144 * 4 - 1, "row-4-8"),
            RangeSpan(URL, 8 * 262144 * 4, 10 * 262144 * 4 - 1, "row-8-10"),
        )
        for span in spans:
            assert span.end - span.start + 1 <= MAX_SPAN_BYTES
        for prev, nxt in itertools.pairwise(spans):
            assert nxt.start == prev.end + 1
        # Every span end lands on a row boundary (stride multiple from row 0).
        for span in spans:
            assert (span.end + 1) % stride == 0

    def test_single_row_larger_than_cap_is_emitted_whole(self) -> None:
        """One row wider than the cap is the only above-cap emission."""
        # A row bigger than 4 MiB cannot end a span on a row boundary and be
        # under the cap; it is emitted as one span and documented as such.
        stride = 2 * MIB * 4
        spans = _plan([0], [stride], samples=2 * MIB, line1=1, col1=2 * MIB)
        assert spans == (RangeSpan(URL, 0, stride - 1, "row-0-1"),)
        assert spans[0].end - spans[0].start + 1 > MAX_SPAN_BYTES


class TestNarrowColumn:
    """Narrow columns build tight per-row-group ranges meeting the floor."""

    def test_rows_group_until_requests_meet_floor(self) -> None:
        """Consecutive rows merge into requests of at least 256 KiB."""
        stride = 100000 * 4  # 400000 B rows
        offsets = [500 + i * stride for i in range(10)]
        counts = [stride] * 10
        # Column slice of 100..1100 -> 4000 B per row, far below the floor.
        spans = _plan(offsets, counts, samples=100000, line1=10, col0=100, col1=1100)
        assert spans == (
            RangeSpan(URL, 900, 404899, "row-0-2"),
            RangeSpan(URL, 800900, 1204899, "row-2-4"),
            RangeSpan(URL, 1600900, 2004899, "row-4-6"),
            RangeSpan(URL, 2400900, 2804899, "row-6-8"),
            RangeSpan(URL, 3200900, 3604899, "row-8-10"),
        )
        for span in spans:
            assert span.end - span.start + 1 >= MERGE_FLOOR_BYTES
            assert span.end - span.start + 1 <= MAX_SPAN_BYTES

    def test_final_leftover_group_may_stay_below_floor(self) -> None:
        """Only the final leftover group may stay below the 256 KiB floor."""
        stride = 100000 * 4
        offsets = [500 + i * stride for i in range(3)]
        counts = [stride] * 3
        spans = _plan(offsets, counts, samples=100000, line1=3, col0=100, col1=1100)
        assert spans == (
            RangeSpan(URL, 900, 404899, "row-0-2"),
            RangeSpan(URL, 800900, 804899, "row-2-3"),
        )
        for span in spans[:-1]:
            assert span.end - span.start + 1 >= MERGE_FLOOR_BYTES
        assert spans[-1].end - spans[-1].start + 1 < MERGE_FLOOR_BYTES

    def test_tiny_roi_single_sub_floor_request_is_allowed(self) -> None:
        """A ROI smaller than the floor yields one bounded sub-floor span."""
        offsets, counts = _tables_8x1000x4()
        spans = _plan(offsets, counts, line0=0, line1=2, col0=0, col1=10)
        assert spans == (RangeSpan(URL, 1000, 1000 + 4000 + 40 - 1, "row-0-2"),)


class TestFailClosed:
    """Compressed, tiled, or inconsistent geometry is rejected."""

    def test_compressed_is_rejected(self) -> None:
        """A compressed TIFF has no fixed row stride and must be rejected."""
        offsets, counts = _tables_8x1000x4()
        with pytest.raises(Sentinel1ProductError):
            _plan(offsets, counts, compressed=True)

    def test_tiled_is_rejected(self) -> None:
        """A tiled TIFF has no row-aligned strips and must be rejected."""
        offsets, counts = _tables_8x1000x4()
        with pytest.raises(Sentinel1ProductError):
            _plan(offsets, counts, tiled=True)

    @pytest.mark.parametrize(
        ("kwargs", "tables"),
        [
            # Table length mismatch.
            ({}, ([0, 4000, 8000], [4000, 4000])),
            # Empty tables.
            ({}, ([], [])),
            # RowsPerStrip must be positive.
            ({"rows_per_strip": 0}, ([0], [4000])),
            # Samples and bytes per sample must be positive.
            ({"samples": 0}, ([0], [4000])),
            ({"bytes_per_sample": 0}, ([0], [4000])),
            # Column bounds must be inside the row.
            ({"col1": 1001}, ([0], [4000])),
            ({"col0": -1}, ([0], [4000])),
            # ROI rows must exist in the tables (8 rows known).
            ({"line1": 9}, (_OFFSETS_8, _COUNTS_8)),
            ({"line0": 3, "line1": 3}, ([0], [4000])),
            # Byte counts must be whole rows.
            (
                {},
                (
                    [0, 4001, 8000, 12000, 16000, 20000, 24000, 28000],
                    [4000, 4001, 4000, 4000, 4000, 4000, 4000, 4000],
                ),
            ),
            # Every strip except the last must be full.
            (
                {"rows_per_strip": 2},
                ([0, 8000, 16000, 24000], [4000, 8000, 8000, 8000]),
            ),
            # Last strip cannot exceed RowsPerStrip rows.
            (
                {"rows_per_strip": 2},
                ([0, 8000, 16000, 24000], [8000, 8000, 8000, 12000]),
            ),
            # Negative strip offset.
            ({}, ([-4000, *_OFFSETS_8[1:]], _COUNTS_8)),
        ],
    )
    def test_invalid_geometry_is_rejected(
        self,
        kwargs: dict[str, object],
        tables: tuple[list[int], list[int]],
    ) -> None:
        """Each inconsistent geometry variant raises a mission error."""
        offsets, counts = tables
        with pytest.raises(Sentinel1ProductError):
            _plan(offsets, counts, **kwargs)  # type: ignore[arg-type]


class TestIfdProbeSpans:
    """Bounded header/IFD/strip-table probes; never a 0..EOF span."""

    def test_header_and_ifd_only(self) -> None:
        """Without strip tables, only the header and IFD spans are emitted."""
        spans = plan_ifd_probe_spans(URL, ifd_offset=900_000, ifd_length=16_384)
        assert spans == (
            RangeSpan(URL, 0, 15, "tiff-header"),
            RangeSpan(URL, 900_000, 916_383, "ifd-900000"),
        )

    def test_strip_tables_outside_ifd_add_third_span(self) -> None:
        """Strip tables outside the IFD produce a third bounded span."""
        spans = plan_ifd_probe_spans(
            URL,
            ifd_offset=900_000,
            ifd_length=16_384,
            strip_table_offset=920_000,
            strip_table_length=1024,
        )
        assert spans == (
            RangeSpan(URL, 0, 15, "tiff-header"),
            RangeSpan(URL, 900_000, 916_383, "ifd-900000"),
            RangeSpan(URL, 920_000, 921_023, "strip-tables-920000"),
        )

    def test_strip_tables_inside_ifd_are_omitted(self) -> None:
        """Strip tables fully inside the IFD need no separate span."""
        spans = plan_ifd_probe_spans(
            URL,
            ifd_offset=900_000,
            ifd_length=16_384,
            strip_table_offset=905_000,
            strip_table_length=512,
        )
        assert spans == (
            RangeSpan(URL, 0, 15, "tiff-header"),
            RangeSpan(URL, 900_000, 916_383, "ifd-900000"),
        )

    def test_never_emits_zero_to_eof(self) -> None:
        """The offset-0 span is a small bounded header, never 0..EOF."""
        for spans in (
            plan_ifd_probe_spans(URL, ifd_offset=900_000, ifd_length=16_384),
            plan_ifd_probe_spans(
                URL,
                ifd_offset=900_000,
                ifd_length=16_384,
                strip_table_offset=920_000,
                strip_table_length=1024,
            ),
        ):
            # The header span starts at 0 but is small and bounded.
            assert not any(span.start == 0 and span.end > 15 for span in spans)
            assert all(span.end >= span.start for span in spans)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ifd_offset": 0, "ifd_length": 16_384},
            {"ifd_offset": 8, "ifd_length": 16_384},
            {"ifd_offset": -1, "ifd_length": 16_384},
            {"ifd_offset": 900_000, "ifd_length": 0},
            {"ifd_offset": 900_000, "ifd_length": -16},
            {
                "ifd_offset": 900_000,
                "ifd_length": 16_384,
                "strip_table_offset": 920_000,
            },
            {
                "ifd_offset": 900_000,
                "ifd_length": 16_384,
                "strip_table_length": 1024,
            },
            {
                "ifd_offset": 900_000,
                "ifd_length": 16_384,
                "strip_table_offset": 920_000,
                "strip_table_length": 0,
            },
        ],
    )
    def test_invalid_probe_geometry_is_rejected(self, kwargs: dict[str, int]) -> None:
        """Each unusable probe offset/length raises a mission error."""
        with pytest.raises(Sentinel1ProductError):
            plan_ifd_probe_spans(URL, **kwargs)
