"""Mission-owned row-to-byte span planning for Sentinel-1 SLC range transfer.

PROPOSAL-0061 splits the concern: :mod:`faninsar.remote` transports
already-sized :class:`~faninsar.remote.fetch.RangeSpan` objects and knows
nothing about TIFF geometry, while the mission layer owns the row-to-byte
mapping and emits the spans.  This module is that owner.  It is pure: no
HTTP, no aiohttp, no ``RangeFetcher`` calls.

The planner reads ``StripOffsets`` / ``StripByteCounts`` / ``RowsPerStrip``
(probe-planned by :func:`plan_ifd_probe_spans`) and never assumes
``line * width * 4`` from byte 0, which is only valid for a contiguous
single-strip layout.  The ZIP-local fast path in
:mod:`faninsar.missions.s1.io.extract` is unrelated to this module.

Contract (PROPOSAL-0061):

- fail closed on compressed or tiled TIFFs;
- never emit a ``0..EOF`` span: every span is bounded by strip-table
  arithmetic or explicit probe lengths;
- narrow columns fetch tight per-row-group ranges, each request at least
  :data:`MERGE_FLOOR_BYTES` (256 KiB) unless the remaining ROI is smaller;
- wide columns fetch whole-row blocks;
- adjacent ranges below the 256 KiB floor merge;
- any contiguous range is split into spans of at most
  :data:`MAX_SPAN_BYTES` (4 MiB) that end on row boundaries using the
  known row stride; a single row wider than the cap is the only exception
  and is emitted whole.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from faninsar.logging import setup_logger
from faninsar.missions.s1.errors import reject_product
from faninsar.remote.fetch import RangeSpan

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

logger = setup_logger(__name__)

MAX_SPAN_BYTES = 4 * 1024 * 1024
"""Retry quantum (PROPOSAL-0061): no span exceeds 4 MiB unless one row does."""

MERGE_FLOOR_BYTES = 256 * 1024
"""Adjacent sub-floor ranges merge; single requests stay above this floor."""

HEADER_PROBE_BYTES = 16
"""Bytes probed at offset 0: classic (8 B) and BigTIFF (16 B) IFD pointers."""

_RowGroup: TypeAlias = tuple[int, int, int, int]
"""Half-open row group ``(line0, line1, byte_start, byte_end)``."""

_RowInterval: TypeAlias = tuple[int, int]
"""Half-open byte interval ``(start, end)`` of one ROI row."""


def plan_measurement_spans(
    *,
    url: str,
    strip_offsets: Sequence[int],
    strip_byte_counts: Sequence[int],
    rows_per_strip: int,
    samples: int,
    bytes_per_sample: int,
    line0: int,
    line1: int,
    col0: int,
    col1: int,
    compressed: bool = False,
    tiled: bool = False,
) -> tuple[RangeSpan, ...]:
    """Plan bounded, row-aligned byte spans for one ROI of a strip TIFF.

    Rows are mapped to bytes through the strip tables: row ``line`` lives in
    strip ``line // rows_per_strip`` at ``strip_offsets[strip] +
    (line % rows_per_strip) * row_stride``.  The row stride is
    ``samples * bytes_per_sample``.  Labels are stable planner-generated
    ``row-{line0}-{line1}`` strings; the fetcher treats them as opaque.

    Parameters
    ----------
    url : str
        Object URL each span requests.
    strip_offsets : sequence of int
        TIFF ``StripOffsets``: byte offset of each strip.
    strip_byte_counts : sequence of int
        TIFF ``StripByteCounts``: byte length of each strip.  Every strip
        except the last must hold ``rows_per_strip`` whole rows; the last
        may hold fewer.
    rows_per_strip : int
        TIFF ``RowsPerStrip``.
    samples : int
        Samples per row.
    bytes_per_sample : int
        Storage bytes per sample (4 for Sentinel-1 complex-int16).
    line0, line1 : int
        ROI row bounds; ``line1`` is exclusive.
    col0, col1 : int
        ROI column bounds; ``col1`` is exclusive.
    compressed : bool, optional
        True for a compressed TIFF; rejected (default False).
    tiled : bool, optional
        True for a tiled TIFF; rejected (default False).

    Returns
    -------
    tuple of RangeSpan
        Bounded spans whose union covers every ROI byte.  Each span is at
        most :data:`MAX_SPAN_BYTES`, ends on a row boundary, and
        consecutive spans stay byte-adjacent within a group.  A narrow
        column (fetched width below one row stride) yields tight
        per-row-group ranges of at least :data:`MERGE_FLOOR_BYTES` — the
        interior gaps between a group's rows are fetched and discarded by
        assembly — except a final leftover group that may stay below the
        floor.

    Raises
    ------
    Sentinel1ProductError
        If the TIFF is compressed or tiled, the strip tables are missing,
        inconsistent with the row stride, or the ROI falls outside them.

    Examples
    --------
    >>> from faninsar.missions.s1.acquisition import plan_measurement_spans
    >>> offsets = [1000 + i * 4000 for i in range(8)]
    >>> spans = plan_measurement_spans(
    ...     url="https://example.test/measurement.tiff",
    ...     strip_offsets=offsets,
    ...     strip_byte_counts=[4000] * 8,
    ...     rows_per_strip=1,
    ...     samples=1000,
    ...     bytes_per_sample=4,
    ...     line0=3,
    ...     line1=7,
    ...     col0=0,
    ...     col1=1000,
    ... )
    >>> [(span.start, span.end) for span in spans]
    [(13000, 28999)]

    """
    if compressed:
        reject_product(
            "compressed measurement TIFF cannot be planned for range fetch; "
            "strip rows have no fixed byte stride"
        )
    if tiled:
        reject_product(
            "tiled measurement TIFF cannot be planned for range fetch; "
            "row-aligned spans require stripped organization"
        )
    if not url:
        reject_product("range span planning requires a non-empty object URL")
    if samples <= 0 or bytes_per_sample <= 0:
        reject_product(
            f"invalid sample geometry: samples={samples}, "
            f"bytes_per_sample={bytes_per_sample}"
        )
    row_stride = samples * bytes_per_sample
    total_rows = _validate_strip_tables(
        strip_offsets,
        strip_byte_counts,
        rows_per_strip=rows_per_strip,
        row_stride=row_stride,
    )
    if not 0 <= line0 < line1 <= total_rows:
        reject_product(
            f"ROI lines [{line0}, {line1}) fall outside the {total_rows} "
            "rows described by the strip tables"
        )
    if not 0 <= col0 < col1 <= samples:
        reject_product(
            f"ROI columns [{col0}, {col1}) fall outside {samples} samples per row"
        )

    whole_rows = (col1 - col0) * bytes_per_sample >= row_stride
    intervals: list[_RowInterval] = [
        _row_byte_interval(
            line,
            strip_offsets=strip_offsets,
            rows_per_strip=rows_per_strip,
            row_stride=row_stride,
            first_byte=col0 * bytes_per_sample,
            last_byte=col1 * bytes_per_sample,
            whole_rows=whole_rows,
        )
        for line in range(line0, line1)
    ]
    spans: list[RangeSpan] = []
    for group in _merge_row_groups(intervals, line0):
        spans.extend(_split_group(url, group, intervals, line0))
    return tuple(spans)


def plan_ifd_probe_spans(
    url: str,
    *,
    ifd_offset: int,
    ifd_length: int,
    strip_table_offset: int | None = None,
    strip_table_length: int | None = None,
) -> tuple[RangeSpan, ...]:
    """Plan bounded header/IFD/strip-table probe spans for a TIFF.

    Emits at most three bounded spans and never a ``0..EOF`` span:

    1. the TIFF header at offset 0 (:data:`HEADER_PROBE_BYTES` bytes, which
       carries the first-IFD pointer for classic and BigTIFF layouts);
    2. the IFD at ``ifd_offset``;
    3. the strip tables, only when both their offset and length are given
       and they do not live entirely inside the IFD.

    Parameters
    ----------
    url : str
        Object URL each span requests.
    ifd_offset : int
        Byte offset of the image file directory.  Must lie beyond the
        header probe; ESA/CDSE measurement TIFFs place the IFD near EOF.
    ifd_length : int
        Bounded byte length to read at ``ifd_offset``.
    strip_table_offset : int or None, optional
        Byte offset of the ``StripOffsets``/``StripByteCounts`` arrays when
        they live outside the IFD.
    strip_table_length : int or None, optional
        Bounded byte length of those arrays; required together with
        ``strip_table_offset``.

    Returns
    -------
    tuple of RangeSpan
        Two spans (header, IFD) or three (plus strip tables), each bounded.

    Raises
    ------
    Sentinel1ProductError
        If any offset or length is unusable, or only one of the strip
        table offset/length pair is given.

    Examples
    --------
    >>> from faninsar.missions.s1.acquisition import plan_ifd_probe_spans
    >>> spans = plan_ifd_probe_spans("u", ifd_offset=900_000, ifd_length=16_384)
    >>> [(span.start, span.end) for span in spans]
    [(0, 15), (900000, 916383)]

    """
    if not url:
        reject_product("range span planning requires a non-empty object URL")
    if ifd_offset < HEADER_PROBE_BYTES:
        reject_product(
            f"IFD offset {ifd_offset} falls inside the {HEADER_PROBE_BYTES} "
            "byte header probe; expected the pointer to land beyond it"
        )
    if ifd_length <= 0:
        reject_product(f"IFD length must be positive, got {ifd_length}")

    spans = [
        RangeSpan(url, 0, HEADER_PROBE_BYTES - 1, "tiff-header"),
        RangeSpan(url, ifd_offset, ifd_offset + ifd_length - 1, f"ifd-{ifd_offset}"),
    ]
    if strip_table_offset is None and strip_table_length is None:
        return tuple(spans)
    if strip_table_offset is None or strip_table_length is None:
        reject_product(
            "strip table probe needs offset and length together; got only one of them"
        )
    if strip_table_offset < HEADER_PROBE_BYTES:
        reject_product(
            f"strip table offset {strip_table_offset} falls inside the "
            f"{HEADER_PROBE_BYTES} byte header probe"
        )
    if strip_table_length <= 0:
        reject_product(f"strip table length must be positive, got {strip_table_length}")
    tables_end = strip_table_offset + strip_table_length
    if not (ifd_offset <= strip_table_offset and tables_end <= ifd_offset + ifd_length):
        spans.append(
            RangeSpan(
                url,
                strip_table_offset,
                tables_end - 1,
                f"strip-tables-{strip_table_offset}",
            )
        )
    return tuple(spans)


def _validate_strip_tables(
    strip_offsets: Sequence[int],
    strip_byte_counts: Sequence[int],
    *,
    rows_per_strip: int,
    row_stride: int,
) -> int:
    """Validate the strip tables against the row stride and count the rows.

    Parameters
    ----------
    strip_offsets, strip_byte_counts : sequence of int
        TIFF strip tables.
    rows_per_strip, row_stride : int
        Declared rows per strip and derived bytes per row.

    Returns
    -------
    int
        Total number of whole rows the tables describe.

    Raises
    ------
    Sentinel1ProductError
        If the tables are empty, mismatched in length, negatively offset,
        or inconsistent with whole rows of the declared stride.

    """
    strip_count = len(strip_offsets)
    if strip_count == 0:
        reject_product("strip tables are empty; cannot plan range spans")
    if len(strip_byte_counts) != strip_count:
        reject_product(
            f"strip table length mismatch: {strip_count} offsets vs "
            f"{len(strip_byte_counts)} byte counts"
        )
    total_rows = 0
    for index, (offset, count) in enumerate(
        zip(strip_offsets, strip_byte_counts, strict=True)
    ):
        if offset < 0:
            reject_product(f"strip {index} has a negative offset ({offset})")
        if count <= 0:
            reject_product(f"strip {index} has a non-positive byte count ({count})")
        if count % row_stride != 0:
            reject_product(
                f"strip {index} byte count {count} is not a multiple of the "
                f"row stride {row_stride}"
            )
        rows = count // row_stride
        if index < strip_count - 1 and rows != rows_per_strip:
            reject_product(
                f"strip {index} holds {rows} rows, expected {rows_per_strip} "
                "(only the last strip may be partial)"
            )
        if index == strip_count - 1 and rows > rows_per_strip:
            reject_product(
                f"final strip holds {rows} rows, more than RowsPerStrip "
                f"{rows_per_strip}"
            )
        total_rows += rows
    return total_rows


def _row_byte_interval(
    line: int,
    *,
    strip_offsets: Sequence[int],
    rows_per_strip: int,
    row_stride: int,
    first_byte: int,
    last_byte: int,
    whole_rows: bool,
) -> _RowInterval:
    """Return the half-open byte interval of one ROI row.

    Parameters
    ----------
    line : int
        ROI row index.
    strip_offsets : sequence of int
        TIFF ``StripOffsets``.
    rows_per_strip, row_stride : int
        Rows per strip and bytes per row.
    first_byte, last_byte : int
        Column bounds in row-relative bytes (``col * bytes_per_sample``).
    whole_rows : bool
        True to span the full row stride, False to span only the column
        slice.

    Returns
    -------
    _RowInterval
        Half-open ``(start, end)`` byte interval of the row.

    """
    row_start = (
        strip_offsets[line // rows_per_strip] + (line % rows_per_strip) * row_stride
    )
    if whole_rows:
        return row_start, row_start + row_stride
    return row_start + first_byte, row_start + last_byte


def _merge_row_groups(
    intervals: Sequence[_RowInterval],
    line0: int,
) -> list[_RowGroup]:
    """Coalesce per-row intervals into byte ranges.

    Byte-adjacent rows always continue the current group.  A group still
    below :data:`MERGE_FLOOR_BYTES` also absorbs the next row across an
    interior gap, so each request reaches the floor unless it is the final
    leftover of the ROI; a group at or above the floor starts a new range
    at the next gap.

    Parameters
    ----------
    intervals : sequence of _RowInterval
        Half-open byte intervals, one per ROI row in row order.
    line0 : int
        First ROI row index (labels and group bounds are row-based).

    Returns
    -------
    list of _RowGroup
        Groups as ``(line0, line1, byte_start, byte_end)`` with ``line1``
        exclusive.

    """
    groups: list[_RowGroup] = []
    for index, (start, end) in enumerate(intervals):
        if not groups:
            groups.append((line0 + index, line0 + index + 1, start, end))
            continue
        group_line0, group_line1, group_start, group_end = groups[-1]
        if group_end == start or group_end - group_start < MERGE_FLOOR_BYTES:
            groups[-1] = (group_line0, group_line1 + 1, group_start, end)
        else:
            groups.append((line0 + index, line0 + index + 1, start, end))
    return groups


def _split_group(
    url: str,
    group: _RowGroup,
    intervals: Sequence[_RowInterval],
    line0: int,
) -> list[RangeSpan]:
    """Split one group into spans of at most :data:`MAX_SPAN_BYTES`.

    Splits happen only between rows, so every span ends on a row boundary.
    A single row wider than the cap cannot satisfy both constraints and is
    emitted whole with a warning.

    Parameters
    ----------
    url : str
        Object URL for the emitted spans.
    group : _RowGroup
        Group to split as ``(line0, line1, byte_start, byte_end)``.
    intervals : sequence of _RowInterval
        Byte interval of every ROI row, indexed from ``line0``.
    line0 : int
        First ROI row index.

    Returns
    -------
    list of RangeSpan
        Row-aligned spans labeled ``row-{line0}-{line1}`` (``line1``
        exclusive).

    """
    group_line0, group_line1, group_start, _ = group
    spans: list[RangeSpan] = []
    span_line0 = group_line0
    span_start = group_start
    prev_row_end = group_start
    for line in range(group_line0, group_line1):
        row_start, row_end = intervals[line - line0]
        if line > group_line0 and row_end - span_start > MAX_SPAN_BYTES:
            spans.append(
                RangeSpan(url, span_start, prev_row_end - 1, f"row-{span_line0}-{line}")
            )
            span_line0 = line
            span_start = row_start
        prev_row_end = row_end
    if prev_row_end - span_start > MAX_SPAN_BYTES:
        logger.warning(
            "row %s-%s needs %d bytes; emitted as one span above the %d "
            "byte cap because spans must end on row boundaries",
            span_line0,
            group_line1,
            prev_row_end - span_start,
            MAX_SPAN_BYTES,
        )
    spans.append(
        RangeSpan(url, span_start, prev_row_end - 1, f"row-{span_line0}-{group_line1}")
    )
    return spans
