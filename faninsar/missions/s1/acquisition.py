"""Mission-owned Sentinel-1 ROI acquisition and row-to-byte span planning.

PROPOSAL-0061 splits the concern: :mod:`faninsar.remote` transports
already-sized :class:`~faninsar.remote.fetch.RangeSpan` objects and knows
nothing about TIFF geometry, while the mission layer owns the row-to-byte
mapping, provider dispatch, and SAFE-subset materialization.  This module
does not implement HTTP or authentication; it orchestrates remote fetch
primitives.

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
  and is emitted whole;
- CDSE ROI/burst paths write a full-burst-height canvas, splice overlap
  strips, then hand the SAFE tree to ``from_safes``.

"""

from __future__ import annotations

import asyncio
import re
import struct
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import reject_product
from faninsar.missions.s1.io.canvas import (
    BurstFetchedWindow,
    splice_overlap_into_canvas,
    write_roi_safe_canvas,
)
from faninsar.missions.s1.overlap import burst_overlap_windows
from faninsar.remote.errors import RemoteAccessError
from faninsar.remote.fetch import RangeSpan, fetch_ranges_async

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping, Sequence
    from datetime import datetime
    from typing import TypeAlias

    from faninsar.data.query import BoundingBox, Points, Polygons
    from faninsar.missions.s1.io.canvas import RoiSafeCanvas
    from faninsar.missions.s1.overlap import BurstOverlapWindow
    from faninsar.missions.s1.types import S1Swath

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

_BYTES_PER_COMPLEX_INT16 = 4
"""Storage bytes per Sentinel-1 complex-int16 sample."""

_ROW_SPAN_LABEL = re.compile(r"^row-(\d+)-(\d+)$")
"""Planner labels ``row-{line0}-{line1}`` with exclusive ``line1``."""

_TIFF_TYPE_SIZE = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    7: 1,  # UNDEFINED
    12: 8,  # DOUBLE
    16: 8,  # LONG8
    17: 8,  # SLONG8
    18: 8,  # IFD8
}
_TIFF_INT_FORMAT = {1: "B", 3: "H", 4: "I", 16: "Q"}
_TAG_COMPRESSION = 259
_TAG_STRIP_OFFSETS = 273
_TAG_ROWS_PER_STRIP = 278
_TAG_STRIP_BYTE_COUNTS = 279
_TAG_TILE_WIDTH = 322
_TAG_TILE_LENGTH = 323
_TAG_TILE_OFFSETS = 324
_TAG_TILE_BYTE_COUNTS = 325
_TILE_TAGS = frozenset(
    {
        _TAG_TILE_WIDTH,
        _TAG_TILE_LENGTH,
        _TAG_TILE_OFFSETS,
        _TAG_TILE_BYTE_COUNTS,
    }
)
_MAX_IFD_ENTRIES = 512


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


RoiGranularity = Literal["roi", "burst", "scene"]
RoiProvider = Literal["cdse", "asf"]

_GRANULARITY_PROVIDERS: dict[str, frozenset[str]] = {
    "roi": frozenset({"cdse"}),
    "burst": frozenset({"cdse", "asf"}),
    "scene": frozenset({"asf"}),
}
_DEFAULT_PROVIDER: dict[str, str] = {
    "roi": "cdse",
    "burst": "cdse",
    "scene": "asf",
}
_CDSE_CATALOG = "sentinel1-cdse"
_CDSE_AUTH_PROFILE = "cdse"
_ANNOTATION_PROBE_BYTES = 2 * 1024 * 1024
_IFD_PROBE_BYTES = 16 * 1024


def resolve_roi_acquisition(
    granularity: str = "roi",
    provider: str | None = None,
) -> tuple[str, str]:
    """Resolve the scientific granularity and derived provider.

    Parameters
    ----------
    granularity : {"roi", "burst", "scene"}, optional
        Acquisition granularity.  Default is ``"roi"``.
    provider : {"cdse", "asf"} or None, optional
        Explicit provider.  When omitted, ``roi`` and ``burst`` infer
        ``cdse`` and ``scene`` infers ``asf``.

    Returns
    -------
    tuple of str
        ``(granularity, provider)`` after applying the compatibility table.

    Raises
    ------
    ValueError
        If ``granularity`` is unknown or the pair is illegal.  Illegal
        combinations cite the provider x granularity table.

    """
    if granularity not in _GRANULARITY_PROVIDERS:
        allowed = ", ".join(sorted(_GRANULARITY_PROVIDERS))
        message = f"granularity must be one of {allowed}; got {granularity!r}"
        logger.error(message)
        raise ValueError(message)
    if provider is not None and provider not in {"cdse", "asf"}:
        message = f"provider must be 'cdse' or 'asf'; got {provider!r}"
        logger.error(message)
        raise ValueError(message)
    resolved_provider = _DEFAULT_PROVIDER[granularity] if provider is None else provider
    allowed = _GRANULARITY_PROVIDERS[granularity]
    if resolved_provider not in allowed:
        table = (
            "granularity | providers | inferred provider\n"
            "roi         | cdse      | cdse\n"
            "burst       | cdse, asf | cdse\n"
            "scene       | asf       | asf"
        )
        message = (
            f"granularity={granularity!r} is incompatible with "
            f"provider={resolved_provider!r}; allowed providers are "
            f"{sorted(allowed)}. See the provider x granularity table:\n{table}"
        )
        logger.error(message)
        raise ValueError(message)
    return granularity, resolved_provider


def run_sync(coro: Awaitable[Any]) -> Any:
    """Bridge an async fetch onto the sync Stack constructor.

    With no running event loop this uses :func:`asyncio.run`.  When a loop
    is already running (Jupyter, QGIS), the coroutine runs on a dedicated
    thread with a fresh loop.  ``nest_asyncio`` is not used, and
    ``ClientSession`` objects are never cached on long-lived Stack objects.

    Parameters
    ----------
    coro : awaitable
        Coroutine produced by a remote fetch primitive.

    Returns
    -------
    Any
        The coroutine's result.

    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    holder: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            holder["result"] = loop.run_until_complete(coro)
        except BaseException as exc:
            holder["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(
        target=_runner,
        name="faninsar-s1-range-fetch",
        daemon=True,
    )
    thread.start()
    thread.join()
    error = holder.get("error")
    if error is not None:
        raise error
    return holder["result"]


def discover_cdse_slc_items(
    roi: BoundingBox | Points | Polygons,
    datetime_range: tuple[datetime, datetime] | None = None,
    *,
    polarizations: Sequence[str] | Literal["all"] = ("VV",),
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Discover CDSE SLC products and bind OData measurement node URLs.

    The catalog adapter is expected to emit HTTPS OData
    ``Nodes(...)/$value`` URLs and that node's ``file:size``.  STAC/S3
    ``eodata`` hrefs are not range URLs.

    Parameters
    ----------
    roi : BoundingBox, Points, or Polygons
        Spatial query.
    datetime_range : tuple, optional
        Inclusive acquisition time range.
    polarizations : sequence of str or ``"all"``, optional
        Local channels used only for logging; CDSE filtering stays in
        ``kwargs["filters"]``.
    **kwargs : Any
        Advanced search options forwarded to :func:`faninsar.remote.search`
        (``limit``, ``budget``, ``filters``, ``collections``).  ``catalog``
        and ``auth_profile`` default to the internal CDSE binding.

    Returns
    -------
    list of dict
        One binding per product with ``product_id``, ``annotation_url``,
        ``measurement_url``, ``measurement_size``, ``polarization``, and
        ``swath``.

    """
    del polarizations
    from faninsar import remote

    catalog = str(kwargs.pop("catalog", _CDSE_CATALOG))
    auth_profile = str(kwargs.pop("auth_profile", _CDSE_AUTH_PROFILE))
    try:
        items = remote.search(
            roi,
            catalog=catalog,
            datetime_range=datetime_range,
            auth_profile=auth_profile,
            **kwargs,
        )
    except RemoteAccessError as exc:
        if exc.reason == "unknown_catalog":
            message = (
                f"CDSE STAC adapter is required and not registered for catalog "
                f"{catalog!r}; cannot discover Sentinel-1 CDSE products"
            )
            logger.exception(message)
            reason = "unknown_catalog"
            raise RemoteAccessError(reason, message) from exc
        raise
    bindings = [_bind_cdse_item(item) for item in items]
    return [binding for binding in bindings if binding is not None]


def windows_from_cdse_fetch(
    item: Mapping[str, Any],
    roi: BoundingBox | Points | Polygons,
    *,
    xml_text: str,
    granularity: str = "roi",
    **kwargs: Any,
) -> tuple[S1Swath, list[BurstFetchedWindow], str]:
    """Build fetched windows for one CDSE measurement from annotation XML.

    Bounded IFD probes and measurement spans are planned here and fetched
    through :func:`fetch_ranges_async`.  Compressed or tiled TIFFs fail
    closed in :func:`plan_measurement_spans`.

    Parameters
    ----------
    item : mapping
        Binding from :func:`discover_cdse_slc_items`.
    roi : BoundingBox, Points, or Polygons
        Spatial query inverted through the geolocation grid.
    xml_text : str
        Annotation XML already fetched for this measurement.
    granularity : {"roi", "burst"}, optional
        ``burst`` fetches full-burst row ranges; ``roi`` uses the inverted
        line/pixel envelope.
    **kwargs : Any
        Unused; accepted so tests can add seams without breaking callers.

    Returns
    -------
    swath, windows, xml_text
        Parsed parent swath, one window per intersecting burst, and the
        annotation text to rewrite into the canvas.

    """
    del kwargs
    swath = parse_annotation_xml(
        xml_text,
        annotation_path=str(item.get("annotation_url", "cdse://annotation.xml")),
        measurement_path=str(item.get("measurement_url", "cdse://measurement.tiff")),
    )
    line0, line1, col0, col1 = invert_geolocation_grid(xml_text, roi)
    if granularity == "burst":
        line0, line1 = _full_intersecting_burst_lines(swath, line0, line1)
        col0, col1 = 0, int(swath.samples_per_burst)
    windows = _windows_for_lines(
        swath,
        line0=line0,
        line1=line1,
        col0=col0,
        col1=col1,
        url=str(item.get("measurement_url", "")),
        file_size=int(item.get("measurement_size") or 0),
    )
    return swath, windows, xml_text


def overlap_windows_for_canvas(
    swath: S1Swath,
    windows: Sequence[BurstFetchedWindow],
    canvas: RoiSafeCanvas | None = None,
) -> tuple[BurstOverlapWindow, ...]:
    """Return parent overlap rectangles among kept canvas bursts.

    Completeness is the parent :func:`burst_overlap_windows` rectangle,
    not a tight azimuth box.  A synthetic parent with no sensing-time
    overlap yields an empty tuple.

    """
    del canvas
    kept = {int(window.burst_index) for window in windows}
    if len(kept) < 2:
        return ()
    return burst_overlap_windows(swath, burst_indices=sorted(kept), min_lines=1)


def invert_geolocation_grid(
    xml_text: str,
    roi: BoundingBox | Points | Polygons,
) -> tuple[int, int, int, int]:
    """Invert annotation geolocation-grid points to a line/pixel envelope.

    Parameters
    ----------
    xml_text : str
        Annotation XML containing ``geolocationGridPoint`` entries.
    roi : BoundingBox, Points, or Polygons
        Spatial query.  A :class:`~faninsar.data.query.BoundingBox` uses
        its four corners; other geometries use their bounds.

    Returns
    -------
    tuple of int
        Half-open ``(line0, line1, col0, col1)`` covering the ROI in
        parent-swath coordinates.

    Raises
    ------
    Sentinel1ProductError
        If the annotation has no geolocation grid.

    """
    points = _geolocation_points(xml_text)
    if not points:
        reject_product("annotation has no geolocationGrid points to invert")
    left, bottom, right, top = _roi_bounds(roi)
    corners = (
        (left, bottom),
        (left, top),
        (right, bottom),
        (right, top),
    )
    lines: list[int] = []
    pixels: list[int] = []
    for lon, lat in corners:
        nearest = min(points, key=lambda p: (p[2] - lat) ** 2 + (p[3] - lon) ** 2)
        lines.append(int(nearest[0]))
        pixels.append(int(nearest[1]))
    line0, line1 = min(lines), max(lines) + 1
    col0, col1 = min(pixels), max(pixels) + 1
    if line1 <= line0:
        line1 = line0 + 1
    if col1 <= col0:
        col1 = col0 + 1
    return line0, line1, col0, col1


def materialize_cdse_roi_safes(
    roi: BoundingBox | Points | Polygons,
    datetime_range: tuple[datetime, datetime] | None = None,
    *,
    polarizations: Sequence[str] | Literal["all"] = ("VV",),
    work_dir: str | Path,
    granularity: str = "roi",
    **kwargs: Any,
) -> list[Path]:
    """Materialize CDSE ROI/burst SAFE-subset canvases under ``work_dir``.

    Order (PROPOSAL-0061): fetch annotation ranges, assemble windows,
    :func:`write_roi_safe_canvas`, splice parent overlap rectangles,
    then return the ``.SAFE`` roots for ``from_safes``.

    Parameters
    ----------
    roi, datetime_range, polarizations
        Search envelope forwarded to :func:`discover_cdse_slc_items`.
    work_dir : path-like
        Stack work directory; canvases are written under ``inputs/``.
    granularity : {"roi", "burst"}, optional
        CDSE acquisition granularity.
    **kwargs : Any
        Forwarded to discovery.

    Returns
    -------
    list of pathlib.Path
        Generated ``.SAFE`` directory roots.

    Raises
    ------
    ValueError
        If discovery returns no products.

    """
    items = discover_cdse_slc_items(
        roi,
        datetime_range=datetime_range,
        polarizations=polarizations,
        **kwargs,
    )
    if not items:
        message = "Sentinel-1 CDSE search returned no products"
        logger.error(message)
        raise ValueError(message)
    out_dir = Path(work_dir) / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for item in items:
        xml_text = _fetch_annotation_xml(item)
        swath, windows, xml_text = windows_from_cdse_fetch(
            item,
            roi,
            xml_text=xml_text,
            granularity=granularity,
        )
        canvas = write_roi_safe_canvas(
            swath,
            windows,
            out_dir,
            annotation_xml=xml_text,
            product_id=str(item.get("product_id", "S1_ROI")),
        )
        by_index = {int(window.burst_index): window for window in windows}
        for overlap in overlap_windows_for_canvas(swath, windows, canvas):
            if (
                overlap.top_index not in by_index
                or overlap.bottom_index not in by_index
            ):
                continue
            top = _overlap_rectangle_samples(
                item,
                swath,
                by_index[int(overlap.top_index)],
                overlap,
                role="top",
            )
            bottom = _overlap_rectangle_samples(
                item,
                swath,
                by_index[int(overlap.bottom_index)],
                overlap,
                role="bottom",
            )
            splice_overlap_into_canvas(canvas, overlap, role="top", samples=top)
            splice_overlap_into_canvas(canvas, overlap, role="bottom", samples=bottom)
        paths.append(canvas.safe_root)
    return paths


def _fetch_annotation_xml(item: Mapping[str, Any]) -> str:
    """Fetch one annotation XML document via the async range engine."""
    url = str(item.get("annotation_url", ""))
    size = int(item.get("annotation_size") or _ANNOTATION_PROBE_BYTES)
    span = RangeSpan(url, 0, max(size - 1, 0), "annotation")
    bodies = run_sync(fetch_ranges_async((span,)))
    return bodies["annotation"].decode("utf-8")


def _bind_cdse_item(item: Any) -> dict[str, Any] | None:
    """Bind one catalog item to OData annotation and measurement node URLs."""
    assets = getattr(item, "assets", {}) or {}
    annotation = None
    measurement = None
    for asset in assets.values():
        href = str(getattr(asset, "href", ""))
        key = str(getattr(asset, "key", "")).casefold()
        media = str(getattr(asset, "media_type", "") or "").casefold()
        if annotation is None and (
            "annotation" in key or href.casefold().endswith(".xml") or "xml" in media
        ):
            annotation = asset
        if measurement is None and (
            "measurement" in key
            or href.casefold().endswith((".tif", ".tiff"))
            or "tiff" in media
        ):
            measurement = asset
    if annotation is None or measurement is None:
        logger.warning(
            "CDSE item %s is missing annotation or measurement OData node",
            getattr(item, "item_id", "?"),
        )
        return None
    annotation_url = str(annotation.href)
    measurement_url = str(measurement.href)
    _require_odata_value_href(annotation_url, role="annotation")
    _require_odata_value_href(measurement_url, role="measurement")
    properties = dict(getattr(measurement, "properties", {}) or {})
    size = getattr(measurement, "size_bytes", None) or properties.get("file:size")
    return {
        "product_id": str(getattr(item, "item_id", "S1_ROI")),
        "annotation_url": annotation_url,
        "measurement_url": measurement_url,
        "measurement_size": int(size or 0),
        "polarization": str(
            getattr(item, "raw_metadata", {}).get("polarization", "VV")
            if hasattr(item, "raw_metadata")
            else "VV"
        ),
        "swath": str(properties.get("swath", "IW1")),
    }


def _require_odata_value_href(href: str, *, role: str) -> None:
    """Reject STAC/S3 ``eodata`` hrefs; CDSE range fetch needs OData ``$value``."""
    parsed = urllib.parse.urlsplit(href)
    scheme = parsed.scheme.casefold()
    path = parsed.path
    host = (parsed.netloc or "").casefold()
    path_folded = path.casefold()
    if scheme not in {"http", "https"} or "eodata" in host or "eodata" in path_folded:
        reject_product(
            f"CDSE {role} href is not an OData Nodes(...)/$value URL: {href}"
        )
    if "/Nodes(" not in path or not path.rstrip("/").endswith("/$value"):
        reject_product(
            f"CDSE {role} href is not an OData Nodes(...)/$value URL: {href}"
        )


def _geolocation_points(xml_text: str) -> list[tuple[int, int, float, float]]:
    """Return ``(line, pixel, lat, lon)`` tuples from annotation XML."""
    root = ET.fromstring(xml_text)
    points: list[tuple[int, int, float, float]] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "geolocationGridPoint":
            continue
        fields = {
            child.tag.rsplit("}", 1)[-1]: (child.text or "").strip()
            for child in element
        }
        if (
            "line" not in fields
            or "latitude" not in fields
            or "longitude" not in fields
        ):
            continue
        pixel = int(fields["pixel"]) if fields.get("pixel") else 0
        points.append(
            (
                int(fields["line"]),
                pixel,
                float(fields["latitude"]),
                float(fields["longitude"]),
            )
        )
    return points


def _roi_bounds(
    roi: BoundingBox | Points | Polygons,
) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, right, top)`` for a FanInSAR spatial query."""
    left = getattr(roi, "left", None)
    if left is not None:
        return (
            float(roi.left),
            float(roi.bottom),
            float(roi.right),
            float(roi.top),
        )
    bounds = roi.to_geodataframe().total_bounds
    return float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])


def _full_intersecting_burst_lines(
    swath: S1Swath,
    line0: int,
    line1: int,
) -> tuple[int, int]:
    """Expand an ROI line envelope to whole intersecting bursts."""
    lpb = int(swath.lines_per_burst)
    first = min(burst.index for burst in swath.bursts)
    last = max(burst.index for burst in swath.bursts)
    start = max(first, line0 // lpb)
    stop = min(last, (line1 - 1) // lpb)
    return start * lpb, (stop + 1) * lpb


def _windows_for_lines(
    swath: S1Swath,
    *,
    line0: int,
    line1: int,
    col0: int,
    col1: int,
    url: str,
    file_size: int,
) -> list[BurstFetchedWindow]:
    """Build per-burst fetched windows, filling samples from range fetches."""
    lpb = int(swath.lines_per_burst)
    samples_per_burst = int(swath.samples_per_burst)
    col0 = max(0, col0)
    col1 = min(samples_per_burst, max(col0 + 1, col1))
    bodies = _fetch_measurement_bodies(
        url,
        file_size=file_size,
        line0=line0,
        line1=line1,
        col0=col0,
        col1=col1,
        samples=samples_per_burst,
    )
    windows: list[BurstFetchedWindow] = []
    for burst in swath.bursts:
        burst_line0 = burst.index * lpb
        burst_line1 = burst_line0 + lpb
        inter0 = max(line0, burst_line0)
        inter1 = min(line1, burst_line1)
        if inter1 <= inter0:
            continue
        row0 = inter0 - burst_line0
        n_lines = inter1 - inter0
        n_cols = col1 - col0
        samples = _assemble_window_samples(
            bodies,
            parent_line0=inter0,
            n_lines=n_lines,
            col0=col0,
            n_cols=n_cols,
            samples_per_burst=samples_per_burst,
        )
        windows.append(
            BurstFetchedWindow(
                burst_index=int(burst.index),
                row0=row0,
                col0=col0,
                samples=samples,
            )
        )
    if not windows:
        reject_product("ROI does not intersect any Sentinel-1 burst")
    return windows


def _fetch_measurement_bodies(
    url: str,
    *,
    file_size: int,
    line0: int,
    line1: int,
    col0: int,
    col1: int,
    samples: int,
) -> dict[str, bytes]:
    """Fetch IFD probes then bounded measurement spans for one ROI window."""
    if not url:
        reject_product("CDSE measurement fetch requires a non-empty object URL")
    ifd_offset = max(file_size - _IFD_PROBE_BYTES, HEADER_PROBE_BYTES)
    probes = plan_ifd_probe_spans(
        url,
        ifd_offset=ifd_offset,
        ifd_length=_IFD_PROBE_BYTES,
    )
    probe_bodies = run_sync(fetch_ranges_async(probes))
    offsets, counts, rows_per_strip = _strip_tables_from_probes(
        probe_bodies, samples=samples
    )
    spans = plan_measurement_spans(
        url=url,
        strip_offsets=offsets,
        strip_byte_counts=counts,
        rows_per_strip=rows_per_strip,
        samples=samples,
        bytes_per_sample=_BYTES_PER_COMPLEX_INT16,
        line0=line0,
        line1=line1,
        col0=col0,
        col1=col1,
    )
    return run_sync(fetch_ranges_async(spans))


def _strip_tables_from_probes(
    bodies: Mapping[str, bytes],
    *,
    samples: int,
) -> tuple[list[int], list[int], int]:
    """Parse strip tables from IFD probe bodies for an uncompressed strip TIFF."""
    del samples
    header = _probe_body(bodies, "tiff-header")
    ifd_offset, endian, bigtiff = _parse_tiff_header(header)
    regions = _probe_regions(bodies)
    ifd_body = _ifd_bytes_from_regions(regions, ifd_offset)
    tags = _parse_ifd_tags(
        ifd_body,
        endian=endian,
        bigtiff=bigtiff,
        regions=regions,
        ifd_file_offset=ifd_offset,
    )
    if any(tag in tags for tag in _TILE_TAGS):
        reject_product(
            "tiled measurement TIFF cannot be planned for range fetch; "
            "row-aligned spans require stripped organization"
        )
    compression = _first_int_tag(tags, _TAG_COMPRESSION, endian=endian, default=1)
    if compression != 1:
        reject_product(
            "compressed measurement TIFF cannot be planned for range fetch; "
            "strip rows have no fixed byte stride"
        )
    rows_per_strip = _first_int_tag(tags, _TAG_ROWS_PER_STRIP, endian=endian)
    if rows_per_strip is None or rows_per_strip <= 0:
        reject_product("measurement TIFF IFD is missing a positive RowsPerStrip")
    offsets = _int_tag_values(tags, _TAG_STRIP_OFFSETS, endian=endian)
    counts = _int_tag_values(tags, _TAG_STRIP_BYTE_COUNTS, endian=endian)
    if not offsets or not counts:
        reject_product(
            "measurement TIFF IFD is missing StripOffsets or StripByteCounts"
        )
    if len(offsets) != len(counts):
        reject_product(
            "StripOffsets length "
            f"{len(offsets)} != StripByteCounts length {len(counts)}"
        )
    return offsets, counts, rows_per_strip


def _assemble_window_samples(
    bodies: Mapping[str, bytes],
    *,
    parent_line0: int,
    n_lines: int,
    col0: int,
    n_cols: int,
    samples_per_burst: int,
) -> np.ndarray:
    """Assemble complex64 samples from fetched row-labeled span bodies."""
    canvas = np.zeros((n_lines, n_cols), dtype=np.complex64)
    parent_line1 = parent_line0 + n_lines
    for label, payload in bodies.items():
        match = _ROW_SPAN_LABEL.fullmatch(label)
        if match is None:
            continue
        span_line0 = int(match.group(1))
        span_line1 = int(match.group(2))
        if span_line1 <= span_line0:
            reject_product(f"row span {label} has an empty line interval")
        n_span_lines = span_line1 - span_line0
        decoded = _bytes_to_complex_span(
            payload,
            n_span_lines=n_span_lines,
            samples_per_burst=samples_per_burst,
            col0=col0,
            n_cols=n_cols,
        )
        overlap0 = max(parent_line0, span_line0)
        overlap1 = min(parent_line1, span_line1)
        if overlap1 <= overlap0:
            continue
        dst0 = overlap0 - parent_line0
        src0 = overlap0 - span_line0
        height = overlap1 - overlap0
        canvas[dst0 : dst0 + height] = decoded[src0 : src0 + height]
    return canvas


def _overlap_rectangle_samples(
    item: Mapping[str, Any],
    swath: S1Swath,
    window: BurstFetchedWindow,
    overlap: BurstOverlapWindow,
    *,
    role: Literal["top", "bottom"],
) -> np.ndarray:
    """Return one overlap rectangle from a fetched window or a dedicated fetch."""
    burst_row0 = int(overlap.top_row0 if role == "top" else overlap.bottom_row0)
    burst_col0 = int(overlap.col0)
    n_lines = int(overlap.n_lines)
    n_samples = int(overlap.n_samples)
    covered = _window_covers_rectangle(
        window,
        row0=burst_row0,
        col0=burst_col0,
        n_lines=n_lines,
        n_cols=n_samples,
    )
    if covered:
        src_row0 = burst_row0 - int(window.row0)
        src_col0 = burst_col0 - int(window.col0)
        rectangle = window.samples[
            src_row0 : src_row0 + n_lines, src_col0 : src_col0 + n_samples
        ]
        return np.ascontiguousarray(rectangle, dtype=np.complex64)
    lpb = int(swath.lines_per_burst)
    parent_line0 = int(window.burst_index) * lpb + burst_row0
    bodies = _fetch_measurement_bodies(
        str(item.get("measurement_url", "")),
        file_size=int(item.get("measurement_size") or 0),
        line0=parent_line0,
        line1=parent_line0 + n_lines,
        col0=burst_col0,
        col1=burst_col0 + n_samples,
        samples=int(swath.samples_per_burst),
    )
    return _assemble_window_samples(
        bodies,
        parent_line0=parent_line0,
        n_lines=n_lines,
        col0=burst_col0,
        n_cols=n_samples,
        samples_per_burst=int(swath.samples_per_burst),
    )


def _window_covers_rectangle(
    window: BurstFetchedWindow,
    *,
    row0: int,
    col0: int,
    n_lines: int,
    n_cols: int,
) -> bool:
    """Return True when ``window`` fully contains the burst-local rectangle."""
    samples = window.samples
    if samples.ndim != 2:
        return False
    win_row1 = int(window.row0) + samples.shape[0]
    win_col1 = int(window.col0) + samples.shape[1]
    return (
        int(window.row0) <= row0
        and int(window.col0) <= col0
        and row0 + n_lines <= win_row1
        and col0 + n_cols <= win_col1
    )


def _probe_body(bodies: Mapping[str, bytes], label: str) -> bytes:
    """Return one named probe body or fail closed."""
    payload = bodies.get(label)
    if payload is None:
        reject_product(f"IFD probe body {label!r} is missing")
    return payload


def _ifd_bytes_from_regions(
    regions: Sequence[tuple[int, bytes]],
    ifd_offset: int,
) -> bytes:
    """Return the IFD starting at the header pointer inside a covering probe."""
    for start, payload in regions:
        relative = ifd_offset - start
        if relative >= 0 and relative < len(payload):
            return payload[relative:]
    return reject_product(
        f"IFD at file offset {ifd_offset} is not covered by any probe body"
    )


def _parse_tiff_header(header: bytes) -> tuple[int, str, bool]:
    """Return ``(ifd_offset, endian, bigtiff)`` from a TIFF header probe."""
    if len(header) < 8:
        reject_product("TIFF header probe is shorter than 8 bytes")
    endian_code = header[:2]
    if endian_code == b"II":
        endian = "<"
    elif endian_code == b"MM":
        endian = ">"
    else:
        reject_product(f"unrecognized TIFF byte order {endian_code!r}")
    magic = struct.unpack_from(f"{endian}H", header, 2)[0]
    if magic == 42:
        ifd_offset = int(struct.unpack_from(f"{endian}I", header, 4)[0])
        return ifd_offset, endian, False
    if magic == 43:
        if len(header) < HEADER_PROBE_BYTES:
            reject_product("BigTIFF header probe is shorter than 16 bytes")
        offset_size, reserved = struct.unpack_from(f"{endian}HH", header, 4)
        if offset_size != 8 or reserved != 0:
            reject_product("BigTIFF header has an unexpected offset size")
        ifd_offset = int(struct.unpack_from(f"{endian}Q", header, 8)[0])
        return ifd_offset, endian, True
    return reject_product(f"unrecognized TIFF magic number {magic}")


def _probe_regions(bodies: Mapping[str, bytes]) -> list[tuple[int, bytes]]:
    """Map probe labels to ``(file_offset, payload)`` regions."""
    regions: list[tuple[int, bytes]] = []
    for label, payload in bodies.items():
        if label == "tiff-header":
            start = 0
        elif label.startswith("ifd-"):
            start = int(label[4:])
        elif label.startswith("strip-tables-"):
            start = int(label[len("strip-tables-") :])
        else:
            continue
        regions.append((start, payload))
    return regions


def _bytes_at(
    regions: Sequence[tuple[int, bytes]], offset: int, nbytes: int
) -> bytes | None:
    """Return ``nbytes`` at a file offset, or None if no probe covers it."""
    for start, payload in regions:
        relative = offset - start
        if relative >= 0 and relative + nbytes <= len(payload):
            return payload[relative : relative + nbytes]
    return None


def _parse_ifd_tags(
    ifd_body: bytes,
    *,
    endian: str,
    bigtiff: bool,
    regions: Sequence[tuple[int, bytes]],
    ifd_file_offset: int,
) -> dict[int, tuple[int, bytes]]:
    """Parse classic or BigTIFF IFD entries from a probe body."""
    count_fmt = "Q" if bigtiff else "H"
    count_size = 8 if bigtiff else 2
    entry_size = 20 if bigtiff else 12
    offset_size = 8 if bigtiff else 4
    if len(ifd_body) < count_size:
        reject_product("IFD probe is shorter than the directory count")
    n_entries = int(struct.unpack_from(f"{endian}{count_fmt}", ifd_body, 0)[0])
    if n_entries <= 0 or n_entries > _MAX_IFD_ENTRIES:
        reject_product(f"IFD entry count {n_entries} is not usable")
    needed = count_size + n_entries * entry_size
    if len(ifd_body) < needed:
        reject_product(
            f"IFD probe is {len(ifd_body)} bytes, need {needed} for {n_entries} entries"
        )
    tags: dict[int, tuple[int, bytes]] = {}
    for index in range(n_entries):
        start = count_size + index * entry_size
        if bigtiff:
            tag, typ, count = struct.unpack_from(f"{endian}HHQ", ifd_body, start)
            value_offset = start + 12
        else:
            tag, typ, count = struct.unpack_from(f"{endian}HHI", ifd_body, start)
            value_offset = start + 8
        payload = _ifd_value_bytes(
            ifd_body,
            typ=int(typ),
            count=int(count),
            value_offset=value_offset,
            endian=endian,
            offset_size=offset_size,
            regions=regions,
            ifd_file_offset=ifd_file_offset,
        )
        tags[int(tag)] = (int(typ), payload)
    return tags


def _ifd_value_bytes(
    ifd_body: bytes,
    *,
    typ: int,
    count: int,
    value_offset: int,
    endian: str,
    offset_size: int,
    regions: Sequence[tuple[int, bytes]],
    ifd_file_offset: int,
) -> bytes:
    """Return the value bytes of one IFD entry, following file offsets."""
    unit = _TIFF_TYPE_SIZE.get(typ)
    if unit is None:
        reject_product(f"unsupported TIFF field type {typ}")
    nbytes = unit * count
    inline = ifd_body[value_offset : value_offset + offset_size]
    if nbytes <= offset_size:
        return inline[:nbytes]
    offset_fmt = "Q" if offset_size == 8 else "I"
    if len(inline) < offset_size:
        reject_product("IFD value offset is truncated")
    pointer = int(struct.unpack_from(f"{endian}{offset_fmt}", inline, 0)[0])
    payload = _bytes_at(regions, pointer, nbytes)
    if payload is not None:
        return payload
    relative = pointer - ifd_file_offset
    if relative >= 0 and relative + nbytes <= len(ifd_body):
        return ifd_body[relative : relative + nbytes]
    return reject_product(
        "IFD extra values at file offset "
        f"{pointer} ({nbytes} bytes) are not in any probe"
    )


def _int_tag_values(
    tags: Mapping[int, tuple[int, bytes]],
    tag: int,
    *,
    endian: str,
) -> list[int]:
    """Decode an integer TIFF tag into a list of ints."""
    entry = tags.get(tag)
    if entry is None:
        return []
    typ, payload = entry
    fmt = _TIFF_INT_FORMAT.get(typ)
    if fmt is None:
        reject_product(f"TIFF tag {tag} has non-integer type {typ}")
    unit = struct.calcsize(fmt)
    if len(payload) % unit != 0:
        reject_product(f"TIFF tag {tag} payload is not a whole number of integers")
    count = len(payload) // unit
    values = struct.unpack(f"{endian}{count}{fmt}", payload)
    return [int(value) for value in values]


def _first_int_tag(
    tags: Mapping[int, tuple[int, bytes]],
    tag: int,
    *,
    endian: str,
    default: int | None = None,
) -> int | None:
    """Return the first integer of a tag, or ``default`` when absent."""
    values = _int_tag_values(tags, tag, endian=endian)
    if not values:
        return default
    return values[0]


def _bytes_to_complex_span(
    raw: bytes,
    *,
    n_span_lines: int,
    samples_per_burst: int,
    col0: int,
    n_cols: int,
) -> np.ndarray:
    """Reinterpret a row-labeled span as complex64, slicing fetched columns."""
    if n_span_lines <= 0 or n_cols <= 0:
        reject_product("row span assembly requires a positive rectangle")
    expected_full = n_span_lines * samples_per_burst * _BYTES_PER_COMPLEX_INT16
    expected_slice = n_span_lines * n_cols * _BYTES_PER_COMPLEX_INT16
    if len(raw) == expected_slice:
        pairs = np.frombuffer(raw, dtype="<i2").reshape(n_span_lines, n_cols, 2)
    elif len(raw) == expected_full:
        pairs = np.frombuffer(raw, dtype="<i2").reshape(
            n_span_lines, samples_per_burst, 2
        )[:, col0 : col0 + n_cols]
    else:
        reject_product(
            f"row span is {len(raw)} bytes, expected {expected_slice} "
            f"(tight columns) or {expected_full} (whole rows)"
        )
    real = pairs[..., 0].astype(np.float32)
    imag = pairs[..., 1].astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)
