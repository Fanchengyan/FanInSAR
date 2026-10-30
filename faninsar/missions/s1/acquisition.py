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
- measurement spans are whole-row blocks by default; callers slice
  columns after download.  A probe-based narrow-ROI rule (measured
  fetch bandwidth below :data:`NARROW_ROI_BANDWIDTH_BPS` *and* an ROI
  column width below :data:`NARROW_ROI_MAX_WIDTH_FRACTION` of the row)
  may instead emit exact-column per-row spans; each such span covers
  only ``[col0, col1)`` of one row and is labeled ``col-{line}-{col0}-
  {col1}`` so assembly can place the tight payload;
- any contiguous whole-row range is split into spans of at most
  :data:`MAX_SPAN_BYTES` (4 MiB) that end on row boundaries using the
  known row stride; a single row wider than the cap is the only
  exception and is emitted whole.  Large full-burst CDSE selections
  therefore split into parallel sub-spans fetched concurrently;
- identical ``(url, start, end)`` byte ranges are fetched once per
  :class:`_RoiFetchContext`; ROI windows and NESD overlap rectangles
  share the context so repeated probes and identical ranges are not
  requested twice;
- CDSE ROI/burst paths write a full-burst-height canvas then hand the
  SAFE tree to ``from_safes``.  Geometry canvases list intersecting
  bursts only.  NESD canvases list every parent burst of the swath and
  splice that frame's overlap strips.

"""

from __future__ import annotations

import asyncio
import itertools
import math
import re
import struct
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pyproj import Transformer
from pyproj.crs import CRS
from pyproj.exceptions import CRSError

from faninsar.data.query import BoundingBox, Points, Polygons
from faninsar.logging import setup_logger
from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import reject_product
from faninsar.missions.s1.io.canvas import (
    BurstFetchedWindow,
    splice_overlap_into_canvas,
    write_roi_safe_canvas,
)
from faninsar.missions.s1.overlap import burst_overlap_windows
from faninsar.processing.geometry.dem import RasterDEM
from faninsar.remote.errors import RemoteAccessError
from faninsar.remote.fetch import RangeSpan, fetch_ranges_async

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping, Sequence
    from datetime import datetime
    from typing import TypeAlias

    from faninsar.missions.s1.io.canvas import RoiSafeCanvas
    from faninsar.missions.s1.overlap import BurstOverlapWindow
    from faninsar.missions.s1.types import S1Swath

logger = setup_logger(__name__)

MAX_SPAN_BYTES = 4 * 1024 * 1024
"""Retry quantum (PROPOSAL-0061): no span exceeds 4 MiB unless one row does."""

HEADER_PROBE_BYTES = 16
"""Bytes probed at offset 0: classic (8 B) and BigTIFF (16 B) IFD pointers."""

POINT_BUFFER_M = 1000.0
"""Ground half-width applied to a point ROI before geolocation inversion."""

_WGS84 = CRS.from_epsg(4326)

_RowGroup: TypeAlias = tuple[int, int, int, int]
"""Half-open row group ``(line0, line1, byte_start, byte_end)``."""

_RowInterval: TypeAlias = tuple[int, int]
"""Half-open byte interval ``(start, end)`` of one ROI row."""

_BYTES_PER_COMPLEX_INT16 = 4
"""Storage bytes per Sentinel-1 complex-int16 sample."""

_ROW_SPAN_LABEL = re.compile(r"^row-(\d+)-(\d+)$")
"""Planner labels ``row-{line0}-{line1}`` with exclusive ``line1``."""

_COL_SPAN_LABEL = re.compile(r"^col-(\d+)-(\d+)-(\d+)$")
"""Exact-column labels ``col-{line}-{col0}-{col1}`` with exclusive ``col1``."""

NARROW_ROI_BANDWIDTH_BPS = 1.3 * 1024 * 1024
"""Probe-bandwidth ceiling (B/s) for exact-column per-row span planning."""

NARROW_ROI_MAX_WIDTH_FRACTION = 0.30
"""ROI column width below this fraction of the row is "narrow"."""

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
    exact_columns: bool = False,
) -> tuple[RangeSpan, ...]:
    """Plan bounded, row-aligned byte spans for one ROI of a strip TIFF.

    Rows are mapped to bytes through the strip tables: row ``line`` lives in
    strip ``line // rows_per_strip`` at ``strip_offsets[strip] +
    (line % rows_per_strip) * row_stride``.  The row stride is
    ``samples * bytes_per_sample``.  Labels are stable planner-generated
    ``row-{line0}-{line1}`` strings; the fetcher treats them as opaque.

    With ``exact_columns=True`` the planner instead emits one tight span per
    ROI row covering only ``[col0, col1)``, labeled
    ``col-{line}-{col0}-{col1}``.  Exact-column spans never merge across
    rows (inter-slice gaps are fetched nowhere), so they trade request
    count for bytes saved; callers enable them only when the probe-based
    narrow-ROI rule says the saved bytes beat the per-request cost.

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
    exact_columns : bool, optional
        True to emit one tight ``[col0, col1)`` span per ROI row instead
        of whole-row blocks (default False; used by the probe-based
        narrow-ROI rule).

    Returns
    -------
    tuple of RangeSpan
        Bounded whole-row spans whose union covers every ROI row.  Each span
        is at most :data:`MAX_SPAN_BYTES`, ends on a row boundary, and
        consecutive spans stay byte-adjacent within a group.  Column bounds
        are validated here and sliced after download; they do not trim the
        HTTP range.  With ``exact_columns=True`` the return is one
        ``col-{line}-{col0}-{col1}`` span per ROI row instead.

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

    intervals: list[_RowInterval] = [
        _row_byte_interval(
            line,
            strip_offsets=strip_offsets,
            rows_per_strip=rows_per_strip,
            row_stride=row_stride,
        )
        for line in range(line0, line1)
    ]
    if exact_columns:
        return _exact_column_spans(
            url,
            intervals,
            line0=line0,
            col0=col0,
            col1=col1,
            bytes_per_sample=bytes_per_sample,
        )
    spans: list[RangeSpan] = []
    for group in _merge_row_groups(intervals, line0):
        spans.extend(_split_group(url, group, intervals, line0))
    return tuple(spans)


def _exact_column_spans(
    url: str,
    intervals: Sequence[_RowInterval],
    *,
    line0: int,
    col0: int,
    col1: int,
    bytes_per_sample: int,
) -> tuple[RangeSpan, ...]:
    """Emit one tight ``[col0, col1)`` span per ROI row.

    Each span is bounded by strip-table arithmetic (the row interval) plus
    the validated column slice, so no span ever escapes the strip tables.
    Spans are never merged across rows: the inter-slice gap is bytes the
    narrow-ROI rule decided not to fetch.

    Parameters
    ----------
    url : str
        Object URL each span requests.
    intervals : sequence of _RowInterval
        Whole-row byte interval of every ROI row, indexed from ``line0``.
    line0 : int
        First ROI row index (labels are row-based).
    col0, col1 : int
        ROI column bounds; ``col1`` is exclusive.
    bytes_per_sample : int
        Storage bytes per sample.

    Returns
    -------
    tuple of RangeSpan
        One span per ROI row labeled ``col-{line}-{col0}-{col1}``.

    """
    spans: list[RangeSpan] = []
    for index, (row_start, _row_end) in enumerate(intervals):
        start = row_start + col0 * bytes_per_sample
        end = row_start + col1 * bytes_per_sample - 1
        spans.append(RangeSpan(url, start, end, f"col-{line0 + index}-{col0}-{col1}"))
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
) -> _RowInterval:
    """Return the half-open whole-row byte interval of one ROI row.

    Parameters
    ----------
    line : int
        ROI row index.
    strip_offsets : sequence of int
        TIFF ``StripOffsets``.
    rows_per_strip, row_stride : int
        Rows per strip and bytes per row.

    Returns
    -------
    _RowInterval
        Half-open ``(start, end)`` covering the full row stride.

    """
    row_start = (
        strip_offsets[line // rows_per_strip] + (line % rows_per_strip) * row_stride
    )
    return row_start, row_start + row_stride


def _merge_row_groups(
    intervals: Sequence[_RowInterval],
    line0: int,
) -> list[_RowGroup]:
    """Coalesce per-row intervals into byte ranges.

    Byte-adjacent whole rows continue the current group.  A gap between
    rows starts a new range so each span stays a contiguous byte interval.

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
        if group_end == start:
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
_MAX_IFD_PROBE_BYTES = 512 * 1024
_CDSE_RANGE_HOSTS = frozenset(
    {
        "zipper.dataspace.copernicus.eu",
        "download.dataspace.copernicus.eu",
    }
)


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
        Local channels kept as measurement bindings.  Default ``("VV",)``
        retains every IW swath in VV; ``"all"`` keeps every OData pair.
    **kwargs : Any
        Advanced search options forwarded to :func:`faninsar.remote.search`
        (``limit``, ``budget``, ``filters``).  ``catalog`` and
        ``auth_profile`` default to the internal CDSE binding.

    Returns
    -------
    list of dict
        One binding per measurement with ``product_id``, ``annotation_url``,
        ``measurement_url``, ``measurement_size``, ``annotation_size``,
        ``polarization``, and ``swath``.

    """
    from faninsar import remote

    catalog = str(kwargs.pop("catalog", _CDSE_CATALOG))
    auth_profile = str(kwargs.pop("auth_profile", _CDSE_AUTH_PROFILE))
    kwargs.pop("collections", None)
    if catalog == _CDSE_CATALOG:
        from faninsar.remote.providers.cdse import ensure_cdse_catalog

        ensure_cdse_catalog()
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
                f"{catalog!r}; cannot discover Sentinel-1 CDSE products. "
                'Use granularity="burst", provider="asf" for ASF burst products.'
            )
            logger.exception(message)
            reason = "unknown_catalog"
            raise RemoteAccessError(reason, message) from exc
        raise
    requested = (
        None
        if polarizations == "all"
        else {str(value).upper() for value in polarizations}
    )
    bindings: list[dict[str, Any]] = []
    for item in items:
        bindings.extend(_bind_cdse_item(item))
    if requested is None:
        return bindings
    return [
        binding
        for binding in bindings
        if str(binding.get("polarization", "")).upper() in requested
    ]


def windows_from_cdse_fetch(
    item: Mapping[str, Any],
    roi: BoundingBox | Points | Polygons,
    *,
    xml_text: str,
    granularity: str = "roi",
    fetch_context: _RoiFetchContext | None = None,
    dem: RasterDEM | None = None,
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
    fetch_context : _RoiFetchContext or None, optional
        Shared per-measurement fetch context.  When supplied by
        :func:`materialize_cdse_roi_safes`, probe results and identical
        byte ranges are shared with later NESD overlap fetches.
    dem : RasterDEM or None, optional
        DEM used by the multi-polygon geo2rdr inversion.  Required when
        ``roi`` is a multi-desired-polygon :class:`Polygons`.
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
    if (
        isinstance(roi, Polygons)
        and granularity == "roi"
        and len(roi.to_desired().geodataframe) > 1
    ):
        if dem is None:
            reject_product("multi-polygon ROI requires a DEM for per-polygon geo2rdr")
        envelopes = _polygon_envelopes(roi, swath, dem)
        windows = _windows_for_polygons(
            swath,
            envelopes,
            url=str(item.get("measurement_url", "")),
            file_size=int(item.get("measurement_size") or 0),
            fetch_context=fetch_context,
        )
        return swath, windows, xml_text

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
        fetch_context=fetch_context,
    )
    return swath, windows, xml_text


def overlap_windows_for_canvas(
    swath: S1Swath,
    windows: Sequence[BurstFetchedWindow],
    canvas: RoiSafeCanvas | None = None,
    *,
    listed_indices: Sequence[int] | None = None,
) -> tuple[BurstOverlapWindow, ...]:
    """Return parent overlap rectangles among listed canvas bursts.

    Completeness is the parent :func:`burst_overlap_windows` rectangle,
    not a tight azimuth box.  A synthetic parent with no sensing-time
    overlap yields an empty tuple.  Geometry canvases pass the fetched
    window set; NESD canvases pass every parent burst of the swath.

    """
    del canvas
    kept = (
        {int(index) for index in listed_indices}
        if listed_indices is not None
        else {int(window.burst_index) for window in windows}
    )
    if len(kept) < 2:
        return ()
    return burst_overlap_windows(swath, burst_indices=sorted(kept), min_lines=1)


def _is_wgs84(crs: object | None) -> bool:
    """Return True when ``crs`` is geographic WGS 84 / EPSG:4326."""
    if crs is None:
        return False
    try:
        parsed = CRS.from_user_input(crs)
    except (CRSError, TypeError, ValueError):
        return False
    epsg = parsed.to_epsg()
    if epsg == 4326:
        return True
    return parsed.equals(_WGS84)


def normalize_roi_wgs84(
    roi: BoundingBox | Points | Polygons,
) -> BoundingBox | Points | Polygons:
    """Return ``roi`` in EPSG:4326, preserving its query type.

    Parameters
    ----------
    roi : BoundingBox, Points, or Polygons
        FanInSAR spatial query.  A CRS is required.

    Returns
    -------
    BoundingBox, Points, or Polygons
        The same type, with coordinates in EPSG:4326.

    Raises
    ------
    ValueError
        If ``roi`` has no CRS, or the CRS cannot be transformed to
        EPSG:4326.

    """
    crs = getattr(roi, "crs", None)
    if crs is None:
        message = "ROI must declare a CRS; spatial queries cannot assume lon/lat"
        logger.error(message)
        raise ValueError(message)
    if _is_wgs84(crs):
        return roi
    try:
        return roi.to_crs(_WGS84)
    except (CRSError, TypeError, ValueError) as exc:
        message = "ROI CRS cannot be transformed to EPSG:4326"
        logger.exception(message)
        raise ValueError(message) from exc


def _utm_ups_epsg(lon: float, lat: float) -> str:
    """Return the UTM or UPS EPSG code for a WGS84 point."""
    if lat >= 84:
        return "EPSG:32661"
    if lat < -80:
        return "EPSG:32761"
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def _point_buffer_bounds(lon: float, lat: float) -> tuple[float, float, float, float]:
    """Return a 1 km ground square around a WGS84 point as lon/lat bounds."""
    projected = CRS.from_user_input(_utm_ups_epsg(lon, lat))
    to_xy = Transformer.from_crs(_WGS84, projected, always_xy=True)
    to_ll = Transformer.from_crs(projected, _WGS84, always_xy=True)
    x, y = to_xy.transform(lon, lat)
    corners = (
        (x - POINT_BUFFER_M, y - POINT_BUFFER_M),
        (x - POINT_BUFFER_M, y + POINT_BUFFER_M),
        (x + POINT_BUFFER_M, y - POINT_BUFFER_M),
        (x + POINT_BUFFER_M, y + POINT_BUFFER_M),
    )
    lons: list[float] = []
    lats: list[float] = []
    for easting, northing in corners:
        corner_lon, corner_lat = to_ll.transform(easting, northing)
        lons.append(float(corner_lon))
        lats.append(float(corner_lat))
    return min(lons), min(lats), max(lons), max(lats)


def inversion_envelope(
    roi: BoundingBox | Points | Polygons,
) -> tuple[float, float, float, float]:
    """Return a WGS84 ``(west, south, east, north)`` with positive area.

    ``BoundingBox`` uses its four sides after CRS normalization.
    ``Polygons`` uses the desired-minus-undesired bounds.
    ``Points`` are expanded by :data:`POINT_BUFFER_M` on the local
    UTM/UPS plane so a GPS pin is not inverted as a 1x1 radar pixel.

    Parameters
    ----------
    roi : BoundingBox, Points, or Polygons
        Spatial query.  A CRS is required.

    Returns
    -------
    tuple of float
        Inclusive-exclusive geographic envelope in EPSG:4326.

    Raises
    ------
    ValueError
        If ``roi`` has no CRS or cannot be transformed to EPSG:4326.

    """
    normalized = normalize_roi_wgs84(roi)
    if isinstance(normalized, Points):
        envelopes = [
            _point_buffer_bounds(float(x), float(y)) for x, y in normalized.values
        ]
        if not envelopes:
            message = "Points ROI is empty"
            logger.error(message)
            raise ValueError(message)
        west = min(item[0] for item in envelopes)
        south = min(item[1] for item in envelopes)
        east = max(item[2] for item in envelopes)
        north = max(item[3] for item in envelopes)
        return west, south, east, north
    if isinstance(normalized, Polygons):
        geometry = (
            normalized.to_desired() if normalized.is_mixed else normalized.desired
        )
        if len(geometry) == 0:
            message = "Polygons ROI has no desired geometry"
            logger.error(message)
            raise ValueError(message)
        bounds = geometry.to_geodataframe().total_bounds
        return float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
    return (
        float(normalized.left),
        float(normalized.bottom),
        float(normalized.right),
        float(normalized.top),
    )


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
        Spatial query.  Coordinates are transformed to EPSG:4326.
        A :class:`~faninsar.data.query.Points` query is buffered on the
        ground before inversion; boxes and polygons use their bounds.

    Returns
    -------
    tuple of int
        Half-open ``(line0, line1, col0, col1)`` covering the ROI in
        parent-swath coordinates.

    Raises
    ------
    Sentinel1ProductError
        If the annotation has no geolocation grid.
    ValueError
        If ``roi`` has no CRS.

    """
    points = _geolocation_points(xml_text)
    if not points:
        reject_product("annotation has no geolocationGrid points to invert")
    left, bottom, right, top = inversion_envelope(roi)
    corners = (
        (left, bottom),
        (left, top),
        (right, bottom),
        (right, top),
    )
    grid = _GeolocationGrid.from_points(points)
    lines: list[float] = []
    pixels: list[float] = []
    for lon, lat in corners:
        line, pixel = grid.invert(lat, lon)
        lines.append(line)
        pixels.append(pixel)
    line0 = math.floor(min(lines))
    line1 = math.ceil(max(lines)) + 1
    col0 = math.floor(min(pixels))
    col1 = math.ceil(max(pixels)) + 1
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
    coreg_mode: Literal["geometry", "nesd"] = "geometry",
    dem: RasterDEM | None = None,
    items: Sequence[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[Path]:
    """Materialize CDSE ROI/burst SAFE-subset canvases under ``work_dir``.

    Order (PROPOSAL-0061): fetch annotation ranges, assemble windows,
    :func:`write_roi_safe_canvas`, optionally splice parent overlap
    rectangles, then return the ``.SAFE`` roots for ``from_safes``.
    Geometry canvases keep intersecting bursts and skip overlap fetch.
    NESD canvases list every parent burst of the swath and splice that
    frame's overlap strips.

    Parameters
    ----------
    roi, datetime_range, polarizations
        Search envelope forwarded to :func:`discover_cdse_slc_items`.
    work_dir : path-like
        Stack work directory; canvases are written under ``inputs/``.
    granularity : {"roi", "burst"}, optional
        CDSE acquisition granularity.
    coreg_mode : {"geometry", "nesd"}, optional
        Canvas listing and overlap policy.  Default ``"geometry"``.
    items : sequence of mapping, optional
        Pre-bound measurement dicts (as produced by ``_bind_cdse_item``).
        When given, the internal ``discover_cdse_slc_items`` search is
        skipped and these bindings are materialized directly — the
        ``from_search`` path.
    dem : RasterDEM, optional
        DEM used for per-polygon geo2rdr inversion when ``roi`` holds
        multiple desired polygons.  Fetched automatically when omitted.
    **kwargs : Any
        Forwarded to discovery (ignored when ``items`` is given).

    Returns
    -------
    list of pathlib.Path
        Generated ``.SAFE`` directory roots.

    Raises
    ------
    ValueError
        If ``roi`` has no CRS, ``coreg_mode`` is unknown, or discovery
        returns no products.

    """
    if coreg_mode not in {"geometry", "nesd"}:
        message = f"coreg_mode must be 'geometry' or 'nesd'; got {coreg_mode!r}"
        logger.error(message)
        raise ValueError(message)
    roi = normalize_roi_wgs84(roi)
    # Multi-polygon ROI needs DEM for per-polygon geo2rdr inversion.
    if (
        isinstance(roi, Polygons)
        and granularity == "roi"
        and len(roi.to_desired().geodataframe) > 1
        and dem is None
    ):
        dem = _fetch_dem_for_polygons(roi, work_dir=work_dir)
    if items is None:
        items = discover_cdse_slc_items(
            roi,
            datetime_range=datetime_range,
            polarizations=polarizations,
            **kwargs,
        )
    else:
        requested = (
            None if polarizations == "all" else {p.upper() for p in polarizations}
        )
        items = [
            binding
            for binding in items
            if requested is None or binding.get("polarization") in requested
        ]
    if not items:
        message = "Sentinel-1 CDSE search returned no products"
        logger.error(message)
        raise ValueError(message)
    out_dir = Path(work_dir) / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for item in items:
        fetch_context = _RoiFetchContext()
        xml_text = _fetch_annotation_xml(item, fetch_context=fetch_context)
        if not _roi_overlaps_geolocation(xml_text, roi):
            logger.warning(
                "Skipping CDSE measurement %s %s %s: "
                "ROI is outside the geolocation envelope",
                item.get("product_id", "?"),
                item.get("swath", "?"),
                item.get("polarization", "?"),
            )
            continue
        swath, windows, xml_text = windows_from_cdse_fetch(
            item,
            roi,
            xml_text=xml_text,
            granularity=granularity,
            fetch_context=fetch_context,
            dem=dem,
        )
        if not windows:
            continue
        product_id = str(item.get("product_id", "S1_ROI"))
        swath_name = str(item.get("swath", "IW1"))
        polarization = str(item.get("polarization", "VV"))
        listed_indices = (
            tuple(int(burst.index) for burst in swath.bursts)
            if coreg_mode == "nesd"
            else None
        )
        canvas = write_roi_safe_canvas(
            swath,
            windows,
            out_dir,
            annotation_xml=xml_text,
            product_id=f"{product_id}-{swath_name}-{polarization}",
            listed_indices=listed_indices,
        )
        if coreg_mode == "nesd":
            by_index = {int(window.burst_index): window for window in windows}
            overlaps = overlap_windows_for_canvas(
                swath,
                windows,
                canvas,
                listed_indices=listed_indices,
            )
            rectangles = _fetch_overlap_rectangles(
                item,
                swath,
                overlaps,
                by_index=by_index,
                fetch_context=fetch_context,
            )
            for overlap, (top, bottom) in zip(overlaps, rectangles, strict=True):
                splice_overlap_into_canvas(canvas, overlap, role="top", samples=top)
                splice_overlap_into_canvas(
                    canvas, overlap, role="bottom", samples=bottom
                )
        paths.append(canvas.safe_root)
    if not paths:
        message = "Sentinel-1 CDSE search returned no products intersecting the ROI"
        logger.error(message)
        raise ValueError(message)
    return paths


def _fetch_dem_for_polygons(
    roi: Polygons,
    work_dir: str | Path,
) -> RasterDEM:
    """Fetch a Copernicus DEM mosaic covering all desired polygons.

    Downloads GLO-30 tiles for the union bounding box and writes a merged
    GeoTIFF under ``work_dir/dem/``.  The result is persisted so a later
    ``prepare`` pass can reuse it without re-downloading.
    """
    from faninsar.processing.geometry.dem_manager import DEMManager

    gdf = roi.to_desired().geodataframe.to_crs(4326)
    bounds = gdf.total_bounds
    out_dir = Path(work_dir) / "dem"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rgv_stack.tif"
    mgr = DEMManager(cache_dir=Path(work_dir) / "dem-cache")
    mgr.fetch_dem(bounds=tuple(bounds), output_path=out_path)
    return RasterDEM(path=out_path)


def _fetch_annotation_xml(
    item: Mapping[str, Any],
    *,
    fetch_context: _RoiFetchContext | None = None,
) -> str:
    """Fetch one annotation XML document via the async range engine.

    The annotation body (up to ~2 MiB) is the largest probe-stage
    download, so when ``fetch_context`` is given its elapsed time feeds
    the observed bandwidth used by the narrow-ROI rule.
    """
    url = str(item.get("annotation_url", ""))
    size = int(item.get("annotation_size") or _ANNOTATION_PROBE_BYTES)
    span = RangeSpan(url, 0, max(size - 1, 0), "annotation")
    started = time.monotonic()
    bodies = _fetch_cdse_ranges((span,))
    if fetch_context is not None:
        fetch_context.record_probe(
            bytes_fetched=span.end - span.start + 1,
            seconds=time.monotonic() - started,
        )
    return bodies["annotation"].decode("utf-8")


def _bind_cdse_item(item: Any) -> list[dict[str, Any]]:
    """Bind one catalog item to OData annotation and measurement node URLs."""
    assets = getattr(item, "assets", {}) or {}
    measurements: dict[str, Any] = {}
    annotations: dict[str, Any] = {}
    fallback_annotation = None
    fallback_measurement = None
    for asset in assets.values():
        href = str(getattr(asset, "href", ""))
        key = str(getattr(asset, "key", "")).casefold()
        media = str(getattr(asset, "media_type", "") or "").casefold()
        is_annotation = (
            "annotation" in key or href.casefold().endswith(".xml") or "xml" in media
        )
        is_measurement = (
            "measurement" in key
            or href.casefold().endswith((".tif", ".tiff"))
            or "tiff" in media
        )
        if is_annotation:
            suffix = _asset_pair_suffix(key, "annotation")
            if suffix:
                annotations[suffix] = asset
            elif fallback_annotation is None:
                fallback_annotation = asset
        if is_measurement:
            suffix = _asset_pair_suffix(key, "measurement")
            if suffix:
                measurements[suffix] = asset
            elif fallback_measurement is None:
                fallback_measurement = asset
    suffixes = sorted(set(measurements) & set(annotations))
    if (
        not suffixes
        and fallback_annotation is not None
        and fallback_measurement is not None
    ):
        suffixes = [""]
        annotations[""] = fallback_annotation
        measurements[""] = fallback_measurement
    if not suffixes:
        logger.warning(
            "CDSE item %s is missing annotation or measurement OData node",
            getattr(item, "item_id", "?"),
        )
        return []
    metadata = (
        getattr(item, "raw_metadata", {}) if hasattr(item, "raw_metadata") else {}
    )
    bindings: list[dict[str, Any]] = []
    for suffix in suffixes:
        annotation = annotations[suffix]
        measurement = measurements[suffix]
        annotation_url = str(annotation.href)
        measurement_url = str(measurement.href)
        _require_odata_value_href(annotation_url, role="annotation")
        _require_odata_value_href(measurement_url, role="measurement")
        properties = dict(getattr(measurement, "properties", {}) or {})
        size = getattr(measurement, "size_bytes", None) or properties.get("file:size")
        annotation_properties = dict(getattr(annotation, "properties", {}) or {})
        annotation_size = getattr(annotation, "size_bytes", None) or (
            annotation_properties.get("file:size")
        )
        fallback_pol = (
            suffix.rsplit("-", 1)[-1] if suffix else metadata.get("polarization", "VV")
        )
        polarization = str(properties.get("polarization") or fallback_pol).upper()
        swath = str(properties.get("swath") or _swath_from_pair_suffix(suffix))
        bindings.append(
            {
                "product_id": str(getattr(item, "item_id", "S1_ROI")),
                "annotation_url": annotation_url,
                "measurement_url": measurement_url,
                "measurement_size": int(size or 0),
                "annotation_size": int(annotation_size or 0),
                "polarization": polarization,
                "swath": swath,
            }
        )
    return bindings


def _asset_pair_suffix(key: str, role: str) -> str:
    """Return the swath-pol suffix of ``measurement-iw1-vv`` / ``annotation-iw1-vv``."""
    prefix = f"{role}-"
    if key.startswith(prefix):
        return key[len(prefix) :]
    return ""


def _swath_from_pair_suffix(suffix: str) -> str:
    """Return ``IW1`` from a pair suffix such as ``iw1-vv``."""
    prefix = suffix.split("-", 1)[0]
    return prefix.upper() if prefix.startswith("iw") else "IW1"


def _lines_per_burst(xml_text: str) -> int | None:
    """Return the annotation ``linesPerBurst``, or None when absent/invalid."""
    root = ET.fromstring(xml_text)
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "linesPerBurst":
            continue
        try:
            value = int((element.text or "").strip())
        except ValueError:
            return None
        return value if value > 0 else None
    return None


def _roi_overlaps_geolocation(
    xml_text: str,
    roi: BoundingBox | Points | Polygons,
) -> bool:
    """Return True when the ROI envelope overlaps a burst geolocation bbox.

    The whole-strip geolocation bbox over-covers a diagonal Sentinel-1
    strip, so the envelope is tested against each burst's own bbox,
    grouped by ``linesPerBurst``; annotations without burst lengths keep
    the conservative whole-strip bbox.
    """
    points = _geolocation_points(xml_text)
    if not points:
        return False
    roi_west, roi_south, roi_east, roi_north = inversion_envelope(roi)
    lines_per_burst = _lines_per_burst(xml_text)
    first_line = min(point[0] for point in points)
    groups: dict[int, list[tuple[int, int, float, float]]] = {}
    for point in points:
        key = (
            0 if lines_per_burst is None else (point[0] - first_line) // lines_per_burst
        )
        groups.setdefault(key, []).append(point)
    for burst_points in groups.values():
        west = min(point[3] for point in burst_points)
        east = max(point[3] for point in burst_points)
        south = min(point[2] for point in burst_points)
        north = max(point[2] for point in burst_points)
        if not (
            roi_east < west or roi_west > east or roi_north < south or roi_south > north
        ):
            return True
    return False


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


class _GeolocationGrid:
    """Bilinear geolocation grid used to invert a WGS84 ROI to radar lines."""

    def __init__(
        self,
        lines: Sequence[float],
        pixels: Sequence[float],
        table: Mapping[tuple[float, float], tuple[float, float]],
    ) -> None:
        self.lines = list(lines)
        self.pixels = list(pixels)
        self.table = dict(table)

    @classmethod
    def from_points(
        cls, points: Sequence[tuple[int, int, float, float]]
    ) -> _GeolocationGrid:
        """Build a regular (line, pixel) table from annotation grid points."""
        table: dict[tuple[float, float], tuple[float, float]] = {}
        lines: set[float] = set()
        pixels: set[float] = set()
        for line, pixel, lat, lon in points:
            key = (float(line), float(pixel))
            table[key] = (float(lat), float(lon))
            lines.add(float(line))
            pixels.add(float(pixel))
        return cls(sorted(lines), sorted(pixels), table)

    def _bilinear(self, line: float, pixel: float) -> tuple[float, float]:
        """Interpolate (lat, lon) at one radar coordinate."""
        lines, pixels, table = self.lines, self.pixels, self.table
        line0 = max((value for value in lines if value <= line), default=lines[0])
        line1 = min((value for value in lines if value >= line), default=lines[-1])
        pixel0 = max((value for value in pixels if value <= pixel), default=pixels[0])
        pixel1 = min((value for value in pixels if value >= pixel), default=pixels[-1])
        t_line = (line - line0) / (line1 - line0) if line1 > line0 else 0.0
        t_pixel = (pixel - pixel0) / (pixel1 - pixel0) if pixel1 > pixel0 else 0.0
        a = table[(line0, pixel0)]
        b = table[(line0, pixel1)]
        c = table[(line1, pixel0)]
        d = table[(line1, pixel1)]
        lat = (a[0] * (1.0 - t_pixel) + b[0] * t_pixel) * (1.0 - t_line) + (
            c[0] * (1.0 - t_pixel) + d[0] * t_pixel
        ) * t_line
        lon = (a[1] * (1.0 - t_pixel) + b[1] * t_pixel) * (1.0 - t_line) + (
            c[1] * (1.0 - t_pixel) + d[1] * t_pixel
        ) * t_line
        return lat, lon

    def invert(self, lat: float, lon: float) -> tuple[float, float]:
        """Coarse-to-fine search from (lat, lon) to (line, pixel)."""
        lines, pixels = self.lines, self.pixels
        line_pts = [
            lines[0],
            *[(a + b) / 2.0 for a, b in itertools.pairwise(lines)],
            lines[-1],
        ]
        pixel_pts = [
            pixels[0],
            *[(a + b) / 2.0 for a, b in itertools.pairwise(pixels)],
            pixels[-1],
        ]
        best: tuple[float, float, float] | None = None
        for line in line_pts:
            for pixel in pixel_pts:
                sample_lat, sample_lon = self._bilinear(line, pixel)
                distance = (sample_lat - lat) ** 2 + ((sample_lon - lon) * 0.7) ** 2
                if best is None or distance < best[0]:
                    best = (distance, line, pixel)
        if best is None:
            reject_product("geolocation grid inversion found no candidate")
        _, line, pixel = best
        for _step in range(8):
            step_line = max(1.0, (lines[-1] - lines[0]) / 200.0)
            step_pixel = max(1.0, (pixels[-1] - pixels[0]) / 300.0)
            candidate = best
            for delta_line in (-step_line, 0.0, step_line):
                for delta_pixel in (-step_pixel, 0.0, step_pixel):
                    trial_line = min(max(line + delta_line, lines[0]), lines[-1])
                    trial_pixel = min(max(pixel + delta_pixel, pixels[0]), pixels[-1])
                    sample_lat, sample_lon = self._bilinear(trial_line, trial_pixel)
                    distance = (sample_lat - lat) ** 2 + ((sample_lon - lon) * 0.7) ** 2
                    if distance < candidate[0]:
                        candidate = (distance, trial_line, trial_pixel)
            best = candidate
            _, line, pixel = best
        return line, pixel


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


class _RoiFetchContext:
    """Per-measurement fetch context shared by ROI and overlap planning.

    A context is created once per measurement item and threaded through
    ``windows_from_cdse_fetch`` and the NESD overlap fetches so:

    - IFD probe results (strip tables) are fetched once per
      ``(url, file_size, samples)`` instead of once per rectangle;
    - identical ``(url, start, end)`` byte ranges are requested once and
      reused, whichever label asked for them;
    - the observed fetch bandwidth (bytes per second across every range
      batch issued through the context) feeds the probe-based narrow-ROI
      rule in :func:`_fetch_measurement_bodies`.
    """

    def __init__(self) -> None:
        self.strip_tables: dict[
            tuple[str, int, int], tuple[list[int], list[int], int]
        ] = {}
        self.bodies: dict[tuple[str, int, int], bytes] = {}
        self.probed_bytes = 0
        self.probed_seconds = 0.0
        # NESD overlap rectangles fetch in a bounded thread pool; every
        # cache/counter mutation goes through this lock.
        self.lock = threading.Lock()

    @property
    def bandwidth_bps(self) -> float | None:
        """Observed bytes/second across all range fetches in this context."""
        if self.probed_seconds <= 0.0:
            return None
        return self.probed_bytes / self.probed_seconds

    def record_probe(self, *, bytes_fetched: int, seconds: float) -> None:
        """Accumulate one probe batch into the bandwidth meter."""
        with self.lock:
            self.probed_seconds += seconds
            self.probed_bytes += bytes_fetched


def _polygon_envelopes(
    polygons: Polygons,
    swath: S1Swath,
    dem: RasterDEM,
) -> list[tuple[int, int, int, int]]:
    """Invert each desired polygon to a per-polygon (line0, line1, col0, col1).

    Each polygon's WGS84 bounding box is sampled on an adaptive N x N grid
    (N from :func:`_geo2rdr_grid_size`, 4x4 to 10x10 log-interpolated by
    polygon size).  DEM heights are sampled at the grid points and
    ``run_geo2rdr`` solves for (line, pixel) per point.  The polygon's
    envelope is the bounding box of converged solutions.

    Returns
    -------
    list of tuple of int
        One ``(line0, line1, col0, col1)`` per desired polygon, clamped to
        the swath extent.  Polygons with no converged geo2rdr points are
        skipped.

    """
    from faninsar.missions.s1.processing import _radar_model
    from faninsar.processing.geometry.prepare_production import run_geo2rdr

    gdf = polygons.to_desired().geodataframe.to_crs(4326)
    if gdf.empty:
        return []

    burst = swath.bursts[0]
    model = _radar_model(
        swath,
        burst,
        shape=(int(swath.lines_per_burst), int(swath.samples_per_burst)),
        row0=0,
        col0=0,
    )
    total_lines = int(swath.lines)
    total_samples = int(swath.samples_per_burst)
    envelopes: list[tuple[int, int, int, int]] = []
    for geom in gdf.geometry:
        minx, miny, maxx, maxy = geom.bounds
        n = _geo2rdr_grid_size(maxx - minx, maxy - miny)
        xs = np.linspace(minx, maxx, n)
        ys = np.linspace(miny, maxy, n)
        lons, lats = np.meshgrid(xs, ys)
        lons = lons.ravel()
        lats = lats.ravel()
        heights = np.asarray(dem.sample(lats, lons), dtype=np.float64)
        heights = np.where(np.isnan(heights), 0.0, heights)
        result = run_geo2rdr(model, lats, lons, heights, device="cpu")
        az = np.asarray(result.azimuth_index).ravel()
        rg = np.asarray(result.range_index).ravel()
        conv = np.asarray(result.converged).ravel().astype(bool)
        if not conv.any():
            continue
        az_ok = az[conv]
        rg_ok = rg[conv]
        line0 = max(0, math.floor(az_ok.min()))
        line1 = min(total_lines, math.ceil(az_ok.max()) + 1)
        col0 = max(0, math.floor(rg_ok.min()))
        col1 = min(total_samples, math.ceil(rg_ok.max()) + 1)
        if line1 > line0 and col1 > col0:
            envelopes.append((line0, line1, col0, col1))
    return envelopes


def _geo2rdr_grid_size(bbox_w_deg: float, bbox_h_deg: float) -> int:
    """Return adaptive geo2rdr sampling grid size 4..10 via log interpolation.

    Small polygons (<0.1 km) get 4x4; large ones (>=20 km) cap at 10x10.
    The mapping is logarithmic so small polygons get denser sampling than
    a linear ramp would give them.
    """
    # Approximate km at mid-latitudes (RGV is ~46 deg N).
    w_km = bbox_w_deg * 111.0 * 0.7
    h_km = bbox_h_deg * 111.0
    max_km = max(w_km, h_km)
    if max_km <= 0.1:
        return 4
    if max_km >= 20.0:
        return 10
    t = math.log(max_km / 0.1) / math.log(20.0 / 0.1)
    return round(4 + t * 6)


def _merge_line_segments(
    segments: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping line segments into disjoint union intervals.

    Parameters
    ----------
    segments : sequence of (line0, line1)
        Per-polygon row ranges (half-open).

    Returns
    -------
    list of (line0, line1)
        Sorted, non-overlapping merged intervals.

    """
    if not segments:
        return []
    sorted_segs = sorted(segments)
    merged: list[list[int]] = [list(sorted_segs[0])]
    for lo, hi in sorted_segs[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(seg[0], seg[1]) for seg in merged]


def _windows_for_polygons(
    swath: S1Swath,
    envelopes: Sequence[tuple[int, int, int, int]],
    *,
    url: str,
    file_size: int,
    fetch_context: _RoiFetchContext | None = None,
) -> list[BurstFetchedWindow]:
    """Build per-burst windows for multiple polygon envelopes.

    Line segments from all envelopes are unioned into disjoint ranges,
    each fetched once via the range planner.  Small polygons whose column
    width falls below the narrow-ROI rule use exact-column spans; larger
    ones fetch whole rows.  Envelopes on the same burst whose column
    ranges overlap are merged into one window so samples do not
    overwrite each other.
    """
    if not envelopes:
        return []
    lpb = int(swath.lines_per_burst)
    samples_per_burst = int(swath.samples_per_burst)

    # Drop envelopes that intersect no burst before any fetch.
    burst_line_ranges = [
        (int(burst.index) * lpb, (int(burst.index) + 1) * lpb) for burst in swath.bursts
    ]
    envelopes = [
        env
        for env in envelopes
        if any(env[0] < b1 and env[1] > b0 for b0, b1 in burst_line_ranges)
    ]
    if not envelopes:
        reject_product("no polygon envelope intersects any Sentinel-1 burst")

    # 1. Group envelopes by burst index; each burst's union envelope
    #    decides whether its rows fetch whole or exact-column spans.
    burst_groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for env in envelopes:
        for burst in swath.bursts:
            burst_line0 = int(burst.index) * lpb
            burst_line1 = burst_line0 + lpb
            if env[0] < burst_line1 and env[1] > burst_line0:
                burst_groups.setdefault(int(burst.index), []).append(env)

    all_bodies: dict[str, bytes] = {}
    for burst_envs in burst_groups.values():
        line0 = min(env[0] for env in burst_envs)
        line1 = max(env[1] for env in burst_envs)
        col0 = min(env[2] for env in burst_envs)
        col1 = max(env[3] for env in burst_envs)
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
            fetch_context=fetch_context,
        )
        all_bodies.update(bodies)

    # 3. Merge envelopes on the same burst whose column ranges overlap,
    #    then slice each merged window's columns from the fetched rows.
    windows: list[BurstFetchedWindow] = []
    for burst in swath.bursts:
        burst_line0 = burst.index * lpb
        burst_line1 = burst_line0 + lpb
        burst_envs = [
            env for env in envelopes if env[0] < burst_line1 and env[1] > burst_line0
        ]
        if not burst_envs:
            continue
        # All envelopes on this burst merge into one window covering the
        # union rectangle (write_roi_safe_canvas allows one window per
        # burst index).
        merged = _merge_burst_windows(burst_envs)
        for line0, line1, env_col0, env_col1 in merged:
            col0 = max(0, env_col0)
            col1 = min(samples_per_burst, max(col0 + 1, env_col1))
            inter0 = max(line0, burst_line0)
            inter1 = min(line1, burst_line1)
            row0 = inter0 - burst_line0
            n_lines = inter1 - inter0
            n_cols = col1 - col0
            samples = _assemble_window_samples(
                all_bodies,
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
        reject_product("no polygon envelope intersects any Sentinel-1 burst")
    return windows


def _merge_burst_windows(
    envelopes: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge all envelopes on one burst into a single union window.

    ``write_roi_safe_canvas`` accepts at most one ``BurstFetchedWindow``
    per burst index, so every polygon covering the same burst is merged
    into one rectangle spanning the union of their line and column
    ranges.
    """
    if not envelopes:
        return []
    line0 = min(env[0] for env in envelopes)
    line1 = max(env[1] for env in envelopes)
    col0 = min(env[2] for env in envelopes)
    col1 = max(env[3] for env in envelopes)
    return [(line0, line1, col0, col1)]


def _windows_for_lines(
    swath: S1Swath,
    *,
    line0: int,
    line1: int,
    col0: int,
    col1: int,
    url: str,
    file_size: int,
    fetch_context: _RoiFetchContext | None = None,
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
        fetch_context=fetch_context,
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
    fetch_context: _RoiFetchContext | None = None,
) -> dict[str, bytes]:
    """Fetch IFD probes then bounded measurement spans for one ROI window.

    With ``fetch_context`` the strip tables come from the context cache
    when an earlier rectangle of the same measurement already probed
    them, measurement spans identical to an earlier request are not
    re-requested, and the probe fetch feeds the context bandwidth used
    by the narrow-ROI rule.  Without a context every call probes and
    fetches independently, exactly as before.
    """
    if not url:
        reject_product("CDSE measurement fetch requires a non-empty object URL")
    cached = _strip_tables_for(
        url, file_size=file_size, samples=samples, fetch_context=fetch_context
    )
    offsets, counts, rows_per_strip = cached
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
        exact_columns=_use_exact_columns(
            fetch_context,
            n_cols=col1 - col0,
            samples=samples,
        ),
    )
    return _fetch_deduped(spans, fetch_context)


def _strip_tables_for(
    url: str,
    *,
    file_size: int,
    samples: int,
    fetch_context: _RoiFetchContext | None,
) -> tuple[list[int], list[int], int]:
    """Return ``(offsets, counts, rows_per_strip)`` for one measurement.

    With ``fetch_context`` the tables are cached per
    ``(url, file_size, samples)``; concurrent callers hold the context
    lock for the whole probe so the header + IFD ranges are fetched once
    and the probe feeds the context bandwidth meter exactly once.
    Without a context this probes independently, exactly as before.
    """
    if fetch_context is None:
        probed, _n = _probe_strip_tables(url, file_size=file_size, samples=samples)
        return probed
    table_key = (url, int(file_size), int(samples))
    with fetch_context.lock:
        cached = fetch_context.strip_tables.get(table_key)
        if cached is not None:
            return cached
        started = time.monotonic()
        probed, probe_bytes = _probe_strip_tables(
            url, file_size=file_size, samples=samples
        )
        fetch_context.probed_seconds += time.monotonic() - started
        fetch_context.probed_bytes += probe_bytes
        fetch_context.strip_tables[table_key] = probed
        return probed


def _probe_strip_tables(
    url: str,
    *,
    file_size: int,
    samples: int,
) -> tuple[tuple[list[int], list[int], int], int]:
    """Fetch header + IFD probes and parse the strip tables.

    Returns ``((offsets, counts, rows_per_strip), bytes_fetched)`` so the
    caller can charge the actual requested bytes to a bandwidth meter.
    """
    header_probe = (RangeSpan(url, 0, HEADER_PROBE_BYTES - 1, "tiff-header"),)
    header_bodies = _fetch_cdse_ranges(header_probe)
    ifd_offset, _endian, _bigtiff = _parse_tiff_header(header_bodies["tiff-header"])
    if file_size > ifd_offset:
        ifd_length = min(file_size - ifd_offset, _MAX_IFD_PROBE_BYTES)
    else:
        ifd_length = _IFD_PROBE_BYTES
    probes = plan_ifd_probe_spans(
        url,
        ifd_offset=ifd_offset,
        ifd_length=max(ifd_length, _IFD_PROBE_BYTES),
    )
    remaining = tuple(span for span in probes if span.label != "tiff-header")
    probe_bodies = dict(header_bodies)
    if remaining:
        probe_bodies.update(_fetch_cdse_ranges(remaining))
    fetched = sum(span.end - span.start + 1 for span in header_probe) + sum(
        span.end - span.start + 1 for span in remaining
    )
    return _strip_tables_from_probes(probe_bodies, samples=samples), fetched


def _use_exact_columns(
    fetch_context: _RoiFetchContext | None,
    *,
    n_cols: int,
    samples: int,
) -> bool:
    """Probe-based narrow-ROI rule for exact-column per-row span planning.

    Exact-column spans trade many small requests for bytes not fetched,
    which only pays off on a slow link with a narrow ROI.  The rule
    engages when both hold:

    - the bandwidth observed through ``fetch_context`` is below
      :data:`NARROW_ROI_BANDWIDTH_BPS` (1.3 MiB/s), and
    - the ROI column width is below
      :data:`NARROW_ROI_MAX_WIDTH_FRACTION` (30%) of the row.

    Without a context (or before its first probe completes) the rule
    stays off and whole-row planning is preserved.
    """
    if fetch_context is None:
        return False
    bandwidth = fetch_context.bandwidth_bps
    if bandwidth is None or bandwidth >= NARROW_ROI_BANDWIDTH_BPS:
        return False
    return n_cols < NARROW_ROI_MAX_WIDTH_FRACTION * samples


def _fetch_deduped(
    spans: Sequence[RangeSpan],
    fetch_context: _RoiFetchContext | None,
) -> dict[str, bytes]:
    """Fetch spans, reusing identical byte ranges already in the context.

    Deduplication is on ``(url, start, end)``: a span whose byte range was
    fetched earlier in this context is answered from cache under its own
    label.  Duplicate ranges inside one batch collapse to a single
    request.  Without a context this is a plain :func:`_fetch_cdse_ranges`
    dispatch.
    """
    span_tuple = tuple(spans)
    if fetch_context is None:
        return _fetch_cdse_ranges(span_tuple)
    bodies: dict[str, bytes] = {}
    missing: list[RangeSpan] = []
    pending: dict[tuple[str, int, int], list[str]] = {}
    with fetch_context.lock:
        cache = fetch_context.bodies
        for span in span_tuple:
            key = (span.url, span.start, span.end)
            if key in cache:
                bodies[span.label] = cache[key]
                continue
            if key in pending:
                pending[key].append(span.label)
                continue
            pending[key] = [span.label]
            missing.append(span)
    if missing:
        fetched = _fetch_cdse_ranges(missing)
        with fetch_context.lock:
            for span in missing:
                key = (span.url, span.start, span.end)
                payload = fetched[span.label]
                fetch_context.bodies[key] = payload
                for label in pending.pop(key):
                    bodies[label] = payload
    return bodies


def _cdse_range_host(url: str) -> bool:
    """Return True when ``url`` is a CDSE OData range origin that needs Bearer."""
    host = (urllib.parse.urlsplit(url).hostname or "").casefold()
    return host in _CDSE_RANGE_HOSTS


def _fetch_cdse_ranges(spans: Sequence[RangeSpan]) -> dict[str, bytes]:
    """Dispatch already-sized spans to remote fetch; CDSE OData uses Bearer.

    Tests and fixture URLs keep the unauthenticated ``fetch_ranges_async``
    path so existing patches still observe probe and measurement requests.
    Live zipper/download OData nodes go through
    :func:`faninsar.remote.providers.cdse.fetch_cdse_ranges`.
    """
    span_tuple = tuple(spans)
    if not span_tuple:
        return {}
    if not _cdse_range_host(span_tuple[0].url):
        return run_sync(fetch_ranges_async(span_tuple))
    from faninsar.remote.providers.cdse import fetch_cdse_ranges

    return run_sync(fetch_cdse_ranges(span_tuple))


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
    """Assemble complex64 samples from fetched span bodies.

    ``row-{line0}-{line1}`` spans carry whole rows sliced to the canvas
    columns; ``col-{line}-{col0}-{col1}`` spans carry one row's tight
    column slice and must match the requested ``(col0, n_cols)``.
    """
    canvas = np.zeros((n_lines, n_cols), dtype=np.complex64)
    parent_line1 = parent_line0 + n_lines
    for label, payload in bodies.items():
        row_match = _ROW_SPAN_LABEL.fullmatch(label)
        if row_match is not None:
            span_line0 = int(row_match.group(1))
            span_line1 = int(row_match.group(2))
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
            continue
        col_match = _COL_SPAN_LABEL.fullmatch(label)
        if col_match is None:
            continue
        span_line = int(col_match.group(1))
        span_col0 = int(col_match.group(2))
        span_col1 = int(col_match.group(3))
        if not parent_line0 <= span_line < parent_line1:
            continue
        if span_col0 != col0 or span_col1 - span_col0 != n_cols:
            reject_product(
                f"column span {label} does not match the requested "
                f"columns [{col0}, {col0 + n_cols})"
            )
        canvas[span_line - parent_line0] = _bytes_to_complex_column(
            payload,
            n_cols=n_cols,
            label=label,
        )
    return canvas


def _bytes_to_complex_column(
    raw: bytes,
    *,
    n_cols: int,
    label: str,
) -> np.ndarray:
    """Decode one exact-column span payload into ``n_cols`` complex64."""
    expected = n_cols * _BYTES_PER_COMPLEX_INT16
    if len(raw) != expected:
        reject_product(f"column span {label} is {len(raw)} bytes, expected {expected}")
    pairs = np.frombuffer(raw, dtype="<i2").reshape(n_cols, 2)
    real = pairs[:, 0].astype(np.float32)
    imag = pairs[:, 1].astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


_MAX_OVERLAP_FETCH_WORKERS = 8
"""Bounded fan-out for independent overlap-rectangle range fetches."""


def _fetch_overlap_rectangles(
    item: Mapping[str, Any],
    swath: S1Swath,
    overlaps: Sequence[BurstOverlapWindow],
    *,
    by_index: Mapping[int, BurstFetchedWindow],
    fetch_context: _RoiFetchContext | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fetch every ``(top, bottom)`` overlap rectangle, parallel across bursts.

    Rectangles already covered by a fetched window are sliced without I/O.
    The remainder are independent ``(url, start, end)`` batches on the same
    measurement, so they are fetched concurrently in a bounded thread pool
    — each worker owns its ``run_sync`` event loop while the shared
    ``fetch_context`` deduplicates byte ranges under a lock.  Splice order
    is preserved by returning results aligned with ``overlaps``.
    """
    pending: list[tuple[int, BurstOverlapWindow]] = []
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, overlap in enumerate(overlaps):
        top_window = by_index.get(int(overlap.top_index))
        bottom_window = by_index.get(int(overlap.bottom_index))
        top_covered = top_window is not None and _window_covers_rectangle(
            top_window,
            row0=int(overlap.top_row0),
            col0=int(overlap.col0),
            n_lines=int(overlap.n_lines),
            n_cols=int(overlap.n_samples),
        )
        bottom_covered = bottom_window is not None and _window_covers_rectangle(
            bottom_window,
            row0=int(overlap.bottom_row0),
            col0=int(overlap.col0),
            n_lines=int(overlap.n_lines),
            n_cols=int(overlap.n_samples),
        )
        if top_covered and bottom_covered:
            results[index] = (
                _overlap_rectangle_samples(
                    item,
                    swath,
                    overlap,
                    role="top",
                    window=top_window,
                    fetch_context=fetch_context,
                ),
                _overlap_rectangle_samples(
                    item,
                    swath,
                    overlap,
                    role="bottom",
                    window=bottom_window,
                    fetch_context=fetch_context,
                ),
            )
        else:
            pending.append((index, overlap))
    if pending:
        workers = min(_MAX_OVERLAP_FETCH_WORKERS, len(pending))
        if workers == 1:
            for index, overlap in pending:
                results[index] = (
                    _overlap_rectangle_samples(
                        item,
                        swath,
                        overlap,
                        role="top",
                        window=by_index.get(int(overlap.top_index)),
                        fetch_context=fetch_context,
                    ),
                    _overlap_rectangle_samples(
                        item,
                        swath,
                        overlap,
                        role="bottom",
                        window=by_index.get(int(overlap.bottom_index)),
                        fetch_context=fetch_context,
                    ),
                )
        else:
            from concurrent.futures import ThreadPoolExecutor

            def _job(overlap: BurstOverlapWindow) -> tuple[np.ndarray, np.ndarray]:
                return (
                    _overlap_rectangle_samples(
                        item,
                        swath,
                        overlap,
                        role="top",
                        window=by_index.get(int(overlap.top_index)),
                        fetch_context=fetch_context,
                    ),
                    _overlap_rectangle_samples(
                        item,
                        swath,
                        overlap,
                        role="bottom",
                        window=by_index.get(int(overlap.bottom_index)),
                        fetch_context=fetch_context,
                    ),
                )

            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="s1-overlap-fetch"
            ) as pool:
                fetched = list(pool.map(lambda pair: _job(pair[1]), pending))
            for (index, _overlap), rectangle in zip(pending, fetched, strict=True):
                results[index] = rectangle
    return [results[index] for index in range(len(overlaps))]


def _overlap_rectangle_samples(
    item: Mapping[str, Any],
    swath: S1Swath,
    overlap: BurstOverlapWindow,
    *,
    role: Literal["top", "bottom"],
    window: BurstFetchedWindow | None = None,
    fetch_context: _RoiFetchContext | None = None,
) -> np.ndarray:
    """Return one overlap rectangle from a fetched window or a dedicated fetch."""
    burst_index = int(overlap.top_index if role == "top" else overlap.bottom_index)
    burst_row0 = int(overlap.top_row0 if role == "top" else overlap.bottom_row0)
    burst_col0 = int(overlap.col0)
    n_lines = int(overlap.n_lines)
    n_samples = int(overlap.n_samples)
    if window is not None and _window_covers_rectangle(
        window,
        row0=burst_row0,
        col0=burst_col0,
        n_lines=n_lines,
        n_cols=n_samples,
    ):
        src_row0 = burst_row0 - int(window.row0)
        src_col0 = burst_col0 - int(window.col0)
        rectangle = window.samples[
            src_row0 : src_row0 + n_lines, src_col0 : src_col0 + n_samples
        ]
        return np.ascontiguousarray(rectangle, dtype=np.complex64)
    lpb = int(swath.lines_per_burst)
    parent_line0 = burst_index * lpb + burst_row0
    bodies = _fetch_measurement_bodies(
        str(item.get("measurement_url", "")),
        file_size=int(item.get("measurement_size") or 0),
        line0=parent_line0,
        line1=parent_line0 + n_lines,
        col0=burst_col0,
        col1=burst_col0 + n_samples,
        samples=int(swath.samples_per_burst),
        fetch_context=fetch_context,
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
    if len(raw) != expected_full:
        reject_product(
            f"row span is {len(raw)} bytes, expected {expected_full} whole-row bytes"
        )
    pairs = np.frombuffer(raw, dtype="<i2").reshape(n_span_lines, samples_per_burst, 2)[
        :, col0 : col0 + n_cols
    ]
    real = pairs[..., 0].astype(np.float32)
    imag = pairs[..., 1].astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)
