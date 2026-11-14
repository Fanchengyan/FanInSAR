"""Mission-owned Sentinel-1 physical sample demand and acquisition orchestration.

PROPOSAL-0061 splits the concern: :mod:`faninsar.remote` transports
already-sized :class:`~faninsar.remote.fetch.RangeSpan` objects and knows
nothing about TIFF geometry, while :mod:`faninsar.io.tiff` owns storage
mapping and decoding. The mission layer owns physical sample demand,
provider dispatch, and SAFE materialization.  This module does not
implement HTTP or authentication; it orchestrates remote fetch
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
- generic planning defaults to row-aligned spans of at most
  :data:`MAX_SPAN_BYTES`; ROI materialization instead unions spatial row
  requirements without artificial subdivision and shares concurrency
  across selected dates, bursts, and tracks;
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
import contextlib
import hashlib
import itertools
import json
import math
import re
import shutil
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from pyproj import Transformer
from pyproj.crs import CRS
from pyproj.exceptions import CRSError

from faninsar.data.query import BoundingBox, Points, Polygons
from faninsar.io import tiff
from faninsar.logging import reject, setup_logger
from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import reject_product
from faninsar.missions.s1.io.canvas import BurstFetchedWindow
from faninsar.missions.s1.overlap import burst_overlap_windows
from faninsar.missions.s1.types import track_group_key
from faninsar.remote.errors import RemoteAccessError, _fail
from faninsar.remote.fetch import RangeFetcher, RangeSpan, fetch_ranges_async
from faninsar.remote.fetch.span_cache import (
    SpanCache,
    _atomic_write_json,
)
from faninsar.remote.providers.cdse import cdse_range_fetcher as _cdse_range_fetcher
from faninsar.stack.layout import staged_acquisition_path, track_acquisitions_dir

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from faninsar.missions.s1.io.canvas import RoiSafeCanvas
    from faninsar.missions.s1.overlap import BurstOverlapWindow
    from faninsar.missions.s1.types import S1Swath
    from faninsar.processing.geometry.dem import RasterDEM

logger = setup_logger(__name__)

MAX_SPAN_BYTES = 4 * 1024 * 1024
_RATE_LIMIT_RETRY_BACKOFF_S = 2.0
"""Retry quantum (PROPOSAL-0061): no span exceeds 4 MiB unless one row does."""

HEADER_PROBE_BYTES = 16
"""Bytes probed at offset 0: classic (8 B) and BigTIFF (16 B) IFD pointers."""

POINT_BUFFER_M = 1000.0
"""Ground half-width applied to a point ROI before geolocation inversion."""

_WGS84 = CRS.from_epsg(4326)

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

_PLANNER_VERSION = (
    f"pv1/span-{MAX_SPAN_BYTES}/header-{HEADER_PROBE_BYTES}/"
    f"narrow-{NARROW_ROI_BANDWIDTH_BPS}-{NARROW_ROI_MAX_WIDTH_FRACTION}"
)
"""Planner identity baked into the canvas fingerprint (PROPOSAL-0070).

Cross-version caches cannot collide: the token covers the span cap, the
header probe length, and the exact-column narrow-ROI regime constants, so a
bandwidth-regime or planner-constant change produces a different
fingerprint and forces re-materialization.
"""

_CANVAS_MANIFEST_NAME = "coverage.json"
"""Visible canvas coverage record inside each materialized item SAFE."""

_LEGACY_CANVAS_MANIFEST_NAME = ".canvas_manifest.json"
"""Pre-coverage completion marker, still honored then promoted."""

_LEGACY_PROGRESS_NAME = ".canvas_progress.json"
"""Pre-coverage in-SAFE burst checkpoint, stripped on promote."""

_SPAN_CACHE_DIRNAME = ".span-cache"
"""Legacy in-SAFE range scratch; new runs keep parts under ``tmp/s1-transfer``."""


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
    max_span_bytes: int | None = MAX_SPAN_BYTES,
) -> tuple[RangeSpan, ...]:
    """Map S1 whole-row demand through provider-free TIFF storage planning.

    Pixel bounds are half-open; HTTP spans are inclusive. Scientific policy
    selects whole rows unless the explicit exact-column compatibility option
    is requested. Strip gaps and the short final strip remain physical facts.
    """
    if compressed or tiled:
        reject_product(
            "compressed or tiled measurement TIFF cannot be planned for range fetch"
        )
    if not url:
        reject_product("range span planning requires a non-empty object URL")
    try:
        layout = tiff.layout_from_tables(
            strip_offsets,
            strip_byte_counts,
            rows_per_strip,
            samples=samples,
            bytes_per_sample=bytes_per_sample,
        )
        if not 0 <= col0 < col1 <= samples:
            reject_product("ROI columns fall outside samples per row")
        plan = tiff.plan_tiff_windows(
            layout,
            (
                tiff.PixelWindow(
                    line0,
                    line1,
                    col0 if exact_columns else 0,
                    col1 if exact_columns else samples,
                ),
            ),
            max_extent_bytes=max_span_bytes,
            exact_columns=exact_columns,
        )
    except ValueError as error:
        reject_product(str(error))
    return tuple(
        RangeSpan(
            url,
            r.extent.offset,
            r.extent.offset + r.extent.length - 1,
            f"col-{r.window.row0}-{col0}-{col1}"
            if exact_columns
            else f"row-{r.window.row0}-{r.window.row1}",
        )
        for r in plan.regions
    )


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
        reject(message, ValueError, logger=logger)
    if provider is not None and provider not in {"cdse", "asf"}:
        message = f"provider must be 'cdse' or 'asf'; got {provider!r}"
        reject(message, ValueError, logger=logger)
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
        reject(message, ValueError, logger=logger)
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

    copied_context = copy_context()
    thread = threading.Thread(
        target=lambda: copied_context.run(_runner),
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
    parent_ids: Sequence[str] | None = None,
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
    parent_ids : sequence of str or None, optional
        Exact ASF-selected parent SAFE ids.  When supplied, a parent that
        the CDSE STAC search did not return is resolved by exact-name CDSE
        OData metadata lookup.  ``None`` disables the fallback because the
        caller could not establish every selected parent's identity.
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
    if parent_ids is not None and catalog == _CDSE_CATALOG:
        if isinstance(parent_ids, str):
            parent_ids = (parent_ids,)
        expected = {
            _safe_product_id(str(value)) or str(value).strip().upper()
            for value in parent_ids
            if str(value).strip()
        }
        present = {_binding_parent_id(binding) for binding in bindings}
        missing = sorted(expected - present)
        if missing:
            bindings.extend(
                _lookup_missing_cdse_parents(
                    missing,
                    roi,
                    requested=requested,
                    budget=kwargs.get("budget"),
                )
            )
    if requested is None:
        return bindings
    return [
        binding
        for binding in bindings
        if str(binding.get("polarization", "")).upper() in requested
    ]


def _lookup_missing_cdse_parents(
    parent_ids: Sequence[str],
    roi: BoundingBox | Points | Polygons,
    *,
    requested: set[str] | None,
    budget: Any | None,
) -> list[dict[str, Any]]:
    """Resolve exact missing ASF parents through CDSE OData metadata.

    The lookup is missing-only: callers pass the exact ASF-selected parent
    ids, and only parents absent from the STAC bindings are queried by their
    full SAFE name.  The shared preparation ledger is reused when present so
    these metadata requests cannot bypass the operation's resource cap.
    """
    from faninsar.remote.protocols import _CallLedger
    from faninsar.remote.providers.cdse import lookup_cdse_safe_record
    from faninsar.remote.records import RemoteResourceBudget

    policy_budget, policy_ledger = _preparation_remote_policy.get()
    ledger = (
        policy_ledger
        if policy_ledger is not None
        else _CallLedger(policy_budget or budget or RemoteResourceBudget())
    )
    resolved: list[dict[str, Any]] = []
    for parent_id in parent_ids:
        item = lookup_cdse_safe_record(parent_id, ledger=ledger)
        if item is None:
            message = (
                f"CDSE OData has no exact product named {parent_id}.SAFE; "
                "refusing to substitute another parent or version"
            )
            _fail(RemoteAccessError, "product_not_found", message)
        if not _exact_parent_covers_roi(item, roi):
            message = (
                f"CDSE exact parent {parent_id} does not intersect the requested "
                "ROI; refusing to substitute another parent"
            )
            _fail(RemoteAccessError, "product_not_found", message)
        bindings = _bind_cdse_item(item)
        if requested is not None:
            bindings = [
                binding
                for binding in bindings
                if str(binding.get("polarization", "")).upper() in requested
            ]
        if not bindings:
            message = (
                f"CDSE exact parent {parent_id} has no requested polarization "
                f"{sorted(requested) if requested is not None else 'all'}; "
                "refusing to substitute another measurement"
            )
            _fail(RemoteAccessError, "invalid_product", message)
        resolved.extend(bindings)
    return resolved


def _exact_parent_covers_roi(
    item: Any,
    roi: BoundingBox | Points | Polygons,
) -> bool:
    """Return whether one exact OData parent intersects the caller's ROI."""
    geometry = getattr(getattr(item, "acquisition", None), "geometry", None)
    if not isinstance(geometry, Mapping):
        return False
    from shapely.geometry import shape

    from faninsar.remote.catalog import _query_geometry

    query, _query_kind, _point_geometries = _query_geometry(roi)
    try:
        return bool(shape(geometry).intersects(query))
    except (TypeError, ValueError):
        return False


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
        Spatial query inverted through the geolocation grid (burst
        granularity) or terrain-aware geo2rdr (roi granularity).
    xml_text : str
        Annotation XML already fetched for this measurement.
    granularity : {"roi", "burst"}, optional
        ``burst`` fetches full-burst row ranges; ``roi`` uses the geo2rdr
        line/pixel envelope.
    fetch_context : _RoiFetchContext or None, optional
        Shared per-measurement fetch context.  When supplied by
        :func:`materialize_cdse_roi_safes`, probe results and identical
        byte ranges are shared with later NESD overlap fetches.
    dem : RasterDEM or None, optional
        DEM used by the geo2rdr inversion on the ``"roi"`` granularity.
        Required whenever ``granularity == "roi"`` and ``roi`` is set —
        every ROI shape now inverts through :func:`_roi_envelopes`.
    **kwargs : Any
        Unused; accepted so tests can add seams without breaking callers.

    Returns
    -------
    swath, windows, xml_text
        Parsed parent swath, one window per intersecting burst, and the
        annotation text to rewrite into the canvas.

    """
    del kwargs
    swath, plans, xml_text = _plan_cdse_windows(
        item,
        roi,
        xml_text=xml_text,
        granularity=granularity,
        fetch_context=fetch_context,
        dem=dem,
    )
    windows = _fetch_planned_windows(
        plans,
        url=str(item.get("measurement_url", "")),
        file_size=int(item.get("measurement_size") or 0),
        samples_per_row=int(swath.samples_per_burst),
        fetch_context=fetch_context,
    )
    return swath, windows, xml_text


def _plan_cdse_windows(
    item: Mapping[str, Any],
    roi: BoundingBox | Points | Polygons,
    *,
    xml_text: str,
    granularity: str = "roi",
    fetch_context: _RoiFetchContext | None = None,
    dem: RasterDEM | None = None,
) -> tuple[S1Swath, list[_WindowPlan], str]:
    """Plan ROI placement without fetching TIFF probes or measurement data.

    Geometry needs annotation XML and, at ROI granularity, a DEM.  Source
    strip tables and transfer spans are deliberately planned separately so
    ROI and NESD physical row requirements can share one source transfer.
    """
    del fetch_context
    swath = parse_annotation_xml(
        xml_text,
        annotation_path=str(item.get("annotation_url", "cdse://annotation.xml")),
        measurement_path=str(item.get("measurement_url", "cdse://measurement.tiff")),
        _original_parent=True,
    )
    if roi is not None and granularity == "roi":
        if dem is None:
            reject_product(
                "ROI-scoped acquisition requires a DEM for geo2rdr inversion"
            )
        plans = _plan_windows_for_polygons(swath, _roi_envelopes(roi, swath, dem))
    else:
        line0, line1, col0, col1 = invert_geolocation_grid(xml_text, roi)
        if granularity == "burst":
            line0, line1 = _full_intersecting_burst_lines(swath, line0, line1)
            col0, col1 = 0, int(swath.samples_per_burst)
        plans = _plan_windows_for_lines(
            swath, line0=line0, line1=line1, col0=col0, col1=col1
        )
    return swath, plans, xml_text


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
        reject(message, ValueError, logger=logger)
    if _is_wgs84(crs):
        return roi
    try:
        return roi.to_crs(_WGS84)
    except (CRSError, TypeError, ValueError) as exc:
        message = "ROI CRS cannot be transformed to EPSG:4326"
        reject(message, ValueError, cause=exc, logger=logger)


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
            reject(message, ValueError, logger=logger)
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
            reject(message, ValueError, logger=logger)
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
    orbit_group: str | None = "auto",
    absolute_burst_ids: Sequence[int] | None = None,
    catalog_sensing_times: Sequence[datetime] | None = None,
    **kwargs: Any,
) -> list[Path]:
    """Materialize CDSE ROI/burst SAFE-subset canvases under ``work_dir``.

    Each annotation XML is fetched on the shared asyncio session; as soon
    as one document is complete it is planned and its measurement spans are
    queued, so later products do not block the first downloads.  Ready
    bursts are written and checkpointed while other downloads continue;
    their raw span bytes are then released.  Whole-SAFE completion follows
    every listed burst and its overlap writes.  Geometry canvases list
    intersecting bursts; NESD canvases list all parent bursts.

    Parameters
    ----------
    roi, datetime_range, polarizations
        Search envelope forwarded to :func:`discover_cdse_slc_items`.
    work_dir : path-like
        Stack work directory; canvases are written under
        ``tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>`` when the items
        carry a full orbit-group identity, else under
        ``inputs/<YYYYMMDD>/<parentBox>``.
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
        DEM used by the terrain-aware geo2rdr inversion that maps every
        ROI shape (BoundingBox, Points, Polygons) to its radar envelope on
        the ``"roi"`` granularity.  Fetched automatically when omitted.
    orbit_group : {"auto", "require"} or str, optional
        How to handle items spanning multiple ``(mission_id, relative_orbit,
        orbit_direction)`` groups.  ``"auto"`` (default) picks the group with
        the most items and skips the rest with a warning.  ``"require"``
        raises ``ValueError`` whenever the selection mixes groups so callers
        must pick one explicitly.  An explicit string like
        ``"S1D/88/ASCENDING"`` keeps only items in that group.  ``None``
        disables grouping entirely (legacy behaviour).
    absolute_burst_ids, catalog_sensing_times : optional
        Existing catalog hints.  When both are given they are associated by
        sensing time onto kept bursts; no extra catalog query is performed.
        Frame-level searches without per-burst IDs leave official IDs unset.
    **kwargs : Any
        Forwarded to discovery (ignored when ``items`` is given).

    Returns
    -------
    list of pathlib.Path
        Generated ``.SAFE`` directory roots.

    Raises
    ------
    ValueError
        If ``roi`` has no CRS, ``coreg_mode`` is unknown, discovery
        returns no products, or ``orbit_group="require"`` and the items
        span multiple orbit groups.

    """
    if coreg_mode not in {"geometry", "nesd"}:
        message = f"coreg_mode must be 'geometry' or 'nesd'; got {coreg_mode!r}"
        reject(message, ValueError, logger=logger)
    roi = normalize_roi_wgs84(roi)
    # Every ROI shape inverts through geo2rdr on the roi granularity, so a
    # DEM is required whenever a spatial bound is given.
    if roi is not None and granularity == "roi" and dem is None:
        dem = _fetch_dem_for_roi(roi, work_dir=work_dir)
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
        reject(message, ValueError, logger=logger)
    items = _select_orbit_group(items, orbit_group)
    if not items:
        message = (
            "Sentinel-1 CDSE search returned no products in the selected "
            f"orbit group {orbit_group!r}"
        )
        reject(message, ValueError, logger=logger)
    from faninsar.missions.s1.roi_transfer import stream_owners

    dem_id = "dem:executed-plan" if granularity == "roi" else "dem:none"
    paths: list[Path] = []
    pending = []
    order: dict[Path, int] = {}
    for item in items:
        product_id = str(item.get("product_id", "S1_ROI"))
        swath_name = str(item.get("swath", "IW1"))
        polarization = str(item.get("polarization", "VV"))
        track_key = _binding_track_key(item)
        out_dir = (
            track_acquisitions_dir(work_dir, track_key)
            if track_key is not None
            else Path(work_dir) / "inputs"
        )
        safe_dir = staged_acquisition_path(
            out_dir,
            f"{product_id}-{swath_name}-{polarization}.SAFE",
        )
        if safe_dir in order:
            continue
        order[safe_dir] = len(order)
        fingerprint = _canvas_fingerprint(
            item,
            roi,
            coreg_mode=coreg_mode,
            granularity=granularity,
            dem_id=dem_id,
        )
        pending.append((item, safe_dir, fingerprint))
    if pending:
        paths.extend(
            owner.safe_dir
            for owner in run_sync(
                stream_owners(
                    pending,
                    roi,
                    work_dir=work_dir,
                    coreg_mode=coreg_mode,
                    granularity=granularity,
                    dem=dem,
                    absolute_burst_ids=absolute_burst_ids,
                    catalog_sensing_times=catalog_sensing_times,
                )
            )
        )
    if not paths:
        message = "Sentinel-1 CDSE search returned no products intersecting the ROI"
        reject(message, ValueError, logger=logger)
    return sorted(paths, key=order.__getitem__)


def _orbit_group_key(item: Mapping[str, Any]) -> tuple[int | None, str]:
    """Return ``(relative_orbit, orbit_direction)`` for an item.

    The orbit group is the *track* — one relative orbit flown in one
    direction.  ``mission_id`` is intentionally dropped so S1A/C/D sharing
    one orbit land in the same group (``track_group_key`` already folds
    them onto one ``"<rel><A|D>"`` key); a mission-qualified group would
    split a single track and drop every satellite but the largest on
    ``"auto"``.
    """
    return (
        item.get("relative_orbit"),
        str(item.get("orbit_direction") or "").upper(),
    )


def _orbit_group_label(key: tuple[int | None, str]) -> str:
    """Return a readable ``"117/ASCENDING"`` label for a group key."""
    rel_orbit, direction = key
    rel = "?" if rel_orbit is None else str(rel_orbit)
    return f"{rel}/{direction or '?'}"


def _normalize_orbit_selector(text: str) -> str:
    """Normalize a selector to the ``"<REL>/<DIRECTION>"`` group label.

    A leading ``"<MISSION>/"`` token (``"S1A/"``, ``"C/"``, ``"sentinel-1d/"``)
    is dropped because the orbit group is mission-agnostic; the remaining
    two tokens are matched verbatim.
    """
    tokens = [token for token in str(text).strip().upper().split("/") if token]
    # Three tokens mean a mission-prefixed selector ("S1A/117/ASCENDING");
    # drop the mission token and keep "<REL>/<DIR>".  A bare two-token
    # selector is already "<REL>/<DIR>".
    if len(tokens) > 2:
        tokens = tokens[-2:]
    return "/".join(tokens)


def _select_orbit_group(
    items: Sequence[dict[str, Any]],
    orbit_group: str | None,
) -> list[dict[str, Any]]:
    """Return items filtered to one ``(relOrbit, direction)`` track group.

    ``orbit_group`` behaviour:

    - ``None``: return items unchanged (legacy).
    - ``"auto"``: pick the group holding the most items; log a warning when
      other groups are skipped.
    - ``"require"``: raise ``ValueError`` when more than one group is present;
      callers must narrow the search or pass an explicit selector.
    - explicit ``"<REL_ORBIT>/<DIRECTION>"`` (a leading ``"<MISSION>/"`` token
      is ignored): keep only items in that track group.  Matching is
      case-insensitive and tolerant to spaces, so both ``"117/ASCENDING"``
      and ``"S1C/117/ASCENDING"`` select the same group.
    """
    if orbit_group is None:
        return list(items)
    groups: dict[tuple[int | None, str], list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(_orbit_group_key(item), []).append(item)
    if len(groups) <= 1:
        return list(items)
    if isinstance(orbit_group, str) and orbit_group not in ("auto", "require"):
        selector = _normalize_orbit_selector(orbit_group)
        for key, members in groups.items():
            if _orbit_group_label(key) == selector:
                logger.info(
                    "Selected orbit group %s: %d item(s)",
                    _orbit_group_label(key),
                    len(members),
                )
                return members
        labels = sorted(_orbit_group_label(k) for k in groups)
        message = (
            f"orbit_group {orbit_group!r} matches no item; available groups: {labels}"
        )
        reject(message, ValueError, logger=logger)
    if orbit_group == "require":
        labels = sorted(_orbit_group_label(k) for k in groups)
        message = (
            "items span multiple orbit groups "
            f"{labels}; pass orbit_group='<REL_ORBIT>/<DIRECTION>' "
            "to pick one"
        )
        reject(message, ValueError, logger=logger)
    # auto
    key, members = max(groups.items(), key=lambda pair: len(pair[1]))
    skipped = sorted(_orbit_group_label(k) for k in groups if k != key)
    logger.warning(
        "Selected orbit group %s (%d item(s)); skipped other groups %s",
        _orbit_group_label(key),
        len(members),
        skipped,
    )
    return members


def _ensure_canvas_ipf_manifest(safe_dir: Path, item: Mapping[str, Any]) -> None:
    """Record the source IPF version on a reused canvas that lacks it."""
    ipf_version = item.get("ipf_version")
    if not ipf_version or (safe_dir / "manifest.safe").is_file():
        return
    from faninsar.missions.s1.io.path_safe import write_minimal_manifest

    names = sorted(
        path.relative_to(safe_dir).as_posix()
        for folder in ("measurement", "annotation")
        for path in (safe_dir / folder).glob("*")
        if path.is_file()
    )
    write_minimal_manifest(safe_dir, safe_dir.name, names, ipf_version=str(ipf_version))


def _canvas_manifest_complete(manifest_path: Path, fingerprint: str) -> bool:
    """Return True when a manifest records ``complete`` for ``fingerprint``."""
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(raw, dict)
        and raw.get("fingerprint") == fingerprint
        and raw.get("complete") is True
    )


def _canvas_coverage_complete(safe_dir: Path, fingerprint: str) -> bool:
    """Return True when coverage or a legacy canvas manifest is complete."""
    return any(
        _canvas_manifest_complete(safe_dir / name, fingerprint)
        for name in (_CANVAS_MANIFEST_NAME, _LEGACY_CANVAS_MANIFEST_NAME)
    )


def _strip_in_safe_scratch(safe_dir: Path) -> None:
    """Remove leftover in-SAFE transfer files from the pre-coverage layout."""
    shutil.rmtree(safe_dir / _SPAN_CACHE_DIRNAME, ignore_errors=True)
    (safe_dir / _LEGACY_PROGRESS_NAME).unlink(missing_ok=True)
    (safe_dir / _LEGACY_CANVAS_MANIFEST_NAME).unlink(missing_ok=True)


def _annotation_burst_coverage(safe_dir: Path) -> list[dict[str, Any]]:
    """Return kept-burst records from canvas annotation XML."""
    annotation_dir = safe_dir / "annotation"
    if not annotation_dir.is_dir():
        return []
    bursts: list[dict[str, Any]] = []
    seen: set[int] = set()
    for annotation in sorted(annotation_dir.glob("*.xml")):
        if not annotation.is_file() or "calibration" in annotation.parts:
            continue
        swath = parse_annotation_xml(
            annotation.read_text(encoding="utf-8"),
            annotation_path=str(annotation),
            measurement_path="",
        )
        for burst in swath.bursts:
            index = int(burst.index)
            if index in seen:
                continue
            seen.add(index)
            record: dict[str, Any] = {
                "index": index,
                "sensing_time": burst.sensing_time.isoformat(),
            }
            if burst.number is not None:
                record["number"] = int(burst.number)
            if burst.absolute_burst_id is not None:
                record["absolute_burst_id"] = int(burst.absolute_burst_id)
            bursts.append(record)
    bursts.sort(key=lambda record: int(record["index"]))
    return bursts


def _promote_canvas_coverage(
    safe_dir: Path,
    fingerprint: str,
    item: Mapping[str, Any],
) -> None:
    """Write ``coverage.json`` and drop hidden in-SAFE transfer leftovers."""
    coverage_path = safe_dir / _CANVAS_MANIFEST_NAME
    if not coverage_path.is_file():
        legacy_path = safe_dir / _LEGACY_CANVAS_MANIFEST_NAME
        created = datetime.now(UTC).isoformat()
        try:
            raw = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        if isinstance(raw, dict) and isinstance(raw.get("created"), str):
            created = raw["created"]
        legacy_item = raw.get("item") if isinstance(raw, dict) else None
        if not isinstance(legacy_item, dict):
            legacy_item = {}
        payload = {
            "complete": True,
            "created": created,
            "fingerprint": fingerprint,
            "parent": str(
                item.get("product_id") or legacy_item.get("product") or "S1_ROI"
            ),
            "swath": str(item.get("swath") or legacy_item.get("swath") or "IW1"),
            "polarization": str(
                item.get("polarization") or legacy_item.get("pol") or "VV"
            ),
            "bursts": _annotation_burst_coverage(safe_dir),
        }
        _atomic_write_json(coverage_path, payload)
    _strip_in_safe_scratch(safe_dir)


def _safe_canvas_intact(safe_dir: Path) -> bool:
    """Re-verify a completed SAFE canvas before skipping the item.

    Mirrors the ``_open_safe_directory`` globs: every ``annotation/*.xml``
    (calibration excluded) must pair with a ``measurement/<stem>.*`` that
    rasterio can open and decode — a truncated-but-present TIFF fails.
    """
    import rasterio
    from rasterio.windows import Window

    annotation_dir = safe_dir / "annotation"
    measurement_dir = safe_dir / "measurement"
    if not annotation_dir.is_dir() or not measurement_dir.is_dir():
        return False
    annotation_files = sorted(
        path
        for path in annotation_dir.glob("*.xml")
        if path.is_file() and "calibration" not in path.parts
    )
    if not annotation_files:
        return False
    for annotation in annotation_files:
        stem = annotation.stem
        matches = [path for path in measurement_dir.glob(f"{stem}.*") if path.is_file()]
        if not matches:
            return False
        try:
            with rasterio.open(matches[0]) as dataset:
                if dataset.width <= 0 or dataset.height <= 0:
                    return False
                # Decoding the last pixel forces the trailing tile to be
                # readable, so a tail-truncated TIFF fails the check.
                tail = Window(dataset.width - 1, dataset.height - 1, 1, 1)
                dataset.read(1, window=tail)
        except Exception:
            return False
    return True


def _roi_canonical(roi: BoundingBox | Points | Polygons) -> str:
    """Return a deterministic canonical form of a WGS84-normalized ROI."""
    if isinstance(roi, BoundingBox):
        return (
            f"bbox:{float(roi.left)!r}:{float(roi.bottom)!r}:"
            f"{float(roi.right)!r}:{float(roi.top)!r}"
        )
    if isinstance(roi, Points):
        values = ",".join(f"{float(x)!r}:{float(y)!r}" for x, y in roi.values)
        return f"points:{values}"
    geometry = roi.to_desired().geodataframe.to_crs(4326)
    members = sorted(geom.wkb.hex() for geom in geometry.geometry if geom is not None)
    return f"polygons:{','.join(members)}"


def _canvas_dem_id(
    roi: BoundingBox | Points | Polygons,
    *,
    granularity: str,
    dem: RasterDEM | None,
) -> str:
    """Return the DEM identity folded into the canvas fingerprint.

    :func:`_plan_cdse_windows` runs ``geo2rdr`` against the DEM for every
    ROI-granularity plan, so the identity is ``dem:file:<path>:<size>:<mtime>``
    under exactly that condition and ``dem:none`` otherwise.  A ``stat`` is
    enough to notice a replaced DEM without reading the raster.
    """
    if roi is None or granularity != "roi" or dem is None:
        return "dem:none"
    path = Path(dem.path).expanduser().resolve(strict=False)
    stat = path.stat()
    return f"dem:file:{path}:{stat.st_size}:{stat.st_mtime_ns}"


def _canvas_fingerprint(
    item: Mapping[str, Any],
    roi: BoundingBox | Points | Polygons,
    *,
    coreg_mode: str,
    granularity: str,
    dem_id: str,
) -> str:
    """Return the planner fingerprint ``F`` for one CDSE item.

    Covers every planner input — annotation/measurement URLs, pol/swath,
    coreg mode, granularity, the canonical ROI, DEM identity, the
    catalog-reported byte sizes (a regenerated product under a stable URL
    trips the wire), and :data:`_PLANNER_VERSION` — so a stale cache can
    never poison a resumed canvas.
    """
    parts = (
        str(item.get("annotation_url", "")),
        str(item.get("measurement_url", "")),
        str(item.get("polarization", "")),
        str(item.get("swath", "")),
        coreg_mode,
        granularity,
        _roi_canonical(roi),
        dem_id,
        str(int(item.get("measurement_size") or 0)),
        str(int(item.get("annotation_size") or 0)),
        _PLANNER_VERSION,
    )
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _fetch_dem_for_roi(
    roi: BoundingBox | Points | Polygons,
    work_dir: str | Path,
) -> RasterDEM:
    """Fetch the canonical workspace DEM covering the whole ROI.

    Downloads source tiles for the ROI envelope and writes
    ``work_dir/dem.tif`` as EPSG:4326 ellipsoidal heights.  A later Stack
    ``prepare_data`` pass reuses that file instead of downloading a second
    copy.  Bounds include the ordinary interpolation margin so biquintic
    DEM sampling along the ROI edge has the kernel support it needs.
    """
    from faninsar.processing.geometry.api import DEM
    from faninsar.processing.geometry.managed_dem import resolve_managed_wgs84_dem
    from faninsar.stack.layout import DEM_RASTER_NAME, TMP_DIRNAME, dem_cache_dir

    bounds = inversion_envelope(roi)
    root = Path(work_dir)
    cache_dir = dem_cache_dir(root)
    source = DEM.from_source("auto", cache_dir=cache_dir)
    return resolve_managed_wgs84_dem(
        managed_path=root / DEM_RASTER_NAME,
        source=source,
        required_bounds=bounds,
        check_coverage=True,
        cache_dir=cache_dir,
        scratch_dir=root / TMP_DIRNAME,
        product=source.product,
    )


def _fetch_dem_for_polygons(
    roi: Polygons,
    work_dir: str | Path,
) -> RasterDEM:
    """Backward-compatible alias for :func:`_fetch_dem_for_roi`."""
    return _fetch_dem_for_roi(roi, work_dir)


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
        product_id = str(getattr(item, "item_id", "S1_ROI"))
        bindings.append(
            {
                "product_id": product_id,
                "annotation_url": annotation_url,
                "measurement_url": measurement_url,
                "measurement_size": int(size or 0),
                "annotation_size": int(annotation_size or 0),
                "polarization": polarization,
                "swath": swath,
                "mission_id": _mission_letter(product_id),
                "relative_orbit": _item_relative_orbit(item, metadata),
                "orbit_direction": _item_orbit_direction(item, metadata),
                "ipf_version": _item_ipf_version(item, metadata),
            }
        )
    return bindings


def _mission_letter(product_id: str) -> str:
    """Return the mission letter (``"A"``, ``"B"``, ``"C"``, or ``"D"``)."""
    name = product_id.upper()
    for letter in ("A", "B", "C", "D"):
        if name.startswith(f"S1{letter}"):
            return letter
    return ""


def _item_ipf_version(item: Any, metadata: Mapping[str, Any]) -> str | None:
    """Return the Sentinel-1 IPF version recorded on a catalog item, if any."""
    for source in (metadata, getattr(item, "raw_metadata", {}) or {}):
        if not isinstance(source, Mapping):
            continue
        value = source.get("ipf_version")
        if value:
            return str(value)
        software = source.get("processing:software")
        if isinstance(software, Mapping) and software.get("Sentinel-1 IPF"):
            return str(software["Sentinel-1 IPF"])
    return None


def _item_relative_orbit(item: Any, metadata: Mapping[str, Any]) -> int | None:
    """Return ``sat:relative_orbit`` from item fields or raw metadata."""
    direct = getattr(item, "relative_orbit", None)
    if isinstance(direct, int):
        return direct
    for source in (metadata, getattr(item, "raw_metadata", {}) or {}):
        if not isinstance(source, Mapping):
            continue
        value = source.get("relative_orbit") or source.get("sat:relative_orbit")
        if isinstance(value, int):
            return value
        try:
            return int(str(value))
        except (TypeError, ValueError):
            continue
    return None


def _item_orbit_direction(item: Any, metadata: Mapping[str, Any]) -> str:
    """Return ``"ASCENDING"``/``"DESCENDING"`` from item fields or metadata."""
    direct = getattr(item, "orbit_direction", None)
    if isinstance(direct, str) and direct:
        return direct.upper()
    for source in (metadata, getattr(item, "raw_metadata", {}) or {}):
        if not isinstance(source, Mapping):
            continue
        value = source.get("orbit_direction") or source.get("sat:orbit_state")
        if isinstance(value, str) and value:
            return value.upper()
    return ""


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


@dataclass(frozen=True, slots=True)
class _WindowPlan:
    """Payload-free placement of one connected ROI row segment.

    ``line0`` and ``col0`` are burst-local; ``parent_line0`` and
    ``parent_line1`` address physical TIFF rows, not azimuth-time rows.
    ``line_valid``, when present, has one entry per parent burst line and
    uses inclusive last columns, as does :class:`BurstFetchedWindow`.
    """

    burst_index: int
    line0: int
    col0: int
    n_lines: int
    n_samples: int
    lines_per_burst: int
    line_valid: tuple[tuple[int, int], ...] | None = None

    @property
    def parent_line0(self) -> int:
        """First physical parent TIFF row, inclusive."""
        return self.burst_index * self.lines_per_burst + self.line0

    @property
    def parent_line1(self) -> int:
        """Last physical parent TIFF row, exclusive."""
        return self.parent_line0 + self.n_lines


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
      rule in :func:`_fetch_measurement_bodies`;
    - when ``span_cache`` is set, measurement spans persist durable
      progress under ``tmp/s1-transfer/<canvas>/<fingerprint>/`` so an
      interrupted item resumes unfinished spans instead of restarting
      them.  Completing the canvas deletes that scratch.
    """

    def __init__(self, span_cache: _SpanCache | None = None) -> None:
        self.strip_tables: dict[
            tuple[str, int, int], tuple[list[int], list[int], int]
        ] = {}
        self.layouts: dict[str, tiff.TiffLayout] = {}
        self.bodies: dict[tuple[str, int, int], bytes] = {}
        self.probed_bytes = 0
        self.probed_seconds = 0.0
        self.span_cache = span_cache
        # Cache/counter mutations (probes, dedup bodies, bandwidth) share
        # this lock; the shared async loop keeps writes single-threaded.
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


def _burst_azimuth_origin_lines(swath: S1Swath) -> np.ndarray:
    """Return each burst's azimuth-line origin measured from ``bursts[0]``.

    Sentinel-1 TOPS bursts overlap in azimuth, so the real per-burst line
    offset is ``(azimuth_time[k] - azimuth_time[0]) / ati`` — smaller than
    ``k * lines_per_burst``.  This is the same convention
    :func:`~faninsar.missions.s1.coregister._unit_placement` and
    :func:`~faninsar.missions.s1.overlap._adjacent_window` use; it is *not*
    the physical TIFF row slot ``index * lines_per_burst``.
    """
    ati = float(swath.azimuth_time_interval_s)
    t0 = swath.bursts[0].azimuth_time
    return np.asarray(
        [
            round((burst.azimuth_time - t0).total_seconds() / ati)
            for burst in swath.bursts
        ],
        dtype=np.float64,
    )


def _azimuth_to_physical_lines(
    swath: S1Swath,
    azimuth_index: np.ndarray,
    origins: np.ndarray,
) -> np.ndarray:
    """Convert geo2rdr azimuth indices into physical TIFF row indices.

    ``azimuth_index`` counts rows linearly from ``bursts[0].azimuth_time``
    (the ``run_geo2rdr`` model convention).  The parent TIFF stores each
    burst contiguously at ``index * lines_per_burst``, so the physical row of
    a solution inside burst ``k`` is ``k * lpb + (az - origins[k])``.  In the
    azimuth overlap between adjacent bursts one ``az`` can map to two
    physical rows; *all* mapped rows are returned (flattened) so the caller
    can take the overall min/max and the enclosing envelope covers every
    burst copy the polygon intersects.  Rows outside every burst are
    returned as NaN and should be filtered by the caller.
    """
    lpb = int(swath.lines_per_burst)
    n_bursts = len(swath.bursts)
    az = np.asarray(azimuth_index, dtype=np.float64).reshape(-1, 1)
    local = az - origins.reshape(1, n_bursts)
    in_burst = (local >= 0.0) & (local < float(lpb))
    slot = np.arange(n_bursts, dtype=np.float64).reshape(1, n_bursts) * lpb
    physical = np.where(in_burst, slot + local, np.nan)
    return physical.ravel()


def _geoms_to_envelopes(
    geometries: Sequence[Any],
    swath: S1Swath,
    dem: RasterDEM,
) -> list[tuple[int, int, int, int]]:
    """Invert WGS84 geometries to per-geometry ``(line0, line1, col0, col1)``.

    Each geometry's bounding box is sampled on an adaptive N x N grid
    (N from :func:`_geo2rdr_grid_size`, 4x4 to 10x10 log-interpolated by
    polygon size).  DEM heights are sampled at the grid points and
    ``run_geo2rdr`` solves for (line, pixel) per point.  The geometry's
    envelope is the bounding box of converged solutions.

    This is the single geo→radar inversion path every ROI kind funnels
    through; the geolocation-grid approximation is not used for ROI
    envelopes because it ignores terrain.

    Returns
    -------
    list of tuple of int
        One ``(line0, line1, col0, col1)`` per geometry, clamped to the
        swath extent.  Geometries with no converged geo2rdr points are
        skipped.

    """
    from faninsar.missions.s1.processing import _radar_model
    from faninsar.processing.geometry.prepare_production import run_geo2rdr

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
    # Geo2rdr azimuth_index counts azimuth lines from bursts[0].azimuth_time,
    # but downstream planners bucket and fetch by physical TIFF row
    # (``index * lines_per_burst``).  Convert each converged solution to its
    # physical row so the returned envelope is in the same coordinate the
    # range fetcher and canvas writer use.
    origins = _burst_azimuth_origin_lines(swath)
    envelopes: list[tuple[int, int, int, int]] = []
    for geom in geometries:
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
        conv = (
            np.asarray(result.converged).ravel().astype(bool)
            & (az >= 0)
            & (az < total_lines)
            & (rg >= 0)
            & (rg < total_samples)
        )
        if not conv.any():
            continue
        lines = _azimuth_to_physical_lines(swath, az[conv], origins)
        lines = lines[np.isfinite(lines)]
        if lines.size == 0:
            continue
        rg_ok = rg[conv]
        line0 = max(0, math.floor(lines.min()))
        line1 = min(total_lines, math.ceil(lines.max()) + 1)
        col0 = max(0, math.floor(rg_ok.min()))
        col1 = min(total_samples, math.ceil(rg_ok.max()) + 1)
        if line1 > line0 and col1 > col0:
            envelopes.append((line0, line1, col0, col1))
    return envelopes


def _polygon_envelopes(
    polygons: Polygons,
    swath: S1Swath,
    dem: RasterDEM,
) -> list[tuple[int, int, int, int]]:
    """Invert each desired polygon via the shared geo2rdr envelope path."""
    gdf = polygons.to_desired().geodataframe.to_crs(4326)
    if gdf.empty:
        return []
    return _geoms_to_envelopes(list(gdf.geometry), swath, dem)


def _roi_envelopes(
    roi: BoundingBox | Points | Polygons,
    swath: S1Swath,
    dem: RasterDEM,
) -> list[tuple[int, int, int, int]]:
    """Invert any spatial ROI to envelopes through ``run_geo2rdr``.

    ``BoundingBox`` becomes one box geometry, ``Points`` becomes one small
    buffered box per point, and ``Polygons`` keeps one geometry per desired
    polygon — every kind then shares the single geo2rdr inversion in
    :func:`_geoms_to_envelopes`, so terrain-aware radar geometry is used for
    all ROI shapes instead of the geolocation-grid approximation.
    """
    from shapely.geometry import box as _box

    normalized = normalize_roi_wgs84(roi)
    geoms: list[Any] = []
    if isinstance(normalized, Polygons):
        gdf = normalized.to_desired().geodataframe.to_crs(4326)
        geoms = list(gdf.geometry)
    elif isinstance(normalized, Points):
        for x, y in normalized.values:
            west, south, east, north = _point_buffer_bounds(float(x), float(y))
            geoms.append(_box(west, south, east, north))
    else:
        geoms.append(
            _box(
                float(normalized.left),
                float(normalized.bottom),
                float(normalized.right),
                float(normalized.top),
            )
        )
    if not geoms:
        return []
    return _geoms_to_envelopes(geoms, swath, dem)


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
    """Fetch disjoint polygon row requirements and assemble per-burst windows."""
    return _fetch_planned_windows(
        _plan_windows_for_polygons(swath, envelopes),
        url=url,
        file_size=file_size,
        samples_per_row=int(swath.samples_per_burst),
        fetch_context=fetch_context,
    )


def _plan_windows_for_polygons(
    swath: S1Swath,
    envelopes: Sequence[tuple[int, int, int, int]],
) -> list[_WindowPlan]:
    """Keep disjoint ROI rows separate while unioning overlapping requirements."""
    if not envelopes:
        return []
    lpb = int(swath.lines_per_burst)
    samples_per_burst = int(swath.samples_per_burst)
    plans: list[_WindowPlan] = []
    for burst in swath.bursts:
        burst_index = int(burst.index)
        burst_line0 = burst_index * lpb
        burst_line1 = burst_line0 + lpb
        burst_envs = [
            (max(line0, burst_line0), min(line1, burst_line1), col0, col1)
            for line0, line1, col0, col1 in envelopes
            if line0 < burst_line1 and line1 > burst_line0
        ]
        for line0, line1 in _merge_line_segments(
            [(env[0], env[1]) for env in burst_envs]
        ):
            segment_envs = [
                env for env in burst_envs if env[0] < line1 and env[1] > line0
            ]
            col0 = max(0, min(env[2] for env in segment_envs))
            col1 = min(
                samples_per_burst, max(col0 + 1, *(env[3] for env in segment_envs))
            )
            line_valid = _line_valid_for_burst(
                segment_envs,
                burst_line0=burst_line0,
                lines_per_burst=lpb,
                samples_per_burst=samples_per_burst,
            )
            plans.append(
                _WindowPlan(
                    burst_index=burst_index,
                    line0=line0 - burst_line0,
                    col0=col0,
                    n_lines=line1 - line0,
                    n_samples=col1 - col0,
                    lines_per_burst=lpb,
                    line_valid=line_valid,
                )
            )
    if not plans:
        reject_product("no polygon envelope intersects any Sentinel-1 burst")
    return plans


def _fetch_planned_windows(
    plans: Sequence[_WindowPlan],
    *,
    url: str,
    file_size: int,
    samples_per_row: int,
    fetch_context: _RoiFetchContext | None,
) -> list[BurstFetchedWindow]:
    """Compatibility fetch facade over payload-free spatial plans.

    Probe one source once, retain physical strip gaps, and leave contiguous
    row requirements unsplit.  Column cropping happens only in assembly.
    """
    if not plans:
        return []
    offsets, counts, rows_per_strip = _strip_tables_for(
        url,
        file_size=file_size,
        samples=samples_per_row,
        fetch_context=fetch_context,
    )
    spans: list[RangeSpan] = []
    for line0, line1 in _merge_line_segments(
        [(plan.parent_line0, plan.parent_line1) for plan in plans]
    ):
        spans.extend(
            plan_measurement_spans(
                url=url,
                strip_offsets=offsets,
                strip_byte_counts=counts,
                rows_per_strip=rows_per_strip,
                samples=samples_per_row,
                bytes_per_sample=_BYTES_PER_COMPLEX_INT16,
                line0=line0,
                line1=line1,
                col0=0,
                col1=samples_per_row,
                max_span_bytes=None,
            )
        )
    bodies = _fetch_deduped(spans, fetch_context)
    return _assemble_planned_windows(plans, bodies, samples_per_row=samples_per_row)


def _assemble_planned_windows(
    plans: Sequence[_WindowPlan],
    bodies: Mapping[str, bytes],
    *,
    samples_per_row: int,
) -> list[BurstFetchedWindow]:
    """Assemble supplied segments into one masked window per burst.

    A caller may pass just one ready burst's plans.  Row gaps are zero-filled
    rather than decoded from unrelated transfer bodies; their XML mask stays
    invalid.  Payload labels remain local to this measurement source.
    """
    by_burst: dict[int, list[_WindowPlan]] = {}
    for plan in plans:
        by_burst.setdefault(plan.burst_index, []).append(plan)
    windows: list[BurstFetchedWindow] = []
    for burst_index, burst_plans in by_burst.items():
        line0 = min(plan.line0 for plan in burst_plans)
        line1 = max(plan.line0 + plan.n_lines for plan in burst_plans)
        col0 = min(plan.col0 for plan in burst_plans)
        col1 = max(plan.col0 + plan.n_samples for plan in burst_plans)
        samples = np.zeros((line1 - line0, col1 - col0), dtype=np.complex64)
        valid = [[-1, -1] for _ in range(burst_plans[0].lines_per_burst)]
        for plan in burst_plans:
            segment = _assemble_window_samples(
                bodies,
                parent_line0=plan.parent_line0,
                n_lines=plan.n_lines,
                col0=plan.col0,
                n_cols=plan.n_samples,
                samples_per_burst=samples_per_row,
            )
            r0, c0 = plan.line0 - line0, plan.col0 - col0
            samples[r0 : r0 + plan.n_lines, c0 : c0 + plan.n_samples] = segment
            for row in range(plan.line0, plan.line0 + plan.n_lines):
                first, last = (
                    plan.line_valid[row]
                    if plan.line_valid is not None
                    else (plan.col0, plan.col0 + plan.n_samples - 1)
                )
                if first < 0:
                    continue
                if valid[row][0] < 0:
                    valid[row] = [first, last]
                else:
                    valid[row][0] = min(valid[row][0], first)
                    valid[row][1] = max(valid[row][1], last)
        windows.append(
            BurstFetchedWindow(
                burst_index=burst_index,
                row0=line0,
                col0=col0,
                samples=samples,
                line_valid=tuple((first, last) for first, last in valid),
            )
        )
    return windows


def _line_valid_for_burst(
    envelopes: Sequence[tuple[int, int, int, int]],
    *,
    burst_line0: int,
    lines_per_burst: int,
    samples_per_burst: int,
) -> tuple[tuple[int, int], ...]:
    """Return per-burst-line ``(first, last)`` valid columns of the envelopes.

    Each envelope is an absolute-frame ``(line0, line1, col0, col1)``
    rectangle.  For every burst-local line the covering envelopes' column
    ranges are unioned into a single span; lines no envelope covers keep
    ``(-1, -1)`` so the canvas records them as unfetched.
    """
    valid: list[list[int]] = [[-1, -1] for _ in range(lines_per_burst)]
    for line0, line1, col0, col1 in envelopes:
        r0 = max(0, int(line0) - burst_line0)
        r1 = min(lines_per_burst, int(line1) - burst_line0)
        c0 = max(0, int(col0))
        c1 = min(samples_per_burst, int(col1))
        for row in range(r0, r1):
            if valid[row][0] < 0:
                valid[row] = [c0, c1 - 1]
            else:
                valid[row][0] = min(valid[row][0], c0)
                valid[row][1] = max(valid[row][1], c1 - 1)
    return tuple((pair[0], pair[1]) for pair in valid)


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
    """Plan, fetch, and assemble a physical row envelope for legacy callers."""
    return _fetch_planned_windows(
        _plan_windows_for_lines(swath, line0=line0, line1=line1, col0=col0, col1=col1),
        url=url,
        file_size=file_size,
        samples_per_row=int(swath.samples_per_burst),
        fetch_context=fetch_context,
    )


def _plan_windows_for_lines(
    swath: S1Swath,
    *,
    line0: int,
    line1: int,
    col0: int,
    col1: int,
) -> list[_WindowPlan]:
    """Intersect a physical row envelope with parent burst slots, without I/O."""
    lpb = int(swath.lines_per_burst)
    samples_per_burst = int(swath.samples_per_burst)
    col0 = max(0, col0)
    col1 = min(samples_per_burst, max(col0 + 1, col1))
    plans: list[_WindowPlan] = []
    for burst in swath.bursts:
        burst_line0 = int(burst.index) * lpb
        inter0 = max(line0, burst_line0)
        inter1 = min(line1, burst_line0 + lpb)
        if inter1 > inter0:
            plans.append(
                _WindowPlan(
                    burst_index=int(burst.index),
                    line0=inter0 - burst_line0,
                    col0=col0,
                    n_lines=inter1 - inter0,
                    n_samples=col1 - col0,
                    lines_per_burst=lpb,
                )
            )
    if not plans:
        reject_product("ROI does not intersect any Sentinel-1 burst")
    return plans


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
    max_span_bytes: int | None = MAX_SPAN_BYTES,
) -> dict[str, bytes]:
    """Fetch IFD probes then bounded measurement spans for one ROI window.

    With ``fetch_context`` the strip tables come from the context cache
    when an earlier rectangle of the same measurement already probed
    them, measurement spans identical to an earlier request are not
    re-requested, and the probe fetch feeds the context bandwidth used
    by the narrow-ROI rule.  Without a context every call probes and
    fetches independently, exactly as before.  ``max_span_bytes=None``
    keeps each merged contiguous row group as one span.
    """
    if not url:
        reject_product("CDSE measurement fetch requires a non-empty object URL")
    spans = _plan_measurement_spans(
        url,
        file_size=file_size,
        line0=line0,
        line1=line1,
        col0=col0,
        col1=col1,
        samples=samples,
        fetch_context=fetch_context,
        max_span_bytes=max_span_bytes,
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


# Tests replace ``_strip_tables_for``; the live ROI path compares against this
# original so TIFF probes stay on the shared asyncio session.
_strip_tables_for_impl = _strip_tables_for


def _ifd_followup_spans(
    url: str,
    header: bytes,
    *,
    file_size: int,
) -> tuple[RangeSpan, ...]:
    """Return IFD/strip-table probes that follow a TIFF header body."""
    ifd_offset, _endian, _bigtiff = _parse_tiff_header(header)
    if file_size > ifd_offset:
        ifd_length = min(file_size - ifd_offset, _MAX_IFD_PROBE_BYTES)
    else:
        ifd_length = _IFD_PROBE_BYTES
    probes = plan_ifd_probe_spans(
        url,
        ifd_offset=ifd_offset,
        ifd_length=max(ifd_length, _IFD_PROBE_BYTES),
    )
    return tuple(span for span in probes if span.label != "tiff-header")


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
    remaining = _ifd_followup_spans(
        url, header_bodies["tiff-header"], file_size=file_size
    )
    probe_bodies = dict(header_bodies)
    if remaining:
        probe_bodies.update(_fetch_cdse_ranges(remaining))
    fetched = sum(span.end - span.start + 1 for span in header_probe) + sum(
        span.end - span.start + 1 for span in remaining
    )
    return _strip_tables_from_probes(probe_bodies, samples=samples), fetched


async def _probe_strip_tables_async(
    url: str,
    *,
    file_size: int,
    samples: int,
    fetcher: Any,
) -> tuple[tuple[list[int], list[int], int], int]:
    """Async variant of :func:`_probe_strip_tables` on an open fetcher.

    Parameters
    ----------
    url : str
        Measurement object URL.
    file_size : int
        Declared measurement size, used to bound the IFD probe.
    samples : int
        Samples per row, forwarded to strip-table decode.
    fetcher : RangeFetcher
        Open session shared with annotation and measurement transfers.

    Returns
    -------
    tables : tuple of (list of int, list of int, int)
        ``(offsets, counts, rows_per_strip)``.
    bytes_fetched : int
        Header plus IFD bytes charged to the bandwidth meter.

    """
    header_probe = (RangeSpan(url, 0, HEADER_PROBE_BYTES - 1, "tiff-header"),)
    header_bodies = await fetch_ranges_async(header_probe, fetcher=fetcher)
    remaining = _ifd_followup_spans(
        url, header_bodies["tiff-header"], file_size=file_size
    )
    probe_bodies = dict(header_bodies)
    if remaining:
        probe_bodies.update(await fetch_ranges_async(remaining, fetcher=fetcher))
    fetched = sum(span.end - span.start + 1 for span in header_probe) + sum(
        span.end - span.start + 1 for span in remaining
    )
    return _strip_tables_from_probes(probe_bodies, samples=samples), fetched


def _use_exact_columns(
    _fetch_context: _RoiFetchContext | None,
    *,
    n_cols: int,
    samples: int,
) -> bool:
    """Return whether the ROI uses per-row exact-column spans.

    Disabled by policy: row-window fetches always download the merged
    contiguous row block as whole-row ranges and slice the polygon's
    columns after download, rather than issuing one ``[col0, col1)`` span
    per row.  Per-row spans cost one request per row and historically
    starved on CDSE; the merged block is a single range per burst group.
    ``n_cols``/``samples``/``_fetch_context`` are accepted so callers keep
    one signature if the heuristic is ever re-enabled.
    """
    del n_cols, samples
    return False


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
        if fetch_context.span_cache is not None:
            fetched = fetch_context.span_cache.fetch(missing)
        else:
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

    When a :class:`_CdseBurstTransfer` operation is active (set by
    :func:`prepare_s1_inputs` or created per direct materialize call) every
    CDSE batch rides that operation's provider-owned fetcher instead of
    opening a fresh session, so the AIMD window, token, and connection pool
    survive across probes, bursts, and dates.  Tests and fixture URLs keep
    the unauthenticated ``fetch_ranges_async`` path so existing patches
    still observe probe and measurement requests.  Live zipper/download
    OData nodes without an operation go through
    :func:`faninsar.remote.providers.cdse.fetch_cdse_ranges` as before.
    """
    span_tuple = tuple(spans)
    if not span_tuple:
        return {}
    operation = _burst_transfer.get()
    if operation is not None:
        return operation.fetch_sync(span_tuple)
    if not _cdse_range_host(span_tuple[0].url):
        return run_sync(fetch_ranges_async(span_tuple))
    from faninsar.remote.providers.cdse import fetch_cdse_ranges

    policy_budget, policy_ledger = _preparation_remote_policy.get()
    policy = (
        {}
        if policy_budget is None and policy_ledger is None
        else {"budget": policy_budget, "ledger": policy_ledger}
    )
    return run_sync(fetch_cdse_ranges(span_tuple, **policy))


_preparation_remote_policy: ContextVar[tuple[Any, Any]] = ContextVar(
    "s1_preparation_remote_policy", default=(None, None)
)


class _CdseBurstTransfer:
    """Operation-scoped provider fetcher shared by one prepare/materialize call.

    PROPOSAL-0061 D2 and the PROPOSAL-0070 amendment keep one remote
    fetcher/window for the selected dates, bursts, and tracks.  The class
    owns a dedicated event-loop thread (named ``faninsar-s1-range-fetch``,
    the P0061 bridge thread) and lazily opens one provider
    :class:`~faninsar.remote.fetch.RangeFetcher` on it — a rerun whose plan
    is fully skipped performs zero network calls.  Sync call sites
    (annotation fetch, TIFF probes, measurement spans) bridge onto that
    loop through ``run_coroutine_threadsafe``; the aiohttp session is never
    cached on a Stack, a module global, or any object outliving
    :meth:`close`.

    Auth boundary: every CDSE URL is validated through
    :func:`faninsar.remote.providers.cdse.validate_cdse_range_urls` before
    its first credentialed request (a runtime allow-list caches the
    provider verdicts), and non-CDSE spans keep today's anonymous
    per-batch path.  Mixed CDSE/non-CDSE batches are rejected.

    Shutdown order is load-bearing: callers stop submitting work, join
    their worker pool, and only then call :meth:`close`, which cancels any
    still-pending bridge futures, closes the fetcher on the live loop, and
    stops the loop thread.  Closing before the pool joins would strand
    ``run_coroutine_threadsafe`` callers on a dead loop.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._cm: Any = None
        self._fetcher: Any = None
        self._open_lock: asyncio.Lock | None = None
        self._lock = threading.Lock()
        self._validated: set[str] = set()
        self._inflight: list[Any] = []
        self._stop = threading.Event()
        # Captured once on the constructing thread: worker threads and the
        # loop thread do not share context variables, so the network budget
        # must not be re-read from a worker's (possibly stale) context.
        self._policy = _preparation_remote_policy.get()

    @property
    def stop_event(self) -> threading.Event:
        """Event set on shutdown; workers must poll it between fetch waves."""
        return self._stop

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start the dedicated bridge loop thread once, on demand."""
        with self._lock:
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="faninsar-s1-range-fetch",
                daemon=True,
            )
            thread.start()
            self._loop = loop
            self._loop_thread = thread
            return loop

    def _bridge(self, coro: Any) -> Any:
        """Run one coroutine on the operation loop and await its result.

        If the calling thread is interrupted while parked on
        ``future.result()`` (SIGINT, cancel), the coroutine would keep
        running on the operation loop unowned and later outlive the
        session close.  The exception path therefore cancels the future
        first and then unwinds; :meth:`close` still gathers the loop
        tasks so the cancellation settles inside the loop.
        """
        loop = self._ensure_loop()
        with self._lock:
            if self._stop.is_set():
                coro.close()
                raise asyncio.CancelledError
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            self._inflight.append(future)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise
        finally:
            with self._lock, contextlib.suppress(ValueError):
                self._inflight.remove(future)

    async def bridge_async(self, coro: Any) -> Any:
        """Await owned-loop work without blocking the caller's event loop.

        Cancellation waits for the actual owned task, not merely the cancelled
        concurrent future, before releasing a caller-owned cache sink.
        """
        loop = self._ensure_loop()
        tasks = []

        async def run() -> Any:
            task = asyncio.current_task()
            tasks.append(task)
            return await coro

        with self._lock:
            if self._stop.is_set():
                coro.close()
                raise asyncio.CancelledError
            future = asyncio.run_coroutine_threadsafe(run(), loop)
            self._inflight.append(future)
        try:
            return await asyncio.wrap_future(future)
        except BaseException:
            future.cancel()

            async def drain() -> None:
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    coro.close()

            drained = asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(drain(), loop)
            )
            # A second caller cancellation must not release a still-writing
            # sink. Shield the same waiter until owned-loop cleanup settles.
            while not drained.done():
                try:
                    await asyncio.shield(drained)
                except asyncio.CancelledError:
                    continue
            drained.result()
            raise
        finally:
            with self._lock, contextlib.suppress(ValueError):
                self._inflight.remove(future)

    async def _validate_and_open(self, new_urls: tuple[str, ...], policy: Any) -> None:
        """Provider-validate new URLs, then open the fetcher with the token.

        Single-flight: concurrent bridge callers queue on one loop-owned
        lock and re-check inside it, so exactly one ``RangeFetcher`` is
        opened.  Validation runs strictly before the token grant so a bad
        origin can never receive a Bearer header.  ``policy`` is captured
        at operation construction because worker threads and the loop
        thread do not share context variables.
        """
        from faninsar.remote.providers.cdse import (
            cdse_range_fetcher,
            validate_cdse_range_urls,
        )

        if self._open_lock is None:
            self._open_lock = asyncio.Lock()
        async with self._open_lock:
            new_urls = tuple(url for url in new_urls if url not in self._validated)
            if new_urls:
                await validate_cdse_range_urls(new_urls)
                self._validated.update(new_urls)
            if self._fetcher is None:
                policy_budget, policy_ledger = policy
                kwargs = (
                    {}
                    if policy_budget is None and policy_ledger is None
                    else {"budget": policy_budget, "ledger": policy_ledger}
                )
                cm = cdse_range_fetcher((), **kwargs)
                try:
                    self._fetcher = await cm.__aenter__()
                except BaseException:
                    self._cm = None
                    raise
                self._cm = cm
                configure = getattr(self._fetcher, "configure_rate_limit_policy", None)
                if configure is not None:
                    configure(max_retries_per_object=10)

    def configure_deferred_pass(self) -> None:
        """Switch the opened fetcher to terminal, serial 429 handling."""
        _reconfigure_deferred_pass_policy(self)

    def cancel_pending(self) -> None:
        """Stop bridge admission and release callers without stopping the loop."""
        with self._lock:
            self._stop.set()
            pending = tuple(self._inflight)
        for future in pending:
            future.cancel()

    def fetch_sync(self, spans: Sequence[RangeSpan]) -> dict[str, bytes]:
        """Fetch one batch from the calling thread via the shared fetcher."""
        if self._stop.is_set():
            reject(
                "burst transfer operation is shutting down",
                RuntimeError,
                logger=logger,
            )
        span_tuple = tuple(spans)
        cdse_spans = tuple(span for span in span_tuple if _cdse_range_host(span.url))
        other_spans = tuple(
            span for span in span_tuple if not _cdse_range_host(span.url)
        )
        if cdse_spans and other_spans:
            reject(
                "mixed CDSE and non-CDSE range origins in one fetch batch",
                ValueError,
                logger=logger,
            )
        if not cdse_spans:
            return run_sync(fetch_ranges_async(other_spans))
        new_urls = tuple(sorted({span.url for span in cdse_spans} - self._validated))
        self._bridge(self._validate_and_open(new_urls, self._policy))
        return self._bridge(self._fetcher.fetch(cdse_spans))

    def close(self) -> None:
        """Ordered shutdown: cancel bridges, drain, close fetcher, stop loop.

        Every pending task owned by the dedicated loop — fetch streams,
        cooldown timers, leftover bridges — is cancelled and gathered with
        ``return_exceptions=True`` before the aiohttp session closes; the
        dedicated loop has no other owners, so there is no fixed drain
        wait and cancellation settles immediately.  The fetcher context is
        read *inside* the loop coroutine, after the drain settles: the
        lazy open runs on that loop, so a close racing an in-flight open
        (e.g. SIGINT while the first synchronous token grant blocks the
        loop) must not trust a snapshot taken on the caller thread.  The
        provider ``__aexit__`` runs last, then the loop thread is joined
        and the loop closed.
        """
        self.cancel_pending()
        loop = self._loop
        if loop is not None:

            async def _drain_and_close() -> None:
                tasks = [
                    task
                    for task in asyncio.all_tasks(loop)
                    if task is not asyncio.current_task()
                ]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                cm = self._cm
                self._cm = None
                self._fetcher = None
                if cm is not None:
                    await cm.__aexit__(None, None, None)

            closer = asyncio.run_coroutine_threadsafe(_drain_and_close(), loop)
            try:
                closer.result()
            except Exception:
                logger.warning("burst transfer fetcher closed with pending error")
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=30.0)
            self._loop_thread = None
        if loop is not None:
            loop.close()
            self._loop = None


def _reconfigure_deferred_pass_policy(operation: _CdseBurstTransfer) -> None:
    """Apply the final-pass policy on the fetcher's owning event loop."""
    if getattr(operation._fetcher, "configure_rate_limit_policy", None) is None:
        return

    async def configure() -> None:
        operation._fetcher.configure_rate_limit_policy(
            max_retries_per_object=0, max_workers=1, reset_counters=True
        )

    operation._bridge(configure())


_burst_transfer: ContextVar[_CdseBurstTransfer | None] = ContextVar(
    "s1_burst_transfer", default=None
)


class _LazySpanFetcher:
    """Provider fetcher proxy that opens the session on first byte transfer.

    A rerun whose executed plan matches a durable canvas performs zero
    network calls; deferring session construction (and its token grant for
    CDSE) keeps that rerun genuinely offline.  Callers keep using the
    ``async with`` shape and the ``fetch``/``fetch_into`` surface.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._cm: Any = None
        self._fetcher: Any = None

    async def _ensure(self) -> Any:
        if self._fetcher is None:
            self._cm = self._factory()
            self._fetcher = await self._cm.__aenter__()
        return self._fetcher

    async def fetch(self, spans: Sequence[RangeSpan]) -> dict[str, bytes]:
        return await (await self._ensure()).fetch(spans)

    async def fetch_into(self, span: RangeSpan, sink: Any, **kwargs: Any) -> int:
        return await (await self._ensure()).fetch_into(span, sink, **kwargs)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> Any:
        if self._cm is not None:
            return await self._cm.__aexit__(*exc_info)
        return None


class _SharedSpanFetcher:
    """Async facade keeping provider sessions and writes on their owned loop."""

    def __init__(self, operation: _CdseBurstTransfer) -> None:
        self._operation = operation

    async def _fetch(
        self, spans: Sequence[RangeSpan], sink: Any = None, **kwargs: Any
    ) -> Any:
        operation = self._operation
        listed = tuple(spans)

        async def transfer() -> Any:
            # Validate every newly encountered URL before using provider auth.
            await operation._validate_and_open(
                tuple(sorted({span.url for span in listed})), operation._policy
            )
            if sink is None:
                return await operation._fetcher.fetch(listed)
            return await operation._fetcher.fetch_into(listed[0], sink, **kwargs)

        return await operation.bridge_async(transfer())

    async def fetch(self, spans: Sequence[RangeSpan]) -> dict[str, bytes]:
        """Fetch a batch through the operation's loop-owned provider."""
        return await self._fetch(spans)

    async def fetch_into(self, span: RangeSpan, sink: Any, **kwargs: Any) -> int:
        """Stream bytes on the owned loop; report progress on the caller loop."""
        callback = kwargs.get("on_bytes")
        if callback is not None:
            caller_loop = asyncio.get_running_loop()
            kwargs["on_bytes"] = lambda count: caller_loop.call_soon_threadsafe(
                callback, count
            )
        return await self._fetch((span,), sink, **kwargs)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        # The prepare wrapper owns the session, including the deferred pass.
        return None


def _span_fetcher_cm(spans: Sequence[RangeSpan]) -> Any:
    """Use provider-owned auth and refresh; fixture URLs remain anonymous."""
    span_tuple = tuple(spans)
    policy_budget, policy_ledger = _preparation_remote_policy.get()
    policy = (
        {}
        if policy_budget is None and policy_ledger is None
        else {"budget": policy_budget, "ledger": policy_ledger}
    )
    if any(_cdse_range_host(span.url) for span in span_tuple):
        operation = _burst_transfer.get()
        if operation is not None:
            return _SharedSpanFetcher(operation)
        return _LazySpanFetcher(lambda: _cdse_range_fetcher(span_tuple, **policy))
    return _LazySpanFetcher(lambda: RangeFetcher(**policy))


class _SpanCache(SpanCache):
    """Synchronous S1 compatibility facade; recovery remains remote-owned."""

    def fetch(self, spans: Sequence[RangeSpan]) -> dict[str, bytes]:
        listed = tuple(spans)
        self._gc_orphans_once(listed)

        async def stage() -> None:
            async with _span_fetcher_cm(listed) as fetcher:
                await asyncio.gather(*(self.stage(span, fetcher) for span in listed))

        run_sync(stage())
        return {
            span.label: self._read_part(span, self._record_for(span)) for span in listed
        }


def _strip_tables_from_probes(
    bodies: Mapping[str, bytes], *, samples: int
) -> tuple[list[int], list[int], int]:
    """Wrap :func:`faninsar.io.tiff._strip_tables_from_probes` with product errors."""
    try:
        return tiff._strip_tables_from_probes(bodies, samples=samples)
    except ValueError as error:
        return reject_product(str(error))


def _validate_strip_tables(
    offsets: Sequence[int],
    counts: Sequence[int],
    *,
    rows_per_strip: int,
    row_stride: int,
) -> int:
    """Wrap :func:`faninsar.io.tiff._validate_strip_tables` with product errors."""
    try:
        return tiff._validate_strip_tables(
            offsets, counts, rows_per_strip=rows_per_strip, row_stride=row_stride
        )
    except ValueError as error:
        return reject_product(str(error))


def _bytes_to_complex_span(
    raw: bytes,
    *,
    n_span_lines: int,
    samples_per_burst: int,
    col0: int,
    n_cols: int,
) -> np.ndarray:
    """Decode one whole-row complex-int16 span with product-level errors."""
    try:
        return tiff.decode_complex_int16(
            raw, rows=n_span_lines, samples=samples_per_burst, col0=col0, n_cols=n_cols
        )
    except ValueError as error:
        return reject_product(str(error))


def _bytes_to_complex_column(raw: bytes, *, n_cols: int, label: str) -> np.ndarray:
    """Decode one single-row column span with product-level errors."""
    try:
        return tiff.decode_complex_int16(raw, rows=1, samples=n_cols)[0]
    except ValueError as error:
        return reject_product(f"column span {label}: {error}")


_parse_tiff_header = tiff._parse_tiff_header


def _span_cache_for_canvas(
    work_dir: str | Path,
    safe_dir: Path,
    fingerprint: str,
) -> _SpanCache:
    """Return the transfer-scratch cache for one CDSE canvas."""
    from faninsar.stack.layout import s1_transfer_root

    return _SpanCache(
        s1_transfer_root(work_dir, safe_dir.name),
        fingerprint,
        label=safe_dir.name,
    )


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
        # ``span_line`` is the absolute physical row; rows in different burst
        # slots cannot share an absolute line index, so a sibling window's
        # span is already filtered by the ``parent_line0 <= line < parent_line1``
        # check above.  A col-span on a row this window owns whose columns do
        # not match the request is a fetch bug, not a sibling span — fail
        # closed rather than silently skipping.
        if span_col0 != col0 or span_col1 - span_col0 != n_cols:
            reject_product(
                f"column span {label} does not match the requested "
                f"columns {col0}..{col0 + n_cols}"
            )
        canvas[span_line - parent_line0] = _bytes_to_complex_column(
            payload,
            n_cols=n_cols,
            label=label,
        )
    return canvas


def _fetch_overlap_rectangles(
    item: Mapping[str, Any],
    swath: S1Swath,
    overlaps: Sequence[BurstOverlapWindow],
    *,
    by_index: Mapping[int, BurstFetchedWindow],
    fetch_context: _RoiFetchContext | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fetch every ``(top, bottom)`` overlap rectangle on one shared batch.

    Rectangles already covered by a fetched window are sliced without I/O.
    The remainder are planned synchronously into ``(url, start, end)``
    spans, deduplicated at plan-merge time through the shared
    ``fetch_context`` (identical byte ranges collapse to one request), and
    fetched in a single ``run_sync`` batch dispatched through
    :func:`_fetch_cdse_ranges` — live CDSE OData spans keep the
    credential-header path while fixture/test URLs route through
    ``fetch_ranges_async``.  Byte ranges are disjoint by construction
    (different bursts -> disjoint parent rows -> disjoint strip byte
    ranges), so no write leases are needed and splice order is preserved
    by assembling results aligned with ``overlaps``.
    """
    lpb = int(swath.lines_per_burst)
    samples_per_burst = int(swath.samples_per_burst)
    url = str(item.get("measurement_url", ""))
    file_size = int(item.get("measurement_size") or 0)

    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    # (overlap_index, role) -> (parent_line0, planned spans)
    plans: dict[tuple[int, str], tuple[int, tuple[RangeSpan, ...]]] = {}
    pending_spans: list[RangeSpan] = []
    for index, overlap in enumerate(overlaps):
        top_window = by_index.get(int(overlap.top_index))
        bottom_window = by_index.get(int(overlap.bottom_index))
        roles = (
            ("top", int(overlap.top_index), int(overlap.top_row0), top_window),
            (
                "bottom",
                int(overlap.bottom_index),
                int(overlap.bottom_row0),
                bottom_window,
            ),
        )
        for role, burst_index, burst_row0, window in roles:
            n_lines = int(overlap.n_lines)
            n_samples = int(overlap.n_samples)
            burst_col0 = int(overlap.col0)
            if window is not None and _window_covers_rectangle(
                window,
                row0=burst_row0,
                col0=burst_col0,
                n_lines=n_lines,
                n_cols=n_samples,
            ):
                continue
            parent_line0 = burst_index * lpb + burst_row0
            planned = _plan_measurement_spans(
                url,
                file_size=file_size,
                line0=parent_line0,
                line1=parent_line0 + n_lines,
                col0=burst_col0,
                col1=burst_col0 + n_samples,
                samples=samples_per_burst,
                fetch_context=fetch_context,
            )
            plans[(index, role)] = (parent_line0, planned)
            pending_spans.extend(planned)

    bodies: dict[str, bytes] = {}
    if pending_spans:
        bodies = _fetch_deduped(tuple(pending_spans), fetch_context)

    for index, overlap in enumerate(overlaps):
        pair: dict[str, np.ndarray] = {}
        for role in ("top", "bottom"):
            n_lines = int(overlap.n_lines)
            n_samples = int(overlap.n_samples)
            burst_col0 = int(overlap.col0)
            plan = plans.get((index, role))
            if plan is None:
                # Covered by a fetched window: slice without I/O.
                burst_index = int(
                    overlap.top_index if role == "top" else overlap.bottom_index
                )
                pair[role] = _overlap_rectangle_samples(
                    item,
                    swath,
                    overlap,
                    role=role,
                    window=by_index.get(burst_index),
                    fetch_context=fetch_context,
                )
                continue
            parent_line0, spans = plan
            rect_bodies = {span.label: bodies[span.label] for span in spans}
            pair[role] = _assemble_window_samples(
                rect_bodies,
                parent_line0=parent_line0,
                n_lines=n_lines,
                col0=burst_col0,
                n_cols=n_samples,
                samples_per_burst=samples_per_burst,
            )
        results[index] = (pair["top"], pair["bottom"])
    return [results[index] for index in range(len(overlaps))]


def _plan_measurement_spans(
    url: str,
    *,
    file_size: int,
    line0: int,
    line1: int,
    col0: int,
    col1: int,
    samples: int,
    fetch_context: _RoiFetchContext | None,
    max_span_bytes: int | None = MAX_SPAN_BYTES,
) -> tuple[RangeSpan, ...]:
    """Plan bounded measurement spans for one parent-line rectangle.

    Shares the probe/strip-table stage with
    :func:`_fetch_measurement_bodies` so overlap rectangles reuse cached
    tables and the narrow-ROI bandwidth regime; only the ``RangeSpan``
    planning is returned — callers own the fetch and assembly.
    ``max_span_bytes=None`` emits each merged contiguous row group as one
    span (whole-burst block fetches).
    """
    if not url:
        reject_product("CDSE measurement fetch requires a non-empty object URL")
    offsets, counts, rows_per_strip = _strip_tables_for(
        url,
        file_size=file_size,
        samples=samples,
        fetch_context=fetch_context,
    )
    return plan_measurement_spans(
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
        max_span_bytes=max_span_bytes,
    )


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


# ---------------------------------------------------------------------------
# Stage-3 provider-decoupled track materialization
# ---------------------------------------------------------------------------


def _seconds_of_day(value: datetime) -> float:
    """Return seconds since UTC midnight of one aware datetime."""
    return (
        value.hour * 3600.0
        + value.minute * 60.0
        + value.second
        + value.microsecond / 1e6
    )


def _burst_sensing_axis_seconds(swath: S1Swath) -> np.ndarray:
    """Return each burst's ``sensing_time`` as seconds-of-day.

    Catalog ``sensingStart`` hints are wall-clock datetimes, so matching
    them against parent bursts uses the sensing-time axis (wrapped daily),
    not ``azimuth_anx_time_s`` — ANX seconds is the cross-mission *pairing*
    coordinate between two swaths, not a wall-clock match key.
    """
    return np.asarray(
        [_seconds_of_day(burst.sensing_time) for burst in swath.bursts],
        dtype=np.float64,
    )


def _burst_window_seconds(swath: S1Swath) -> float:
    """Return one burst's azimuth extent in seconds (linesPerBurst * ATI)."""
    return float(swath.lines_per_burst) * float(swath.azimuth_time_interval_s)


def _resolve_burst_indices(
    swath: S1Swath,
    *,
    absolute_burst_ids: Sequence[int] | None,
    catalog_sensing_times: Sequence[datetime] | None = None,
) -> tuple[int, ...]:
    """Map requested burst identities to parent ``burst_index`` values.

    CDSE SAFE annotation XML carries no ``absoluteBurstID`` (stage-1
    finding), so an ``absolute_burst_ids`` selection is resolved through
    the sensing-time axis: each requested id's catalog ``sensing_time``
    hint is matched to the parent burst nearest on the swath's time axis
    within half a burst window — the same tolerance
    :func:`_common_burst_pairs` uses for cross-mission pairing.  When no
    ``catalog_sensing_times`` hints are given the ids select bursts by
    order (``absolute_burst_ids[i]`` picks the ``i``-th nearest burst to
    that id's relative position), which only holds when ids arrive in
    ascending azimuth order — callers should prefer the hint form.

    Parameters
    ----------
    swath : S1Swath
        Parsed parent sub-swath.
    absolute_burst_ids : sequence of int, optional
        Catalog absolute burst ids to keep.  ``None`` selects every burst.
    catalog_sensing_times : sequence of datetime, optional
        Catalog ``sensingStart`` hints aligned with
        ``absolute_burst_ids``; required for position-correct matching
        when ids are given.

    Returns
    -------
    tuple of int
        Sorted unique parent burst indices to fetch.

    Raises
    ------
    Sentinel1ProductError
        If a requested burst cannot be matched within half a burst
        window.

    """
    n = len(swath.bursts)
    if absolute_burst_ids is None:
        return tuple(int(burst.index) for burst in swath.bursts)
    if not absolute_burst_ids:
        reject_product("absolute_burst_ids selects no bursts")
    axis = _burst_sensing_axis_seconds(swath)
    tolerance = 0.5 * _burst_window_seconds(swath)
    resolved: set[int] = set()
    for position, _burst_id in enumerate(absolute_burst_ids):
        if catalog_sensing_times is not None and position < len(catalog_sensing_times):
            hint = _seconds_of_day(catalog_sensing_times[position])
            gaps = np.abs(axis - hint)
            # Wrap the daily cycle when the axis is a time-of-day axis.
            gaps = np.minimum(gaps, 86400.0 - gaps)
            best = int(np.argmin(gaps))
            if gaps[best] > tolerance:
                reject_product(
                    f"no parent burst matches catalog burst "
                    f"{_burst_id} within half a burst window "
                    f"(best gap {gaps[best]:.3f}s > {tolerance:.3f}s)"
                )
            resolved.add(best)
            continue
        # Positional fallback: pick the burst whose ordinal slot matches
        # the id's rank in the request — only valid for contiguous
        # ascending-id selections.
        if len(absolute_burst_ids) == 1:
            rank = 0
        else:
            rank = round(position * (n - 1) / (len(absolute_burst_ids) - 1))
        resolved.add(int(swath.bursts[min(rank, n - 1)].index))
        logger.warning(
            "absolute_burst_id %s resolved positionally to burst index %d; "
            "pass catalog_sensing_times for an exact sensing_time match",
            _burst_id,
            min(rank, n - 1),
        )
    return tuple(sorted(resolved))


def _associate_catalog_absolute_ids(
    swath: S1Swath,
    *,
    absolute_burst_ids: Sequence[int] | None,
    catalog_sensing_times: Sequence[datetime] | None,
) -> dict[int, int]:
    """Map parent burst index to a catalog official ID by sensing time.

    Association uses the same half-burst-window sensing-time match as
    :func:`_resolve_burst_indices`. IDs are never assigned by list order.
    Missing sensing hints yield an empty mapping so official IDs stay
    ``None`` rather than a positional guess.

    Parameters
    ----------
    swath : S1Swath
        Parsed parent sub-swath.
    absolute_burst_ids : sequence of int, optional
        Catalog official identifiers.
    catalog_sensing_times : sequence of datetime, optional
        Catalog ``sensingStart`` hints aligned with ``absolute_burst_ids``.

    Returns
    -------
    dict of int to int
        ``{burst_index: official_id}`` for bursts that matched a hint.

    Raises
    ------
    Sentinel1ProductError
        If two different catalog IDs match the same parent burst.

    """
    if not absolute_burst_ids or not catalog_sensing_times:
        return {}
    axis = _burst_sensing_axis_seconds(swath)
    tolerance = 0.5 * _burst_window_seconds(swath)
    mapping: dict[int, int] = {}
    count = min(len(absolute_burst_ids), len(catalog_sensing_times))
    for position in range(count):
        burst_id = int(absolute_burst_ids[position])
        hint = _seconds_of_day(catalog_sensing_times[position])
        gaps = np.abs(axis - hint)
        gaps = np.minimum(gaps, 86400.0 - gaps)
        best = int(np.argmin(gaps))
        if gaps[best] > tolerance:
            continue
        index = int(swath.bursts[best].index)
        previous = mapping.get(index)
        if previous is not None and previous != burst_id:
            reject_product(
                f"conflicting official absolute burst IDs {previous} vs {burst_id} "
                f"for burst index {index}"
            )
        mapping[index] = burst_id
    return mapping


def _fetch_burst_block(
    item: Mapping[str, Any],
    swath: S1Swath,
    burst_index: int,
    *,
    fetch_context: _RoiFetchContext | None,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Fetch one whole burst's rows and derive its valid-column mask.

    Rows ``burst_index * lines_per_burst`` through ``+lines_per_burst``
    are range-fetched via :func:`_fetch_measurement_bodies` (reusing the
    shared strip tables and span cache through ``fetch_context``) and
    assembled into a full ``(lines_per_burst, samples_per_burst)``
    complex64 block.  The returned ``line_valid`` keeps only columns the
    annotation's own ``firstValidSample``/``lastValidSample`` declares
    valid, so the path-SAFE mask mirrors the parent burst rather than the
    fetched rectangle.
    """
    lpb = int(swath.lines_per_burst)
    spb = int(swath.samples_per_burst)
    line0 = int(burst_index) * lpb
    bodies = _fetch_measurement_bodies(
        str(item.get("measurement_url", "")),
        file_size=int(item.get("measurement_size") or 0),
        line0=line0,
        line1=line0 + lpb,
        col0=0,
        col1=spb,
        samples=spb,
        fetch_context=fetch_context,
    )
    samples = _assemble_window_samples(
        bodies,
        parent_line0=line0,
        n_lines=lpb,
        col0=0,
        n_cols=spb,
        samples_per_burst=spb,
    )
    burst = swath.bursts[int(burst_index)]
    first = tuple(int(v) for v in burst.first_valid_sample)
    last = tuple(int(v) for v in burst.last_valid_sample)
    if len(first) == lpb and len(last) == lpb:
        line_valid = tuple(
            (first_i, last_i) for first_i, last_i in zip(first, last, strict=True)
        )
    else:
        line_valid = tuple((0, spb - 1) for _ in range(lpb))
    return samples, line_valid


def _binding_track_key(item: Mapping[str, Any]) -> str | None:
    """Return the track key from one CDSE binding's orbit-group fields."""
    return track_group_key(
        str(item.get("mission_id") or ""),
        item.get("relative_orbit"),
        str(item.get("orbit_direction") or ""),
    )


def materialize_cdse_burst_safes(
    roi: BoundingBox | Points | Polygons | None = None,
    datetime_range: tuple[datetime, datetime] | None = None,
    *,
    polarizations: Sequence[str] | Literal["all"] = ("VV",),
    work_dir: str | Path,
    absolute_burst_ids: Sequence[int] | None = None,
    catalog_sensing_times: Sequence[datetime] | None = None,
    items: Sequence[dict[str, Any]] | None = None,
    orbit_group: str | None = None,
    **kwargs: Any,
) -> list[Path]:
    """Materialize same-track burst path-SAFEs from CDSE range fetches.

    The CDSE counterpart of the ASF burst path: instead of ROI-clipped
    canvases, each measurement's selected bursts are fetched as whole
    burst rows and assembled into a
    ``tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>`` path-SAFE readable by
    :func:`open_safe_product` — the identical
    structure the ASF burst-extractor path produces, so the provider does
    not leak upward.

    All CDSE requests (annotation, TIFF/IFD probes, measurement rows)
    share one operation-scoped provider ``RangeFetcher`` (PROPOSAL-0061
    D2, PROPOSAL-0070): when called under :func:`prepare_s1_inputs` the
    fetcher spans every selected date; a direct call owns one temporarily
    for this invocation only.  The fetcher's AIMD window — or
    ``RemoteResourceBudget.max_workers`` when a budget is set — bounds the
    total in-flight HTTP, so per-batch windows no longer reset to the
    seed.  Bursts within an item are fetched serially.

    Parameters
    ----------
    roi : BoundingBox, Points, or Polygons, optional
        Spatial search bound.  Only needed to discover items and to
        restrict the burst selection to bursts intersecting it; when
        ``absolute_burst_ids`` is given the id selection already defines
        the bursts and ``roi`` may be omitted (discovery still uses it
        when ``items`` is not given).
    datetime_range : tuple, optional
        Inclusive acquisition window forwarded to discovery.
    polarizations : sequence of str or ``"all"``, optional
        Channels retained as measurement bindings.
    work_dir : path-like
        Stack work directory; path-SAFEs land under
        ``tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>`` (or
        ``inputs/<YYYYMMDD>/<parentBox>`` when no track identity
        resolves).
    absolute_burst_ids : sequence of int, optional
        Catalog absolute burst ids to keep per measurement.  ``None``
        keeps every burst intersecting ``roi`` (or every burst when
        ``roi`` is also ``None``).
    catalog_sensing_times : sequence of datetime, optional
        Catalog ``sensingStart`` hints aligned with
        ``absolute_burst_ids``; enables exact sensing_time matching.
    items : sequence of mapping, optional
        Pre-bound measurement dicts (``_bind_cdse_item`` output);
        discovery is skipped when given.
    orbit_group : {"auto", "require"} or str, optional
        ``None`` (default) keeps every matched track so each product lands
        under its own ``tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>``
        directory; pass an
        explicit selector (or ``"auto"``/``"require"``) to narrow to one
        track.  Same selection contract as
        :func:`materialize_cdse_roi_safes`.
    **kwargs : Any
        Forwarded to :func:`discover_cdse_slc_items` when ``items`` is
        not given.

    Returns
    -------
    list of pathlib.Path
        Generated path-SAFE directory roots.

    """
    from faninsar.missions.s1.io.path_safe import PathBurstPayload, write_path_safe

    roi = normalize_roi_wgs84(roi) if roi is not None else None
    if items is None:
        if roi is None:
            message = (
                "CDSE burst materialization needs roi for discovery; "
                "pass roi=... or pre-bound items=..."
            )
            reject(message, ValueError, logger=logger)
        items = discover_cdse_slc_items(
            roi,
            datetime_range=datetime_range,
            polarizations=polarizations,
            **kwargs,
        )
    else:
        requested = (
            None if polarizations == "all" else {str(p).upper() for p in polarizations}
        )
        items = [
            binding
            for binding in items
            if requested is None or binding.get("polarization") in requested
        ]
    if not items:
        message = "Sentinel-1 CDSE search returned no products"
        reject(message, ValueError, logger=logger)
    items = _select_orbit_group(items, orbit_group)
    if not items:
        message = (
            "Sentinel-1 CDSE search returned no products in the selected "
            f"orbit group {orbit_group!r}"
        )
        reject(message, ValueError, logger=logger)

    def _out_dir_for(item: Mapping[str, Any]) -> tuple[Path, str | None]:
        key = _binding_track_key(item)
        if key is None:
            out_dir = Path(work_dir) / "inputs"
        else:
            out_dir = track_acquisitions_dir(work_dir, key)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir, key

    def _binding_origin_kind() -> str:
        """Reject mixed-origin bindings; return ``"cdse"`` or ``"other"``."""
        kinds: set[str] = set()
        for binding in items:
            binding_kinds = {
                "cdse"
                if _cdse_range_host(str(binding.get("annotation_url", "")))
                else "other",
                "cdse"
                if _cdse_range_host(str(binding.get("measurement_url", "")))
                else "other",
            }
            if len(binding_kinds) > 1:
                reject(
                    "mixed CDSE and non-CDSE range origins within one "
                    "measurement binding; split the pools by provider",
                    ValueError,
                    logger=logger,
                )
            kinds |= binding_kinds
        if len(kinds) > 1:
            reject(
                "mixed CDSE and non-CDSE range origins in one burst "
                "materialization; one credentialed session cannot serve "
                "both pools",
                ValueError,
                logger=logger,
            )
        return kinds.pop() if kinds else "other"

    origin_kind = _binding_origin_kind()

    operation = _burst_transfer.get()
    owner = None
    owner_token = None
    if origin_kind == "cdse" and operation is None:
        owner = _CdseBurstTransfer()
        owner_token = _burst_transfer.set(owner)
        operation = owner
    paths: list[Path] = []
    try:
        for item in items:
            out_dir, track_key = _out_dir_for(item)
            product_id = str(item.get("product_id", "S1_ROI"))
            fetch_context = _RoiFetchContext()
            xml_text = _fetch_annotation_xml(item, fetch_context=fetch_context)
            if roi is not None and not _roi_overlaps_geolocation(xml_text, roi):
                logger.warning(
                    "Skipping CDSE measurement %s %s %s: "
                    "ROI is outside the geolocation envelope",
                    item.get("product_id", "?"),
                    item.get("swath", "?"),
                    item.get("polarization", "?"),
                )
                continue
            swath = parse_annotation_xml(
                xml_text,
                annotation_path=str(
                    item.get("annotation_url", "cdse://annotation.xml")
                ),
                measurement_path=str(
                    item.get("measurement_url", "cdse://measurement.tiff")
                ),
                _original_parent=True,
            )
            hints = item.get("_asf_burst_hints")
            if hints is not None:
                selected_ids = [int(burst_id) for burst_id, _time in hints]
                selected_times = [time for _burst_id, time in hints]
            else:
                selected_ids = absolute_burst_ids
                selected_times = catalog_sensing_times
            indices = _resolve_burst_indices(
                swath,
                absolute_burst_ids=selected_ids,
                catalog_sensing_times=selected_times,
            )
            if roi is not None:
                # Drop resolved bursts outside the ROI's line envelope so an
                # explicit id selection can still be bounded spatially.
                l0, l1, _c0, _c1 = invert_geolocation_grid(xml_text, roi)
                lpb = int(swath.lines_per_burst)
                indices = tuple(
                    index
                    for index in indices
                    if (index * lpb) < l1 and ((index + 1) * lpb) > l0
                )
            if not indices:
                logger.warning(
                    "Skipping CDSE measurement %s: "
                    "no resolved burst intersects the ROI",
                    product_id,
                )
                continue
            id_by_index = _associate_catalog_absolute_ids(
                swath,
                absolute_burst_ids=selected_ids,
                catalog_sensing_times=selected_times,
            )
            payloads: list[PathBurstPayload] = []
            for index in indices:
                block, line_valid = _fetch_burst_block(
                    item,
                    swath,
                    index,
                    fetch_context=fetch_context,
                )
                catalog_id = id_by_index.get(int(index))
                parsed_id = swath.bursts[int(index)].absolute_burst_id
                if (
                    catalog_id is not None
                    and parsed_id is not None
                    and int(catalog_id) != int(parsed_id)
                ):
                    reject_product(
                        f"conflicting official absolute burst IDs {parsed_id} vs "
                        f"{catalog_id} for burst index {index}"
                    )
                payloads.append(
                    PathBurstPayload(
                        burst_index=int(index),
                        samples=block,
                        line_valid=line_valid,
                        absolute_burst_id=(
                            int(catalog_id) if catalog_id is not None else parsed_id
                        ),
                        number=getattr(swath.bursts[int(index)], "number", None),
                    )
                )
            result = write_path_safe(
                swath,
                payloads,
                track_key or "000?",
                out_dir,
                annotation_xml=xml_text,
                product_id=None,
                ipf_version=item.get("ipf_version"),
            )
            useful_bytes = (
                sum(len(body) for body in fetch_context.bodies.values())
                + fetch_context.probed_bytes
            )
            logger.info(
                "cdse burst materialized %s: bursts=%d useful_bytes=%d",
                product_id,
                len(payloads),
                useful_bytes,
            )
            paths.append(result.safe_root)
    finally:
        if owner_token is not None:
            _burst_transfer.reset(owner_token)
        if owner is not None:
            owner.close()
    if not paths:
        message = "Sentinel-1 CDSE search produced no burst SAFEs"
        reject(message, ValueError, logger=logger)
    return paths


_SAFE_PRODUCT_RE = re.compile(
    r"^(?P<mission>S1[ABCD])_(?P<beam>[A-Z]{2})_SLC__1S(?P<pols>[A-Z]{2})_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_"
    r"(?P<absorbit>\d{6})_(?P<datatake>[0-9A-F]{6})_(?P<uid>[0-9A-F]{4})$"
)
"""ESA IW SLC product-id grammar; same pattern as ``path_safe._SAFE_PRODUCT_RE``."""


def _safe_product_id(segment: str) -> str | None:
    """Return an ESA IW SLC product id from one path or filename segment."""
    text = urllib.parse.unquote(str(segment).strip())
    if text.casefold().endswith(".zip"):
        text = text[:-4]
    if text[-5:].upper() == ".SAFE":
        text = text[:-5]
    text = text.upper()
    return text if _SAFE_PRODUCT_RE.fullmatch(text) else None


def _parse_safe_sensing_window(product_id: str) -> tuple[datetime, datetime] | None:
    """Return parent sensing start/stop encoded in an ESA SLC product id."""
    match = _SAFE_PRODUCT_RE.fullmatch(product_id)
    if match is None:
        return None
    fmt = "%Y%m%dT%H%M%S"
    start = datetime.strptime(match.group("start"), fmt).replace(tzinfo=UTC)
    stop = datetime.strptime(match.group("stop"), fmt).replace(tzinfo=UTC)
    return start, stop


def _asf_data_href(item: Any) -> str | None:
    """Return the ``data`` zip href, not a thumbnail or metadata asset."""
    assets = getattr(item, "assets", None) or {}
    if not isinstance(assets, Mapping):
        return None
    preferred = assets.get("data")
    href = getattr(preferred, "href", None) if preferred is not None else None
    if href:
        return str(href)
    for asset in assets.values():
        href = getattr(asset, "href", None)
        if href and str(href).casefold().endswith(".zip"):
            return str(href)
    return None


def _asf_parent_id(item: Any) -> str | None:
    """Return the parent SAFE id from the data href path or the item id."""
    measurement = _asf_burst_measurement(item)
    if measurement is not None:
        return measurement[0]
    href = _asf_data_href(item)
    if href:
        path = urllib.parse.unquote(urllib.parse.urlsplit(href).path)
        for segment in path.split("/"):
            parent = _safe_product_id(segment)
            if parent is not None:
                return parent
    return _safe_product_id(str(getattr(item, "item_id", "")))


_SWATH_NAMES = frozenset({"IW1", "IW2", "IW3"})
_POL_NAMES = frozenset({"VV", "VH", "HH", "HV"})


def _asf_burst_measurement(item: Any) -> tuple[str, str, str] | None:
    """Return ``(parent, swath, polarization)`` from an SLC-BURST data href.

    The extractor layout is ``.../<parent SAFE>/IW1/VV/8.zip``.  The trailing
    ordinal is the parent ``burstList`` index, not a canvas slot, and is not
    used as a resolver key.
    """
    assets = getattr(item, "assets", None) or {}
    if not isinstance(assets, Mapping):
        return None
    hrefs = []
    data_href = _asf_data_href(item)
    if data_href:
        hrefs.append(data_href)
    for asset in assets.values():
        if getattr(asset, "representation", None) != "burst-source":
            continue
        href = str(getattr(asset, "href", ""))
        parsed = urllib.parse.urlsplit(href)
        if (
            parsed.scheme != "https"
            or parsed.netloc.casefold() != "sentinel1-burst.asf.alaska.edu"
            or parsed.query
            or parsed.fragment
            or not parsed.path.casefold().endswith((".tif", ".tiff", ".xml"))
            or "%" in parsed.path
        ):
            return None
        hrefs.append(href)
    measurements = set()
    for href in hrefs:
        path = urllib.parse.unquote(urllib.parse.urlsplit(href).path)
        parts = [part for part in path.split("/") if part]
        for index, segment in enumerate(parts):
            parent = _safe_product_id(segment)
            if parent is None:
                continue
            if index + 3 != len(parts) - 1:
                return None
            swath = parts[index + 1].upper()
            pol = parts[index + 2].upper()
            filename = parts[index + 3]
            if (
                swath not in _SWATH_NAMES
                or pol not in _POL_NAMES
                or not re.fullmatch(
                    r"\d+\.(?:zip|tif|tiff|xml)", filename, re.IGNORECASE
                )
            ):
                return None
            measurements.add((parent, swath, pol))
            break
        else:
            return None
    return next(iter(measurements)) if len(measurements) == 1 else None


def _asf_burst_groups(
    items: Sequence[Any],
) -> dict[tuple[str, str, str], list[tuple[int, datetime]]]:
    """Group ASF SLC-BURST items by ``(parent, swath, polarization)``.

    Each group carries ``(absoluteBurstID, sensing_time)`` pairs for that
    measurement only.  Identified burst items that lack the href triple or
    a sensing time fail closed so the time-of-day resolver is never fed
    another swath or date's hints.
    """
    groups: dict[tuple[str, str, str], list[tuple[int, datetime]]] = {}
    for item in items:
        if not _asf_is_burst_item(item):
            continue
        measurement = _asf_burst_measurement(item)
        if measurement is None:
            message = (
                "ASF SLC-BURST item "
                f"{getattr(item, 'item_id', '?')!r} has no parent+IW+pol "
                "identity in its data href"
            )
            reject(message, ValueError, logger=logger)
        parent, swath, pol = measurement
        science = _asf_item_science(item)
        absolute = science.get("absoluteBurstID")
        sensing = _asf_sensing_time(item)
        if absolute is None or sensing is None:
            message = (
                "ASF SLC-BURST item "
                f"{getattr(item, 'item_id', '?')!r} lacks absoluteBurstID "
                "or sensing time; refusing positional burst-index fallback"
            )
            reject(message, ValueError, logger=logger)
        groups.setdefault((parent, swath, pol), []).append((int(absolute), sensing))
    return groups


def _asf_frame_parent_ids(items: Sequence[Any]) -> frozenset[str]:
    """Return parent SAFE ids of whole-frame ASF items (not SLC-BURST)."""
    parents: set[str] = set()
    for item in items:
        if _asf_is_burst_item(item):
            continue
        parent = _asf_parent_id(item)
        if parent is not None:
            parents.add(parent)
    return frozenset(parents)


def _apply_asf_burst_hints(
    bindings: Sequence[Mapping[str, Any]],
    groups: Mapping[tuple[str, str, str], Sequence[tuple[int, datetime]]],
    *,
    frame_parents: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Attach per-measurement ASF burst hints; drop unselected IW/pol.

    Burst-only parents keep only the selected ``(parent, swath, pol)``
    groups.  A parent that was also an explicit whole-frame selection
    stays unhinted on every measurement, including ones that also match
    a burst group, so frame-union semantics are preserved.  A selected
    burst group missing from rediscovery fails closed.
    """
    if not groups:
        return [dict(binding) for binding in bindings]
    burst_only_parents = {
        parent for parent, _swath, _pol in groups if parent not in frame_parents
    }
    kept: list[dict[str, Any]] = []
    matched: set[tuple[str, str, str]] = set()
    for binding in bindings:
        parent = _binding_parent_id(binding)
        swath = str(binding.get("swath") or "").upper()
        pol = str(binding.get("polarization") or "").upper()
        key = (parent, swath, pol)
        if key in groups:
            matched.add(key)
            if parent in frame_parents:
                kept.append(dict(binding))
            else:
                updated = dict(binding)
                updated["_asf_burst_hints"] = list(groups[key])
                kept.append(updated)
            continue
        if parent in burst_only_parents:
            continue
        kept.append(dict(binding))
    missing = sorted(key for key in groups if key not in matched)
    if missing:
        message = (
            "CDSE rediscovery is missing ASF-selected measurements "
            f"{missing!r}; refusing to apply another swath or date's "
            "burst hints"
        )
        reject(message, ValueError, logger=logger)
    return kept


def _asf_is_burst_item(item: Any) -> bool:
    """Return whether one catalog item is an ASF SLC-BURST, not a whole frame."""
    collection = str(getattr(item, "collection", "") or "").casefold()
    if "burst" in collection:
        return True
    item_id = str(getattr(item, "item_id", "") or "")
    if item_id.casefold().endswith("-burst"):
        return True
    props = getattr(getattr(item, "acquisition", None), "properties", {}) or {}
    return bool(props.get("absoluteBurstID") or props.get("BURST_ID_ABSOLUTE"))


def _binding_parent_id(item: Mapping[str, Any]) -> str:
    """Return the SAFE product id of one CDSE binding."""
    product_id = str(item.get("product_id") or "").strip()
    parsed = _safe_product_id(product_id)
    return parsed if parsed is not None else product_id.upper()


def _asf_href_burst_number(href: str | None) -> int | None:
    """Return the parent burstList ordinal encoded as ``.../IW1/VV/8.zip``.

    The trailing zip stem is the original parent sub-swath ordinal, not the
    inner one-burst annotation index.  Hrefs that are not this extractor
    layout return ``None``; the ordinal is never guessed from a local list.
    """
    if not href:
        return None
    path = urllib.parse.unquote(urllib.parse.urlsplit(str(href)).path)
    parts = [part for part in path.split("/") if part]
    if len(parts) < 3:
        return None
    last = parts[-1]
    stem, dot, suffix = last.rpartition(".")
    if not dot or suffix.casefold() != "zip" or not stem.isdigit():
        return None
    polarization = parts[-2].upper()
    swath = parts[-3].upper()
    if polarization not in {"VV", "VH", "HH", "HV"}:
        return None
    if not swath.startswith(("IW", "EW")):
        return None
    return int(stem)


def _asf_item_data_href(item: Any) -> str:
    """Return the SLC-BURST data zip href, not a thumbnail or metadata asset."""
    return _asf_data_href(item) or ""


def _asf_item_science(item: Any) -> dict[str, Any]:
    """Return the stage-1 ASF science properties of one catalog item."""
    props = getattr(item.acquisition, "properties", {}) or {}
    raw = getattr(item, "raw_metadata", {}) or {}
    nested = raw.get("properties") if isinstance(raw, Mapping) else {}
    return {
        "absoluteBurstID": (
            props.get("absoluteBurstID") or props.get("BURST_ID_ABSOLUTE")
        ),
        "relativeBurstID": props.get("relativeBurstID")
        or props.get("BURST_ID_RELATIVE"),
        "relativeOrbit": props.get("relativeOrbit") or props.get("PATH_NUMBER"),
        "orbitDirection": props.get("orbitDirection")
        or props.get("ASCENDING_DESCENDING"),
        "mission": props.get("mission") or props.get("MISSION_NAME"),
        "sensingStart": props.get("sensingStart") or props.get("SENSING_START"),
        "nested": nested if isinstance(nested, Mapping) else {},
    }


def _asf_item_track_key(item: Any) -> str | None:
    """Return the track key from one ASF catalog item's science fields."""
    science = _asf_item_science(item)
    return track_group_key(
        str(science.get("mission") or ""),
        science.get("relativeOrbit"),
        str(science.get("orbitDirection") or ""),
    )


def _asf_sensing_time(item: Any) -> datetime | None:
    """Return the burst sensing time of one ASF item.

    Catalog ``sensingStart`` wins when present.  Real ASF ``SLC-BURST``
    properties leave that field unset; ``acquisition.start`` is then the
    parent burstList sensingTime (verified against IW XML, not a frame
    start).
    """
    science = _asf_item_science(item)
    value = science.get("sensingStart")
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if value is not None:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            parsed = None
        else:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    start = getattr(getattr(item, "acquisition", None), "start", None)
    if isinstance(start, datetime):
        return start if start.tzinfo else start.replace(tzinfo=UTC)
    return None


def asf_burst_selectors(
    items: Sequence[Any],
) -> tuple[list[int], list[datetime]]:
    """Map ASF burst items to CDSE ``absolute_burst_ids``/``sensing`` hints.

    ``from_search`` searches on ASF but may download on CDSE.  The two
    catalogs share the ``absoluteBurstID`` namespace and the sensing-time
    axis, so an ASF item's ``absoluteBurstID`` plus sensing time re-finds
    the same bursts inside a CDSE parent frame
    (``_resolve_burst_indices``).  Sensing time is ``sensingStart`` when
    present, otherwise ``acquisition.start``.

    Identified burst items (those with ``absoluteBurstID``) never return
    an id without an aligned time: a missing time would otherwise fall
    through to the positional index resolver, which is unsafe for
    non-contiguous selections.  Whole-frame items have no
    ``absoluteBurstID`` and are skipped.

    Returns
    -------
    tuple of (list of int, list of datetime)
        Aligned ``absolute_burst_ids`` and ``catalog_sensing_times`` for
        :func:`materialize_s1_acquisitions`.

    Raises
    ------
    ValueError
        If an item has ``absoluteBurstID`` but neither ``sensingStart``
        nor ``acquisition.start``.

    """
    ids: list[int] = []
    times: list[datetime] = []
    for item in items:
        science = _asf_item_science(item)
        absolute = science.get("absoluteBurstID")
        sensing = _asf_sensing_time(item)
        if absolute is None:
            continue
        if sensing is None:
            message = (
                "ASF SLC-BURST item "
                f"{getattr(item, 'item_id', '?')!r} has absoluteBurstID "
                f"{absolute} but no sensingStart or acquisition.start; "
                "refusing positional burst-index fallback"
            )
            reject(message, ValueError, logger=logger)
        ids.append(int(absolute))
        times.append(sensing)
    return ids, times


def asf_frame_selectors(
    items: Sequence[Any],
) -> tuple[tuple[datetime, datetime] | None, set[str], frozenset[str] | None]:
    """Map ASF items to a CDSE rediscovery window, tracks, and parent ids.

    ASF ``SLC-BURST`` items are one-burst extractor zips, not whole
    frames.  Their data href encodes the parent SAFE
    (``.../<parent>/IW1/VV/8.zip``).  The CDSE rediscovery window must
    cover that parent's sensing start/end, not the child burst start,
    because :func:`faninsar.remote.search` keeps items by
    ``acquisition.start`` only.

    Whole-frame SLC items keep the parent id and sensing window from the
    SAFE zip stem (or catalog interval when the stem is not a SAFE id).
    Parent product ids are returned only when every selected item has a
    confident parent, so adjacent same-track frames cannot leak into
    rediscovery; otherwise the id filter is omitted rather than dropping
    frame-only identities.

    Returns
    -------
    tuple of (datetime_range or None, set of str, frozenset of str or None)
        Inclusive parent sensing window, resolved ``track_key`` set
        (e.g. ``{"88A", "117A"}``), and parent SAFE ids when every item
        identified a parent.

    Raises
    ------
    ValueError
        If an SLC-BURST extractor href is present but its parent SAFE
        identity cannot be parsed.  A child burst start is never used as
        a silent fallback window.

    """
    starts: list[datetime] = []
    ends: list[datetime] = []
    tracks: set[str] = set()
    parents: list[str] = []
    all_parents_known = True
    for item in items:
        key = _asf_item_track_key(item)
        if key is not None:
            tracks.add(key)
        parent_id = _asf_parent_id(item)
        parent_window = (
            _parse_safe_sensing_window(parent_id) if parent_id is not None else None
        )
        if _asf_is_burst_item(item) and parent_id is None:
            message = (
                "ASF SLC-BURST item "
                f"{getattr(item, 'item_id', '?')!r} has no parent SAFE "
                "identity in its data href; refusing to rediscover CDSE "
                "products from the child burst start"
            )
            reject(message, ValueError, logger=logger)
        if parent_id is None:
            all_parents_known = False
            acquisition = getattr(item, "acquisition", None)
            start = getattr(acquisition, "start", None)
            end = getattr(acquisition, "end", None)
        else:
            parents.append(parent_id)
            if parent_window is not None:
                start, end = parent_window
            else:
                acquisition = getattr(item, "acquisition", None)
                start = getattr(acquisition, "start", None)
                end = getattr(acquisition, "end", None)
        if isinstance(start, datetime):
            starts.append(start if start.tzinfo else start.replace(tzinfo=UTC))
        if isinstance(end, datetime):
            ends.append(end if end.tzinfo else end.replace(tzinfo=UTC))
    window = (min(starts), max(ends)) if starts and ends else None
    known = frozenset(parents) if all_parents_known and parents else None
    return window, tracks, known


def materialize_asf_burst_safes(
    items: Sequence[Any],
    *,
    work_dir: str | Path,
    download: Any | None = None,
    polarizations: Sequence[str] | Literal["all"] = ("VV",),
    on_exclusion: Any | None = None,
) -> list[Path]:
    """Download ASF burst products and reassemble them into path-SAFEs.

    Each ASF ``SLC-BURST`` catalog item is a one-burst SAFE zip.  The
    items are downloaded into the track-aware staging root (already
    ``tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>`` via
    :func:`~faninsar.missions.s1.search.local_destination`), opened, and
    each same-track group's bursts are reassembled into a single
    path-SAFE directory — the same on-disk structure the CDSE burst path
    emits, so ``provider`` never surfaces above this layer.

    Parameters
    ----------
    items : sequence of CatalogItem
        ASF burst catalog items (``dataset=SLC-BURST``).  Their
        ``acquisition.properties`` carry the stage-1 pass-through fields
        ``absoluteBurstID``/``relativeOrbit``/``orbitDirection``/
        ``mission``/``sensingStart``.
    work_dir : path-like
        Stack work directory; downloads and path-SAFEs land under
        ``tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>`` when the track
        resolves.
    download : callable, optional
        ``(asset, destination) -> Path`` transfer hook; defaults to
        :func:`faninsar.remote.download`.  Tests inject a fake.
    polarizations : sequence of str or ``"all"``, optional
        Channels retained when opening downloaded products.
    on_exclusion : callable, optional
        ``(item, destination)`` hook invoked when one item's download
        fails provider checksum verification (``RemoteIntegrityError``
        reason ``checksum_mismatch``); the item is skipped and other
        downloads continue.  Other download errors stay fail-closed.

    Returns
    -------
    list of pathlib.Path
        Generated path-SAFE directory roots, one per same-track group.

    """
    from faninsar.missions.s1.io.export import _read_annotation_source
    from faninsar.missions.s1.io.path_safe import PathBurstPayload, write_path_safe
    from faninsar.missions.s1.io.read import read_full_burst
    from faninsar.missions.s1.safe import open_safe_product
    from faninsar.missions.s1.search import (
        complete_safe_asset,
        local_destination,
    )

    if not items:
        message = "Sentinel-1 ASF search returned no burst products"
        reject(message, ValueError, logger=logger)
    if download is None:
        from faninsar import remote

        policy_budget, policy_ledger = _preparation_remote_policy.get()
        download = (
            remote.download
            if policy_budget is None and policy_ledger is None
            else lambda asset, destination: remote.download(
                asset, destination, budget=policy_budget, ledger=policy_ledger
            )
        )

    requested = (
        None if polarizations == "all" else {str(p).upper() for p in polarizations}
    )
    root = Path(work_dir)

    # Download every burst zip into the track-aware staging root.  A
    # download whose bytes fail provider checksum verification is skipped
    # (reported through ``on_exclusion``) while other items continue;
    # every other download failure stays fail-closed.
    from faninsar import remote as _remote

    staged: list[tuple[Any, Path]] = []
    for item in items:
        asset = complete_safe_asset(item)
        destination = local_destination(root, item, asset)
        try:
            staged.append((item, download(asset, destination)))
        except _remote.RemoteIntegrityError as integrity_error:
            if integrity_error.reason != "checksum_mismatch":
                raise
            logger.warning(
                "Sentinel-1 input failed download verification and was "
                "excluded: %s; delete the recorded file and re-download it "
                "before processing.",
                destination,
            )
            if on_exclusion is not None:
                on_exclusion(item, destination)
    if not staged:
        message = (
            "every Sentinel-1 input failed download verification; delete "
            "the recorded files and re-download them before preparing"
        )
        reject(message, ValueError, logger=logger)

    # Group by (track key, sensing date, swath, polarization) so one
    # path-SAFE gathers one date's bursts of one measurement.
    groups: dict[tuple[str, str, str, str], list[tuple[Any, Path]]] = {}
    for item, path in staged:
        science = _asf_item_science(item)
        track_key = _asf_item_track_key(item) or "000?"
        sensing = _asf_sensing_time(item)
        date_key = sensing.strftime("%Y%m%d") if sensing else "undated"
        swath_name = str(
            science.get("nested", {}).get("swath")
            or getattr(item.acquisition, "beam_mode", None)
            or "IW"
        )
        polarization = str(
            science.get("nested", {}).get("polarization")
            or getattr(item.acquisition, "polarization", None)
            or "VV"
        ).upper()
        if requested is not None and polarization not in requested:
            continue
        groups.setdefault((track_key, date_key, swath_name, polarization), []).append(
            (item, path)
        )

    paths: list[Path] = []
    for (track_key, _date, _swath_name, _pol), members in sorted(groups.items()):
        out_dir = (
            track_acquisitions_dir(root, track_key)
            if track_key != "000?"
            else root / "inputs"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        # Each member is an independent one-burst product; every payload
        # carries its own annotation so write_path_safe sees the group as
        # N documents ordered by sensing_time.
        parent_swath = None
        ipf_versions: set[str | None] = set()
        payloads: list[PathBurstPayload] = []
        for item, path in members:
            product = open_safe_product(
                path,
                polarizations=(None if requested is None else tuple(sorted(requested))),
            )
            if not product.swaths:
                continue
            swath = product.swaths[0]
            ipf_versions.add(product.ipf_version)
            if parent_swath is None:
                parent_swath = swath
            science = _asf_item_science(item)
            burst = swath.bursts[0]
            array = read_full_burst(
                swath,
                burst_index=int(burst.index),
                full_range=True,
            ).samples
            absolute = science.get("absoluteBurstID")
            payloads.append(
                PathBurstPayload(
                    burst_index=int(burst.index),
                    samples=array,
                    line_valid=None,
                    absolute_burst_id=(int(absolute) if absolute is not None else None),
                    number=_asf_href_burst_number(_asf_data_href(item)),
                    annotation_xml=_read_annotation_source(swath),
                )
            )
        if parent_swath is None or not payloads:
            continue
        if len(ipf_versions) > 1:
            reject_product(
                "same-date ASF bursts come from different IPF versions: "
                f"{sorted(str(v) for v in ipf_versions)}"
            )
        result = write_path_safe(
            parent_swath,
            payloads,
            track_key,
            out_dir,
            ipf_version=next(iter(ipf_versions)),
        )
        paths.append(result.safe_root)
    if not paths:
        message = "Sentinel-1 ASF downloads produced no burst SAFEs"
        reject(message, ValueError, logger=logger)
    return paths


def _search_asf_burst_items(
    roi: BoundingBox | Points | Polygons,
    *,
    datetime_range: tuple[datetime, datetime] | None,
    polarizations: Sequence[str] | Literal["all"],
    **kwargs: Any,
) -> list[Any]:
    """Search ASF ``SLC-BURST`` catalog items for the burst path.

    Delegates to :meth:`S1Stack.search` so the mission-owned query
    composition (dataset/processingLevel/polarization filters, canonical
    ordering) is reused rather than duplicated here.
    """
    from faninsar.missions.s1.stack import S1Stack

    return list(
        S1Stack.search(
            roi,
            polarizations=polarizations,
            datetime_range=datetime_range,
            **kwargs,
        )
    )


def materialize_s1_acquisitions(
    roi: BoundingBox | Points | Polygons,
    datetime_range: tuple[datetime, datetime] | None = None,
    *,
    provider: Literal["cdse", "asf"] | None = None,
    granularity: str = "burst",
    polarizations: Sequence[str] | Literal["all"] = ("VV",),
    work_dir: str | Path,
    absolute_burst_ids: Sequence[int] | None = None,
    items: Sequence[Any] | None = None,
    orbit_group: str | None = None,
    **kwargs: Any,
) -> list[Path]:
    """Unified provider-decoupled acquisition materialization entry.

    Dispatches to the CDSE range-fetch path (default) or the ASF
    burst-extractor path; both emit
    ``tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>`` path-SAFE directories
    readable by :func:`open_safe_product`, so the chosen provider never
    surfaces above this seam.

    Parameters
    ----------
    roi : BoundingBox, Points, or Polygons
        Spatial search/selection bound.
    datetime_range : tuple, optional
        Inclusive acquisition window.
    provider : {"cdse", "asf"} or None, optional
        ``None`` (default) lets :func:`resolve_roi_acquisition` infer the
        provider from ``granularity`` (``"roi"``/``"burst"`` → ``"cdse"``).
        ``"cdse"`` range-fetches windows out of CDSE full products;
        ``"asf"`` downloads ASF burst-extractor zips and reassembles them
        into the same path-SAFE structure.  An explicit provider must be
        compatible with ``granularity`` (``"roi"`` is CDSE-only;
        ``"burst"`` works on both).
    granularity : {"roi", "burst"}, optional
        ``"burst"`` (default) materializes whole-burst path-SAFEs;
        ``"roi"`` keeps the legacy CDSE ROI-canvas behaviour
        (``materialize_cdse_roi_safes``).  ``"scene"`` is rejected — this
        entry only emits path-SAFE acquisitions; whole-frame scenes are
        downloaded externally and enter through ``from_safes``.
    polarizations : sequence of str or ``"all"``, optional
        Channels retained.
    work_dir : path-like
        Stack work directory; output lands under
        ``tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>``.
    absolute_burst_ids : sequence of int, optional
        Catalog absolute burst ids to keep (matched through
        ``sensing_time`` on CDSE).  ``None`` keeps every burst
        intersecting ``roi``.  CDSE burst path only.
    items : sequence, optional
        Pre-selected catalog items (CDSE bindings or ASF CatalogItems);
        skips internal search when given.
    orbit_group : {"auto", "require"} or str, optional
        Orbit-group selection policy (CDSE path).  ``None`` (default) keeps
        every matched track so each lands under its own
        ``tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>`` directory — the
        multi-track behaviour.  ``"auto"``/``"require"``/an explicit selector
        narrow to a
        single track.  ROI downloads across selected tracks share one
        transfer pool while retaining per-track canvas destinations.
    **kwargs : Any
        Routed extras.  ``dem`` and ``coreg_mode`` reach the CDSE roi
        canvas path (geo2rdr inversion / NESD ``listed_indices``);
        ``download`` and ``on_exclusion`` reach the ASF burst path
        (transfer hook / checksum-skip exclusion recorder); the rest are
        forwarded to CDSE item discovery.

    Returns
    -------
    list of pathlib.Path
        Generated SAFE directory roots.

    Raises
    ------
    ValueError
        If the provider/granularity pair is incompatible or discovery
        returns nothing.

    """
    resolved_granularity, resolved_provider = resolve_roi_acquisition(
        granularity,
        provider,
    )
    if resolved_granularity == "scene":
        # This entry only emits path-SAFE acquisitions (burst rows or the
        # CDSE roi canvas); whole-frame scene downloads belong to the
        # caller-driven ``from_safes`` path, not this provider seam.
        message = (
            "materialize_s1_acquisitions does not emit whole-frame scenes; "
            'use granularity="burst"/"roi" or download scenes externally '
            "and start from from_safes"
        )
        reject(message, ValueError, logger=logger)
    catalog_sensing_times = kwargs.pop("catalog_sensing_times", None)
    if resolved_provider != "asf":
        kwargs.pop("on_exclusion", None)
    if resolved_provider == "asf":
        if items is None:
            items = _search_asf_burst_items(
                roi,
                datetime_range=datetime_range,
                polarizations=polarizations,
                **kwargs,
            )
        return materialize_asf_burst_safes(
            list(items),
            work_dir=work_dir,
            download=kwargs.pop("download", None),
            polarizations=polarizations,
            on_exclusion=kwargs.pop("on_exclusion", None),
        )
    if resolved_granularity == "burst":
        return materialize_cdse_burst_safes(
            roi,
            datetime_range=datetime_range,
            polarizations=polarizations,
            work_dir=work_dir,
            absolute_burst_ids=absolute_burst_ids,
            catalog_sensing_times=catalog_sensing_times,
            items=items,
            orbit_group=orbit_group,
            **kwargs,
        )
    return materialize_cdse_roi_safes(
        roi,
        datetime_range=datetime_range,
        polarizations=polarizations,
        work_dir=work_dir,
        granularity=resolved_granularity,
        items=items,
        orbit_group=orbit_group,
        absolute_burst_ids=absolute_burst_ids,
        catalog_sensing_times=catalog_sensing_times,
        **kwargs,
    )
