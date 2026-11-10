"""Operation-scoped ROI downloads with durable, burst-ready canvas writes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from faninsar.io import tiff
from faninsar.logging import setup_logger
from faninsar.missions.s1.io.canvas import _RoiCanvasWriter
from faninsar.remote.fetch import RangeSpan
from faninsar.remote.fetch.progress import DownloadProgress

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from faninsar.missions.s1.acquisition import _RoiFetchContext, _WindowPlan
    from faninsar.missions.s1.overlap import BurstOverlapWindow
    from faninsar.missions.s1.types import S1Swath

logger = setup_logger(__name__)

_MAX_ACTIVE_BURSTS = 20
_MAX_TRANSFER_TASKS = 20
_PROGRESS_NAME = "progress.json"
_PROGRESS_SCHEMA = "p0082-completion-v1"
_DECODE_WORKING_BYTES = 4 * 1024 * 1024
_WRITER_FIXED_BYTES = 32 * 1024 * 1024
_CHECKPOINT_HEADROOM = 2 * 1024 * 1024


def _annotation_cache_path(
    work_dir: str | Path, safe_dir: Path, item: Mapping[str, Any]
) -> Path:
    """Return the durable original-annotation cache path for one measurement.

    The canvas annotation is a rewritten cropped product, so the original
    XML needed to recompute the executed plan is kept under the transfer
    scratch root, keyed by URL and catalog size.  It survives the
    fingerprint-scoped cleanup in :meth:`_SpanCache.finalize`.
    """
    from faninsar.stack.layout import s1_transfer_root

    url = str(item.get("annotation_url", ""))
    size = int(item.get("annotation_size") or 0)
    key = hashlib.sha256(f"{url}\x00{size}".encode()).hexdigest()[:16]
    return s1_transfer_root(work_dir, safe_dir.name) / "annotations" / f"{key}.xml"


def _cached_annotation_text(path: Path) -> str | None:
    """Return the cached original annotation text, or ``None`` when absent."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _store_annotation_text(path: Path, text: str) -> None:
    """Atomically persist one original annotation for future local replans."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


@dataclass
class _BurstPlan:
    index: int
    windows: list[_WindowPlan]
    overlaps: list[tuple[BurstOverlapWindow, str]]
    spans: tuple[RangeSpan, ...] = ()
    read_plan: tiff.TiffReadPlan | None = None


@dataclass
class _OwnerPlan:
    item: Mapping[str, Any]
    safe_dir: Path
    fingerprint: str
    context: _RoiFetchContext
    swath: S1Swath
    xml_text: str
    listed: tuple[int, ...]
    bursts: list[_BurstPlan]
    granularity: str = "roi"
    writer: _RoiCanvasWriter | None = None
    completed: dict[str, bool] = field(default_factory=dict)
    abort: threading.Event = field(default_factory=threading.Event)
    skipped: bool = False
    absolute_burst_ids: dict[int, int] = field(default_factory=dict)

    @property
    def progress_path(self) -> Path:
        cache = self.context.span_cache
        if cache is not None:
            return cache.dir / _PROGRESS_NAME
        return self.safe_dir / _PROGRESS_NAME

    def open_writer(self, *, resume: bool = False) -> _RoiCanvasWriter:
        """Create the single writer belonging to this measurement."""
        return _RoiCanvasWriter(
            self.swath,
            self.safe_dir.parent,
            annotation_xml=self.xml_text,
            product_id=self.safe_dir.name.removesuffix(".SAFE"),
            listed_indices=self.listed,
            ipf_version=self.item.get("ipf_version"),
            resume=resume,
            absolute_burst_ids=self.absolute_burst_ids or None,
        )


def plan_owner(
    item: Mapping[str, Any],
    roi: Any,
    *,
    safe_dir: Path,
    fingerprint: str,
    context: _RoiFetchContext,
    coreg_mode: str,
    granularity: str,
    dem: Any,
    xml_text: str | None = None,
    plan_spans: bool = True,
    absolute_burst_ids: Sequence[int] | None = None,
    catalog_sensing_times: Sequence[datetime] | None = None,
) -> _OwnerPlan | None:
    """Plan metadata, spatial unions and burst consumers without bulk transfer.

    ``xml_text`` may be supplied when the caller already fetched the
    annotation; otherwise this function fetches the one XML.  Pass
    ``plan_spans=False`` when the caller will fill strip tables on the
    shared asyncio session before :func:`_plan_owner_spans`.
    """
    from faninsar.missions.s1 import acquisition as source

    if xml_text is None:
        xml_text = source._fetch_annotation_xml(item, fetch_context=context)
    if not source._roi_overlaps_geolocation(xml_text, roi):
        return None
    swath, windows, xml_text = source._plan_cdse_windows(
        item,
        roi,
        xml_text=xml_text,
        granularity=granularity,
        fetch_context=context,
        dem=dem,
    )
    if not windows:
        return None
    listed = (
        tuple(sorted(int(burst.index) for burst in swath.bursts))
        if coreg_mode == "nesd"
        else tuple(sorted({w.burst_index for w in windows}))
    )
    overlaps = (
        source.overlap_windows_for_canvas(
            swath,
            windows,
            listed_indices=listed,
        )
        if coreg_mode == "nesd"
        else ()
    )
    bursts = {index: _BurstPlan(index, [], []) for index in listed}
    for window in windows:
        bursts[window.burst_index].windows.append(window)
    for overlap in overlaps:
        bursts[int(overlap.top_index)].overlaps.append((overlap, "top"))
        bursts[int(overlap.bottom_index)].overlaps.append((overlap, "bottom"))
    owner = _OwnerPlan(
        item,
        safe_dir,
        fingerprint,
        context,
        swath,
        xml_text,
        listed,
        list(bursts.values()),
        granularity,
    )
    owner.absolute_burst_ids = source._associate_catalog_absolute_ids(
        swath,
        absolute_burst_ids=absolute_burst_ids,
        catalog_sensing_times=catalog_sensing_times,
    )
    if plan_spans:
        _plan_owner_spans(owner)
    return owner


def _layout_cache_path(
    work_dir: str | Path, safe_dir: Path, url: str, file_size: int, samples: int
) -> Path:
    """Return the durable TIFF strip-table cache path for one measurement."""
    from faninsar.stack.layout import s1_transfer_root

    key = hashlib.sha256(f"{url}\x00{file_size}\x00{samples}".encode()).hexdigest()[:16]
    return s1_transfer_root(work_dir, safe_dir.name) / "layouts" / f"{key}.json"


def _load_layout_tables(
    path: Path,
) -> tuple[list[int], list[int], int] | None:
    """Load persisted strip tables, or ``None`` when absent or malformed."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        offsets = [int(value) for value in raw["offsets"]]
        counts = [int(value) for value in raw["counts"]]
        rows = int(raw["rows_per_strip"])
    except (KeyError, TypeError, ValueError):
        return None
    if not offsets or len(offsets) != len(counts) or rows < 1:
        return None
    return offsets, counts, rows


def _store_layout_tables(path: Path, tables: tuple[list[int], list[int], int]) -> None:
    """Atomically persist strip tables so reruns need no TIFF header probe."""
    offsets, counts, rows = tables
    payload = json.dumps({"offsets": offsets, "counts": counts, "rows_per_strip": rows})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


async def _plan_owner_spans_async(
    owner: _OwnerPlan, fetcher: Any, work_dir: str | Path | None = None
) -> None:
    """Fill strip tables on the open session, then plan measurement spans.

    Tests that replace :func:`~faninsar.missions.s1.acquisition._strip_tables_for`
    keep that fake; live calls probe the TIFF header/IFD through ``fetcher``
    instead of opening a nested ``asyncio.run`` session.  Probed strip
    tables and the original annotation are cached durably under the
    transfer scratch root so an identical rerun plan needs no network call.

    Parameters
    ----------
    owner : _OwnerPlan
        Planned measurement whose windows still need byte spans.
    fetcher : RangeFetcher
        Open session shared with annotation and measurement transfers.
    work_dir : str or pathlib.Path, optional
        Stack work directory owning the durable metadata cache.

    """
    from faninsar.missions.s1 import acquisition as source

    if source._strip_tables_for is source._strip_tables_for_impl:
        url = str(owner.item["measurement_url"])
        file_size = int(owner.item.get("measurement_size") or 0)
        samples = int(owner.swath.samples_per_burst)
        table_key = (url, file_size, samples)
        cached = owner.context.strip_tables.get(table_key)
        if cached is None:
            durable = None
            layout_cache = (
                _layout_cache_path(work_dir, owner.safe_dir, url, file_size, samples)
                if work_dir is not None
                else None
            )
            if layout_cache is not None:
                durable = _load_layout_tables(layout_cache)
            if durable is not None:
                owner.context.strip_tables[table_key] = durable
                owner.context.layouts[url] = tiff.layout_from_tables(
                    *durable, samples=samples
                )
            else:
                started = time.monotonic()
                probe_bytes = 0

                async def read_at(offset: int, length: int) -> bytes:
                    nonlocal probe_bytes
                    label = "tiff-header" if offset == 0 else f"ifd-{offset}"
                    span = RangeSpan(url, offset, offset + length - 1, label)
                    bodies = await source.fetch_ranges_async((span,), fetcher=fetcher)
                    probe_bytes += length
                    return bodies[label]

                layout = await tiff.inspect_tiff(
                    read_at, size_bytes=file_size, samples=samples
                )
                owner.context.layouts[url] = layout
                probed = (
                    list(layout.strip_offsets),
                    list(layout.strip_byte_counts),
                    layout.rows_per_strip,
                )
                owner.context.record_probe(
                    bytes_fetched=probe_bytes,
                    seconds=time.monotonic() - started,
                )
                owner.context.strip_tables[table_key] = probed
                if layout_cache is not None:
                    _store_layout_tables(layout_cache, probed)
    _plan_owner_spans(owner)


def _plan_owner_spans(owner: _OwnerPlan) -> None:
    """Turn planned windows into measurement spans using cached strip tables."""
    from faninsar.missions.s1 import acquisition as source

    lpb = int(owner.swath.lines_per_burst)
    url = str(owner.item["measurement_url"])
    layout = owner.context.layouts.get(url)
    if layout is None:
        offsets, counts, rows_per_strip = source._strip_tables_for(
            url,
            file_size=int(owner.item.get("measurement_size") or 0),
            samples=int(owner.swath.samples_per_burst),
            fetch_context=owner.context,
        )
        layout = tiff.layout_from_tables(
            offsets, counts, rows_per_strip, samples=int(owner.swath.samples_per_burst)
        )
        owner.context.layouts[url] = layout
    all_spans = []
    for burst in owner.bursts:
        if str(burst.index) in owner.completed:
            continue
        rows = [(w.parent_line0, w.parent_line1) for w in burst.windows]
        for overlap, role in burst.overlaps:
            row0 = int(overlap.top_row0 if role == "top" else overlap.bottom_row0)
            start = burst.index * lpb + row0
            rows.append((start, start + int(overlap.n_lines)))
        physical = tuple(
            tiff.PixelWindow(line0, line1, 0, layout.width)
            for line0, line1 in source._merge_line_segments(rows)
        )
        burst.read_plan = tiff.plan_tiff_windows(layout, physical)
        burst.spans = tuple(
            RangeSpan(
                url,
                r.extent.offset,
                r.extent.offset + r.extent.length - 1,
                f"row-{r.window.row0}-{r.window.row1}",
            )
            for r in burst.read_plan.regions
        )
        all_spans.extend(burst.spans)
    # Identity is constructed from the immutable plan actually executed once.
    # DEM effects are represented by placement/validity/overlap demand, while
    # annotation hashing preserves its original orbit, timing and phase facts.
    facts = {
        "source": owner.fingerprint,
        "annotation": hashlib.sha256(owner.xml_text.encode()).hexdigest(),
        "layout": layout.digest,
        "listed": owner.listed,
        "targets": [
            {
                "index": b.index,
                "windows": [asdict(w) for w in b.windows],
                "overlaps": [(asdict(w), role) for w, role in b.overlaps],
                "read_plan": asdict(b.read_plan),
            }
            for b in owner.bursts
        ],
    }
    owner.fingerprint = hashlib.sha256(
        json.dumps(facts, sort_keys=True).encode()
    ).hexdigest()
    old_cache = owner.context.span_cache
    if old_cache is not None:
        owner.context.span_cache = source._SpanCache(
            old_cache.root, owner.fingerprint, label=old_cache.label
        )
        owner.context.span_cache._gc_orphans_once(all_spans)
    _restore_progress(owner)


def _progress_record(owner: _OwnerPlan) -> dict[str, Any]:
    """Return the durable burst-completion record for ``owner``."""
    return {
        "schema": _PROGRESS_SCHEMA,
        "fingerprint": owner.fingerprint,
        "bursts": dict.fromkeys(owner.completed, True),
    }


def _restore_progress(owner: _OwnerPlan) -> None:
    """Honor completed burst indices after the canvas structure still parses.

    Legacy digest maps are accepted as completion only when the plan fingerprint
    and canvas structure match. Payload bytes are not rehashed.
    """
    try:
        raw = json.loads(owner.progress_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict) or raw.get("fingerprint") != owner.fingerprint:
        return
    schema = raw.get("schema")
    if schema is not None and schema != _PROGRESS_SCHEMA:
        return
    completed = raw.get("bursts")
    if not isinstance(completed, dict):
        return
    try:
        owner.writer = owner.open_writer(resume=True)
    except (OSError, ValueError, RuntimeError):
        return
    legacy = schema is None
    for index in owner.listed:
        marker = completed.get(str(index))
        if marker is True or (legacy and isinstance(marker, str) and marker):
            owner.completed[str(index)] = True


def _write_burst(owner: _OwnerPlan, burst: _BurstPlan) -> None:
    """Decode verified local blocks incrementally on the sole writer thread."""
    import rasterio

    from faninsar.missions.s1.io.canvas import BurstFetchedWindow

    cache = owner.context.span_cache
    assert cache is not None
    assert burst.read_plan is not None
    read_at = cache.reader(burst.spans)
    plan = burst.read_plan
    lpb = int(owner.swath.lines_per_burst)

    def blocks(window: tiff.PixelWindow) -> Iterator[tiff.RasterBlock]:
        cancel_message = "canvas assembly cancelled between safe blocks"
        for block in tiff.decode_tiff_blocks(
            plan, read_at, max_working_bytes=_DECODE_WORKING_BYTES, window=window
        ):
            if owner.abort.is_set():
                raise InterruptedError(cancel_message)
            yield block

    def windows() -> Iterator[BurstFetchedWindow]:
        for w in burst.windows:
            crop = tiff.PixelWindow(
                w.parent_line0, w.parent_line1, w.col0, w.col0 + w.n_samples
            )
            for block in blocks(crop):
                yield BurstFetchedWindow(
                    burst.index,
                    block.window.row0 - burst.index * lpb,
                    block.window.col0,
                    block.samples,
                    w.line_valid,
                )

    def rectangles() -> Iterator[tuple[BurstOverlapWindow, str, Any]]:
        for overlap, role in burst.overlaps:
            row0 = int(overlap.top_row0 if role == "top" else overlap.bottom_row0)
            crop = tiff.PixelWindow(
                burst.index * lpb + row0,
                burst.index * lpb + row0 + int(overlap.n_lines),
                int(overlap.col0),
                int(overlap.col0 + overlap.n_samples),
            )
            for block in blocks(crop):
                local_row = block.window.row0 - burst.index * lpb
                change = {"top_row0" if role == "top" else "bottom_row0": local_row}
                subwindow = replace(
                    overlap,
                    **change,
                    col0=block.window.col0,
                    n_lines=block.window.row1 - block.window.row0,
                    n_samples=block.window.col1 - block.window.col0,
                )
                yield subwindow, role, block.samples

    # Reserve fixed GDAL/tile buffers separately from the decoder budget.
    with rasterio.Env(GDAL_CACHEMAX=_WRITER_FIXED_BYTES // 4):
        if owner.writer is None:
            owner.writer = owner.open_writer()
        owner.writer.write_burst(burst.index, windows(), rectangles())


async def _write_ready_burst(owner: _OwnerPlan, burst: _BurstPlan) -> None:
    """Publish a durable checkpoint before deletion; join repeated cancellation."""
    from faninsar.missions.s1 import acquisition as source

    # Invalidate any stale receipt before modifying this target's pixels.
    owner.completed.pop(str(burst.index), None)
    if owner.progress_path.exists():
        source._atomic_write_json(owner.progress_path, _progress_record(owner))
    task = asyncio.create_task(asyncio.to_thread(_write_burst, owner, burst))
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
            owner.abort.set()
            if task.done():
                # Retrieve the actual worker error; raw bytes remain recoverable.
                await asyncio.gather(task, return_exceptions=True)
                raise
            continue
        except BaseException:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    owner.completed[str(burst.index)] = True
    source._atomic_write_json(owner.progress_path, _progress_record(owner))
    cache = owner.context.span_cache
    assert cache is not None
    # The coordinator's admitted consumers define last use, never download order.
    remaining = {
        cache.span_key(span)
        for b in owner.bursts
        if str(b.index) not in owner.completed
        for span in b.spans
    }
    cache.discard(
        tuple(span for span in burst.spans if cache.span_key(span) not in remaining)
    )
    if len(owner.completed) == len(owner.listed):
        _finish_owner(owner)


def _coverage_bursts(owner: _OwnerPlan) -> list[dict[str, Any]]:
    """Return kept-burst records for ``coverage.json``."""
    by_index = {int(burst.index): burst for burst in owner.swath.bursts}
    records: list[dict[str, Any]] = []
    for index in owner.listed:
        record: dict[str, Any] = {"index": int(index)}
        burst = by_index.get(int(index))
        if burst is not None:
            record["sensing_time"] = burst.sensing_time.isoformat()
            if burst.number is not None:
                record["number"] = int(burst.number)
            if burst.absolute_burst_id is not None:
                record["absolute_burst_id"] = int(burst.absolute_burst_id)
        records.append(record)
    return records


def _finish_owner(owner: _OwnerPlan) -> None:
    """Publish whole-SAFE completion only after every listed burst is durable."""
    from faninsar.missions.s1 import acquisition as source

    assert owner.writer is not None
    owner.writer.finish()
    source._atomic_write_json(
        owner.safe_dir / source._CANVAS_MANIFEST_NAME,
        {
            "complete": True,
            "created": datetime.now(UTC).isoformat(),
            "fingerprint": owner.fingerprint,
            "parent": str(owner.item.get("product_id", "S1_ROI")),
            "swath": str(owner.item.get("swath", "IW1")),
            "polarization": str(owner.item.get("polarization", "VV")),
            "granularity": owner.granularity,
            "bursts": _coverage_bursts(owner),
        },
    )
    owner.progress_path.unlink(missing_ok=True)
    assert owner.context.span_cache is not None
    owner.context.span_cache.finalize()
    source._strip_in_safe_scratch(owner.safe_dir)


def _interleaved_bursts(
    owners: Sequence[_OwnerPlan],
) -> deque[tuple[_OwnerPlan, _BurstPlan]]:
    """Alternate owners without making a date or track a transfer barrier."""
    queues = [
        deque(
            (owner, burst)
            for burst in owner.bursts
            if str(burst.index) not in owner.completed
        )
        for owner in owners
    ]
    result = deque()
    while any(queues):
        for queue in queues:
            if queue:
                result.append(queue.popleft())
    return result


def _incomplete_jobs(owner: _OwnerPlan) -> list[RangeSpan]:
    """Return unfinished measurement spans, registering them in the span cache."""
    cache = owner.context.span_cache
    assert cache is not None
    jobs: list[RangeSpan] = []
    for burst in owner.bursts:
        if str(burst.index) in owner.completed:
            continue
        for span in burst.spans:
            record = cache._record_for(span)
            if record["state"] != "complete":
                jobs.append(span)
    return jobs


async def stream_owners(
    pending: Sequence[tuple[Mapping[str, Any], Path, str]],
    roi: Any,
    *,
    work_dir: Path,
    coreg_mode: str,
    granularity: str,
    dem: Any,
    absolute_burst_ids: Sequence[int] | None = None,
    catalog_sensing_times: Sequence[datetime] | None = None,
) -> list[_OwnerPlan]:
    """Fetch each XML, plan it, and queue its measurement spans on one session.

    Annotation documents, TIFF probes and measurement ranges share one
    :class:`~faninsar.remote.fetch.RangeFetcher` and one worker pool of
    :data:`_MAX_TRANSFER_TASKS` slots.  Workers pull XML and measurement
    jobs 1:1 from that pool so remaining annotations keep pace with bulk
    transfer instead of draining first or waiting behind it.  As soon as
    one XML finishes, that product is planned and its bursts join the
    measurement queue.

    Parameters
    ----------
    pending : sequence of ``(item, safe_dir, fingerprint)``
        Incomplete CDSE bindings still needing a canvas.
    roi : BoundingBox, Points, or Polygons
        Spatial envelope used to plan burst windows.
    work_dir : pathlib.Path
        Stack work directory; transfer scratch lands under ``tmp/s1-transfer``.
    coreg_mode : {"geometry", "nesd"}
        Canvas listing and overlap policy.
    granularity : {"roi", "burst"}
        Acquisition granularity forwarded to window planning.
    dem : RasterDEM or None
        DEM required for ROI-granularity geo2rdr.
    absolute_burst_ids, catalog_sensing_times : optional
        Existing catalog hints associated by sensing time onto listed bursts.
        No additional catalog query is performed.

    Returns
    -------
    list of _OwnerPlan
        Owners that intersected the ROI and were materialized.

    """
    from faninsar.missions.s1 import acquisition as source

    listed = list(pending)
    if not listed:
        return []
    xml_spans = []
    for index, (item, _safe_dir, _fingerprint) in enumerate(listed):
        url = str(item.get("annotation_url", ""))
        size = int(item.get("annotation_size") or source._ANNOTATION_PROBE_BYTES)
        xml_spans.append(RangeSpan(url, 0, max(size - 1, 0), f"annotation-{index}"))
    progress = DownloadProgress(label="CDSE ROI", total_bytes=0, total_spans=0)
    ready: asyncio.Queue[Any] = asyncio.Queue()
    xml_jobs: deque[tuple[int, Mapping[str, Any], Path, str]] = deque(
        (index, item, safe_dir, fingerprint)
        for index, (item, safe_dir, fingerprint) in enumerate(listed)
    )
    meas_jobs: deque[tuple[_OwnerPlan, _BurstPlan, RangeSpan]] = deque()
    burst_left: dict[tuple[int, int], int] = {}
    cond = asyncio.Condition()
    in_xml = 0
    in_meas = 0
    n_planned = 0
    n_items = len(listed)
    results: list[_OwnerPlan | None] = [None] * n_items

    async def run(fetcher: Any) -> list[_OwnerPlan]:
        async def fetch_span(span: RangeSpan) -> bytes:
            bodies = await source.fetch_ranges_async((span,), fetcher=fetcher)
            return bodies[span.label]

        async def download(owner: _OwnerPlan, span: RangeSpan) -> None:
            cache = owner.context.span_cache
            assert cache is not None
            record = cache._records[cache.span_key(span)]
            if record["state"] == "complete":
                return
            label = f"{owner.safe_dir.name}:{span.label}"
            progress.span_start(label)
            await cache.stage(
                span,
                fetcher,
                on_bytes=progress.add_bytes,
                allow_restart=lambda: not owner.completed,
            )
            progress.span_done(label)

        def _idle() -> bool:
            return (
                n_planned == n_items
                and not xml_jobs
                and not meas_jobs
                and in_xml == 0
                and in_meas == 0
            )

        async def take() -> tuple[str, Any] | None:
            nonlocal in_xml, in_meas
            async with cond:
                while True:
                    if xml_jobs and meas_jobs:
                        if in_xml <= in_meas:
                            in_xml += 1
                            return "xml", xml_jobs.popleft()
                        in_meas += 1
                        return "meas", meas_jobs.popleft()
                    if xml_jobs:
                        in_xml += 1
                        return "xml", xml_jobs.popleft()
                    if meas_jobs:
                        in_meas += 1
                        return "meas", meas_jobs.popleft()
                    if _idle():
                        return None
                    await cond.wait()

        async def release(kind: str) -> None:
            nonlocal in_xml, in_meas
            async with cond:
                if kind == "xml":
                    in_xml -= 1
                else:
                    in_meas -= 1
                cond.notify_all()

        async def do_xml(
            index: int,
            item: Mapping[str, Any],
            safe_dir: Path,
            fingerprint: str,
        ) -> None:
            nonlocal n_planned
            # The original annotation is cached locally after the first
            # fetch, so the executed plan can be recomputed without any
            # network round trip on a rerun.
            annotation_cache = _annotation_cache_path(work_dir, safe_dir, item)
            xml_text = _cached_annotation_text(annotation_cache)
            if xml_text is None:
                xml_text = (await fetch_span(xml_spans[index])).decode("utf-8")
                _store_annotation_text(annotation_cache, xml_text)
                logger.info("Fetched annotation XML for %s", safe_dir.name)
            # Keep completion evidence until the actual executed plan is known.
            context = source._RoiFetchContext(
                span_cache=source._span_cache_for_canvas(
                    work_dir, safe_dir, fingerprint
                )
            )
            owner = plan_owner(
                item,
                roi,
                safe_dir=safe_dir,
                fingerprint=fingerprint,
                context=context,
                coreg_mode=coreg_mode,
                granularity=granularity,
                dem=dem,
                xml_text=xml_text,
                plan_spans=False,
                absolute_burst_ids=absolute_burst_ids,
                catalog_sensing_times=catalog_sensing_times,
            )
            ready_bursts: list[tuple[_OwnerPlan, _BurstPlan]] = []
            new_jobs: list[tuple[_OwnerPlan, _BurstPlan, RangeSpan]] = []
            if owner is None:
                assert context.span_cache is not None
                context.span_cache.finalize()
            else:
                await _plan_owner_spans_async(owner, fetcher, work_dir)
                if source._canvas_coverage_complete(
                    safe_dir, owner.fingerprint
                ) and source._safe_canvas_intact(safe_dir):
                    # Same executed plan and intact pixels: nothing to transfer.
                    logger.info("Skipping completed CDSE canvas %s", safe_dir.name)
                    owner.skipped = True
                    source._promote_canvas_coverage(safe_dir, owner.fingerprint, item)
                    source._ensure_canvas_ipf_manifest(safe_dir, item)
                    results[index] = owner
                    async with cond:
                        n_planned += 1
                        cond.notify_all()
                    return
                jobs = _incomplete_jobs(owner)
                if jobs:
                    progress.add_job(
                        nbytes=sum(span.end - span.start + 1 for span in jobs),
                        nspans=len(jobs),
                    )
                cache = owner.context.span_cache
                assert cache is not None
                incomplete = [
                    burst
                    for burst in owner.bursts
                    if str(burst.index) not in owner.completed
                ]
                if not incomplete and len(owner.completed) == len(owner.listed):
                    _finish_owner(owner)
                for burst in incomplete:
                    pending = [
                        span
                        for span in burst.spans
                        if cache._records[cache.span_key(span)]["state"] != "complete"
                    ]
                    if not pending:
                        ready_bursts.append((owner, burst))
                        continue
                    burst_left[id(owner), burst.index] = len(pending)
                    new_jobs.extend((owner, burst, span) for span in pending)
                results[index] = owner
            async with cond:
                n_planned += 1
                meas_jobs.extend(new_jobs)
                cond.notify_all()
            for owner_burst in ready_bursts:
                await ready.put(owner_burst)

        async def do_meas(job: tuple[_OwnerPlan, _BurstPlan, RangeSpan]) -> None:
            owner, burst, span = job
            await download(owner, span)
            key = (id(owner), burst.index)
            async with cond:
                burst_left[key] -= 1
                burst_done = burst_left[key] == 0
            if burst_done:
                await ready.put((owner, burst))

        async def worker() -> None:
            while True:
                job = await take()
                if job is None:
                    return
                kind, payload = job
                try:
                    if kind == "xml":
                        await do_xml(*payload)
                    else:
                        await do_meas(payload)
                finally:
                    await release(kind)

        async def assemble() -> None:
            while True:
                owner, burst = await ready.get()
                try:
                    await _write_ready_burst(owner, burst)
                finally:
                    ready.task_done()

        workers = [
            asyncio.create_task(worker()) for _ in range(max(1, _MAX_TRANSFER_TASKS))
        ]
        consumer = asyncio.create_task(assemble())
        group = asyncio.gather(*workers)
        drain: asyncio.Task[None] | None = None
        try:
            done, _pending = await asyncio.wait(
                (group, consumer),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for future in done:
                future.result()
            drain = asyncio.create_task(ready.join())
            done, _pending = await asyncio.wait(
                (drain, consumer),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for future in done:
                future.result()
            return [owner for owner in results if owner is not None]
        finally:
            closing = [*workers, consumer, group]
            if drain is not None:
                closing.append(drain)
            for task in closing:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*closing, return_exceptions=True)
            progress.finish()

    async with source._span_fetcher_cm(xml_spans) as fetcher:
        return await run(fetcher)


async def transfer_owners(owners: Sequence[_OwnerPlan]) -> None:
    """Download on one asyncio session while a single consumer writes ready bursts."""
    from faninsar.missions.s1 import acquisition as source

    pending = _interleaved_bursts(owners)
    jobs = []
    for owner, burst in pending:
        cache = owner.context.span_cache
        assert cache is not None
        for span in burst.spans:
            record = cache._record_for(span)
            if record["state"] != "complete":
                jobs.append(span)
    for owner in owners:
        if len(owner.completed) == len(owner.listed):
            _finish_owner(owner)
    if not pending:
        return
    progress = DownloadProgress(
        label="CDSE ROI",
        total_bytes=sum(s.end - s.start + 1 for s in jobs),
        total_spans=len(jobs),
    )
    admission = asyncio.Semaphore(_MAX_TRANSFER_TASKS)
    ready: asyncio.Queue[Any] = asyncio.Queue()

    async def run(fetcher: Any) -> None:
        async def download(owner: _OwnerPlan, span: RangeSpan) -> None:
            cache = owner.context.span_cache
            assert cache is not None
            record = cache._records[cache.span_key(span)]
            if record["state"] == "complete":
                return
            label = f"{owner.safe_dir.name}:{span.label}"
            async with admission:
                progress.span_start(label)
                part = cache._part_path(record["key"])
                await fetcher.fetch_into(span, part, on_bytes=progress.add_bytes)
                cache._mark_complete(record, part)
                progress.span_done(label)

        async def burst_worker() -> None:
            while pending:
                owner, burst = pending.popleft()
                spans = iter(burst.spans)

                async def span_worker(
                    owner: _OwnerPlan = owner,
                    spans: Iterator[RangeSpan] = spans,
                ) -> None:
                    for span in spans:
                        await download(owner, span)

                tasks = [
                    asyncio.create_task(span_worker())
                    for _ in range(min(_MAX_TRANSFER_TASKS, len(burst.spans)))
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                done = asyncio.get_running_loop().create_future()
                await ready.put((owner, burst, done))
                await done

        async def assemble() -> None:
            while True:
                owner, burst, done = await ready.get()
                await _write_ready_burst(owner, burst)
                if not done.done():
                    done.set_result(None)

        workers = [
            asyncio.create_task(burst_worker())
            for _ in range(min(_MAX_ACTIVE_BURSTS, len(pending)))
        ]
        consumer = asyncio.create_task(assemble())
        group = asyncio.gather(*workers)
        try:
            done, _ = await asyncio.wait(
                (group, consumer),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for future in done:
                future.result()
        finally:
            for worker in workers:
                worker.cancel()
            consumer.cancel()
            await asyncio.gather(*workers, consumer, return_exceptions=True)
            await asyncio.gather(group, return_exceptions=True)
            progress.finish()

    if jobs:
        async with source._span_fetcher_cm(jobs) as fetcher:
            await run(fetcher)
    else:
        for owner, burst in list(pending):
            await _write_ready_burst(owner, burst)
