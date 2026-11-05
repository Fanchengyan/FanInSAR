"""Verified resumable encoded spans, independent of provider and mission policy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import urllib.parse
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from faninsar.logging import reject, setup_logger
from faninsar.remote.errors import RemoteIntegrityError
from faninsar.remote.fetch.ranges import RangeFetcher, RangeSpan

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import NoReturn

logger = setup_logger(__name__)

_METADATA_DIR_NAMES = frozenset({"annotations", "layouts"})


def _invalid(message: str) -> NoReturn:
    return reject(message, ValueError, logger=logger)


def _scrub_url(url: str) -> str:
    """Strip query/fragment so persisted span records stay OData-canonical."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a tmp sibling + rename so readers never see halves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    if os.name != "nt":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


class SpanCache:
    """Durable per-item span cache under ``tmp/s1-transfer/<canvas>/<F>/``.

    ``manifest.json`` records ``{fingerprint, spans: [{key, url, start,
    end, label, state, part_size, sha1}]}``; ``key`` is
    ``sha1(url|start|end)`` of the scrubbed URL so manifest ``url`` fields
    never carry signed query parameters.  Each span body streams into an
    append-only ``<key>.part`` via ``RangeFetcher.fetch_into``, which
    already resumes from the sink's file size — a crash mid-span leaves a
    partial ``.part`` the next run resumes from ``start + received``.

    A ``state="complete"`` record is honored only when
    ``part_size == actual .part size == end-start+1``; any drift refetches
    the span from byte 0.  The optional ``sha1`` is verified on read-back
    so a same-size corrupted ``.part`` is never mistaken for a clean span.
    Completing the canvas deletes this directory; product coverage lives
    in the SAFE as ``coverage.json``.
    """

    def __init__(self, root: Path, fingerprint: str, *, label: str = "") -> None:
        """Open the cache rooted at ``root / fingerprint``.

        Parameters
        ----------
        root : pathlib.Path
            Per-canvas transfer root (``tmp/s1-transfer/<canvasStem>``).
        fingerprint : str
            Planner fingerprint ``F`` naming this plan's subdirectory.
        label : str, optional
            Human-readable download unit (e.g. the SAFE name) shown in
            progress reports.  Defaults to the parent directory name.

        """
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.dir = self.root / fingerprint
        self.label = label or self.root.parent.name
        self._records: dict[str, dict[str, Any]] = {}
        self._gc_done = False
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._load_manifest()

    @staticmethod
    def span_key(span: RangeSpan) -> str:
        """Return ``sha1(url|start|end)`` of the query-scrubbed span URL."""
        material = f"{_scrub_url(span.url)}|{span.start}|{span.end}"
        return hashlib.sha1(material.encode("utf-8")).hexdigest()

    def _load_manifest(self) -> None:
        """Load span records, tolerating a missing or torn manifest."""
        manifest_path = self.dir / "manifest.json"
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict) or raw.get("fingerprint") != self.fingerprint:
            return
        spans = raw.get("spans")
        if not isinstance(spans, list):
            return
        for record in spans:
            if isinstance(record, dict) and isinstance(record.get("key"), str):
                self._records[record["key"]] = dict(record)

    def _save_manifest(self) -> None:
        """Atomically persist the current span records."""
        payload = {
            "fingerprint": self.fingerprint,
            "spans": list(self._records.values()),
        }
        _atomic_write_json(self.dir / "manifest.json", payload)

    def _part_path(self, key: str) -> Path:
        """Return the ``.part`` path for one span key."""
        return self.dir / f"{key}.part"

    def _record_for(self, span: RangeSpan) -> dict[str, Any]:
        """Return the reconciled record for ``span``, honoring resume rules."""
        key = self.span_key(span)
        expected = span.end - span.start + 1
        part = self._part_path(key)
        part_size = part.stat().st_size if part.exists() else 0
        record = self._records.get(key)
        if record is not None:
            if record.get("state") == "complete":
                honored = record.get(
                    "part_size"
                ) == part_size == expected and self._sha1_ok(record, part)
                if honored:
                    return record
                # Drift or corruption: a recorded-complete span whose bytes
                # no longer verify is refetched from byte 0.
                part.unlink(missing_ok=True)
                part_size = 0
                record = None
            else:
                # Pending record: the .part file size is the progress record.
                if part_size > expected:
                    part.unlink(missing_ok=True)
                    part_size = 0
                record["part_size"] = part_size
                record["state"] = "pending"
                return record
        if part_size > expected:
            part.unlink(missing_ok=True)
            part_size = 0
        record = {
            "key": key,
            "url": _scrub_url(span.url),
            "start": span.start,
            "end": span.end,
            "label": span.label,
            "state": "pending",
            "part_size": part_size,
        }
        self._records[key] = record
        return record

    def _sha1_ok(self, record: Mapping[str, Any], part: Path) -> bool:
        """Verify a recorded sha1 against the current ``.part`` bytes."""
        recorded = record.get("sha1")
        if not isinstance(recorded, str) or not recorded:
            return True
        try:
            with part.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha1").hexdigest()
        except OSError:
            return False
        return digest == recorded

    def _mark_complete(self, record: dict[str, Any], part: Path) -> None:
        """Record one span complete after ``fetch_into`` returned."""
        size = part.stat().st_size
        expected = int(record["end"]) - int(record["start"]) + 1
        if size != expected:
            _invalid(f"span cache part length {size} differs from expected {expected}")
        record["part_size"] = size
        with part.open("rb") as stream:
            record["sha1"] = hashlib.file_digest(stream, "sha1").hexdigest()
        record["state"] = "complete"
        self._save_manifest()

    def _read_part(self, span: RangeSpan, record: Mapping[str, Any]) -> bytes:
        """Read back one completed span's ``.part`` bytes."""
        del span
        part = self._part_path(record["key"])
        try:
            return part.read_bytes()
        except OSError:
            return _invalid(f"span cache part file unreadable: {part}")

    def _gc_orphans_once(self, planned: Sequence[RangeSpan]) -> None:
        """Remove orphan ``<F'>`` dirs, keeping plan-overlapping partials.

        A pending partial whose span overlaps the current plan's
        ``(url, start, end)`` set still carries usable bytes and is kept;
        every other orphan tree is removed so bandwidth-regime flips and
        planner-version bumps cannot grow the disk unboundedly.
        """
        if self._gc_done or not self.root.is_dir():
            self._gc_done = True
            return
        planned_keys = {self.span_key(span) for span in planned}
        planned_ranges = [(_scrub_url(s.url), s.start, s.end) for s in planned]
        for child in self.root.iterdir():
            # Sibling metadata caches (original annotations, TIFF strip
            # tables) are durable plan inputs, not orphan transfer states.
            if not child.is_dir() or child.name == self.fingerprint:
                continue
            if child.name in _METADATA_DIR_NAMES:
                continue
            if self._orphan_overlaps(child, planned_keys, planned_ranges):
                logger.info(
                    "keeping orphan span cache %s: pending partial overlaps "
                    "the current plan",
                    child,
                )
                continue
            logger.info("removing orphan span cache %s", child)
            shutil.rmtree(child, ignore_errors=True)
        self._gc_done = True

    @staticmethod
    def _orphan_overlaps(
        orphan: Path,
        planned_keys: set[str],
        planned_ranges: Sequence[tuple[str, int, int]],
    ) -> bool:
        """Return True when ``orphan`` holds a pending partial in the plan."""
        try:
            raw = json.loads((orphan / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = None
        if isinstance(raw, dict) and isinstance(raw.get("spans"), list):
            for record in raw["spans"]:
                if not isinstance(record, dict) or record.get("state") == "complete":
                    continue
                if record.get("key") in planned_keys:
                    return True
                url = str(record.get("url", ""))
                try:
                    start = int(record.get("start", -1))
                    end = int(record.get("end", -2))
                except (TypeError, ValueError):
                    continue
                for p_url, p_start, p_end in planned_ranges:
                    if url == p_url and start <= p_end and end >= p_start:
                        return True
        # Manifest missing/torn: a .part whose filename key is in the plan
        # still carries resumable bytes.
        return any(part.stem in planned_keys for part in orphan.glob("*.part"))

    async def stage(
        self,
        span: RangeSpan,
        fetcher: RangeFetcher,
        *,
        on_bytes: Callable[[int], None] | None = None,
        allow_restart: Callable[[], bool] = lambda: True,
    ) -> None:
        """Stage one exact region using the caller's provider-configured fetcher.

        Concurrent requests for the same region share one producer. A changed
        strong validator abandons the entire old prefix before one bounded
        retry; callers can forbid restart after related targets have committed.
        """
        key = self.span_key(span)
        existing = self._tasks.get(key)
        if existing is not None:
            await asyncio.shield(existing)
            return

        async def produce() -> None:
            record = self._record_for(span)
            if record["state"] == "complete":
                return
            part = self._part_path(key)
            # Persist identity before any payload can arrive, including aborts.
            self._save_manifest()

            def learned(etag: str) -> None:
                record["etag"] = etag
                self._save_manifest()

            for attempt in range(2):
                options: dict[str, Any] = {"on_bytes": on_bytes}
                # Legacy injected fetchers need no new keywords until qualified.
                if isinstance(fetcher, RangeFetcher):
                    options.update(etag=record.get("etag"), on_etag=learned)
                try:
                    await fetcher.fetch_into(span, part, **options)
                except RemoteIntegrityError as error:
                    if error.reason != "revision_changed":
                        raise
                    part.unlink(missing_ok=True)
                    record.update(state="pending", part_size=0)
                    record.pop("etag", None)
                    record.pop("sha1", None)
                    self._save_manifest()
                    if attempt or not allow_restart():
                        raise
                    continue
                self._mark_complete(record, part)
                return

        task = asyncio.create_task(produce())
        self._tasks[key] = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            self._tasks.pop(key, None)

    def reader(self, spans: Sequence[RangeSpan]) -> Callable[[int, int], bytes]:
        """Verify staged regions and return a bounded absolute-offset local reader.

        SHA1 payload integrity is checked without loading whole spans. Reads
        outside the verified region set fail before touching disk.
        """
        regions = []
        for span in spans:
            record = self._record_for(span)
            if record["state"] != "complete":
                _invalid("local range reader requires verified complete spans")
            regions.append((span.start, span.end + 1, self._part_path(record["key"])))

        def read_at(offset: int, length: int) -> bytes:
            if length < 0:
                _invalid("local range read length must be nonnegative")
            for start, end, part in regions:
                if start <= offset and offset + length <= end:
                    with part.open("rb") as stream:
                        stream.seek(offset - start)
                        payload = stream.read(length)
                    if len(payload) != length:
                        _invalid("verified span was truncated during local reading")
                    return payload
            return _invalid("local range read is outside the staged extents")

        return read_at

    def discard(self, spans: Sequence[RangeSpan]) -> None:
        """Release last-use encoded regions after the caller's durable checkpoint."""
        for span in spans:
            try:
                self._part_path(self.span_key(span)).unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not reclaim committed span %s", span.label, exc_info=True
                )

    def finalize(self) -> None:
        """Drop this fingerprint's transfer directory after the canvas is durable."""
        shutil.rmtree(self.dir, ignore_errors=True)
        with suppress(OSError):
            self.root.rmdir()
