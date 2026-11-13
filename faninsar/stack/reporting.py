"""Durable diagnostic reports for Stack coregistration attempts (PROPOSAL-0088).

Reports are diagnostics, never completion authority. A report describes what an
attempt did or failed to do; durable product receipts and intent remain the
authority for reuse. Each attempt owns one directory::

    <work_dir>/reports/coregistration/<attempt-id>/report.json
    <work_dir>/reports/coregistration/<attempt-id>/report.md

Both files derive from one :class:`CoregistrationReport` model. ``report.json``
is the sole authority snapshot and is replaced atomically; ``report.md`` is a
derived view and is republished after the JSON snapshot. A JSON write failure
leaves the previous snapshot and the in-memory model unchanged and raises
:class:`ReportWriteError`. A Markdown-only failure never hides a saved JSON
snapshot; it is logged and exposed through
:attr:`ReportWriter.markdown_error`. Loading is read-only and does not create or
claim any writer lease: an unfinished snapshot stays ``"running"`` unless the
caller explicitly reconciles it while holding the workspace writer lock.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from faninsar.logging import reject, setup_logger

logger = setup_logger(__name__)

REPORT_SCHEMA = "faninsar.coregistration.report.v1"
"""Schema identifier persisted in every report snapshot."""

REPORTS_DIRNAME = "reports"
"""Workspace subtree holding diagnostic reports."""

COREGISTRATION_REPORTS_DIRNAME = "coregistration"
"""Per-attempt coregistration report subtree."""

REPORT_JSON_NAME = "report.json"
"""Machine-readable snapshot name inside one attempt directory."""

REPORT_MARKDOWN_NAME = "report.md"
"""Human-readable snapshot name inside one attempt directory."""

RecordStatus: TypeAlias = Literal[
    "completed",
    "reused",
    "failed",
    "blocked",
    "pending",
    "interrupted",
    "skipped",
]
ReportOutcome: TypeAlias = Literal[
    "running",
    "complete",
    "partial",
    "failed",
    "blocked",
    "interrupted",
    "skipped",
    "no_work",
]

_RECORD_STATUSES = frozenset(
    {
        "completed",
        "reused",
        "failed",
        "blocked",
        "pending",
        "interrupted",
        "skipped",
    }
)
_FINISH_OUTCOMES = frozenset(
    {"complete", "partial", "failed", "blocked", "skipped", "no_work"}
)
_REPORT_OUTCOMES = _FINISH_OUTCOMES | {"running", "interrupted"}
_INCOMPLETE_STATUSES = frozenset(
    {"failed", "blocked", "interrupted", "pending", "skipped"}
)
_ATTEMPT_ID_PATTERN = re.compile(r"^\d{8}T\d{6}\d{6}Z-[0-9a-f]{8}$")
_STRING_RECORD_FIELDS = (
    "stage",
    "unit_id",
    "acquisition",
    "date",
    "roi",
    "reason_code",
    "dependency",
    "retry",
    "output",
    "log",
)
_STRING_REPORT_FIELDS = ("mode", "grid", "reference", "verification")
_SECRET_AUTHORIZATION = re.compile(
    r"(?i)\b(?:authorization|proxy-authorization)\s*[:=]\s*"
    r"(?:(?:bearer|basic)\s+)?\S+"
)
_SECRET_BEARER = re.compile(r"(?i)\b(?:bearer|basic)\s+\S+")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:token|password|passwd|secret|signature|credential|api[-_]?key)\b"
    r"\s*[:=]\s*\S+"
)
_URL_IN_TEXT = re.compile(r"(?i)\b[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"<>]+")


class ReportWriteError(OSError):
    """A diagnostic snapshot could not be published."""


def _utc_now() -> str:
    """Return the current time as a UTC ISO-8601 string."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize(value: Any) -> Any:  # noqa: PLR0911
    """Convert supported Python values into JSON-compatible data recursively."""
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_normalize(item) for item in sorted(value, key=str)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _redact_url(match: re.Match[str]) -> str:
    """Remove userinfo and signed query material from one embedded URL."""
    from faninsar.remote.records import _sanitize

    sanitized = _sanitize(match.group(0))
    return sanitized if isinstance(sanitized, str) else "[REDACTED]"


def _redact_text(value: str) -> str:
    """Remove credential-bearing URLs and exception-like secret text."""
    text = _URL_IN_TEXT.sub(_redact_url, value)
    text = _SECRET_AUTHORIZATION.sub("[REDACTED]", text)
    text = _SECRET_BEARER.sub("[REDACTED]", text)
    return _SECRET_ASSIGNMENT.sub("[REDACTED]", text)


def _redact_strings(value: Any) -> Any:
    """Apply text redaction to every string in a normalized structure."""
    if isinstance(value, Mapping):
        return {str(key): _redact_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_strings(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact(value: Any) -> Any:
    """Normalize and sanitize one value before it enters a report model."""
    from faninsar.remote.records import _sanitize

    return _redact_strings(_sanitize(_normalize(value)))


def _deep_freeze(value: Any) -> Any:
    """Return a recursively immutable copy of JSON-compatible data."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    """Return a mutable JSON-compatible copy of an immutable model value."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _redacted_message(value: Any) -> Any:
    """Sanitize and recursively freeze one free-form diagnostic message."""
    return _deep_freeze(_redact(value))


def _redacted_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Sanitize and freeze an optional report metadata mapping."""
    if value is None:
        return None
    redacted = _redact(value)
    if not isinstance(redacted, Mapping):
        reject(
            "coregistration report metadata must be a mapping",
            ValueError,
            logger=logger,
        )
    return _deep_freeze(redacted)


def _cell(value: Any) -> str:
    """Escape one value for a Markdown table cell."""
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _derive_outcome(records: tuple[CoregistrationRecord, ...]) -> str:
    """Derive a final outcome from recorded dispositions without inventing facts."""
    if not records:
        return "no_work"
    statuses = {record.status for record in records}
    if statuses <= {"failed"}:
        return "failed"
    if statuses <= {"blocked"}:
        return "blocked"
    if statuses <= {"skipped"}:
        return "skipped"
    if statuses & _INCOMPLETE_STATUSES:
        return "partial"
    return "complete"


@dataclass(frozen=True)
class CoregistrationRecord:
    """One stable unit's disposition inside a coregistration attempt.

    Parameters
    ----------
    stage : str
        Stage name such as ``"prepare"``, ``"geometry"``, or
        ``"coregistration"``.
    unit_id : str
        Stable unit identity; never a provider result position.
    status : {"completed", "reused", "failed", "blocked", "pending", \
"interrupted", "skipped"}
        Disposition of this unit in the attempt.
    acquisition, date, burst, roi, pair : optional
        Stable scope metadata. ``pair`` is a pair label or a two-date tuple.
    reason_code, message, dependency, retry : optional
        Diagnostic facts: machine reason, sanitized human message, root
        prerequisite, and retry advice.
    output, log : optional
        Output identity or path and a workspace-relative log pointer.
    quality : mapping, optional
        Applicable real quality evidence. Unavailable metrics stay absent.

    """

    stage: str
    unit_id: str
    status: RecordStatus
    acquisition: str | None = None
    date: str | None = None
    burst: int | str | None = None
    roi: str | None = None
    pair: str | tuple[str, str] | None = None
    reason_code: str | None = None
    message: str | BaseException | Mapping[str, Any] | None = None
    dependency: str | None = None
    retry: str | None = None
    output: str | None = None
    quality: Mapping[str, Any] | None = None
    log: str | None = None

    def __post_init__(self) -> None:
        """Validate disposition and sanitize every free-form value."""
        if self.status not in _RECORD_STATUSES:
            reject(
                f"unsupported coregistration record status {self.status!r}",
                ValueError,
                logger=logger,
            )
        for name in _STRING_RECORD_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _redact_text(str(value)))
        if isinstance(self.burst, str):
            object.__setattr__(self, "burst", _redact_text(self.burst))
        if isinstance(self.pair, tuple):
            object.__setattr__(
                self,
                "pair",
                tuple(_redact_text(str(item)) for item in self.pair),
            )
        elif self.pair is not None:
            object.__setattr__(self, "pair", _redact_text(str(self.pair)))
        if self.message is not None:
            object.__setattr__(self, "message", _redacted_message(self.message))
        if self.quality is not None:
            object.__setattr__(self, "quality", _redacted_mapping(self.quality))

    def to_dict(self) -> dict[str, Any]:
        """Encode this record as JSON-compatible data."""
        return {
            "stage": self.stage,
            "unit_id": self.unit_id,
            "status": self.status,
            "acquisition": self.acquisition,
            "date": self.date,
            "burst": self.burst,
            "roi": self.roi,
            "pair": _jsonable(self.pair),
            "reason_code": self.reason_code,
            "message": _jsonable(self.message),
            "dependency": self.dependency,
            "retry": self.retry,
            "output": self.output,
            "quality": _jsonable(self.quality),
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> CoregistrationRecord:
        """Decode one persisted record, rejecting malformed keys."""
        pair = document.get("pair")
        if isinstance(pair, list):
            pair = tuple(pair)
        return cls(
            stage=document["stage"],
            unit_id=document["unit_id"],
            status=document["status"],
            acquisition=document.get("acquisition"),
            date=document.get("date"),
            burst=document.get("burst"),
            roi=document.get("roi"),
            pair=pair,
            reason_code=document.get("reason_code"),
            message=document.get("message"),
            dependency=document.get("dependency"),
            retry=document.get("retry"),
            output=document.get("output"),
            quality=document.get("quality"),
            log=document.get("log"),
        )


@dataclass(frozen=True)
class CoregistrationReport:
    """One durable coregistration attempt and its current reconciled view.

    The persisted ``outcome`` is the attempt's reported outcome. A snapshot with
    no ``finished_at`` is an unfinished attempt. Loading it keeps
    ``outcome="running"`` and ``unfinished=True``; only an explicit
    reconciliation under the owning workspace lock sets ``interrupted=True`` and
    ``current_outcome="interrupted"``. The stored ``outcome`` remains
    ``"running"``.
    """

    attempt_id: str
    intent_revision: int | str
    mode: str
    grid: str
    reference: str | None
    verification: str
    created_at: str
    updated_at: str
    outcome: ReportOutcome = "running"
    finished_at: str | None = None
    records: tuple[CoregistrationRecord, ...] = ()
    scope: Mapping[str, Any] | None = None
    revisions: Mapping[str, str] | None = None
    message: str | Mapping[str, Any] | None = None
    interrupted: bool = False

    def __post_init__(self) -> None:
        """Validate the outcome and sanitize report context and metadata."""
        if self.outcome not in _REPORT_OUTCOMES:
            reject(
                f"unsupported coregistration report outcome {self.outcome!r}",
                ValueError,
                logger=logger,
            )
        object.__setattr__(self, "records", tuple(self.records))
        for name in _STRING_REPORT_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _redact_text(str(value)))
        object.__setattr__(self, "scope", _redacted_mapping(self.scope))
        object.__setattr__(self, "revisions", _redacted_mapping(self.revisions))
        if self.message is not None:
            object.__setattr__(self, "message", _redacted_message(self.message))

    @property
    def counts(self) -> dict[str, int]:
        """Return nonzero record counts by disposition."""
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    @property
    def stage_counts(self) -> dict[str, dict[str, int]]:
        """Return nonzero record counts grouped by stage then disposition."""
        counts: dict[str, dict[str, int]] = {}
        for record in self.records:
            stage = counts.setdefault(record.stage, {})
            stage[record.status] = stage.get(record.status, 0) + 1
        return counts

    @property
    def unfinished(self) -> bool:
        """Whether the attempt has no recorded finalization timestamp."""
        return self.finished_at is None

    @property
    def current_outcome(self) -> str:
        """Return the view, interrupted only after explicit reconciliation."""
        return "interrupted" if self.interrupted else self.outcome

    def to_dict(self) -> dict[str, Any]:
        """Encode this report as JSON-compatible data."""
        return {
            "schema": REPORT_SCHEMA,
            "attempt_id": self.attempt_id,
            "intent_revision": self.intent_revision,
            "mode": self.mode,
            "grid": self.grid,
            "reference": self.reference,
            "verification": self.verification,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "counts": self.counts,
            "stage_counts": self.stage_counts,
            "scope": _jsonable(self.scope),
            "revisions": _jsonable(self.revisions),
            "message": _jsonable(self.message),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> CoregistrationReport:
        """Decode one persisted report snapshot.

        Raises
        ------
        ValueError
            If the document is not a supported report schema.

        """
        if not isinstance(document, Mapping):
            reject(
                "coregistration report snapshot must be a mapping",
                ValueError,
                logger=logger,
            )
        if document.get("schema") != REPORT_SCHEMA:
            reject(
                f"unsupported coregistration report schema {document.get('schema')!r}",
                ValueError,
                logger=logger,
            )
        records = tuple(
            CoregistrationRecord.from_dict(item) for item in document.get("records", ())
        )
        return cls(
            attempt_id=document["attempt_id"],
            intent_revision=document["intent_revision"],
            mode=document["mode"],
            grid=document["grid"],
            reference=document["reference"],
            verification=document["verification"],
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            outcome=document.get("outcome", "running"),
            finished_at=document.get("finished_at"),
            records=records,
            scope=document.get("scope"),
            revisions=document.get("revisions"),
            message=document.get("message"),
        )

    def to_markdown(self) -> str:
        """Render the same model as a human-readable Markdown snapshot."""
        outcome = (
            f"{self.current_outcome} (stored outcome: {self.outcome})"
            if self.interrupted
            else self.outcome
        )
        lines = [
            f"# Coregistration report {self.attempt_id}",
            "",
            f"- Schema: `{REPORT_SCHEMA}`",
            f"- Intent revision: `{self.intent_revision}`",
            f"- Mode: `{self.mode}`",
            f"- Grid: `{self.grid}`",
            f"- Reference: `{self.reference}`",
            f"- Verification: `{self.verification}`",
            f"- Created: `{self.created_at}`",
            f"- Updated: `{self.updated_at}`",
            f"- Finished: `{self.finished_at}`",
            f"- Outcome: **{outcome}**",
            f"- Counts: `{json.dumps(self.counts, sort_keys=True)}`",
            f"- Stage counts: `{json.dumps(self.stage_counts, sort_keys=True)}`",
        ]
        if self.scope:
            scope = json.dumps(_jsonable(self.scope), sort_keys=True)
            lines.append(f"- Scope: `{scope}`")
        if self.revisions:
            revisions = json.dumps(_jsonable(self.revisions), sort_keys=True)
            lines.append(f"- Revisions: `{revisions}`")
        if self.message:
            lines.append(f"- Message: {_cell(_jsonable(self.message))}")
        lines.extend(["", "## Records", ""])
        if not self.records:
            lines.append("No records were reported.")
        else:
            lines.append(
                "| Stage | Unit | Status | Acquisition | Date | Burst | Pair | "
                "Reason | Message | Output |"
            )
            lines.append(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            )
            for record in self.records:
                pair = (
                    "-".join(record.pair)
                    if isinstance(record.pair, tuple)
                    else record.pair
                )
                lines.append(
                    "| "
                    + " | ".join(
                        _cell(value)
                        for value in (
                            record.stage,
                            record.unit_id,
                            record.status,
                            record.acquisition,
                            record.date,
                            record.burst,
                            pair,
                            record.reason_code,
                            _jsonable(record.message),
                            record.output,
                        )
                    )
                    + " |"
                )
        lines.append("")
        return "\n".join(lines)


def _new_attempt_id() -> str:
    """Return a lexically sortable, unique attempt directory name."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _write_and_sync(path: Path, text: str) -> None:
    """Write one temporary snapshot and flush it to durable storage."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Flush directory entries so a completed replace survives a crash."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write one text file atomically, preserving any prior version on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        _write_and_sync(temporary, text)
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_snapshot(json_path: Path, report: CoregistrationReport) -> None:
    """Publish the authoritative JSON snapshot atomically."""
    json_text = (
        json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    _atomic_write_text(json_path, json_text)


def _write_markdown_snapshot(markdown_path: Path, report: CoregistrationReport) -> None:
    """Publish the derived Markdown view after the JSON authority snapshot."""
    _atomic_write_text(markdown_path, report.to_markdown())


class ReportWriter:
    """Own one durable coregistration attempt and publish its snapshots.

    Parameters
    ----------
    work_dir : path-like
        Stack work directory receiving ``reports/coregistration/``.
    intent_revision, mode, grid, reference, verification : str or int
        Attempt context copied into every snapshot.
    scope, revisions : mapping, optional
        Optional scope and semantic revision evidence. Values are sanitized and
        frozen before publication.

    Notes
    -----
    The writer publishes a ``"running"`` snapshot at construction. Every
    :meth:`record` and :meth:`finish` call first replaces the authoritative
    ``report.json`` snapshot atomically, then republishes the derived
    ``report.md``. A JSON failure raises :class:`ReportWriteError` and leaves
    both the prior JSON file and :attr:`report` unchanged. A Markdown-only
    failure is logged and exposed through :attr:`markdown_error`; the JSON
    snapshot remains saved and :attr:`report` advances. This class never reads
    products and never decides scientific completion.

    """

    def __init__(
        self,
        work_dir: str | Path,
        *,
        intent_revision: int | str,
        mode: str,
        grid: str,
        reference: str | None,
        verification: str,
        scope: Mapping[str, Any] | None = None,
        revisions: Mapping[str, str] | None = None,
    ) -> None:
        """Create the attempt directory and publish its running snapshot."""
        self._work_dir = Path(work_dir)
        self._attempt_id = _new_attempt_id()
        attempt_dir = (
            self._work_dir
            / REPORTS_DIRNAME
            / COREGISTRATION_REPORTS_DIRNAME
            / self._attempt_id
        )
        self._path = attempt_dir / REPORT_JSON_NAME
        self._markdown_path = attempt_dir / REPORT_MARKDOWN_NAME
        self._markdown_error: str | None = None
        now = _utc_now()
        report = CoregistrationReport(
            attempt_id=self._attempt_id,
            intent_revision=intent_revision,
            mode=mode,
            grid=grid,
            reference=reference,
            verification=verification,
            created_at=now,
            updated_at=now,
            outcome="running",
            scope=scope,
            revisions=revisions,
        )
        self._publish(report)
        self._report = report

    @property
    def report(self) -> CoregistrationReport:
        """Return the last successfully published attempt snapshot."""
        return self._report

    @property
    def path(self) -> Path:
        """Return the attempt's ``report.json`` path."""
        return self._path

    @property
    def markdown_path(self) -> Path:
        """Return the attempt's ``report.md`` path."""
        return self._markdown_path

    @property
    def markdown_error(self) -> str | None:
        """Return the latest derived-Markdown failure, if any.

        The JSON snapshot remains the authority. This value is informational: a
        non-``None`` string means the last JSON snapshot was saved but its
        Markdown view was not successfully republished.
        """
        return self._markdown_error

    def record(
        self,
        *,
        stage: str,
        unit_id: str,
        status: RecordStatus,
        acquisition: str | None = None,
        date: str | None = None,
        burst: int | str | None = None,
        roi: str | None = None,
        pair: str | tuple[str, str] | None = None,
        reason_code: str | None = None,
        message: str | BaseException | Mapping[str, Any] | None = None,
        dependency: str | None = None,
        retry: str | None = None,
        output: str | None = None,
        quality: Mapping[str, Any] | None = None,
        log: str | None = None,
    ) -> CoregistrationRecord:
        """Publish one unit disposition and return the immutable record."""
        if self._report.finished_at is not None:
            reject(
                "cannot record into a finished coregistration report",
                RuntimeError,
                logger=logger,
            )
        record = CoregistrationRecord(
            stage=stage,
            unit_id=unit_id,
            status=status,
            acquisition=acquisition,
            date=date,
            burst=burst,
            roi=roi,
            pair=pair,
            reason_code=reason_code,
            message=message,
            dependency=dependency,
            retry=retry,
            output=output,
            quality=quality,
            log=log,
        )
        report = replace(
            self._report,
            records=(*self._report.records, record),
            updated_at=_utc_now(),
            outcome="running",
        )
        self._publish(report)
        self._report = report
        return record

    def finish(
        self,
        *,
        outcome: str | None = None,
        message: str | BaseException | Mapping[str, Any] | None = None,
    ) -> CoregistrationReport:
        """Finalize the attempt, deriving the outcome from records by default.

        An empty attempt derives ``"no_work"`` rather than claiming success.
        All-failed, all-blocked, and all-skipped attempts derive their matching
        outcome. Any mix containing failed, blocked, interrupted, pending, or
        skipped work derives ``"partial"``; only fully completed/reused work
        derives ``"complete"``.
        """
        if self._report.finished_at is not None:
            return self._report
        resolved = _derive_outcome(self._report.records) if outcome is None else outcome
        if resolved not in _FINISH_OUTCOMES:
            reject(
                f"unsupported coregistration finish outcome {resolved!r}",
                ValueError,
                logger=logger,
            )
        now = _utc_now()
        report = replace(
            self._report,
            outcome=resolved,
            finished_at=now,
            updated_at=now,
            message=self._report.message if message is None else message,
        )
        self._publish(report)
        self._report = report
        return report

    @classmethod
    def load_latest(
        cls, work_dir: str | Path, *, reconcile_interrupted: bool = False
    ) -> CoregistrationReport | None:
        """Load the newest readable attempt without creating a writer lease.

        An attempt whose snapshot has no ``finished_at`` is unfinished. By
        default it is returned unchanged with ``outcome="running"``,
        ``unfinished=True``, and ``interrupted=False``: reading a report cannot
        prove that another live writer died. Pass ``reconcile_interrupted=True``
        only while holding the workspace writer lock and after confirming no
        writer owns the live attempt; then the returned view has
        ``interrupted=True`` and ``current_outcome="interrupted"`` while the
        stored ``outcome`` stays ``"running"``. OS writer-lock ownership remains
        the only authority on whether a writer is alive.

        Only directories named like a writer-generated attempt are considered,
        and a snapshot must declare the attempt id of the directory holding it.
        Foreign or misplaced directories are skipped rather than treated as the
        latest attempt.
        """
        root = Path(work_dir) / REPORTS_DIRNAME / COREGISTRATION_REPORTS_DIRNAME
        if not root.is_dir():
            return None
        attempts = sorted(
            (
                entry
                for entry in root.iterdir()
                if entry.is_dir() and _ATTEMPT_ID_PATTERN.fullmatch(entry.name)
            ),
            key=lambda entry: entry.name,
            reverse=True,
        )
        for attempt in attempts:
            snapshot = attempt / REPORT_JSON_NAME
            if not snapshot.is_file():
                continue
            try:
                document = json.loads(snapshot.read_text(encoding="utf-8"))
                report = CoregistrationReport.from_dict(document)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "skipping unreadable coregistration report %s: %s", snapshot, exc
                )
                continue
            if report.attempt_id != attempt.name:
                logger.warning(
                    "skipping coregistration report %s: attempt id %r does not "
                    "match its directory",
                    snapshot,
                    report.attempt_id,
                )
                continue
            if reconcile_interrupted and report.unfinished:
                report = replace(report, interrupted=True)
            return report
        return None

    def _publish(self, report: CoregistrationReport) -> None:
        """Publish the JSON authority, then truthfully republish Markdown."""
        try:
            _write_json_snapshot(self._path, report)
        except (OSError, TypeError, ValueError) as exc:
            reject(
                f"failed to write coregistration report {self._path}: {exc}",
                ReportWriteError,
                logger=logger,
                cause=exc,
            )
        try:
            _write_markdown_snapshot(self._markdown_path, report)
        except Exception as exc:
            message = _redact_text(f"failed to write {self._markdown_path}: {exc}")
            self._markdown_error = message
            logger.error(  # noqa: TRY400 - sanitized text; no traceback secrets
                "coregistration report JSON saved but Markdown failed: %s", message
            )
        else:
            self._markdown_error = None


__all__ = [
    "COREGISTRATION_REPORTS_DIRNAME",
    "REPORTS_DIRNAME",
    "REPORT_JSON_NAME",
    "REPORT_MARKDOWN_NAME",
    "REPORT_SCHEMA",
    "CoregistrationRecord",
    "CoregistrationReport",
    "ReportWriteError",
    "ReportWriter",
]
