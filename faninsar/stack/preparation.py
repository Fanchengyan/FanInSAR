"""Explicit, recoverable Stack input preparation (PROPOSAL-0073)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from faninsar.logging import reject, setup_logger
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.search import MissionPreparedInputs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from faninsar.remote.records import RemoteResourceBudget
    from faninsar.stack.session import Stack

logger = setup_logger(__name__)
REQUEST_SCHEMA = "faninsar.preparation.v1"


def topology_request(
    reference: Any,
    pairs: Any,
    misreg_pairs: Any,
    pair_max_interval: int,
    pair_max_days: int,
    misreg_max_interval: int,
    misreg_max_days: int,
) -> dict[str, Any]:
    """Capture explicit networks separately from default ready-set policies."""
    from faninsar.stack.session import _date_to_yyyymmdd, _iter_pair_dates

    return {
        "reference": None if reference is None else _date_to_yyyymmdd(reference),
        "pairs": None
        if pairs is None
        else tuple(f"{a}_{b}" for a, b in _iter_pair_dates(pairs)),
        "misreg_pairs": None
        if misreg_pairs is None
        else tuple(f"{a}_{b}" for a, b in _iter_pair_dates(misreg_pairs)),
        "pair_max_interval": pair_max_interval,
        "pair_max_days": pair_max_days,
        "misreg_max_interval": misreg_max_interval,
        "misreg_max_days": misreg_max_days,
    }


def freeze(value: Any) -> Any:
    """Capture a recursively immutable request snapshot."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    """Return a mutable copy suitable for mission normalization."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(thaw(item) for item in value)
    return value


@dataclass(frozen=True)
class PreparationFailure:
    """An expected unavailable target and its preparation phase."""

    target: str
    phase: str
    reason: str
    cause: Exception | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class PreparationReport:
    """Original selection, qualified membership and unavailable target facts."""

    requested: tuple[str, ...]
    ready: tuple[str, ...]
    failed: tuple[PreparationFailure, ...] = ()
    reused: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()

    @property
    def outcome(self) -> Literal["complete", "partial", "failed"]:
        """Return completeness independently of usable subset readiness."""
        if not self.ready:
            return "failed"
        return "partial" if self.failed else "complete"

    @property
    def complete(self) -> bool:
        """Whether every selected target qualified."""
        return self.outcome == "complete"


class InputUnavailableError(OSError, RuntimeError):
    """An expected mission dependency could not be acquired or located."""


class DataPreparationError(RuntimeError):
    """Strict preparation failed; ``report`` retains qualified target facts."""

    def __init__(self, report: PreparationReport) -> None:
        """Attach the structured report without disclosing source secrets."""
        self.report = report
        super().__init__(
            f"Data preparation {report.outcome}: {len(report.failed)} failed targets"
        )


@dataclass(frozen=True)
class PreparationContext:
    """Immutable execution policy and original science/source snapshots."""

    request: Mapping[str, Any]
    science: Mapping[str, Any]
    offline: bool = False
    budget: RemoteResourceBudget | None = None
    previous: PreparationReport | None = None
    on_ready: Callable[[PreparedInputs], None] | None = None
    """Mission-owned per-date handoff invoked after SLC+EOF+AUX qualification.

    The mission calls this on the controlling acquisition thread, so the
    callback may publish an incremental snapshot and run bounded science for
    the date while other acquisition futures stay in flight.  ``None`` keeps
    the historical all-dates-then-finalize path.
    """
    previous_document: Mapping[str, Any] | None = None
    """Legacy request snapshot from the previous preparation, if any.

    Missions compare their per-target plan identity against this snapshot
    before reusing an already-qualified path, so an additive ROI/burst
    extension for the same date re-plans instead of trusting old coverage.
    """


@dataclass(frozen=True)
class PreparedInputs:
    """Mission handoff of qualified local sources and mission-owned bindings."""

    sources: Mapping[str, tuple[Path, ...]]
    requested: tuple[str, ...] = ()
    failures: tuple[PreparationFailure, ...] = ()
    mission: MissionPreparedInputs = field(default_factory=MissionPreparedInputs)
    extra: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    scene_provider: Any = None

    def __post_init__(self) -> None:
        """Freeze membership and configuration handed to finalization."""
        object.__setattr__(
            self,
            "sources",
            freeze(
                {
                    key: tuple(Path(p) for p in paths)
                    for key, paths in self.sources.items()
                }
            ),
        )
        object.__setattr__(self, "extra", freeze(self.extra))
        object.__setattr__(self, "runtime", MappingProxyType(dict(self.runtime)))


@dataclass(frozen=True)
class PreparedStackData:
    """Installed resolved catalog, reference and networks as one value."""

    catalog: SceneCatalog
    pairs: Any
    misreg_pairs: Any
    reference: str
    acquisitions: Any = None


def _request_options(stack: Stack) -> Mapping[str, Any]:
    """Return the request's frozen topology options, ignoring malformed values."""
    options = stack.request.get("topology", {})
    return options if isinstance(options, Mapping) else {}


def _network_plan_dates(
    stack: Stack, sources: Mapping[str, Any], requested: Sequence[str]
) -> tuple[str, ...]:
    """Return the deterministic date plan used for default networks.

    Mission requests may name local paths or acquisition identities rather
    than ``YYYYMMDD`` dates (for example a NISAR RSLC path).  A search request
    always carries the full stable target vocabulary, so an unparseable
    ``start`` still pins its acquisition-id order instead of falling back to
    whichever source qualified first.  Non-search missions whose requested
    membership is not a date vocabulary keep the qualified catalog.
    """
    plan = tuple(sorted(set(requested)))
    if plan and (
        all(len(item) == 8 and item.isascii() and item.isdigit() for item in plan)
        or stack.request.get("kind") == "search"
    ):
        return plan
    return tuple(sorted(sources))


def _resolve_networks(
    stack: Stack,
    sources: Mapping[str, Any],
    *,
    requested: Sequence[str],
) -> tuple[Any, Any, str, tuple[str, ...]]:
    """Return deterministic networks, reference and blocking prerequisites.

    The reference is a scientific decision resolved from the full requested
    target before acquisition completes, never from whichever date happened
    to qualify first.  An explicit request reference is preserved even when
    it is unavailable and produces a blocking prerequisite instead of a
    silent fallback.
    """
    from faninsar.core.pair import Pairs
    from faninsar.stack.session import _iter_pair_dates, _pairs_from_factory

    options = _request_options(stack)
    ready = tuple(sorted(sources))
    plan = _network_plan_dates(stack, sources, requested)
    explicit = options.get("reference") or stack.request.get("reference")
    if explicit is not None:
        reference = str(explicit)
    elif plan:
        reference = plan[0]
    elif stack._inputs_state is not None:
        reference = stack._inputs_state.reference
    else:
        reference = ready[0]
    prerequisites: list[str] = []
    if reference not in sources:
        prerequisites.append(
            f"{'explicit' if explicit is not None else 'default'} reference "
            f"{reference} is unavailable"
        )

    def network(name: str, prefix: str, interval: int, days: int) -> Any:
        names = options.get(name)
        if names is not None:
            result = Pairs.from_names(list(names))
            missing = {
                endpoint for pair in _iter_pair_dates(result) for endpoint in pair
            } - set(ready)
            if missing:
                prerequisites.append(f"explicit {name} endpoints are unavailable")
            return result
        return _pairs_from_factory(
            plan,
            max_interval=options.get(prefix + "max_interval", interval),
            max_days=options.get(prefix + "max_days", days),
        )

    return (
        network("pairs", "pair_", 3, 72),
        network("misreg_pairs", "misreg_", 2, 36),
        reference,
        tuple(prerequisites),
    )


def _candidate_from_inputs(
    stack: Stack,
    inputs: PreparedInputs,
    *,
    unavailable: Mapping[str, Any],
) -> Stack:
    """Build an unpublished candidate sharing the caller's requested recipe."""
    candidate = copy.copy(stack)
    candidate._preparation_preserve_artifacts = stack.data_ready
    if any(entry.get("kind") == "dem" for entry in unavailable.values()):
        candidate._dem_wgs84 = None
    # Copy the construction snapshot: preparation may rematerialize the
    # managed DEM in place, and the requested recipe must stay the
    # caller's recipe for durable intent comparison. The DEM is shallow
    # copied because the managed-raster path attribute is rebound while
    # its immutable grid stays shared.
    candidate.config = replace(
        stack._request_config,
        nlooks_divisor=stack.config.nlooks_divisor,
        ref_points=stack.config.ref_points,
        ref_policy=stack.config.ref_policy,
        extra=thaw(freeze(stack._request_config.extra)),
    )
    if candidate.config.dem is not None:
        candidate.config = replace(
            candidate.config, dem=copy.copy(candidate.config.dem)
        )
    candidate.config.extra.update(thaw(inputs.extra))
    if stack._inputs_state is not None:
        candidate.config.grid = stack.config.grid
        if not any(entry.get("kind") == "dem" for entry in unavailable.values()):
            live_dem = stack.config.dem
            candidate.config.dem = None if live_dem is None else copy.copy(live_dem)
    candidate.__dict__.update(inputs.runtime)
    candidate.scene_provider = (
        inputs.scene_provider or stack.scene_provider or stack._default_scene_provider()
    )
    return candidate


def _bind_mission(
    candidate: Stack,
    inputs: PreparedInputs,
    sources: Mapping[str, Any],
) -> MissionPreparedInputs:
    """Bind only mission resources whose owning source qualified."""
    mission = replace(
        inputs.mission,
        orbit_paths_by_acquisition={
            key: value
            for key, value in inputs.mission.orbit_paths_by_acquisition.items()
            if key in sources
        },
        aux_cal_paths_by_acquisition={
            key: value
            for key, value in inputs.mission.aux_cal_paths_by_acquisition.items()
            if key in sources
        },
    )
    candidate._mission_prepared = mission
    if mission.orbit_paths_by_acquisition:
        candidate.config.extra["orbit_paths"] = dict(mission.orbit_paths_by_acquisition)
    if mission.aux_cal_paths_by_acquisition:
        candidate.config.extra["aux_cal_paths"] = dict(
            mission.aux_cal_paths_by_acquisition
        )
    return mission


def _prepare_shared_inputs(
    candidate: Stack,
    mission: MissionPreparedInputs,
    *,
    offline: bool,
) -> None:
    """Resolve the fixed grid/ROI and shared DEM once for a candidate."""
    if (
        candidate.config.grid != "utm_ups"
        or candidate.config.roi is not None
        or mission.selected_footprints
    ):
        candidate.config.grid = candidate.resolve_grid()
    _prepare_mask_dependencies(candidate, offline=offline)
    dem = candidate._ensure_raster_dem()
    if (
        dem is None
        and candidate._sensor_registry_key is not None
        and candidate.config.extra.get("height_m", candidate.config.extra.get("height"))
        is None
    ):
        message = "shared DEM has no usable extent"
        raise FileNotFoundError(message)


def _normalize_identity_request(root: Path) -> None:
    """Keep the durable request view canonical after resolved-input publication.

    ``save_stack_identity`` encodes the request payload; ``Stack.extend``
    consumes the authoritative request as normalized plain values.  Re-writing
    only the request field preserves the same full target while preventing an
    additive extension from starting with an encoded (and therefore stale)
    source view.
    """
    from faninsar.stack.identity import decode_value
    from faninsar.stack.workspace import normalize_request

    path = Path(root) / "stack.json"
    if not path.is_file():
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    request = document.get("request") if isinstance(document, dict) else None
    if not request:
        return
    decoded = decode_value(request)
    normalized = normalize_request(str(document.get("mission") or ""), decoded)
    if normalized == decoded:
        return
    document["request"] = normalized
    atomic_json(path, document)


def _install_candidate(
    stack: Stack,
    candidate: Stack,
    *,
    track_metadata: bool = True,
) -> None:
    """Publish one committed candidate snapshot and preserve prior records."""
    from faninsar.stack.identity import save_stack_identity

    if track_metadata:
        candidate._track_metadata()
    _sync_dem_artifacts(candidate)
    current_records = local_input_records(candidate)
    historical = {
        path: {**entry, "active": False}
        for path, entry in stack.config.extra.get(
            "preparation_local_inputs", {}
        ).items()
    }
    historical.update(current_records)
    candidate.config.extra["preparation_local_inputs"] = historical
    save_stack_identity(candidate)
    _normalize_identity_request(stack.config.work_dir)
    stack.__dict__.update(candidate.__dict__)


def _known_ready_sources(
    stack: Stack, unavailable: Mapping[str, Any]
) -> dict[str, tuple[Path, ...]]:
    """Return locally present committed sources not already known stale."""
    if stack._inputs_state is None:
        return {}
    changed = {
        str(entry.get("target"))
        for path, entry in unavailable.items()
        if Path(path).exists()
    }
    known = {}
    for target, paths in stack.catalog.paths.items():
        resolved = tuple(Path(path) for path in paths)
        if target not in changed and all(path.exists() for path in resolved):
            known[target] = resolved
    return known


class _StreamingPreparation:
    """Publish qualified dates incrementally while acquisition keeps running.

    The mission invokes one instance on the controlling acquisition thread
    after each date's SLC+EOF+AUX qualification.  Only this instance mutates
    the live Stack; acquisition workers keep reading their immutable
    construction snapshot.  A private staging area is unnecessary: each
    snapshot reuses the existing durable identity publication and keeps the
    original full request untouched.
    """

    def __init__(
        self,
        stack: Stack,
        on_ready: Callable[[Stack, tuple[str, ...]], None],
        *,
        offline: bool,
    ) -> None:
        self.stack = stack
        self.on_ready = on_ready
        self.offline = offline
        self.strict = stack.verification == "strict"
        self.sources: dict[str, tuple[Path, ...]] = {}
        self.orbit_paths: dict[str, Path] = {}
        self.aux_cal_paths: dict[str, Path] = {}
        self.footprints: list[Any] = []
        self.provenance: dict[str, Any] = {}
        self.requested: tuple[str, ...] = ()
        self.pending: set[str] = set()
        self.emitted: set[str] = set()
        self.shared_ready = False
        self.callback_failed = False
        if stack._inputs_state is not None:
            for target, paths in stack.catalog.paths.items():
                self.sources[target] = tuple(Path(path) for path in paths)
        for name, target in (
            ("orbit_paths", self.orbit_paths),
            ("aux_cal_paths", self.aux_cal_paths),
        ):
            for key, value in stack._request_config.extra.get(name, {}).items():
                target[str(key)] = Path(value)

    def __call__(self, inputs: PreparedInputs) -> None:
        """Merge one qualified date and publish it when science may proceed."""
        if not isinstance(inputs, PreparedInputs):
            message = "on_ready requires PreparedInputs"
            reject(message, TypeError, logger=logger)
        if inputs.requested:
            self.requested = tuple(dict.fromkeys((*self.requested, *inputs.requested)))
        incoming = {
            target: tuple(Path(path) for path in paths)
            for target, paths in inputs.sources.items()
            if paths
        }
        incoming = {
            target: paths
            for target, paths in incoming.items()
            if all(path.exists() for path in paths)
        }
        if not incoming:
            return
        if self.strict:
            _establish_admitted_source_baselines(self.stack, incoming)
            verified: dict[str, tuple[Path, ...]] = {}
            for target, paths in incoming.items():
                reason = strict_source_failure(self.stack, paths)
                if reason is None:
                    verified[target] = paths
                else:
                    logger.warning(
                        "Strict source admission excluded target %s: %s", target, reason
                    )
            incoming = verified
            if not incoming:
                return
        self.sources.update(incoming)
        self.orbit_paths.update(
            {
                str(key): Path(value)
                for key, value in inputs.mission.orbit_paths_by_acquisition.items()
            }
        )
        self.aux_cal_paths.update(
            {
                str(key): Path(value)
                for key, value in inputs.mission.aux_cal_paths_by_acquisition.items()
            }
        )
        self.footprints.extend(inputs.mission.selected_footprints)
        self.provenance.update(inputs.mission.provenance)
        self.pending.update(incoming)
        if not self._publish():
            return
        if self.stack.reference not in self.sources:
            return
        batch = tuple(sorted(self.pending))
        self.pending.clear()
        if batch:
            self.emitted.update(batch)
            try:
                self.on_ready(self.stack, batch)
            except BaseException:
                self.callback_failed = True
                raise

    def _snapshot(self) -> PreparedInputs:
        """Return the cumulative qualified snapshot for identity publication."""
        extra: dict[str, Any] = {}
        if self.orbit_paths:
            extra["orbit_paths"] = dict(self.orbit_paths)
        if self.aux_cal_paths:
            extra["aux_cal_paths"] = dict(self.aux_cal_paths)
        return PreparedInputs(
            sources=dict(self.sources),
            requested=self.requested,
            mission=MissionPreparedInputs(
                selected_footprints=tuple(self.footprints),
                orbit_paths_by_acquisition=dict(self.orbit_paths),
                aux_cal_paths_by_acquisition=dict(self.aux_cal_paths),
                provenance=dict(self.provenance),
            ),
            extra=extra,
        )

    def _publish(self) -> bool:
        """Install one cumulative snapshot after shared inputs resolve.

        Returns ``False`` without publishing anything when the fixed grid/ROI
        or shared DEM cannot resolve, so a caller never observes a qualified
        date before science prerequisites exist.  The next qualified date
        retries the shared resolution with the cumulative snapshot and then
        releases every pending date in one batch.
        """
        from faninsar.core.acquisition import Acquisition

        with reentrant_writer_guard(self.stack.config.work_dir):
            snapshot = self._snapshot()
            candidate = _candidate_from_inputs(self.stack, snapshot, unavailable={})
            pairs, misreg_pairs, reference, _prerequisites = _resolve_networks(
                self.stack,
                snapshot.sources,
                requested=snapshot.requested,
            )
            candidate._inputs_state = PreparedStackData(
                SceneCatalog(snapshot.sources),
                pairs,
                misreg_pairs,
                reference,
                Acquisition(sorted(snapshot.sources)),
            )
            mission = _bind_mission(candidate, snapshot, snapshot.sources)
            if mission.selected_footprints:
                candidate.config.extra["selected_footprints"] = (
                    mission.selected_footprints
                )
            candidate._validate_mission_prepared(mission)
            if not self.shared_ready:
                try:
                    _prepare_shared_inputs(candidate, mission, offline=self.offline)
                except (OSError, ConnectionError) as error:
                    logger.warning(
                        "Incremental snapshot cannot resolve shared inputs: %s",
                        error,
                    )
                    return False
                self.shared_ready = True
            candidate._prepared = True
            _install_candidate(self.stack, candidate, track_metadata=False)
        return True


@contextmanager
def writer_guard(root: Path) -> Iterator[None]:
    """Hold an OS-owned workspace lock until all mission writers have joined."""
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".preparation.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            message = f"another preparation writer owns {root}"
            reject(message, RuntimeError, logger=logger)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass
class _WriterLockState:
    """Same-thread ownership of one OS writer lock."""

    owner: int
    depth: int
    handle: Any


_WRITER_LOCKS: dict[Path, _WriterLockState] = {}
_WRITER_LOCKS_GUARD = threading.Lock()


@contextmanager
def reentrant_writer_guard(root: Path) -> Iterator[None]:
    """Hold one workspace OS lock, re-entering only on the owning thread.

    A single preparation pass keeps the OS ``flock`` for its whole lifetime
    so incremental snapshots do not open a second descriptor and so another
    process (or another thread) still cannot write the workspace.  Nested
    acquisition on the owning thread increments a depth counter; any other
    thread and any other process observes the real kernel lock conflict.
    """
    import fcntl

    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    key = target.resolve()
    handle = None
    reentered = False
    with _WRITER_LOCKS_GUARD:
        state = _WRITER_LOCKS.get(key)
        if state is not None and state.owner == threading.get_ident():
            state.depth += 1
            reentered = True
        elif state is None:
            handle = (target / ".preparation.lock").open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                handle = None
            else:
                _WRITER_LOCKS[key] = _WriterLockState(threading.get_ident(), 1, handle)
    if not reentered and handle is None:
        message = f"another preparation writer owns {target}"
        reject(message, RuntimeError, logger=logger)
    try:
        yield
    finally:
        with _WRITER_LOCKS_GUARD:
            state = _WRITER_LOCKS.get(key)
            if state is not None and state.owner == threading.get_ident():
                state.depth -= 1
                if state.depth == 0:
                    _WRITER_LOCKS.pop(key, None)
                    try:
                        fcntl.flock(state.handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        state.handle.close()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    """Publish one durable JSON record without altering its prior version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.pending")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _preparation_identity(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a request document with unwrap-only science stripped.

    Legacy ``inputs/request.json`` files omit ``nlooks_divisor``. Newer
    snapshots must compare equal to those files so a coefficient-only
    change cannot reject ``prepare_data``.
    """
    from faninsar.stack.identity import PREPARATION_EXCLUDED_SCIENCE_FIELDS

    payload = copy.deepcopy(dict(document))
    science = payload.get("science")
    if isinstance(science, dict):
        for name in PREPARATION_EXCLUDED_SCIENCE_FIELDS:
            science.pop(name, None)
    return payload


def request_document(stack: Stack) -> dict[str, Any]:
    """Validate and encode the captured science and mission source request.

    ``_request_config`` is the immutable construction snapshot. The live
    unwrap coefficient is overlaid onto a copy so request-only reopen
    keeps the current numeric divisor without mutating the snapshot.
    """
    from faninsar.stack.identity import UNSAVED_FIELDS, encode_value

    snapshot = replace(
        stack._request_config,
        nlooks_divisor=stack.config.nlooks_divisor,
        ref_points=stack.config.ref_points,
        ref_policy=stack.config.ref_policy,
    )
    science = {
        item.name: encode_value(getattr(snapshot, item.name))
        for item in fields(snapshot)
        if item.name not in UNSAVED_FIELDS and item.name != "work_dir"
    }
    from faninsar.remote.records import _sanitize

    return _sanitize(
        {
            "schema": REQUEST_SCHEMA,
            "mission": type(stack).__name__,
            "source": encode_value(thaw(stack.request)),
            "science": science,
            "extra": encode_value(snapshot.extra),
        }
    )


def report_document(report: PreparationReport) -> dict[str, Any]:
    """Encode diagnostics without persisting exception text or runtime objects."""
    return {
        "requested": list(report.requested),
        "ready": list(report.ready),
        "failed": [
            {"target": f.target, "phase": f.phase, "reason": f.reason}
            for f in report.failed
        ],
        "reused": list(report.reused),
        "prerequisites": list(report.prerequisites),
        "outcome": report.outcome,
    }


def restore_request(root: Path, runtime: Mapping[str, Any]) -> Stack:
    """Restore a configured mission locally from its supported request schema."""
    from faninsar.stack.config import StackConfig
    from faninsar.stack.identity import decode_value
    from faninsar.stack.session import _mission_class

    document = json.loads((root / "inputs" / "request.json").read_text())
    if document.get("schema") != REQUEST_SCHEMA:
        message = (
            "unsupported preparation request version; reconstruct the mission request"
        )
        reject(message, ValueError, logger=logger)
    cls = _mission_class(document["mission"])
    science = {name: decode_value(value) for name, value in document["science"].items()}
    science.update(
        {name: value for name, value in runtime.items() if name != "scene_provider"}
    )
    science["work_dir"] = root
    science["extra"] = decode_value(document["extra"])
    stack = cls.configure(
        config=StackConfig(**science), request=decode_value(document["source"])
    )
    stack.scene_provider = runtime.get("scene_provider")
    return stack


def _legacy_request_is_stale(stack: Stack, root: Path) -> bool:
    """Whether authoritative ``stack.json`` already records the live request.

    ``stack.json`` is the single durable intent (PROPOSAL-0088).  A legacy
    ``inputs/request.json`` remains a compatibility view for old workspaces;
    after an additive extension it must be refreshed, not used to reject the
    authoritative target as a conflicting request.
    """
    try:
        from faninsar.stack.identity import decode_value
        from faninsar.stack.workspace import normalize_request, read_intent
    except ImportError:  # pragma: no cover - defensive boundary
        return False
    intent = read_intent(root)
    if intent is None or not intent.request:
        return False
    try:
        current = normalize_request(type(stack).__name__, thaw(stack.request))
        recorded = normalize_request(type(stack).__name__, decode_value(intent.request))
    except (TypeError, ValueError):
        return False
    return recorded == current


def prepare_stack_data(
    stack: Stack,
    *,
    on_error: str,
    offline: bool,
    budget: RemoteResourceBudget | None,
    on_ready: Callable[[Stack, tuple[str, ...]], None] | None = None,
) -> PreparationReport:
    """Prepare under one geometry-resource policy, including mission DEM demands.

    When ``on_ready`` is supplied, each mission-qualified date is installed as
    an incremental durable snapshot on the controlling acquisition thread and
    handed to ``on_ready`` while the other acquisition futures keep running.
    """
    from faninsar.processing.geometry.fetch import offline_resources

    with offline_resources(offline):
        return _prepare_stack_data(
            stack,
            on_error=on_error,
            offline=offline,
            budget=budget,
            on_ready=on_ready,
        )


def _prepare_stack_data(
    stack: Stack,
    *,
    on_error: str,
    offline: bool,
    budget: RemoteResourceBudget | None,
    on_ready: Callable[[Stack, tuple[str, ...]], None] | None = None,
) -> PreparationReport:
    """Prepare a candidate and atomically install only a qualified ready set."""
    from faninsar.core.acquisition import Acquisition

    if on_error not in ("warn", "raise") or not isinstance(offline, bool):
        message = "on_error must be 'warn' or 'raise' and offline must be bool"
        reject(message, ValueError, logger=logger)
    document = request_document(stack)
    root = stack.config.work_dir
    with reentrant_writer_guard(root):
        request_path = root / "inputs" / "request.json"
        stored = None
        request_changed = False
        if request_path.exists():
            stored = json.loads(request_path.read_text())
            if _preparation_identity(stored) != _preparation_identity(document):
                request_changed = True
                if _legacy_request_is_stale(stack, root):
                    logger.info(
                        "Refreshing legacy %s from authoritative stack.json",
                        request_path,
                    )
                else:
                    message = (
                        "workspace holds a different preparation request; "
                        "choose a different work_dir"
                    )
                    reject(message, ValueError, logger=logger)
        if (
            not request_changed
            and stack.preparation_report is not None
            and stack.preparation_report.complete
            and stack.data_ready
            and all(
                path.exists()
                for paths in stack.catalog.paths.values()
                for path in paths
            )
            and not (
                stack.verification == "strict"
                and any(
                    strict_source_failure(stack, paths) is not None
                    for paths in stack.catalog.paths.values()
                )
            )
        ):
            return stack.preparation_report
        unavailable = unavailable_local_inputs(stack, include_history=True)
        atomic_json(request_path, document)
        observer = (
            None
            if on_ready is None
            else _StreamingPreparation(stack, on_ready, offline=offline)
        )
        context = PreparationContext(
            stack.request,
            freeze(document["science"]),
            offline,
            budget,
            stack.preparation_report,
            observer,
            stored,
        )
        try:
            inputs = stack._prepare_inputs(context)
        except BaseException as error:
            atomic_json(
                root / "inputs" / "attempt.json",
                {
                    "outcome": "cancelled"
                    if isinstance(error, (KeyboardInterrupt, SystemExit))
                    else "interrupted",
                    "phase": (
                        "on_ready"
                        if observer is not None and observer.callback_failed
                        else "prepare_inputs"
                    ),
                    "error_type": type(error).__name__,
                },
            )
            raise
        if not isinstance(inputs, PreparedInputs):
            message = "_prepare_inputs must return PreparedInputs"
            reject(message, TypeError, logger=logger)
        failures = list(inputs.failures)
        sources = {}
        changed_sources = {
            entry["target"]
            for path, entry in unavailable.items()
            if entry["kind"] == "source" and Path(path).exists()
        }
        for target, paths in inputs.sources.items():
            if (
                target not in changed_sources
                and paths
                and all(path.exists() for path in paths)
            ):
                sources[target] = paths
            else:
                failures.append(
                    PreparationFailure(
                        target,
                        "qualification",
                        "missing_source",
                        FileNotFoundError(target),
                    )
                )
        for target, paths in _known_ready_sources(stack, unavailable).items():
            if target not in {failure.target for failure in failures}:
                sources.setdefault(target, paths)
        for bindings in (
            inputs.mission.orbit_paths_by_acquisition,
            inputs.mission.aux_cal_paths_by_acquisition,
        ):
            for target, path in bindings.items():
                prior = unavailable.get(str(path))
                if not Path(path).is_file() or (
                    prior is not None and prior["kind"] in {"orbit", "aux"}
                ):
                    sources.pop(target, None)
                    failures.append(
                        PreparationFailure(
                            target,
                            "mission_inputs",
                            "unqualified_auxiliary",
                            FileNotFoundError(str(path)),
                        )
                    )
        requested = inputs.requested or tuple(
            dict.fromkeys((*sources, *(f.target for f in failures)))
        )
        sources = _filter_strict_sources(stack, sources, failures)
        if stack.verification == "strict" and stack._inputs_state is not None:
            reference = stack._inputs_state.reference
            if reference not in sources and not any(
                failure.target == reference for failure in failures
            ):
                failures.append(
                    PreparationFailure(
                        reference,
                        "source_integrity",
                        "reference_unverifiable",
                        RuntimeError(reference),
                    )
                )
        if failures and on_error == "raise":
            report = PreparationReport(
                tuple(requested), tuple(sorted(sources)), tuple(failures)
            )
            atomic_json(root / "inputs" / "attempt.json", report_document(report))
            error = DataPreparationError(report)
            logger.error("%s", error)
            raise error from failures[0].cause
        candidate = _candidate_from_inputs(stack, inputs, unavailable=unavailable)
        mission = _bind_mission(candidate, inputs, sources)
        prerequisites: tuple[str, ...] = ()
        if sources:
            dates = tuple(sorted(sources))
            pairs, misreg_pairs, reference, prerequisites = _resolve_networks(
                stack, sources, requested=requested
            )
            candidate._inputs_state = PreparedStackData(
                SceneCatalog(sources),
                pairs,
                misreg_pairs,
                reference,
                Acquisition(list(dates)),
            )
            if inputs.mission.selected_footprints:
                candidate.config.extra["selected_footprints"] = (
                    inputs.mission.selected_footprints
                )
            if stack._prepared and stack._inputs_state is not None:
                candidate._preparation_preserve_artifacts = False
                candidate._dem_wgs84 = None
            candidate._validate_mission_prepared(mission)
            try:
                _prepare_shared_inputs(candidate, mission, offline=offline)
            except (OSError, ConnectionError) as error:
                failures.extend(
                    PreparationFailure(
                        target, "shared_inputs", type(error).__name__, error
                    )
                    for target in sources
                )
                sources = {}
        report = PreparationReport(
            tuple(requested),
            tuple(sorted(sources)),
            tuple(failures),
            prerequisites=prerequisites,
        )
        atomic_json(root / "inputs" / "attempt.json", report_document(report))
        if failures and on_error == "raise":
            error = DataPreparationError(report)
            logger.error("%s", error)
            raise error from failures[0].cause
        for failure in failures:
            logger.warning(
                "Preparation target %s unavailable during %s: %s",
                failure.target,
                failure.phase,
                failure.reason,
            )
        if sources:
            candidate._prepared = True
            candidate.preparation_report = report
            candidate.config.extra["preparation_report"] = report_document(report)
            _install_candidate(stack, candidate)
        elif not stack.data_ready:
            stack.preparation_report = report
        return report


def _prepare_mask_dependencies(stack: Stack, *, offline: bool) -> None:
    """Fetch configured external sources without rasterizing a science mask."""
    for definition in stack.config.mask_plan.masks.values():
        path = getattr(definition, "path", None)
        if path is not None and not Path(path).is_file():
            message = "configured mask source is unavailable"
            raise InputUnavailableError(message)
        if getattr(definition, "kind", None) != "water":
            continue
        from pyproj import Transformer

        from faninsar.processing.masking.mask_manager import get_mask_manager
        from faninsar.processing.masking.mask_sources import expand_tile_parts

        grid = stack.grid
        bounds = grid.bounds
        if str(grid.crs) != "EPSG:4326":
            bounds = Transformer.from_crs(
                grid.crs, "EPSG:4326", always_xy=True
            ).transform_bounds(*bounds)
        manager = get_mask_manager(source="water:" + (definition.provider or "auto"))
        if offline:
            plan = manager.source_entry.plan(bounds)
            if any(
                not (manager.cache_dir / tile.cache_path).is_file()
                for raw in plan.tiles
                for tile in expand_tile_parts(raw)
            ):
                message = "offline water mask prerequisites are unavailable"
                raise InputUnavailableError(message)
        else:
            from faninsar.processing.masking.mask_manager import (
                MaskProviderUnavailableError,
            )

            try:
                manager.fetch_water(bounds)
            except MaskProviderUnavailableError as error:
                message = "water mask prerequisites are unavailable"
                raise InputUnavailableError(message) from error


def _sync_dem_artifacts(stack: Stack) -> None:
    """Make candidate raster files durable before publishing their identity."""
    paths = (
        stack._stack_dem_raster_path(),
        stack._stack_wgs84_raster_path(),
        stack._stack_dem_sidecar_path(),
    )
    for path in paths:
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directory = stack._dem_artifact_root()
    if directory.is_dir():
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def source_integrity_root(stack: Stack) -> Path:
    """Return the workspace-local root for strict source baselines."""
    return Path(stack.config.work_dir) / "integrity" / "sources"


def strict_source_failure(stack: Stack, paths: Sequence[Path]) -> str | None:
    """Return why one target's source paths fail strict qualification.

    This is a hard gate used only by ``verification="strict"``. Fast
    preparation never calls it, so the default path performs no extra source
    walk or payload hash.
    """
    from faninsar.processing.errors import InvalidProcessingStateError
    from faninsar.stack.source_integrity import source_identity, verify_source_integrity

    root = source_integrity_root(stack)
    cache = getattr(stack, "_verified_source_identities", None)
    for path in paths:
        resolved = Path(path).resolve()
        try:
            current = source_identity(resolved)
        except InvalidProcessingStateError as error:
            return f"strict source verification failed for {path}: {error}"
        if cache is not None and cache.get(str(resolved)) == current:
            continue
        try:
            verdict = verify_source_integrity(root, resolved, strict=True)
        except InvalidProcessingStateError as error:
            return f"strict source verification failed for {path}: {error}"
        if verdict.status not in {"metadata_verified", "verified"}:
            detail = verdict.detail or verdict.basis or "no trusted baseline"
            return (
                f"strict source verification failed for {path}: "
                f"{verdict.status}: {detail}"
            )
        if cache is not None:
            cache[str(resolved)] = current
    return None


def _establish_admitted_source_baselines(
    stack: Stack, sources: Mapping[str, Sequence[Path]]
) -> None:
    """Baseline only fresh provider-verified sources, never old local files.

    A source already present in the prior preparation records is an existing
    local input; strict verification must find its trusted baseline or report
    it unverifiable. A newly admitted single-file source with a qualified
    provider SHA-256 receipt may record one baseline from that external digest;
    a receipt without a digest never becomes a self-baseline.
    """
    if stack.verification != "strict":
        return
    from faninsar.remote.fetch.complete import qualified_download_receipt
    from faninsar.stack.source_integrity import establish_source_integrity

    prior = stack.config.extra.get("preparation_local_inputs", {})
    root = source_integrity_root(stack)
    for paths in sources.values():
        for path in paths:
            if str(path) in prior:
                continue
            receipt = qualified_download_receipt(Path(path))
            if receipt is None:
                continue
            digest = receipt.get("sha256")
            if not isinstance(digest, str):
                continue
            try:
                establish_source_integrity(root, path, provider_digest=digest)
            except Exception as error:  # pragma: no cover - local integrity I/O
                logger.warning(
                    "Strict source baseline could not be established for %s: %s",
                    path,
                    error,
                )


def _filter_strict_sources(
    stack: Stack,
    sources: Mapping[str, tuple[Path, ...]],
    failures: list[PreparationFailure],
) -> dict[str, tuple[Path, ...]]:
    """Drop targets whose original sources fail strict verification."""
    if stack.verification != "strict":
        return dict(sources)
    _establish_admitted_source_baselines(stack, sources)
    verified: dict[str, tuple[Path, ...]] = {}
    for target, paths in sources.items():
        reason = strict_source_failure(stack, paths)
        if reason is None:
            verified[target] = tuple(paths)
            continue
        failures.append(
            PreparationFailure(
                target,
                "source_integrity",
                "strict_source_unverifiable",
                RuntimeError(reason),
            )
        )
        logger.warning("Strict source admission excluded target %s: %s", target, reason)
    return verified


def _path_signature(path: Path) -> dict[str, Any]:
    """Record local metadata without reading large scientific payloads."""
    stat = path.stat()
    record = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if path.is_dir():
        record["members"] = {
            str(member.relative_to(path)): {
                "size": item.st_size,
                "mtime_ns": item.st_mtime_ns,
            }
            for member in sorted(path.rglob("*"))
            if member.is_file()
            for item in (member.stat(),)
        }
    return record


def local_input_records(stack: Stack) -> dict[str, Any]:
    """Capture all installed local input and shared artifact dependencies."""
    records = {}
    for target, paths in stack.catalog.paths.items():
        for path in paths:
            records[str(path)] = {
                "kind": "source",
                "target": target,
                "stat": _path_signature(path),
            }
    for kind, key in (("orbit", "orbit_paths"), ("aux", "aux_cal_paths")):
        for target, raw_path in stack.config.extra.get(key, {}).items():
            path = Path(raw_path)
            records[str(path)] = {
                "kind": kind,
                "target": target,
                "stat": _path_signature(path),
            }
    for path in (
        stack._stack_dem_raster_path(),
        stack._stack_wgs84_raster_path(),
        stack._stack_dem_sidecar_path(),
    ):
        if path.is_file():
            records[str(path)] = {
                "kind": "dem",
                "target": "shared",
                "stat": _path_signature(path),
            }
    dem_path = getattr(stack.config.dem, "path", None)
    if dem_path is not None:
        path = Path(dem_path)
        records[str(path)] = {
            "kind": "dem",
            "target": "shared",
            "stat": _path_signature(path),
        }
    for definition in stack.config.mask_plan.masks.values():
        path = getattr(definition, "path", None)
        if path is not None:
            path = Path(path)
            records[str(path)] = {
                "kind": "mask",
                "target": "shared",
                "stat": _path_signature(path),
            }
    for path, entry in records.items():
        entry["content"] = _bounded_content(Path(path))
        entry["active"] = True
        from faninsar.remote.fetch.complete import qualified_download_receipt

        entry["receipt"] = qualified_download_receipt(Path(path))
    return records


def _bounded_content(path: Path) -> str | None:
    """Qualify small repairable artifacts without hashing bulk payloads."""
    paths = (
        sorted(member for member in path.rglob("*") if member.is_file())
        if path.is_dir()
        else [path]
    )
    if sum(member.stat().st_size for member in paths) > 1024 * 1024:
        return None
    digest = hashlib.sha256()
    for member in paths:
        digest.update(
            str(member.relative_to(path) if path.is_dir() else "file").encode()
        )
        digest.update(member.read_bytes())
    return digest.hexdigest()


def unavailable_local_inputs(
    stack: Stack, *, include_history: bool = False
) -> dict[str, Any]:
    """Stat-qualify the installed snapshot without acquisition or payload hashes."""
    records = stack.config.extra.get("preparation_local_inputs", {})
    unavailable = {}
    for recorded, entry in records.items():
        if not include_history and not entry.get("active", True):
            continue
        try:
            matches = _path_signature(Path(recorded)) == entry["stat"]
            if not matches and entry.get("content") is not None:
                matches = _bounded_content(Path(recorded)) == entry["content"]
            if not matches and entry.get("receipt") is not None:
                from faninsar.remote.fetch.complete import qualified_download_receipt

                matches = qualified_download_receipt(Path(recorded)) == entry["receipt"]
        except OSError:
            matches = False
        if not matches:
            unavailable[recorded] = entry
    if not records and stack._inputs_state is not None:
        for target, paths in stack.catalog.paths.items():
            for path in paths:
                if not path.exists():
                    unavailable[str(path)] = {"kind": "source", "target": target}
    return unavailable
