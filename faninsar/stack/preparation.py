"""Explicit, recoverable Stack input preparation (PROPOSAL-0073)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from faninsar.logging import reject, setup_logger
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.search import MissionPreparedInputs

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def request_document(stack: Stack) -> dict[str, Any]:
    """Validate and encode the captured science and mission source request."""
    from faninsar.stack.identity import UNSAVED_FIELDS, encode_value

    science = {
        item.name: encode_value(getattr(stack._request_config, item.name))
        for item in fields(stack._request_config)
        if item.name not in UNSAVED_FIELDS and item.name != "work_dir"
    }
    from faninsar.remote.records import _sanitize

    return _sanitize(
        {
            "schema": REQUEST_SCHEMA,
            "mission": type(stack).__name__,
            "source": encode_value(thaw(stack.request)),
            "science": science,
            "extra": encode_value(stack._request_config.extra),
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


def prepare_stack_data(
    stack: Stack, *, on_error: str, offline: bool, budget: RemoteResourceBudget | None
) -> PreparationReport:
    """Prepare under one geometry-resource policy, including mission DEM demands."""
    from faninsar.processing.geometry.fetch import offline_resources

    with offline_resources(offline):
        return _prepare_stack_data(
            stack, on_error=on_error, offline=offline, budget=budget
        )


def _prepare_stack_data(
    stack: Stack, *, on_error: str, offline: bool, budget: RemoteResourceBudget | None
) -> PreparationReport:
    """Prepare a candidate and atomically install only a qualified ready set."""
    from faninsar.core.acquisition import Acquisition
    from faninsar.core.pair import Pairs
    from faninsar.stack.identity import save_stack_identity
    from faninsar.stack.session import _pairs_from_factory

    if on_error not in ("warn", "raise") or not isinstance(offline, bool):
        message = "on_error must be 'warn' or 'raise' and offline must be bool"
        reject(message, ValueError, logger=logger)
    document = request_document(stack)
    root = stack.config.work_dir
    with writer_guard(root):
        request_path = root / "inputs" / "request.json"
        if request_path.exists() and json.loads(request_path.read_text()) != document:
            message = (
                "workspace holds a different preparation request; "
                "choose a different work_dir"
            )
            reject(message, ValueError, logger=logger)
        if (
            stack.preparation_report is not None
            and stack.preparation_report.complete
            and stack.data_ready
            and all(
                path.exists()
                for paths in stack.catalog.paths.values()
                for path in paths
            )
        ):
            return stack.preparation_report
        unavailable = unavailable_local_inputs(stack, include_history=True)
        atomic_json(request_path, document)
        context = PreparationContext(
            stack.request,
            freeze(document["science"]),
            offline,
            budget,
            stack.preparation_report,
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
        if failures and on_error == "raise":
            report = PreparationReport(
                tuple(requested), tuple(sorted(sources)), tuple(failures)
            )
            atomic_json(root / "inputs" / "attempt.json", report_document(report))
            error = DataPreparationError(report)
            logger.error("%s", error)
            raise error from failures[0].cause
        candidate = copy.copy(stack)
        candidate._preparation_preserve_artifacts = stack.data_ready
        if any(entry["kind"] == "dem" for entry in unavailable.values()):
            candidate._dem_wgs84 = None
        candidate.config = replace(
            stack._request_config, extra=thaw(freeze(stack._request_config.extra))
        )
        candidate.config.extra.update(thaw(inputs.extra))
        if stack._inputs_state is not None:
            candidate.config.grid = stack.config.grid
            if not any(entry["kind"] == "dem" for entry in unavailable.values()):
                candidate.config.dem = stack.config.dem
        candidate.__dict__.update(inputs.runtime)
        candidate.scene_provider = (
            inputs.scene_provider
            or stack.scene_provider
            or stack._default_scene_provider()
        )
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
        if inputs.mission.orbit_paths_by_acquisition:
            candidate.config.extra["orbit_paths"] = dict(
                mission.orbit_paths_by_acquisition
            )
        if inputs.mission.aux_cal_paths_by_acquisition:
            candidate.config.extra["aux_cal_paths"] = dict(
                mission.aux_cal_paths_by_acquisition
            )
        prerequisites = []
        if sources:
            dates = tuple(sorted(sources))
            options = stack.request.get("topology", {})
            reference = options.get("reference") or (
                stack._inputs_state.reference
                if stack._inputs_state is not None
                and stack._inputs_state.reference in sources
                else dates[0]
            )
            if reference not in sources:
                prerequisites.append(f"explicit reference {reference} is unavailable")

            def network(name: str, prefix: str, interval: int, days: int) -> Any:
                names = options.get(name)
                if names is not None:
                    result = Pairs.from_names(list(names))
                    from faninsar.stack.session import _iter_pair_dates

                    missing = {
                        endpoint
                        for pair in _iter_pair_dates(result)
                        for endpoint in pair
                    } - set(dates)
                    if missing:
                        prerequisites.append(
                            f"explicit {name} endpoints are unavailable"
                        )
                    return result
                return _pairs_from_factory(
                    dates,
                    max_interval=options.get(prefix + "max_interval", interval),
                    max_days=options.get(prefix + "max_days", days),
                )

            candidate._inputs_state = PreparedStackData(
                SceneCatalog(sources),
                network("pairs", "pair_", 3, 72),
                network("misreg_pairs", "misreg_", 2, 36),
                reference,
                Acquisition(list(dates)),
            )
            if inputs.mission.selected_footprints:
                candidate.config.extra["selected_footprints"] = (
                    inputs.mission.selected_footprints
                )
            if stack._prepared and stack._inputs_state is not None:
                from faninsar.stack.identity import encode_value

                scope_identity = {
                    "request": document,
                    "sources": {
                        key: [
                            {
                                "path": str(path),
                                "size": path.stat().st_size,
                                "mtime_ns": path.stat().st_mtime_ns,
                            }
                            for path in paths
                        ]
                        for key, paths in sorted(sources.items())
                    },
                    "footprints": encode_value(inputs.mission.selected_footprints),
                    "grid": encode_value(candidate.config.grid),
                    "repair_of": {
                        path: entry.get("stat")
                        for path, entry in unavailable.items()
                        if entry["kind"] == "dem"
                    },
                }
                fingerprint = hashlib.sha256(
                    json.dumps(scope_identity, sort_keys=True).encode()
                ).hexdigest()
                candidate.config.extra["preparation_dem_scope"] = (
                    f"inputs/dem/{fingerprint}"
                )
                candidate._preparation_preserve_artifacts = False
                candidate._dem_wgs84 = None
            candidate._validate_mission_prepared(mission)
            try:
                if (
                    candidate.config.grid != "utm_ups"
                    or candidate.config.roi is not None
                    or inputs.mission.selected_footprints
                ):
                    candidate.config.grid = candidate.resolve_grid()
                _prepare_mask_dependencies(candidate, offline=offline)
                dem = candidate._ensure_raster_dem()
                if (
                    dem is None
                    and candidate._sensor_registry_key is not None
                    and candidate.config.extra.get(
                        "height_m", candidate.config.extra.get("height")
                    )
                    is None
                ):
                    message = "shared DEM has no usable extent"
                    raise FileNotFoundError(message)
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
            prerequisites=tuple(prerequisites),
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
            stack.__dict__.update(candidate.__dict__)
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
