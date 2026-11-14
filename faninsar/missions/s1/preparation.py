"""Sentinel-1 source requests and explicit input qualification."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import fields, is_dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger
from faninsar.remote.errors import RemoteError
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.preparation import (
    PreparationContext,
    PreparationFailure,
    PreparedInputs,
    freeze,
    thaw,
)
from faninsar.stack.search import MissionPreparedInputs

logger = setup_logger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable

    from faninsar.missions.s1.s1search import S1Search
    from faninsar.missions.s1.stack import S1Stack


def _record(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _record(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_record(item) for item in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


_SOURCE_PLAN_SCIENCE_KEYS = ("roi", "bursts", "swaths", "coreg_mode")


def _serialized_target_groups(source: Any) -> dict[str, list[Any]]:
    """Group serialized search items by acquisition date without opening a SAFE."""
    plain = thaw(source) if isinstance(source, Mapping) else {}
    search = plain.get("search")
    if not isinstance(search, Mapping):
        return {}
    items = search.get("items")
    if not isinstance(items, (list, tuple)):
        return {}
    groups: dict[str, list[Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        acquisition = item.get("acquisition")
        if not isinstance(acquisition, Mapping):
            continue
        start = acquisition.get("start")
        if isinstance(start, str) and len(start) >= 10:
            target = start[:10].replace("-", "")
        else:
            target = str(acquisition.get("acquisition_id"))
        groups.setdefault(target, []).append(item)
    return groups


def target_plan_identities(source: Any, science: Any) -> dict[str, str]:
    """Return a cheap per-date source/support plan identity.

    The identity covers the full serialized source request (including every
    item for the date), the provider/granularity, and the source-affecting
    target science (ROI, bursts, swaths and coreg mode).  It intentionally
    excludes mutable run records and unrelated science, so an unchanged plan
    reuses a qualified local path while an ROI/burst extension for the same
    date forces a conservative re-plan.  Small metadata only; never a payload
    hash.
    """
    plain_science = thaw(science) if isinstance(science, Mapping) else {}
    plain_source = thaw(source) if isinstance(source, Mapping) else {}
    shared = {
        "kind": plain_source.get("kind"),
        "provider": plain_source.get("provider"),
        "granularity": plain_source.get("granularity"),
        "orbit_group": plain_source.get("orbit_group"),
        "science": {key: plain_science.get(key) for key in _SOURCE_PLAN_SCIENCE_KEYS},
    }
    identities = {}
    for target, items in _serialized_target_groups(source).items():
        canonical_items = sorted(
            json.dumps(thaw(item), sort_keys=True, default=str) for item in items
        )
        document = {**shared, "target": target, "items": canonical_items}
        identities[target] = hashlib.sha256(
            json.dumps(document, sort_keys=True, default=str).encode()
        ).hexdigest()
    return identities


def serialize_search(search: S1Search) -> dict[str, Any]:
    """Capture sanitized stable provider descriptors, without live sessions."""
    from faninsar.remote.records import _sanitize

    return _sanitize({"items": _record(search.items), "catalog": search.catalog})


def restore_search(record: Mapping[str, Any], spatial: Any) -> S1Search:
    """Reconstruct selected descriptors without querying a catalog."""
    from faninsar.missions.s1.s1search import S1Search
    from faninsar.remote.records import AcquisitionMetadata, CatalogItem, RemoteAsset

    items = []
    for raw in record["items"]:
        item = thaw(raw)
        item["acquisition"] = AcquisitionMetadata(**item["acquisition"])
        item["assets"] = {
            key: RemoteAsset(**asset) for key, asset in item["assets"].items()
        }
        items.append(CatalogItem(**item))
    return S1Search(items=tuple(items), catalog=record.get("catalog"), spatial=spatial)


def prepare_s1_inputs(stack: S1Stack, context: PreparationContext) -> PreparedInputs:
    """Scope one transfer meter across selected targets, including async bridging.

    The wrapper also owns the operation-scoped CDSE burst transfer
    (PROPOSAL-0061 D2, PROPOSAL-0070): one provider ``RangeFetcher`` on one
    dedicated loop thread serves every date's annotation, probe, and
    measurement requests, so the AIMD window and token survive the
    per-date loop in :func:`_prepare_s1_inputs`.  The session lives only
    for this call — it is never cached on the Stack or on a module global —
    and the ordered shutdown joins the bridge before the fetcher closes.
    """
    from faninsar.missions.s1.acquisition import (
        _burst_transfer,
        _CdseBurstTransfer,
        _preparation_remote_policy,
    )
    from faninsar.remote.protocols import _CallLedger

    ledger = _CallLedger(context.budget) if context.budget is not None else None
    token = _preparation_remote_policy.set((context.budget, ledger))
    transfer = _CdseBurstTransfer()
    transfer_token = _burst_transfer.set(transfer)
    prefetch = None
    if (
        context.request["kind"] == "search"
        and not context.offline
        and context.request.get("orbit_paths") is None
        and stack._request_config.extra.get("orbit_auto_fetch", True)
    ):
        from faninsar.missions.s1.orbit_fetch import start_orbit_prefetch
        from faninsar.stack.layout import workspace_orbits_dir

        search = restore_search(context.request["search"], stack._request_config.roi)
        prefetch = start_orbit_prefetch(
            search.items,
            workspace_orbits_dir(stack.config.work_dir),
            budget=context.budget,
        )
    try:
        return _prepare_s1_inputs(stack, context)
    finally:
        if prefetch is not None:
            prefetch.join()
        _preparation_remote_policy.reset(token)
        _burst_transfer.reset(transfer_token)
        transfer.close()


def _acquire_workers(context: PreparationContext) -> int:
    """Return the operation's date-task ceiling, defaulting to four."""
    return max(1, getattr(context.budget, "max_active_files", 4))


def _reconfigure_deferred_pass_policy(operation: Any) -> None:
    """Reconfigure the shared transfer only after all regular work drains."""
    operation.configure_deferred_pass()


def _acquire_dates(
    acquire: Any,
    pending: Mapping[str, Any],
    context: PreparationContext,
    *,
    on_complete: Callable[[str, Any], None] | None = None,
) -> dict[str, Any]:
    """Acquire eligible dates with bounded admission and one deferred pass.

    ``on_complete`` runs on the controlling thread as each date settles, so
    qualification and publication of a finished date overlap the acquisition
    futures that are still in flight.  Workers never receive the callback and
    never mutate the Stack.
    """
    from faninsar.missions.s1.acquisition import _burst_transfer

    operation = _burst_transfer.get()
    logger.info(
        "CDSE regular acquire started: pending=%d active_files=%d total_ranges=%s",
        len(pending),
        _acquire_workers(context),
        getattr(context.budget, "max_workers", "default"),
    )
    results = {}
    targets = iter(pending)
    pool = ThreadPoolExecutor(
        max_workers=_acquire_workers(context), thread_name_prefix="faninsar-s1-acquire"
    )
    active = {}

    def deferred(result: Any) -> bool:
        """Whether this result waits for the final single-flight pass."""
        return (
            isinstance(result, PreparationFailure)
            and result.reason == "rate_limit_exhausted"
        )

    try:
        for _ in range(_acquire_workers(context)):
            target = next(targets, None)
            if target is not None:
                future = pool.submit(
                    copy_context().run, acquire, target, pending[target]
                )
                active[future] = target
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                target = active.pop(future)
                results[target] = future.result()
                if on_complete is not None and not deferred(results[target]):
                    on_complete(target, results[target])
            for _ in completed:
                target = next(targets, None)
                if target is not None:
                    future = pool.submit(
                        copy_context().run, acquire, target, pending[target]
                    )
                    active[future] = target
    except BaseException:
        operation.cancel_pending()
        for future in active:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    deferred_targets = [target for target in pending if deferred(results[target])]
    if deferred_targets:
        logger.warning(
            "CDSE regular acquire drained: deferred_targets=%s", deferred_targets
        )
        _reconfigure_deferred_pass_policy(operation)
        logger.info(
            "CDSE final pass started: targets=%d "
            "active_files=1 total_ranges=1 retries=0",
            len(deferred_targets),
        )
        for target in deferred_targets:
            results[target] = acquire(target, pending[target])
            if on_complete is not None:
                on_complete(target, results[target])
    return results


def _prepare_s1_inputs(stack: S1Stack, context: PreparationContext) -> PreparedInputs:
    """Acquire selected targets, qualifying and publishing each settled date."""
    from faninsar.missions.s1.stack import _configure_s1_bindings
    from faninsar.stack.session import Stack

    failures: list[PreparationFailure] = []
    requested: list[str] = []
    qualified: dict[str, tuple[Path, ...]] = {}
    footprints: list[Any] = []
    orbits: dict[str, Path] = {}
    aux_cal: dict[str, Path] = {}
    provenance: dict[str, Any] = {}
    observer = getattr(context, "on_ready", None)

    def qualify(
        target: str, target_paths: Sequence[Path]
    ) -> PreparedInputs | PreparationFailure:
        """Qualify one date's SLC plus required EOF/AUX dependencies."""
        config = replace(
            stack._request_config, extra=thaw(freeze(stack._request_config.extra))
        )
        if context.offline:
            config.extra.update(
                {"orbit_auto_fetch": False, "aux_cal_auto_fetch": False}
            )
        local = type(stack)(
            config=config, catalog=SceneCatalog({target: tuple(target_paths)})
        )
        local._preparation_budget = context.budget
        try:
            if not all(path.exists() for path in target_paths):
                raise FileNotFoundError(target)
            _configure_s1_bindings(
                local,
                orbit_paths=thaw(context.request.get("orbit_paths")),
                aux_cal_paths=thaw(context.request.get("aux_cal_paths")),
            )
            local._bind_precise_orbits()
            mission = local._local_mission_inputs()
            external_required = (
                bool(stack._request_config.extra.get("orbit_auto_fetch", True))
                or context.request.get("orbit_paths") is not None
            )
            if external_required and target not in mission.orbit_paths_by_acquisition:
                message = f"required covering EOF is unavailable for {target}"
                raise FileNotFoundError(message)
        except (OSError, RemoteError) as error:
            return PreparationFailure(
                target,
                "mission_inputs",
                getattr(error, "reason", type(error).__name__),
                error,
            )
        date_orbits = (
            {target: mission.orbit_paths_by_acquisition[target]}
            if target in mission.orbit_paths_by_acquisition
            else {}
        )
        date_aux = (
            {target: mission.aux_cal_paths_by_acquisition[target]}
            if target in mission.aux_cal_paths_by_acquisition
            else {}
        )
        extra: dict[str, Any] = {}
        if date_orbits:
            extra["orbit_paths"] = date_orbits
        if date_aux:
            extra["aux_cal_paths"] = date_aux
        return PreparedInputs(
            sources={target: tuple(target_paths)},
            requested=tuple(requested),
            mission=MissionPreparedInputs(
                selected_footprints=tuple(mission.selected_footprints),
                orbit_paths_by_acquisition=date_orbits,
                aux_cal_paths_by_acquisition=date_aux,
                provenance=mission.provenance,
            ),
            extra=extra,
        )

    def accept(partial: PreparedInputs) -> None:
        """Accumulate one qualified date and hand it to the observer."""
        for target, paths in partial.sources.items():
            qualified[target] = tuple(paths)
        mission = partial.mission
        footprints.extend(mission.selected_footprints)
        orbits.update(mission.orbit_paths_by_acquisition)
        aux_cal.update(mission.aux_cal_paths_by_acquisition)
        provenance.update(mission.provenance)
        if callable(observer):
            observer(partial)

    def process(target: str, result: Any) -> None:
        """Qualify one settled acquisition result on the controlling thread."""
        if isinstance(result, PreparationFailure):
            failures.append(result)
            return
        target_paths = tuple(Path(path) for path in result)
        if not target_paths:
            # An empty materialization is not a qualified source; the old
            # all-dates path also skipped it without inventing a failure.
            return
        partial = qualify(target, target_paths)
        if isinstance(partial, PreparationFailure):
            failures.append(partial)
        else:
            accept(partial)

    if context.request["kind"] == "search":
        search = restore_search(context.request["search"], stack._request_config.roi)
        groups: dict[str, list[Any]] = {}
        for item in search.items:
            start = item.acquisition.start
            target = (
                start.strftime("%Y%m%d")
                if start is not None
                else item.acquisition.acquisition_id
            )
            groups.setdefault(target, []).append(item)
        requested.extend(groups)
        previous_document = getattr(context, "previous_document", None)
        previous_plans = target_plan_identities(
            previous_document.get("source")
            if isinstance(previous_document, Mapping)
            else {},
            previous_document.get("science")
            if isinstance(previous_document, Mapping)
            else {},
        )
        current_plans = target_plan_identities(
            context.request, getattr(context, "science", {})
        )
        ready_targets: list[str] = []
        pending = {}
        for target, items in groups.items():
            # A committed qualified source is a local snapshot only when the
            # current per-date request/support plan is the one that produced
            # it.  An additive ROI/burst extension for the same date must
            # re-plan instead of trusting an old partial canvas.
            if (
                stack._inputs_state is not None
                and target in stack.catalog.paths
                and all(p.exists() for p in stack.catalog.paths_for(target))
                and current_plans.get(target) is not None
                and previous_plans.get(target) == current_plans.get(target)
            ):
                ready_targets.append(target)
                continue
            if context.offline:
                failures.append(
                    PreparationFailure(
                        target,
                        "acquisition",
                        "offline_missing_source",
                        FileNotFoundError(target),
                    )
                )
                continue
            pending[target] = items

        # Publish restart-ready dates before remote work so the reference can
        # unblock science even when every pending download is still in flight.
        for target in sorted(ready_targets):
            process(target, list(stack.catalog.paths_for(target)))

        def acquire(target: str, items: list[Any]) -> Any:
            allowed = inspect.signature(Stack._from_safes).parameters
            options = {
                item.name: getattr(stack._request_config, item.name)
                for item in fields(stack._request_config)
                if item.name in allowed
                and item.name not in {"work_dir", "coreg_mode", "extra"}
            }
            options["extra"] = thaw(freeze(stack._request_config.extra))
            try:
                local = type(stack)._materialize_from_search(
                    replace(search, items=tuple(items)),
                    work_dir=stack.config.work_dir,
                    provider=context.request["provider"],
                    granularity=context.request["granularity"],
                    polarizations=stack._request_config.extra["polarizations"],
                    coreg_mode=stack._request_config.coreg_mode,
                    orbit_group=context.request.get("orbit_group"),
                    orbit_paths=context.request.get("orbit_paths"),
                    aux_cal_paths=context.request.get("aux_cal_paths"),
                    effective_roi=stack._request_config.roi,
                    kwargs=options,
                )
                return local.request["paths"]
            except ValueError as error:
                message = str(error)
                if "every Sentinel-1 input failed download verification" in message:
                    return PreparationFailure(
                        target, "acquisition", "checksum_mismatch", error
                    )
                if "CDSE rediscovery is missing ASF-selected measurements" in message:
                    # A missing parent frame is one date's unavailable source,
                    # not a reason to abandon its downloaded siblings.  Report
                    # it per target so the report stays partial and the strict
                    # policy can still refuse a reduced ready set after drain.
                    return PreparationFailure(
                        target,
                        "acquisition",
                        "rediscovery_missing_parent",
                        error,
                    )
                raise
            except (OSError, RemoteError) as error:
                return PreparationFailure(
                    target,
                    "acquisition",
                    getattr(error, "reason", type(error).__name__),
                    error,
                )

        parallel = (
            context.request["provider"] == "cdse"
            and context.request["granularity"] in ("roi", "burst")
            and pending
            and all(len(items) == 1 for items in pending.values())
        )
        if parallel:
            _acquire_dates(acquire, pending, context, on_complete=process)
        else:
            for target, items in pending.items():
                process(target, acquire(target, items))
    else:
        local_paths = [Path(path) for path in context.request["paths"]]
        if not local_paths:
            return PreparedInputs(
                sources={}, requested=tuple(requested), failures=tuple(failures)
            )
        catalog = SceneCatalog.from_paths(local_paths)
        requested.extend(catalog.dates)
        for target in catalog.dates:
            process(target, list(catalog.paths_for(target)))
    if not qualified:
        return PreparedInputs(
            sources={}, requested=tuple(requested), failures=tuple(failures)
        )
    extra: dict[str, Any] = {}
    if orbits:
        extra["orbit_paths"] = orbits
    if aux_cal:
        extra["aux_cal_paths"] = aux_cal
    return PreparedInputs(
        sources=qualified,
        requested=tuple(requested),
        failures=tuple(failures),
        mission=MissionPreparedInputs(
            selected_footprints=tuple(footprints),
            orbit_paths_by_acquisition=orbits,
            aux_cal_paths_by_acquisition=aux_cal,
            provenance=provenance,
        ),
        extra=extra,
    )
