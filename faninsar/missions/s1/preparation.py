"""Sentinel-1 source requests and explicit input qualification."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import fields, is_dataclass, replace
from datetime import date, datetime
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
    acquire: Any, pending: Mapping[str, Any], context: PreparationContext
) -> dict[str, Any]:
    """Acquire eligible dates with bounded admission and one deferred pass."""
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

    deferred = [
        target
        for target in pending
        if isinstance(results[target], PreparationFailure)
        and results[target].reason == "rate_limit_exhausted"
    ]
    if deferred:
        logger.warning("CDSE regular acquire drained: deferred_targets=%s", deferred)
        _reconfigure_deferred_pass_policy(operation)
        logger.info(
            "CDSE final pass started: targets=%d "
            "active_files=1 total_ranges=1 retries=0",
            len(deferred),
        )
        for target in deferred:
            results[target] = acquire(target, pending[target])
    return results


def _prepare_s1_inputs(stack: S1Stack, context: PreparationContext) -> PreparedInputs:
    """Acquire selected targets, then qualify each one's EOF/AUX dependencies."""
    from faninsar.missions.s1.stack import _configure_s1_bindings
    from faninsar.stack.session import Stack

    failures = []
    requested = []
    paths = []
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
        pending = {}
        for target, items in groups.items():
            # A committed qualified source is a local snapshot, not a new remote query.
            if (
                stack._inputs_state is not None
                and target in stack.catalog.paths
                and all(p.exists() for p in stack.catalog.paths_for(target))
            ):
                paths.extend(stack.catalog.paths_for(target))
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
                if "every Sentinel-1 input failed download verification" not in str(
                    error
                ):
                    raise
                return PreparationFailure(
                    target, "acquisition", "checksum_mismatch", error
                )
            except (OSError, RemoteError) as error:
                return PreparationFailure(
                    target,
                    "acquisition",
                    getattr(error, "reason", type(error).__name__),
                    error,
                )

        parallel = (
            context.request["provider"] == "cdse"
            and context.request["granularity"] == "burst"
            and pending
            and all(len(items) == 1 for items in pending.values())
        )
        results = (
            _acquire_dates(acquire, pending, context)
            if parallel
            else {target: acquire(target, items) for target, items in pending.items()}
        )
        for target in pending:
            result = results[target]
            if isinstance(result, PreparationFailure):
                failures.append(result)
            else:
                paths.extend(result)
    else:
        paths.extend(context.request["paths"])
    if not paths:
        return PreparedInputs(
            sources={}, requested=tuple(requested), failures=tuple(failures)
        )
    paths = list(dict.fromkeys(paths))
    catalog = SceneCatalog.from_paths(paths)
    if not requested:
        requested.extend(catalog.dates)
    qualified = {}
    footprints = []
    orbits = {}
    aux_cal = {}
    provenance = {}
    for target in catalog.dates:
        config = replace(
            stack._request_config, extra=thaw(freeze(stack._request_config.extra))
        )
        if context.offline:
            config.extra.update(
                {"orbit_auto_fetch": False, "aux_cal_auto_fetch": False}
            )
        local = type(stack)(
            config=config, catalog=SceneCatalog({target: catalog.paths_for(target)})
        )
        local._preparation_budget = context.budget
        try:
            if not all(path.exists() for path in catalog.paths_for(target)):
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
            qualified[target] = catalog.paths_for(target)
            footprints.extend(mission.selected_footprints)
            orbits.update(mission.orbit_paths_by_acquisition)
            aux_cal.update(mission.aux_cal_paths_by_acquisition)
            provenance.update(mission.provenance)
        except (OSError, RemoteError) as error:
            failures.append(
                PreparationFailure(
                    target,
                    "mission_inputs",
                    getattr(error, "reason", type(error).__name__),
                    error,
                )
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
