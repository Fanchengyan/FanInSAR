"""Sentinel-1 SAFE adapter for the mission-neutral Stack (PROPOSAL-0037)."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_pair_configuration
from faninsar.stack.provider import StackSceneProvider
from faninsar.stack.search import (
    MissionPreparedInputs,
    MissionSearchProfile,
    SelectionRequest,
)
from faninsar.stack.session import Stack

from .acquisition import (
    materialize_cdse_roi_safes,
    normalize_roi_wgs84,
    resolve_roi_acquisition,
)
from .search import (
    S1PolarizationRequest,
    S1ProductType,
    canonical_items,
    catalog_for_product,
    complete_safe_asset,
    ensure_asf_catalog,
    filter_burst_polarizations,
    local_destination,
    normalize_polarizations,
    product_filters,
    safe_requires_aux_cal,
    select_orbit_bindings,
    validate_calibration_paths,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from faninsar.data.query import BoundingBox, Points, Polygons
    from faninsar.missions.s1.processing import BurstSelection
    from faninsar.remote.records import CatalogItem, RemoteResourceBudget
    from faninsar.stack.config import CoregMode

logger = setup_logger(__name__)

_DEFAULT_S1_DOWNLOAD_WORKERS = 8


def _acquisition_date_identity(item: CatalogItem) -> str:
    """Return the stable acquisition date used to record download exclusions."""
    start = item.acquisition.start
    if start is not None:
        return start.isoformat()
    return str(item.acquisition.acquisition_id or item.item_id)


def _excluded_acquisition_record(
    item: CatalogItem, destination: Path
) -> dict[str, str]:
    """Build the persisted exclusion record for one failed download.

    PROPOSAL-0055: a download whose bytes fail the provider checksum is
    recorded as ``status: failed`` and excluded from pair formation.  The
    transfer layer owns the ``checksum_mismatch`` reason; the mission layer
    only reacts to it.
    """
    return {
        "date": _acquisition_date_identity(item),
        "path": str(destination),
        "status": "failed",
        "reason": "checksum_mismatch",
    }


def _excluded_acquisition_provenance(
    records: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    """Serialize download exclusions into the mission provenance record.

    Returns
    -------
    dict
        Empty when nothing was excluded; otherwise one deterministic JSON
        entry consumed by the shared input manifest's ``provenance`` block.

    """
    if not records:
        return {}
    payload = [dict(record) for record in records]
    return {"excluded_acquisitions": json.dumps(payload, sort_keys=True)}


def _input_verification_provenance(
    paths: Sequence[str | Path],
) -> dict[str, str]:
    """Surface each downloaded input's verification record in provenance.

    PROPOSAL-0060 makes ``verification.status`` a consumed provenance fact.
    Reads the download-time PROPOSAL-0055 verification record of each input
    (via :func:`faninsar.remote.fetch.complete.read_verification`) and serializes
    ``{path, status, algorithm, size}`` per input into one deterministic JSON
    entry consumed by the shared input manifest's ``provenance`` block. An
    input with no record is surfaced as ``status: unverifiable`` (the honest
    outcome when no provider digest was available).

    Parameters
    ----------
    paths : sequence of str or Path
        Resolved local paths of the downloaded inputs.

    Returns
    -------
    dict
        Always one ``{"input_verification": <json>}`` entry.

    """
    from faninsar.remote.fetch.complete import read_verification

    entries: list[dict[str, object]] = []
    for path in paths:
        resolved = Path(path)
        record = read_verification(resolved) or {}
        entries.append(
            {
                "path": str(resolved),
                "status": record.get("status", "unverifiable"),
                "algorithm": record.get("algorithm"),
                "size": record.get("size", resolved.stat().st_size),
            }
        )
    return {"input_verification": json.dumps(entries, sort_keys=True)}


def _configure_s1_bindings(
    stack: S1Stack,
    *,
    orbit_paths: Mapping[str, object] | Sequence[str | Path] | None,
    calibration_paths: Sequence[str | Path] | None,
) -> None:
    """Resolve explicit local S1 orbit and calibration bindings."""
    intervals: dict[str, tuple[Any, Any]] = {}
    mission_ids: dict[str, str] = {}
    if orbit_paths is not None:
        from faninsar.missions.s1.safe import open_safe_product

        for date_id in stack.catalog.dates:
            products = [
                open_safe_product(path) for path in stack.catalog.paths_for(date_id)
            ]
            starts = [
                swath.sensing_start for product in products for swath in product.swaths
            ]
            stops = [
                swath.sensing_stop for product in products for swath in product.swaths
            ]
            if starts and stops:
                intervals[date_id] = (min(starts), max(stops))
            mission = next(
                (product.mission_id for product in products if product.mission_id),
                "S1",
            )
            mission_ids[date_id] = str(mission)
    selected = select_orbit_bindings(
        orbit_paths,
        intervals,
        mission_ids=mission_ids or None,
    )
    calibrations = validate_calibration_paths(calibration_paths)
    if any(
        safe_requires_aux_cal(path)
        for date_id in stack.catalog.dates
        for path in stack.catalog.paths_for(date_id)
    ):
        message = (
            "Sentinel-1 IPF 002.36 requires AUX_CAL EAP correction, which the "
            "FanInSAR SAFE reader does not yet implement"
        )
        logger.error(message)
        raise NotImplementedError(message)
    if selected:
        stack.config.extra["orbit_paths"] = selected
    else:
        stack.config.extra.pop("orbit_paths", None)
    if calibrations:
        stack.config.extra["calibration_paths"] = calibrations
    else:
        stack.config.extra.pop("calibration_paths", None)


class S1Stack(Stack):
    """Concrete Stack entry point for Sentinel-1 SAFE source paths.

    The adapter owns raw SAFE initialization and installs the provider used by
    every shared Stack stage.  The base session never imports a pair runner or
    otherwise guesses how a source path should be opened.
    """

    @classmethod
    def _search_profile(cls) -> Any:
        """Return the default Sentinel-1 burst search profile."""
        ensure_asf_catalog(product_type="burst")
        return MissionSearchProfile(
            mission="sentinel-1",
            catalog=catalog_for_product("burst"),
            collections=("sentinel-1",),
            default_filters={"dataset": "SLC-BURST", "processingLevel": "BURST"},
            normalizer=canonical_items,
            selector=lambda items, request: canonical_items(items)[: request.limit],
        )

    @classmethod
    def search(
        cls,
        spatial: Points | BoundingBox | Polygons,
        *,
        product_type: S1ProductType = "burst",
        polarizations: S1PolarizationRequest | Sequence[str] | str = ("VV",),
        catalog: str | None = None,
        collections: Sequence[str] | None = None,
        datetime_range: tuple[datetime, datetime] | None = None,
        flight_direction: Literal["ASCENDING", "DESCENDING"] | None = None,
        filters: Mapping[str, Any] | None = None,
        auth_profile: str | None = "earthdata-asf",
        limit: int = 100,
        budget: RemoteResourceBudget | None = None,
    ) -> list[CatalogItem]:
        """Search Sentinel-1 ASF records with mission defaults applied."""
        selected_polarizations = (
            "all" if polarizations is None else normalize_polarizations(polarizations)
        )
        if catalog is None:
            catalog = catalog_for_product(product_type)
        if catalog == "sentinel1-asf":
            ensure_asf_catalog(catalog, product_type=product_type)
        from faninsar import remote

        merged_filters = product_filters(product_type, selected_polarizations, filters)
        profile = MissionSearchProfile(
            mission="sentinel-1",
            catalog=catalog,
            collections=tuple(collections or ()),
            default_filters=merged_filters,
            normalizer=canonical_items,
            selector=lambda values, request: canonical_items(values)[: request.limit],
        )
        items = remote.search(
            spatial,
            catalog=catalog,
            collections=collections,
            datetime_range=datetime_range,
            flight_direction=flight_direction,
            filters=merged_filters,
            auth_profile=auth_profile,
            limit=limit,
            budget=budget,
        )
        ordered = profile.normalize(items)
        if product_type == "burst" and selected_polarizations != "all":
            ordered = filter_burst_polarizations(ordered, selected_polarizations)
        return profile.select(
            ordered,
            SelectionRequest(
                limit=limit,
                datetime_range=datetime_range,
                flight_direction=flight_direction,
                filters=merged_filters,
            ),
        )

    @classmethod
    def from_safes(
        cls,
        paths: Sequence[str | Path],
        *,
        reference: object | None = None,
        polarizations: S1PolarizationRequest | Sequence[str] | str | None = None,
        orbit_paths: Mapping[str, object] | Sequence[str | Path] | None = None,
        calibration_paths: Sequence[str | Path] | None = None,
        coreg_mode: CoregMode = "nesd",
        swaths: tuple[str, ...] | None = None,
        bursts: BurstSelection | None = None,
        **kwargs: Any,
    ) -> Self:
        """Build an S1 Stack and optionally select its reference acquisition.

        Parameters
        ----------
        paths : sequence of path-like
            Sentinel-1 SAFE directories or archives.
        reference : date-like, optional
            Reference acquisition.  Defaults to the existing Stack default.
        polarizations : sequence of str or "all", optional
            Local Sentinel-1 channels to retain when opening SAFE inputs.
        orbit_paths : mapping or sequence, optional
            Explicit local POEORB/RESORB candidates or date bindings.
        calibration_paths : sequence, optional
            Explicit local AUX_CAL/calibration files.
        coreg_mode : {"geometry", "nesd"}, optional
            Stack coregistration mode. Defaults to ``"nesd"``.
        swaths : tuple of str, optional
            Sub-swaths to process. ``None`` selects every sub-swath.
        bursts : BurstSelection, optional
            Per-swath burst selection. ``None`` selects every burst.
        **kwargs : Any
            Remaining :meth:`Stack._from_safes` options.

        Returns
        -------
        S1Stack
            A Stack whose scene production is explicitly provider-owned.

        """
        selected_polarizations = (
            "all" if polarizations is None else normalize_polarizations(polarizations)
        )
        extra = dict(kwargs.pop("extra", {}) or {})
        if "master" in kwargs:
            reject_pair_configuration(
                "S1Stack.from_safes no longer accepts 'master'; use 'reference'"
            )
        if kwargs.get("flatten_stage", "coregistration") != "coregistration":
            message = (
                "Sentinel-1 SAFE does not support non-default flatten_stage; "
                "use 'coregistration'"
            )
            logger.error(message)
            raise ValueError(message)
        if reference is not None:
            kwargs["reference"] = reference

        # The classmethod remains bound to this concrete adapter through
        # ``super()``, so the raw constructor creates an S1Stack directly
        # without duplicating Stack's large configuration signature.
        stack = super()._from_safes(
            paths,
            coreg_mode=coreg_mode,
            swaths=swaths,
            bursts=bursts,
            **kwargs,
        )
        stack.scene_provider = cls._default_scene_provider()
        stack.config.extra.update(extra)
        stack.config.extra.setdefault("polarizations", selected_polarizations)
        _configure_s1_bindings(
            stack,
            orbit_paths=orbit_paths,
            calibration_paths=calibration_paths,
        )
        return stack  # type: ignore[return-value]

    @classmethod
    def from_roi(
        cls,
        roi: Points | BoundingBox | Polygons,
        datetime_range: tuple[datetime, datetime] | None = None,
        *,
        granularity: str = "roi",
        provider: str | None = None,
        polarizations: S1PolarizationRequest | Sequence[str] | str = ("VV",),
        work_dir: str | Path,
        orbit_paths: Mapping[str, object] | Sequence[str | Path] | None = None,
        calibration_paths: Sequence[str | Path] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Search, acquire, and prepare a Sentinel-1 Stack for one ROI.

        Scientific granularity is the user-facing axis; the provider is
        derived.  Bytes transferred depend on ROI shape and burst coverage;
        this method does not promise a fixed byte fraction.

        Provider x granularity table:

        ============  ==========  =================  =================================
        granularity   providers   inferred provider  notes
        ============  ==========  =================  =================================
        roi (default) cdse        cdse               byte-range windows; bytes depend
                                                     on ROI shape
        burst         cdse, asf   cdse               full-burst contiguous row ranges
                                                     (cdse) or ASF burst products
        scene         asf         asf                complete scene/frame products
        ============  ==========  =================  =================================

        Both ``granularity`` and ``provider`` omitted → ``roi`` + ``cdse``.
        Illegal combinations (``roi``+``asf``, ``scene``+``cdse``) raise
        :class:`ValueError` citing this table.  Overlap strips for NESD/ESD
        are fetched internally and are not a user knob.  ``catalog`` /
        ``collections`` are adapter configuration, not constructor
        arguments.

        CDSE paths (``roi`` and ``burst@cdse``) materialize a full-burst-
        height SAFE-subset canvas, splice overlap rows, then call
        :meth:`from_safes` → :meth:`prepare`.  ASF paths download complete
        objects and follow the same ``from_safes`` → ``prepare`` join.

        Parameters
        ----------
        roi : Points, BoundingBox, or Polygons
            FanInSAR spatial query.  A CRS is required; non-WGS84
            coordinates are transformed to EPSG:4326.  A single
            :class:`~faninsar.data.query.Points` query is buffered by
            1 km on the ground before CDSE window inversion.
        datetime_range : tuple, optional
            Inclusive acquisition time range.
        granularity : {"roi", "burst", "scene"}, default="roi"
            Acquisition granularity (not a provider product type).
        provider : {"cdse", "asf"} or None, optional
            Explicit provider; inferred from ``granularity`` when omitted.
        polarizations : sequence of str or "all", default=("VV",)
            Sentinel-1 single/dual channels to retain.
        work_dir : path-like
            Stack work directory and local input cache.
        orbit_paths, calibration_paths : optional
            Explicit local orbit candidates/bindings and AUX_CAL files passed
            to :meth:`from_safes`.
        **kwargs : Any
            Existing :meth:`from_safes` options plus internal ASF search
            options (``flight_direction``, ``filters``, ``limit``,
            ``budget``).  ``catalog`` / ``collections`` remain internal.

        Returns
        -------
        S1Stack
            A locally materialized and prepared Stack.

        Raises
        ------
        ValueError
            If the ROI has no CRS, the granularity x provider pair is
            illegal, the query returns no products, every download fails
            verification, or local construction fails.

        Notes
        -----
        A download whose bytes fail provider checksum verification excludes
        only that acquisition (PROPOSAL-0055): it is warned, recorded as
        ``status: failed`` in ``config.extra["excluded_acquisitions"]``, and
        excluded from pair formation.  Any other download error remains
        operation-fatal.

        """
        resolved_granularity, resolved_provider = resolve_roi_acquisition(
            granularity,
            provider,
        )
        roi = normalize_roi_wgs84(roi)
        extra = dict(kwargs.pop("extra", {}) or {})
        extra["roi_granularity"] = resolved_granularity
        extra["roi_provider"] = resolved_provider
        kwargs["extra"] = extra
        if resolved_provider == "cdse":
            return cls._from_cdse_roi(
                roi,
                datetime_range=datetime_range,
                granularity=resolved_granularity,
                polarizations=polarizations,
                work_dir=work_dir,
                orbit_paths=orbit_paths,
                calibration_paths=calibration_paths,
                **kwargs,
            )
        product_type: S1ProductType = (
            "scene" if resolved_granularity == "scene" else "burst"
        )
        return cls._from_asf_complete(
            roi,
            work_dir=work_dir,
            product_type=product_type,
            polarizations=polarizations,
            datetime_range=datetime_range,
            orbit_paths=orbit_paths,
            calibration_paths=calibration_paths,
            **kwargs,
        )

    @classmethod
    def _from_cdse_roi(
        cls,
        roi: Points | BoundingBox | Polygons,
        *,
        datetime_range: tuple[datetime, datetime] | None,
        granularity: str,
        polarizations: S1PolarizationRequest | Sequence[str] | str,
        work_dir: str | Path,
        orbit_paths: Mapping[str, object] | Sequence[str | Path] | None,
        calibration_paths: Sequence[str | Path] | None,
        **kwargs: Any,
    ) -> Self:
        """Materialize CDSE canvases then join through ``from_safes``."""
        selected_polarizations = normalize_polarizations(polarizations)
        root = Path(work_dir)
        kwargs.pop("collections", None)
        search_keys = (
            "filters",
            "limit",
            "budget",
            "flight_direction",
            "catalog",
            "auth_profile",
        )
        search_kwargs = {key: kwargs.pop(key) for key in search_keys if key in kwargs}
        local_paths = materialize_cdse_roi_safes(
            roi,
            datetime_range=datetime_range,
            polarizations=selected_polarizations,
            work_dir=root,
            granularity=granularity,
            **search_kwargs,
        )
        extra = dict(kwargs.pop("extra", {}) or {})
        extra.update(
            {
                "search_product_type": granularity,
                "search_polarizations": selected_polarizations,
            }
        )
        if orbit_paths is not None:
            kwargs["orbit_paths"] = orbit_paths
        if calibration_paths is not None:
            kwargs["calibration_paths"] = calibration_paths
        kwargs.setdefault("roi", roi)
        kwargs["extra"] = extra
        stack = cls.from_safes(
            local_paths,
            work_dir=root,
            polarizations=selected_polarizations,
            **kwargs,
        )
        prepare = getattr(stack, "prepare", None)
        if callable(prepare):
            prepare()
        else:
            stack.prepare_scenes()
        return stack

    @classmethod
    def _from_asf_complete(
        cls,
        spatial: Points | BoundingBox | Polygons,
        *,
        work_dir: str | Path,
        product_type: S1ProductType = "burst",
        polarizations: S1PolarizationRequest | Sequence[str] | str = ("VV",),
        catalog: str | None = None,
        collections: Sequence[str] | None = None,
        datetime_range: tuple[datetime, datetime] | None = None,
        flight_direction: Literal["ASCENDING", "DESCENDING"] | None = None,
        filters: Mapping[str, Any] | None = None,
        auth_profile: str | None = "earthdata-asf",
        limit: int = 100,
        budget: RemoteResourceBudget | None = None,
        orbit_paths: Mapping[str, object] | Sequence[str | Path] | None = None,
        calibration_paths: Sequence[str | Path] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Search ASF, download complete SAFE inputs, and prepare a Stack.

        Absorbed from the deleted ``from_search`` constructor
        (PROPOSAL-0061).  Burst products poll until ready; scene products
        download immediately.

        """
        selected_polarizations = normalize_polarizations(polarizations)
        catalog_name = catalog or catalog_for_product(product_type)
        if catalog_name == "sentinel1-asf":
            ensure_asf_catalog(catalog_name, product_type=product_type)
        from faninsar import remote

        budget = budget or remote.RemoteResourceBudget()
        items = cls.search(
            spatial,
            product_type=product_type,
            polarizations=selected_polarizations,
            catalog=catalog_name,
            collections=collections,
            datetime_range=datetime_range,
            flight_direction=flight_direction,
            filters=filters,
            auth_profile=auth_profile,
            limit=limit,
            budget=budget,
        )
        selected_items = canonical_items(items)
        if not selected_items:
            message = "Sentinel-1 search returned no products"
            logger.error(message)
            raise ValueError(message)
        root = Path(work_dir)
        poll_scheduler = None
        poll_tokens: list[Any | None] = [None] * len(selected_items)
        if product_type == "burst":
            from faninsar.remote.fetch.complete import _BurstPollScheduler

            poll_scheduler = _BurstPollScheduler()
            poll_tokens = [
                poll_scheduler.register(index) for index in range(len(selected_items))
            ]
        download_plans: list[tuple[CatalogItem, Any, Path]] = []
        identities: list[dict[str, str]] = []
        for item in selected_items:
            asset = complete_safe_asset(item)
            destination = local_destination(root, item, asset)
            # Resolve every SAFE representation before starting any transfer so
            # a malformed later item cannot leave an earlier download crossing
            # the construction boundary.
            download_plans.append((item, asset, destination))
            identities.append(
                {
                    "provider": item.provider,
                    "catalog": item.catalog,
                    "item_id": item.item_id,
                    "asset_key": asset.key,
                }
            )
        worker_limit = getattr(budget, "max_workers", _DEFAULT_S1_DOWNLOAD_WORKERS)
        worker_count = min(len(download_plans), max(1, worker_limit))
        operation_ledger = remote._CallLedger(budget)
        downloaded_paths: list[Path | None] = [None] * len(download_plans)
        excluded_indices: set[int] = set()
        excluded_acquisitions: list[dict[str, str]] = []
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="faninsar-s1-download",
        )
        futures: dict[Any, tuple[str, int]] = {}
        waiting: set[int] = set()
        probe_urls = [asset.href for _item, asset, _destination in download_plans]

        def submit_probe(index: int) -> None:
            """Submit one provider observation without sleeping in a worker."""
            token = poll_tokens[index]
            assert token is not None
            poll_scheduler.set_state(token, "submitted")
            future = executor.submit(
                remote.probe,
                download_plans[index][1],
                budget=budget,
                url=probe_urls[index],
                attempt=poll_scheduler.details()[index]["observations"],
                ledger=operation_ledger,
            )
            futures[future] = ("probe", index)

        if poll_scheduler is not None:
            for index in range(len(download_plans)):
                submit_probe(index)
        else:
            for index, (_item, asset, destination) in enumerate(download_plans):
                future = executor.submit(
                    remote.download,
                    asset,
                    destination,
                    budget=budget,
                    ledger=operation_ledger,
                )
                futures[future] = ("download", index)
        try:
            while futures or waiting:
                if poll_scheduler is not None:
                    claimed = poll_scheduler.claim_due()
                    if claimed is not None and claimed.index in waiting:
                        waiting.remove(claimed.index)
                        submit_probe(claimed.index)
                pending = set(futures)
                if not pending:
                    next_due = poll_scheduler.next_due() if poll_scheduler else None
                    if next_due is not None:
                        time.sleep(max(0.0, min(0.25, next_due - time.monotonic())))
                    continue
                done, pending = wait(
                    pending,
                    timeout=0.25,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue
                for future in done:
                    kind, index = futures.pop(future)
                    if kind == "download":
                        try:
                            downloaded_paths[index] = future.result()
                        except remote.BurstPendingError as pending_error:
                            token = poll_tokens[index]
                            assert token is not None
                            observations = poll_scheduler.observe(token)
                            if observations >= getattr(budget, "max_poll_attempts", 50):
                                remote._fail(
                                    remote.RemoteLimitError, "max_poll_attempts"
                                )
                            probe_urls[index] = pending_error.location
                            poll_scheduler.mark_waiting(
                                token, pending_error.delay_seconds
                            )
                            waiting.add(index)
                        except remote.RemoteIntegrityError as integrity_error:
                            # PROPOSAL-0055 amends the all-or-nothing clause for
                            # download verification only: a bytes-vs-checksum
                            # mismatch skips that acquisition (warned, recorded,
                            # excluded from pair formation) while the remaining
                            # downloads continue.  Every other download error
                            # re-raises into the fail-closed boundary below.
                            if integrity_error.reason != "checksum_mismatch":
                                raise
                            _item, _asset, destination = download_plans[index]
                            record = _excluded_acquisition_record(_item, destination)
                            logger.warning(
                                "Sentinel-1 input failed download verification "
                                "and is excluded from this Stack: %s "
                                "(acquisition %s). Delete the file and "
                                "re-download it before processing.",
                                destination,
                                record["date"],
                            )
                            excluded_indices.add(index)
                            excluded_acquisitions.append(record)
                        continue
                    token = poll_tokens[index]
                    assert token is not None
                    observation = future.result()
                    observations = poll_scheduler.observe(token)
                    if observation.state == "ready":
                        poll_scheduler.set_state(token, "ready")
                        poll_scheduler.set_state(token, "downloading")
                        download_future = executor.submit(
                            remote.download,
                            download_plans[index][1],
                            download_plans[index][2],
                            budget=budget,
                            transfer_url=observation.url,
                            poll_scheduler=poll_scheduler,
                            poll_token=token,
                            ledger=operation_ledger,
                        )
                        futures[download_future] = ("download", index)
                    else:
                        if observations >= getattr(budget, "max_poll_attempts", 50):
                            remote._fail(remote.RemoteLimitError, "max_poll_attempts")
                        probe_urls[index] = observation.url
                        poll_scheduler.mark_waiting(token, observation.delay_seconds)
                        waiting.add(index)
        except BaseException:
            # A failed input must never produce a partially constructed Stack.
            # Verification failures are excluded above (PROPOSAL-0055) and
            # never reach this boundary; any other download error stays
            # operation-fatal.  Pending jobs are cancelled; already-running
            # downloads are allowed to finish so their temporary files and
            # destination locks unwind.
            logger.exception("Sentinel-1 SAFE download failed")
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        local_paths: list[Path] = []
        for index, path in enumerate(downloaded_paths):
            if index in excluded_indices:
                continue
            if path is None:  # pragma: no cover - executor invariant
                message = "Sentinel-1 download completed without a local path"
                logger.error(message)
                raise RuntimeError(message)
            local_paths.append(path)
        if not local_paths:
            message = (
                "every Sentinel-1 input failed download verification; delete "
                "the recorded files and re-download them before preparing"
            )
            logger.error(message)
            raise ValueError(message)
        extra = dict(kwargs.pop("extra", {}) or {})
        if excluded_acquisitions:
            extra["excluded_acquisitions"] = tuple(excluded_acquisitions)
        extra.update(
            {
                "search_product_type": product_type,
                "search_polarizations": selected_polarizations,
                "search_items": tuple(identities),
            }
        )
        if orbit_paths is not None:
            kwargs["orbit_paths"] = orbit_paths
        if calibration_paths is not None:
            kwargs["calibration_paths"] = calibration_paths
        kwargs["extra"] = extra
        stack = cls.from_safes(
            local_paths,
            work_dir=root,
            polarizations=selected_polarizations,
            **kwargs,
        )
        prepare = getattr(stack, "prepare", None)
        if callable(prepare):
            prepare()
        else:
            stack.prepare_scenes()
        return stack

    @classmethod
    def _default_scene_provider(cls) -> Any:
        """Return the Sentinel-1 SAFE production provider."""
        return StackSceneProvider(name="Sentinel-1 SAFE")

    def prepare(self, *, refresh: bool = False) -> Self:
        """Prepare shared inputs, re-enabling orbit auto-fetch on refresh."""
        if refresh:
            self.config.extra["orbit_auto_fetch"] = True
        return super().prepare()

    def _ensure_precise_orbits(self) -> None:
        """Download covering EOFs from s1-orbits before coregistration."""
        from faninsar.missions.s1.orbit_fetch import ensure_covering_orbit_paths
        from faninsar.missions.s1.safe import open_safe_product

        auto_fetch = bool(self.config.extra.get("orbit_auto_fetch", True))
        existing = self.config.extra.get("orbit_paths")
        existing_paths: Mapping[str, Path] | None
        if isinstance(existing, Mapping) and existing:
            existing_paths = {str(key): Path(value) for key, value in existing.items()}
        else:
            existing_paths = None
        polarizations = self.config.extra.get("polarizations")
        selected_polarizations = (
            None if polarizations == "all" else tuple(polarizations or ())
        )
        intervals: dict[str, tuple[Any, Any]] = {}
        mission_ids: dict[str, str] = {}
        for date_id in self.catalog.dates:
            products = [
                open_safe_product(path, polarizations=selected_polarizations)
                for path in self.catalog.paths_for(date_id)
            ]
            starts = [
                swath.sensing_start for product in products for swath in product.swaths
            ]
            stops = [
                swath.sensing_stop for product in products for swath in product.swaths
            ]
            if starts and stops:
                intervals[date_id] = (min(starts), max(stops))
            mission = next(
                (product.mission_id for product in products if product.mission_id),
                "S1",
            )
            mission_ids[date_id] = str(mission)
        if not intervals:
            return
        cache = self.config.extra.get("orbit_cache_dir")
        cache_dir = (
            Path(cache)
            if cache is not None
            else self.config.work_dir / "inputs" / "orbits"
        )
        selected = ensure_covering_orbit_paths(
            intervals=intervals,
            mission_ids=mission_ids,
            cache_dir=cache_dir,
            existing=existing_paths,
            auto_fetch=auto_fetch,
        )
        if selected:
            self.config.extra["orbit_paths"] = selected
        elif existing_paths is None:
            self.config.extra.pop("orbit_paths", None)

    def _prepare_mission(self) -> MissionPreparedInputs:
        """Return SAFE-derived S1 footprints and local mission bindings."""
        from faninsar.missions.s1.safe import open_safe_product

        footprints: list[object] = []
        input_paths: list[Path] = []
        for date_id in self.catalog.dates:
            for path in self.catalog.paths_for(date_id):
                input_paths.append(Path(path))
                product = open_safe_product(
                    path,
                    polarizations=(
                        None
                        if self.config.extra.get("polarizations") == "all"
                        else tuple(self.config.extra.get("polarizations", ()))
                    ),
                )
                footprints.extend(
                    burst.footprint
                    for swath in product.swaths
                    for burst in swath.bursts
                    if burst.footprint
                )
        excluded = self.config.extra.get("excluded_acquisitions", ())
        return MissionPreparedInputs(
            selected_footprints=tuple(footprints),
            orbit_paths_by_acquisition=dict(self.config.extra.get("orbit_paths", {})),
            calibration_paths=tuple(self.config.extra.get("calibration_paths", ())),
            provenance={
                "mission": "sentinel-1",
                "orbit_source": (
                    "external" if self.config.extra.get("orbit_paths") else "embedded"
                ),
                "calibration": str(len(self.config.extra.get("calibration_paths", ()))),
                "polarizations": (
                    "all"
                    if self.config.extra.get("polarizations") == "all"
                    else ",".join(tuple(self.config.extra.get("polarizations", ())))
                ),
                **_excluded_acquisition_provenance(excluded),
                **_input_verification_provenance(input_paths),
            },
        )

    def _requires_inverted_nesd_constants(self) -> bool:
        """Sentinel-1 NESD apply requires inverted date constants (F11)."""
        return self.config.coreg_mode == "nesd"

    def _resolve_s1_nesd_selection(
        self,
    ) -> tuple[tuple[str, ...] | None, BurstSelection | None]:
        """Resolve swaths/bursts for NESD measure and apply.

        No ROI keeps the configured selection (``None`` means all). An ROI
        selects bursts by footprint intersection, including neighboring bursts
        at edges whose footprints overlap the ROI. Geometry apply still crops
        to the ROI window; NESD measures the **full** overlap strips of those
        selected bursts and does not require the overlap pixels to lie inside
        the ROI crop. A single-burst interior ROI has no adjacent pair, so
        measure warns and skips NESD.
        """
        cached = getattr(self, "_s1_nesd_selection", None)
        if cached is not None:
            return cached
        roi = self._effective_roi_with_mask().roi
        if roi is None:
            result = (self.config.swaths, self.config.bursts)
            self._s1_nesd_selection = result
            return result
        from faninsar.missions.s1.processing import _select_bursts_by_roi
        from faninsar.missions.s1.safe import open_safe_product

        logger.info("ROI provided; explicit swaths/bursts selection ignored")
        ref_paths = list(self.catalog.paths_for(self.reference))
        polarizations = self.config.extra.get("polarizations")
        selected_polarizations = (
            None if polarizations == "all" else tuple(polarizations or ())
        )
        products = [
            open_safe_product(path, polarizations=selected_polarizations)
            for path in ref_paths
        ]
        ordered = sorted(
            products[0].swaths,
            key=lambda item: item.slant_range_time_s,
        )
        swath_tuple = tuple(item.swath for item in ordered)
        ref_orbit = self._orbit_path_for(self.reference)
        orbits = [ref_orbit] * len(ref_paths)
        resolved = _select_bursts_by_roi(
            roi,
            ref_paths,
            swath_tuple,
            orbits=orbits,
            dem=self._geometry_dem(),
        )
        bursts = _burst_selection_from_resolved(resolved, len(ref_paths), swath_tuple)
        bursts, swath_tuple = _drop_empty_swaths(bursts, swath_tuple)
        if not swath_tuple:
            reject_pair_configuration("ROI selection contains no bursts")
        result = (swath_tuple, bursts)
        self._s1_nesd_selection = result
        return result

    def _measure_pair_misreg(self, primary: str, secondary: str) -> Any:
        """Measure one pair with overlap ESD + overlap Ampcor (not the pair engine)."""
        from faninsar.missions.s1.measure import measure_s1_pair_misreg

        swaths, bursts = self._resolve_s1_nesd_selection()
        return measure_s1_pair_misreg(
            _source_input(self.catalog.paths_for(primary)),
            _source_input(self.catalog.paths_for(secondary)),
            dem=self._geometry_dem(),
            swaths=swaths,
            bursts=bursts,
            coherence_threshold=self.config.esd_coherence_threshold,
            ampcor_snr_threshold=self.config.ampcor_snr_threshold,
            primary_orbit_path=self._orbit_path_for(primary),
            secondary_orbit_path=self._orbit_path_for(secondary),
            device=self.config.device,
        )

    def _geo_lut_cache_dir(self) -> Path:
        """Return the shared content-addressed geo2rdr LUT cache."""
        configured = self.config.extra.get("geo_lut_cache_dir")
        if configured is not None:
            return Path(configured)
        return self.config.work_dir / "cache" / "geo-lut"

    def _prepare_reference_scene(self, *, overwrite: bool) -> Any:
        """Compute (or reuse) the Reference scene product once per run."""
        from faninsar.missions.s1.coregister import prepare_s1_reference

        swaths, bursts = self._resolve_s1_nesd_selection()
        return prepare_s1_reference(
            _source_input(self.catalog.paths_for(self.reference)),
            output_scene=self._scene_dir(self.reference),
            swaths=swaths,
            bursts=bursts,
            roi=self._effective_roi_with_mask().roi,
            dem=self._geometry_dem(),
            coregistration_grid=self.config.coregistration_grid,
            geo_grid=self.config.geo_grid,
            geo_lut_cache_dir=self._geo_lut_cache_dir(),
            orbit_path=self._orbit_path_for(self.reference),
            device=self.config.device,
            overwrite=overwrite,
        )

    def _coregister_one_scene(
        self,
        date_id: str,
        *,
        output_dir: Path,
        misreg_az_px: float,
        misreg_rg_px: float,
        overwrite: bool,
        reference_context: Any = None,
        inputs: Mapping[str, str] | None = None,
        properties: Mapping[str, object] | None = None,
    ) -> Any:
        """Resample one secondary onto the once-prepared Reference context."""
        from faninsar.missions.s1.coregister import coregister_s1_secondary

        if reference_context is None:
            reference_context = self._prepare_reference_scene(overwrite=False)
        coregister_s1_secondary(
            reference_context,
            _source_input(self.catalog.paths_for(date_id)),
            output_scene=output_dir,
            misreg_az_px=misreg_az_px,
            misreg_rg_px=misreg_rg_px,
            secondary_orbit_path=self._orbit_path_for(date_id),
            device=self.config.device,
            inputs=inputs,
            properties=properties,
            overwrite=overwrite,
        )
        return None


def _source_input(paths: tuple[Path, ...]) -> Path | tuple[Path, ...]:
    """Return a scalar path for one-frame dates, else the frame tuple."""
    if len(paths) == 1:
        return paths[0]
    return paths


def _burst_selection_from_resolved(
    resolved: dict[tuple[int, str], list[int]],
    frame_count: int,
    swaths: tuple[str, ...],
) -> BurstSelection:
    """Convert ROI ``(frame, swath)`` indices into a BurstSelection mapping."""
    if frame_count == 1:
        return {swath: list(resolved.get((0, swath), [])) for swath in swaths}
    return [
        {swath: list(resolved.get((frame, swath), [])) for swath in swaths}
        for frame in range(frame_count)
    ]


def _drop_empty_swaths(
    bursts: BurstSelection,
    swaths: tuple[str, ...],
) -> tuple[BurstSelection, tuple[str, ...]]:
    """Drop swaths with no selected bursts so normalization does not reject."""
    if isinstance(bursts, list):
        kept = tuple(
            swath for swath in swaths if any(frame.get(swath) for frame in bursts)
        )
        trimmed = [
            {swath: list(frame.get(swath, [])) for swath in kept} for frame in bursts
        ]
        return trimmed, kept
    kept = tuple(swath for swath in swaths if bursts.get(swath))
    return {swath: list(bursts.get(swath, [])) for swath in kept}, kept


__all__ = ["S1Stack"]
