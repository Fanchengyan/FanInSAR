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
from faninsar.stack.provider import SourceHandle, StackSceneProvider
from faninsar.stack.search import (
    MissionPreparedInputs,
    MissionSearchProfile,
    SelectionRequest,
)
from faninsar.stack.session import Stack

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


def _produce_s1_pair(
    primary_path: SourceHandle,
    secondary_path: SourceHandle,
    *,
    output_dir: Path,
    options: dict[str, Any],
) -> Any:
    """Dispatch one admitted SAFE pair through the S1 production adapter."""
    from faninsar.missions.s1.processing import produce_interferogram_pair

    primary_sources = primary_path._resolve()
    secondary_sources = secondary_path._resolve()

    # Preserve the scalar callback shape for ordinary one-frame acquisitions;
    # frame stacks remain tuples and are consumed by the production adapter.
    primary_input: Path | tuple[Path, ...] = (
        primary_sources[0] if len(primary_sources) == 1 else primary_sources
    )
    secondary_input: Path | tuple[Path, ...] = (
        secondary_sources[0] if len(secondary_sources) == 1 else secondary_sources
    )

    return produce_interferogram_pair(
        primary_input,
        secondary_input,
        output_dir=output_dir,
        **options,
    )


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
        stack.scene_provider = StackSceneProvider(
            name="Sentinel-1 SAFE",
            produce_pair=_produce_s1_pair,
        )
        stack.config.extra.update(extra)
        stack.config.extra.setdefault("polarizations", selected_polarizations)
        _configure_s1_bindings(
            stack,
            orbit_paths=orbit_paths,
            calibration_paths=calibration_paths,
        )
        return stack  # type: ignore[return-value]

    @classmethod
    def from_search(
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

        Parameters
        ----------
        spatial : Points, BoundingBox, or Polygons
            FanInSAR spatial query.
        work_dir : path-like
            Stack work directory and local input cache.
        product_type : {"burst", "scene"}, default="burst"
            ASF virtual burst or complete full-scene/frame product family.
        polarizations : sequence of str or "all", default=("VV",)
            Sentinel-1 single/dual channels to retain.  ``"all"`` leaves
            filtering to the local SAFE reader.
        **kwargs : Any
            Existing :meth:`from_safes` and Stack preparation options.
        catalog, collections : str or sequence of str, optional
            ASF catalog and collection identities.
        datetime_range : tuple, optional
            Inclusive acquisition time range.
        flight_direction : {"ASCENDING", "DESCENDING"}, optional
            Orbit direction filter.
        filters : mapping, optional
            Advanced ASF filters.
        auth_profile : str, optional
            Remote authentication profile.
        limit : int, optional
            Maximum candidate count.
        budget : RemoteResourceBudget, optional
            Remote request and transfer resource limits.
        orbit_paths, calibration_paths : optional
            Explicit local orbit candidates/bindings and AUX_CAL files passed
            to :meth:`from_safes`.

        Returns
        -------
        S1Stack
            A locally downloaded and prepared Stack.

        Raises
        ------
        ValueError
            If the query has no complete SAFE representation, every download
            fails verification, or local construction fails.

        Notes
        -----
        A download whose bytes fail provider checksum verification excludes
        only that acquisition (PROPOSAL-0055): it is warned, recorded as
        ``status: failed`` in ``config.extra["excluded_acquisitions"]``, and
        excluded from pair formation.  Any other download error remains
        operation-fatal.

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
            from faninsar.remote.transfer import _BurstPollScheduler

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
    def open(cls, work_dir: str | Path, **processing_options: Any) -> Self:
        """Reopen a local S1 Stack from ``input_manifest.json``.

        The method deliberately performs no remote search.  It accepts the
        compact manifest forms emitted by the shared Stack seam and delegates
        local source validation to :meth:`from_safes`.
        """
        root = Path(work_dir)
        manifest_path = root / "input_manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.exception("cannot read S1 input manifest %s", manifest_path)
            message = f"invalid S1 input manifest: {manifest_path}"
            raise ValueError(message) from error
        if not isinstance(payload, Mapping):
            message = "S1 input manifest must contain a JSON object"
            logger.error(message)
            raise TypeError(message)
        if payload.get("schema") is not None:
            from faninsar.stack.input_manifest import read_manifest

            try:
                payload = read_manifest(manifest_path, work_dir=root)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                logger.exception("S1 input manifest validation failed")
                message = f"invalid S1 input manifest: {manifest_path}"
                raise ValueError(message) from error
        sources = payload.get("sources")
        raw_paths: list[str] | tuple[str, ...] | None = None
        if isinstance(sources, Mapping):
            source_values = [
                entry["path"]
                for entries in sources.values()
                if isinstance(entries, (list, tuple))
                for entry in entries
                if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
            ]
            raw_paths = source_values
        if raw_paths is None:
            raw_paths = payload.get("safe_paths", payload.get("local_safe_paths"))
        if raw_paths is None:
            inputs = payload.get("inputs", ())
            raw_paths = [
                value.get("local_path")
                for value in inputs
                if isinstance(value, Mapping) and value.get("local_path")
            ]
        if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
            message = "S1 input manifest has no local SAFE paths"
            logger.error(message)
            raise ValueError(message)
        paths = tuple(Path(path) for path in raw_paths)
        if any(not path.exists() for path in paths):
            missing = next(path for path in paths if not path.exists())
            message = f"S1 SAFE input is missing: {missing}"
            logger.error(message)
            raise ValueError(message)
        options = dict(processing_options)
        manifest_options = payload.get("processing_options")
        if isinstance(manifest_options, Mapping):
            for key in ("coreg_mode", "multilook", "reference"):
                if key not in options and key in manifest_options:
                    options[key] = manifest_options[key]
            extra = dict(options.get("extra", {}) or {})
            for key in ("product_type", "polarizations"):
                if key in manifest_options:
                    extra[f"search_{key}"] = manifest_options[key]
            options["extra"] = extra
        manifest_orbits = payload.get("orbits")
        if isinstance(manifest_orbits, Mapping):
            orbit_paths = {
                str(key): value["path"]
                for key, value in manifest_orbits.items()
                if isinstance(value, Mapping) and isinstance(value.get("path"), str)
            }
            if orbit_paths:
                options.setdefault("orbit_paths", orbit_paths)
        manifest_calibration = payload.get("calibration")
        if isinstance(manifest_calibration, (list, tuple)):
            options.setdefault(
                "calibration_paths",
                [
                    value["path"]
                    for value in manifest_calibration
                    if isinstance(value, Mapping) and isinstance(value.get("path"), str)
                ],
            )
        provenance = payload.get("provenance")
        if isinstance(provenance, Mapping) and "polarizations" in provenance:
            encoded = provenance["polarizations"]
            if encoded == "all":
                options.setdefault("polarizations", "all")
            elif isinstance(encoded, str):
                values = tuple(
                    value.strip() for value in encoded.split(",") if value.strip()
                )
                options.setdefault("polarizations", normalize_polarizations(values))
        options.pop("work_dir", None)
        stack = cls.from_safes(paths, work_dir=root, **options)
        extra = getattr(getattr(stack, "config", None), "extra", None)
        if isinstance(extra, dict):
            extra["orbit_auto_fetch"] = False
        prepare = getattr(stack, "prepare", None)
        if callable(prepare):
            prepare(refresh=False)
        else:
            stack.prepare_scenes()
        return stack

    def prepare(self, *, refresh: bool = False) -> Self:
        """Prepare shared inputs, re-enabling orbit auto-fetch on refresh."""
        if refresh:
            self.config.extra["orbit_auto_fetch"] = True
        return super().prepare(refresh=refresh)

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
        for date_id in self.catalog.dates:
            for path in self.catalog.paths_for(date_id):
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
            },
        )


__all__ = ["S1Stack"]
