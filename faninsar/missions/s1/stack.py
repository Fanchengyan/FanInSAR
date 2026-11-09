"""Sentinel-1 SAFE adapter for the mission-neutral Stack (PROPOSAL-0037)."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from faninsar.logging import reject, setup_logger
from faninsar.processing.errors import (
    reject_invalid_state,
    reject_pair_configuration,
)
from faninsar.stack.provider import StackSceneProvider
from faninsar.stack.search import MissionPreparedInputs
from faninsar.stack.session import Stack

from .acquisition import (
    _bind_cdse_item,
    _binding_track_key,
    asf_burst_selectors,
    asf_frame_selectors,
    discover_cdse_slc_items,
    materialize_cdse_roi_safes,
    materialize_s1_acquisitions,
    normalize_roi_wgs84,
)
from .s1search import search_acquisitions
from .search import (
    S1PolarizationRequest,
    canonical_items,
    complete_safe_asset,
    ensure_asf_catalog,
    local_destination,
    normalize_polarizations,
    select_orbit_bindings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from faninsar.data.query import BoundingBox, Points, Polygons
    from faninsar.missions.s1.processing import BurstSelection
    from faninsar.remote.records import CatalogItem, RemoteResourceBudget
    from faninsar.stack.config import CoregMode
    from faninsar.stack.preparation import PreparationContext, PreparedInputs

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
    aux_cal_paths: Mapping[str, object] | Sequence[str | Path] | None,
) -> None:
    """Resolve explicit local S1 orbit and AUX_CAL bindings.

    AUX_CAL pins are recorded per IPF 002.36 date; dates they do not cover
    are filled in by :meth:`S1Stack._prepare_mission`.
    """
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
    aux_cal: dict[str, Path] = {}
    if aux_cal_paths is not None:
        from faninsar.missions.s1.aux_cal import select_aux_cal_bindings

        aux_cal = select_aux_cal_bindings(aux_cal_paths, _eap_acquisitions(stack))
    if selected:
        stack.config.extra["orbit_paths"] = selected
    else:
        stack.config.extra.pop("orbit_paths", None)
    if aux_cal:
        stack.config.extra["aux_cal_paths"] = aux_cal
    else:
        stack.config.extra.pop("aux_cal_paths", None)


def _eap_acquisitions(stack: Stack) -> dict[str, tuple[str, Any]]:
    """Return ``date -> (mission, sensing_mid)`` of IPF 002.36 acquisitions."""
    from faninsar.missions.s1.aux_cal import requires_eap_correction
    from faninsar.missions.s1.safe import open_acquisition_products

    polarizations = stack.config.extra.get("polarizations")
    selected = None if polarizations in (None, "all") else tuple(polarizations)
    result: dict[str, tuple[str, Any]] = {}
    for date_id in stack.catalog.dates:
        products = open_acquisition_products(
            stack.catalog.paths_for(date_id), polarizations=selected
        )
        flagged = [product for product in products if requires_eap_correction(product)]
        if not flagged:
            continue
        swaths = [swath for product in flagged for swath in product.swaths]
        starts = [swath.sensing_start for swath in swaths]
        stops = [swath.sensing_stop for swath in swaths]
        midpoint = min(starts) + (max(stops) - min(starts)) / 2
        result[date_id] = (str(flagged[0].mission_id).upper(), midpoint)
    return result


class S1Stack(Stack):
    """Concrete Stack entry point for Sentinel-1 SAFE source paths.

    The adapter owns raw SAFE initialization and installs the provider used by
    every shared Stack stage.  The base session never imports a pair runner or
    otherwise guesses how a source path should be opened.
    """

    # The single Sentinel-1 search implementation (PROPOSAL-0072 §2):
    # ``S1Stack.search`` and ``search_acquisitions`` are the same function.
    search = staticmethod(search_acquisitions)

    _sensor_registry_key = "sentinel1"
    _default_nlooks_divisor: ClassVar[float] = 0.64

    @classmethod
    def from_safes(
        cls,
        paths: Sequence[str | Path],
        *,
        reference: object | None = None,
        polarizations: S1PolarizationRequest | Sequence[str] | str | None = None,
        orbit_paths: Mapping[str, object] | Sequence[str | Path] | None = None,
        aux_cal_paths: Mapping[str, object] | Sequence[str | Path] | None = None,
        coreg_mode: CoregMode = "geometry",
        swaths: tuple[str, ...] | None = None,
        bursts: BurstSelection | None = None,
        **kwargs: Any,
    ) -> Self:
        """Build an S1 Stack and optionally select its reference acquisition.

        Parameters
        ----------
        paths : sequence of path-like
            Sentinel-1 SAFE directories or ``.zip`` archives.  Accepts whole
            frame SAFEs, single-burst ASF ``-BURST`` zips, and the
            ``tracks/<trackKey>/acquisitions`` path-SAFEs that
            :meth:`from_search` materializes — so acquisitions downloaded
            externally (whole frames or burst products) can start here
            directly.  Paths whose track cannot be resolved (e.g. an old
            frame name with no relative orbit) still load; they simply do
            not participate in track/polygon ``sel`` routing.
        reference : date-like, optional
            Reference acquisition.  Defaults to the existing Stack default.
        polarizations : sequence of str or "all", optional
            Local Sentinel-1 channels to retain when opening SAFE inputs.
        orbit_paths : mapping or sequence, optional
            Explicit local POEORB/RESORB candidates or date bindings.
        aux_cal_paths : mapping or sequence, optional
            Explicit local Sentinel-1 AUX_CAL files (``.SAFE.zip``, ``.SAFE``,
            or ``s1?-aux-cal.xml``) or date bindings.  Only IPF 002.36
            products use them, for the elevation antenna phase correction
            applied on load; uncovered dates are downloaded from SAR-MPC
            during :meth:`prepare_data`.
        coreg_mode : {"geometry", "nesd"}, optional
            Stack coregistration mode. Defaults to ``"geometry"``.
            Tests that exercise NESD pass ``coreg_mode="nesd"``.
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
            reject(message, ValueError, logger=logger)
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
        from faninsar.stack.preparation import freeze, thaw

        request = thaw(stack.request)
        request.update({"orbit_paths": orbit_paths, "aux_cal_paths": aux_cal_paths})
        stack.request = freeze(request)
        stack._request_config.extra = dict(stack.config.extra)
        return stack  # type: ignore[return-value]

    @classmethod
    def from_search(
        cls,
        search: Any,
        *,
        work_dir: str | Path,
        roi: Points | BoundingBox | Polygons | None = None,
        provider: Literal["cdse", "asf"] = "cdse",
        granularity: Literal["roi", "burst"] = "roi",
        polarizations: S1PolarizationRequest | Sequence[str] | str | None = None,
        coreg_mode: CoregMode = "geometry",
        orbit_group: str | None = None,
        orbit_paths: Mapping[str, object] | Sequence[str | Path] | None = None,
        aux_cal_paths: Mapping[str, object] | Sequence[str | Path] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Build a Stack from a pre-filtered :class:`S1Search` result set.

        The acquisitions to materialize come from ``search.items`` — already
        filtered/composed client-side — so the provider search step is skipped
        entirely.  ``provider`` picks how those items are downloaded; the
        search catalog itself is not a download constraint:

        - ``"cdse"`` (default) → byte-range windows materialized via the
          CDSE range path; ``granularity`` selects the window shape.  When
          ``search.catalog`` is ``"sentinel1-asf"`` the searched items are
          re-discovered inside the CDSE catalog and their
          ``absoluteBurstID``/``sensingStart`` select the same bursts in the
          CDSE parent frames (``asf_burst_selectors``).
        - ``"asf"`` → complete burst products via the ASF burst-extractor
          path.  ASF always downloads whole bursts, so ``granularity`` is
          forced to ``"burst"``.

        ``roi`` is optional.  When given it bounds processing — CDSE crops to
        byte ranges on download; ASF downloads whole products but ``prepare_data``/
        ``coregister`` still crop to the ROI.  When omitted the full
        frame/burst extent is used.

        Parameters
        ----------
        search : S1Search
            Filtered/composed acquisitions returned by :meth:`search`
            (:func:`~faninsar.missions.s1.s1search.search_acquisitions`).
        work_dir : path-like
            Stack work directory and local input cache.  Required — the
            Stack persists ``stack.json``, acquisitions, and products here.
        roi : spatial, optional
            Optional processing bound.  ``search.spatial`` is reused when the
            search was ROI-scoped and ``roi`` is omitted and CDSE windows
            still need one — pass it explicitly to be safe.
        provider : {"cdse", "asf"}, optional
            Download provider, default ``"cdse"``.  ``"cdse"`` re-discovers
            the acquisitions on the CDSE catalog and range-fetches the
            selected windows; ``"asf"`` downloads the ASF whole-burst
            products directly.  The provider is independent of the search
            catalog — an ASF-searched result can still be downloaded
            through CDSE.
        granularity : {"roi", "burst"}, optional
            Download granularity.  ``"roi"`` (default) range-fetches the
            polygon-union row windows on the CDSE path; ``"burst"``
            materializes whole bursts.  Always ``"burst"`` on the ASF path.
        polarizations : sequence of str or "all", optional
            Sentinel-1 single/dual channels to retain.
        coreg_mode : {"geometry", "nesd"}, optional
            Stack coregistration mode.
        orbit_group : {"auto", "require"} or str, optional
            Track selection when ``search.items`` spans multiple orbit groups
            (``relative_orbit`` + ``orbit_direction``; satellites sharing one
            orbit land in the same group).  ``None`` (default) keeps every
            selected track.  ROI range requests share one transfer pool;
            each track retains its own acquisition directory.  Pass an
            explicit ``"<REL_ORBIT>/<DIRECTION>"`` selector to pick one.
        orbit_paths, aux_cal_paths : optional
            Explicit local orbit and AUX_CAL candidates/bindings passed to
            :meth:`from_safes`.  When ``orbit_paths`` is omitted, covering
            EOFs are prefetched from anonymous s1-orbits on a dedicated
            thread while XML/burst bytes transfer, then bound at the first
            precise-orbit stage.
        **kwargs
            Forwarded to the provider constructor / ``from_safes``.

        Returns
        -------
        S1Stack
            Configured Stack over the selected acquisition request.

        """
        from faninsar.missions.s1.s1search import S1Search

        if not isinstance(search, S1Search):
            message = (
                "from_search expects an S1Search, got "
                f"{type(search).__name__}; produce one via "
                "S1Stack.search(...) or search_acquisitions(...)"
            )
            raise TypeError(message)
        if not search.items:
            message = "from_search received an empty S1Search; nothing to download"
            reject(message, ValueError, logger=logger)
        selected_polarizations = normalize_polarizations(
            search.query.get("polarizations", ("VV",))
            if polarizations is None
            else polarizations
        )
        effective_roi = roi if roi is not None else search.spatial
        if effective_roi is not None:
            effective_roi = normalize_roi_wgs84(effective_roi)

        # ``provider`` selects the download path; the search itself is
        # catalog-agnostic.  ``provider="cdse"`` re-discovers the same
        # acquisitions inside the CDSE catalog (the two catalogs share the
        # ``absoluteBurstID`` namespace and the sensing-time axis), so the
        # searched items only act as the selection, never as the asset.
        if provider not in ("cdse", "asf"):
            message = f"provider must be 'cdse' or 'asf'; got {provider!r}"
            reject(message, ValueError, logger=logger)
        if provider == "asf":
            # ASF only ships whole burst products; there is no roi/scene
            # granularity to select.
            granularity = "burst"

        from faninsar.missions.s1.preparation import serialize_search
        from faninsar.stack.preparation import freeze, thaw

        stack = cls.from_safes(
            (),
            work_dir=work_dir,
            roi=effective_roi,
            polarizations=selected_polarizations,
            coreg_mode=coreg_mode,
            orbit_paths=orbit_paths,
            aux_cal_paths=aux_cal_paths,
            **kwargs,
        )
        request = thaw(stack.request)
        request.update(
            {
                "kind": "search",
                "search": serialize_search(search),
                "provider": provider,
                "granularity": granularity,
                "orbit_group": orbit_group,
            }
        )
        stack.request = freeze(request)
        return stack

    @classmethod
    def _materialize_from_search(
        cls,
        search: Any,
        *,
        work_dir: str | Path,
        provider: Literal["cdse", "asf"],
        granularity: str,
        polarizations: Any,
        coreg_mode: CoregMode,
        orbit_group: str | None,
        orbit_paths: Mapping[str, object] | Sequence[str | Path] | None,
        aux_cal_paths: Mapping[str, object] | Sequence[str | Path] | None,
        effective_roi: Any,
        kwargs: dict[str, Any],
    ) -> Self:
        """Download selected acquisitions then join through ``from_safes``."""
        selected_polarizations = polarizations
        if provider == "cdse":
            requested = (
                None
                if selected_polarizations == "all"
                else {str(p).upper() for p in selected_polarizations}
            )
            # Split items by their own catalog: CDSE items already carry the
            # OData node URLs the range path reads (bind directly), while ASF
            # items carry whole-product assets the CDSE range path cannot
            # read — those are re-discovered on CDSE narrowed to the ASF
            # items' sensing window and track identities.  A mixed search
            # (e.g. S1A from ASF + S1B/C/D from CDSE) splits per item so both
            # catalogs feed the same download.
            cdse_items = [
                i for i in search.items if getattr(i, "catalog", "") == "sentinel1-cdse"
            ]
            asf_items = [
                i for i in search.items if getattr(i, "catalog", "") == "sentinel1-asf"
            ]
            bindings: list[dict[str, Any]] | None = []
            for item in cdse_items:
                bindings.extend(_bind_cdse_item(item))
            if asf_items:
                absolute_ids, sensing_times = asf_burst_selectors(asf_items)
                frame_window, frame_tracks = asf_frame_selectors(asf_items)
            else:
                absolute_ids, sensing_times = [], []
                frame_window, frame_tracks = None, set()
            if bindings is not None and requested is not None:
                bindings = [b for b in bindings if b.get("polarization") in requested]
            if effective_roi is None:
                message = (
                    "CDSE from_search needs a spatial bound for byte-range "
                    "windows; pass roi=... or use a spatial-scoped search"
                )
                reject(message, ValueError, logger=logger)
            dem = kwargs.pop("dem", None)
            if asf_items:
                # Re-discover the ASF-selected acquisitions on CDSE, narrowed
                # to the ASF items' sensing window, then keep only bindings
                # in the searched tracks so other orbits on the same ROI do
                # not leak in.  Burst-level items additionally pass their
                # absoluteBurstID/sensingStart selector through.  Any CDSE
                # items in the same search are merged in afterwards.
                rediscovered = discover_cdse_slc_items(
                    effective_roi,
                    datetime_range=frame_window,
                    polarizations=selected_polarizations,
                )
                if frame_tracks:
                    rediscovered = [
                        b for b in rediscovered if _binding_track_key(b) in frame_tracks
                    ]
                per_mission = kwargs.pop("per_mission_limit", None)
                if per_mission is not None:
                    rediscovered = _cap_per_mission(rediscovered, int(per_mission))
                if bindings:
                    seen = {str(b.get("measurement_url")) for b in bindings}
                    bindings.extend(
                        b
                        for b in rediscovered
                        if str(b.get("measurement_url")) not in seen
                    )
                else:
                    bindings = rediscovered
                local_paths = materialize_s1_acquisitions(
                    effective_roi,
                    provider="cdse",
                    granularity=granularity,
                    polarizations=selected_polarizations,
                    work_dir=work_dir,
                    items=bindings,
                    absolute_burst_ids=absolute_ids or None,
                    catalog_sensing_times=sensing_times or None,
                    orbit_group=orbit_group,
                    dem=dem,
                    coreg_mode=coreg_mode,
                )
            else:
                local_paths = materialize_s1_acquisitions(
                    effective_roi,
                    provider="cdse",
                    granularity=granularity,
                    polarizations=selected_polarizations,
                    work_dir=work_dir,
                    items=bindings,
                    orbit_group=orbit_group,
                    dem=dem,
                    coreg_mode=coreg_mode,
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
            if aux_cal_paths is not None:
                kwargs["aux_cal_paths"] = aux_cal_paths
            if effective_roi is not None:
                kwargs.setdefault("roi", effective_roi)
            kwargs["extra"] = extra
            return cls.from_safes(
                local_paths,
                work_dir=work_dir,
                polarizations=selected_polarizations,
                coreg_mode=coreg_mode,
                **kwargs,
            )

        # ASF burst-extractor path: whole-burst products reassembled into
        # the same tracks/<key>/acquisitions path-SAFE layout the CDSE
        # burst path emits.
        spatial_arg = effective_roi if effective_roi is not None else search.spatial
        if spatial_arg is None:
            message = (
                "from_search needs a spatial bound for the ASF path; the "
                "search was not spatial-scoped and roi= was not given"
            )
            reject(message, ValueError, logger=logger)
        excluded: list[dict[str, str]] = []
        local_paths = materialize_s1_acquisitions(
            spatial_arg,
            provider="asf",
            granularity="burst",
            polarizations=selected_polarizations,
            work_dir=work_dir,
            items=list(search.items),
            on_exclusion=lambda item, destination: excluded.append(
                _excluded_acquisition_record(item, destination)
            ),
        )
        extra = dict(kwargs.pop("extra", {}) or {})
        if excluded:
            extra["excluded_acquisitions"] = tuple(excluded)
        extra.update(
            {
                "search_product_type": "burst",
                "search_polarizations": selected_polarizations,
            }
        )
        if orbit_paths is not None:
            kwargs["orbit_paths"] = orbit_paths
        if aux_cal_paths is not None:
            kwargs["aux_cal_paths"] = aux_cal_paths
        kwargs.setdefault("roi", spatial_arg)
        kwargs["extra"] = extra
        return cls.from_safes(
            local_paths,
            work_dir=work_dir,
            polarizations=selected_polarizations,
            coreg_mode=coreg_mode,
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
        aux_cal_paths: Mapping[str, object] | Sequence[str | Path] | None,
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
        coreg_mode = kwargs.get("coreg_mode", "geometry")
        dem = kwargs.pop("dem", None)
        local_paths = materialize_cdse_roi_safes(
            roi,
            datetime_range=datetime_range,
            polarizations=selected_polarizations,
            work_dir=root,
            granularity=granularity,
            coreg_mode=coreg_mode,
            dem=dem,
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
        if aux_cal_paths is not None:
            kwargs["aux_cal_paths"] = aux_cal_paths
        kwargs.setdefault("roi", roi)
        kwargs["extra"] = extra
        return cls.from_safes(
            local_paths,
            work_dir=root,
            polarizations=selected_polarizations,
            **kwargs,
        )

    @classmethod
    def _from_asf_complete(
        cls,
        roi: Points | BoundingBox | Polygons,
        *,
        work_dir: str | Path,
        missions: str | Sequence[str] | None = None,
        polarizations: S1PolarizationRequest | Sequence[str] | str = ("VV",),
        collections: Sequence[str] | None = None,
        datetime_range: tuple[Any, Any] | None = None,
        flight_direction: Literal["ASCENDING", "DESCENDING"] | None = None,
        track: int | Sequence[int] | None = None,
        burst_id: int | Sequence[int] | None = None,
        auth_profile: str | None = "earthdata-asf",
        limit: int = 10_000,
        budget: RemoteResourceBudget | None = None,
        orbit_paths: Mapping[str, object] | Sequence[str | Path] | None = None,
        aux_cal_paths: Mapping[str, object] | Sequence[str | Path] | None = None,
        items: Sequence[CatalogItem] | None = None,
        **filters: Any,
    ) -> Self:
        """Search ASF, download complete SAFE inputs, and prepare a Stack.

        Absorbed from the deleted ``from_search`` constructor
        (PROPOSAL-0061).  Burst products poll until ready.  When ``items`` is
        given the internal :meth:`search` call is skipped and those items are
        downloaded directly — the ``from_search`` path.
        """
        selected_polarizations = normalize_polarizations(polarizations)
        ensure_asf_catalog("sentinel1-asf", product_type="burst")
        from faninsar import remote
        from faninsar.remote.errors import _fail
        from faninsar.remote.protocols import _CallLedger

        budget = budget or remote.RemoteResourceBudget()
        if items is None:
            items = list(
                cls.search(
                    roi,
                    missions=missions,
                    polarizations=selected_polarizations,
                    collections=collections,
                    datetime_range=datetime_range,
                    flight_direction=flight_direction,
                    track=track,
                    burst_id=burst_id,
                    auth_profile=auth_profile,
                    limit=limit,
                    budget=budget,
                    **filters,
                )
            )
        selected_items = canonical_items(items)
        if not selected_items:
            message = "Sentinel-1 search returned no products"
            reject(message, ValueError, logger=logger)
        root = Path(work_dir)
        poll_scheduler = None
        poll_tokens: list[Any | None] = [None] * len(selected_items)
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
        operation_ledger = _CallLedger(budget)
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
                                _fail(remote.RemoteLimitError, "max_poll_attempts")
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
                            _fail(remote.RemoteLimitError, "max_poll_attempts")
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
                reject(message, RuntimeError, logger=logger)
            local_paths.append(path)
        if not local_paths:
            message = (
                "every Sentinel-1 input failed download verification; delete "
                "the recorded files and re-download them before preparing"
            )
            reject(message, ValueError, logger=logger)
        extra = dict(filters.pop("extra", {}) or {})
        if excluded_acquisitions:
            extra["excluded_acquisitions"] = tuple(excluded_acquisitions)
        extra.update(
            {
                "search_product_type": "burst",
                "search_polarizations": selected_polarizations,
                "search_items": tuple(identities),
            }
        )
        if orbit_paths is not None:
            filters["orbit_paths"] = orbit_paths
        if aux_cal_paths is not None:
            filters["aux_cal_paths"] = aux_cal_paths
        filters.setdefault("roi", roi)
        filters["extra"] = extra
        return cls.from_safes(
            local_paths,
            work_dir=root,
            polarizations=selected_polarizations,
            **filters,
        )

    @classmethod
    def _default_scene_provider(cls) -> Any:
        """Return the Sentinel-1 SAFE production provider."""
        return StackSceneProvider(name="Sentinel-1 SAFE")

    def _prepare_inputs(self, context: PreparationContext) -> PreparedInputs:
        from faninsar.missions.s1.preparation import prepare_s1_inputs

        return prepare_s1_inputs(self, context)

    def _bind_precise_orbits(self) -> None:
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
        if cache is not None:
            cache_dir = Path(cache)
        else:
            from faninsar.stack.layout import resolve_orbit_cache_dir

            cache_dir = resolve_orbit_cache_dir(self.config.work_dir)
        selected = ensure_covering_orbit_paths(
            intervals=intervals,
            mission_ids=mission_ids,
            cache_dir=cache_dir,
            existing=existing_paths,
            auto_fetch=auto_fetch,
            budget=getattr(self, "_preparation_budget", None),
        )
        if selected:
            self.config.extra["orbit_paths"] = selected
        elif existing_paths is None:
            self.config.extra.pop("orbit_paths", None)

    def _ensure_aux_cal(self) -> None:
        """Bind (and download if missing) AUX_CAL files for IPF 002.36 dates."""
        acquisitions = _eap_acquisitions(self)
        if not acquisitions:
            self.config.extra.pop("aux_cal_paths", None)
            return
        from faninsar.missions.s1.aux_cal_fetch import ensure_aux_cal_paths
        from faninsar.stack.layout import aux_cal_dir

        existing = self.config.extra.get("aux_cal_paths")
        self.config.extra["aux_cal_paths"] = ensure_aux_cal_paths(
            acquisitions=acquisitions,
            cache_dir=aux_cal_dir(self.config.work_dir),
            existing=existing if isinstance(existing, Mapping) else None,
            auto_fetch=bool(self.config.extra.get("aux_cal_auto_fetch", True)),
            budget=getattr(self, "_preparation_budget", None),
        )

    def _local_mission_inputs(self) -> MissionPreparedInputs:
        """Return SAFE-derived S1 footprints and local mission bindings."""
        from faninsar.missions.s1.safe import open_safe_product

        self._ensure_aux_cal()
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
            aux_cal_paths_by_acquisition=dict(
                self.config.extra.get("aux_cal_paths", {})
            ),
            provenance={
                "mission": "sentinel-1",
                "orbit_source": (
                    "external" if self.config.extra.get("orbit_paths") else "embedded"
                ),
                "eap_correction": ",".join(
                    sorted(self.config.extra.get("aux_cal_paths", {}))
                )
                or "none",
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

    # ------------------------------------------------------------------
    # Track-axis metadata provider (stage 5 sel() support)
    # ------------------------------------------------------------------

    def _track_metadata(self) -> dict[str, object] | None:
        """Resolve each catalog date's track metadata for sel().

        Reads the persisted ``science_extra.track_metadata`` first so a
        reopened Stack keeps the recorded track map without reopening
        SAFEs; otherwise resolves from staged path-SAFE layout and
        annotation parsing through
        :func:`~faninsar.missions.s1.track_map.resolve_track_acquisition`.
        The result is cached per session.
        """
        cached = getattr(self, "_track_metadata_cache", None)
        if cached is not None:
            return cached  # type: ignore[return-value]
        persisted = self.config.extra.get("track_metadata")
        persisted_dates = (
            persisted.get("dates") if isinstance(persisted, dict) else None
        )
        if isinstance(persisted_dates, dict) and persisted_dates:
            from faninsar.missions.s1.track_map import (
                decode_track_metadata_dates,
            )

            result = {"dates": decode_track_metadata_dates(persisted_dates)}
            self._track_metadata_cache = result
            return result
        from faninsar.missions.s1.track_map import (
            disambiguate_track_letters,
            resolve_track_acquisition,
        )

        polarizations = self.config.extra.get("polarizations")
        selected = (
            None
            if polarizations in (None, "all")
            else tuple(str(value) for value in polarizations)
        )
        infos: dict[str, Any] = {}
        for date_id in self.catalog.dates:
            infos[date_id] = resolve_track_acquisition(
                self.catalog.paths_for(date_id),
                polarizations=selected,
            )
        infos = disambiguate_track_letters(infos)
        result = {"dates": infos}
        self._track_metadata_cache = result
        # Persist the resolved map into the science ``extra`` so
        # ``stack.json`` keeps it for later reopens.
        try:
            self.config.extra["track_metadata"] = {
                "dates": {
                    date_id: {
                        "track_key": info.track_key,
                        "direction": info.direction,
                        "relative_orbit": info.relative_orbit,
                        "mission_id": info.mission_id,
                        "polarizations": tuple(info.polarizations),
                        "absolute_burst_ids": tuple(sorted(info.absolute_burst_ids)),
                    }
                    for date_id, info in infos.items()
                }
            }
        except Exception as error:  # pragma: no cover - defensive boundary
            logger.warning("track metadata could not be serialized: %s", error)
        return result

    def _date_footprints(self, date_id: str) -> tuple[object, ...]:
        """Return burst-footprint geometries of one catalog date."""
        from faninsar.missions.s1.track_map import footprints_for_date

        polarizations = self.config.extra.get("polarizations")
        selected = (
            None
            if polarizations in (None, "all")
            else tuple(str(value) for value in polarizations)
        )
        return footprints_for_date(
            self.catalog.paths_for(date_id), polarizations=selected
        )

    def _date_track_key(self, date_id: str) -> str | None:
        """Return the resolved ``track_key`` of one catalog date, if any."""
        infos = self._catalog_track_infos()
        info = infos.get(date_id)
        return None if info is None else getattr(info, "track_key", None)

    def _resolve_s1_roi_burst_selection(
        self,
        *,
        polygon: int | None = None,
    ) -> tuple[tuple[str, ...] | None, BurstSelection | None]:
        """Resolve swaths/bursts whose footprints intersect the ROI.

        No ROI keeps the configured selection (``None`` means all). An ROI
        selects bursts by footprint intersection, including neighboring
        bursts at edges whose footprints overlap the ROI. Geometry apply
        still crops to the ROI window. ``polygon`` resolves one part of a
        multi-polygon ROI instead of the whole union.
        """
        cached_map = getattr(self, "_s1_roi_burst_selection_cache", None)
        if cached_map is None:
            cached_map = {}
            self._s1_roi_burst_selection_cache = cached_map
        if polygon in cached_map:
            return cached_map[polygon]
        polygon_rois = self._roi_polygon_rois()
        if polygon is not None and polygon_rois is not None:
            roi = polygon_rois[polygon]
        else:
            roi = self._effective_roi_with_mask().roi
        if roi is None:
            result = (self.config.swaths, self.config.bursts)
            cached_map[polygon] = result
            return result
        from faninsar.missions.s1.processing import (
            _select_bursts_by_roi,
            _union_swath_names,
        )
        from faninsar.missions.s1.safe import open_acquisition_products

        logger.info("ROI provided; explicit swaths/bursts selection ignored")
        ref_paths = list(self.catalog.paths_for(self.reference))
        polarizations = self.config.extra.get("polarizations")
        selected_polarizations = (
            None if polarizations == "all" else tuple(polarizations or ())
        )
        products = open_acquisition_products(
            ref_paths, polarizations=selected_polarizations
        )
        swath_tuple = _union_swath_names(products)
        ref_orbit = self._orbit_path_for(self.reference)
        orbits = [ref_orbit] * len(products)
        resolved = _select_bursts_by_roi(
            roi,
            ref_paths,
            swath_tuple,
            orbits=orbits,
            dem=self._geometry_dem(),
        )
        bursts = _burst_selection_from_resolved(resolved, len(products), swath_tuple)
        bursts, swath_tuple = _drop_empty_swaths(bursts, swath_tuple)
        if not swath_tuple:
            reject_pair_configuration("ROI selection contains no bursts")
        result = (swath_tuple, bursts)
        cached_map[polygon] = result
        return result

    def _resolve_s1_nesd_selection(
        self,
    ) -> tuple[tuple[str, ...] | None, BurstSelection | None]:
        """Resolve swaths/bursts for NESD measure.

        Intersecting swaths keep **every listed parent burst** so NESD can
        measure adjacent overlap strips, including slots that stay
        ``firstValidSample = -1`` outside the fetched overlap. Apply still
        uses :meth:`_resolve_s1_roi_burst_selection` and crops to the ROI.
        A swath with fewer than two listed bursts has no adjacent pair, so
        measure warns and skips NESD.
        """
        cached = getattr(self, "_s1_nesd_selection", None)
        if cached is not None:
            return cached
        roi = self._effective_roi_with_mask().roi
        swaths, bursts = self._resolve_s1_roi_burst_selection()
        if roi is None:
            self._s1_nesd_selection = (swaths, bursts)
            return swaths, bursts
        from faninsar.missions.s1.processing import _union_swath_names
        from faninsar.missions.s1.safe import open_acquisition_products

        ref_paths = list(self.catalog.paths_for(self.reference))
        polarizations = self.config.extra.get("polarizations")
        selected_polarizations = (
            None if polarizations == "all" else tuple(polarizations or ())
        )
        products = open_acquisition_products(
            ref_paths, polarizations=selected_polarizations
        )
        swath_tuple = swaths or _union_swath_names(products)
        resolved: dict[tuple[int, str], list[int]] = {}
        for frame_index, product in enumerate(products):
            for swath_name in swath_tuple:
                swath = product.find_swath(swath_name)
                if swath is None:
                    resolved[(frame_index, swath_name)] = []
                    continue
                resolved[(frame_index, swath_name)] = [
                    int(burst.index) for burst in swath.bursts
                ]
        expanded = _burst_selection_from_resolved(resolved, len(products), swath_tuple)
        expanded, swath_tuple = _drop_empty_swaths(expanded, swath_tuple)
        if not swath_tuple:
            reject_pair_configuration("ROI selection contains no bursts")
        result = (swath_tuple, expanded)
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
            primary_aux_cal_path=self._aux_cal_path_for(primary),
            secondary_aux_cal_path=self._aux_cal_path_for(secondary),
            device=self.config.device,
        )

    def _geo_lut_cache_dir(self) -> Path:
        """Return the shared content-addressed geo2rdr LUT cache."""
        configured = self.config.extra.get("geo_lut_cache_dir")
        if configured is not None:
            return Path(configured)
        return self.config.work_dir / "cache" / "geo-lut"

    def _source_processing_identity(self, date_id: str) -> Mapping[str, object]:
        """Bind the S1 scientific-validity revision into scene reuse identity."""
        from faninsar.missions.s1.coregister import S1_SLC_VALIDITY_REVISION

        identity = dict(super()._source_processing_identity(date_id))
        identity["slc_validity_revision"] = S1_SLC_VALIDITY_REVISION
        return identity

    def _prepare_reference_scene(
        self, *, overwrite: bool, polygon: int | None = None
    ) -> Any:
        """Compute (or reuse) the Reference scene product once per run.

        A ``polygon`` that maps to a different track than the Reference's
        produces no scene units — it is skipped with an explanatory log
        rather than coregistered against another track's Reference.
        """
        from faninsar.missions.s1.coregister import prepare_s1_reference

        if polygon is not None:
            reference_track = self._date_track_key(self.reference)
            polygon_track = self.polygon_tracks.get(polygon)
            if (
                reference_track is not None
                and polygon_track is not None
                and polygon_track != reference_track
            ):
                logger.info(
                    "ROI polygon %d lies on track %s; the Reference is on %s — "
                    "no scene units in this subtree, skipped",
                    polygon,
                    polygon_track,
                    reference_track,
                )
                # ``InvalidProcessingStateError`` carrying the canonical
                # "no scene units" phrase is what ``coregister_scenes``
                # catches to skip the off-track polygon subtree.
                reject_invalid_state(
                    f"Reference ROI selection produced no scene units "
                    f"(polygon {polygon} lies on track {polygon_track!r}, "
                    f"Reference on {reference_track!r})"
                )
        polygon_rois = self._roi_polygon_rois()
        if polygon is not None and polygon_rois is not None:
            roi = polygon_rois[polygon]
        else:
            roi = self._effective_roi_with_mask().roi
        swaths, bursts = self._resolve_s1_roi_burst_selection(polygon=polygon)
        return prepare_s1_reference(
            _source_input(self.catalog.paths_for(self.reference)),
            output_scene=self._scene_dir(self.reference, polygon),
            swaths=swaths,
            bursts=bursts,
            roi=roi,
            dem=self._geometry_dem(),
            coregistration_grid=self.config.coregistration_grid,
            geo_grid=self.config.geo_grid,
            geo_lut_cache_dir=self._geo_lut_cache_dir(),
            orbit_path=self._orbit_path_for(self.reference),
            aux_cal_path=self._aux_cal_path_for(self.reference),
            polarizations=self.config.extra.get("polarizations"),
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
        polygon: int | None = None,
    ) -> Any:
        """Resample one secondary onto the once-prepared Reference context.

        The single-track constraint is enforced before any pixel work: a
        secondary whose resolved track differs from the Reference's fails
        closed with a message naming both tracks — Sentinel-1 pairs never
        span two orbit groups.
        """
        from faninsar.missions.s1.coregister import coregister_s1_secondary

        reference_track = self._date_track_key(self.reference)
        secondary_track = self._date_track_key(date_id)
        if (
            reference_track is not None
            and secondary_track is not None
            and secondary_track != reference_track
        ):
            reject_invalid_state(
                f"cannot coregister {date_id} onto Reference {self.reference}: "
                f"{date_id} lies on track {secondary_track!r} but the "
                f"Reference is on {reference_track!r}; different tracks "
                "cannot form a pair"
            )
        if reference_context is None:
            reference_context = self._prepare_reference_scene(
                overwrite=False, polygon=polygon
            )
        coregister_s1_secondary(
            reference_context,
            _source_input(self.catalog.paths_for(date_id)),
            output_scene=output_dir,
            misreg_az_px=misreg_az_px,
            misreg_rg_px=misreg_rg_px,
            secondary_orbit_path=self._orbit_path_for(date_id),
            secondary_aux_cal_path=self._aux_cal_path_for(date_id),
            polarizations=self.config.extra.get("polarizations"),
            device=self.config.device,
            inputs=inputs,
            properties=properties,
            overwrite=overwrite,
        )
        return None


def _cap_per_mission(
    bindings: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Keep at most ``limit`` products per satellite among CDSE bindings.

    CDSE re-discovery on one track returns every satellite flying that
    orbit (S1A/C/D interleaved ~6 days apart).  Each binding is one
    ``(product, swath, polarization)`` measurement, so products — not
    bindings — are the unit being limited: the ``limit`` most recent
    ``product_id`` groups per mission keep all of their swath bindings.
    """
    import re

    def _product_key(binding: Mapping[str, Any]) -> tuple[str, str, str]:
        product_id = str(binding.get("product_id", ""))
        mission = str(binding.get("mission_id", ""))
        match = re.search(r"_(\d{8})T", product_id)
        return mission, (match.group(1) if match else ""), product_id

    # mission -> {product_id: latest-date}
    by_mission: dict[str, dict[str, str]] = {}
    for binding in bindings:
        mission, date, product_id = _product_key(binding)
        by_mission.setdefault(mission, {})[product_id] = max(
            date, by_mission.setdefault(mission, {}).get(product_id, "")
        )
    keep: set[str] = set()
    for products in by_mission.values():
        ranked = sorted(products.items(), key=lambda kv: kv[1], reverse=True)
        keep.update(product_id for product_id, _ in ranked[:limit])
    return [b for b in bindings if str(b.get("product_id", "")) in keep]


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
