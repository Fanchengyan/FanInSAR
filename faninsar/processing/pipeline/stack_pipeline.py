"""Multi-scene stack pipeline — thin wrapper around :class:`Stack` (PROPOSAL-0017)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.stack.catalog import scene_id_from_path
from faninsar.processing.timeseries.inversion import (
    TimeSeriesResult,
    write_timeseries_zarr,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from faninsar.processing.contracts.prepared_geometry import (
        ActivationToken,
        StackActivationBinding,
    )
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.merge.grid import GeoGridSpec
    from faninsar.processing.pipeline.production import (
        BurstSelection,
        CoregistrationGrid,
    )
    from faninsar.processing.stack.config import ActivationMode
    from faninsar.processing.stack.stack_generation import StackResultGeneration
    from faninsar.query import BoundingBox, Polygons

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class StackInterferogramResult:
    """Immutable public reference to one persisted Stack interferogram.

    Attributes
    ----------
    pair_id : str
        Canonical ``primary_secondary`` identifier.
    artifact_root : pathlib.Path
        Reopenable manifest-bound IFG artifact root.
    manifest_digest : str
        SHA-256 identity of the committed IFG manifest.
    shape : tuple[int, int]
        Common-grid product shape after multilooking.
    multilook : tuple[int, int]
        Azimuth and range look factors.
    domain : {"radar", "geo"}
        Artifact coordinate domain.

    """

    pair_id: str
    artifact_root: Path
    manifest_digest: str
    shape: tuple[int, int]
    multilook: tuple[int, int]
    domain: Literal["radar", "geo"]


@dataclass(frozen=True, slots=True)
class StackPipelineResult:
    """Public persisted outputs of a multi-scene Stack run."""

    scene_ids: tuple[str, ...]
    pair_results: tuple[StackInterferogramResult, ...]
    timeseries: TimeSeriesResult | None
    timeseries_zarr: Path | None
    stack_generation: StackResultGeneration | None


def default_pair_list(scene_ids: Sequence[str]) -> list[tuple[str, str]]:
    """Build chronological unique pair list from sorted scene ids."""
    ordered = tuple(sorted(scene_ids))
    if len(ordered) < 2:
        reject_invalid_state("at least two scenes are required to form pairs")
    return [(a, b) for a, b in combinations(ordered, 2)]


def execute_stack_pipeline(
    scene_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    roi: BoundingBox | Polygons | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    swath: str = "IW1",
    burst_index: int = 0,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    coregistration_grid: CoregistrationGrid = "radar",
    geo_grid: GeoGridSpec | None = None,
    height: int = 256,
    width: int = 256,
    multilook: tuple[int, int] = (2, 8),
    goldstein_alpha: float = 0.5,
    invert_timeseries: bool = True,
    executor: str = "torch",
    device: str = "auto",
    invert_device: str = "cpu",
    dem: DEMSampler | None = None,
    coreg_mode: str = "pair",
    activation_mode: ActivationMode,
    activation_binding: StackActivationBinding | None = None,
    activation_token: ActivationToken | None = None,
    activation_authority_root: str | Path | None = None,
    record_scientific_lineage: bool = False,
) -> StackPipelineResult:
    """Process an arbitrary SAFE stack via :class:`~faninsar.processing.stack.Stack`.

    Parameters
    ----------
    scene_paths : iterable of path
        SAFE ZIP/directory paths (N >= 2).
    output_dir : path
        Output root for pairs/ and timeseries.zarr.
    roi : BoundingBox or Polygons, optional
        Geographic region used to select and crop burst processing.
    pairs : sequence of (ref_id, sec_id), optional
        Optional explicit pairs using scene ids. Default: all combinations
        for legacy behavior of this helper (Stack defaults use short baseline
        when constructed via :meth:`Stack.from_safes` without pairs).
    swath, burst_index : optional
        Backwards-compatible single-burst selection.
    swaths, bursts : optional
        Explicit multi-swath and multi-burst selection. When supplied, these
        override ``swath`` and ``burst_index``.
    coregistration_grid : {"radar", "geo"}, optional
        Common Stack artifact coordinate domain.
    geo_grid : GeoGridSpec, optional
        Required projected grid when ``coregistration_grid="geo"``.
    height, width : optional
        Deprecated window size (ignored; production uses full burst).
    multilook, goldstein_alpha : optional
        Interferogram parameters.
    invert_timeseries : bool, optional
        Run spatial unwrap, temporal phase reconciliation, and SBAS after all
        common-grid pair artifacts are complete.
    executor : {"torch"}, optional
        Unified Torch coregistration and LUT Lanczos path.
    device : {"auto","cpu","cuda"}, optional
        Torch compute device.
    invert_device : str, optional
        Device for SBAS inversion. Default ``"cpu"``.
    dem : DEMSampler, optional
        DEM for coreg/flatten/geocode.
    coreg_mode : {"geometry", "pair", "network"}, optional
        Stack coregistration mode (PROPOSAL-0017).
    activation_mode : {"reference", "qualified"}
        Explicit namespace selection.  ``reference`` is correctness-only;
        ``qualified`` requires a signed activation token, binding, and local
        authority root.
    activation_binding, activation_token, activation_authority_root : optional
        Required together for qualified production execution.
    record_scientific_lineage : bool, optional
        Persist ordered residual/carrier/phase operation records with payload
        hashes during scene production. Disabled by default to avoid hashing
        overhead in ordinary runs.

    Returns
    -------
    StackPipelineResult
        Per-pair production states and optional timeseries product.

    """
    if height != 256 or width != 256:
        warnings.warn(
            "height/width are deprecated no-ops; "
            "production uses full burst",
            DeprecationWarning,
            stacklevel=2,
        )
    paths = [Path(p) for p in scene_paths]
    if len(paths) < 2:
        reject_invalid_state("stack pipeline requires at least two SAFE scenes")

    from faninsar.core.pairs import Pairs
    from faninsar.processing.stack.session import Stack

    resolved_swaths = swaths or (swath,)
    resolved_bursts = bursts or {name: [burst_index] for name in resolved_swaths}
    if coregistration_grid == "geo" and geo_grid is None:
        reject_invalid_state("Geo Stack pipeline requires geo_grid")

    stack = Stack.from_safes(
        paths,
        work_dir=output_dir,
        dem=dem,
        roi=roi,
        multilook=multilook,
        goldstein_alpha=goldstein_alpha,
        executor=executor,
        device=device,
        invert_device=invert_device,
        coreg_mode=coreg_mode,  # type: ignore[arg-type]
        coregistration_grid=coregistration_grid,
        geo_grid=geo_grid,
        activation_mode=activation_mode,
        activation_binding=activation_binding,
        activation_token=activation_token,
        activation_authority_root=activation_authority_root,
        swaths=resolved_swaths,
        bursts=resolved_bursts,
        # The persisted IFG/unwrap/SBAS path never consumes in-memory Pair
        # states. Keeping them would scale RSS with acquisition count.
        retain_pair_states=False,
        record_scientific_lineage=record_scientific_lineage,
    )
    if pairs is not None:
        names = [f"{a}_{b}" for a, b in pairs]
        stack.pairs = Pairs.from_names(names)
        stack.misreg_pairs = stack.pairs

    stack.prepare_scenes()
    if stack.config.coreg_mode == "network":
        stack.measure_misreg()
        stack.invert_misreg()
    stack.coregister_scenes()
    stack.form_interferograms()

    from faninsar.processing.stack.ifg_store import InterferogramArtifactStore

    pair_results = tuple(
        StackInterferogramResult(
            pair_id=f"{store.pair[0]}_{store.pair[1]}",
            artifact_root=store.root,
            manifest_digest=store.manifest_digest,
            shape=store.shape,
            multilook=store.looks,
            domain=store.domain,
        )
        for store in (InterferogramArtifactStore.open(path) for path in stack.ifg_dirs)
    )
    timeseries = None
    timeseries_zarr = None
    stack_generation = None
    if invert_timeseries:
        stack.unwrap()
        timeseries = stack.invert_timeseries(device=invert_device)
        timeseries_zarr = write_timeseries_zarr(
            timeseries,
            Path(output_dir) / "timeseries.zarr",
        )
        stack_generation = stack.publish_generation(timeseries_zarr)

    logger.info(
        "Stack complete: %s scenes, %s pair states under %s mode=%s",
        len(stack.catalog),
        len(pair_results),
        output_dir,
        stack.config.coreg_mode,
    )
    return StackPipelineResult(
        scene_ids=stack.catalog.dates,
        pair_results=pair_results,
        timeseries=timeseries,
        timeseries_zarr=timeseries_zarr,
        stack_generation=stack_generation,
    )


def load_safe_burst_windows(
    scene_paths: Iterable[str | Path],
    *,
    swath: str = "IW1",
    burst_index: int = 0,
    window: tuple[int, int] = (128, 128),
) -> dict[str, object]:
    """Load scene burst windows for inspection or custom workflows."""
    from faninsar.processing.pipeline.workflow import stage_read_scene

    height, width = window
    out: dict[str, object] = {}
    for path in scene_paths:
        sid = scene_id_from_path(path)
        out[sid] = stage_read_scene(
            path,
            swath=swath,
            burst_index=burst_index,
            height=height,
            width=width,
        )
    return out
