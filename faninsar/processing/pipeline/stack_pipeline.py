"""Multi-scene stack pipeline — thin wrapper around :class:`Stack` (PROPOSAL-0017)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.stack.catalog import scene_id_from_path
from faninsar.processing.timeseries.inversion import (
    TimeSeriesResult,
    invert_unwrapped_pairs,
    write_timeseries_zarr,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.pipeline.production import ProductionPairState

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class StackPipelineResult:
    """Outputs of a multi-scene pair stack run."""

    scene_ids: tuple[str, ...]
    pair_results: tuple[ProductionPairState, ...]
    timeseries: TimeSeriesResult | None
    timeseries_zarr: Path | None


def default_pair_list(scene_ids: Sequence[str]) -> list[tuple[str, str]]:
    """Build chronological unique pair list from sorted scene ids."""
    ordered = tuple(sorted(scene_ids))
    if len(ordered) < 2:
        reject_invalid_state("at least two scenes are required to form pairs")
    return [(a, b) for a, b in combinations(ordered, 2)]


def run_stack_pipeline(
    scene_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    pairs: Sequence[tuple[str, str]] | None = None,
    swath: str = "IW1",
    burst_index: int = 0,
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
) -> StackPipelineResult:
    """Process an arbitrary SAFE stack via :class:`~faninsar.processing.stack.Stack`.

    Parameters
    ----------
    scene_paths : iterable of path
        SAFE ZIP/directory paths (N >= 2).
    output_dir : path
        Output root for pairs/ and timeseries.zarr.
    pairs : sequence of (ref_id, sec_id), optional
        Optional explicit pairs using scene ids. Default: all combinations
        for legacy behavior of this helper (Stack defaults use short baseline
        when constructed via :meth:`Stack.from_safes` without pairs).
    swath, burst_index : optional
        Common burst selection for every scene.
    height, width : optional
        Deprecated window size (ignored; production uses full burst).
    multilook, goldstein_alpha : optional
        Interferogram parameters.
    invert_timeseries : bool, optional
        Run SBAS after pairs complete (requires unwrapped phases; currently
        only records None unless pair states carry unwrapped products).
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

    Returns
    -------
    StackPipelineResult
        Per-pair production states and optional timeseries product.

    """
    if height != 256 or width != 256:
        warnings.warn(
            "run_stack_pipeline height/width are deprecated no-ops; "
            "production uses full burst",
            DeprecationWarning,
            stacklevel=2,
        )
    paths = [Path(p) for p in scene_paths]
    if len(paths) < 2:
        reject_invalid_state("stack pipeline requires at least two SAFE scenes")

    from faninsar.core.pairs import Pairs
    from faninsar.processing.stack.session import Stack

    stack = Stack.from_safes(
        paths,
        work_dir=output_dir,
        dem=dem,
        multilook=multilook,
        goldstein_alpha=goldstein_alpha,
        executor=executor,
        device=device,
        invert_device=invert_device,
        coreg_mode=coreg_mode,  # type: ignore[arg-type]
        swaths=(swath,),
        bursts={swath: [burst_index]},
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

    pair_states = tuple(stack.pair_states.values())
    timeseries = None
    timeseries_zarr = None
    if invert_timeseries:
        pair_phases: dict[str, object] = {}
        for pid, state in stack.pair_states.items():
            if state.unwrapped_phase is not None:
                pair_phases[pid] = state.unwrapped_phase
        if pair_phases:
            timeseries = invert_unwrapped_pairs(
                pair_phases,  # type: ignore[arg-type]
                device=invert_device,
            )
            timeseries_zarr = write_timeseries_zarr(
                timeseries,
                Path(output_dir) / "timeseries.zarr",
            )

    logger.info(
        "Stack complete: %s scenes, %s pair states under %s mode=%s",
        len(stack.catalog),
        len(pair_states),
        output_dir,
        stack.config.coreg_mode,
    )
    return StackPipelineResult(
        scene_ids=stack.catalog.dates,
        pair_results=pair_states,
        timeseries=timeseries,
        timeseries_zarr=timeseries_zarr,
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
