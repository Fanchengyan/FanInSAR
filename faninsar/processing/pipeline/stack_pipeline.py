"""General multi-scene stack pipeline via production pair stages."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.pipeline.production import (
    ProductionPairState,
    run_pair,
)
from faninsar.processing.timeseries.inversion import (
    TimeSeriesResult,
    invert_unwrapped_pairs,
    write_timeseries_zarr,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from faninsar.processing.geometry.dem import DEMSampler

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class StackPipelineResult:
    """Outputs of a multi-scene pair stack run."""

    scene_ids: tuple[str, ...]
    pair_results: tuple[ProductionPairState, ...]
    timeseries: TimeSeriesResult | None
    timeseries_zarr: Path | None


def scene_id_from_path(path: str | Path) -> str:
    """Extract a YYYYMMDD-style scene id from a SAFE path when possible."""
    stem = Path(path).stem.replace(".SAFE", "")
    for part in stem.split("_"):
        if len(part) >= 8 and part[:8].isdigit():
            return part[:8]
    return stem


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
) -> StackPipelineResult:
    """Process an arbitrary SAFE stack with the production pair pipeline.

    Parameters
    ----------
    scene_paths : iterable of path
        SAFE ZIP/directory paths (N >= 2).
    output_dir : path
        Output root for pairs/ and timeseries.zarr.
    pairs : sequence of (ref_id, sec_id), optional
        Optional explicit pairs using scene ids. Default: all combinations.
    swath, burst_index : optional
        Common burst selection for every scene.
    height, width : optional
        Deprecated window size (ignored; production uses full burst).
    multilook, goldstein_alpha : optional
        Interferogram parameters.
    invert_timeseries : bool, optional
        Run SBAS after pairs complete.
    executor : {"torch"}, optional
        Unified Torch coregistration and LUT Lanczos path.
    device : {"auto","cpu","cuda"}, optional
        Torch compute device.
    invert_device : str, optional
        Device for SBAS inversion. Default ``"cpu"``.
    dem : DEMSampler, optional
        DEM for coreg/flatten/geocode.

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

    id_to_path = {scene_id_from_path(p): p for p in paths}
    if len(id_to_path) != len(paths):
        reject_invalid_state("duplicate scene ids in stack input")
    scene_ids = tuple(sorted(id_to_path))
    pair_list = list(pairs) if pairs is not None else default_pair_list(scene_ids)

    out = Path(output_dir)
    pair_states: list[ProductionPairState] = []
    pair_phases: dict[str, object] = {}
    for ref_id, sec_id in pair_list:
        if ref_id not in id_to_path or sec_id not in id_to_path:
            reject_invalid_state(f"unknown scene id in pair ({ref_id}, {sec_id})")
        state = run_pair(
            id_to_path[ref_id],
            id_to_path[sec_id],
            output_dir=out / "pairs" / f"{ref_id}_{sec_id}",
            swaths=(swath,),
            bursts={swath: [burst_index]},
            dem=dem,
            multilook=multilook,
            goldstein_alpha=goldstein_alpha,
            executor=executor,
            device=device,
            unwrap=True,
        )
        pair_states.append(state)
        assert state.unwrapped_phase is not None
        pair_phases[state.pair_id] = state.unwrapped_phase

    timeseries = None
    timeseries_zarr = None
    if invert_timeseries:
        timeseries = invert_unwrapped_pairs(pair_phases, device=invert_device)
        timeseries_zarr = write_timeseries_zarr(timeseries, out / "timeseries.zarr")

    logger.info(
        "Stack complete: %s scenes, %s pairs under %s mode=%s",
        len(scene_ids),
        len(pair_states),
        out,
        "radar",
    )
    return StackPipelineResult(
        scene_ids=scene_ids,
        pair_results=tuple(pair_states),
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
        scene = stage_read_scene(
            path,
            swath=swath,
            burst_index=burst_index,
            height=height,
            width=width,
        )
        out[scene.scene_id] = scene
    return out
