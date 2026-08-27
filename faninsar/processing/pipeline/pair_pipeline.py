"""Compatibility wrapper around the explicit pair workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger
from faninsar.processing.pipeline.products import PairProductArrays
from faninsar.processing.pipeline.workflow import (
    PairWorkflowState,
    execute_pair_workflow,
)

if TYPE_CHECKING:
    from faninsar.processing.tops.deramp import TOPSCarrierModel

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class PairPipelineResult:
    """Outputs of a completed pair workflow."""

    product: PairProductArrays
    zarr_path: Path
    stac_path: Path
    range_shift_px: float
    azimuth_shift_px: float
    state: PairWorkflowState


def execute_pair_pipeline(
    primary: str | Path,
    secondary: str | Path,
    *,
    pair_id: str | None = None,
    output_dir: str | Path,
    carrier: TOPSCarrierModel | None = None,
    multilook: tuple[int, int] = (2, 8),
    goldstein_alpha: float = 0.5,
    metadata: dict[str, Any] | None = None,
    swath: str = "IW1",
    burst_index: int = 0,
    height: int = 256,
    width: int = 256,
) -> PairPipelineResult:
    """Run the full pair workflow from two SAFE paths."""
    _ = (carrier, metadata, pair_id)
    if not isinstance(primary, (str, Path)) or not isinstance(secondary, (str, Path)):
        message = (
            "the pair pipeline requires SAFE paths so deramp/coreg/geocode "
            "use annotation metadata."
        )
        logger.error(message)
        raise TypeError(message)

    state = execute_pair_workflow(
        primary,
        secondary,
        output_dir=output_dir,
        swath=swath,
        burst_index=burst_index,
        height=height,
        width=width,
        multilook=multilook,
        goldstein_alpha=goldstein_alpha,
    )
    if state.zarr_path is None or state.stac_path is None:
        message = "workflow did not write products"
        raise RuntimeError(message)
    if (
        state.complex_ifg is None
        or state.coherence is None
        or state.wrapped_phase is None
        or state.unwrapped_phase is None
        or state.connected_components is None
    ):
        message = "workflow missing product arrays"
        raise RuntimeError(message)

    product = PairProductArrays(
        pair_id=state.pair_id,
        complex_ifg=state.complex_ifg,
        coherence=state.coherence,
        wrapped_phase=state.wrapped_phase,
        unwrapped_phase=state.unwrapped_phase,
        connected_components=state.connected_components,
        metadata={"stages": list(state.log)},
    )
    return PairPipelineResult(
        product=product,
        zarr_path=state.zarr_path,
        stac_path=state.stac_path,
        range_shift_px=float(state.range_shift_px or 0.0),
        azimuth_shift_px=float(state.azimuth_shift_px or 0.0),
        state=state,
    )
