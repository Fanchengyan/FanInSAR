"""Stack session configuration (PROPOSAL-0017)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.pipeline.production import (
        BurstSelection,
        CoregistrationGrid,
    )

CoregMode = Literal["geometry", "pair", "network"]
EsdMethod = Literal["auto", "splitband", "overlap"]
OnNetworkFailure = Literal["error", "degrade_to_pair"]


@dataclass
class StackConfig:
    """Session-level defaults for :class:`~faninsar.processing.stack.session.Stack`.

    Step methods may override individual fields for a single call; overrides
    do not mutate this config unless the step is written to do so.
    """

    work_dir: Path
    coreg_mode: CoregMode = "pair"
    coregistration_grid: CoregistrationGrid = "radar"
    esd_method: EsdMethod = "auto"
    multilook: tuple[int, int] = (2, 10)
    goldstein_alpha: float = 0.5
    executor: str = "torch"
    device: str = "auto"
    invert_device: str = "cpu"
    dem: DEMSampler | None = None
    swaths: tuple[str, ...] = ("IW1",)
    bursts: BurstSelection | None = None
    roi: object | None = None
    on_network_failure: OnNetworkFailure = "error"
    control_spacing: int | None = None
    n_jobs: int = 1
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize path and multilook types."""
        self.work_dir = Path(self.work_dir)
        ml = self.multilook
        self.multilook = (int(ml[0]), int(ml[1]))
