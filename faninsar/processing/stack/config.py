"""Stack session configuration (PROPOSAL-0017)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from faninsar.logging import setup_logger

if TYPE_CHECKING:
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

CoregMode = Literal["geometry", "pair", "network"]
EsdMethod = Literal["auto", "splitband", "overlap"]
OnNetworkFailure = Literal["error"]
ActivationMode = Literal["reference", "qualified"]

logger = setup_logger(__name__)


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
    geo_grid: GeoGridSpec | None = None
    swaths: tuple[str, ...] = ("IW1",)
    bursts: BurstSelection | None = None
    roi: object | None = None
    on_network_failure: OnNetworkFailure = "error"
    activation_mode: ActivationMode = "reference"
    activation_binding: StackActivationBinding | None = None
    activation_token: ActivationToken | None = None
    control_spacing: int | None = None
    n_jobs: int = 1
    retain_pair_states: bool = False
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize path and multilook types."""
        self.work_dir = Path(self.work_dir)
        if self.on_network_failure != "error":
            message = (
                "P19 qualified Stack mode is fail-closed; "
                "on_network_failure must be 'error'"
            )
            logger.error(message)
            raise ValueError(message)
        if self.activation_mode not in ("reference", "qualified"):
            message = f"unsupported Stack activation mode: {self.activation_mode!r}"
            logger.error(message)
            raise ValueError(message)
        if self.activation_mode == "qualified":
            if self.activation_binding is None:
                message = "qualified Stack mode requires an activation binding"
                logger.error(message)
                raise ValueError(message)
            if self.activation_token is None:
                message = "qualified Stack mode requires an activation token"
                logger.error(message)
                raise ValueError(message)
            if self.activation_binding.activation_mode != "qualified":
                message = "qualified Stack mode requires a qualified binding"
                logger.error(message)
                raise ValueError(message)
            if self.activation_token.mode != "qualified":
                message = "qualified Stack mode requires a qualified token"
                logger.error(message)
                raise ValueError(message)
        elif self.activation_binding is not None and (
            self.activation_binding.activation_mode != "reference"
        ):
            message = "reference Stack mode cannot use a qualified binding"
            logger.error(message)
            raise ValueError(message)
        elif self.activation_token is not None and (
            self.activation_token.mode != "reference"
        ):
            message = "reference Stack mode cannot use a qualified token"
            logger.error(message)
            raise ValueError(message)
        ml = self.multilook
        self.multilook = (int(ml[0]), int(ml[1]))
