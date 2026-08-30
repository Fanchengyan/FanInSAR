"""Stack session configuration (PROPOSAL-0017, PROPOSAL-0039)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from faninsar.logging import setup_logger
from faninsar.processing.masking.mask import MaskSampler
from faninsar.processing.resources import ResourceBudget

if TYPE_CHECKING:
    from faninsar._core.device import GpuMemoryReclaim
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
FlattenStage = Literal["coregistration", "interferogram"]
ActivationMode = Literal["reference", "qualified"]
MaskFailurePolicy = Literal["error", "warning", "skip"]

#: Stack default: the automatic water mask resolved through the
#: :mod:`faninsar.processing.masking` manager (PROPOSAL-0039 G5).
AUTO_WATER_MASK = "water"
#: String spelling that explicitly disables masking (``mask: none`` in a
#: mapping config; normalized to ``None`` in :meth:`StackConfig.__post_init__`).
MASK_DISABLED = "none"

_MASK_FAILURE_POLICIES = frozenset({"error", "warning", "skip"})

logger = setup_logger(__name__)


@dataclass
class StackConfig:
    """Session-level defaults for :class:`~faninsar.processing.stack.session.Stack`.

    Step methods may override individual fields for a single call; overrides
    do not mutate this config unless the step is written to do so.

    .. note::
        The ``mask`` fields implement the PROPOSAL-0039 Stack surface: the
        default is the automatic water mask (resolved through the masking
        manager and subtracted from the ROI at burst-selection level), while
        ``mask=None`` explicitly disables masking and restores unmasked
        processing.
    """

    work_dir: Path
    activation_mode: ActivationMode
    coreg_mode: CoregMode = "pair"
    flatten_stage: FlattenStage = "coregistration"
    coregistration_grid: CoregistrationGrid = "radar"
    esd_method: EsdMethod = "auto"
    multilook: tuple[int, int] = (5, 2)
    goldstein_alpha: float = 0.5
    executor: str = "torch"
    device: str = "auto"
    invert_device: str = "cpu"
    dem: DEMSampler | None = None
    geo_grid: GeoGridSpec | None = None
    swaths: tuple[str, ...] = ("IW1",)
    bursts: BurstSelection | None = None
    roi: object | None = None
    # -- PROPOSAL-0039 mask surface -----------------------------------------
    # ``mask`` selects the removed region: ``"water"`` (the default) resolves
    # the automatic water mask through the masking manager, a MaskSampler
    # instance is a user-supplied mask, and ``None`` (or the ``"none"``
    # spelling accepted from mapping configs) explicitly disables masking and
    # restores unmasked processing.
    mask: str | MaskSampler | None = AUTO_WATER_MASK
    mask_source: str | None = None
    mask_resolution_m: float | None = None
    mask_buffer_km: float = 1.0
    ocean_water_buffer_km: float | None = None
    inland_water_buffer_km: float | None = None
    mask_on_failure: MaskFailurePolicy = "warning"
    mask_apply_ionosphere: bool = False
    on_network_failure: OnNetworkFailure = "error"
    activation_binding: StackActivationBinding | None = None
    activation_token: ActivationToken | None = None
    activation_authority_root: Path | None = None
    control_spacing: int | None = None
    n_jobs: int = 1
    retain_pair_states: bool = False
    record_scientific_lineage: bool = False
    gpu_memory_reclaim: GpuMemoryReclaim = "adaptive"
    resource_budget: ResourceBudget | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize path and multilook types, and validate mask fields."""
        self.work_dir = Path(self.work_dir)
        if self.resource_budget is not None and not isinstance(
            self.resource_budget, ResourceBudget
        ):
            message = "resource_budget must be a ResourceBudget or None"
            logger.error(message)
            raise ValueError(message)
        if self.flatten_stage not in {"coregistration", "interferogram"}:
            message = (
                "flatten_stage must be 'coregistration' or 'interferogram'; "
                f"got {self.flatten_stage!r}"
            )
            logger.error(message)
            raise ValueError(message)
        if self.gpu_memory_reclaim not in {"lazy", "eager", "adaptive"}:
            message = (
                "gpu_memory_reclaim must be 'lazy', 'eager', or 'adaptive'; "
                f"got {self.gpu_memory_reclaim!r}"
            )
            logger.error(message)
            raise ValueError(message)
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
            if self.activation_authority_root is None:
                message = "qualified Stack mode requires an activation authority root"
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
        if self.activation_authority_root is not None:
            self.activation_authority_root = Path(self.activation_authority_root)
        self._validate_mask_fields()
        ml = self.multilook
        self.multilook = (int(ml[0]), int(ml[1]))

    def _validate_mask_fields(self) -> None:
        """Normalize and validate the PROPOSAL-0039 mask fields (fail closed)."""
        if isinstance(self.mask, str):
            selection = self.mask.strip()
            if selection == MASK_DISABLED:
                self.mask = None
            elif selection != AUTO_WATER_MASK:
                message = (
                    "mask must be 'water' (automatic water mask), 'none' "
                    "(explicitly disabled), a MaskSampler instance, or None; "
                    f"got {self.mask!r}"
                )
                logger.error(message)
                raise ValueError(message)
        elif self.mask is not None and not isinstance(self.mask, MaskSampler):
            message = (
                "mask must be 'water', 'none', a MaskSampler instance, or "
                f"None; got {type(self.mask).__name__}"
            )
            logger.error(message)
            raise ValueError(message)
        if self.mask_on_failure not in _MASK_FAILURE_POLICIES:
            message = (
                "mask_on_failure must be 'error', 'warning', or 'skip'; "
                f"got {self.mask_on_failure!r}"
            )
            logger.error(message)
            raise ValueError(message)
        if not float(self.mask_buffer_km) >= 0.0:
            message = f"mask_buffer_km must be >= 0; got {self.mask_buffer_km!r}"
            logger.error(message)
            raise ValueError(message)
        self.mask_buffer_km = float(self.mask_buffer_km)
        if (self.ocean_water_buffer_km is None) != (
            self.inland_water_buffer_km is None
        ):
            message = (
                "ocean_water_buffer_km and inland_water_buffer_km must be "
                "configured together (dual buffers are both-or-neither)"
            )
            logger.error(message)
            raise ValueError(message)
        if self.ocean_water_buffer_km is not None:
            if not float(self.ocean_water_buffer_km) >= 0.0:
                message = (
                    "ocean_water_buffer_km must be >= 0; "
                    f"got {self.ocean_water_buffer_km!r}"
                )
                logger.error(message)
                raise ValueError(message)
            if not float(self.inland_water_buffer_km) >= 0.0:
                message = (
                    "inland_water_buffer_km must be >= 0; "
                    f"got {self.inland_water_buffer_km!r}"
                )
                logger.error(message)
                raise ValueError(message)
            self.ocean_water_buffer_km = float(self.ocean_water_buffer_km)
            self.inland_water_buffer_km = float(self.inland_water_buffer_km)
        if (
            self.mask_resolution_m is not None
            and not float(self.mask_resolution_m) > 0.0
        ):
            message = (
                "mask_resolution_m must be a positive number or None; "
                f"got {self.mask_resolution_m!r}"
            )
            logger.error(message)
            raise ValueError(message)
