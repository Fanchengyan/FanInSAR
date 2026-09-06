"""Stack session configuration (PROPOSAL-0017, PROPOSAL-0040)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.runtime.compute.numpy_backend import NumpyBackend
from faninsar.processing.runtime.resources import ResourceBudget
from faninsar.stack.mask_plan import MaskPlan

if TYPE_CHECKING:
    from faninsar.processing.geometry import DEM, GridSpec
    from faninsar.processing.geometry.prepared import (
        ActivationToken,
        StackActivationBinding,
    )
    from faninsar.processing.mosaicking.grid import GeoGridSpec
    from faninsar.processing.runtime.device import GpuMemoryReclaim
    from faninsar.processing.runtime.protocols import ComputeBackend
    from faninsar.processing.stages import (
        BurstSelection,
        CoregistrationGrid,
    )

CoregMode = Literal["geometry", "pair", "network"]
EsdMethod = Literal["auto", "splitband", "overlap"]
OnNetworkFailure = Literal["error"]
FlattenStage = Literal["coregistration", "interferogram"]
ActivationMode = Literal["reference", "qualified"]
logger = setup_logger(__name__)


@dataclass
class StackConfig:
    """Session-level defaults for :class:`~faninsar.stack.session.Stack`.

    Step methods may override individual fields for a single call; overrides
    do not mutate this config unless the step is written to do so.

    ``mask_plan`` is the only mask configuration surface.  An empty plan is
    unmasked processing; every non-empty stage is explicit and normalized
    before this object is constructed. ``grid`` is either an explicit shared
    :class:`~faninsar.processing.geometry.GridSpec` or ``"auto"``. In automatic
    mode, ``roi`` has precedence over selected acquisition/swath/burst
    footprints; the footprint centre selects UTM or UPS. Seam and large-range
    cases warn and continue, while an explicit seam-crossing ROI fails before
    planning. ``resolution_m`` supplies automatic projected pixel spacing;
    pixel edges are expanded to whole pixels around the projected ROI.
    ``dem_cache_dir`` supplies the source cache at materialization. An
    explicit ``GridSpec`` always wins over automatic ROI selection.
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
    dem: DEM | None = None
    geo_grid: GeoGridSpec | None = None
    swaths: tuple[str, ...] = ("IW1",)
    bursts: BurstSelection | None = None
    roi: object | None = None
    # An empty plan is the deliberate default; water is opt-in through
    # ``MaskPlan.water()`` or an explicit registry definition.
    mask_plan: MaskPlan = field(default_factory=MaskPlan)
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
    grid: GridSpec | Literal["auto"] = "auto"
    resolution_m: float = 30.0
    dem_cache_dir: Path | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize path and multilook types, and validate the mask plan."""
        self.work_dir = Path(self.work_dir)
        valid_resolution = False
        if isinstance(self.resolution_m, (int, float)):
            valid_resolution = bool(
                np.isfinite(self.resolution_m) and self.resolution_m > 0
            )
        if not valid_resolution:
            message = "resolution_m must be a finite positive number"
            logger.error(message)
            raise ValueError(message)
        if self.grid != "auto":
            from faninsar.processing.geometry import GridSpec

            if not isinstance(self.grid, GridSpec):
                message = "grid must be a GridSpec or 'auto'"
                logger.error(message)
                raise TypeError(message)
        if self.dem_cache_dir is not None:
            self.dem_cache_dir = Path(self.dem_cache_dir)
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
        if not isinstance(self.mask_plan, MaskPlan):
            message = "mask_plan must be a MaskPlan"
            logger.error(message)
            raise TypeError(message)
        ml = self.multilook
        self.multilook = (int(ml[0]), int(ml[1]))


def _load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load and validate a Stack configuration mapping.

    Parameters
    ----------
    config : str, pathlib.Path, or dict
        JSON/YAML path or an in-memory mapping.

    Returns
    -------
    dict
        A shallow copy of the configuration mapping.

    Raises
    ------
    TypeError
        If an in-memory or decoded document is not a mapping.
    ValueError
        If a legacy mask option is present.

    """
    if isinstance(config, dict):
        data = dict(config)
        _reject_legacy_mask_config(data)
        return data
    path = Path(config)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            message = "PyYAML required for YAML configs"
            logger.exception(message)
            raise ImportError(message) from exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            message = "YAML config must be a mapping"
            logger.error(message)
            raise ValueError(message)
        _reject_legacy_mask_config(data)
        return data
    data = json.loads(text)
    if not isinstance(data, dict):
        message = "JSON config must be a mapping"
        logger.error(message)
        raise TypeError(message)
    _reject_legacy_mask_config(data)
    return data


def _load_mask_plan(
    config: str | Path | dict[str, Any], cfg: dict[str, Any]
) -> MaskPlan:
    """Normalize the explicit ``mask_plan`` section of a Stack config."""
    if "mask_plan" not in cfg:
        return MaskPlan()
    value = cfg["mask_plan"]
    if not isinstance(value, dict):
        message = "'mask_plan' must be a mapping"
        logger.error(message)
        raise TypeError(message)
    base_dir = Path(config).resolve().parent if not isinstance(config, dict) else None
    return MaskPlan.from_mapping(value, base_dir=base_dir)


def _reject_legacy_mask_config(cfg: dict[str, Any]) -> None:
    """Reject removed mask spellings at the Stack configuration boundary."""
    legacy = {
        "mask",
        "mask_source",
        "mask_on_failure",
        "mask_apply_ionosphere",
        "water_mask",
        "auto_water_mask",
    } & cfg.keys()
    if legacy:
        message = (
            "legacy mask configuration is not supported; use explicit "
            "'mask_plan' definitions and stage references: " + ", ".join(sorted(legacy))
        )
        logger.error(message)
        raise ValueError(message)


def _resolve_backend(backend: str | ComputeBackend) -> ComputeBackend:
    """Resolve a configured compute backend without implicit fallback."""
    if isinstance(backend, str):
        if backend == "numpy":
            return NumpyBackend()
        if backend in {"dask_torch", "dask"}:
            from faninsar.processing.runtime.compute.dask_torch import DaskTorchBackend

            return DaskTorchBackend()
        message = f"unknown backend {backend!r}"
        logger.error(message)
        raise ValueError(message)
    return backend


__all__ = [
    "ActivationMode",
    "CoregMode",
    "EsdMethod",
    "FlattenStage",
    "OnNetworkFailure",
    "StackConfig",
]
