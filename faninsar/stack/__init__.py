"""Stack session API (PROPOSAL-0017)."""

from __future__ import annotations

from faninsar.io.storage.ifg_store import (
    InterferogramArtifact,
    InterferogramArtifactStore,
    UnwrappedArtifact,
)
from faninsar.processing.runtime.device import GpuMemoryReclaim
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.config import (
    CoregMode,
    FlattenStage,
    StackConfig,
)
from faninsar.stack.grid import resolve_stack_grid, utm_ups_grid
from faninsar.stack.mask_plan import (
    MASK_KINDS,
    STAGES,
    MaskDefinition,
    MaskPlan,
)
from faninsar.stack.masks import (
    CombinedMaskView,
    StackMasks,
    StackMaskView,
)
from faninsar.stack.provider import (
    SceneProductionCallback,
    SourceHandle,
    StackSceneProvider,
    UnsupportedStackCapabilityError,
    unavailable_scene_provider,
    unsupported_stack_capability,
)
from faninsar.stack.search import (
    MissionPreparedInputs,
    MissionSearchProfile,
    SelectionRequest,
    merge_profile_filters,
)
from faninsar.stack.session import Stack
from faninsar.stack.stage import Stage

__all__ = [
    "MASK_KINDS",
    "STAGES",
    "CombinedMaskView",
    "CoregMode",
    "FlattenStage",
    "GpuMemoryReclaim",
    "InterferogramArtifact",
    "InterferogramArtifactStore",
    "MaskDefinition",
    "MaskPlan",
    "MissionPreparedInputs",
    "MissionSearchProfile",
    "SceneCatalog",
    "SceneProductionCallback",
    "SelectionRequest",
    "SourceHandle",
    "Stack",
    "StackConfig",
    "StackMaskView",
    "StackMasks",
    "StackSceneProvider",
    "Stage",
    "UnsupportedStackCapabilityError",
    "UnwrappedArtifact",
    "merge_profile_filters",
    "resolve_stack_grid",
    "unavailable_scene_provider",
    "unsupported_stack_capability",
    "utm_ups_grid",
]
