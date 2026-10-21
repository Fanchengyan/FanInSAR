"""Stack session API (PROPOSAL-0017)."""

from __future__ import annotations

from faninsar.io.storage.ifg_store import (
    InterferogramArtifact,
    InterferogramArtifactStore,
    UnwrappedArtifact,
)
from faninsar.processing.runtime.device import GpuMemoryReclaim
from faninsar.stack.activation import (
    ActivationIssuerRecord,
    LocalActivationAuthority,
    StackGateEvent,
)
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.config import (
    ActivationMode,
    CoregMode,
    FlattenStage,
    StackConfig,
)
from faninsar.stack.grid import automatic_grid, resolve_stack_grid
from faninsar.stack.mask_plan import (
    MASK_KINDS,
    STAGES,
    MaskDefinition,
    MaskPlan,
)
from faninsar.stack.provider import (
    SceneProductionCallback,
    SourceHandle,
    StackProviderError,
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
    "ActivationIssuerRecord",
    "ActivationMode",
    "CoregMode",
    "FlattenStage",
    "GpuMemoryReclaim",
    "InterferogramArtifact",
    "InterferogramArtifactStore",
    "LocalActivationAuthority",
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
    "StackGateEvent",
    "StackProviderError",
    "StackSceneProvider",
    "Stage",
    "UnsupportedStackCapabilityError",
    "UnwrappedArtifact",
    "automatic_grid",
    "merge_profile_filters",
    "resolve_stack_grid",
    "unavailable_scene_provider",
    "unsupported_stack_capability",
]
