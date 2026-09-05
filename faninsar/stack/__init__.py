"""Stack session API (PROPOSAL-0017)."""

from __future__ import annotations

from faninsar._core.device import GpuMemoryReclaim
from faninsar.stack.activation import (
    ActivationIssuerRecord,
    LocalActivationAuthority,
    StackGateEvent,
)
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.config import (
    ActivationMode,
    CoregMode,
    EsdMethod,
    FlattenStage,
    StackConfig,
)
from faninsar.stack.grid import automatic_grid, resolve_stack_grid
from faninsar.stack.ifg_store import (
    InterferogramArtifact,
    InterferogramArtifactStore,
    UnwrappedArtifact,
)
from faninsar.stack.mask_plan import (
    MASK_KINDS,
    STAGES,
    MaskDefinition,
    MaskPlan,
)
from faninsar.stack.nisar import NISARStack
from faninsar.stack.provider import (
    SceneProductionCallback,
    SourceHandle,
    StackProviderError,
    StackSceneProvider,
    UnsupportedStackCapabilityError,
    unavailable_scene_provider,
    unsupported_stack_capability,
)
from faninsar.stack.s1 import S1Stack
from faninsar.stack.session import Stack
from faninsar.stack.stack_generation import (
    UnwrapResultGeneration,
    open_unwrap_generation,
    publish_unwrap_generation,
)

__all__ = [
    "MASK_KINDS",
    "STAGES",
    "ActivationIssuerRecord",
    "ActivationMode",
    "CoregMode",
    "EsdMethod",
    "FlattenStage",
    "GpuMemoryReclaim",
    "InterferogramArtifact",
    "InterferogramArtifactStore",
    "LocalActivationAuthority",
    "MaskDefinition",
    "MaskPlan",
    "NISARStack",
    "S1Stack",
    "SceneCatalog",
    "SceneProductionCallback",
    "SourceHandle",
    "Stack",
    "StackConfig",
    "StackGateEvent",
    "StackProviderError",
    "StackSceneProvider",
    "UnsupportedStackCapabilityError",
    "UnwrapResultGeneration",
    "UnwrappedArtifact",
    "automatic_grid",
    "open_unwrap_generation",
    "publish_unwrap_generation",
    "resolve_stack_grid",
    "unavailable_scene_provider",
    "unsupported_stack_capability",
]
