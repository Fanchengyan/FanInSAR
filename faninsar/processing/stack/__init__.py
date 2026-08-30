"""Stack session API (PROPOSAL-0017)."""

from __future__ import annotations

from faninsar._core.device import GpuMemoryReclaim
from faninsar.processing.stack.activation import (
    ActivationIssuerRecord,
    LocalActivationAuthority,
    StackGateEvent,
)
from faninsar.processing.stack.catalog import SceneCatalog
from faninsar.processing.stack.config import (
    ActivationMode,
    CoregMode,
    EsdMethod,
    FlattenStage,
    StackConfig,
)
from faninsar.processing.stack.ifg_store import (
    InterferogramArtifact,
    InterferogramArtifactStore,
    UnwrappedArtifact,
)
from faninsar.processing.stack.ion_store import (
    IonosphereArtifact,
    IonosphereArtifactStore,
)
from faninsar.processing.stack.mask_plan import (
    MASK_KINDS,
    STAGES,
    MaskDefinition,
    MaskPlan,
)
from faninsar.processing.stack.nisar import NISARStack
from faninsar.processing.stack.provider import (
    SceneProductionCallback,
    SourceHandle,
    StackProviderError,
    StackSceneProvider,
    UnsupportedStackCapabilityError,
    unavailable_scene_provider,
    unsupported_stack_capability,
)
from faninsar.processing.stack.s1 import S1Stack
from faninsar.processing.stack.session import Stack
from faninsar.processing.stack.stack_generation import (
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
    "IonosphereArtifact",
    "IonosphereArtifactStore",
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
    "open_unwrap_generation",
    "publish_unwrap_generation",
    "unavailable_scene_provider",
    "unsupported_stack_capability",
]
