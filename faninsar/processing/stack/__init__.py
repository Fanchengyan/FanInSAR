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
    StackConfig,
)
from faninsar.processing.stack.nisar import NISARStack
from faninsar.processing.stack.provider import (
    SceneProductionCallback,
    StackProviderError,
    StackSceneProvider,
    UnsupportedStackCapabilityError,
    unavailable_scene_provider,
    unsupported_stack_capability,
)
from faninsar.processing.stack.session import Stack

__all__ = [
    "ActivationIssuerRecord",
    "ActivationMode",
    "CoregMode",
    "EsdMethod",
    "GpuMemoryReclaim",
    "LocalActivationAuthority",
    "NISARStack",
    "SceneCatalog",
    "SceneProductionCallback",
    "Stack",
    "StackConfig",
    "StackGateEvent",
    "StackProviderError",
    "StackSceneProvider",
    "UnsupportedStackCapabilityError",
    "unavailable_scene_provider",
    "unsupported_stack_capability",
]
