"""Operation-aware native geometry preparation contracts.

The v2 builder deliberately owns preparation metadata only.  Numerical
transforms and the native ABI adapters are separate layers; dispatch consumes
an already prepared candidate and never invokes a compiler.
"""

from __future__ import annotations

from .bindings import NATIVE_RESULT_FIELDS, result_from_native_outputs
from .builder import (
    BuildPlan,
    GeometryOperation,
    NativeBackend,
    NativeBuilder,
    NativeBuildRequest,
    NativePreparationError,
    PreparationStatus,
    PreparedNativeCandidate,
    select_native_sources,
)
from .openmp import (
    OpenMPFlagProvider,
    OpenMPFlags,
    OpenMPProviderResult,
    openmp_provider_for_platform,
)
from .qualification import (
    OpenMPTelemetry,
    QualificationResult,
    qualify_openmp_telemetry,
)

__all__ = [
    "NATIVE_RESULT_FIELDS",
    "BuildPlan",
    "GeometryOperation",
    "NativeBackend",
    "NativeBuildRequest",
    "NativeBuilder",
    "NativePreparationError",
    "OpenMPFlagProvider",
    "OpenMPFlags",
    "OpenMPProviderResult",
    "OpenMPTelemetry",
    "PreparationStatus",
    "PreparedNativeCandidate",
    "QualificationResult",
    "openmp_provider_for_platform",
    "qualify_openmp_telemetry",
    "result_from_native_outputs",
    "select_native_sources",
]
