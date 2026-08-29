"""Phase unwrapping backends and reconciliation."""

from __future__ import annotations

from .api import UnwrapBackend, unwrap
from .common import (
    SpatialUnwrapper,
    SpatialUnwrapResult,
)
from .errors import NoValidSupportError, UnwrapFailedError
from .irls import IRLSUnwrapResult, SpatialIRLS, irls_unwrap, wrap_phase
from .quality import (
    MetricDistribution,
    StackQualityCriteria,
    StackQualityReport,
    evaluate_stack_quality,
)
from .reconcile import (
    ComponentCorrection,
    ReconciliationResult,
    align_components_to_reference,
    loop_closure_phase,
    reconcile_components,
)
from .snaphu_backend import (
    Snaphu,
    SnaphuConfig,
    SnaphuNotAvailableError,
    snaphu_available,
    snaphu_unwrap,
)
from .stack import StackUnwrapResult, unwrap_stack

__all__ = [
    "ComponentCorrection",
    "IRLSUnwrapResult",
    "MetricDistribution",
    "NoValidSupportError",
    "ReconciliationResult",
    "Snaphu",
    "SnaphuConfig",
    "SnaphuNotAvailableError",
    "SpatialIRLS",
    "SpatialUnwrapResult",
    "SpatialUnwrapper",
    "StackQualityCriteria",
    "StackQualityReport",
    "StackUnwrapResult",
    "UnwrapBackend",
    "UnwrapFailedError",
    "align_components_to_reference",
    "evaluate_stack_quality",
    "irls_unwrap",
    "loop_closure_phase",
    "reconcile_components",
    "snaphu_available",
    "snaphu_unwrap",
    "unwrap",
    "unwrap_stack",
    "wrap_phase",
]
