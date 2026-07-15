"""Phase unwrapping backends and reconciliation."""

from __future__ import annotations

from .api import UnwrapBackend, unwrap
from .common import CommonUnwrapResult, UnwrapMethod, build_common_result
from .dct_irls import dct_irls_unwrap
from .irls import IRLSUnwrapResult, irls_unwrap, wrap_phase
from .reconcile import (
    ComponentCorrection,
    ReconciliationResult,
    align_components_to_reference,
    loop_closure_phase,
    reconcile_components,
)
from .snaphu_backend import (
    SnaphuConfig,
    SnaphuNotAvailableError,
    snaphu_available,
    snaphu_unwrap,
)
from .stack import StackUnwrapResult, unwrap_stack
from .temporal_irls import TemporalUnwrapResult, unwrap_temporal_irls

__all__ = [
    "CommonUnwrapResult",
    "ComponentCorrection",
    "IRLSUnwrapResult",
    "ReconciliationResult",
    "SnaphuConfig",
    "SnaphuNotAvailableError",
    "StackUnwrapResult",
    "TemporalUnwrapResult",
    "UnwrapBackend",
    "UnwrapMethod",
    "align_components_to_reference",
    "build_common_result",
    "dct_irls_unwrap",
    "irls_unwrap",
    "loop_closure_phase",
    "reconcile_components",
    "snaphu_available",
    "snaphu_unwrap",
    "unwrap",
    "unwrap_stack",
    "unwrap_temporal_irls",
    "wrap_phase",
]
