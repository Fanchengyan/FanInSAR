"""Common unwrapping result contract shared by IRLS and optional backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.unwrap.irls import wrap_phase

logger = setup_logger(__name__)

UnwrapMethod = Literal["irls", "dct_irls", "snaphu", "temporal_irls"]


@dataclass(frozen=True, slots=True)
class CommonUnwrapResult:
    """Normalized unwrapping product returned by every backend.

    Attributes
    ----------
    unwrapped_phase
        Unwrapped phase in radians.
    connected_components
        Integer connected-component labels (1-based; 0 = invalid).
    rewrap_residual
        ``wrap(unwrapped - wrapped)`` residual.
    method
        Explicit backend identity (never a silent fallback).
    metrics
        Backend-specific scalar diagnostics.
    configuration
        Provenance of the unwrap configuration.

    """

    unwrapped_phase: np.ndarray
    connected_components: np.ndarray
    rewrap_residual: np.ndarray
    method: UnwrapMethod
    metrics: dict[str, float]
    configuration: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate shapes and method identity."""
        if self.unwrapped_phase.shape != self.connected_components.shape:
            reject_invalid_state("unwrap phase and components must share a shape")
        if self.unwrapped_phase.shape != self.rewrap_residual.shape:
            reject_invalid_state("unwrap residual must match phase shape")
        if self.method not in ("irls", "dct_irls", "snaphu", "temporal_irls"):
            reject_invalid_state(f"unsupported unwrap method: {self.method}")


def build_common_result(
    *,
    wrapped_phase: np.ndarray,
    unwrapped_phase: np.ndarray,
    connected_components: np.ndarray,
    method: UnwrapMethod,
    metrics: dict[str, float] | None = None,
    configuration: dict[str, Any] | None = None,
) -> CommonUnwrapResult:
    """Assemble a normalized unwrap result with rewrap residual.

    Parameters
    ----------
    wrapped_phase : numpy.ndarray
        Original wrapped phase.
    unwrapped_phase : numpy.ndarray
        Backend unwrapped phase.
    connected_components : numpy.ndarray
        Component labels.
    method : {"irls", "dct_irls", "snaphu"}
        Explicit backend name.
    metrics, configuration : dict, optional
        Diagnostics and provenance.

    Returns
    -------
    CommonUnwrapResult
        Normalized product.

    """
    residual = wrap_phase(
        np.asarray(unwrapped_phase, dtype=np.float64)
        - np.asarray(wrapped_phase, dtype=np.float64)
    )
    return CommonUnwrapResult(
        unwrapped_phase=np.asarray(unwrapped_phase, dtype=np.float32),
        connected_components=np.asarray(connected_components, dtype=np.int32),
        rewrap_residual=residual.astype(np.float32),
        method=method,
        metrics=dict(metrics or {}),
        configuration=dict(configuration or {}),
    )
