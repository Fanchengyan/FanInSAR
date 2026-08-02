"""Backend dispatcher for explicit IRLS and snaphu unwrapping."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.unwrap.common import CommonUnwrapResult, build_common_result
from faninsar.processing.unwrap.irls import irls_unwrap
from faninsar.processing.unwrap.snaphu_backend import SnaphuConfig, snaphu_unwrap
from faninsar.processing.unwrap.temporal_irls import (
    TemporalUnwrapResult,
    unwrap_temporal_irls,
)

logger = setup_logger(__name__)

UnwrapBackend = Literal["irls", "snaphu"]

__all__ = [
    "TemporalUnwrapResult",
    "UnwrapBackend",
    "unwrap",
    "unwrap_temporal_irls",
]


def unwrap(
    wrapped_or_complex: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    method: UnwrapBackend = "snaphu",
    snaphu_config: SnaphuConfig | None = None,
    irls_kwargs: dict[str, Any] | None = None,
) -> CommonUnwrapResult:
    """Unwrap phase with an explicitly selected backend.

    Parameters
    ----------
    wrapped_or_complex : numpy.ndarray
        Wrapped phase (IRLS) or complex interferogram (snaphu).
    coherence : numpy.ndarray, optional
        Coherence weights.
    method : {"irls", "snaphu"}
        Backend selection. There is no silent fallback between backends.
    snaphu_config : SnaphuConfig, optional
        Configuration for the snaphu backend.
    irls_kwargs : dict, optional
        Extra keyword arguments for :func:`irls_unwrap`.

    Returns
    -------
    CommonUnwrapResult
        Normalized unwrap product.

    """
    if method == "irls":
        phase = np.asarray(wrapped_or_complex)
        if np.iscomplexobj(phase):
            phase = np.angle(phase)
        result = irls_unwrap(phase, coherence, **(irls_kwargs or {}))
        return build_common_result(
            wrapped_phase=phase,
            unwrapped_phase=result.unwrapped_phase,
            connected_components=result.connected_components,
            method="irls",
            metrics={
                "iterations": float(result.iterations),
                "converged": float(result.converged),
            },
            configuration=dict(irls_kwargs or {}),
        )
    if method == "snaphu":
        if not np.iscomplexobj(wrapped_or_complex):
            reject_invalid_state(
                "snaphu backend requires a complex interferogram input"
            )
        if coherence is None:
            reject_invalid_state("snaphu backend requires coherence")
        return snaphu_unwrap(
            np.asarray(wrapped_or_complex),
            np.asarray(coherence),
            config=snaphu_config,
        )
    return reject_invalid_state(f"unknown unwrap method: {method}")
