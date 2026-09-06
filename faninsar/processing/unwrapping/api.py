"""Backend dispatcher for explicit IRLS and snaphu unwrapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.unwrapping.irls import SpatialIRLS, _resolve_device

if TYPE_CHECKING:
    from faninsar.processing.unwrapping.common import SpatialUnwrapResult
    from faninsar.processing.unwrapping.snaphu_backend import SnaphuConfig

logger = setup_logger(__name__)

UnwrapBackend = Literal["irls", "snaphu"]

__all__ = [
    "UnwrapBackend",
    "unwrap",
]


def unwrap(
    wrapped_or_complex: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    method: UnwrapBackend = "snaphu",
    snaphu_config: SnaphuConfig | None = None,
    irls_kwargs: dict[str, Any] | None = None,
) -> SpatialUnwrapResult:
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
        Extra keyword arguments for :class:`~.irls.SpatialIRLS`. The
        ``device`` entry selects the Torch device before input conversion.

    Returns
    -------
    SpatialUnwrapResult
        Torch spatial result on the explicitly selected device.

    """
    if method == "irls":
        import torch

        phase = np.asarray(wrapped_or_complex)
        if np.iscomplexobj(phase):
            phase = np.angle(phase)
        options = dict(irls_kwargs or {})
        requested_device = options.pop("device", "auto")
        resolved_device = _resolve_device(requested_device)
        phase_tensor = torch.as_tensor(
            phase,
            dtype=torch.float32,
            device=resolved_device,
        )
        coherence_tensor = (
            None
            if coherence is None
            else torch.as_tensor(
                coherence,
                dtype=torch.float32,
                device=resolved_device,
            )
        )
        return SpatialIRLS(**options).unwrap(
            phase_tensor,
            coherence=coherence_tensor,
        )
    if method == "snaphu":
        # Keep the optional backend out of the import graph until it is
        # explicitly selected.  In particular, importing the dispatcher for
        # IRLS must not load the external snaphu package or its adapter.
        from faninsar.processing.unwrapping.snaphu_backend import Snaphu

        if not np.iscomplexobj(wrapped_or_complex):
            reject_invalid_state(
                "snaphu backend requires a complex interferogram input"
            )
        import torch

        if coherence is None:
            reject_invalid_state("snaphu backend requires coherence")
        if not isinstance(wrapped_or_complex, np.ndarray):
            wrapped_or_complex = np.asarray(wrapped_or_complex)
        phase = np.angle(wrapped_or_complex)
        phase_tensor = torch.as_tensor(phase, dtype=torch.float32)
        coherence_tensor = torch.as_tensor(coherence, dtype=torch.float32)
        return Snaphu(snaphu_config).unwrap(
            phase_tensor,
            coherence=coherence_tensor,
        )
    return reject_invalid_state(f"unknown unwrap method: {method}")
