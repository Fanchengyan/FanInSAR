"""Common Torch-oriented contract for spatial phase unwrapping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


def _raise_contract(message: str) -> None:
    """Log and raise a public contract error."""
    logger.error("spatial unwrap contract rejected: %s", message)
    raise ValueError(message)


if TYPE_CHECKING:
    import torch

FailureReason = Literal[
    "pcg_breakdown",
    "nonfinite_state",
    "inner_iteration_limit",
    "outer_iteration_limit",
    "backend_failure",
]


@dataclass(frozen=True, slots=True)
class SpatialUnwrapResult:
    """Result of unwrapping one two-dimensional interferometric pair.

    Parameters
    ----------
    phase : torch.Tensor
        Unwrapped phase in radians. It has the same shape, floating dtype, and
        device as the input phase.
    valid_mask : torch.Tensor
        Boolean support mask. Invalid output pixels are represented by NaN in
        ``phase``.
    component_labels : torch.Tensor
        Deterministic row-major component labels, with ``-1`` for invalid
        pixels.
    reference_values : torch.Tensor
        Principal wrapped phase at each component's row-major anchor.
    converged : bool
        Whether all numerical work completed successfully.
    iterations : int
        Number of completed outer IRLS iterations. Initial convergence is zero.
    pcg_iterations : int
        Total completed inner PCG iterations across all components and outers.
    residual_norm : float
        Global L2 norm of active-edge wrapped-gradient residuals.
    failure_reason : FailureReason or None
        Closed numerical/backend failure code, or ``None`` on success.

    Notes
    -----
    Component labels are zero-based. A component's reference value is an
    anchor value, not a post-solve scene-wide phase shift.

    """

    phase: torch.Tensor
    valid_mask: torch.Tensor
    component_labels: torch.Tensor
    reference_values: torch.Tensor
    converged: bool
    iterations: int
    pcg_iterations: int
    residual_norm: float
    failure_reason: FailureReason | None

    def __post_init__(self) -> None:
        """Validate the shape and state invariants of a result."""
        import torch

        for name in ("phase", "valid_mask", "component_labels", "reference_values"):
            if not hasattr(getattr(self, name), "shape"):
                _raise_contract(f"{name} must be a torch tensor")
        if self.phase.ndim != 2:
            _raise_contract("phase must be a 2-D tensor")
        if self.valid_mask.shape != self.phase.shape:
            _raise_contract("valid_mask must match phase shape")
        if self.component_labels.shape != self.phase.shape:
            _raise_contract("component_labels must match phase shape")
        if self.reference_values.ndim != 1:
            _raise_contract("reference_values must be one-dimensional")
        if self.valid_mask.dtype is not torch.bool:
            _raise_contract("valid_mask must be boolean")
        if self.phase.device != self.valid_mask.device:
            _raise_contract("result tensors must share a device")
        if self.component_labels.device != self.phase.device:
            _raise_contract("result tensors must share a device")
        if self.reference_values.device != self.phase.device:
            _raise_contract("result tensors must share a device")
        if self.converged != (self.failure_reason is None):
            _raise_contract("converged must agree with failure_reason")
        if self.iterations < 0 or self.pcg_iterations < 0:
            _raise_contract("iteration diagnostics cannot be negative")


class SpatialUnwrapper(ABC):
    """Abstract strategy for one spatial interferometric pair.

    Implementations consume a two-dimensional Torch tensor in
    ``(azimuth, range)`` order and return :class:`SpatialUnwrapResult` on the
    same device. This seam intentionally covers spatial unwrapping only;
    temporal/network or composite orchestration is outside this interface.
    A strategy may use any numerical backend, but it must preserve the input
    shape and report unsupported pixels through the result mask and NaNs.
    """

    @abstractmethod
    def unwrap(
        self,
        wrapped_phase: torch.Tensor,
        *,
        coherence: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> SpatialUnwrapResult:
        """Unwrap one 2-D phase tensor without changing its device.

        Parameters
        ----------
        wrapped_phase : torch.Tensor
            Phase in radians, shape ``(azimuth, range)``. It must be a real
            floating tensor and remain on its original Torch device.
        coherence : torch.Tensor, optional
            Dimensionless quality values in ``[0, 1]`` with the same shape.
            ``None`` means uniform quality; finite zero values may disconnect
            incident graph edges while retaining their pixels.
        valid_mask : torch.Tensor, optional
            Boolean authoritative support mask in ``(azimuth, range)`` order.
            ``None`` means the finite phase support is authoritative.

        Returns
        -------
        SpatialUnwrapResult
            Same-shape, same-device phase, support mask, component labels,
            reference anchors, and numerical diagnostics.

        Raises
        ------
        NoValidSupportError
            If no pixel remains supported for the selected algorithm.
        ValueError
            If shape, dtype, device, or quality bounds are invalid.

        """
