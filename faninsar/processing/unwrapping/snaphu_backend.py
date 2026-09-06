"""snaphu-py unwrapping backend with lazy import and no silent fallback."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import ProcessingContractError, reject_invalid_state
from faninsar.processing.runtime.capabilities import snaphu_capability
from faninsar.processing.unwrapping.common import (
    SpatialUnwrapper,
    SpatialUnwrapResult,
)
from faninsar.processing.unwrapping.errors import NoValidSupportError

logger = setup_logger(__name__)

SnaphuCostMode = Literal["defo", "smooth"]


class SnaphuNotAvailableError(ProcessingContractError):
    """Raised when the required snaphu-py package is not installed."""


@dataclass(frozen=True, slots=True)
class SnaphuConfig:
    """Typed configuration for the snaphu-py backend."""

    cost: SnaphuCostMode = "defo"
    nlooks: float = 1.0
    ntiles: tuple[int, int] = (1, 1)
    nproc: int = 1


class Snaphu(SpatialUnwrapper):
    """Adapt the SNAPHU spatial backend to the Torch spatial seam.

    Parameters
    ----------
    config : SnaphuConfig, optional
        Typed configuration passed to ``snaphu-py``.

    Notes
    -----
    The adapter accepts Torch tensors and performs the only NumPy boundary in
    this strategy around the external SNAPHU package. It never falls back to
    another unwrapper. Input tensors are returned on their original device.

    """

    def __init__(self, config: SnaphuConfig | None = None) -> None:
        """Store the explicit SNAPHU configuration."""
        self.config = config or SnaphuConfig()

    def unwrap(
        self,
        wrapped_phase: Any,
        *,
        coherence: Any | None = None,
        valid_mask: Any | None = None,
    ) -> SpatialUnwrapResult:
        """Unwrap one 2-D Torch phase tensor with SNAPHU.

        Parameters
        ----------
        wrapped_phase : torch.Tensor
            Wrapped phase in radians, ordered ``(azimuth, range)``.
        coherence : torch.Tensor, optional
            Finite values in ``[0, 1]``. Nonfinite values remove support.
        valid_mask : torch.Tensor, optional
            Boolean support mask matching ``wrapped_phase``.

        Returns
        -------
        SpatialUnwrapResult
            The parsed SNAPHU phase and deterministic spatial diagnostics.

        Raises
        ------
        NoValidSupportError
            If no input pixel is supported.
        SnaphuNotAvailableError
            If the external SNAPHU package is unavailable.
        ValueError
            If tensor inputs violate the spatial contract.

        """
        import torch

        self._validate_tensors(wrapped_phase, coherence, valid_mask)
        support = torch.isfinite(wrapped_phase)
        if valid_mask is not None:
            support &= valid_mask
        if coherence is not None:
            finite = torch.isfinite(coherence)
            finite_values = coherence[finite]
            if torch.any(finite_values < 0) or torch.any(finite_values > 1):
                reject_invalid_state("coherence finite values must be within [0, 1]")
            support &= finite
            backend_coherence = torch.nan_to_num(coherence, nan=0.0)
        else:
            backend_coherence = torch.ones_like(wrapped_phase)
        if not bool(torch.any(support)):
            logger.error("SNAPHU input has no valid support")
            raise NoValidSupportError

        phase = torch.remainder(wrapped_phase + torch.pi, 2 * torch.pi) - torch.pi
        if coherence is None:
            coherence = backend_coherence
        try:
            backend_phase, _backend_components = _snaphu_numpy(
                torch.exp(1j * torch.nan_to_num(phase)).detach().cpu().numpy(),
                backend_coherence.detach().cpu().numpy(),
                config=self.config,
            )
        except SnaphuNotAvailableError:
            raise
        except Exception:
            logger.exception("SNAPHU backend failed")
            return self._failed_result(phase, support)

        output = torch.as_tensor(
            backend_phase,
            dtype=phase.dtype,
            device=phase.device,
        )
        output_valid = support & torch.isfinite(output)
        if tuple(output.shape) != tuple(phase.shape) or not bool(
            torch.all(torch.isfinite(output[output_valid]))
        ):
            logger.error("SNAPHU returned malformed or nonfinite output")
            return self._failed_result(phase, support)
        labels, references, active_h, active_v = self._labels_and_edges(
            phase, output_valid, coherence
        )
        residual = self._residual_norm(phase, output, active_h, active_v)
        output = output.clone()
        output[~output_valid] = torch.nan
        return SpatialUnwrapResult(
            phase=output,
            valid_mask=output_valid,
            component_labels=labels,
            reference_values=references,
            converged=True,
            iterations=0,
            pcg_iterations=0,
            residual_norm=residual,
            failure_reason=None,
        )

    @staticmethod
    def _validate_tensors(phase: Any, coherence: Any, valid_mask: Any) -> None:
        """Validate Torch tensor shape, dtype, and device constraints."""
        import torch

        if not isinstance(phase, torch.Tensor) or phase.ndim != 2:
            reject_invalid_state("wrapped_phase must be a 2-D torch tensor")
        if not phase.is_floating_point() or phase.is_complex():
            reject_invalid_state("wrapped_phase must be a real floating tensor")
        for name, value in (("coherence", coherence), ("valid_mask", valid_mask)):
            if value is not None and not isinstance(value, torch.Tensor):
                reject_invalid_state(f"{name} must be a torch tensor")
            if value is not None and value.shape != phase.shape:
                reject_invalid_state(f"{name} must match wrapped_phase shape")
            if value is not None and value.device != phase.device:
                reject_invalid_state(f"{name} must share wrapped_phase device")
        if coherence is not None and not coherence.is_floating_point():
            reject_invalid_state("coherence must be a floating tensor")
        if valid_mask is not None and valid_mask.dtype is not torch.bool:
            reject_invalid_state("valid_mask must have dtype torch.bool")

    @staticmethod
    def _labels_and_edges(
        phase: Any,
        support: Any,
        coherence: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Build row-major support labels and positive-quality edge masks."""
        import torch

        height, width = phase.shape
        active_h = support[:, :-1] & support[:, 1:]
        active_v = support[:-1, :] & support[1:, :]
        if coherence is not None:
            active_h &= coherence[:, :-1] > 0
            active_h &= coherence[:, 1:] > 0
            active_v &= coherence[:-1, :] > 0
            active_v &= coherence[1:, :] > 0
        parent = list(range(height * width))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for row, col in zip(*torch.where(active_h), strict=True):
            union(int(row) * width + int(col), int(row) * width + int(col) + 1)
        for row, col in zip(*torch.where(active_v), strict=True):
            union(int(row) * width + int(col), int(row) * width + int(col) + width)
        labels = torch.full_like(support, -1, dtype=torch.int64)
        anchors: list[int] = []
        root_labels: dict[int, int] = {}
        for index in range(height * width):
            row, col = divmod(index, width)
            if not bool(support[row, col]):
                continue
            root = find(index)
            label = root_labels.setdefault(root, len(root_labels))
            labels[row, col] = label
            if label == len(anchors):
                anchors.append(index)
        anchor_tensor = torch.as_tensor(anchors, device=phase.device)
        return labels, phase.flatten()[anchor_tensor], active_h, active_v

    @staticmethod
    def _residual_norm(phase: Any, output: Any, active_h: Any, active_v: Any) -> float:
        """Compute active-edge wrapped-gradient residual norm."""
        import torch

        def wrap(values: Any) -> Any:
            return torch.remainder(values + torch.pi, 2 * torch.pi) - torch.pi

        residual_h = output[:, 1:] - output[:, :-1] - wrap(phase[:, 1:] - phase[:, :-1])
        residual_v = output[1:, :] - output[:-1, :] - wrap(phase[1:, :] - phase[:-1, :])
        values = torch.cat((residual_h[active_h], residual_v[active_v]))
        return float(torch.linalg.vector_norm(values)) if values.numel() else 0.0

    @staticmethod
    def _failed_result(phase: Any, support: Any) -> SpatialUnwrapResult:
        """Return a closed failed result after an admitted backend failure."""
        import torch

        labels = torch.full_like(phase, -1, dtype=torch.int64)
        output = torch.full_like(phase, torch.nan)
        return SpatialUnwrapResult(
            phase=output,
            valid_mask=torch.zeros_like(support, dtype=torch.bool),
            component_labels=labels,
            reference_values=torch.empty(0, dtype=phase.dtype, device=phase.device),
            converged=False,
            iterations=0,
            pcg_iterations=0,
            residual_norm=float("inf"),
            failure_reason="backend_failure",
        )


def snaphu_available() -> bool:
    """Return whether the snaphu package is importable."""
    return find_spec("snaphu") is not None


def require_snaphu() -> Any:
    """Import snaphu lazily or raise a typed capability error.

    Returns
    -------
    module
        The imported ``snaphu`` module.

    Raises
    ------
    SnaphuNotAvailableError
        If the required ``snaphu-py`` package is not installed.

    """
    capability = snaphu_capability()
    if not capability.available:
        message = (
            "snaphu-py is not installed. Reinstall FanInSAR with "
            "`pip install faninsar` "
            f"and review the license caveat: {capability.license_caveat}"
        )
        logger.error(message)
        raise SnaphuNotAvailableError(message)
    return import_module("snaphu")


def _snaphu_numpy(
    complex_ifg: np.ndarray,
    coherence: np.ndarray,
    *,
    config: SnaphuConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Unwrap a complex interferogram with the snaphu-py API.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram (not phase-only).
    coherence : numpy.ndarray
        Coherence in ``[0, 1]`` matching the interferogram shape.
    config : SnaphuConfig, optional
        Typed snaphu configuration.

    Raises
    ------
    SnaphuNotAvailableError
        If snaphu is not installed.
    InvalidProcessingStateError
        If inputs are invalid.

    Notes
    -----
    This backend never silently falls back to IRLS. Callers must choose the
    method explicitly. SNAPHU license terms are surfaced via capabilities.

    """
    if complex_ifg.ndim != 2 or not np.iscomplexobj(complex_ifg):
        reject_invalid_state("snaphu unwrap requires a 2-D complex interferogram")
    if coherence.shape != complex_ifg.shape:
        reject_invalid_state("coherence must match the interferogram shape")

    cfg = config or SnaphuConfig()
    snaphu = require_snaphu()
    capability = snaphu_capability()

    # snaphu-py public API: snaphu.unwrap(igram, corr, nlooks=..., cost=...)
    kwargs: dict[str, Any] = {
        "nlooks": float(cfg.nlooks),
        "cost": cfg.cost,
        "nproc": int(cfg.nproc),
        "ntiles": (int(cfg.ntiles[0]), int(cfg.ntiles[1])),
    }
    valid = (
        np.isfinite(complex_ifg.real)
        & np.isfinite(complex_ifg.imag)
        & np.isfinite(coherence)
        & (coherence > 0.0)
    )
    kwargs["mask"] = valid.astype(np.uint8)

    logger.info(
        "Calling snaphu-py unwrap (wrapper=%s, bundled=%s, cost=%s)",
        capability.wrapper_version,
        capability.bundled_snaphu_version,
        cfg.cost,
    )
    unwrapped, conncomp = snaphu.unwrap(
        np.asarray(complex_ifg),
        np.asarray(coherence, dtype=np.float32),
        **kwargs,
    )
    return (
        np.asarray(unwrapped, dtype=np.float32),
        np.asarray(conncomp, dtype=np.int32),
    )


def snaphu_unwrap(
    complex_ifg: np.ndarray,
    coherence: np.ndarray,
    *,
    config: SnaphuConfig | None = None,
) -> SpatialUnwrapResult:
    """Unwrap a complex interferogram and return the spatial result contract.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Two-dimensional complex interferogram.
    coherence : numpy.ndarray
        Coherence in ``[0, 1]`` matching ``complex_ifg``.
    config : SnaphuConfig, optional
        Explicit SNAPHU configuration.

    Returns
    -------
    SpatialUnwrapResult
        Torch result on CPU. Use :class:`Snaphu` to preserve a caller's
        existing Torch device.

    """
    import torch

    if not isinstance(complex_ifg, np.ndarray):
        complex_ifg = np.asarray(complex_ifg)
    if not isinstance(coherence, np.ndarray):
        coherence = np.asarray(coherence)
    phase = np.angle(complex_ifg).astype(np.float32)
    return Snaphu(config).unwrap(
        torch.as_tensor(phase),
        coherence=torch.as_tensor(coherence, dtype=torch.float32),
    )
