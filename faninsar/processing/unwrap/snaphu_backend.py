"""snaphu-py unwrapping backend with lazy import and no silent fallback."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Literal

import numpy as np

from faninsar.capabilities import snaphu_capability
from faninsar.logging import setup_logger
from faninsar.processing.errors import ProcessingContractError, reject_invalid_state
from faninsar.processing.unwrap.common import CommonUnwrapResult, build_common_result

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


def snaphu_unwrap(
    complex_ifg: np.ndarray,
    coherence: np.ndarray,
    *,
    config: SnaphuConfig | None = None,
) -> CommonUnwrapResult:
    r"""Unwrap a complex interferogram with the snaphu-py API.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram (not phase-only).
    coherence : numpy.ndarray
        Coherence in ``[0, 1]`` matching the interferogram shape.
    config : SnaphuConfig, optional
        Typed snaphu configuration.

    Returns
    -------
    CommonUnwrapResult
        Normalized unwrap product with method ``"snaphu"``.

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
    wrapped = np.angle(complex_ifg)
    return build_common_result(
        wrapped_phase=wrapped,
        unwrapped_phase=np.asarray(unwrapped, dtype=np.float32),
        connected_components=np.asarray(conncomp, dtype=np.int32),
        method="snaphu",
        metrics={
            "mean_coherence": float(np.nanmean(coherence)),
        },
        configuration={
            "cost": cfg.cost,
            "nlooks": cfg.nlooks,
            "ntiles": cfg.ntiles,
            "nproc": cfg.nproc,
            "wrapper_version": capability.wrapper_version,
            "bundled_snaphu_version": capability.bundled_snaphu_version,
            "license_caveat": capability.license_caveat,
        },
    )
