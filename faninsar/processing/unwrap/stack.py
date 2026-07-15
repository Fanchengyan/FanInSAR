"""Thin orchestrator: 2D spatial → 1D temporal → batch_lstsq inversion.

Stages are orthogonal and independently skippable. This module does **not**
embed dask scheduling (outer pipeline only) and does not checkpoint to zarr
in the P0 minimal version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.timeseries.inversion import invert_unwrapped_pairs
from faninsar.processing.unwrap._incidence import build_incidence_matrix
from faninsar.processing.unwrap.api import unwrap
from faninsar.processing.unwrap.temporal_irls import unwrap_temporal_irls

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar import Pairs

logger = setup_logger(__name__)

SpatialMethod = Literal["irls", "dct_irls"]


@dataclass(frozen=True, slots=True)
class StackUnwrapResult:
    """Product of the multi-stage stack unwrap orchestrator.

    Attributes
    ----------
    phase_2d_unw : numpy.ndarray or None
        Pair phases after the 2D spatial stage (or the input when spatial is
        skipped), shape ``(n_pairs, H, W)``.
    phase_1d_unw : numpy.ndarray or None
        Pair phases after the 1D temporal stage, or ``None`` when skipped.
    corrections_k : numpy.ndarray or None
        Integer 2π corrections from the temporal stage, or ``None``.
    timeseries : numpy.ndarray or None
        Cumulative displacement ``(n_dates, H, W)`` when inversion ran.
    pair_ids : tuple of str
        Pair identifiers aligned with axis 0 of the phase stacks.
    method : str
        Orchestrator identity (``"stack"``).
    spatial_method : str
        Spatial backend name requested (even if spatial was skipped).
    temporal_applied : bool
        Whether the 1D temporal stage ran.
    inverted : bool
        Whether the batch least-squares inversion stage ran.

    """

    phase_2d_unw: np.ndarray | None
    phase_1d_unw: np.ndarray | None
    corrections_k: np.ndarray | None
    timeseries: np.ndarray | None
    pair_ids: tuple[str, ...]
    method: str
    spatial_method: str
    temporal_applied: bool
    inverted: bool


def _resolve_pair_ids(
    pair_dates: Pairs | Sequence[tuple[str, str]] | np.ndarray,
    n_pairs: int,
) -> tuple[str, ...]:
    """Resolve pair ids from ``pair_dates``, validating count."""
    _a_mat, pair_ids = build_incidence_matrix(pair_dates)
    if len(pair_ids) != n_pairs:
        reject_invalid_state(
            f"pair_dates length ({len(pair_ids)}) must match phase_stack "
            f"axis-0 ({n_pairs})",
        )
    return pair_ids


def unwrap_stack(
    phase_stack: np.ndarray,
    pair_dates: Pairs | Sequence[tuple[str, str]] | np.ndarray,
    *,
    do_spatial: bool = True,
    do_temporal: bool = True,
    do_invert: bool = True,
    spatial_method: SpatialMethod = "irls",
    spatial_device: str = "cpu",
    temporal_device: str = "cpu",
    lstsq_device: str = "cpu",
    temporal_kwargs: dict[str, Any] | None = None,
    spatial_kwargs: dict[str, Any] | None = None,
) -> StackUnwrapResult:
    """Run the 2D → 1D → inversion unwrap chain on a pair-phase stack.

    Parameters
    ----------
    phase_stack : numpy.ndarray
        Wrapped (or pre-unwrapped) pair phases of shape ``(n_pairs, H, W)``.
    pair_dates : Pairs or sequence of (primary, secondary) or ndarray
        Temporal pair definitions aligned with axis 0.
    do_spatial : bool, optional
        If ``True`` (default), run per-pair 2D spatial unwrap. If ``False``,
        the input is treated as already spatially unwrapped.
    do_temporal : bool, optional
        If ``True`` (default), run 1D temporal/network IRLS.
    do_invert : bool, optional
        If ``True`` (default), invert unwrapped pairs to a cumulative
        time series via
        :func:`~faninsar.processing.timeseries.inversion.invert_unwrapped_pairs`
        (batch least-squares -- **not** a 1D unwrap stage).
    spatial_method : {"irls", "dct_irls"}, optional
        2D spatial backend. Both backends are CPU-only today; ``spatial_device``
        is accepted for API stability but does not route GPU work.
    spatial_device : str, optional
        Reserved device hint for spatial backends (currently CPU-only).
    temporal_device : str, optional
        Device for :func:`unwrap_temporal_irls` (default ``"cpu"`` preferred).
    lstsq_device : str, optional
        Device for the inversion stage.
    temporal_kwargs : dict, optional
        Extra keyword arguments for :func:`unwrap_temporal_irls`.
        ``wrapped_input`` defaults to ``False`` after a spatial stage (or when
        spatial is skipped and the caller already provides unwrapped phases).
    spatial_kwargs : dict, optional
        Extra keyword arguments forwarded to the spatial backend via
        :func:`~faninsar.processing.unwrap.api.unwrap` ``irls_kwargs``.

    Returns
    -------
    StackUnwrapResult
        Stage products and flags.

    Notes
    -----
    No silent method/device downgrade. Stages are independent: any combination
    of skip flags is valid. This orchestrator is sequential; dask parallelism
    belongs at the outer pipeline level.

    """
    phase = np.asarray(phase_stack, dtype=np.float64)
    if phase.ndim != 3:
        reject_invalid_state(
            "unwrap_stack requires phase_stack with shape (n_pairs, H, W)",
        )
    n_pairs, height, width = phase.shape
    if height < 1 or width < 1:
        reject_invalid_state("spatial dimensions must be positive")

    pair_ids = _resolve_pair_ids(pair_dates, n_pairs)
    spatial_kw = dict(spatial_kwargs or {})
    temporal_kw = dict(temporal_kwargs or {})

    # --- Stage 1: 2D spatial unwrap (per pair) ---
    if do_spatial:
        if spatial_method not in ("irls", "dct_irls"):
            reject_invalid_state(
                f"spatial_method must be 'irls' or 'dct_irls', got {spatial_method!r}",
            )
        if spatial_device not in ("cpu", "auto"):
            # irls / dct_irls are CPU-only; refuse silent GPU claims
            logger.warning(
                "spatial backends are CPU-only today; spatial_device=%r is ignored",
                spatial_device,
            )
        unwrapped_pairs: list[np.ndarray] = []
        for i in range(n_pairs):
            result_2d = unwrap(
                phase[i],
                method=spatial_method,
                irls_kwargs=spatial_kw,
            )
            unwrapped_pairs.append(
                np.asarray(result_2d.unwrapped_phase, dtype=np.float64),
            )
        phase_2d = np.stack(unwrapped_pairs, axis=0)
        logger.info(
            "Spatial unwrap (%s) finished for %s pairs on %sx%s",
            spatial_method,
            n_pairs,
            height,
            width,
        )
    else:
        phase_2d = phase.copy()

    # --- Stage 2: 1D temporal / network IRLS ---
    phase_1d: np.ndarray | None = None
    corrections_k: np.ndarray | None = None
    temporal_applied = False
    if do_temporal:
        # After spatial (or when spatial is skipped), phases are treated as
        # already-unwrapped unless the caller overrides wrapped_input.
        temporal_kw.setdefault("wrapped_input", False)
        temporal_kw.setdefault("device", temporal_device)
        temporal_result = unwrap_temporal_irls(
            phase_2d,
            pair_dates,
            **temporal_kw,
        )
        phase_1d = temporal_result.phase_unw
        corrections_k = temporal_result.corrections_k
        temporal_applied = True
        logger.info(
            "Temporal IRLS finished (iter=%s, converged=%s, device=%s)",
            temporal_result.iterations,
            temporal_result.converged,
            temporal_result.device,
        )

    # --- Stage 3: batch least-squares inversion (not 1D unwrap) ---
    timeseries: np.ndarray | None = None
    inverted = False
    if do_invert:
        source = phase_1d if phase_1d is not None else phase_2d
        pair_phases = {
            pid: np.asarray(source[i], dtype=np.float64)
            for i, pid in enumerate(pair_ids)
        }
        ts_result = invert_unwrapped_pairs(pair_phases, device=lstsq_device)
        timeseries = np.asarray(ts_result.cumulative, dtype=np.float64)
        inverted = True
        logger.info(
            "Inversion finished: %s dates, device=%s",
            timeseries.shape[0],
            lstsq_device,
        )

    return StackUnwrapResult(
        phase_2d_unw=phase_2d,
        phase_1d_unw=phase_1d,
        corrections_k=corrections_k,
        timeseries=timeseries,
        pair_ids=pair_ids,
        method="stack",
        spatial_method=spatial_method,
        temporal_applied=temporal_applied,
        inverted=inverted,
    )
