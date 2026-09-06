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
from faninsar.processing.unwrapping._incidence import build_incidence_matrix
from faninsar.processing.unwrapping.api import unwrap
from faninsar.processing.unwrapping.quality import (
    StackQualityCriteria,
    StackQualityReport,
    evaluate_stack_quality,
)
from faninsar.processing.unwrapping.temporal_irls import unwrap_temporal_irls
from faninsar.timeseries.results import invert_unwrapped_pairs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar import Pairs

logger = setup_logger(__name__)

SpatialMethod = Literal["irls"]
SpatialExecutor = Literal["serial", "dask"]


@dataclass(frozen=True, slots=True)
class StackUnwrapResult:
    """Product of the multi-stage stack unwrap orchestrator.

    Attributes
    ----------
    phase_2d_unw : numpy.ndarray or None
        Pair phases after the 2D spatial stage (or the input when spatial is
        skipped), shape ``(n_pairs, H, W)``.
    connected_components : numpy.ndarray or None
        Spatial component labels aligned with ``phase_2d_unw``. Independent
        islands remain available for per-pixel temporal rank validation.
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
    temporal_iterations : int
        Iterations reported by the temporal solver, or zero when skipped.
    temporal_converged : bool
        Whether every numerically solvable temporal pixel converged.
    temporal_converged_mask : numpy.ndarray or None
        Spatial mask of temporal solutions safe to publish.
    temporal_converged_pixels, temporal_unconverged_pixels : int
        Published and masked solvable-pixel counts.
    temporal_converged_fraction : float
        Fraction of solvable temporal pixels that converged.
    quality_report : StackQualityReport or None
        Independent algebraic quality diagnostics when temporal processing ran.
    inverted : bool
        Whether the batch least-squares inversion stage ran.

    """

    phase_2d_unw: np.ndarray | None
    connected_components: np.ndarray | None
    phase_1d_unw: np.ndarray | None
    corrections_k: np.ndarray | None
    timeseries: np.ndarray | None
    pair_ids: tuple[str, ...]
    method: str
    spatial_method: str
    temporal_applied: bool
    temporal_iterations: int
    temporal_converged: bool
    temporal_converged_mask: np.ndarray | None
    temporal_converged_pixels: int
    temporal_unconverged_pixels: int
    temporal_converged_fraction: float
    quality_report: StackQualityReport | None
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
    spatial_executor: SpatialExecutor = "dask",
    spatial_device: str = "cpu",
    temporal_device: str = "cpu",
    lstsq_device: str = "cpu",
    temporal_kwargs: dict[str, Any] | None = None,
    spatial_kwargs: dict[str, Any] | None = None,
    quality_criteria: StackQualityCriteria | None = None,
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
        :func:`~faninsar.timeseries.results.invert_unwrapped_pairs`
        (batch least-squares -- **not** a 1D unwrap stage).
    spatial_method : {"irls"}, optional
        2D spatial backend.
    spatial_executor : {"serial", "dask"}, optional
        Pair-level scheduler. Dask runs independent full-raster solves in
        parallel without introducing spatial tile seams.
    spatial_device : str, optional
        Torch device for the IRLS numerical kernel.
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
        :func:`~faninsar.processing.unwrapping.api.unwrap` ``irls_kwargs``.
    quality_criteria : StackQualityCriteria, optional
        Explicit campaign quality limits. Exact integer-cycle, finite-mask,
        and published-rank invariants are always checked before inversion.

    Returns
    -------
    StackUnwrapResult
        Stage products and flags.

    Notes
    -----
    No silent method/device downgrade. Spatial IRLS remains global per pair;
    Dask parallelizes pairs rather than cutting a phase field into independently
    referenced tiles.

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
        if spatial_method != "irls":
            reject_invalid_state(
                f"spatial_method must be 'irls', got {spatial_method!r}",
            )
        spatial_kw.setdefault("device", spatial_device)

        def unwrap_pair(index: int) -> tuple[np.ndarray, np.ndarray]:
            result_2d = unwrap(
                phase[index],
                method=spatial_method,
                irls_kwargs=spatial_kw,
            )
            return (
                result_2d.phase.detach().cpu().numpy().astype(np.float64),
                result_2d.component_labels.detach().cpu().numpy().astype(np.int32),
            )

        if spatial_executor == "dask" and n_pairs > 1:
            import dask

            tasks = [dask.delayed(unwrap_pair)(index) for index in range(n_pairs)]
            gpu_device = spatial_device in ("cuda", "mps")
            workers = 1 if gpu_device else min(n_pairs, 8)
            unwrapped_results = list(
                dask.compute(*tasks, scheduler="threads", num_workers=workers)
            )
        else:
            unwrapped_results = [unwrap_pair(index) for index in range(n_pairs)]
        phase_2d = np.stack([item[0] for item in unwrapped_results], axis=0)
        component_stack = np.stack(
            [item[1] for item in unwrapped_results],
            axis=0,
        )
        logger.info(
            "Spatial unwrap (%s) finished for %s pairs on %sx%s",
            spatial_method,
            n_pairs,
            height,
            width,
        )
    else:
        phase_2d = phase.copy()
        component_stack = np.where(np.isfinite(phase_2d), 1, 0).astype(np.int32)

    # --- Stage 2: 1D temporal / network IRLS ---
    phase_1d: np.ndarray | None = None
    corrections_k: np.ndarray | None = None
    temporal_applied = False
    temporal_iterations = 0
    temporal_converged = False
    temporal_converged_mask: np.ndarray | None = None
    temporal_converged_pixels = 0
    temporal_unconverged_pixels = 0
    temporal_converged_fraction = 0.0
    quality_report: StackQualityReport | None = None
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
        temporal_iterations = temporal_result.iterations
        temporal_converged = temporal_result.converged
        temporal_converged_mask = temporal_result.converged_mask
        temporal_converged_pixels = temporal_result.converged_pixels
        temporal_unconverged_pixels = temporal_result.unconverged_pixels
        temporal_converged_fraction = temporal_result.converged_fraction
        logger.info(
            "Temporal IRLS finished (iter=%s, converged=%s, fraction=%.6f, device=%s)",
            temporal_result.iterations,
            temporal_result.converged,
            temporal_result.converged_fraction,
            temporal_result.device,
        )
        quality_report = evaluate_stack_quality(
            phase_2d,
            temporal_result.phase_unw,
            temporal_result.corrections_k,
            pair_dates,
            converged_mask=temporal_result.converged_mask,
            criteria=quality_criteria,
        )
        if not quality_report.passed:
            reject_invalid_state(
                "temporal Stack quality gate failed: "
                + "; ".join(quality_report.failures)
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
        connected_components=component_stack,
        phase_1d_unw=phase_1d,
        corrections_k=corrections_k,
        timeseries=timeseries,
        pair_ids=pair_ids,
        method="stack",
        spatial_method=spatial_method,
        temporal_applied=temporal_applied,
        temporal_iterations=temporal_iterations,
        temporal_converged=temporal_converged,
        temporal_converged_mask=temporal_converged_mask,
        temporal_converged_pixels=temporal_converged_pixels,
        temporal_unconverged_pixels=temporal_unconverged_pixels,
        temporal_converged_fraction=temporal_converged_fraction,
        quality_report=quality_report,
        inverted=inverted,
    )
