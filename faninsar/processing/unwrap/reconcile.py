"""Connected-component integer-cycle reconciliation and closure QA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class ComponentCorrection:
    """Integer-cycle correction applied to one connected component."""

    component_id: int
    cycles: int
    confidence: float


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Corrected unwrapped phase and diagnostics."""

    corrected_phase: np.ndarray
    corrections: tuple[ComponentCorrection, ...]
    before_closure_rad: float
    after_closure_rad: float
    unresolved: bool


def align_components_to_reference(
    unwrapped_phase: np.ndarray,
    connected_components: np.ndarray,
    *,
    reference_component: int | None = None,
    min_pixels: int = 16,
) -> tuple[np.ndarray, tuple[ComponentCorrection, ...]]:
    """Align component medians to a reference component by integer cycles.

    Parameters
    ----------
    unwrapped_phase : numpy.ndarray
        Unwrapped phase in radians.
    connected_components : numpy.ndarray
        Integer labels; 0 is treated as invalid/background.
    reference_component : int, optional
        Component used as the absolute reference. Defaults to the largest.
    min_pixels : int, optional
        Ignore components smaller than this size.

    Returns
    -------
    tuple
        Corrected phase and the applied component corrections.

    """
    phase = np.asarray(unwrapped_phase, dtype=np.float64)
    labels = np.asarray(connected_components, dtype=np.int32)
    if phase.shape != labels.shape:
        reject_invalid_state("phase and components must share a shape")

    unique = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    if not unique:
        return phase.astype(np.float32), ()

    sizes = {cid: int(np.count_nonzero(labels == cid)) for cid in unique}
    if reference_component is None:
        reference_component = max(sizes, key=sizes.get)
    if reference_component not in sizes:
        reject_invalid_state(
            f"reference component {reference_component} is not present"
        )

    ref_mask = labels == reference_component
    ref_median = float(np.median(phase[ref_mask]))
    corrected = phase.copy()
    corrections: list[ComponentCorrection] = []

    for cid in unique:
        if cid == reference_component or sizes[cid] < min_pixels:
            continue
        mask = labels == cid
        median = float(np.median(phase[mask]))
        delta = ref_median - median
        cycles = int(np.rint(delta / (2.0 * np.pi)))
        if cycles == 0:
            continue
        corrected[mask] = phase[mask] + cycles * 2.0 * np.pi
        confidence = min(1.0, sizes[cid] / max(sizes[reference_component], 1))
        corrections.append(
            ComponentCorrection(
                component_id=cid,
                cycles=cycles,
                confidence=float(confidence),
            )
        )
        logger.info(
            "Aligned component %s by %s cycles (confidence=%.3f)",
            cid,
            cycles,
            confidence,
        )
    return corrected.astype(np.float32), tuple(corrections)


def loop_closure_phase(
    pair_phases: dict[str, np.ndarray],
    loop: tuple[str, str, str],
    *,
    masks: dict[str, np.ndarray] | None = None,
) -> float:
    """Compute mean absolute loop-closure residual for a three-pair loop.

    Parameters
    ----------
    pair_phases : dict[str, numpy.ndarray]
        Mapping of pair id to unwrapped phase arrays on a common grid.
    loop : tuple[str, str, str]
        Ordered pair ids ``(ab, bc, ac)`` such that closure is
        ``ab + bc - ac``.
    masks : dict, optional
        Optional boolean masks per pair; the intersection is used.

    Returns
    -------
    float
        Mean absolute closure residual in radians.

    """
    ab_id, bc_id, ac_id = loop
    for pair_id in loop:
        if pair_id not in pair_phases:
            reject_invalid_state(f"missing pair phase for loop member {pair_id}")
    ab = np.asarray(pair_phases[ab_id], dtype=np.float64)
    bc = np.asarray(pair_phases[bc_id], dtype=np.float64)
    ac = np.asarray(pair_phases[ac_id], dtype=np.float64)
    if ab.shape != bc.shape or ab.shape != ac.shape:
        reject_invalid_state("loop pair phases must share a common grid")

    valid = np.ones(ab.shape, dtype=bool)
    if masks is not None:
        for pair_id in loop:
            if pair_id in masks:
                valid &= np.asarray(masks[pair_id], dtype=bool)
    if not np.any(valid):
        return float("nan")
    closure = ab + bc - ac
    return float(np.mean(np.abs(closure[valid])))


def reconcile_components(
    unwrapped_phase: np.ndarray,
    connected_components: np.ndarray,
    *,
    pair_phases_for_closure: dict[str, np.ndarray] | None = None,
    loop: tuple[str, str, str] | None = None,
    confidence_threshold: float = 0.05,
) -> ReconciliationResult:
    """Reconcile integer-cycle component offsets and optional loop closure.

    Parameters
    ----------
    unwrapped_phase, connected_components : numpy.ndarray
        Single-pair unwrap product.
    pair_phases_for_closure : dict, optional
        Optional multi-pair phases for closure diagnostics.
    loop : tuple, optional
        Three-pair loop identity for closure comparison.
    confidence_threshold : float, optional
        Corrections with confidence below this threshold are skipped.

    Returns
    -------
    ReconciliationResult
        Corrected phase, applied corrections, and closure metrics.

    """
    corrected, corrections = align_components_to_reference(
        unwrapped_phase,
        connected_components,
    )
    kept = tuple(c for c in corrections if c.confidence >= confidence_threshold)
    if len(kept) != len(corrections):
        # re-apply only confident corrections from the original phase
        corrected, _ = align_components_to_reference(
            unwrapped_phase,
            connected_components,
        )
        # filter by rebuilding from original with only kept cycles
        phase = np.asarray(unwrapped_phase, dtype=np.float64)
        labels = np.asarray(connected_components, dtype=np.int32)
        corrected = phase.copy()
        for item in kept:
            corrected[labels == item.component_id] = (
                phase[labels == item.component_id] + item.cycles * 2.0 * np.pi
            )
        corrected = corrected.astype(np.float32)

    before = float("nan")
    after = float("nan")
    unresolved = False
    if pair_phases_for_closure is not None and loop is not None:
        before = loop_closure_phase(pair_phases_for_closure, loop)
        updated = dict(pair_phases_for_closure)
        # assume first loop member is the corrected pair when present
        if loop[0] in updated:
            updated[loop[0]] = corrected
        after = loop_closure_phase(updated, loop)
        unresolved = bool(np.isfinite(after) and after > before + 1e-6)

    return ReconciliationResult(
        corrected_phase=corrected,
        corrections=kept,
        before_closure_rad=before,
        after_closure_rad=after,
        unresolved=unresolved,
    )
