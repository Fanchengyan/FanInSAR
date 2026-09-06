"""Independent algebraic quality diagnostics for temporal Stack products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.unwrapping._incidence import build_incidence_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar import Pairs

_TWO_PI = 2.0 * np.pi
_ALGEBRA_TOLERANCE = 1.0e-8


@dataclass(frozen=True, slots=True)
class MetricDistribution:
    """Finite-sample absolute-value distribution.

    Attributes
    ----------
    count : int
        Number of finite scalar samples.
    median, p95, maximum : float or None
        Distribution statistics, or ``None`` when no sample exists.

    """

    count: int
    median: float | None
    p95: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class StackQualityCriteria:
    """Optional coverage limits for a Stack product.

    Temporal loop closure and least-squares residuals are deliberately
    diagnostics only.  They are not universal correctness criteria for
    multilooked InSAR products: multilooking, pair-dependent weighting and
    phase unwrapping can produce physically meaningful non-zero residuals.
    Product qualification is therefore performed against the ISCE2 oracle
    and explicit rank/coverage requirements, not against a zero-closure
    assumption.

    Attributes
    ----------
    min_converged_fraction, min_rank_coverage_fraction : float or None
        Optional minimum fractions in the closed interval ``[0, 1]``.

    """

    min_converged_fraction: float | None = None
    min_rank_coverage_fraction: float | None = None

    def __post_init__(self) -> None:
        """Validate configured quality limits."""
        for name in ("min_converged_fraction", "min_rank_coverage_fraction"):
            value = getattr(self, name)
            if value is not None and (
                not np.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                reject_invalid_state(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class StackQualityReport:
    """Independent temporal convergence, diagnostic, and rank report.

    ``modulo_closure_abs_rad`` and ``sbas_residual_abs_rad`` are retained as
    observational fields for scientific reporting.  Neither field affects
    :attr:`passed`; ISCE2 parity and explicit coverage/rank policy determine
    qualification.
    """

    passed: bool
    failures: tuple[str, ...]
    observed_pixels: int
    full_rank_pixels: int
    rank_coverage_fraction: float
    published_pixels: int
    published_full_rank_fraction: float
    converged_fraction: float
    cycle_count: int
    modulo_closure_abs_rad: MetricDistribution
    sbas_residual_abs_rad: MetricDistribution
    integer_correction_max_error: float | None
    phase_reconstruction_max_error_rad: float | None


def _distribution(values: np.ndarray) -> MetricDistribution:
    """Summarize finite absolute values without inventing missing samples."""
    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return MetricDistribution(count=0, median=None, p95=None, maximum=None)
    return MetricDistribution(
        count=int(finite.size),
        median=float(np.median(finite)),
        p95=float(np.percentile(finite, 95.0)),
        maximum=float(np.max(finite)),
    )


def _fundamental_cycle_matrix(design_matrix: np.ndarray) -> np.ndarray:
    """Return an integer fundamental-cycle basis for an SBAS design matrix."""
    n_pairs, n_intervals = design_matrix.shape
    basis_indices: list[int] = []
    rank = 0
    for index in range(n_pairs):
        candidate = design_matrix[[*basis_indices, index]]
        candidate_rank = int(np.linalg.matrix_rank(candidate))
        if candidate_rank > rank:
            basis_indices.append(index)
            rank = candidate_rank
        if rank == n_intervals:
            break
    if rank != n_intervals:
        reject_invalid_state("pair network is disconnected or rank deficient")

    basis = design_matrix[basis_indices]
    cycle_rows: list[np.ndarray] = []
    for index in range(n_pairs):
        if index in basis_indices:
            continue
        coefficients = np.linalg.solve(basis.T, design_matrix[index])
        rounded = np.rint(coefficients)
        if not np.allclose(coefficients, rounded, atol=_ALGEBRA_TOLERANCE, rtol=0.0):
            reject_invalid_state("pair network did not yield an integer cycle basis")
        cycle = np.zeros(n_pairs, dtype=np.float64)
        cycle[index] = 1.0
        cycle[basis_indices] = -rounded
        cycle_rows.append(cycle)
    if not cycle_rows:
        return np.empty((0, n_pairs), dtype=np.float64)
    return np.stack(cycle_rows, axis=0)


def _rank_mask(
    finite_pairs_by_pixel: np.ndarray,
    design_matrix: np.ndarray,
) -> np.ndarray:
    """Return pixels whose available pair rows span all date intervals."""
    full_rank = np.zeros(finite_pairs_by_pixel.shape[1], dtype=bool)
    unique_masks, inverse = np.unique(
        finite_pairs_by_pixel.T,
        axis=0,
        return_inverse=True,
    )
    required_rank = design_matrix.shape[1]
    for mask_index, finite_pairs in enumerate(unique_masks):
        if np.count_nonzero(finite_pairs) < required_rank:
            continue
        if np.linalg.matrix_rank(design_matrix[finite_pairs]) == required_rank:
            full_rank[inverse == mask_index] = True
    return full_rank


def _sbas_residuals(
    phase_flat: np.ndarray,
    design_matrix: np.ndarray,
    published_mask: np.ndarray,
) -> np.ndarray:
    """Compute least-squares residual samples for each finite subnetwork."""
    finite_pairs_by_pixel = np.isfinite(phase_flat)
    residual_samples: list[np.ndarray] = []
    selected = np.flatnonzero(published_mask)
    if selected.size == 0:
        return np.empty(0, dtype=np.float64)
    unique_masks, inverse = np.unique(
        finite_pairs_by_pixel[:, selected].T,
        axis=0,
        return_inverse=True,
    )
    for mask_index, finite_pairs in enumerate(unique_masks):
        pixel_indices = selected[inverse == mask_index]
        local_design = design_matrix[finite_pairs]
        observations = phase_flat[np.ix_(finite_pairs, pixel_indices)]
        solution = np.linalg.lstsq(local_design, observations, rcond=None)[0]
        residual_samples.append((observations - local_design @ solution).ravel())
    return np.concatenate(residual_samples) if residual_samples else np.empty(0)


def evaluate_stack_quality(
    phase_input: np.ndarray,
    phase_output: np.ndarray,
    corrections_k: np.ndarray,
    pair_dates: Pairs | Sequence[tuple[str, str]] | np.ndarray,
    *,
    converged_mask: np.ndarray,
    criteria: StackQualityCriteria | None = None,
) -> StackQualityReport:
    """Evaluate a temporal Stack product's structural invariants.

    Parameters
    ----------
    phase_input, phase_output, corrections_k : numpy.ndarray
        Input pair phase, published pair phase, and integer-cycle correction
        stacks with shape ``(n_pairs, ...)``.
    pair_dates : Pairs or sequence of pair dates or numpy.ndarray
        Pair network aligned with stack axis zero.
    converged_mask : numpy.ndarray
        Spatial mask identifying pixels selected for publication.
    criteria : StackQualityCriteria, optional
        Explicit rank/coverage limits. Multilook loop closure and SBAS
        residuals are reported for diagnostics only and are never used as
        universal publication gates.

    Returns
    -------
    StackQualityReport
        Immutable diagnostics and every failed criterion.

    Notes
    -----
    Modulo closure is evaluated on an integer fundamental-cycle basis as a
    diagnostic, ``angle(exp(1j * C @ phase))``. SBAS residuals are independently
    recomputed by least squares on every published pixel's finite full-rank
    subnetwork. Neither diagnostic is interpreted as a zero-error physical
    requirement after multilooking.

    """
    source = np.asarray(phase_input, dtype=np.float64)
    output = np.asarray(phase_output, dtype=np.float64)
    corrections = np.asarray(corrections_k, dtype=np.float64)
    mask = np.asarray(converged_mask, dtype=bool)
    if (
        source.ndim < 2
        or output.shape != source.shape
        or corrections.shape != source.shape
    ):
        reject_invalid_state(
            "phase and correction stacks must share shape (n_pairs, ...)"
        )
    if mask.shape != source.shape[1:]:
        reject_invalid_state("converged_mask must match the phase spatial shape")

    design_matrix, _pair_ids = build_incidence_matrix(pair_dates)
    if design_matrix.shape[0] != source.shape[0]:
        reject_invalid_state("pair_dates must align with phase stack axis zero")
    if np.linalg.matrix_rank(design_matrix) != design_matrix.shape[1]:
        reject_invalid_state("pair network is disconnected or rank deficient")

    n_pairs = source.shape[0]
    source_flat = source.reshape(n_pairs, -1)
    output_flat = output.reshape(n_pairs, -1)
    corrections_flat = corrections.reshape(n_pairs, -1)
    published = mask.reshape(-1)
    finite_source = np.isfinite(source_flat)
    finite_output = np.isfinite(output_flat)
    finite_corrections = np.isfinite(corrections_flat)
    observed = finite_source.any(axis=0)
    full_rank = _rank_mask(finite_source, design_matrix)

    observed_pixels = int(np.count_nonzero(observed))
    full_rank_pixels = int(np.count_nonzero(full_rank))
    published_pixels = int(np.count_nonzero(published))
    rank_coverage = full_rank_pixels / observed_pixels if observed_pixels else 0.0
    published_full_rank = int(np.count_nonzero(published & full_rank))
    published_rank_fraction = (
        published_full_rank / published_pixels if published_pixels else 0.0
    )
    converged_fraction = (
        published_pixels / full_rank_pixels if full_rank_pixels else 0.0
    )

    failures: list[str] = []
    if published_pixels == 0:
        failures.append("no converged pixels were selected for publication")
    if np.any(published & ~full_rank):
        failures.append("one or more published temporal pixels are rank deficient")
    expected_finite = finite_source & published[None, :]
    if not np.array_equal(finite_output, expected_finite):
        failures.append(
            "published phase finite mask does not match source observations"
        )
    if not np.array_equal(finite_corrections, expected_finite):
        failures.append(
            "integer-correction finite mask does not match source observations"
        )

    valid_values = expected_finite & finite_output & finite_corrections
    correction_values = corrections_flat[valid_values]
    if correction_values.size:
        integer_error = float(
            np.max(np.abs(correction_values - np.rint(correction_values)))
        )
        reconstruction_error = float(
            np.max(
                np.abs(
                    output_flat[valid_values]
                    - source_flat[valid_values]
                    - _TWO_PI * corrections_flat[valid_values]
                )
            )
        )
    else:
        integer_error = None
        reconstruction_error = None
    if integer_error is not None and integer_error > _ALGEBRA_TOLERANCE:
        failures.append("published temporal corrections are not integer cycle counts")
    if reconstruction_error is not None and reconstruction_error > _ALGEBRA_TOLERANCE:
        failures.append("published phase does not equal input plus 2pi corrections")

    cycle_matrix = _fundamental_cycle_matrix(design_matrix)
    closure_samples: list[np.ndarray] = []
    for cycle in cycle_matrix:
        cycle_pairs = np.flatnonzero(cycle)
        cycle_valid = published & finite_output[cycle_pairs].all(axis=0)
        if np.any(cycle_valid):
            closure = cycle @ output_flat[:, cycle_valid]
            closure_samples.append(np.arctan2(np.sin(closure), np.cos(closure)))
    modulo_closure = _distribution(
        np.concatenate(closure_samples) if closure_samples else np.empty(0)
    )
    sbas_residual = _distribution(
        _sbas_residuals(output_flat, design_matrix, published & full_rank)
    )

    limits = criteria or StackQualityCriteria()
    if (
        limits.min_converged_fraction is not None
        and converged_fraction < limits.min_converged_fraction
    ):
        failures.append(
            "converged fraction "
            f"{converged_fraction:.6f} is below "
            f"{limits.min_converged_fraction:.6f}"
        )
    if (
        limits.min_rank_coverage_fraction is not None
        and rank_coverage < limits.min_rank_coverage_fraction
    ):
        failures.append(
            "rank coverage fraction "
            f"{rank_coverage:.6f} is below "
            f"{limits.min_rank_coverage_fraction:.6f}"
        )
    return StackQualityReport(
        passed=not failures,
        failures=tuple(failures),
        observed_pixels=observed_pixels,
        full_rank_pixels=full_rank_pixels,
        rank_coverage_fraction=rank_coverage,
        published_pixels=published_pixels,
        published_full_rank_fraction=published_rank_fraction,
        converged_fraction=converged_fraction,
        cycle_count=int(cycle_matrix.shape[0]),
        modulo_closure_abs_rad=modulo_closure,
        sbas_residual_abs_rad=sbas_residual,
        integer_correction_max_error=integer_error,
        phase_reconstruction_max_error_rad=reconstruction_error,
    )
