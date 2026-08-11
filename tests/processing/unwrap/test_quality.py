"""Tests for independent temporal-network and SBAS quality diagnostics."""

from __future__ import annotations

import numpy as np

from faninsar.processing.unwrap.quality import (
    StackQualityCriteria,
    evaluate_stack_quality,
)

PAIRS = [
    ("20160101", "20160113"),
    ("20160113", "20160125"),
    ("20160101", "20160125"),
]


def test_quality_report_proves_exact_planted_network_solution() -> None:
    """Exact integer correction closes the network and leaves zero residual."""
    true_pair_phase = np.array([0.4, 0.7, 1.1], dtype=np.float64)[:, None, None]
    observed = true_pair_phase.copy()
    observed[0] += 2.0 * np.pi
    corrections = np.array([-1.0, 0.0, 0.0])[:, None, None]

    report = evaluate_stack_quality(
        observed,
        true_pair_phase,
        corrections,
        PAIRS,
        converged_mask=np.ones((1, 1), dtype=bool),
    )

    assert report.passed
    assert report.failures == ()
    assert report.converged_fraction == 1.0
    assert report.rank_coverage_fraction == 1.0
    assert report.published_full_rank_fraction == 1.0
    assert report.modulo_closure_abs_rad.count == 1
    assert report.modulo_closure_abs_rad.maximum < 1e-12
    assert report.sbas_residual_abs_rad.maximum < 1e-12
    assert report.integer_correction_max_error == 0.0
    assert report.phase_reconstruction_max_error_rad < 1e-12


def test_multilook_closure_is_diagnostic_only() -> None:
    """Non-zero multilook closure is reported without blocking publication."""
    phase = np.array([1.0, 2.0, 3.3], dtype=np.float64)[:, None, None]
    corrections = np.zeros_like(phase)

    diagnostic_only = evaluate_stack_quality(
        phase,
        phase,
        corrections,
        PAIRS,
        converged_mask=np.ones((1, 1), dtype=bool),
    )
    assert diagnostic_only.passed
    np.testing.assert_allclose(
        diagnostic_only.modulo_closure_abs_rad.p95,
        0.3,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        diagnostic_only.sbas_residual_abs_rad.p95,
        0.1,
        atol=1e-12,
    )
    assert diagnostic_only.failures == ()


def test_quality_rejects_published_rank_deficient_pixel() -> None:
    """A published pixel must independently span every temporal interval."""
    phase = np.array(
        [
            [[1.0, 1.0]],
            [[2.0, np.nan]],
            [[3.0, np.nan]],
        ],
        dtype=np.float64,
    )
    corrections = np.where(np.isfinite(phase), 0.0, np.nan)

    report = evaluate_stack_quality(
        phase,
        phase,
        corrections,
        PAIRS,
        converged_mask=np.ones((1, 2), dtype=bool),
    )

    assert not report.passed
    assert report.observed_pixels == 2
    assert report.full_rank_pixels == 1
    assert report.rank_coverage_fraction == 0.5
    assert report.published_full_rank_fraction == 0.5
    assert any("rank deficient" in failure for failure in report.failures)


def test_quality_configures_minimum_converged_and_rank_coverage() -> None:
    """Configured coverage requirements reject an otherwise safe masked subset."""
    phase = np.broadcast_to(
        np.array([1.0, 2.0, 3.0])[:, None, None],
        (3, 1, 2),
    ).copy()
    output = phase.copy()
    output[:, :, 1] = np.nan
    corrections = np.zeros_like(phase)
    corrections[:, :, 1] = np.nan

    report = evaluate_stack_quality(
        phase,
        output,
        corrections,
        PAIRS,
        converged_mask=np.array([[True, False]]),
        criteria=StackQualityCriteria(
            min_converged_fraction=0.75,
            min_rank_coverage_fraction=1.0,
        ),
    )

    assert not report.passed
    assert report.converged_fraction == 0.5
    assert report.rank_coverage_fraction == 1.0
    assert any("converged fraction" in failure for failure in report.failures)
