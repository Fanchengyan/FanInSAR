"""Tests for canonical v2 convergence-boundary evaluation."""

from __future__ import annotations

import numpy as np

from faninsar.processing.geometry import (
    BoundaryDecision,
    TransformResultV2,
    evaluate_canonical_boundary,
    normalize_result_boundary,
)


def test_boundary_band_recomputes_strictly_below_equal_and_above() -> None:
    """A current-attempt residual is recomputed inside the inclusive ULP band."""
    eps_band = 32 * np.finfo(np.float64).eps
    seen: list[bool] = []

    def callback(latitude: np.ndarray) -> tuple[float, float]:
        seen.append(not latitude.flags.writeable)
        return latitude.item(), 0.0

    for residual, expected in (
        (1.0 - eps_band, True),
        (1.0, False),
        (1.0 + eps_band, False),
    ):
        decision = evaluate_canonical_boundary(
            "geo2rdr",
            4,
            (np.array([residual], dtype=np.float64),),
            callback,
            decision_residual=(residual, 0.0),
            range_tolerance_m=1.0,
            doppler_tolerance_hz=1.0,
        )
        assert decision.converged is expected
        assert decision.boundary_rechecked
    assert seen == [True, True, True]


def test_boundary_outside_band_preserves_original_decision() -> None:
    """Far from the boundary, no callback or residual replacement occurs."""
    called = False

    def callback(*_: np.ndarray) -> float:
        nonlocal called
        called = True
        return 0.0

    decision = evaluate_canonical_boundary(
        "rdr2geo",
        2,
        (np.array([1.0], dtype=np.float64),),
        callback,
        decision_residual=0.25,
        slant_range_tolerance_m=1.0,
    )
    assert decision.decision_residual == 0.25
    assert not decision.boundary_rechecked
    assert not called


def test_boundary_only_owns_current_attempt() -> None:
    """The narrow API calls the callback once and never revisits old attempts."""
    calls = 0

    def callback(*_: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return 0.5

    decision = evaluate_canonical_boundary(
        "rdr2geo",
        7,
        (np.array([1.0], dtype=np.float64),),
        callback,
        decision_residual=1.0,
        slant_range_tolerance_m=1.0,
    )
    assert decision.converged
    assert decision.boundary_rechecked
    assert calls == 1


def test_invalid_boundary_and_result_normalization_keep_sentinels() -> None:
    """Invalid coordinates do not recheck and normalization restores sentinels."""
    decision = evaluate_canonical_boundary(
        "geo2rdr",
        1,
        (np.array([np.nan], dtype=np.float64),),
        lambda *_: (0.0, 0.0),
        decision_residual=(1.0, 0.0),
        range_tolerance_m=1.0,
        doppler_tolerance_hz=1.0,
    )
    assert not decision.converged
    assert not decision.boundary_rechecked
    assert np.isnan(decision.decision_residual)
    assert np.isnan(decision.normalized_q)
    result = TransformResultV2.invalid(1)
    normalized = normalize_result_boundary(
        result,
        BoundaryDecision(True, True, 0.0, 0.0),
        invalid_mask=np.array([True], dtype=bool),
    )
    assert normalized.iterations[0] == -1
    assert not normalized.converged[0]
    assert not normalized.boundary_rechecked[0]
    assert np.isnan(normalized.tolerance[0])
