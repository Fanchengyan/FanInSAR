"""Tests for date-level ionosphere network inversion (PROPOSAL-0036)."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.atmosphere import invert_ionosphere_network


def _grid(value: float) -> np.ndarray:
    return np.full((4, 5), value, dtype=np.float64)


def _triangle() -> dict[str, np.ndarray]:
    """Three-date triangle with true screens a=1, b=-2, c=3 rad."""
    return {
        "a_b": _grid(3.0),
        "b_c": _grid(-5.0),
        "a_c": _grid(-2.0),
    }


def test_triangle_recovers_reference_relative_screens() -> None:
    result = invert_ionosphere_network(_triangle(), screening_iterations=0)
    assert result.dates == ("a", "b", "c")
    assert result.reference_date == "a"
    assert result.observation_rank == 2
    # Screens are relative to the reference date (zro_date convention).
    np.testing.assert_allclose(result.screens[:, 0, 0], [0.0, -3.0, 2.0], atol=1e-12)
    assert result.screened_pair_ids == ()


def test_explicit_reference_date_pinned_to_zero() -> None:
    result = invert_ionosphere_network(
        _triangle(), reference_date="b", screening_iterations=0
    )
    assert result.reference_date == "b"
    np.testing.assert_allclose(result.screens[:, 0, 0], [3.0, 0.0, 5.0], atol=1e-12)


def test_window_weighting_pulls_solution_toward_confident_pairs() -> None:
    confident = {"a_b": _grid(100.0), "b_c": _grid(1.0), "a_c": _grid(1.0)}
    screens = {"a_b": _grid(4.0), "b_c": _grid(-5.0), "a_c": _grid(-2.0)}
    chain = invert_ionosphere_network(
        {"b_c": _grid(-5.0), "a_c": _grid(-2.0)}, screening_iterations=0
    )
    unweighted = invert_ionosphere_network(screens, screening_iterations=0)
    weighted = invert_ionosphere_network(screens, confident, screening_iterations=0)
    assert weighted.screened_pair_ids == ()
    assert np.abs(weighted.screens - chain.screens).max() < np.abs(
        unweighted.screens - chain.screens
    ).max()


def test_zero_or_nan_window_pixels_exclude_the_pair_at_that_pixel() -> None:
    windows = {"a_b": _grid(1.0), "b_c": _grid(0.0), "a_c": _grid(1.0)}
    result = invert_ionosphere_network(_triangle(), windows, screening_iterations=0)
    chain = invert_ionosphere_network(
        {"a_b": _grid(3.0), "a_c": _grid(-2.0)}, screening_iterations=0
    )
    np.testing.assert_allclose(result.screens, chain.screens, atol=1e-12)


def test_irls_screening_downweights_inconsistent_pair() -> None:
    screens = {
        "a_b": _grid(3.0),
        "b_c": _grid(-5.0),
        "a_c": _grid(-2.0),
        "c_d": _grid(7.0),
        "a_d": _grid(5.0),
    }
    truth = np.array([0.0, -3.0, 2.0, -5.0])
    clean = invert_ionosphere_network(screens, screening_iterations=3)
    assert clean.screened_pair_ids == ()
    np.testing.assert_allclose(clean.screens[:, 0, 0], truth, atol=1e-12)

    corrupted = dict(screens)
    corrupted["a_c"] = _grid(3.0)
    biased = invert_ionosphere_network(corrupted, screening_iterations=0)
    assert np.abs(biased.screens[:, 0, 0] - truth).max() > 0.5
    robust = invert_ionosphere_network(corrupted, screening_iterations=6)
    assert "a_c" in robust.screened_pair_ids
    np.testing.assert_allclose(robust.screens[:, 0, 0], truth, atol=1e-6)


def test_excluded_dates_drop_every_touching_pair() -> None:
    result = invert_ionosphere_network(_triangle(), excluded_dates=["c"])
    assert result.dates == ("a", "b")
    assert result.used_pair_ids == ("a_b",)


def test_excluded_pairs_leave_a_connected_chain() -> None:
    result = invert_ionosphere_network(_triangle(), excluded_pairs=["a_c"])
    assert result.used_pair_ids == ("a_b", "b_c")
    np.testing.assert_allclose(result.screens[:, 0, 0], [0.0, -3.0, 2.0], atol=1e-12)


def test_unknown_exclusions_fail_closed() -> None:
    with pytest.raises(ValueError, match="not part of the network"):
        invert_ionosphere_network(_triangle(), excluded_pairs=["zz"])
    with pytest.raises(ValueError, match="not part of the network"):
        invert_ionosphere_network(_triangle(), excluded_dates=["zz"])


def test_disconnected_network_fails_closed() -> None:
    with pytest.raises(ValueError, match="not fully connected"):
        invert_ionosphere_network(
            {"a_b": _grid(3.0), "c_d": _grid(11.0)}, screening_iterations=0
        )


def test_unknown_reference_date_fails_closed() -> None:
    with pytest.raises(ValueError, match="reference date"):
        invert_ionosphere_network(_triangle(), reference_date="zz")


def test_rank_deficient_pixels_are_nan_while_healthy_pixels_solve() -> None:
    screens = {key: value.copy() for key, value in _triangle().items()}
    screens["a_b"][:, 0] = np.nan
    screens["b_c"][:, 0] = np.nan
    result = invert_ionosphere_network(screens, screening_iterations=0)
    # Date b has no valid observation at column 0: the whole pixel is
    # conservatively reported as NaN (documented contract).
    assert np.isnan(result.screens[1:, 0, 0]).all()
    assert result.screens[0, 0, 0] == 0.0
    assert np.isfinite(result.screens[:, 1, 1]).all()


def test_fully_invalid_pixels_never_crash_the_solve() -> None:
    screens = {key: value.copy() for key, value in _triangle().items()}
    for value in screens.values():
        value[:, 0] = np.nan
    result = invert_ionosphere_network(screens, screening_iterations=0)
    assert np.isnan(result.screens[1:, 0, 0]).all()
    assert np.isfinite(result.screens[:, 1, 1]).all()


def test_empty_and_malformed_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least one pair"):
        invert_ionosphere_network({})
    with pytest.raises(ValueError, match="sharing one shape"):
        invert_ionosphere_network({"a_b": _grid(1.0), "b_c": np.ones((3, 5))})
    with pytest.raises(ValueError, match="primary_secondary"):
        invert_ionosphere_network({"abc": _grid(1.0)})
