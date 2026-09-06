"""Tests for 1D temporal/network IRLS phase unwrapping."""

from __future__ import annotations

import numpy as np

from faninsar.processing.unwrapping.temporal_irls import unwrap_temporal_irls


def _three_date_pairs() -> list[tuple[str, str]]:
    """Return a redundant 3-date / 3-pair network."""
    return [
        ("20160101", "20160113"),
        ("20160113", "20160125"),
        ("20160101", "20160125"),
    ]


def test_temporal_irls_recovers_planted_2pi() -> None:
    """Planted integer 2π jump that breaks loop closure is recovered."""
    pairs = _three_date_pairs()
    # True interval phases (2 intervals for 3 dates)
    true_x = np.array([0.4, 0.7], dtype=np.float64)
    # Incidence: pair0 spans int0, pair1 spans int1, pair2 spans both
    a_mat = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    true_phi = a_mat @ true_x  # (3,)
    # Single-pair jump breaks closure (cancelling jumps would stay consistent)
    k_plant = np.array([1, 0, 0], dtype=np.int64)
    phi_obs = true_phi + 2.0 * np.pi * k_plant
    # Broadcast to a small spatial grid
    phase_stack = np.broadcast_to(phi_obs[:, None, None], (3, 4, 5)).copy()

    result = unwrap_temporal_irls(
        phase_stack,
        pairs,
        wrapped_input=True,
        max_iter=20,
        tol=1e-4,
        device="cpu",
    )

    assert result.phase_unw.shape == phase_stack.shape
    assert result.method == "temporal_irls"
    recovered = result.phase_unw[:, 0, 0]
    np.testing.assert_allclose(recovered, true_phi, atol=0.05)
    np.testing.assert_array_equal(
        result.corrections_k[:, 0, 0].astype(np.int64),
        -k_plant,
    )


def test_temporal_irls_singular_pixel_returns_nan() -> None:
    """Zero-weight / all-NaN pixel yields NaN output without crashing."""
    pairs = _three_date_pairs()
    # NaN observations → zero weight → singular normal equations per pixel
    phase_stack = np.full((3, 1, 1), np.nan, dtype=np.float64)

    result = unwrap_temporal_irls(
        phase_stack,
        pairs,
        wrapped_input=False,
        device="cpu",
    )

    assert result.phase_unw.shape == (3, 1, 1)
    assert np.isnan(result.phase_unw).all()


def test_temporal_irls_shape_contract() -> None:
    """Input (n_pairs, H, W) maps to output of the same shape."""
    pairs = _three_date_pairs()
    rng = np.random.default_rng(0)
    phase_stack = rng.uniform(-np.pi, np.pi, size=(3, 6, 7)).astype(np.float64)

    result = unwrap_temporal_irls(
        phase_stack,
        pairs,
        wrapped_input=True,
        device="cpu",
    )

    assert result.phase_unw.shape == (3, 6, 7)
    assert result.corrections_k.shape == (3, 6, 7)
    assert result.iterations >= 1
    assert result.device == "cpu"


def test_temporal_irls_wrapped_input_false_passthrough() -> None:
    """Already-consistent unwrapped input yields near-zero integer corrections."""
    pairs = _three_date_pairs()
    true_x = np.array([0.25, -0.15], dtype=np.float64)
    a_mat = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    true_phi = a_mat @ true_x
    phase_stack = np.broadcast_to(true_phi[:, None, None], (3, 3, 3)).copy()

    result = unwrap_temporal_irls(
        phase_stack,
        pairs,
        wrapped_input=False,
        max_iter=10,
        device="cpu",
    )

    np.testing.assert_array_equal(result.corrections_k, 0)
    np.testing.assert_allclose(result.phase_unw, phase_stack, atol=1e-5)


def test_temporal_irls_masks_pixels_that_do_not_converge() -> None:
    """A bounded solve never publishes pixels without convergence evidence."""
    pairs = _three_date_pairs()
    phase_stack = np.full((3, 2, 2), 0.25, dtype=np.float64)

    result = unwrap_temporal_irls(
        phase_stack,
        pairs,
        wrapped_input=False,
        max_iter=1,
        device="cpu",
    )

    assert result.converged is False
    assert result.converged_pixels == 0
    assert result.unconverged_pixels == 4
    assert result.converged_fraction == 0.0
    assert not result.converged_mask.any()
    assert np.isnan(result.phase_unw).all()
    assert np.isnan(result.corrections_k).all()


def test_temporal_irls_ignores_singular_pixels_in_convergence_fraction() -> None:
    """Missing pixels are not misreported as failed temporal solutions."""
    pairs = _three_date_pairs()
    true_phi = np.array([0.25, -0.1, 0.15], dtype=np.float64)
    phase_stack = np.broadcast_to(true_phi[:, None, None], (3, 1, 2)).copy()
    phase_stack[:, 0, 1] = np.nan

    result = unwrap_temporal_irls(
        phase_stack,
        pairs,
        wrapped_input=False,
        max_iter=10,
        device="cpu",
    )

    assert result.converged is True
    assert result.converged_pixels == 1
    assert result.unconverged_pixels == 0
    assert result.converged_fraction == 1.0
    np.testing.assert_array_equal(result.converged_mask, [[True, False]])
    assert np.isfinite(result.phase_unw[:, 0, 0]).all()
    assert np.isnan(result.phase_unw[:, 0, 1]).all()
