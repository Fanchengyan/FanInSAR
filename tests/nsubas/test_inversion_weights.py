"""Public-seam tests for NSBASSolver coherence weighting."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from faninsar import Pairs
from faninsar.NSBAS import LinearModel, NSBASSolver
from faninsar.NSBAS.coherence_weight import coherence_to_weight


def _sample_pairs() -> Pairs:
    return Pairs.from_names(
        [
            "20170101_20170113",
            "20170101_20170125",
            "20170113_20170125",
        ]
    )


def _inconsistent_unw(pairs: Pairs) -> np.ndarray:
    """Return an overdetermined, inconsistent interferogram matrix."""
    design = pairs.sbas_matrix().astype(np.float64)
    increments = np.array([[1.0, 0.5], [2.0, -1.0]], dtype=np.float64)
    unw = design @ increments
    unw = unw + np.array(
        [[0.20, -0.15], [-0.35, 0.25], [0.40, -0.30]],
        dtype=np.float64,
    )
    return unw


def _weighted_lstsq(
    design: np.ndarray,
    observations: np.ndarray,
    weight_sqrt: np.ndarray,
) -> np.ndarray:
    n_pixel = observations.shape[1]
    n_param = design.shape[1]
    result = np.empty((n_param, n_pixel), dtype=np.float64)
    for pixel in range(n_pixel):
        scaled_g = design * weight_sqrt[:, pixel : pixel + 1]
        scaled_d = observations[:, pixel] * weight_sqrt[:, pixel]
        result[:, pixel], *_ = np.linalg.lstsq(scaled_g, scaled_d, rcond=None)
    return result


def test_inverse_fim_weighting_matches_sqrt_weight_lstsq() -> None:
    """WLS with FIM weights must scale G and d by sqrt(w), not w."""
    pairs = _sample_pairs()
    unw = _inconsistent_unw(pairs)
    coh = np.array(
        [[0.90, 0.80], [0.70, 0.95], [0.60, 0.85]],
        dtype=np.float64,
    )
    n_looks = 4
    weight = 2.0 * n_looks * coh**2 / (1.0 - coh**2)
    expected_incs = _weighted_lstsq(
        pairs.sbas_matrix().astype(np.float64),
        unw,
        np.sqrt(weight),
    )
    wrong_incs = _weighted_lstsq(
        pairs.sbas_matrix().astype(np.float64),
        unw,
        weight,
    )

    solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
    incs, params, *_ = solver.inverse(
        coh=coh,
        weight_func="fim",
        n_looks=n_looks,
        coh_threshold=0.0,
    )

    np.testing.assert_allclose(incs, expected_incs, rtol=1e-8, atol=1e-8)
    assert not np.allclose(incs, wrong_incs)
    assert params.shape[0] == 0


def test_inverse_can_change_coherence_after_construction() -> None:
    """inverse() must accept a later coh/weight_func and rebuild the system."""
    pairs = _sample_pairs()
    unw = _inconsistent_unw(pairs)
    coh_low = np.array(
        [[0.50, 0.55], [0.90, 0.60], [0.55, 0.92]],
        dtype=np.float64,
    )
    coh_high = np.array(
        [[0.95, 0.90], [0.60, 0.94], [0.92, 0.55]],
        dtype=np.float64,
    )

    solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
    incs_low, *_ = solver.inverse(
        coh=coh_low,
        weight_func="coh",
        coh_threshold=0.0,
    )
    incs_high, *_ = solver.inverse(
        coh=coh_high,
        weight_func="fim",
        n_looks=8,
        coh_threshold=0.0,
    )

    assert not np.allclose(incs_low, incs_high)


def test_nsbas_constraint_rows_are_not_coherence_weighted() -> None:
    """NSBAS model rows stay at gamma scale; only interferogram rows take sqrt(w)."""
    pairs = _sample_pairs()
    unw = _inconsistent_unw(pairs)
    coh = np.array(
        [[0.90, 0.80], [0.70, 0.95], [0.60, 0.85]],
        dtype=np.float64,
    )
    model = LinearModel(pairs.dates)
    gamma = 0.01
    solver = NSBASSolver(
        unw, pairs, model=model, gamma=gamma, verbose=False, device="cpu"
    )
    solver.inverse(coh=coh, weight_func="coh", coh_threshold=0.0)

    n_pair = len(pairs)
    constraint = solver.G.detach().cpu().numpy()[:, n_pair:, :]
    expected_bottom = (
        np.hstack(
            (
                np.tril(np.ones((len(pairs.dates), len(pairs.dates) - 1)), k=-1),
                model.G_br,
            )
        )
        * gamma
    )
    np.testing.assert_allclose(constraint[0], expected_bottom, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(constraint[1], expected_bottom, rtol=1e-6, atol=1e-6)


def test_constructor_and_inverse_default_shared_kwargs_to_none() -> None:
    """Constructor and inverse() must share None sentinels for weighting kwargs."""
    shared = ("coh", "coh_threshold", "weight_func", "n_looks")
    ctor = inspect.signature(NSBASSolver.__init__)
    inverse = inspect.signature(NSBASSolver.inverse)
    assert "inv_alg" not in ctor.parameters
    assert "inv_alg" not in inverse.parameters
    for name in shared:
        assert ctor.parameters[name].default is None
        assert inverse.parameters[name].default is None


def test_weighted_inversion_without_coherence_raises() -> None:
    """Selecting a weight function without coherence must fail closed."""
    pairs = _sample_pairs()
    unw = _inconsistent_unw(pairs)
    with pytest.raises(ValueError, match="coherence"):
        NSBASSolver(unw, pairs, weight_func="fim", verbose=False, device="cpu")
    solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
    with pytest.raises(ValueError, match="coherence"):
        solver.inverse(weight_func="var")


def test_inverse_without_kwargs_keeps_constructor_weighting() -> None:
    """Omitting inverse() kwargs must keep the constructor weighting, not reset it."""
    pairs = _sample_pairs()
    unw = _inconsistent_unw(pairs)
    coh = np.array(
        [[0.90, 0.80], [0.70, 0.95], [0.60, 0.85]],
        dtype=np.float64,
    )
    n_looks = 4
    weight = 2.0 * n_looks * coh**2 / (1.0 - coh**2)
    expected_incs = _weighted_lstsq(
        pairs.sbas_matrix().astype(np.float64),
        unw,
        np.sqrt(weight),
    )

    solver = NSBASSolver(
        unw,
        pairs,
        coh=coh,
        weight_func="fim",
        n_looks=n_looks,
        coh_threshold=0.0,
        verbose=False,
        device="cpu",
    )
    incs, *_ = solver.inverse()
    np.testing.assert_allclose(incs, expected_incs, rtol=1e-8, atol=1e-8)


def test_repr_includes_weight_func() -> None:
    """Solver repr must show the stored weight function."""
    pairs = _sample_pairs()
    unw = _inconsistent_unw(pairs)
    solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
    text = repr(solver)
    assert "weight_func: no" in text
    assert "G shape: (3, 2)" in text

    coh = np.array(
        [[0.90, 0.80], [0.70, 0.95], [0.60, 0.85]],
        dtype=np.float64,
    )
    solver.inverse(coh=coh, weight_func="fim", coh_threshold=0.0)
    text = repr(solver)
    assert "weight_func: fim" in text
    assert "G shape: (2, 3, 2)" in text


def test_var_weights_are_below_fim_and_increase_with_coherence() -> None:
    """DS variance weights stay below the FIM/CRB and grow with coherence."""
    n_looks = 20
    coh = np.array([[0.60, 0.90]], dtype=np.float64)
    var_w = coherence_to_weight(coh, weight_func="var", n_looks=n_looks)
    fim_w = coherence_to_weight(coh, weight_func="fim", n_looks=n_looks)
    assert np.all(var_w > 0)
    assert np.all(var_w < fim_w)
    assert var_w[0, 1] > var_w[0, 0]
