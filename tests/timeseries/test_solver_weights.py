"""Multi-weight WLS solver port coverage for the post-P0050 layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from faninsar import Pairs
from faninsar.timeseries.coherence_weight import (
    coherence_to_weight,
    normalize_weight_func,
)
from faninsar.timeseries.solver import NSBASSolver, censored_lstsq

_NAMES = [
    "20170111_20170204",
    "20170111_20170222",
    "20170111_20170318",
    "20170204_20170222",
    "20170204_20170318",
    "20170222_20170318",
]


@pytest.fixture
def pairs() -> Pairs:
    """Small connected pair network (4 dates, 6 pairs)."""
    return Pairs.from_names(_NAMES)


@pytest.fixture
def unw(pairs: Pairs) -> np.ndarray:
    """Deterministic unwrapped phase matrix (6 pairs, 8 pixels)."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(len(pairs), 8))


class TestNormalizeWeightFunc:
    """Canonical names and aliases."""

    @pytest.mark.parametrize("name", ["no", "coh", "fim", "var"])
    def test_canonical(self, name: str) -> None:
        """Canonical names pass through unchanged."""
        assert normalize_weight_func(name) == name

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("sbas", "no"),
            ("uniform", "no"),
            ("lin", "coh"),
            ("fisher", "fim"),
            ("wls", "fim"),
        ],
    )
    def test_aliases(self, alias: str, canonical: str) -> None:
        """Legacy spellings map to their canonical weight function."""
        assert normalize_weight_func(alias) == canonical

    def test_none_maps_to_no(self) -> None:
        """An unset weight function keeps the unweighted default."""
        assert normalize_weight_func(None) == "no"

    def test_unknown_raises(self) -> None:
        """Unknown weight names are rejected."""
        with pytest.raises(ValueError, match="weight_func"):
            normalize_weight_func("bogus")


class TestCoherenceToWeight:
    """Weight shapes, positivity, and monotonic behaviour."""

    def test_shapes_and_positivity_fim_var(self) -> None:
        """Fim and var weights are finite, positive, and shape-preserving."""
        coh = np.linspace(0.05, 0.95, 24).reshape(6, 4)
        for func in ("fim", "var"):
            weight = coherence_to_weight(coh, weight_func=func, n_looks=10)
            assert weight.shape == coh.shape
            assert np.all(np.isfinite(weight))
            assert np.all(weight > 0)

    def test_no_returns_ones(self) -> None:
        """The ``no`` weight function returns unit weights."""
        coh = np.full((3, 4), 0.5)
        assert np.all(coherence_to_weight(coh, "no") == 1.0)

    def test_coh_returns_coherence(self) -> None:
        """The ``coh`` weight function returns the coherence itself."""
        coh = np.linspace(0.2, 0.9, 12).reshape(3, 4)
        np.testing.assert_allclose(coherence_to_weight(coh, "coh"), coh)

    def test_fim_increases_with_coherence(self) -> None:
        """Fisher-information weights grow with coherence."""
        coh = np.array([[0.2, 0.5, 0.8]])
        weight = coherence_to_weight(coh, "fim", n_looks=8)
        assert np.all(np.diff(weight) > 0)

    def test_nan_clipped_to_epsilon(self) -> None:
        """NaN coherence clips to the epsilon floor and stays finite."""
        coh = np.array([[np.nan, 0.9]])
        weight = coherence_to_weight(coh, "fim", n_looks=4)
        assert np.all(np.isfinite(weight))


class TestSolverWeights:
    """Weighted solver construction and fail-closed behaviour."""

    def test_weighted_without_coherence_fails(
        self, pairs: Pairs, unw: np.ndarray
    ) -> None:
        """A weighted construction without coherence raises ValueError."""
        with pytest.raises(ValueError, match="coherence is required"):
            NSBASSolver(unw, pairs, weight_func="fim", verbose=False, device="cpu")

    def test_weighted_in_inverse_without_coherence_fails(
        self, pairs: Pairs, unw: np.ndarray
    ) -> None:
        """Switching to a weighted solve without coherence raises ValueError."""
        solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
        with pytest.raises(ValueError, match="coherence is required"):
            solver.inverse(weight_func="var")

    def test_fim_builds_3d_g(self, pairs: Pairs, unw: np.ndarray) -> None:
        """Weighted systems expand the design matrix to per-pixel 3-D form."""
        coh = np.full(unw.shape, 0.8)
        solver = NSBASSolver(
            unw,
            pairs,
            coh=coh,
            weight_func="fim",
            n_looks=12,
            verbose=False,
            device="cpu",
        )
        assert solver.G.ndim == 3
        assert solver.G.shape[0] == unw.shape[1]

    def test_weighted_inverse_runs(self, pairs: Pairs, unw: np.ndarray) -> None:
        """A coherence-weighted solver inverts to full-shaped increments."""
        coh = np.full(unw.shape, 0.7)
        solver = NSBASSolver(
            unw,
            pairs,
            coh=coh,
            weight_func="coh",
            verbose=False,
            device="cpu",
        )
        result = solver.inverse()
        n_dates = len(pairs.dates)
        assert result.increments.shape == (n_dates - 1, unw.shape[1])


class TestRankRobustLstsq:
    """Minimum-norm fallback and diagnostics on censored_lstsq."""

    def test_rank_deficient_pixel_min_norm(self) -> None:
        """Rank-deficient pixels receive the minimum-norm solution."""
        # G maps 2 increments; pixel 0 sees only the first observation row,
        # leaving its system rank deficient.
        g_mat = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        d = np.array([[2.0, 1.0], [np.nan, 3.0], [np.nan, 2.0]])
        out = censored_lstsq(g_mat, d, device="cpu", return_numpy=False)
        # Pixel 0: only row 0 valid -> increments share the observation.
        sol = out.solution.numpy()
        assert np.isfinite(sol[:, 0]).all()
        # Minimum norm over the constrained direction only: x0 == d0.
        assert sol[0, 0] == pytest.approx(2.0)
        assert out.is_constrained.shape == (2, 2)
        # Parameter 1 of pixel 0 is unobserved -> unconstrained flag.
        assert not bool(out.is_constrained[1, 0])
        assert bool(out.is_constrained[0, 0])
        # Pixel 1 is fully observed and full rank.
        assert int(out.rank[1]) == 2
        assert bool(out.is_constrained[:, 1].all())

    def test_empty_pixel_stays_nan(self) -> None:
        """Pixels with no valid observation stay NaN with rank zero."""
        g_mat = np.eye(2)
        d = np.array([[1.0, np.nan], [2.0, np.nan]])
        out = censored_lstsq(g_mat, d, device="cpu", return_numpy=False)
        assert np.isnan(out.solution[:, 1].numpy()).all()
        assert int(out.rank[1]) == 0
        assert int(out.n_observations[1]) == 0

    def test_diagnostics_populated(self) -> None:
        """LstsqResult always fills rank, counts, and condition estimates."""
        g_mat = np.eye(3)
        d = np.arange(9, dtype=np.float64).reshape(3, 3)
        out = censored_lstsq(g_mat, d, device="cpu", return_numpy=False)
        assert out.rank.shape == (3,)
        assert out.n_observations.shape == (3,)
        assert out.condition_number.shape == (3,)
        assert torch.isfinite(out.condition_number).all()


class TestNSBASResultDiagnostics:
    """NSBASResult diagnostics opt-in via solver.inverse(diagnostics=True)."""

    def test_diagnostics_none_by_default(self, pairs: Pairs, unw: np.ndarray) -> None:
        """Diagnostics stay None unless requested."""
        solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
        result = solver.inverse()
        assert result.is_constrained is None
        assert result.rank is None

    def test_diagnostics_populated(self, pairs: Pairs, unw: np.ndarray) -> None:
        """diagnostics=True fills every per-pixel reliability field."""
        solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
        result = solver.inverse(diagnostics=True)
        n_dates = len(pairs.dates)
        assert result.is_constrained.shape == (n_dates - 1, unw.shape[1])
        assert result.n_observations.shape == (unw.shape[1],)
        assert result.rank.shape == (unw.shape[1],)
        assert result.condition_number.shape == (unw.shape[1],)

    def test_tuple_unpacking_still_works(self, pairs: Pairs, unw: np.ndarray) -> None:
        """The historical 4-tuple unpacking pattern keeps working."""
        solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
        incs, params, res_pair, _res_tsm = solver.inverse()
        n_dates = len(pairs.dates)
        assert incs.shape == (n_dates - 1, unw.shape[1])
        assert params.shape == (0, unw.shape[1])
        assert res_pair.shape == (len(pairs), unw.shape[1])
