"""Rank-robust censored least squares and per-pixel reliability diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from faninsar import Pairs
from faninsar.NSBAS import (
    LinearModel,
    LstsqDiagnostics,
    NSBASResult,
    NSBASSolver,
    batch_lstsq,
    censored_lstsq,
)


def _sbas_design(
    n_date: int, seed: int = 0
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return an incidence-like design matrix with consecutive chain pairs."""
    rng = np.random.default_rng(seed)
    edges = [(i, i + 1) for i in range(n_date - 1)]
    while len(edges) < 3 * (n_date - 1):
        i = int(rng.integers(0, n_date - 2))
        j = int(rng.integers(i + 1, n_date))
        edges.append((i, j))
    g = np.zeros((len(edges), n_date - 1))
    for row, (i, j) in enumerate(edges):
        g[row, i:j] = 1.0
    return g, edges


def _oracle(g: np.ndarray, d: np.ndarray, rcond: float) -> np.ndarray:
    """NumPy minimum-norm least squares per pixel."""
    n_param, n_pt = g.shape[1], d.shape[1]
    out = np.full((n_param, n_pt), np.nan)
    for px in range(n_pt):
        rows = ~np.isnan(d[:, px])
        if not rows.any():
            continue
        out[:, px] = np.linalg.lstsq(g[rows], d[rows, px], rcond=rcond)[0]
    return out


class TestCensoredLstsqSolver:
    """Rank-robust behavior of censored_lstsq."""

    def test_exactly_determined_pixel_is_solved(self) -> None:
        """Count == n_param must solve, not NaN (regression for dropped filter)."""
        rng = np.random.default_rng(1)
        g = rng.normal(size=(6, 4))
        d = rng.normal(size=(6, 3))
        d[[1, 3], 0] = np.nan  # exactly 4 valid rows == n_param
        x, diag = censored_lstsq(
            g, d, device="cpu", diagnostics=True, return_numpy=True
        )
        rows = ~np.isnan(d[:, 0])
        expected = np.linalg.lstsq(g[rows], d[rows, 0], rcond=None)[0]
        assert not np.isnan(x[:, 0]).any()
        np.testing.assert_allclose(x[:, 0], expected, rtol=1e-10, atol=1e-10)
        assert diag.n_observations[0] == 4
        assert diag.rank[0] == 4
        assert diag.is_constrained[:, 0].all()

    def test_zero_observation_pixel_stays_nan(self) -> None:
        """Pixels without observations stay NaN with rank 0."""
        rng = np.random.default_rng(2)
        g, _ = _sbas_design(5)
        d = rng.normal(size=(g.shape[0], 2))
        d[:, 1] = np.nan
        x, diag = censored_lstsq(g, d, device="cpu", diagnostics=True)
        assert np.isnan(x[:, 1]).all()
        assert diag.n_observations[1] == 0
        assert diag.rank[1] == 0
        assert not diag.is_constrained[:, 1].any()
        assert np.isfinite(x[:, 0]).all()

    def test_isolated_date_gets_min_norm_and_gap_label(self) -> None:
        """Isolated date: min-norm solution + False labels on its increments."""
        n_date = 6
        g, edges = _sbas_design(n_date, seed=3)
        isolated = 3
        assert g.shape == (3 * (n_date - 1), n_date - 1)
        rng = np.random.default_rng(3)
        increments = rng.normal(size=(n_date - 1, 2)) * 0.5
        d = g @ increments + rng.normal(scale=0.05, size=(g.shape[0], 2))
        # pixel 1: drop every row having the isolated date as an endpoint
        touch = np.array([isolated in (i, j) for i, j in edges], dtype=bool)
        d[touch, 1] = np.nan
        assert (~np.isnan(d[:, 1])).sum() > g.shape[1]  # still overdetermined

        x, diag = censored_lstsq(g, d, device="cpu", diagnostics=True, rcond=1e-10)
        expected = _oracle(g, d, 1e-10)
        np.testing.assert_allclose(x, expected, rtol=1e-8, atol=1e-8)

        assert diag.rank[1] == n_date - 2  # one unconstrained direction
        # the two increments around the isolated date are exactly the gap
        assert not diag.is_constrained[isolated - 1 : isolated + 1, 1].any()
        assert np.delete(diag.is_constrained[:, 1], [isolated - 1, isolated]).all()
        assert diag.is_constrained[:, 0].all()

    def test_random_dropout_matches_numpy_oracle(self) -> None:
        """Random dropout solutions match the numpy min-norm oracle."""
        g, _ = _sbas_design(9, seed=4)
        rng = np.random.default_rng(4)
        d = g @ rng.normal(size=(8, 40)) + rng.normal(scale=0.1, size=(g.shape[0], 40))
        d[rng.random(d.shape) < 0.15] = np.nan
        d[:, 7] = np.nan  # one unsolvable pixel

        x, diag = censored_lstsq(g, d, device="cpu", diagnostics=True)
        expected = _oracle(g, d, 1e-10)
        solvable = diag.n_observations > 0
        np.testing.assert_allclose(
            x[:, solvable], expected[:, solvable], rtol=1e-8, atol=1e-8
        )
        assert np.isnan(x[:, 7]).all()
        assert (diag.condition_number[solvable] > 0).all()
        assert (diag.rank[solvable] >= 1).all()
        # rank-deficient pixels must carry at least one False label
        deficient = solvable & (diag.rank < g.shape[1])
        assert (np.atleast_2d(~diag.is_constrained[:, deficient]).sum(axis=0) > 0).all()

    def test_full_rank_diagnostics_fast_path(self) -> None:
        """Full-rank systems report fast-path diagnostics."""
        g, _ = _sbas_design(5, seed=5)
        rng = np.random.default_rng(5)
        d = g @ rng.normal(size=(4, 3))
        _x, diag = censored_lstsq(g, d, device="cpu", diagnostics=True)
        assert diag.is_constrained.all()
        assert (diag.rank == g.shape[1]).all()
        assert np.isfinite(diag.condition_number).all()
        assert (diag.condition_number > 1e-6).all()  # pivot-ratio proxy
        assert (diag.n_observations == g.shape[0]).all()

    def test_weighted_3d_g_matches_per_pixel_oracle(self) -> None:
        """3D per-pixel weighted systems match a per-pixel numpy oracle."""
        g, _ = _sbas_design(5, seed=6)
        rng = np.random.default_rng(6)
        n_pt = 4
        d = g @ rng.normal(size=(4, n_pt)) + rng.normal(
            scale=0.1, size=(g.shape[0], n_pt)
        )
        d[rng.random(d.shape) < 0.1] = np.nan
        w_sqrt = rng.uniform(0.5, 1.5, size=d.shape)
        d_w = np.where(np.isnan(d), np.nan, d * w_sqrt)
        g_3d = np.repeat(g[None, :, :], n_pt, axis=0) * w_sqrt.T[:, :, None]

        x3, _ = censored_lstsq(g_3d, d_w, device="cpu", diagnostics=True)
        for px in range(n_pt):
            rows = ~np.isnan(d_w[:, px])
            sol, *_ = np.linalg.lstsq(g_3d[px][rows], d_w[rows, px], rcond=1e-10)
            np.testing.assert_allclose(x3[:, px], sol, rtol=1e-9, atol=1e-9)
        # identical unweighted systems must agree between 2D and 3D layouts
        x2 = censored_lstsq(g, d, device="cpu")
        x3u = censored_lstsq(np.repeat(g[None], n_pt, axis=0), d, device="cpu")
        np.testing.assert_allclose(x2, x3u, rtol=1e-9, atol=1e-9)

    def test_diagnostics_shapes_and_types(self) -> None:
        """Diagnostics fields have the documented shapes and dtypes."""
        g, _ = _sbas_design(4, seed=7)
        rng = np.random.default_rng(7)
        d = g @ rng.normal(size=(3, 5))
        x, diag = censored_lstsq(g, d, device="cpu", diagnostics=True)
        assert isinstance(diag, LstsqDiagnostics)
        assert diag.is_constrained.shape == x.shape
        assert diag.is_constrained.dtype == bool
        assert diag.rank.shape == (5,)
        assert diag.n_observations.shape == (5,)
        assert diag.condition_number.shape == (5,)

    def test_torch_return(self) -> None:
        """Torch tensors are returned when return_numpy is False."""
        g, _ = _sbas_design(4, seed=8)
        rng = np.random.default_rng(8)
        d = g @ rng.normal(size=(3, 2))
        x, diag = censored_lstsq(
            g, d, device="cpu", return_numpy=False, diagnostics=True
        )
        assert isinstance(x, torch.Tensor)
        assert isinstance(diag.is_constrained, torch.Tensor)

    def test_batch_lstsq_passes_diagnostics_through(self) -> None:
        """batch_lstsq stitches diagnostics across patches."""
        g, _ = _sbas_design(6, seed=9)
        rng = np.random.default_rng(9)
        d = g @ rng.normal(size=(5, 12))
        d[rng.random(d.shape) < 0.1] = np.nan
        x, diag = batch_lstsq(g, d, device="cpu", verbose=False, diagnostics=True)
        assert x.shape == (g.shape[1], 12)
        assert diag.is_constrained.shape == x.shape
        assert diag.n_observations.shape == (12,)


def _solver_pairs() -> Pairs:
    return Pairs.from_names(
        [
            "20170101_20170113",
            "20170101_20170125",
            "20170113_20170125",
            "20170113_20170206",
            "20170125_20170206",
        ]
    )


class TestNSBASSolverDiagnostics:
    """NSBASSolver-level diagnostics API."""

    def test_default_tuple_unchanged(self) -> None:
        """The default inverse() return stays a four-element tuple."""
        pairs = _solver_pairs()
        design = pairs.sbas_matrix().astype(np.float64)
        incs_true = np.array([[1.0, 0.5, -0.2], [2.0, -1.0, 0.3], [0.7, -0.4, 1.1]])
        unw = design @ incs_true
        solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
        result = solver.inverse()
        assert isinstance(result, tuple)
        assert len(result) == 4
        incs, params, res_pair, res_tsm = result
        assert incs.shape == (3, 3)
        assert params.shape[0] == 0
        assert res_pair.shape == (5, 3)
        assert res_tsm.shape[0] == 0

    def test_diagnostics_result_object_pure_sbas(self) -> None:
        """Pure SBAS diagnostics fill every field of NSBASResult."""
        pairs = _solver_pairs()
        design = pairs.sbas_matrix().astype(np.float64)
        incs_true = np.array([[1.0, 0.5, -0.2], [2.0, -1.0, 0.3], [0.7, -0.4, 1.1]])
        unw = design @ incs_true
        solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
        result = solver.inverse(diagnostics=True)
        assert isinstance(result, NSBASResult)
        assert result.increments.shape == (3, 3)
        assert result.is_constrained.shape == (3, 3)
        assert result.is_constrained.all()
        assert (result.rank == 3).all()
        assert (result.n_observations == 5).all()
        assert np.isfinite(result.condition_number).all()
        assert result.model_parameters.shape[0] == 0
        assert result.model_residuals.shape[0] == 0
        assert result.pair_residuals.shape == (5, 3)

    def test_diagnostics_flags_model_filled_gap(self) -> None:
        """NSBAS mode: increments stay finite, gap columns are labelled."""
        pairs = _solver_pairs()
        dates = pairs.dates
        design = pairs.sbas_matrix().astype(np.float64)
        incs_true = np.array([[1.0, 0.5, -0.2], [2.0, -1.0, 0.3], [0.7, -0.4, 1.1]])
        unw = design @ incs_true
        # pixel 1 loses every interferogram touching date index 2 (20170125)
        str_dates = [d.strftime("%Y%m%d") for d in dates]
        touch = np.array([str_dates[2] in name.split("_") for name in pairs.to_names()])
        unw[touch, 1] = np.nan
        assert (~np.isnan(unw[:, 1])).sum() >= 1

        model = LinearModel(dates)
        solver = NSBASSolver(unw, pairs, model=model, verbose=False, device="cpu")
        result = solver.inverse(diagnostics=True)
        assert np.isfinite(result.increments[:, 1]).all()  # model fills the gap
        # the increments around the isolated date are labelled unconstrained
        assert not result.is_constrained[1, 1]
        assert result.is_constrained[0, 1]
        assert result.is_constrained[:, 0].all()

    def test_rank_deficient_pure_sbas_no_longer_raises(self) -> None:
        """Rank-deficient pixels return min-norm values instead of raising."""
        pairs = _solver_pairs()
        design = pairs.sbas_matrix().astype(np.float64)
        incs_true = np.array([[1.0, 0.5, -0.2], [2.0, -1.0, 0.3], [0.7, -0.4, 1.1]])
        unw = design @ incs_true
        unw[:, 2] = np.nan  # no observations at all
        unw[0, 1] = np.nan
        unw[2, 1] = np.nan
        unw[3, 1] = np.nan
        unw[4, 1] = np.nan
        # pixel 1 keeps only pair "20170101_20170113": one direction unconstrained
        solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
        result = solver.inverse(diagnostics=True)
        assert np.isfinite(result.increments[:, 1]).all()
        assert not result.is_constrained[:, 1].all()
        assert np.isnan(result.increments[:, 2]).all()
        assert result.n_observations[2] == 0

    def test_rcond_sentinel_and_validation(self) -> None:
        """The rcond parameter follows the None-sentinel pattern and is validated."""
        pairs = _solver_pairs()
        design = pairs.sbas_matrix().astype(np.float64)
        unw = design @ np.array([[1.0, 0.5], [2.0, -1.0], [0.7, -0.4]])
        solver = NSBASSolver(unw, pairs, verbose=False, device="cpu")
        assert solver.rcond == 1e-10
        solver.inverse(rcond=1e-6)
        assert solver.rcond == 1e-6
        with pytest.raises(ValueError, match="rcond"):
            solver.inverse(rcond=0.0)
        with pytest.raises(ValueError, match="rcond"):
            NSBASSolver(unw, pairs, verbose=False, device="cpu", rcond=2.0)
