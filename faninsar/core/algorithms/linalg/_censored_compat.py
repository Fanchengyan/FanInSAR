"""Private historical Gram solver, separate from public singular-cutoff OLS.

Only legacy domain adapters use this kernel. Its NaN-only exclusion, Gram
cutoff, pivot proxy and 1e-8 null-mass tolerance intentionally remain unchanged.
It does not extend the public algorithms device/dtype contract or resolve devices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from faninsar.core.algorithms._validation import _require_torch

if TYPE_CHECKING:
    from torch import Tensor

_PIVOT_RATIO_MIN = 1e-6
_NULL_MASS_TOL = 1e-8


def _censored_lstsq_kernel(
    G: Tensor, d: Tensor, *, rcond: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return legacy solution, constraints, rank, counts and condition proxy."""
    torch = _require_torch()
    dtype, device = d.dtype, d.device
    d_nan = torch.isnan(d)
    valid = ~d_nan
    values = torch.where(d_nan, torch.zeros((), dtype=dtype, device=device), d)
    n_param = G.shape[-1]
    n_pt = d.shape[-1]
    n_obs = valid.sum(dim=0)
    x = torch.full((n_param, n_pt), torch.nan, dtype=dtype, device=device)
    constrained = torch.zeros((n_param, n_pt), dtype=torch.bool, device=device)
    rank = torch.zeros(n_pt, dtype=torch.int64, device=device)
    count = n_obs.to(torch.int64)
    condition = torch.full((n_pt,), torch.nan, dtype=dtype, device=device)
    solvable = torch.nonzero(n_obs > 0).flatten()
    if solvable.numel() > 0:
        m_sel = valid[:, solvable]
        d_sel = values[:, solvable]
        if G.ndim == 2:
            gw = m_sel.T.unsqueeze(2) * G.unsqueeze(0)
        else:
            gw = G[solvable] * m_sel.T.unsqueeze(-1)
        t_mat = torch.matmul(gw.mT, gw)
        rhs = torch.matmul(gw.mT, d_sel.T.unsqueeze(2))
        ell, info = torch.linalg.cholesky_ex(t_mat)
        pivots = torch.diagonal(ell, dim1=1, dim2=2)
        pivot_ratio = torch.where(
            info == 0,
            pivots.amin(dim=1) / pivots.amax(dim=1).clamp_min(torch.finfo(dtype).tiny),
            torch.full((solvable.numel(),), torch.inf, dtype=dtype, device=device),
        )
        spd = (info == 0) & (pivot_ratio > _PIVOT_RATIO_MIN)
        n_sel = solvable.numel()
        x_sel = torch.empty((n_param, n_sel), dtype=dtype, device=device)
        rank_sel = torch.empty(n_sel, dtype=torch.int64, device=device)
        cond_sel = torch.empty(n_sel, dtype=dtype, device=device)
        constrained_sel = torch.ones((n_param, n_sel), dtype=torch.bool, device=device)
        if bool(spd.any()):
            x_spd = torch.cholesky_solve(rhs[spd], ell[spd]).squeeze(-1).T
            x_sel[:, spd] = x_spd
            rank_sel[spd] = n_param
            cond_sel[spd] = pivot_ratio[spd]
        deficient = ~spd
        if bool(deficient.any()):
            w_e, v_e = torch.linalg.eigh(t_mat[deficient])
            lam_max = w_e[:, -1:]
            keep = w_e > lam_max * rcond
            rank_sel[deficient] = keep.sum(dim=1).to(torch.int64)
            lam_min = torch.where(keep, w_e, torch.full_like(w_e, torch.inf)).amin(
                dim=1
            )
            cond_sel[deficient] = lam_max[:, 0] / lam_min
            w_inv = torch.where(keep, 1.0 / w_e, torch.zeros_like(w_e))
            x_def = torch.matmul(
                v_e,
                w_inv.unsqueeze(-1) * torch.matmul(v_e.mT, rhs[deficient]),
            )
            x_sel[:, deficient] = x_def.squeeze(-1).T
            null_mass = (v_e * v_e * (~keep).unsqueeze(1)).sum(dim=2)
            constrained_sel[:, deficient] = (null_mass < _NULL_MASS_TOL).T
        x[:, solvable] = x_sel
        rank[solvable] = rank_sel
        condition[solvable] = cond_sel
        constrained[:, solvable] = constrained_sel
    return x, constrained, rank, count, condition
