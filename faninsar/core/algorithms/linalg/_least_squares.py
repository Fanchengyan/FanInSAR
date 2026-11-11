"""Rank-revealing real least squares with independent observation subsets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Literal

from faninsar.core.algorithms._validation import (
    _as_tensor,
    _checked_size,
    _execution_device,
    _inspect_array,
    _require_torch,
)
from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from torch import Tensor

logger = setup_logger(__name__)
_INPUT_LIMIT = 1024**3
_OUTPUT_LIMIT = 256 * 1024**2
_WORKSPACE_LIMIT = 256 * 1024**2
_NULL_MASS_TOLERANCE = 1e-10
_RHS_BATCH = 1024
_RHS_WORKSPACE_LIMIT = 16 * 1024**2


@dataclass(frozen=True)
class LeastSquaresResult:
    """Minimum-norm solution and same-device mathematical diagnostics.

    Attributes
    ----------
    solution : torch.Tensor
        Float64 coefficients, shaped ``(n,k)`` or ``(n,)`` for a single RHS.
    rank : torch.Tensor
        Number of singular values strictly greater than ``rcond * max(s)``.
    n_observations : torch.Tensor
        Number of admitted equations per RHS, before singular truncation.
    is_constrained : torch.Tensor
        Per-parameter identification flags, shaped like ``solution``. True
        means squared projection onto the discarded/structural null space is
        at most 1e-10. Computed from the retained right-singular complement,
        including unrepresented null directions in wide systems.
    condition_number : torch.Tensor
        Actual largest/smallest retained singular-value ratio, not a proxy.
        NaN for no observations and infinity for observed rank-zero systems.
        In deficient systems this describes only the retained subspace.

    """

    solution: Tensor
    rank: Tensor
    n_observations: Tensor
    is_constrained: Tensor
    condition_number: Tensor


def _admitted(data: Tensor, invalid: Tensor | bool | None) -> Tensor:
    """Construct availability for a bounded numerical system or column."""
    torch = _require_torch()
    available = torch.isfinite(data)
    if invalid is True:
        available.fill_(False)
    elif invalid is not None and invalid is not False:
        available &= ~invalid
    return available


def _column(data: Tensor, column: int) -> Tensor:
    return data if data.ndim == 1 else data[:, column]


def _mask_column(invalid: Tensor | bool | None, column: int) -> Tensor | bool | None:
    if invalid is None or isinstance(invalid, bool) or invalid.ndim < 2:
        return invalid
    return invalid[:, column]


def _validate_options(method: str, weights: object, rcond: float) -> None:
    if method not in {"ols", "wls"}:
        reject("method must be 'ols' or 'wls'.", ValueError, logger=logger)
    if (method == "ols" and weights is not None) or (
        method == "wls" and weights is None
    ):
        reject("OLS forbids weights; WLS requires weights.", ValueError, logger=logger)
    if (
        isinstance(rcond, bool)
        or not isinstance(rcond, Real)
        or not math.isfinite(rcond)
        or rcond < 0
    ):
        reject(
            "rcond must be a finite nonnegative real scalar.", ValueError, logger=logger
        )


def _preflight(m: int, n: int, k: int) -> None:
    """Bound all outputs and a conservative reduced-SVD system workspace."""
    _checked_size(
        (n, k), itemsize=9, name="solution and constraints", limit_bytes=_OUTPUT_LIMIT
    )
    _checked_size(
        (k,), itemsize=24, name="system diagnostics", limit_bytes=_OUTPUT_LIMIT
    )
    reduced = min(m, n)
    elements = 4 * m * n + 2 * m * reduced + 2 * reduced * n + 8 * (m + n)
    _checked_size(
        (elements,),
        itemsize=8,
        name="single-system SVD workspace",
        limit_bytes=_WORKSPACE_LIMIT,
    )


def _rhs_view(data: Tensor | None, start: int, stop: int) -> Tensor | None:
    if data is None:
        return None
    return data[:, None] if data.ndim == 1 else data[:, start:stop]


def _rhs_mask(
    mask: Tensor | bool | None, start: int, stop: int
) -> Tensor | bool | None:
    if mask is None or isinstance(mask, bool):
        return mask
    return _rhs_view(mask, start, stop)


def _solve_shared(
    matrix: Tensor,
    rhs: Tensor,
    columns: Tensor,
    result: LeastSquaresResult,
    rcond: float,
) -> None:
    """Factor one effective design, applying its retained subspace to many RHS."""
    torch = _require_torch()
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    retained = singular > rcond * singular[0]
    result.rank[columns] = retained.sum()
    if not bool(retained.any()):
        result.solution[:, columns] = 0
        result.condition_number[columns] = torch.inf
        return
    right = vh[retained]
    result.solution[:, columns] = right.T @ (
        (u[:, retained].T @ rhs) / singular[retained, None]
    )
    constrained = (1 - right.square().sum(dim=0)).clamp_min(0) <= _NULL_MASS_TOLERANCE
    result.is_constrained[:, columns] = constrained[:, None]
    result.condition_number[columns] = singular[0] / singular[retained][-1]


def _solve_batched(
    matrix: Tensor,
    rhs: Tensor,
    columns: Tensor,
    result: LeastSquaresResult,
    rcond: float,
) -> None:
    """Solve a memory-bounded batch of genuinely distinct effective designs."""
    torch = _require_torch()
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    retained = singular > rcond * singular[:, :1]
    ranks = retained.sum(dim=1)
    inverse = torch.where(retained, singular, 1).reciprocal() * retained
    values = vh.mT @ (inverse[:, :, None] * (u.mT @ rhs[:, :, None]))
    observed = result.n_observations[columns] > 0
    result.solution[:, columns] = torch.where(
        observed[:, None], values[:, :, 0], torch.nan
    ).T
    result.rank[columns] = ranks
    null_mass = (1 - (vh.square() * retained[:, :, None]).sum(dim=1)).clamp_min(0)
    result.is_constrained[:, columns] = (
        (null_mass <= _NULL_MASS_TOLERANCE) & observed[:, None]
    ).T
    smallest = torch.where(retained, singular, torch.inf).amin(dim=1)
    condition = torch.where(ranks > 0, singular[:, 0] / smallest, torch.inf)
    result.condition_number[columns] = torch.where(observed, condition, torch.nan)


def least_squares(
    A: object,
    B: object,
    *,
    method: Literal["ols", "wls"] = "ols",
    valid: object = None,
    weights: object = None,
    rcond: float = 1e-10,
) -> LeastSquaresResult:
    """Solve real OLS/WLS with an independent observed row subset per RHS.

    Parameters
    ----------
    A : numpy.ndarray or torch.Tensor
        Shared design ``(m,n)`` or per-RHS design ``(k,m,n)``. All coefficients
        in an admitted row must be finite and native-unmasked. n must be positive.
    B : numpy.ndarray or torch.Tensor
        Observations ``(m,k)`` or one RHS ``(m,)``. Missingness stays RHS-local.
    method : {"ols", "wls"}, default="ols"
        WLS minimizes sum(weight * residual**2) using square-root row scaling.
    valid : Boolean array, optional
        Additional admission, shaped ``(m,)`` or exactly like B. Native masks
        exclude; True cannot restore unavailable A, B, or weights.
    weights : real array, optional
        WLS-only, shaped ``(m,)`` or exactly like B. Unmasked weights must be
        finite and nonnegative; zero and native-masked weights exclude rows.
    rcond : float, default=1e-10
        Nonnegative finite relative singular-value cutoff.

    Returns
    -------
    LeastSquaresResult
        Float64 solution/condition and integer/Boolean diagnostics on the input
        CPU/CUDA device. No-data solutions are NaN; observed rank-zero solutions
        are zero. Rank-deficient solutions have minimum Euclidean norm.

    Raises
    ------
    ValueError
        For incompatible dimensions, options, devices, weights, gradients, or
        requests exceeding the conservative allocation limits.
    TypeError
        For unsupported array containers or non-real/non-floating data.

    Notes
    -----
    Uses same-device reduced Torch SVD without full tall U, normal equations,
    NumPy execution, device fallback, or outlier rejection. Shared designs reuse
    a factorization for identical admitted rows/weights within bounded RHS
    chunks. Distinct designs or many mask patterns use memory-bounded batched
    SVD. Source storage/masks are never mutated. The single-system working budget
    is 256 MiB; source storage is capped at 1 GiB per input and outputs at
    256 MiB per component. These are admission limits, not a peak-memory promise.

    """
    _validate_options(method, weights, rcond)
    device = _execution_device(A, B, valid, weights)
    torch = _require_torch()
    a_info = _inspect_array(A, name="A")
    b_info = _inspect_array(B, name="B")
    if len(a_info.shape) not in {2, 3} or len(b_info.shape) not in {1, 2}:
        reject(
            "A must be (m,n)/(k,m,n); B must be (m,)/(m,k).", ValueError, logger=logger
        )
    m, n = a_info.shape[-2:]
    single = len(b_info.shape) == 1
    k = 1 if single else b_info.shape[1]
    if (
        n == 0
        or b_info.shape[0] != m
        or (len(a_info.shape) == 3 and a_info.shape[0] != k)
    ):
        reject(
            "A and B dimensions disagree or A has no parameters.",
            ValueError,
            logger=logger,
        )
    _preflight(m, n, k)
    for name, info in (("A", a_info), ("B", b_info)):
        _checked_size(info.shape, itemsize=9, name=name, limit_bytes=_INPUT_LIMIT)
    for name, value in (("valid", valid), ("weights", weights)):
        if value is None:
            continue
        info = _inspect_array(value, name=name, allow_bool=name == "valid")
        if info.shape not in {(m,), b_info.shape}:
            reject(
                f"{name} must have shape (m,) or exactly B.shape.",
                ValueError,
                logger=logger,
            )
        if name == "valid" and str(info.dtype) not in {"bool", "torch.bool"}:
            reject("valid must have Boolean dtype.", TypeError, logger=logger)
        _checked_size(info.shape, itemsize=9, name=name, limit_bytes=_INPUT_LIMIT)
    a, a_mask = _as_tensor(A, device=device, name="A")
    b, b_mask = _as_tensor(B, device=device, name="B")
    v, v_mask = (
        (None, None)
        if valid is None
        else _as_tensor(valid, device=device, name="valid", allow_bool=True)
    )
    w, w_mask = (
        (None, None)
        if weights is None
        else _as_tensor(weights, device=device, name="weights")
    )
    if a_mask is not None and not isinstance(a_mask, bool):
        a_mask = a_mask.expand(a.shape)
    if b_mask is not None and not isinstance(b_mask, bool):
        b_mask = b_mask.expand(b.shape)
    if v_mask is not None and not isinstance(v_mask, bool):
        v_mask = v_mask.expand(v.shape)
    if w_mask is not None and not isinstance(w_mask, bool):
        w_mask = w_mask.expand(w.shape)
    if w is not None:
        for j in range(1 if w.ndim == 1 else k):
            column = _column(w, j)
            mask = _mask_column(w_mask, j)
            bad = ~torch.isfinite(column) | (column < 0)
            if mask is True:
                bad.fill_(False)
            elif mask is not None and mask is not False:
                bad &= ~mask
            if bool(bad.any()):
                reject(
                    "Unmasked weights must be finite and nonnegative.",
                    ValueError,
                    logger=logger,
                )
    solution = torch.full((n, k), torch.nan, dtype=torch.float64, device=device)
    rank = torch.zeros(k, dtype=torch.int64, device=device)
    count = torch.zeros_like(rank)
    constrained = torch.zeros((n, k), dtype=torch.bool, device=device)
    condition = torch.full((k,), torch.nan, dtype=torch.float64, device=device)
    result = LeastSquaresResult(solution, rank, count, constrained, condition)
    # A batch is bounded by both query storage and conservative SVD workspace.
    reduced = min(m, n)
    svd_bytes = 8 * (4 * m * n + 2 * m * reduced + 2 * reduced * n + 8 * (m + n))
    shared = a.ndim == 2 and (w is None or w.ndim == 1)
    batch = max(1, min(_RHS_BATCH, _RHS_WORKSPACE_LIMIT // max(1, 64 * m)))
    batch = min(batch, max(1, _WORKSPACE_LIMIT // max(1, svd_bytes)))
    shared_rows = _admitted(a, a_mask).all(dim=1) if a.ndim == 2 else None
    for start in range(0, k, batch):
        stop = min(start + batch, k)
        values = _rhs_view(b, start, stop)
        observed = _admitted(values, _rhs_mask(b_mask, start, stop))
        if a.ndim == 2:
            observed &= shared_rows[:, None]
        else:
            native = (
                a_mask
                if a_mask is None or isinstance(a_mask, bool)
                else a_mask[start:stop]
            )
            observed &= _admitted(a[start:stop], native).all(dim=2).T
        if v is not None:
            allowed = _rhs_view(v, start, stop)
            observed &= allowed & _admitted(allowed, _rhs_mask(v_mask, start, stop))
        weight = _rhs_view(w, start, stop)
        if weight is not None:
            observed &= (weight > 0) & _admitted(weight, _rhs_mask(w_mask, start, stop))
        count[start:stop] = observed.sum(dim=0)
        if m == 0 or not bool(observed.any()):
            continue
        patterns = None
        if shared:
            # Check the frequent common-mask case without sorting/unique overhead.
            if bool((observed == observed[:, :1]).all()):
                patterns = observed[:, :1].T
                inverse = torch.zeros(stop - start, dtype=torch.int64, device=device)
            else:
                patterns, inverse = torch.unique(observed.T, dim=0, return_inverse=True)
                if patterns.shape[0] > 8:
                    patterns = None
        if patterns is not None:
            for group, rows in enumerate(patterns):
                if not bool(rows.any()):
                    continue
                local = torch.nonzero(inverse == group).flatten()
                columns = local + start
                matrix = a[rows].to(dtype=torch.float64)
                rhs = values[:, local][rows].to(dtype=torch.float64)
                if weight is not None:
                    scale = torch.sqrt(weight[rows, 0].to(dtype=torch.float64))
                    matrix = matrix * scale[:, None]
                    rhs = rhs * scale[:, None]
                    if not bool(
                        torch.isfinite(matrix).all() & torch.isfinite(rhs).all()
                    ):
                        reject(
                            "Weighted system overflows float64 row scaling.",
                            ValueError,
                            logger=logger,
                        )
                _solve_shared(matrix, rhs, columns, result, rcond)
        else:
            design = a[None, :, :] if a.ndim == 2 else a[start:stop]
            # Excluded equations are safe zero rows, not recovered observations.
            # Padding by zero preserves all nonzero singular values and null mass.
            matrix = torch.where(observed.T[:, :, None], design, 0).to(
                dtype=torch.float64
            )
            rhs = torch.where(observed, values, 0).T.to(dtype=torch.float64)
            if weight is not None:
                scale = torch.sqrt(
                    torch.where(observed, weight, 0).T.to(dtype=torch.float64)
                )
                matrix = matrix * scale[:, :, None]
                rhs = rhs * scale
                if not bool(torch.isfinite(matrix).all() & torch.isfinite(rhs).all()):
                    reject(
                        "Weighted system overflows float64 row scaling.",
                        ValueError,
                        logger=logger,
                    )
            columns = torch.arange(start, stop, device=device)
            _solve_batched(matrix, rhs, columns, result, rcond)
    if single:
        return LeastSquaresResult(
            solution[:, 0], rank[0], count[0], constrained[:, 0], condition[0]
        )
    return LeastSquaresResult(solution, rank, count, constrained, condition)
