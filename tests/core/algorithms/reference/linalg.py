"""Independent per-RHS NumPy/SciPy least-squares oracle."""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space


def reference_least_squares(A, B, *, weights=None, valid=None, rcond=1e-10):
    """Select each observed system independently and use NumPy LAPACK."""
    a = np.ma.asarray(A)
    b = np.ma.asarray(B)
    single = b.ndim == 1
    if single:
        b = b[:, None]
    m, k = b.shape
    n = a.shape[-1]
    solution = np.full((n, k), np.nan)
    rank = np.zeros(k, dtype=np.int64)
    count = np.zeros(k, dtype=np.int64)
    constrained = np.zeros((n, k), dtype=bool)
    condition = np.full(k, np.nan)
    for j in range(k):
        design = a if a.ndim == 2 else a[j]
        observed = np.all(
            np.isfinite(design.data) & ~np.ma.getmaskarray(design), axis=1
        )
        observed &= np.isfinite(b.data[:, j]) & ~np.ma.getmaskarray(b)[:, j]
        if valid is not None:
            v = np.ma.asarray(valid)
            column = v if v.ndim == 1 else v[:, j]
            observed &= column.data & ~np.ma.getmaskarray(column)
        if weights is not None:
            w = np.ma.asarray(weights)
            column = w if w.ndim == 1 else w[:, j]
            observed &= ~np.ma.getmaskarray(column) & (column.data > 0)
        count[j] = observed.sum()
        if not count[j]:
            continue
        design = np.asarray(design.data[observed], dtype=np.float64)
        values = np.asarray(b.data[observed, j], dtype=np.float64)
        if weights is not None:
            scale = np.sqrt(column.data[observed]).astype(np.float64)
            design = design * scale[:, None]
            values = values * scale
        solution[:, j], _, rank[j], singular = np.linalg.lstsq(
            design, values, rcond=rcond
        )
        if not rank[j]:
            condition[j] = np.inf
            continue
        condition[j] = singular[0] / singular[rank[j] - 1]
        null = null_space(design, rcond=rcond)
        constrained[:, j] = np.sum(null**2, axis=1) <= 1e-10
    if single:
        return solution[:, 0], rank[0], count[0], constrained[:, 0], condition[0]
    return solution, rank, count, constrained, condition
