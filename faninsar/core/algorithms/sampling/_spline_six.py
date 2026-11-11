"""Vectorized six-sample natural-spline leaves for fixed-shape tensor callers.

These private leaves retain the qualified weighted evaluation graph. The public
sampler uses the equivalent recurrence in ``_kernels`` to retain the historical
NumPy expression order. Both evaluate the natural cubic spline through six
samples at ``1 + fraction``; neither is a degree-five polynomial or B-spline.
Independent natural-spline and cross-form tests qualify this shared contract.
Callers own input admission, query batching, and physical-coordinate conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def _spline_six_weights(fraction: Tensor) -> Tensor:
    """Return float64 natural-spline weights without changing input placement."""
    import torch

    if fraction.dtype != torch.float64:
        fraction = fraction.to(dtype=torch.float64)
    second_one = (
        1.6076555023923444,
        -3.6459330143540667,
        2.5837320574162677,
        -0.6889952153110048,
        0.1722488038277512,
        -0.0287081339712919,
    )
    second_two = (
        -0.4306220095693780,
        2.5837320574162677,
        -4.3349282296650715,
        2.7559808612440193,
        -0.6889952153110048,
        0.1148325358851674,
    )
    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    weights = [
        fraction * (-one / 3.0 - two / 6.0)
        + fraction2 * (one / 2.0)
        + fraction3 * (two - one) / 6.0
        for one, two in zip(second_one, second_two, strict=True)
    ]
    weights[1] = weights[1] + 1.0 - fraction
    weights[2] = weights[2] + fraction
    return torch.stack(weights, dim=-1)


def _natural_spline_six_weighted(values: Tensor, fraction: Tensor) -> Tensor:
    """Evaluate six samples along the last axis with the qualified weights."""
    import torch

    return torch.sum(values * _spline_six_weights(fraction), dim=-1)


def _sample_six(source: Tensor, row: Tensor, column: Tensor) -> Tensor:
    """Sample a 2D tensor at array indices with the full asymmetric 6x6 stencil.

    Coordinates name array samples directly; no pixel-centre shift is inserted.
    Stencils span ``floor(index) - 1`` through ``floor(index) + 4``. Every tap
    must be finite, including zero-weight taps at integer coordinates. Invalid
    support and undersized rasters return NaN. Float32 source storage remains
    resident while coordinates, weighted arithmetic, and results use float64.
    """
    import torch

    if row.dtype != torch.float64:
        row = row.to(dtype=torch.float64)
    if column.dtype != torch.float64:
        column = column.to(dtype=torch.float64)
    row_base = torch.floor(row).to(torch.int64)
    column_base = torch.floor(column).to(torch.int64)
    rows, columns = source.shape[-2:]
    # No valid six-point base exists on an undersized raster. In particular,
    # avoid passing a negative upper bound to clamp and then gathering it.
    if rows < 6 or columns < 6:
        return torch.full_like(row, torch.nan)
    valid = (
        torch.isfinite(row)
        & torch.isfinite(column)
        & (row_base >= 1)
        & (row_base <= rows - 5)
        & (column_base >= 1)
        & (column_base <= columns - 5)
    )
    safe_row = row_base.clamp(1, rows - 5)
    safe_column = column_base.clamp(1, columns - 5)
    column_fraction = column - safe_column
    row_fraction = row - safe_row
    column_weights = _spline_six_weights(column_fraction)
    # Retain the qualified row-major unfold view and column-then-row reduction;
    # do not replace this fixed-shape graph with the public chunked sampler.
    windows = source.unfold(0, 6, 1).unfold(1, 6, 1)
    window = windows[safe_row - 1, safe_column - 1]
    row_values = torch.sum(window * column_weights.unsqueeze(-2), dim=-1)
    row_weights = _spline_six_weights(row_fraction)
    sampled = torch.sum(row_values * row_weights, dim=-1)
    return torch.where(
        valid & torch.isfinite(sampled), sampled, torch.full_like(sampled, torch.nan)
    )
