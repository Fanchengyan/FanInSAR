"""Tensor-only polynomial leaves preserving legacy arithmetic contracts.

These private kernels intentionally do not adopt the normalized SVD solver or
Horner evaluation of the public fitting API. Callers own basis selection,
coordinate construction, sampling, placement, validation, and error context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from faninsar.core.algorithms._validation import _require_torch

if TYPE_CHECKING:
    import torch


def _separable_polynomial_fit(
    slices: torch.Tensor,
    valid: torch.Tensor,
    basis: list[tuple[int, int]],
    max_power: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Solve scaled float64 moment equations for caller-selected (row, col) powers.

    Return coefficients and separable powers without changing the solve order.
    Singular systems raise the original Torch error for the caller to contextualize.
    """
    torch = _require_torch()
    n_slice, height, width = slices.shape
    n_param = len(basis)
    acc = torch.float64
    device = slices.device
    y_lin = torch.arange(height, dtype=acc, device=device)
    x_lin = torch.arange(width, dtype=acc, device=device)
    y_pow = [torch.ones(height, dtype=acc, device=device)]
    x_pow = [torch.ones(width, dtype=acc, device=device)]
    for _ in range(max_power):
        y_pow.append(y_pow[-1] * y_lin)
        x_pow.append(x_pow[-1] * x_lin)

    values = slices.to(acc)
    valid_f = valid.to(acc)
    masked = torch.where(valid, values, torch.zeros((), dtype=acc, device=device))
    moments = torch.empty(n_param, n_slice, dtype=acc, device=device)
    for idx, (i, j) in enumerate(basis):
        row = torch.einsum("h,ehw->ew", y_pow[i], masked)
        moments[idx] = torch.einsum("w,ew->e", x_pow[j], row)

    rows_valid: dict[int, torch.Tensor] = {}
    normal = torch.empty(n_param, n_param, n_slice, dtype=acc, device=device)
    for ai, (i1, j1) in enumerate(basis):
        for bi, (i2, j2) in enumerate(basis):
            power = i1 + i2
            if power not in rows_valid:
                rows_valid[power] = torch.einsum("h,ehw->ew", y_pow[power], valid_f)
            normal[ai, bi] = torch.einsum("w,ew->e", x_pow[j1 + j2], rows_valid[power])

    scale = torch.sqrt(
        torch.diagonal(normal, dim1=0, dim2=1).clamp_min(torch.finfo(acc).tiny)
    )
    normal_scaled = normal / (scale.T[:, None, :] * scale.T[None, :, :])
    moments_scaled = moments / scale.T
    coef_scaled = torch.linalg.solve(
        normal_scaled.permute(2, 0, 1),
        moments_scaled.T.unsqueeze(-1),
    ).squeeze(-1)
    return coef_scaled / scale, y_pow, x_pow


def _evaluate_separable_polynomial(
    coefficients: torch.Tensor,
    y_powers: list[torch.Tensor],
    x_powers: list[torch.Tensor],
    basis: list[tuple[int, int]],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Accumulate selected powers in output dtype, preserving legacy rounding."""
    torch = _require_torch()
    y_eval = [tensor.to(dtype) for tensor in y_powers]
    x_eval = [tensor.to(dtype) for tensor in x_powers]
    coef_eval = coefficients.to(dtype)
    surface = torch.zeros(
        coefficients.shape[0],
        y_powers[0].numel(),
        x_powers[0].numel(),
        dtype=dtype,
        device=coefficients.device,
    )
    for idx, (i, j) in enumerate(basis):
        surface += (
            coef_eval[:, idx, None, None]
            * y_eval[i][None, :, None]
            * x_eval[j][None, None, :]
        )
    return surface


def _weighted_total_degree_surface(
    y: torch.Tensor,
    coordinates: torch.Tensor,
    weight: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Fit supplied (x, y) coordinates with legacy total-degree weighted lstsq.

    Coordinates are promoted only after the caller constructs them. This retains
    float32 normalized grids, negative-weight clamping, the default lstsq rank
    policy, and evaluation in float64 before casting to the input value dtype.
    """
    torch = _require_torch()
    orig_shape = y.shape
    xn = coordinates[..., 0].reshape(-1).to(torch.float64)
    yn = coordinates[..., 1].reshape(-1).to(torch.float64)
    flat_y = y.reshape(-1).to(torch.float64)
    w = weight.reshape(-1).to(torch.float64)
    columns = []
    for total in range(degree + 1):
        for py in range(total + 1):
            px = total - py
            columns.append((yn**py) * (xn**px))
    design = torch.stack(columns, dim=-1)
    sqrt_w = torch.sqrt(w.clamp(min=0.0))
    theta = torch.linalg.lstsq(design * sqrt_w.unsqueeze(-1), flat_y * sqrt_w).solution
    surface = design @ theta
    return surface.reshape(orig_shape).to(y.dtype)


def _explicit_power_polynomial(
    coefficients: tuple[float, ...], x: torch.Tensor
) -> torch.Tensor:
    """Evaluate ascending coefficients without changing legacy operation order."""
    torch = _require_torch()
    result = torch.zeros_like(x)
    power = torch.ones_like(x)
    for coeff in coefficients:
        result = result + float(coeff) * power
        power = power * x
    return result
