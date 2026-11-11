"""Bounded Torch gather kernels for real array-index interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from faninsar.core.algorithms._validation import _require_torch

if TYPE_CHECKING:
    from torch import Tensor

_BIQUINTIC_OFFSETS = (-1, 0, 1, 2, 3, 4)
_QUERY_BATCH_SIZE = 4096


def _natural_spline_six(values: Tensor, fraction: Tensor) -> Tensor:
    """Evaluate the original six-sample recurrence, preserving operation order."""
    torch = _require_torch()
    samples = values.unbind(dim=-1)
    second = [torch.zeros_like(samples[0])]
    recurrence = [0.0] * 6
    for index in range(1, 5):
        denominator = recurrence[index - 1] / 2.0 + 2.0
        recurrence[index] = -0.5 / denominator
        second.append(
            (
                3.0 * (samples[index + 1] - 2.0 * samples[index] + samples[index - 1])
                - second[index - 1] / 2.0
            )
            / denominator
        )
    second.append(torch.zeros_like(samples[0]))
    for index in range(4, 0, -1):
        second[index] = recurrence[index] * second[index + 1] + second[index]
    return samples[1] + fraction * (
        samples[2]
        - samples[1]
        - second[1] / 3.0
        - second[2] / 6.0
        + fraction * (second[1] / 2.0 + fraction * (second[2] - second[1]) / 6.0)
    )


def _query_chunk(
    rows: Tensor,
    cols: Tensor,
    row_invalid: Tensor | bool | None,
    col_invalid: Tensor | bool | None,
    start: int,
    stop: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gather broadcast queries without materializing the full broadcast grid."""
    torch = _require_torch()
    flat = torch.arange(start, stop, device=rows.device, dtype=torch.int64)
    indices = []
    for size in reversed(rows.shape):
        indices.append(flat.remainder(size))
        flat = torch.div(flat, size, rounding_mode="floor")
    key = tuple(reversed(indices))
    row = rows[key].reshape(-1).to(dtype=torch.float64)
    col = cols[key].reshape(-1).to(dtype=torch.float64)
    available = torch.isfinite(row) & torch.isfinite(col)
    for invalid in (row_invalid, col_invalid):
        if isinstance(invalid, bool):
            if invalid:
                available = torch.zeros_like(available)
        elif invalid is not None:
            available = available & ~invalid.expand(rows.shape)[key].reshape(-1)
    return row, col, available


def _tap(
    source: Tensor,
    invalid: Tensor | bool | None,
    valid: Tensor | None,
    valid_invalid: Tensor | bool | None,
    rows: Tensor,
    cols: Tensor,
) -> tuple[Tensor, Tensor]:
    """Gather finite samples and intersect native and optional validity."""
    torch = _require_torch()
    samples = source[rows, cols].to(dtype=torch.float64)
    available = torch.isfinite(samples)
    if invalid is True:
        available = torch.zeros_like(available)
    elif invalid is not None and invalid is not False:
        available = available & ~invalid[rows, cols]
    if valid is not None:
        available = available & valid[rows, cols]
        if valid_invalid is True:
            available = torch.zeros_like(available)
        elif valid_invalid is not None and valid_invalid is not False:
            available = available & ~valid_invalid[rows, cols]
    return torch.where(available, samples, 0.0), available


def _sample_chunk(
    source: Tensor,
    invalid: Tensor | bool | None,
    valid: Tensor | None,
    valid_invalid: Tensor | bool | None,
    rows: Tensor,
    cols: Tensor,
    coordinate_valid: Tensor,
    *,
    method: Literal["nearest", "bilinear", "biquintic"],
    boundary: Literal["nan", "clamp"],
) -> tuple[Tensor, Tensor]:
    """Evaluate one image/query chunk with safe indices and strict support."""
    torch = _require_torch()
    height, width = source.shape
    if boundary == "clamp":
        rows = rows.clamp(0, height - 1)
        cols = cols.clamp(0, width - 1)
    if method == "nearest":
        supported = (
            coordinate_valid
            & (rows >= -1)
            & (rows <= height)
            & (cols >= -1)
            & (cols <= width)
        )
        # Replace huge coordinates before floor; check actual rounded indices too:
        # adding 0.5 can round an adjacent-to-half float onto the next integer.
        row = torch.floor(torch.where(supported, rows, 0.0) + 0.5).to(torch.int64)
        col = torch.floor(torch.where(supported, cols, 0.0) + 0.5).to(torch.int64)
        supported = supported & (row >= 0) & (row < height) & (col >= 0) & (col < width)
        sampled, available = _tap(
            source,
            invalid,
            valid,
            valid_invalid,
            row.clamp(0, height - 1),
            col.clamp(0, width - 1),
        )
        return sampled, supported & available
    if method == "bilinear":
        supported = (
            coordinate_valid
            & (rows >= 0)
            & (rows <= height - 1)
            & (cols >= 0)
            & (cols <= width - 1)
        )
        safe_rows = torch.where(supported, rows, 0.0)
        safe_cols = torch.where(supported, cols, 0.0)
        row = torch.floor(safe_rows).to(torch.int64)
        col = torch.floor(safe_cols).to(torch.int64)
        fr, fc = safe_rows - row, safe_cols - col
        sampled = torch.zeros_like(safe_rows)
        for dr, wr in ((0, 1.0 - fr), (1, fr)):
            for dc, wc in ((0, 1.0 - fc), (1, fc)):
                weight = wr * wc
                rr, cc = (
                    (row + dr).clamp(max=height - 1),
                    (col + dc).clamp(max=width - 1),
                )
                value, available = _tap(source, invalid, valid, valid_invalid, rr, cc)
                supported = supported & ((wr == 0) | (wc == 0) | available)
                sampled = sampled + weight * value
        return sampled, supported & torch.isfinite(sampled)
    first, last = _BIQUINTIC_OFFSETS[0], _BIQUINTIC_OFFSETS[-1]
    supported = (
        coordinate_valid
        & (rows >= -first)
        & (rows < height - last)
        & (cols >= -first)
        & (cols < width - last)
    )
    safe_rows = torch.where(supported, rows, float(-first))
    safe_cols = torch.where(supported, cols, float(-first))
    row = torch.floor(safe_rows).to(torch.int64)
    col = torch.floor(safe_cols).to(torch.int64)
    offsets = torch.tensor(_BIQUINTIC_OFFSETS, device=source.device)
    patch, patch_valid = _tap(
        source,
        invalid,
        valid,
        valid_invalid,
        row[:, None, None] + offsets[None, :, None],
        col[:, None, None] + offsets[None, None, :],
    )
    along_columns = _natural_spline_six(patch, (safe_cols - col)[:, None])
    sampled = _natural_spline_six(along_columns, safe_rows - row)
    return sampled, supported & patch_valid.all(dim=-1).all(dim=-1) & torch.isfinite(
        sampled
    )
