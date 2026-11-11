"""Independent scalar sampling references, using SciPy natural splines."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_reference(
    source: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    method: str = "bilinear",
    valid: np.ndarray | None = None,
    boundary: str = "nan",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample small arrays with explicit support checks and scalar loops.

    The six-sample reference uses SciPy's independently implemented spline
    construction, not production weights, recurrence, or indexing helpers.
    """
    values = np.asarray(np.ma.getdata(source), dtype=np.float64)
    available = ~np.ma.getmaskarray(source) & np.isfinite(values)
    if valid is not None:
        available &= np.asarray(np.ma.getdata(valid), dtype=bool)
        available &= ~np.ma.getmaskarray(valid)
    row_values, col_values = np.broadcast_arrays(
        np.asarray(np.ma.getdata(rows), dtype=np.float64),
        np.asarray(np.ma.getdata(cols), dtype=np.float64),
    )
    query_mask = np.broadcast_to(np.ma.getmaskarray(rows), row_values.shape)
    query_mask = query_mask | np.broadcast_to(
        np.ma.getmaskarray(cols), row_values.shape
    )
    height, width = values.shape[-2:]
    output = np.full((*values.shape[:-2], *row_values.shape), np.nan)
    output_valid = np.zeros(output.shape, dtype=bool)
    for image in np.ndindex(values.shape[:-2]):
        for query in np.ndindex(row_values.shape):
            row, col = float(row_values[query]), float(col_values[query])
            if query_mask[query] or not (math.isfinite(row) and math.isfinite(col)):
                continue
            if boundary == "clamp":
                row = min(max(row, 0.0), height - 1)
                col = min(max(col, 0.0), width - 1)
            if method == "nearest":
                rr, cc = math.floor(row + 0.5), math.floor(col + 0.5)
                if not (0 <= rr < height and 0 <= cc < width):
                    continue
                if not available[(*image, rr, cc)]:
                    continue
                sampled = values[(*image, rr, cc)]
            elif method == "bilinear":
                rr, cc = math.floor(row), math.floor(col)
                fr, fc = row - rr, col - cc
                sampled = 0.0
                supported = True
                for dr, wr in ((0, 1 - fr), (1, fr)):
                    for dc, wc in ((0, 1 - fc), (1, fc)):
                        weight = wr * wc
                        if wr == 0 or wc == 0:
                            continue
                        r, c = rr + dr, cc + dc
                        if not (0 <= r < height and 0 <= c < width):
                            supported = False
                        elif not available[(*image, r, c)]:
                            supported = False
                        else:
                            sampled += weight * values[(*image, r, c)]
                if not supported:
                    continue
            else:
                rr, cc = math.floor(row), math.floor(col)
                if not (1 <= rr <= height - 5 and 1 <= cc <= width - 5):
                    continue
                patch_key = (*image, slice(rr - 1, rr + 5), slice(cc - 1, cc + 5))
                if not available[patch_key].all():
                    continue
                patch = values[patch_key]
                along_columns = CubicSpline(
                    np.arange(6), patch, axis=-1, bc_type="natural"
                )(1 + col - cc)
                sampled = CubicSpline(np.arange(6), along_columns, bc_type="natural")(
                    1 + row - rr
                )
            if np.isfinite(sampled):
                output[(*image, *query)] = sampled
                output_valid[(*image, *query)] = True
    return output, output_valid
