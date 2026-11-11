"""Independent, deliberately scalar NumPy reference for block reduction."""

from __future__ import annotations

import itertools

import numpy as np


def block_reference(
    source,
    block_shape,
    *,
    method="mean",
    valid=None,
    invalid="propagate",
    remainder="drop",
):
    """Reduce each available block without importing production algorithms."""
    data = np.asarray(np.ma.getdata(source), dtype=np.float64)
    allowed = np.isfinite(data) & ~np.ma.getmaskarray(source)
    if valid is not None:
        allowed &= np.asarray(valid) & ~np.ma.getmaskarray(valid)
    by, bx = block_shape
    height, width = data.shape[-2:]
    rows = (height + by - 1) // by if remainder == "partial" else height // by
    cols = (width + bx - 1) // bx if remainder == "partial" else width // bx
    result = np.full((*data.shape[:-2], rows, cols), np.nan)
    admission = np.zeros(result.shape, dtype=bool)
    reducers = {"mean": np.mean, "sum": np.sum, "min": np.min, "max": np.max}
    for batch in itertools.product(*(range(size) for size in data.shape[:-2])):
        for row in range(rows):
            for col in range(cols):
                index = (
                    *batch,
                    slice(row * by, (row + 1) * by),
                    slice(col * bx, (col + 1) * bx),
                )
                members = data[index]
                selected = allowed[index]
                if selected.any() and (invalid == "omit" or selected.all()):
                    result[(*batch, row, col)] = reducers[method](members[selected])
                    admission[(*batch, row, col)] = True
    return result, admission
