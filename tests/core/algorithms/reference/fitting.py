"""Independent polynomial evaluation and surface-design references."""

from __future__ import annotations

import numpy as np

from .linalg import reference_least_squares


def reference_polyval(coefficients, x):
    c = np.ma.asarray(coefficients)
    query = np.ma.asarray(x)
    invalid = np.ma.getmaskarray(query) | ~np.isfinite(query.data)
    if np.any(np.ma.getmaskarray(c) | ~np.isfinite(c.data)):
        return np.full(query.shape, np.nan)
    output = np.polynomial.polynomial.polyval(np.where(invalid, 0, query.data), c.data)
    return np.where(invalid, np.nan, output)


def reference_fit(source, exponents, *, valid=None, weights=None):
    source = np.ma.asarray(source)
    height, width = source.shape[-2:]
    rows = np.linspace(-1, 1, height) if height > 1 else np.zeros(1)
    cols = np.linspace(-1, 1, width) if width > 1 else np.zeros(1)
    xx, yy = np.meshgrid(cols, rows)
    design = np.column_stack([(xx**i * yy**j).ravel() for i, j in exponents])
    leading = source.shape[:-2]
    flat = source.reshape((-1, height * width)).T
    if valid is not None:
        valid = np.ma.asarray(valid)
        valid = (
            np.ma.MaskedArray(
                np.broadcast_to(valid.data, source.shape),
                mask=np.broadcast_to(np.ma.getmaskarray(valid), source.shape),
            )
            .reshape((-1, height * width))
            .T
        )
    if weights is not None:
        weights = np.ma.asarray(weights)
        weights = (
            np.ma.MaskedArray(
                np.broadcast_to(weights.data, source.shape),
                mask=np.broadcast_to(np.ma.getmaskarray(weights), source.shape),
            )
            .reshape((-1, height * width))
            .T
        )
    result = reference_least_squares(design, flat, weights=weights, valid=valid)
    return tuple(
        item.T.reshape((*leading, item.shape[0]))
        if item.ndim == 2
        else item.reshape(leading)
        for item in result
    )
