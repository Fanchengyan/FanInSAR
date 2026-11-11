"""Independent SciPy normalized-convolution references with explicit support."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter


def filter_reference(
    source,
    *,
    method,
    window=None,
    sigma=None,
    min_radius=None,
    weights=None,
    valid=None,
    boundary="constant",
):
    """Use SciPy's separate execution and explicit Gaussian radius policy."""
    data = np.asarray(np.ma.getdata(source), dtype=np.float64)
    allowed = np.isfinite(data) & ~np.ma.getmaskarray(source)
    if valid is not None:
        allowed &= np.asarray(np.ma.getdata(valid)) & ~np.ma.getmaskarray(valid)
    weight = (
        np.ones(data.shape)
        if weights is None
        else np.broadcast_to(
            np.asarray(np.ma.getdata(weights), dtype=np.float64), data.shape
        ).copy()
    )
    if weights is not None:
        allowed &= ~np.ma.getmaskarray(weights)
    weight = np.where(allowed, weight, 0.0)
    numerator = np.where(allowed, data, 0.0) * weight
    mode = "mirror" if boundary == "reflect" else "constant"
    if method == "gaussian":
        minimum = 1 if min_radius is None else min_radius
        radii = tuple(max(minimum, int(3 * value + 0.5)) for value in sigma)
        # Zero sigmas on batch axes are SciPy's explicit identity convention.
        sigmas = (0.0,) * (data.ndim - 2) + tuple(sigma)
        support = (0,) * (data.ndim - 2) + radii
        numerator = gaussian_filter(numerator, sigmas, radius=support, mode=mode)
        denominator = gaussian_filter(weight, sigmas, radius=support, mode=mode)
    else:
        denominator = weight
        for axis, length in zip((-2, -1), window, strict=True):
            kernel = np.ones(length) / length
            numerator = convolve1d(numerator, kernel, axis=axis, mode=mode)
            denominator = convolve1d(denominator, kernel, axis=axis, mode=mode)
    result = np.full(data.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result, denominator > 0
