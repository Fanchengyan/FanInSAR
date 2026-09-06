"""String formatting routines for __repr__."""

from __future__ import annotations

import numpy as np
from xarray.core.formatting import short_array_repr


def short_data_repr(array) -> str:  # noqa: ANN001
    """Format "data" for Acquisition and other classes in FanInSAR."""
    if isinstance(array, np.ndarray):
        return short_array_repr(array)
    if getattr(array, "_in_memory", None):
        return short_array_repr(array)
    # internal xarray array type
    return f"[{array.size} values with dtype={array.dtype}]"
