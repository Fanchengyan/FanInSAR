"""Utility functions for plotting in FanInSAR."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap
from matplotlib.legend_handler import HandlerBase
from matplotlib.collections import LineCollection

class HandlerGradientLine(HandlerBase):
    """Custom handler for rendering gradient line in legend."""

    def __init__(self, cmap, norm, **kwargs):
        super().__init__(**kwargs)
        self._cmap = cmap
        self._norm = norm

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Create gradient line segments
        n_segments = 50
        x = np.linspace(xdescent, xdescent + width, n_segments + 1)
        y_mid = ydescent + height / 2
        
        points = np.array([x, np.full_like(x, y_mid)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(
            segments, cmap=self._cmap, norm=self._norm, linewidth=1.5
        )
        lc.set_array(np.linspace(self._norm.vmin, self._norm.vmax, n_segments))
        lc.set_transform(trans)
        return [lc]

def create_discrete_colormap(
    unique_values: np.ndarray, cmap: Colormap | str
) -> tuple[ListedColormap, BoundaryNorm]:
    """Create discrete colormap and normalization from unique values.

    Parameters
    ----------
    unique_values : np.ndarray
        Array of unique values to create discrete colormap for.
    cmap : str
        Base colormap name.

    Returns
    -------
    tuple[ListedColormap, BoundaryNorm]
        A tuple containing the discrete colormap and BoundaryNorm object.

    """
    n_colors = len(unique_values)
    base_cmap = plt.get_cmap(cmap)
    colors = base_cmap(np.linspace(0, 1, n_colors))
    discrete_cmap = ListedColormap(colors)

    # Create boundaries
    if n_colors == 1:
        boundaries = [unique_values[0] - 0.5, unique_values[0] + 0.5]
    else:
        boundaries = [unique_values[0] - (unique_values[1] - unique_values[0]) / 2]
        for i in range(len(unique_values) - 1):
            boundaries.append((unique_values[i] + unique_values[i + 1]) / 2)
        boundaries.append(
            unique_values[-1] + (unique_values[-1] - unique_values[-2]) / 2
        )

    return discrete_cmap, BoundaryNorm(boundaries, discrete_cmap.N)
