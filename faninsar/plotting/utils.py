"""Utility functions for plotting in FanInSAR."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize
from matplotlib.legend_handler import HandlerBase


class HandlerGradientLine(HandlerBase):
    """Custom handler for rendering gradient line in legend."""

    def __init__(self, cmap: Colormap, norm: Normalize, **kwargs: Any) -> None:
        """Initialize a gradient-line legend handler.

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap
            Colormap used to color the legend line.
        norm : matplotlib.colors.Normalize
            Normalization mapping values into *cmap*.
        **kwargs : Any
            Additional keyword arguments forwarded to ``HandlerBase``.

        """
        super().__init__(**kwargs)
        self._cmap = cmap
        self._norm = norm

    def create_artists(
        self,
        _legend: Any,
        _orig_handle: Any,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        _fontsize: float,
        trans: Any,
    ) -> list[LineCollection]:
        """Create gradient segments for a legend entry.

        Parameters
        ----------
        _legend : Any
            Matplotlib legend requesting the artists.
        _orig_handle : Any
            Original legend handle, unused by this handler.
        xdescent, ydescent, width, height : float
            Bounding-box coordinates supplied by Matplotlib.
        _fontsize : float
            Legend font size, unused by this handler.
        trans : Any
            Matplotlib transform applied to the generated artists.

        Returns
        -------
        list[matplotlib.collections.LineCollection]
            The generated gradient line collection.

        """
        # Create gradient line segments
        n_segments = 50
        x = np.linspace(xdescent, xdescent + width, n_segments + 1)
        y_mid = ydescent + height / 2

        points = np.array([x, np.full_like(x, y_mid)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=self._cmap, norm=self._norm, linewidth=1.5)
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
