"""Plotting and rich representations for FanInSAR products.

The package owns the public plotting utilities and colormap registry.  Rich
HTML/SVG representations remain available from :mod:`faninsar.plotting.render`.
"""

from __future__ import annotations

from matplotlib.figure import Figure, SubFigure

from . import colormaps, render
from .colorbar import HistColorbar, _hist_colorbar
from .colormaps import Cmaps, ColormapLoader, EnhancedLinearSegmentedColormap, cmaps
from .formatters import PiFormatter, PiLocator, setup_phase_axis
from .ucm import DEFAULT_UCM_MOSAIC, UCM, create_ucm_mosaic
from .utils import HandlerGradientLine, create_discrete_colormap

Figure.hist_colorbar = _hist_colorbar
SubFigure.hist_colorbar = _hist_colorbar

__all__ = [
    "DEFAULT_UCM_MOSAIC",
    "UCM",
    "Cmaps",
    "ColormapLoader",
    "EnhancedLinearSegmentedColormap",
    "HandlerGradientLine",
    "HistColorbar",
    "PiFormatter",
    "PiLocator",
    "cmaps",
    "colormaps",
    "create_discrete_colormap",
    "create_ucm_mosaic",
    "render",
    "setup_phase_axis",
]
