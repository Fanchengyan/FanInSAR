from matplotlib.figure import Figure, SubFigure

from .cm import cmaps
from .formatters import PiFormatter, PiLocator, setup_phase_axis
from .hist_colorbar import HistColorbar, _hist_colorbar
from .ucm import (
    DEFAULT_UCM_MOSAIC,
    create_ucm_mosaic,
    plot_ucm,
    plot_ucm_heatmap,
    plot_ucm_spatial_profile,
    plot_ucm_surface_3d,
    plot_ucm_temporal_profile,
)
from .utils import HandlerGradientLine, create_discrete_colormap

Figure.hist_colorbar = _hist_colorbar
SubFigure.hist_colorbar = _hist_colorbar
