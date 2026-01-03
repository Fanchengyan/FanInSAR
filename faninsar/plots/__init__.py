from matplotlib.figure import Figure, SubFigure

from .formatters import PiFormatter, PiLocator, setup_phase_axis
from .hist_colorbar import HistColorbar, _hist_colorbar
from .utils import create_discrete_colormap, HandlerGradientLine

Figure.hist_colorbar = _hist_colorbar
SubFigure.hist_colorbar = _hist_colorbar
