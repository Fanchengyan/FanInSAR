from matplotlib.figure import Figure, SubFigure

from .formatters import PiFormatter, PiLocator, setup_phase_axis
from .hist_colorbar import HistColorbar, _hist_colorbar
from .utils import HandlerGradientLine, create_discrete_colormap

Figure.hist_colorbar = _hist_colorbar
SubFigure.hist_colorbar = _hist_colorbar
