from matplotlib.figure import Figure, SubFigure

from .formatters import PiFormatter, PiLocator
from .hist_colorbar import HistColorbar, _hist_colorbar

Figure.hist_colorbar = _hist_colorbar
SubFigure.hist_colorbar = _hist_colorbar

__all__ = ["HistColorbar", "PiFormatter", "PiLocator"]
