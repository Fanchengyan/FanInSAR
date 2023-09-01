import matplotlib.colors as mcolors

from .SCM import *
from .SCM import __all__ as SCM__all__

__all__ = SCM__all__.copy()
__all__.extend(['GnBu_RdPl', 'WtBuPl', 'WtHeatRed'])

white = '0.95'

colors = ["#8f07ff", "#d5734a", white, "#0571b0", "#01ef6c"]
GnBu_RdPl = mcolors.LinearSegmentedColormap.from_list(
    "GnBuRdPl", colors, N=100)

colors = [white, "#0571b0", "#8f07ff", "#d5734a"]
WtBuPl = mcolors.LinearSegmentedColormap.from_list("BuPl", colors, N=100)

colors = [white, "#fff7b3", "#fb9d59", '#aa0526']
WtHeatRed = mcolors.LinearSegmentedColormap.from_list("HeatRed", colors, N=100)

del vars()['mcolors']
del vars()['colors']
del vars()['white']
del vars()['SCM__all__']