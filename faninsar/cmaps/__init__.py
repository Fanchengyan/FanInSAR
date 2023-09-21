import matplotlib.colors as mcolors

from .cmocean import *
from .cmocean import __all__ as cmocean__all__
from .colorcet import *
from .colorcet import __all__ as colorcet__all__
from .GMT import *
from .GMT import __all__ as GMT__all__
from .SCM import *
from .SCM import __all__ as SCM__all__

__all__ = SCM__all__.copy()
__all__.extend(GMT__all__)
__all__.extend(cmocean__all__)
__all__.extend(colorcet__all__)
__all__.extend(['GnBu_RdPl', 'WtBuPl', 'WtHeatRed'])

white = '0.95'

colors = ["#8f07ff", "#d5734a", white, "#0571b0", "#01ef6c"]
GnBu_RdPl = mcolors.LinearSegmentedColormap.from_list(
    "GnBu_RdPl", colors, N=100)
GnBu_RdPl_r = mcolors.LinearSegmentedColormap.from_list(
    "GnBu_RdPl_r", colors[::-1], N=100)

colors = [white, "#0571b0", "#8f07ff", "#d5734a"]
WtBuPl = mcolors.LinearSegmentedColormap.from_list("WtBuPl", colors, N=100)
WtBuPl_r = mcolors.LinearSegmentedColormap.from_list(
    "WtBuPl_r", colors[::-1], N=100)

colors = [white, "#0571b0", "#01ef6c"]
WtBuGn = mcolors.LinearSegmentedColormap.from_list("WtBuGn", colors, N=100)
WtBuGn_r = mcolors.LinearSegmentedColormap.from_list(
    "WtBuGn_r", colors[::-1], N=100)

colors = [white, "#d5734a", "#8f07ff"]
WtRdPl = mcolors.LinearSegmentedColormap.from_list("WtRdPl", colors, N=100)
WtRdPl_r = mcolors.LinearSegmentedColormap.from_list(
    "WtRdPl_r", colors[::-1], N=100)

colors = [white, "#fff7b3", "#fb9d59", '#aa0526']
WtHeatRed = mcolors.LinearSegmentedColormap.from_list(
    "WtHeatRed", colors, N=100)
WtHeatRed_r = mcolors.LinearSegmentedColormap.from_list(
    "WtHeatRed_r", colors[::-1], N=100)


del vars()['mcolors']
del vars()['colors']
del vars()['white']
del vars()['SCM__all__']
