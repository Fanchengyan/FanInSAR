import matplotlib.colors as mcolors

from . import GMT, SCM, cmocean, colorcet
from .cmocean import *
from .colorcet import *
from .GMT import *
from .SCM import *

__all__ = ['GnBu_RdPl', 'WtBuPl', 'WtHeatRed']
__all__ += SCM.__all__
__all__ += GMT.__all__
__all__ += cmocean.__all__
__all__ += colorcet.__all__

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


del mcolors
del colors
del white
