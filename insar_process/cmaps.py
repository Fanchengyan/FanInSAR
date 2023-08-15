import matplotlib.colors as mcolors

white = '0.95'

colors = ["#8f07ff", "#d5734a", white, "#0571b0", "#01ef6c"]
GnBu_RdPl = mcolors.LinearSegmentedColormap.from_list(
    "GnBuRdPl", colors, N=100)

colors = [white, "#0571b0", "#8f07ff", "#d5734a"]
BuPl = mcolors.LinearSegmentedColormap.from_list("BuPl", colors, N=100)

colors = [white, "#fff7b3", "#fb9d59", '#aa0526']
HeatRed = mcolors.LinearSegmentedColormap.from_list("HeatRed", colors, N=100)
