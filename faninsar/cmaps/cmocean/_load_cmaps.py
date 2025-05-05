from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

names = [
    "algae",
    "amp",
    "balance",
    "curl",
    "deep",
    "delta",
    "dense",
    "diff",
    "gray",
    "haline",
    "ice",
    "matter",
    "oxy",
    "phase",
    "rain",
    "solar",
    "speed",
    "tarn",
    "tempo",
    "thermal",
    "topo",
    "turbid",
]
_all = names.copy()


cwd = Path(__file__).parent.absolute()

for name in names:
    cmap_file = cwd / name / f"{name}.txt"
    cm_data = np.loadtxt(cmap_file)

    _all.append(f"{name}_r")

    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[f"{name}_r"] = LinearSegmentedColormap.from_list(f"{name}_r", cm_data[::-1])

__all__ = tuple(_all)
