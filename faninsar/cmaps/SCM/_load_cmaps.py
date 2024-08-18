from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

version = "8.0.1"

cwd = Path(__file__).parent.absolute()

names = [
    "acton",
    "bam",
    "bamako",
    "bamO",
    "batlow",
    "batlowK",
    "batlowW",
    "berlin",
    "bilbao",
    "broc",
    "brocO",
    "buda",
    "bukavu",
    "cork",
    "corkO",
    "davos",
    "devon",
    "fes",
    "glasgow",
    "grayC",
    "hawaii",
    "imola",
    "lajolla",
    "lapaz",
    "lisbon",
    "lipari",
    "managua",
    "navia",
    "nuuk",
    "oleron",
    "oslo",
    "roma",
    "romaO",
    "tofino",
    "tokyo",
    "turku",
    "vanimo",
    "vik",
    "vikO",
]

_all = names.copy()

for name in names:
    file = cwd / name / f"{name}.txt"
    cm_data = np.loadtxt(file)

    _all.append(f"{name}_r")

    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[f"{name}_r"] = LinearSegmentedColormap.from_list(
        f"{name}_r",
        np.flip(cm_data, axis=0),
    )

__all__ = tuple(_all)

del name
del file
del cm_data
del cwd
del Path
del LinearSegmentedColormap
del np
