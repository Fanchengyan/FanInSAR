from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

names = [
    "colorwheel",
    "bkr",
    "bky",
    "bwy",
    "cwr",
    "coolwarm",
    "gwv",
    "bjy",
    "isolum",
    "bgy",
    "bgyw",
    "kbc",
    "blues",
    "bmw",
    "bmy",
    "kgy",
    "gray",
    "dimgray",
    "fire",
    "kb",
    "kg",
    "kr",
    "rainbow",
]
_all = names.copy()


cwd = Path(__file__).parent.absolute()

for name in names:
    file = cwd / f"{name}.csv"
    with Path(file).open() as f:
        f_new = f.readlines()[0]
    file = cwd / f_new
    cm_data = np.loadtxt(file, delimiter=",")

    _all.append(f"{name}_r")

    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[f"{name}_r"] = LinearSegmentedColormap.from_list(f"{name}_r", cm_data[::-1])

__all__ = tuple(_all)

del name
del file
del f_new
del cm_data
del cwd
del Path
del LinearSegmentedColormap
del np
