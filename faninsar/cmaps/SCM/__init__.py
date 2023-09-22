from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

cwd = Path(__file__).parent.absolute()

names = ['acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc',
         'brocO', 'buda', 'cork', 'corkO', 'davos', 'devon', 'grayC',
         'hawaii', 'imola', 'lajolla', 'lapaz', 'lisbon', 'nuuk',
         'oleron', 'oslo', 'roma', 'romaO', 'tofino', 'tokyo',
         'turku', 'vik', 'vikO']

__all__ = names.copy()

for name in names:
    file = cwd / name / f'{name}.txt'
    cm_data = np.loadtxt(file)

    __all__.append(f'{name}_r')

    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[f"{name}_r"] = LinearSegmentedColormap.from_list(
        f"{name}_r", cm_data[::-1])

del name
del names
del file
del cm_data
del cwd
del Path
del LinearSegmentedColormap
del np
