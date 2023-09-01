from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

cwd = Path(__file__).parent.absolute()

__all__ = ['acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc',
           'brocO', 'buda', 'cork', 'corkO', 'davos', 'devon', 'grayC',
           'hawaii', 'imola', 'lajolla', 'lapaz', 'lisbon', 'nuuk',
           'oleron', 'oslo', 'roma', 'romaO', 'tofino', 'tokyo',
           'turku', 'vik', 'vikO']


for name in __all__:
    file = cwd / name / f'{name}.txt'
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
