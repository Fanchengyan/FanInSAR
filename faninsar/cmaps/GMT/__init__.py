from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

names = [
    'abyss', 'bathy', 'cool', 'copper', 'cubhelix', 'cyclic', 'dem1', 'dem2', 'dem3', 'dem4', 'drywet', 'earth', 'elevation', 'etopo1', 'geo', 'globe', 'gray', 'haxby', 'hot', 'inferno', 'jet',
    'magma', 'nighttime', 'no_green', 'ocean', 'plasma', 'polar', 'rainbow', 'red2green', 'relief', 'seafloor', 'sealand', 'seis', 'split', 'srtm', 'terra', 'topo', 'turbo', 'viridis', 'world', 'wysiwyg'
]
__all__ = names.copy()


cwd = Path(__file__).parent.absolute()

for name in names:
    file = cwd / name / f'{name}.txt'
    cm_data = np.loadtxt(file)

    __all__.append(f'{name}_r')

    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[f"{name}_r"] = LinearSegmentedColormap.from_list(
        f"{name}_r", cm_data[::-1])

del name
del file
del cm_data
del cwd
del Path
del LinearSegmentedColormap
del np
