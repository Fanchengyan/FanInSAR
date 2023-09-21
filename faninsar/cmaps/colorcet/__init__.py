from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

names = [
    'CET_C1', 'CET_C1s', 'CET_C2', 'CET_C2s', 'CET_C4', 'CET_C4s', 'CET_C5', 'CET_C5s', 'CET_CBC1', 'CET_CBC2', 'CET_CBD1', 'CET_CBL1', 'CET_CBL2', 'CET_CBTC1', 'CET_CBTC2', 'CET_CBTD1', 'CET_CBTL1', 'CET_CBTL2', 'CET_D1', 'CET_D10', 'CET_D11', 'CET_D12', 'CET_D13', 'CET_D1A', 'CET_D2', 'CET_D3', 'CET_D4', 'CET_D6', 'CET_D7', 'CET_D8', 'CET_D9', 'CET_I1', 'CET_I2', 'CET_I3', 'CET_L1', 'CET_L10', 'CET_L11', 'CET_L12', 'CET_L13', 'CET_L14', 'CET_L15', 'CET_L16', 'CET_L17', 'CET_L18', 'CET_L19', 'CET_L2', 'CET_L3', 'CET_L4', 'CET_L5', 'CET_L6', 'CET_L7', 'CET_L8', 'CET_L9', 'CET_R1', 'CET_R2', 'CET_R3', 'colorwheel', 'bkr', 'bky', 'bwy', 'cwr', 'coolwarm', 'gwv', 'bjy', 'isolum', 'bgy', 'bgyw', 'kbc', 'blues', 'bmw', 'bmy', 'kgy', 'gray', 'dimgray', 'fire', 'kb', 'kg', 'kr', 'rainbow'
]
__all__ = names.copy()


cwd = Path(__file__).parent.absolute()

for name in names:
    file = cwd / f'{name}.txt'
    cm_data = np.loadtxt(file)

    __all__.append(f'{name}_r')

    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[f"{name}_r"] = LinearSegmentedColormap.from_list(
        f"{name}_r", cm_data[::-1])
