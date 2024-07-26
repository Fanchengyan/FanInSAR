import matplotlib.colors as mcolors

from . import GMT, SCM, cmocean, colorcet
from .cmocean import (
    algae,
    amp,
    balance,
    curl,
    deep,
    delta,
    dense,
    diff,
    gray,
    haline,
    ice,
    matter,
    oxy,
    phase,
    rain,
    solar,
    speed,
    tarn,
    tempo,
    thermal,
    topo,
    turbid,
)
from .colorcet import (
    bgy,
    bgyw,
    bjy,
    bkr,
    bky,
    blues,
    bmw,
    bmy,
    bwy,
    colorwheel,
    coolwarm,
    cwr,
    dimgray,
    fire,
    gray,
    gwv,
    isolum,
    kb,
    kbc,
    kg,
    kgy,
    kr,
    rainbow,
)
from .GMT import (
    abyss,
    bathy,
    cool,
    copper,
    cubhelix,
    cyclic,
    dem1,
    dem2,
    dem3,
    dem4,
    drywet,
    earth,
    elevation,
    etopo1,
    geo,
    globe,
    gray,
    haxby,
    hot,
    inferno,
    jet,
    magma,
    nighttime,
    no_green,
    ocean,
    plasma,
    polar,
    rainbow,
    red2green,
    relief,
    seafloor,
    sealand,
    seis,
    split,
    srtm,
    terra,
    topo,
    turbo,
    viridis,
    world,
    wysiwyg,
)
from .SCM import (
    acton,
    bam,
    bamako,
    bamO,
    batlow,
    batlowK,
    batlowW,
    berlin,
    bilbao,
    broc,
    brocO,
    buda,
    bukavu,
    cork,
    corkO,
    davos,
    devon,
    fes,
    glasgow,
    grayC,
    hawaii,
    imola,
    lajolla,
    lapaz,
    lipari,
    lisbon,
    managua,
    navia,
    nuuk,
    oleron,
    oslo,
    roma,
    romaO,
    tofino,
    tokyo,
    turku,
    vanimo,
    vik,
    vikO,
)

__all__ = ["GnBu_RdPl", "WtBuPl", "WtHeatRed", "RdGyBu"]
__all__ += SCM.__all__
__all__ += GMT.__all__
__all__ += cmocean.__all__
__all__ += colorcet.__all__


names = __all__.copy()[3:]

white = "0.95"

colors = [
    "#68011f",
    "#bb2832",
    "#e48066",
    "#fbccb4",
    "#ededed",
    "#c2ddec",
    "#6bacd1",
    "#2a71b2",
    "#0d3061",
]
RdGyBu = mcolors.LinearSegmentedColormap.from_list("RdGrBu", colors, N=100)

colors = ["#8f07ff", "#d5734a", white, "#0571b0", "#01ef6c"]
GnBu_RdPl = mcolors.LinearSegmentedColormap.from_list("GnBu_RdPl", colors, N=100)
GnBu_RdPl_r = mcolors.LinearSegmentedColormap.from_list(
    "GnBu_RdPl_r", colors[::-1], N=100
)

colors = [white, "#0571b0", "#8f07ff", "#d5734a"]
WtBuPl = mcolors.LinearSegmentedColormap.from_list("WtBuPl", colors, N=100)
WtBuPl_r = mcolors.LinearSegmentedColormap.from_list("WtBuPl_r", colors[::-1], N=100)

colors = [white, "#0571b0", "#01ef6c"]
WtBuGn = mcolors.LinearSegmentedColormap.from_list("WtBuGn", colors, N=100)
WtBuGn_r = mcolors.LinearSegmentedColormap.from_list("WtBuGn_r", colors[::-1], N=100)

colors = [white, "#d5734a", "#8f07ff"]
WtRdPl = mcolors.LinearSegmentedColormap.from_list("WtRdPl", colors, N=100)
WtRdPl_r = mcolors.LinearSegmentedColormap.from_list("WtRdPl_r", colors[::-1], N=100)

colors = [white, "#fff7b3", "#fb9d59", "#aa0526"]
WtHeatRed = mcolors.LinearSegmentedColormap.from_list("WtHeatRed", colors, N=100)
WtHeatRed_r = mcolors.LinearSegmentedColormap.from_list(
    "WtHeatRed_r", colors[::-1], N=100
)


del mcolors
del colors
del white
