from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from os import PathLike


def cpt_to_colormap(
    cpt_file: PathLike, name: str | None = None
) -> LinearSegmentedColormap:
    """Convert a CPT file to a matplotlib colormap.

    Parameters
    ----------
    cpt_file : str or Path
        Path to the CPT file
    name : str, optional
        Name of the colormap. If None, uses the filename without extension

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The colormap created from the CPT file

    """
    cpt_file = Path(cpt_file)
    if name is None:
        name = cpt_file.stem

    # Read the CPT file, but exclude special lines for now
    data = np.loadtxt(cpt_file, comments=("#", "B", "F", "N"))

    # Extract RGB values and normalize to [0, 1]
    rgb_data = [[row[1] / 255, row[2] / 255, row[3] / 255] for row in data]

    # Create the colormap
    cmap = LinearSegmentedColormap.from_list(name, rgb_data, N=256)

    # Process special lines (B, F, N) if they exist
    with cpt_file.open() as f:
        for _line in f:
            line = _line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("B"):  # Background color (under)
                parts = line.split()
                if len(parts) >= 4:
                    r, g, b = (
                        int(parts[1]) / 255,
                        int(parts[2]) / 255,
                        int(parts[3]) / 255,
                    )
                    cmap.set_under((r, g, b))

            elif line.startswith("F"):  # Foreground color (over)
                parts = line.split()
                if len(parts) >= 4:
                    r, g, b = (
                        int(parts[1]) / 255,
                        int(parts[2]) / 255,
                        int(parts[3]) / 255,
                    )
                    cmap.set_over((r, g, b))

            elif line.startswith("N"):  # NaN color (bad)
                parts = line.split()
                if len(parts) >= 4:
                    r, g, b = (
                        int(parts[1]) / 255,
                        int(parts[2]) / 255,
                        int(parts[3]) / 255,
                    )
                    cmap.set_bad((r, g, b))

    # Default if N wasn't specified
    if not any(line.startswith("N") for line in cpt_file.open()):
        cmap.set_bad("w", 0.0)  # Set bad values to transparent white

    return cmap


clist = [
    "#f579cd",
    "#f67fc6",
    "#f686bf",
    "#f68cb9",
    "#f692b3",
    "#f698ad",
    "#f69ea7",
    "#f6a5a1",
    "#f6ab9a",
    "#f6b194",
    "#f6b78e",
    "#f6bd88",
    "#f6c482",
    "#f6ca7b",
    "#f6d075",
    "#f6d66f",
    "#f6dc69",
    "#f6e363",
    "#efe765",
    "#e5eb6b",
    "#dbf071",
    "#d0f477",
    "#c8f67d",
    "#c2f684",
    "#bbf68a",
    "#b5f690",
    "#aff696",
    "#a9f69c",
    "#a3f6a3",
    "#9cf6a9",
    "#96f6af",
    "#90f6b5",
    "#8af6bb",
    "#84f6c2",
    "#7df6c8",
    "#77f6ce",
    "#71f6d4",
    "#6bf6da",
    "#65f6e0",
    "#5ef6e7",
    "#58f0ed",
    "#52e8f3",
    "#4cdbf9",
    "#7bccf6",
    "#82c4f6",
    "#88bdf6",
    "#8eb7f6",
    "#94b1f6",
    "#9aabf6",
    "#a1a5f6",
    "#a79ef6",
    "#ad98f6",
    "#b392f6",
    "#b98cf6",
    "#bf86f6",
    "#c67ff6",
    "#cc79f6",
    "#d273f6",
    "#d86df6",
    "#de67f6",
    "#e561f6",
    "#e967ec",
    "#ed6de2",
    "#f173d7",
]
dismph = LinearSegmentedColormap.from_list("dismph", clist, N=256)
dismph.set_bad("w", 0.0)

dismph_r = LinearSegmentedColormap.from_list("dismph_r", clist[::-1], N=256)
dismph_r.set_bad("w", 0.0)


rgbs = np.zeros((256, 3), dtype=np.uint8)

for kk in range(85):
    rgbs[kk, 0] = kk * 3
    rgbs[kk, 1] = 255 - kk * 3
    rgbs[kk, 2] = 255

rgbs[85:170, 0] = rgbs[0:85, 2]
rgbs[85:170, 1] = rgbs[0:85, 0]
rgbs[85:170, 2] = rgbs[0:85, 1]

rgbs[170:255, 0] = rgbs[0:85, 1]
rgbs[170:255, 1] = rgbs[0:85, 2]
rgbs[170:255, 2] = rgbs[0:85, 0]

rgbs[255, 0] = 0
rgbs[255, 1] = 255
rgbs[255, 2] = 255

rgbs = np.roll(rgbs, int(256 / 2 - 214), axis=0)  # shift green to the center
rgbs = np.flipud(rgbs)  # flip up-down so that orange is in the later half (positive)

# color list --> colormap object
cmy = LinearSegmentedColormap.from_list("cmy", rgbs / 255.0, N=256)
cmy_r = LinearSegmentedColormap.from_list("cmy_r", rgbs[::-1] / 255.0, N=256)

names = ["romanian"]
_all = names.copy()


cwd = Path(__file__).parent.absolute()

for name in names:
    cmap_file = cwd / name / f"{name}.cpt"
    # Use the new function instead of np.loadtxt directly
    vars()[name] = cpt_to_colormap(cmap_file, name)
    vars()[f"{name}_r"] = LinearSegmentedColormap.from_list(
        f"{name}_r", list(reversed(vars()[name](np.linspace(0, 1, 256))))
    )
    _all.append(f"{name}_r")

__all__ = tuple(_all)
