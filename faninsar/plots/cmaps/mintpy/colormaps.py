"""Mintpy Colormap Loader with dynamic loading capability."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import numpy as np

from faninsar.logging import setup_logger

from .. import ColormapLoader  # noqa: TID252

if TYPE_CHECKING:
    from os import PathLike

    from ..enhanced_colormap import EnhancedLinearSegmentedColormap  # noqa: TID252


logger = setup_logger(__name__)


def cpt_to_colormap(
    cpt_file: PathLike, name: str | None = None
) -> mcolors.LinearSegmentedColormap:
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
    cmap = mcolors.LinearSegmentedColormap.from_list(name, rgb_data, N=256)

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


class MintpyColormaps(ColormapLoader):
    """Mintpy colormap loader with dynamic loading.

    This loader provides access to mintpy colormaps with dynamic loading capability.
    mintpy provides specialized colormaps for InSAR processing.
    """

    def __init__(self) -> None:
        """Initialize mintpy colormap loader."""
        super().__init__(Path(__file__).parent.absolute())
        self._colormap_names = ["romanian"]
        self._custom_colormaps: dict[str, mcolors.LinearSegmentedColormap] = {}
        self._create_custom_colormaps()

    def _create_custom_colormaps(self) -> None:
        """Create custom colormaps that are defined programmatically."""
        # Create dismph colormap
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
        # Import here to avoid circular imports
        from faninsar.cmaps.enhanced_colormap import (
            EnhancedLinearSegmentedColormap,
        )

        dismph = EnhancedLinearSegmentedColormap.from_list("dismph", clist, N=256)
        dismph.set_bad("w", 0.0)
        self._custom_colormaps["dismph"] = dismph

        # Create cmy colormap
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
        rgbs = np.flipud(
            rgbs
        )  # flip up-down so that orange is in the later half (positive)

        cmy = EnhancedLinearSegmentedColormap.from_list("cmy", rgbs / 255.0, N=256)
        self._custom_colormaps["cmy"] = cmy

    @property
    def names(self) -> list[str]:
        """Return list of available mintpy colormap names."""
        return self._colormap_names + list(self._custom_colormaps.keys())

    def _load_colormap_data(self, name: str) -> np.ndarray:
        """Load mintpy colormap data from file.

        Args:
            name: Name of the colormap

        Returns:
            Numpy array containing colormap data

        """
        # Check if it's a custom colormap first
        if name in self._custom_colormaps:
            # For custom colormaps, we return the colormap directly in _get_colormap
            # This method shouldn't be called for custom colormaps
            msg = f"Custom colormap {name} should not use _load_colormap_data"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)

        # For file-based colormaps (like romanian)
        cmap_file = self.data_dir / name / f"{name}.cpt"
        if not cmap_file.exists():
            msg = f"Mintpy colormap file not found: {cmap_file}"
            logger.error(msg, stacklevel=2)
            raise FileNotFoundError(msg)

        # Load CPT file and convert to colormap, then extract data
        cmap = cpt_to_colormap(cmap_file, name)
        # Extract the colormap data as RGB values
        return np.array([cmap(i) for i in np.linspace(0, 1, 256)])[:, :3]

    def _get_colormap(self, name: str) -> mcolors.LinearSegmentedColormap:
        """Get a colormap, loading it if necessary.

        Override to handle custom colormaps.
        """
        # Check if it's a reversed colormap
        is_reversed = name.endswith("_r")
        base_name = name[:-2] if is_reversed else name

        # Check if base colormap is available
        if base_name not in self.names:
            msg = f"Colormap '{base_name}' not found"
            raise AttributeError(msg)

        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Handle custom colormaps
        if base_name in self._custom_colormaps:
            if base_name not in self._cache:
                self._cache[base_name] = self._custom_colormaps[base_name]

            if is_reversed and name not in self._cache:
                # Create reversed version of custom colormap
                from ..enhanced_colormap import (  # noqa: TID252
                    EnhancedLinearSegmentedColormap,
                )

                base_cmap = self._custom_colormaps[base_name]
                colors = [base_cmap(i) for i in np.linspace(0, 1, 256)]
                reversed_colors = colors[::-1]
                self._cache[name] = EnhancedLinearSegmentedColormap.from_list(
                    name, reversed_colors
                )

            return self._cache[name]

        # Handle file-based colormaps using parent method
        return super()._get_colormap(name)


# Create a global instance
_mintpy_colormaps = MintpyColormaps()

# Export all colormap names for backward compatibility
__all__ = _mintpy_colormaps.__all__


# Create module-level attributes for backward compatibility
def __getattr__(name: str) -> EnhancedLinearSegmentedColormap:
    """Module-level attribute access for colormaps."""
    return getattr(_mintpy_colormaps, name)


def __dir__() -> list[str]:
    """Return list of available module attributes."""
    return _mintpy_colormaps.__all__


# For convenience, also expose the names
names = _mintpy_colormaps.names
