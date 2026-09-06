"""Enhanced LinearSegmentedColormap with additional tools functionality.

This module provides an enhanced colormap class that extends matplotlib's
LinearSegmentedColormap with additional utility methods for colormap manipulation.

Reference
---------
https://github.com/matplotlib/cmocean/blob/main/cmocean/tools.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.colors as mcolors
import numpy as np

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import ArrayLike


class EnhancedLinearSegmentedColormap(mcolors.LinearSegmentedColormap):
    """Enhanced LinearSegmentedColormap with additional utility methods.

    This class extends matplotlib's LinearSegmentedColormap to provide
    additional functionality for colormap manipulation, including cropping,
    merging, transparentizing, RGB extraction, and saving to file.

    All methods from the original LinearSegmentedColormap are preserved,
    with additional utility methods for advanced colormap operations.
    """

    def __add__(
        self, other: EnhancedLinearSegmentedColormap
    ) -> EnhancedLinearSegmentedColormap:
        """Merge two colormaps.

        Parameters
        ----------
        other : EnhancedLinearSegmentedColormap
            Other colormap to add

        Returns
        -------
        EnhancedLinearSegmentedColormap
            New merged colormap

        """
        return self.merge(other)

    def to_rgb_array(self, N: int = 256) -> np.ndarray:
        """Extract RGB values from the colormap.

        Parameters
        ----------
        N : int, default 256
            Number of color samples to extract

        Returns
        -------
        np.ndarray
            Array of shape (N, 3) containing RGB values in range [0, 1]

        """
        return self(np.linspace(0, 1, N))[:, :3]

    def save_rgb(self, filename: PathLike, N: int = 256) -> None:
        """Save colormap RGB values to a text file.

        Parameters
        ----------
        filename : PathLike
            Output filename for the RGB data
        N : int, default 256
            Number of color samples to save

        """
        rgb_array = self.to_rgb_array(N)
        np.savetxt(filename, rgb_array)

    def to_dict(self, N: int = 256) -> dict[str, list[tuple[float, float, float]]]:
        """Convert colormap to dictionary format for LinearSegmentedColormap.

        This method converts the colormap to the dictionary format expected
        by matplotlib's LinearSegmentedColormap constructor.

        Parameters
        ----------
        N : int, default 256
            Number of samples for the conversion

        Returns
        -------
        dict
            Dictionary with 'red', 'green', 'blue' keys containing tuples
            of (position, left_value, right_value) for each color channel

        Notes
        -----
        Code adapted from:
        https://mycarta.wordpress.com/2014/04/25/convert-color-palettes-to-python-matplotlib-colormaps/

        """
        x = np.linspace(0, 1, N)
        rgb = self(x)

        # Flip colormap to follow matplotlib standard if needed
        if rgb[0, :].sum() < rgb[-1, :].sum():
            rgb = np.flipud(rgb)

        # Extract color channels
        r_values = rgb[:, 0]
        g_values = rgb[:, 1]
        b_values = rgb[:, 2]

        # Create tuples for each channel
        r = list(zip(x, r_values, r_values, strict=True))
        g = list(zip(x, g_values, g_values, strict=True))
        b = list(zip(x, b_values, b_values, strict=True))

        return {"red": r, "green": g, "blue": b}

    def merge(
        self, other: EnhancedLinearSegmentedColormap
    ) -> EnhancedLinearSegmentedColormap:
        """Merge two colormaps.

        Parameters
        ----------
        other : EnhancedLinearSegmentedColormap
            Other colormap to merge with

        Returns
        -------
        EnhancedLinearSegmentedColormap
            New merged colormap

        """
        name = f"{self.name}_{other.name}"
        colors = np.concatenate(
            [
                self(np.linspace(0, 1, self.N)),
                other(np.linspace(0, 1, other.N)),
            ]
        )
        return self.from_list(name, colors, N=self.N + other.N)

    def transparentize(self, alpha: float) -> EnhancedLinearSegmentedColormap:
        """Make colormap transparent.

        Parameters
        ----------
        alpha : float
            Alpha/transparency value to apply. 1.0 is opaque, 0.0 is transparent

        Returns
        -------
        EnhancedLinearSegmentedColormap
            New transparent colormap

        Notes
        -----
        This creates a new colormap with reduced opacity, which will appear
        lighter when used in plots.

        """
        rgba_array = self(np.linspace(0, 1, self.N))
        rgba_array[:, 3] = alpha  # Set alpha channel

        return self.from_list(f"{self.name}_transparentize", rgba_array, N=self.N)

    def crop(
        self,
        vmin: float,
        vmax: float,
        pivot: float,
        N: int | None = None,
        dmax: float | None = None,
    ) -> EnhancedLinearSegmentedColormap:
        """Crop ends of a diverging colormap by vmin/vmax values.

        Parameters
        ----------
        vmin : float
            Minimum value for use in plot with colormap
        vmax : float
            Maximum value for use in plot with colormap
        pivot : float
            Center point for diverging colormap
        N : int, optional
            Number of colors in output colormap. If None, uses original N
        dmax : float, optional
            Maximum magnitude to include. Used when vmin and vmax are
            symmetric around pivot

        Returns
        -------
        EnhancedLinearSegmentedColormap
            New cropped colormap

        Raises
        ------
        AssertionError
            If pivot is not between vmin and vmax, or if dmax is required
            but not provided for symmetric ranges

        Notes
        -----
        This function is useful for plotting bathymetry and topography data
        where the data range is asymmetric around the pivot point.

        """
        if not (vmin <= pivot <= vmax):
            msg = "Pivot must be between vmin and vmax"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)

        # dmax required if ranges are equal
        if abs((vmax - pivot) - (pivot - vmin)) < 1e-10 and dmax is None:
            msg = "dmax required when ranges are symmetric"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)

        # Use original N if not specified
        if N is None:
            N = self.N  # noqa: N806

        # Calculate ranges
        below = pivot - vmin  # below pivot
        above = vmax - pivot  # above pivot

        ranges = (above, below)
        half_range = max(ranges)
        full_range = half_range * 2
        reduced_range = min(ranges)
        range_to_keep = half_range + reduced_range

        ratio = (full_range - range_to_keep) / full_range

        # Get original colormap data
        original_colors = self(np.linspace(0, 1, N))

        if below < above:  # reducing colormap on side below pivot
            # Start colormap partway through
            shortcmap = original_colors[int(np.ceil(N * ratio)) :]
        elif above < below:  # reducing colormap on side above pivot
            # End colormap early
            shortcmap = original_colors[: -int(np.ceil(N * ratio))]
        elif dmax is not None:  # equal ranges
            ratio = dmax / full_range
            shortcmap = original_colors[
                int(np.ceil(N * ratio)) : -int(np.ceil(N * ratio))
            ]
        else:
            shortcmap = original_colors

        # Interpolate to original number of rows
        newrgb = np.zeros((N, 4))
        shnum = shortcmap.shape[0]
        for i in range(4):  # Loop through RGBA channels
            newrgb[:, i] = np.interp(
                np.linspace(0, shnum, N), np.arange(0, shnum), shortcmap[:, i]
            )

        return self.from_list(f"{self.name}_cropped", newrgb, N=N)

    def crop_by_percent(
        self,
        percent: float,
        which: Literal["both", "min", "max"] = "both",
        N: int | None = None,
    ) -> EnhancedLinearSegmentedColormap:
        """Crop ends of colormap by percentage.

        This is a convenience wrapper around the crop() method that makes
        it easier to crop based on percentages rather than data values.

        Parameters
        ----------
        percent : float
            Percentage of colormap to remove
        which : {"both", "min", "max"}, default "both"
            Which end(s) to crop:
            - "both": remove from both ends
            - "min": remove from bottom end only
            - "max": remove from top end only
        N : int, optional
            Number of colors in output colormap. If None, uses original N

        Returns
        -------
        EnhancedLinearSegmentedColormap
            New cropped colormap

        """
        if which == "both":  # Remove from both ends
            vmin = -100
            vmax = 100
            pivot = 0
            dmax = percent
        elif which == "min":  # Remove from bottom
            vmax = 10
            pivot = 5
            vmin = (0 + percent / 100) * 2 * pivot
            dmax = None
        elif which == "max":  # Remove from top
            vmin = 0
            pivot = 5
            vmax = (1 - percent / 100) * 2 * pivot
            dmax = None
        else:
            msg = "which must be 'both', 'min', or 'max'"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)

        return self.crop(vmin, vmax, pivot, N=N, dmax=dmax)

    @staticmethod
    def from_list(
        name: str, colors: ArrayLike, N: int = 256, gamma: float = 1.0
    ) -> EnhancedLinearSegmentedColormap:
        """Create an EnhancedLinearSegmentedColormap from a list of colors.

        Parameters
        ----------
        name : str
            Name for the new colormap
        colors : array-like
            List of colors or color specifications
        N : int, default 256
            Number of colors in the output colormap
        gamma : float, default 1.0
            Gamma correction factor

        Returns
        -------
        EnhancedLinearSegmentedColormap
            New enhanced colormap created from the color list

        """
        # Create a regular LinearSegmentedColormap first
        regular_cmap = mcolors.LinearSegmentedColormap.from_list(
            name, colors, N=N, gamma=gamma
        )

        # Create our enhanced version by copying the segment data
        # Access the private attribute safely
        segmentdata = getattr(regular_cmap, "_segmentdata", {})
        enhanced_cmap = EnhancedLinearSegmentedColormap(
            name, segmentdata, N=N, gamma=gamma
        )

        # Copy any special color settings safely
        for attr in ["_rgba_bad", "_rgba_under", "_rgba_over"]:
            if hasattr(regular_cmap, attr):
                setattr(enhanced_cmap, attr, getattr(regular_cmap, attr))

        return enhanced_cmap

    @classmethod
    def from_rgb_array(
        cls, name: str, rgb_array: np.ndarray, N: int = 256
    ) -> EnhancedLinearSegmentedColormap:
        """Create colormap from RGB array.

        Parameters
        ----------
        name : str
            Name for the new colormap
        rgb_array : np.ndarray
            Array of RGB values, shape (m, 3) where m is number of colors.
            Values should be in range [0, 1] or [0, 255]
        N : int, default 256
            Number of colors in the output colormap

        Returns
        -------
        EnhancedLinearSegmentedColormap
            New colormap created from the RGB array

        Raises
        ------
        ValueError
            If RGB values are not in range [0, 1] or [0, 255]

        """
        # Make a copy to avoid modifying the original
        rgb_array = np.array(rgb_array, dtype=float)

        # Normalize to [0, 1] if values appear to be in [0, 255] range
        if rgb_array.max() > 1:
            if rgb_array.max() > 255:
                msg = (
                    "RGB values should be in range [0, 1] or [0, 255], "
                    f"but got {rgb_array.max()}"
                )
                logger.error(msg, stacklevel=2)
                raise ValueError(msg)
            rgb_array = rgb_array / 255

        return cls.from_list(name, rgb_array, N=N)

    @classmethod
    def from_hex_colors(
        cls, name: str, hex_colors: list[str], N: int = 256
    ) -> EnhancedLinearSegmentedColormap:
        """Create colormap from list of hex color strings.

        Parameters
        ----------
        name : str
            Name for the new colormap
        hex_colors : list of str
            List of hex color strings (e.g., ['#FF0000', '#00FF00', '#0000FF'])
        N : int, default 256
            Number of colors in the output colormap

        Returns
        -------
        EnhancedLinearSegmentedColormap
            New colormap created from the hex colors

        """
        return cls.from_list(name, hex_colors, N=N)
