"""Xarray accessor registrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xarray as xr

from faninsar._core.geo.kmz import dataarray2kmz

if TYPE_CHECKING:
    from os import PathLike


@xr.register_dataarray_accessor("fis")
class FanInSARDataArrayAccessor:
    """FanInSAR convenience methods for :class:`xarray.DataArray`.

    Parameters
    ----------
    data_array : xarray.DataArray
        Data array bound to the accessor.

    Attributes
    ----------
    data_array : xarray.DataArray
        Data array bound to the accessor.

    """

    def __init__(self, data_array: xr.DataArray) -> None:
        self.data_array = data_array

    def to_kmz(
        self,
        out_file: PathLike,
        render_scale: int = 4,
        img_kwargs: dict[str, Any] | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        verbose: bool = True,
        *,
        tile_size: int = 256,
        min_lod_pixels: int = 128,
    ) -> None:
        """Write the data array into a KMZ file.

        Parameters
        ----------
        out_file : str or PathLike
            Path of the KMZ file.
        render_scale : int, optional
            Positive integer scale factor used to repeat source pixels before
            rendering, improving pixel-level clarity in Google Earth.
        img_kwargs : dict[str, Any] | None, optional
            Keyword arguments for :func:`matplotlib.pyplot.imshow`.
        cbar_kwargs : dict[str, Any] | None, optional
            Keyword arguments for :func:`faninsar._core.geo.save_colorbar`,
            excluding ``out_file`` and ``mappable``.
        verbose : bool, optional
            Whether to log the output path.
        tile_size : int, optional
            Maximum tile size in pixels.
        min_lod_pixels : int, optional
            Minimum screen-space threshold used by child ``NetworkLink``
            regions.

        Raises
        ------
        ValueError
            If the data array does not have rioxarray CRS or spatial dimension
            metadata, or if any tiling parameter is invalid.

        """
        dataarray2kmz(
            self.data_array,
            out_file,
            render_scale=render_scale,
            img_kwargs=img_kwargs,
            cbar_kwargs=cbar_kwargs,
            verbose=verbose,
            tile_size=tile_size,
            min_lod_pixels=min_lod_pixels,
        )
