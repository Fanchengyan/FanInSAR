"""Xarray accessors for geospatial export helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xarray as xr

from .kml import dataarray2kml, dataarray2kmz

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

    def to_kml(
        self,
        out_file: PathLike,
        img_kwargs: dict[str, Any] | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> None:
        """Write the data array into a KML file.

        Parameters
        ----------
        out_file : str or PathLike
            Path of the KML file.
        img_kwargs : dict[str, Any] | None, optional
            Keyword arguments for :func:`matplotlib.pyplot.imshow`.
        cbar_kwargs : dict[str, Any] | None, optional
            Keyword arguments for :func:`faninsar._core.geo.save_colorbar`,
            excluding ``out_file`` and ``mappable``.
        verbose : bool, optional
            Whether to log the output path.

        Raises
        ------
        ValueError
            If the data array does not have rioxarray CRS or spatial dimension
            metadata.

        """
        dataarray2kml(
            self.data_array,
            out_file,
            img_kwargs=img_kwargs,
            cbar_kwargs=cbar_kwargs,
            verbose=verbose,
        )

    def to_kmz(
        self,
        out_file: PathLike,
        img_kwargs: dict[str, Any] | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        verbose: bool = True,
        *,
        tiled: bool = False,
        tile_size: int = 256,
        min_lod_pixels: int = 128,
        render_scale: float = 1.0,
    ) -> None:
        """Write the data array into a KMZ file.

        Parameters
        ----------
        out_file : str or PathLike
            Path of the KMZ file.
        img_kwargs : dict[str, Any] | None, optional
            Keyword arguments for :func:`matplotlib.pyplot.imshow`.
        cbar_kwargs : dict[str, Any] | None, optional
            Keyword arguments for :func:`faninsar._core.geo.save_colorbar`,
            excluding ``out_file`` and ``mappable``.
        verbose : bool, optional
            Whether to log the output path.
        tiled : bool, optional
            Whether to write a tiled KMZ SuperOverlay instead of a single
            overlay.
        tile_size : int, optional
            Maximum tile size in pixels. Only used when ``tiled`` is True.
        min_lod_pixels : int, optional
            Minimum screen-space threshold used by child ``NetworkLink``
            regions. Only used when ``tiled`` is True.
        render_scale : float, optional
            Scale factor applied to the rendered image size before tiling. Only
            used when ``tiled`` is True.

        Raises
        ------
        ValueError
            If the data array does not have rioxarray CRS or spatial dimension
            metadata, or if any tiling parameter is invalid.

        """
        dataarray2kmz(
            self.data_array,
            out_file,
            img_kwargs=img_kwargs,
            cbar_kwargs=cbar_kwargs,
            verbose=verbose,
            tiled=tiled,
            tile_size=tile_size,
            min_lod_pixels=min_lod_pixels,
            render_scale=render_scale,
        )
