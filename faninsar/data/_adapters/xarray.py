"""Xarray accessor registrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import rioxarray  # noqa: F401
import xarray as xr

from faninsar.io.export.kmz import dataarray2kmz
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from os import PathLike

    from faninsar.core.types import CrsLike

logger = setup_logger(__name__)

_AUTO_DIM_NAME = "auto"
_SPATIAL_DIM_PAIRS = (
    ("x", "y"),
    ("xs", "ys"),
    ("x_coord", "y_coord"),
    ("x_coords", "y_coords"),
    ("xcoord", "ycoord"),
    ("xcoords", "ycoords"),
    ("lon", "lat"),
    ("lons", "lats"),
    ("lng", "lat"),
    ("lngs", "lats"),
    ("long", "lat"),
    ("longs", "lats"),
    ("longitude", "latitude"),
    ("longitudes", "latitudes"),
    ("easting", "northing"),
    ("eastings", "northings"),
    ("east", "north"),
    ("easts", "norths"),
    ("col", "row"),
    ("cols", "rows"),
    ("column", "row"),
    ("columns", "rows"),
    ("sample", "line"),
    ("samples", "lines"),
    ("range", "azimuth"),
    ("ranges", "azimuths"),
)


def _resolve_auto_spatial_dims(
    data_array: xr.DataArray,
    x_dim: str,
    y_dim: str,
) -> tuple[str, str]:
    """Resolve spatial dimension names for a data array.

    Parameters
    ----------
    data_array : xarray.DataArray
        Data array whose dimensions are inspected.
    x_dim : str
        Name of the x dimension, or ``"auto"`` to infer it.
    y_dim : str
        Name of the y dimension, or ``"auto"`` to infer it.

    Returns
    -------
    tuple[str, str]
        Resolved x and y dimension names.

    Raises
    ------
    ValueError
        If either dimension cannot be inferred or is not present in the data
        array dimensions.

    """
    dimension_names = tuple(data_array.dims)
    lower_to_dimension = {name.lower(): name for name in dimension_names}

    resolved_x_dim = x_dim
    resolved_y_dim = y_dim
    if x_dim == _AUTO_DIM_NAME and y_dim == _AUTO_DIM_NAME:
        for x_candidate, y_candidate in _SPATIAL_DIM_PAIRS:
            if x_candidate in lower_to_dimension and y_candidate in lower_to_dimension:
                return lower_to_dimension[x_candidate], lower_to_dimension[y_candidate]

        msg = (
            "Cannot infer spatial dimensions automatically. Expected one of "
            f"{_SPATIAL_DIM_PAIRS}, but found dimensions {dimension_names}. "
            "Please pass x_dim and y_dim explicitly."
        )
        logger.error(msg)
        raise ValueError(msg)

    if x_dim == _AUTO_DIM_NAME:
        for x_candidate, y_candidate in _SPATIAL_DIM_PAIRS:
            if (
                y_candidate in lower_to_dimension
                and lower_to_dimension[y_candidate] == y_dim
                and x_candidate in lower_to_dimension
            ):
                resolved_x_dim = lower_to_dimension[x_candidate]
                break
        else:
            msg = (
                f"Cannot infer x dimension for y_dim={y_dim!r}. "
                f"Found dimensions {dimension_names}. Please pass x_dim explicitly."
            )
            logger.error(msg)
            raise ValueError(msg)

    if y_dim == _AUTO_DIM_NAME:
        for x_candidate, y_candidate in _SPATIAL_DIM_PAIRS:
            if (
                x_candidate in lower_to_dimension
                and lower_to_dimension[x_candidate] == x_dim
                and y_candidate in lower_to_dimension
            ):
                resolved_y_dim = lower_to_dimension[y_candidate]
                break
        else:
            msg = (
                f"Cannot infer y dimension for x_dim={x_dim!r}. "
                f"Found dimensions {dimension_names}. Please pass y_dim explicitly."
            )
            logger.error(msg)
            raise ValueError(msg)

    missing_dimensions = [
        dimension_name
        for dimension_name in (resolved_x_dim, resolved_y_dim)
        if dimension_name not in dimension_names
    ]
    if missing_dimensions:
        msg = (
            f"Spatial dimensions {missing_dimensions} are not present in the "
            f"data array dimensions {dimension_names}."
        )
        logger.error(msg)
        raise ValueError(msg)

    return resolved_x_dim, resolved_y_dim


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

    def set_spatial_ref(
        self,
        crs: CrsLike,
        x_dim: str = _AUTO_DIM_NAME,
        y_dim: str = _AUTO_DIM_NAME,
    ) -> xr.DataArray:
        """Set rioxarray spatial dimensions and coordinate reference system.

        Parameters
        ----------
        crs : CrsLike
            Coordinate reference system. It can be any value accepted by
            :meth:`pyproj.crs.CRS.from_user_input`.
        x_dim : str, optional
            Name of the x dimension. Use ``"auto"`` to infer a common spatial
            dimension name. Default is ``"auto"``.
        y_dim : str, optional
            Name of the y dimension. Use ``"auto"`` to infer a common spatial
            dimension name. Default is ``"auto"``.

        Returns
        -------
        xarray.DataArray
            Data array with rioxarray spatial dimensions and CRS metadata.

        Raises
        ------
        ValueError
            If spatial dimensions cannot be inferred or the selected dimension
            names are not present in the data array.

        Examples
        --------
        >>> import faninsar  # Register the ``fis`` accessor.
        >>> import xarray as xr
        >>> data_array = xr.DataArray([[1]], dims=("y", "x"))
        >>> spatial_ref = data_array.fis.set_spatial_ref(crs="EPSG:4326")
        >>> spatial_ref.rio.x_dim
        'x'

        """
        resolved_x_dim, resolved_y_dim = _resolve_auto_spatial_dims(
            self.data_array,
            x_dim,
            y_dim,
        )
        data_array = self.data_array.rio.set_spatial_dims(
            x_dim=resolved_x_dim,
            y_dim=resolved_y_dim,
        )
        return data_array.rio.write_crs(crs)

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
            Keyword arguments for :func:`faninsar.io.export.kmz.save_colorbar`,
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
