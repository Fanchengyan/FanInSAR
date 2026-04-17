"""Helpers for writing geospatial metadata to xarray objects and NetCDF files."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import xarray as xr

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from os import PathLike

    from faninsar.typing import CrsLike

logger = setup_logger(__name__)


@overload
def write_geoinfo_into_ds(
    ds: xr.DataArray,
    var: None = None,
    crs: CrsLike = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat",
) -> xr.DataArray: ...
@overload
def write_geoinfo_into_ds(
    ds: xr.Dataset,
    var: str | tuple | list,
    crs: CrsLike = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat",
) -> xr.Dataset: ...
def write_geoinfo_into_ds(
    ds: xr.DataArray | xr.Dataset,
    var: str | tuple | list | None = None,
    crs: CrsLike = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat",
) -> xr.DataArray | xr.Dataset:
    """Write geoinformation in to the given xr DataArray or DataSet.

    Parameters
    ----------
    ds: xarray DataArray or DataSet object
        data to be written into geoinfo.If type of ds is DataSet,
        var should be set
    var: str, tuple or list
        variables that need to be added geoinformation
    crs: CrsLike
        the coordinate reference system. Could be any type that
        :meth:`rasterio.crs.CRS.from_user_input` accepts.
    x_dim: str
        the coordinate name that presents the x dimension
    y_dim: str
        the coordinate name that presents the y dimension

    """
    if isinstance(ds, xr.DataArray):
        ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
        ds = ds.rio.write_crs(crs)
    elif isinstance(var, str):
        ds[var] = ds[var].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
        ds[var] = ds[var].rio.write_crs(crs)
    elif isinstance(var, (tuple, list)):
        for var_ in var:
            ds[var_] = ds[var_].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
            ds[var_] = ds[var_].rio.write_crs(crs)
    elif var is None:
        msg = "Detected type of ds is a xr.Dataset. var must be set"
        raise TypeError(msg)
    else:
        msg = "var type must be one of [str,tuple,list]"
        raise TypeError(msg)
    return ds


def write_geoinfo_into_nc(
    nc_file: PathLike,
    var: str | tuple | list | None = None,
    crs: CrsLike = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat",
    encode_time: bool = False,
) -> None:
    """Write geoinformation in to the given nc file.

    This make it could be opened with geoinformation in QGIS directly.

    Parameters
    ----------
    nc_file: str or pathlib.Path object
        the path of nc file
    var: str, tuple or list
        variables that need to be added geoinformation
    crs: CrsLike
        the coordinate reference system. Could be any type that
        :meth:`rasterio.crs.CRS.from_user_input` accepts.
    x_dim: str
        the coordinate name that presents the x dimension
    y_dim: str
        the coordinate name that presents the y dimension
    encode_time: bool
        whether to encode the time since "2000-01-01 00:00:00" if
        "time" coordinate is exists. Default is False.

    """
    ds = xr.load_dataset(nc_file)

    ds = write_geoinfo_into_ds(ds, var, crs, x_dim, y_dim)

    encode = {}
    if encode_time:
        if "time" in ds:
            encode.update({"time": {"units": "days since 2000-01-01 00:00:00"}})
        else:
            info = (
                f'there is no "time" dimension in {nc_file}, '
                "encoding process will be ignored"
            )
            logger.warning(info)
    ds.to_netcdf(nc_file, encoding=encode)
