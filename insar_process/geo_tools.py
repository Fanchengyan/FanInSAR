from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import xarray as xr
from rasterio import transform
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from tqdm import tqdm


def geoinfo_from_latlon(lat, lon):
    west, south, east, north, width, height = (
        np.nanmin(lon),
        np.nanmin(lat),
        np.nanmax(lon),
        np.nanmax(lat),
        len(lon),
        len(lat),
    )

    xsize = (east - west) / (width - 1)
    ysize = (north - south) / (height - 1)
    return (west, north, xsize, ysize, width, height)


def transform_from_latlon(lat, lon) -> Affine:
    """get the rasterio.transform from latitude and longitude.
    the pixel location will shift from center to upper-left corner

    Parameters:
    -----------
    lat, lon: numpy.ndarray or list
        latitudes and longitudes
    """
    west, north, xsize, ysize, _, _ = geoinfo_from_latlon(lat, lon)

    tf = transform.from_origin(
        west - 0.5 * xsize,  # center to left
        north + 0.5 * ysize,  # center to top
        xsize,
        ysize,
    )
    return tf


def latlon_from_meta(meta: Dict) -> np.ndarray:
    '''get the latitude and longitude from rasterio meta data

    Parameters:
    -----------
    meta: dict
        the meta data of rasterio dataset. It can be get from
        rasterio.open().meta

    Returns:
    --------
    lat, lon: numpy.ndarray
    '''
    tf = meta["transform"]
    width = meta["width"]
    height = meta["height"]
    lon = tf.xoff + tf.a * np.arange(width) + tf.a*0.5
    lat = tf.yoff + tf.e * np.arange(height) + tf.e*0.5
    return lat, lon


def write_geoinfo_into_ds(
    ds: Union[xr.DataArray, xr.Dataset],
    vars: Optional[Union[str, Tuple, List]] = None,
    crs: Union[str, int, Dict] = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat"
):
    """write geoinformation in to the given xr DataArray or DataSet.

    Parameters:
    -----------
    ds: xarray DataArray or DataSet object
        data to be written into geoinfo.If type of ds is DataSet,
        vars should be set
    vars: str, tuple or list
        variables that need to be added geoinformation
    crs: str or int
        the coordinate reference system. Could be any type that
        rasterio.crs.from_user_input accepts, such as:
          - EPSG code/string
          - proj4 string
          - wkt string
          - dict of PROJ parameters or PROJ JSON
    x_dim/y_dim: str
        the coordinate name that presents the x/y dimension
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
        ds = ds.rio.write_crs(crs)
    else:
        if isinstance(vars, str):
            ds[vars] = ds[vars].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
            ds[vars] = ds[vars].rio.write_crs(crs)
        elif isinstance(vars, (tuple, list)):
            for var in vars:
                ds[var] = ds[var].rio.set_spatial_dims(
                    x_dim=x_dim, y_dim=y_dim)
                ds[var] = ds[var].rio.write_crs(crs)
        elif vars is None:
            raise TypeError(
                "Detected type of ds is a xr.Dataset." " vars must be set")
        else:
            raise TypeError("vars type must be one of [str,tuple,list]")
    return ds


def write_geoinfo_into_nc(
    nc_file, vars, crs="EPSG:4326", x_dim="lon", y_dim="lat", encode_time=False
):
    """write geoinformation in to the given nc file and making it could be
    opened with geoinformation in QGIS directly.

    Parameters:
    -----------
    nc_file: str or pathlib.Path object
        the path of nc file
    vars: str, tuple or list
        variables that need to be added geoinformation
    x_dim/y_dim: str
        the coordinate name that presents the x/y dimension
    crs: str or int
        the coordinate reference system. Could be any type that
        rasterio.crs.from_user_input accepts, such as:
          - EPSG code/string
          - proj4 string
          - wkt string
          - dict of PROJ parameters or PROJ JSON
    encode_time: bool
        whether to encode the time since "2000-01-01 00:00:00" if
        "time" coordinate is exists. Default is False.
    """
    ds = xr.load_dataset(nc_file)

    ds = write_geoinfo_into_ds(ds, vars, crs, x_dim, y_dim)

    encode = {}
    if encode_time:
        if "time" in ds:
            encode.update(
                {"time": {"units": "days since 2000-01-01 00:00:00"}})
        else:
            print(
                f'there is no "time" dimension in {nc_file}, '
                "encoding process will be ignored"
            )
    ds.to_netcdf(nc_file, encoding=encode)


def match_to_raster(
    src_arr,
    src_crs,
    src_tf,
    dst_height,
    dst_width,
    dst_crs,
    dst_tf,
    nodata=np.nan,
    resampleAlg=Resampling.nearest,
):
    if src_arr.ndim == 2:
        indexes = 1
        src_n_band = 1
        src_height, src_width = src_arr.shape
    elif src_arr.ndim == 3:
        src_n_band, src_height, src_width = src_arr.shape
        indexes = np.arange(1, src_n_band + 1).tolist()
    else:
        raise ValueError("dimension of src_arr must be 2 or 3")
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            count=src_n_band,
            height=src_height,
            width=src_width,
            dtype="float32",
            crs=src_crs,
            transform=src_tf,
        ) as src:
            src.write(src_arr, indexes)

            with MemoryFile() as memfile1:
                with memfile1.open(
                    driver="GTiff",
                    count=src_n_band,
                    height=dst_height,
                    width=dst_width,
                    dtype="float32",
                    crs=dst_crs,
                    transform=dst_tf,
                ) as dst:
                    if indexes == 1:
                        indexes = [1]
                    for i in tqdm(indexes, desc="matching raster"):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst.transform,
                            dst_crs=dst.crs,
                            resampling=resampleAlg,
                            dst_nodata=nodata,
                        )
                    arr_dst = dst.read(indexes)
    return arr_dst
