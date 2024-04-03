from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.enums import Resampling

from faninsar._core.pair_tools import Pairs
from faninsar.datasets.ifg import InterferogramDataset

from ..query.query import BoundingBox


class HyP3S1(InterferogramDataset):
    """A dataset manages the data of HyP3 Sentinel-1 product.

    `Hyp3 <https://hyp3-docs.asf.alaska.edu/>`_ is a service for processing
    Synthetic Aperture Radar (SAR) imagery. This class is used to manage the
    data of Hyp3 product.
    """

    pattern_unw = "*unw_phase.tif"
    pattern_coh = "*corr.tif"

    def __init__(
        self,
        root_dir: str = "data",
        paths_unw: Optional[Sequence[Union[str, Path]]] = None,
        paths_coh: Optional[Sequence[Union[str, Path]]] = None,
        crs: Optional[CRS] = None,
        res: Optional[Union[float, tuple[float, float]]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[Union[float, int, Any]] = None,
        roi: Optional[BoundingBox] = None,
        bands_unw: Optional[Sequence[str]] = None,
        bands_coh: Optional[Sequence[str]] = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        masked: bool = True,
        fill_nodata: bool = False,
        verbose=True,
        keep_common: bool = True,
    ) -> None:
        """Initialize a new HyP3S1 instance.

        Parameters
        ----------
        root_dir: str
            Root directory where dataset can be found.
        paths_unw: list of str, optional
            list of unwrapped interferogram file paths to use instead of searching
            for files in ``root_dir``. If None, files will be searched for in ``root_dir``.
        paths_coh: list of str, optional
            list of coherence file paths to use instead of searching for files in
            ``root_dir``. If None, files will be searched for in ``root_dir``.
        crs: CRS, optional
            the output term:`coordinate reference system (CRS)` of the dataset.
            If None, the CRS of the first file found will be used.
        res: float, optional
            resolution of the output dataset in units of CRS. If None, the resolution
            of the first file found will be used.
        dtype: numpy.dtype, optional
            data type of the output dataset. If None, the data type of the first
            file found will be used.
        nodata: float or int, optional
            no data value of the output dataset. If None, the no data value of the
            first file found will be used.
        roi: BoundingBox, optional
            region of interest to load from the dataset. If None, the union of all
            files bounds in the dataset will be used.
        bands_unw: list of str, optional
            names of bands to return (defaults to all bands) for unwrapped interferograms.
        bands_coh: list of str, optional
            names of bands to return (defaults to all bands) for coherence.
        cache: bool, optional
            if True, cache file handle to speed up repeated sampling
        resampling: Resampling, optional
            Resampling algorithm used when reading input files.
            Default: `Resampling.nearest`.
        masked : bool, optional
            if True, the returned will be a masked array with a mask
            for no data values. Default: True.

            .. note::
                If parameter ``fill_nodata`` is True, the array will be interpolated and the returned array will always be a normal numpy array.
        fill_nodata : bool, optional
            Whether to fill holes in raster data by interpolation using the
            ``rasterio.fill.fillnodata`` function. Default: False.
        verbose: bool, optional
            if True, print verbose output.
        keep_common: bool, optional, default: True
            Only used when the number of interferograms and coherence files are
            not equal. If True, keep the common pairs of interferograms and
            coherence files and raise a warning. If False, raise an error.
        """
        super().__init__(
            root_dir=root_dir,
            paths_unw=paths_unw,
            paths_coh=paths_coh,
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands_unw=bands_unw,
            bands_coh=bands_coh,
            cache=cache,
            resampling=resampling,
            masked=masked,
            fill_nodata=fill_nodata,
            verbose=verbose,
            keep_common=keep_common,
        )

    @classmethod
    def parse_pairs(cls, paths: list[Path]) -> Pairs:
        """Parse the primary and secondary date/acquisition of the interferogram
        to generate Pairs object.
        """
        names = [Path(f).name for f in paths]
        pair_names = ["_".join(i.split("_")[1:3]) for i in names]
        pairs = Pairs.from_names(pair_names)
        return pairs

    @classmethod
    def parse_datetime(cls, paths: list[Path]) -> pd.DatetimeIndex:
        """Parse the datetime of the interferogram to generate DatetimeIndex object."""
        names = [Path(f).name for f in paths]
        pair_names = ["_".join(i.split("_")[1:3]) for i in names]
        date_names = np.unique([i.split("_") for i in pair_names])
        return pd.DatetimeIndex(date_names)


class HyP3S1Burst(InterferogramDataset):
    """A dataset manages the data of HyP3 Sentinel-1 Burst product.

    `Hyp3 <https://hyp3-docs.asf.alaska.edu/>`_ is a service for processing
    Synthetic Aperture Radar (SAR) imagery. This class is used to manage the
    data of Hyp3 product.
    """

    pattern_unw = "*unw_phase.tif"
    pattern_coh = "*corr.tif"

    def __init__(
        self,
        root_dir: str = "data",
        paths_unw: Optional[Sequence[Union[str, Path]]] = None,
        paths_coh: Optional[Sequence[Union[str, Path]]] = None,
        crs: Optional[CRS] = None,
        res: Optional[Union[float, tuple[float, float]]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[Union[float, int, Any]] = None,
        roi: Optional[BoundingBox] = None,
        bands_unw: Optional[Sequence[str]] = None,
        bands_coh: Optional[Sequence[str]] = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        masked: bool = True,
        fill_nodata: bool = False,
        verbose=True,
        keep_common: bool = True,
    ) -> None:
        """Initialize a new HyP3 instance.

        Parameters
        ----------
        root_dir: str
            Root directory where dataset can be found.
        paths_unw: list of str, optional
            list of unwrapped interferogram file paths to use instead of searching
            for files in ``root_dir``. If None, files will be searched for in ``root_dir``.
        paths_coh: list of str, optional
            list of coherence file paths to use instead of searching for files in
            ``root_dir``. If None, files will be searched for in ``root_dir``.
        crs: CRS, optional
            the output term:`coordinate reference system (CRS)` of the dataset.
            If None, the CRS of the first file found will be used.
        res: float, optional
            resolution of the output dataset in units of CRS. If None, the resolution
            of the first file found will be used.
        dtype: numpy.dtype, optional
            data type of the output dataset. If None, the data type of the first
            file found will be used.
        nodata: float or int, optional
            no data value of the output dataset. If None, the no data value of the
            first file found will be used.
        roi: BoundingBox, optional
            region of interest to load from the dataset. If None, the union of all
            files bounds in the dataset will be used.
        bands_unw: list of str, optional
            names of bands to return (defaults to all bands) for unwrapped interferograms.
        bands_coh: list of str, optional
            names of bands to return (defaults to all bands) for coherence.
        cache: bool, optional
            if True, cache file handle to speed up repeated sampling
        resampling: Resampling, optional
            Resampling algorithm used when reading input files.
            Default: `Resampling.nearest`.
        masked : bool, optional
            if True, the returned will be a masked array with a mask
            for no data values. Default: True.

            .. note::
                If parameter ``fill_nodata`` is True, the array will be interpolated and the returned array will always be a normal numpy array.
        fill_nodata : bool, optional
            Whether to fill holes in raster data by interpolation using the
            ``rasterio.fill.fillnodata`` function. Default: False.
        verbose: bool, optional
            if True, print verbose output.
        keep_common: bool, optional, default: True
            Only used when the number of interferograms and coherence files are
            not equal. If True, keep the common pairs of interferograms and
            coherence files and raise a warning. If False, raise an error.
        """
        super().__init__(
            root_dir=root_dir,
            paths_unw=paths_unw,
            paths_coh=paths_coh,
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands_unw=bands_unw,
            bands_coh=bands_coh,
            cache=cache,
            resampling=resampling,
            masked=masked,
            fill_nodata=fill_nodata,
            verbose=verbose,
            keep_common=keep_common,
        )

    @classmethod
    def parse_pairs(cls, paths: list[Path]) -> Pairs:
        """Parse the primary and secondary date/acquisition of the interferogram
        to generate Pairs object.
        """
        names = [Path(f).name for f in paths]
        pair_names = ["_".join(i.split("_")[3:5]) for i in names]
        pairs = Pairs.from_names(pair_names)
        return pairs

    @classmethod
    def parse_datetime(self, paths: list[Path]) -> pd.DatetimeIndex:
        """Parse the datetime of the interferogram to generate DatetimeIndex object."""
        names = [Path(f).name for f in paths]
        pair_names = ["_".join(i.split("_")[3:5]) for i in names]
        date_names = np.unique([i.split("_") for i in pair_names])
        return pd.DatetimeIndex(date_names)
