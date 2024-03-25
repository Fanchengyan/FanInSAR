from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.enums import Resampling

from faninsar._core.pair_tools import Pairs
from faninsar.datasets.base import ApsDataset, ApsPairs
from faninsar.query.query import BoundingBox, Points


class GACOS(ApsDataset):
    """A dataset manages the data of GACOS product.

    `GACOS <http://www.gacos.net/>`_ (Generic Atmospheric Correction Online
    Service for InSAR) is a online service for processing zenith total delay
    maps to correct Atmospheric delays. This class is used to manage the data
    of GACOS product.

    Examples
    --------

    >>> from faninsar.datasets import GACOS
    >>> from faninsar.datasets import HyP3
    >>> from faninsar.query import BoundingBox, Points

    >>> hyp3_dir = Path("/Volumes/Data/Hyp3/descending_roi")
    >>> home_dir = Path("/Volumes/Data/Hyp3/descending_gacos")
    >>> out_dir = Path("/Volumes/Data/Hyp3/descending_gacos_pairs")

    prepare reference points and roi (region of interest)

    >>> ref_points_file = Path("/Volumes/Data/ARPs.geojson")
    >>> ref_points = Points.from_shapefile(ref_points_file)
    >>> roi = BoundingBox(98.57726618, 38.52546262, 99.41100273, 39.13802703, crs=4326)

    initialize HyP3

    >>> ds_hyp3 = HyP3(hyp3_dir)

    using HyP3 crs and res as the output crs and res of GACOS dataset

    >>> gacos = GACOS(home_dir, crs=ds_hyp3.crs, res=ds_hyp3.res, nodata=np.nan)

    using reference points, roi and HyP3 pairs to generate gacos pair files

    >>> gacos.to_pair_files(out_dir, ds_hyp3.pairs, ref_points, roi)
    """

    #: This expression is used to find the GACOS files.
    pattern = "*.ztd.tif"

    def __init__(
        self,
        root_dir: str = "data",
        paths: Optional[Sequence[str]] = None,
        crs: Optional[CRS] = None,
        res: Optional[Union[float, tuple[float, float]]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[Union[float, int, Any]] = None,
        roi: Optional[BoundingBox] = None,
        bands: Optional[Sequence[str]] = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        masked: bool = True,
        fill_nodata: bool = False,
        verbose: bool = True,
        ds_name: str = "GACOS",
    ) -> None:
        """Initialize a new GACOS instance.

        Parameters
        ----------
        root_dir : str or Path
            root_dir directory where dataset can be found.
        paths : list of str, optional
            list of file paths to use instead of searching for files in ``root_dir``.
            If None, files will be searched for in ``root_dir``.
        crs : CRS, optional
            the output term:`coordinate reference system (CRS)` of the dataset.
            If None, the CRS of the first file found will be used.
        res : float, optional
            resolution of the output dataset in units of CRS. If None, the
            resolution of the first file found will be used.
        dtype : numpy.dtype, optional
            data type of the output dataset. If None, the data type of the first
            file found will be used.
        nodata : float or int, optional
            no data value of the output dataset. If None, the no data value of
            the first file found will be used.
        roi : BoundingBox, optional
            region of interest to load from the dataset. If None, the union of
            all files bounds in the dataset will be used.
        bands : list of str, optional
            names of bands to return (defaults to all bands)
        cache : bool, optional
            if True, cache file handle to speed up repeated sampling
        resampling : Resampling, optional
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
        verbose : bool, optional
            if True, print verbose output, default: True
        ds_name : str, optional
            name of the dataset. used for printing verbose output, default: "GACOS"
        """
        super().__init__(
            root_dir=root_dir,
            paths=paths,
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands=bands,
            cache=cache,
            resampling=resampling,
            masked=masked,
            fill_nodata=fill_nodata,
            verbose=verbose,
            ds_name=ds_name,
        )

    def parse_dates(self):
        dates_str = [i.stem.split(".")[0] for i in self.files.paths]
        dates = pd.to_datetime(dates_str, format="%Y%m%d")
        return dates

    def to_pair_files(
        self,
        out_dir: Union[str, Path],
        pairs: Pairs,
        ref_points: Points,
        roi: Union[BoundingBox, None] = None,
        overwrite: bool = False,
        prefix: str = "GACOS",
    ):
        """Generate aps-pair files for given pairs and reference points.

        Parameters
        ----------
        out_dir : str or Path
            path to the directory to save the aps-pair files
        pairs : Pairs
            pairs to generate aps pair files
        ref_points : Points
            reference points which values are subtracted for all aps pair files
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be used.
        overwrite : bool, optional
            if True, overwrite existing files, default: False
        prefix : str, optional
            prefix of the aps-pair files, default: "GACOS"
        """
        return super().to_pair_files(out_dir, pairs, ref_points, roi, overwrite, prefix)


class GACOSPairs(ApsPairs):
    """
    A dataset manages the data of GACOS pairs.
    """

    #: This expression is used to find the GACOSPairs files.
    pattern = "*.tif"

    def __init__(
        self,
        root_dir: str = "data",
        paths: Optional[Sequence[str]] = None,
        crs: Optional[CRS] = None,
        res: Optional[Union[float, tuple[float, float]]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[Union[float, int, Any]] = None,
        roi: Optional[BoundingBox] = None,
        bands: Optional[Sequence[str]] = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        masked: bool = True,
        fill_nodata: bool = False,
        verbose: bool = True,
        ds_name: str = "",
    ) -> None:
        """Initialize a new GACOSPairs instance.

        Parameters
        ----------
        root_dir : str or Path
            root_dir directory where dataset can be found.
        paths : list of str, optional
            list of file paths to use instead of searching for files in ``root_dir``.
            If None, files will be searched for in ``root_dir``.
        crs : CRS, optional
            the output term:`coordinate reference system (CRS)` of the dataset.
            If None, the CRS of the first file found will be used.
        res : float, optional
            resolution of the output dataset in units of CRS. If None, the resolution
            of the first file found will be used.
        dtype : numpy.dtype, optional
            data type of the output dataset. If None, the data type of the first file
            found will be used.
        nodata : float or int, optional
            no data value of the output dataset. If None, the no data value of the first
            file found will be used.
        roi : BoundingBox, optional
            region of interest to load from the dataset. If None, the union of all files
            bounds in the dataset will be used.
        bands : list of str, optional
            names of bands to return (defaults to all bands)
        cache : bool, optional
            if True, cache file handle to speed up repeated sampling
        resampling : Resampling, optional
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
        verbose : bool, optional
            if True, print verbose output, default: True
        ds_name : str, optional
            name of the dataset. used for printing verbose output, default: ""

        Raises
        ------
            FileNotFoundError: if no files are found in ``root_dir``
        """
        super().__init__(
            root_dir=root_dir,
            paths=paths,
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands=bands,
            cache=cache,
            resampling=resampling,
            masked=masked,
            fill_nodata=fill_nodata,
            verbose=verbose,
            ds_name=ds_name,
        )
        self._pairs = self.parse_pairs(self.files.paths[self.valid])
        self._datetime = self.parse_datetime(self.files.paths[self.valid])

    def parse_pairs(self, paths: list[Path]) -> Pairs:
        """Parse pairs from a list of GACOS-pair file paths."""
        return super().parse_pairs(paths)

    def parse_datetime(self, paths: list[Path]) -> pd.DatetimeIndex:
        """Parse datetime from a list of GACOS-pair file paths."""
        return super().parse_datetime(paths)
