from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.enums import Resampling

from faninsar._core.pair_tools import Pairs
from faninsar.datasets.ifg import InterferogramDataset
from faninsar.query.query import BoundingBox


class LiCSAR(InterferogramDataset):
    """A dataset manages the data of LiCSAR product.

    `LiCSAR <https://www.mdpi.com/2072-4292/12/15/2430>`_ is an open-source
    SAR interferometry (InSAR) time series analysis package that integrates with
    the automated Sentinel-1 InSAR processor, which products can be downloaded
    from `COMET-LiCS-portal <https://comet.nerc.ac.uk/COMET-LiCS-portal/>`_.
    """

    filename_glob_unw = "*geo.unw.tif"
    filename_glob_coh = "*geo.cc.tif"

    def __init__(
        self,
        root_dir: str = "data",
        paths_unw: Optional[Sequence[str]] = None,
        paths_coh: Optional[Sequence[str]] = None,
        dem_file: Optional[Any] = None,
        mask_file: Optional[Any] = None,
        crs: Optional[CRS] = None,
        res: Optional[Union[float, Tuple[float, float]]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[Union[float, int, Any]] = None,
        roi: Optional[BoundingBox] = None,
        bands: Optional[Sequence[str]] = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        verbose=False,
    ) -> None:
        """Initialize a new LiCSAR instance.

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
        dem_file: Any, optional
            DEM data. If None, no DEM data will be used.
        mask_file: Any, optional
            Mask data. If None, no Mask data will be used.
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
        bands: list of str, optional
            names of bands to return (defaults to all bands)
        cache: bool, optional
            if True, cache file handle to speed up repeated sampling
        resampling: Resampling, optional
            Resampling algorithm used when reading input files.
            Default: `Resampling.nearest`.
        verbose: bool, optional
            if True, print verbose output.
        """
        super().__init__(
            root_dir=root_dir,
            paths_unw=paths_unw,
            paths_coh=paths_coh,
            dem_file=dem_file,
            mask_file=mask_file,
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands=bands,
            cache=cache,
            resampling=resampling,
            verbose=verbose,
        )

    def parse_pairs(self, paths: list[Path]) -> Pairs:
        """Parse the primary and secondary date/acquisition of the interferogram
        to generate Pairs object.
        """
        pair_names = [f.parent.stem for f in paths]
        pairs = Pairs.from_names(pair_names)
        return pairs

    def parse_datetime(self, paths: list[Path]) -> pd.DatetimeIndex:
        """Parse the datetime of the interferogram to generate DatetimeIndex object."""
        pass
        # pair_names = self.parse_pairs(paths)
        # date_names = np.unique([i.split("_") for i in pair_names])
        # return pd.DatetimeIndex(date_names)
