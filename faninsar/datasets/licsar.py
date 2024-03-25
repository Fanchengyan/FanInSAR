import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from faninsar._core.pair_tools import Pairs
from faninsar.datasets.ifg import InterferogramDataset
from faninsar.query.query import BoundingBox
from rasterio.crs import CRS
from rasterio.enums import Resampling


class LiCSAR(InterferogramDataset):
    """A dataset manages the data of LiCSAR product.

    `LiCSAR <https://www.mdpi.com/2072-4292/12/15/2430>`_ is an open-source
    SAR interferometry (InSAR) time series analysis package that integrates with
    the automated Sentinel-1 InSAR processor, which products can be downloaded
    from `COMET-LiCS-portal <https://comet.nerc.ac.uk/COMET-LiCS-portal/>`_.
    """

    pattern_unw = "*geo.unw.tif"
    pattern_coh = "*geo.cc.tif"

    pattern_dem = "*geo.hgt.tif"
    pattern_E = "*geo.E.tif"
    pattern_N = "*geo.N.tif"
    pattern_U = "*geo.U.tif"
    pattern_baselines = "baselines"
    pattern_polygon = "*-poly.txt"

    def __init__(
        self,
        root_dir: str = "data",
        paths_unw: Optional[Sequence[str]] = None,
        paths_coh: Optional[Sequence[str]] = None,
        crs: Optional[CRS] = None,
        res: Optional[Union[float, Tuple[float, float]]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[Union[float, int, Any]] = None,
        roi: Optional[BoundingBox] = None,
        bands_unw: Optional[Sequence[str]] = None,
        bands_coh: Optional[Sequence[str]] = None,
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
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands_unw=bands_unw,
            bands_coh=bands_coh,
            cache=cache,
            resampling=resampling,
            verbose=verbose,
        )

    @property
    def meta_files(self) -> pd.DataFrame:
        """return the metadata files of the dataset in a DataFrame.
        metadata files include: DEM, U, E, N, baseline, polygon.
        """
        def parse_file(pattern: str) -> Path:
            result = list(self.root_dir.rglob(pattern))
            if len(result) == 0:
                warnings.warn(f"File not found: {pattern}")
                return None
            return result[0]
        
        dem_file = parse_file(self.pattern_dem)
        U_file = parse_file(self.pattern_U)
        E_file = parse_file(self.pattern_E)
        N_file = parse_file(self.pattern_N)
        baseline_file = parse_file(self.pattern_baselines)
        polygon_file = parse_file(self.pattern_polygon)

        df = pd.Series(
            [dem_file, U_file, E_file, N_file, baseline_file, polygon_file],
            index=['dem','U','E','N','baseline','polygon']
        )

        return df

    def parse_pairs(self, paths: list[Path]) -> Pairs:
        """Parse the primary and secondary date/acquisition of the interferogram
        to generate Pairs object.
        """
        pair_names = [f.name.split(".")[0] for f in paths]
        pairs = Pairs.from_names(pair_names)
        return pairs

    def parse_datetime(self, paths: list[Path]) -> pd.DatetimeIndex:
        """Parse the datetime of the interferogram to generate DatetimeIndex object."""
        pair_names = [f.name.split(".")[0] for f in paths]
        date_names = np.unique([i.split("_") for i in pair_names])
        return pd.DatetimeIndex(date_names)

home_dir = "/Volumes/Data/GeoData/YNG/Sentinel1/LiCSAR/106D_05248_131313"
