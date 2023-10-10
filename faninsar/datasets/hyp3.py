from collections.abc import Sequence
from typing import Any, Optional, Tuple, Union

import numpy as np
from faninsar.datasets.base import InterferogramDataset
from rasterio.crs import CRS
from rasterio.enums import Resampling

from .query import BoundingBox


class HyP3(InterferogramDataset):
    """A dataset manages the data of Hyp3 product.

    `Hyp3 <https://hyp3-docs.asf.alaska.edu/>`_ is a service for processing
    Synthetic Aperture Radar (SAR) imagery. This class is used to manage the
    data of Hyp3 product.
    """

    filename_glob_unw = "*unw_phase.tif"
    filename_glob_coh = "*corr.tif"

    def __init__(
        self,
        root: str = "data",
        file_paths_unw: Optional[Sequence[str]] = None,
        file_paths_coh: Optional[Sequence[str]] = None,
        dem: Optional[Any] = None,
        mask: Optional[Any] = None,
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
        """Initialize a new InterferogramDataset instance.
        # TODO: add dem and mask support

        Parameters
        ----------
        root: str
            Root directory where dataset can be found.
        file_paths_unw: list of str, optional
            list of unwrapped interferogram file paths to use instead of searching
            for files in ``root``. If None, files will be searched for in ``root``.
        file_paths_coh: list of str, optional
            list of coherence file paths to use instead of searching for files in
            ``root``. If None, files will be searched for in ``root``.
        dem: Any, optional
            DEM data. If None, no DEM data will be used.
        mask: Any, optional
            Mask data. If None, no mask data will be used.
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
            root=root,
            file_paths_unw=file_paths_unw,
            file_paths_coh=file_paths_coh,
            dem=dem,
            mask=mask,
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
