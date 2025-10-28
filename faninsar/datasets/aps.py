"""Abstract base classes for Atmospheric Phase Screen (APS) datasets."""

from __future__ import annotations

import re
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from faninsar._core.sar.pairs import Pairs
from faninsar.datasets.base import PairDataset, TimeSeriesDataset
from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from os import PathLike

    from pyproj.crs.crs import CRS

    from faninsar.query import BoundingBox, Points


class ApsDataset(TimeSeriesDataset, ABC):
    """A base class for aps (atmospheric phase screen) datasets."""

    #: This expression is used to find the APS files.
    pattern = "*"

    _pairs = None

    def to_pair_files(
        self,
        out_dir: PathLike,
        pairs: Pairs,
        ref_points: Points,
        roi: BoundingBox | None = None,
        overwrite: bool = False,
        prefix: str = "APS",
    ) -> None:
        """Generate aps-pair files for given pairs and reference points.

        Parameters
        ----------
        out_dir : str or Path
            path to the directory to save the aps-pair files
        pairs : Pairs
            pairs to generate aps-pair files
        ref_points : Points
            reference points which values are subtracted for all aps-pair files
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be used.
        overwrite : bool, optional
            if True, overwrite existing files, default: False
        prefix : str, optional
            prefix of the aps-pair files, default: "APS"

        """
        if roi is None:
            roi = self.roi

        profile = self.get_profile(roi)
        profile["count"] = 1
        dates = self.parse_dates(self.files.paths.tolist())

        dates_missing = np.setdiff1d(pairs.dates, dates)
        if len(dates_missing) > 0:
            msg = (
                f"Following dates are missing in the {self.ds_name} "
                f"dataset. \n{dates_missing}",
            )
            logger.warning(msg, stacklevel=2)

        df_paths = pd.Series(self.files.paths.values, index=dates)

        mask = ~np.any(np.isin(pairs.values, dates_missing), axis=1)
        pairs = pairs[mask]

        pairs_names = self._ensure_saving_verbose(
            pairs.to_names().tolist(),
            ds_name=f"{self.ds_name} Pair",
            unit=" pairs",
        )

        for pair_name in pairs_names:
            primary, secondary = pair_name.split("_")
            out_file = Path(out_dir) / f"{prefix}_{pair_name}.tif"
            if out_file.exists() and not overwrite:
                msg = f"{out_file.name} already exists, skipping"
                logger.info(msg)
                continue
            with rasterio.open(out_file, "w", **profile.profile) as dst:
                src_primary = self._load_warp_file(df_paths[primary])
                src_secondary = self._load_warp_file(df_paths[secondary])
                dest_arr = (
                    self._file_query_bbox(roi, src_primary).squeeze(0)
                    - self._file_query_bbox(roi, src_secondary).squeeze(0)
                    - (
                        self._file_query_points(ref_points, src_primary)
                        - self._file_query_points(ref_points, src_secondary)
                    ).mean()
                )

                dst.write(dest_arr, 1)


class ApsPairs(PairDataset, ABC):
    """A dataset manages the data of APS pairs."""

    #: This expression is used to find the GACOSPairs files.
    pattern = "*.tif"

    def __init__(
        self,
        root_dir: str = "data",
        paths: Sequence[str] | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        dtype: np.dtype | None = None,
        nodata: float | None = None,
        roi: BoundingBox | None = None,
        bands: Sequence[str] | None = None,
        cache: bool = True,
        resampling: Resampling = Resampling.nearest,
        fill_nodata: bool = False,
        verbose: bool = True,
        ds_name: str = "",
    ) -> None:
        """Initialize a new ApsPairs instance.

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
            no data value of the output dataset. If None, the no data value of
            the first file found will be used. This parameter is useful when the
            no data value is not stored in the file.
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
        fill_nodata : bool, optional
            Whether to fill holes in the queried data by interpolating them using
            inverse distance weighting method provided by the
            :func:`rasterio.fill.fillnodata`. Default: False.

            .. note::
                This parameter is only used when sampling data using bounding
                boxes or polygons queries, and will not work for points queries.

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
            fill_nodata=fill_nodata,
            verbose=verbose,
            ds_name=ds_name,
        )

    @classmethod
    def parse_pairs(cls, paths: Sequence[str] | Sequence[Path]) -> Pairs:
        """Parse APS pair names from file paths."""
        pair_names = []
        for path in paths:
            stem = Path(path).stem
            matches = re.findall(r"\d{8}", stem)
            if len(matches) >= 2:
                pair_names.append("_".join(matches[:2]))
            else:
                msg = f"Unable to infer APS pair dates from '{stem}'."
                raise ValueError(msg)
        return Pairs.from_names(pair_names)
