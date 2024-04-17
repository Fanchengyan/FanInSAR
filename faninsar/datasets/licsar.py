from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from faninsar._core.pair_tools import Pairs
from faninsar.datasets.ifg import InterferogramDataset


class LiCSAR(InterferogramDataset):
    """A dataset manages the data of LiCSAR product.

    `LiCSAR <https://www.mdpi.com/2072-4292/12/15/2430>`_ is an open-source
    SAR interferometry (InSAR) time series analysis package that integrates with
    the automated Sentinel-1 InSAR processor, which products can be downloaded
    from `COMET-LiCS-portal <https://comet.nerc.ac.uk/COMET-LiCS-portal/>`_.
    """

    pattern_unw = "*geo.unw.tif"
    pattern_coh = "*geo.cc.tif"

    #: pattern used to find dem file
    pattern_dem = "*geo.hgt.tif"

    #: pattern used to find E files
    pattern_E = "*geo.E.tif"

    #: pattern used to find N files
    pattern_N = "*geo.N.tif"
    #: pattern used to find U files
    pattern_U = "*geo.U.tif"
    #: pattern used to find baselines file
    pattern_baselines = "baselines"
    #: pattern used to find polygon file
    pattern_polygon = "*-poly.txt"

    @property
    def meta_files(self) -> pd.Series:
        """return the paths of LiCSAR metadata files in a pandas Series.
        metadata files include: DEM, U, E, N, baselines, polygon.
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
            index=["DEM", "U", "E", "N", "baselines", "polygon"],
        )
        return df

    @classmethod
    def parse_pairs(cls, paths: list[Path]) -> Pairs:
        """Parse the primary and secondary date/acquisition of the interferogram
        to generate Pairs object.
        """
        pair_names = [Path(f).name.split(".")[0] for f in paths]
        pairs = Pairs.from_names(pair_names)
        return pairs

    @classmethod
    def parse_datetime(cls, paths: list[Path]) -> pd.DatetimeIndex:
        """Parse the datetime of the interferogram to generate DatetimeIndex object."""
        pair_names = [Path(f).name.split(".")[0] for f in paths]
        date_names = np.unique([i.split("_") for i in pair_names])
        return pd.DatetimeIndex(date_names)
