"""A dataset manages the data of LiCSAR product."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from faninsar._core.sar.pairs import Pairs
from faninsar._core.sar.sar_missions import Sentinel1
from faninsar.datasets.ifg import InterferogramDataset


class LiCSAR(InterferogramDataset, Sentinel1):
    """A dataset manages the data of LiCSAR product.

    `LiCSAR <https://www.mdpi.com/2072-4292/12/15/2430>`_ is an open-source
    SAR interferometry (InSAR) time series analysis package that integrates with
    the automated Sentinel-1 InSAR processor, which products can be downloaded
    from `COMET-LiCS-portal <https://comet.nerc.ac.uk/COMET-LiCS-portal/>`_.
    """

    pattern_unw = "*geo.unw.tif"
    pattern_coh = "*geo.cc.tif"
    coh_range: tuple[float, float] = (0, 255)

    #: pattern used to find dem file
    pattern_dem = "*geo.hgt.tif"

    #: pattern used to find E files
    pattern_e = "*geo.E.tif"

    #: pattern used to find N files
    pattern_n = "*geo.N.tif"
    #: pattern used to find U files
    pattern_u = "*geo.U.tif"
    #: pattern used to find baselines file
    pattern_baselines = "baselines"
    #: pattern used to find polygon file
    pattern_polygon = "*-poly.txt"

    @property
    def meta_files(self) -> pd.Series:
        """Return the paths of LiCSAR metadata files in a pandas Series.

        metadata files include: DEM, U, E, N, baselines, polygon.
        """

        def parse_file(pattern: str) -> Path:
            result = list(self.root_dir.rglob(pattern))
            if len(result) == 0:
                warnings.warn(f"File not found: {pattern}", stacklevel=2)
                return None
            return result[0]

        dem_file = parse_file(self.pattern_dem)
        u_file = parse_file(self.pattern_u)
        e_file = parse_file(self.pattern_e)
        n_file = parse_file(self.pattern_n)
        baseline_file = parse_file(self.pattern_baselines)
        polygon_file = parse_file(self.pattern_polygon)

        return pd.Series(
            [dem_file, u_file, e_file, n_file, baseline_file, polygon_file],
            index=["DEM", "U", "E", "N", "baselines", "polygon"],
        )

    @classmethod
    def parse_pairs(cls, paths: list[Path]) -> Pairs:
        """Parse the Pairs from the paths of the interferogram."""
        pair_names = [Path(f).name.split(".")[0] for f in paths]
        return Pairs.from_names(pair_names)
