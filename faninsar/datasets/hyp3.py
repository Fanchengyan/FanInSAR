from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from faninsar._core.file_tools import retrieve_meta_value
from faninsar._core.pair_tools import Pairs
from faninsar._core.sar_tools import Baselines
from faninsar.datasets.ifg import InterferogramDataset


class HyP3S1(InterferogramDataset):
    """A dataset manages the data of HyP3 Sentinel-1 product.

    `Hyp3 <https://hyp3-docs.asf.alaska.edu/>`_ is a service for processing
    Synthetic Aperture Radar (SAR) imagery. This class is used to manage the
    data of Hyp3 product.
    """

    pattern_unw = "*unw_phase.tif"
    pattern_coh = "*corr.tif"

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

    def parse_baselines(self, pairs: Pairs | None = None) -> Baselines:
        """Parse the baseline of the interferogram for given pairs.

        Parameters
        ----------
        pairs : Pairs
            The pairs which the baseline will be parsed. Default is None, which
            means all pairs will be parsed.

        returns
        -------
        baselines : Baselines
            The baseline of the interferogram for given pairs.
        """
        if pairs is None:
            pairs = self.pairs

        mask = self.pairs.where(pairs, return_type="mask")

        files = self.files[self.valid][mask].paths
        baselines = []
        for f in files:
            try:
                meta_file = str(f).replace("_unw_phase.tif", ".txt")
                value = float(retrieve_meta_value(meta_file, "Baseline"))
                baselines.append(value)
            except:
                baselines.append(np.nan)
        bs = Baselines.from_pair_wise(pairs, np.array(baselines))
        return bs


class HyP3S1Burst(InterferogramDataset):
    """A dataset manages the data of HyP3 Sentinel-1 Burst product.

    `Hyp3 <https://hyp3-docs.asf.alaska.edu/>`_ is a service for processing
    Synthetic Aperture Radar (SAR) imagery. This class is used to manage the
    data of Hyp3 product.
    """

    pattern_unw = "*unw_phase.tif"
    pattern_coh = "*corr.tif"

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
