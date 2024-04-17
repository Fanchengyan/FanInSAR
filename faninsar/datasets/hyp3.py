from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from faninsar._core.pair_tools import Pairs
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
