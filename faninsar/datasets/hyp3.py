"""A module for managing the data of HyP3 Sentinel-1 interferograms."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from faninsar._core.file_tools import load_meta
from faninsar._core.sar.pairs import Pairs
from faninsar._core.sar.sar_missions import Sentinel1
from faninsar._core.sar.sar_tools import Baselines
from faninsar.datasets.ifg import InterferogramDataset


class HyP3S1(InterferogramDataset, Sentinel1):
    """A dataset manages the data of HyP3 Sentinel-1 product.

    `Hyp3 <https://hyp3-docs.asf.alaska.edu/>`_ is a service for processing
    Synthetic Aperture Radar (SAR) imagery. This class is used to manage the
    data of Hyp3 product.
    """

    pattern_unw = "*unw_phase.tif"
    pattern_coh = "*corr.tif"

    @classmethod
    def parse_pairs(cls, paths: list[Path]) -> Pairs:
        """Parse the Pairs from the paths of the interferogram."""
        names = [Path(f).name for f in paths]
        pair_names = ["_".join(i.split("_")[1:3]) for i in names]
        return Pairs.from_names(pair_names)

    def parse_baselines(self, pairs: Pairs | None = None) -> Baselines:
        """Parse the baseline of the interferogram for given pairs.

        Parameters
        ----------
        pairs : Pairs
            The pairs which the baseline will be parsed. Default is None, which
            means all pairs will be parsed.

        Returns
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
                value = float(load_meta(meta_file, "Baseline"))
                baselines.append(value)
            except Exception:  # noqa: PERF203
                baselines.append(np.nan)
        return Baselines.from_pair_wise(pairs, np.array(baselines))


class HyP3S1Burst(InterferogramDataset, Sentinel1):
    """A dataset manages the data of HyP3 Sentinel-1 Burst product.

    `Hyp3 <https://hyp3-docs.asf.alaska.edu/>`_ is a service for processing
    Synthetic Aperture Radar (SAR) imagery. This class is used to manage the
    data of Hyp3 product.
    """

    pattern_unw = "*unw_phase.tif"
    pattern_coh = "*corr.tif"

    @classmethod
    def parse_pairs(cls, paths: list[Path]) -> Pairs:
        """Parse pairs from the paths of the interferogram."""
        names = [Path(f).name for f in paths]
        pair_names = ["_".join(i.split("_")[3:5]) for i in names]
        return Pairs.from_names(pair_names)
