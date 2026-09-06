"""HyP3 Sentinel-1 interferogram dataset."""

from __future__ import annotations

import numpy as np

from faninsar.io.file_tools import load_meta
from faninsar.core import Baselines, Pairs
from faninsar.core.sar_missions import Sentinel1
from faninsar.data.datasets.ifg import InterferogramDataset


class HyP3S1(InterferogramDataset, Sentinel1):
    """HyP3 Sentinel-1 InSAR product dataset.

    Supports both full-frame and burst products from the
    `HyP3 <https://hyp3-docs.asf.alaska.edu/>`_ processing service.
    """

    pattern_unw = "*unw_phase.tif"
    pattern_coh = "*corr.tif"

    def parse_baselines(self, pairs: Pairs | None = None) -> Baselines:
        """Parse baselines from HyP3 metadata files.

        Parameters
        ----------
        pairs : Pairs
            Pairs to parse baselines for. Default is None (all pairs).

        Returns
        -------
        baselines : Baselines

        """
        if pairs is None:
            pairs = self.pairs

        mask = self.pairs.where(pairs, return_type="mask")

        files = self.files[self.valid][mask].paths
        baselines = []
        for f in files:
            meta_file = str(f).replace("_unw_phase.tif", ".txt")
            value = load_meta(meta_file, "Baseline")
            if value is None:
                value = np.nan
            baselines.append(value)
        return Baselines.from_pair_wise(pairs, np.array(baselines))
