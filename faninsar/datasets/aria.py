"""ARIA dataset class for Sentinel-1 interferograms."""

from __future__ import annotations

from pathlib import Path

from faninsar._core.sar.pairs import Pairs
from faninsar._core.sar.sar_missions import Sentinel1

from .ifg import HierarchicalInterferogramDataset


class ARIA(HierarchicalInterferogramDataset, Sentinel1):
    """ARIA dataset class for Sentinel-1 interferograms."""

    pattern_files = "*.nc"

    @classmethod
    def parse_pairs(cls, paths: list[Path]) -> Pairs:
        """Parse the Pairs from the paths of the interferogram."""
        names = [Path(str(f).split(":")[1]).name for f in paths]
        pair_names = [i.split("-")[6] for i in names]
        return Pairs.from_names(pair_names)
