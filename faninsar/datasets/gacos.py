"""A module for managing the data of GACOS product."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from faninsar.datasets.aps import ApsDataset, ApsPairs

if TYPE_CHECKING:
    from os import PathLike

    from faninsar.core.pairs import Pairs
    from faninsar.query.query import BoundingBox, Points


class GACOS(ApsDataset):
    """A dataset manages the data of GACOS product.

    `GACOS <http://www.gacos.net/>`_ (Generic Atmospheric Correction Online
    Service for InSAR) is a online service for processing zenith total delay
    maps to correct Atmospheric delays. This class is used to manage the data
    of GACOS product.

    Examples
    --------
    >>> from faninsar.datasets import GACOS
    >>> from faninsar.datasets import HyP3
    >>> from faninsar.query import BoundingBox, Points

    >>> hyp3_dir = Path("data/hyp3/descending_roi")
    >>> home_dir = Path("data/hyp3/descending_gacos")
    >>> out_dir = Path("data/hyp3/descending_gacos_pairs")

    prepare reference points and roi (region of interest)

    >>> ref_points_file = Path("data/reference_points.geojson")
    >>> ref_points = Points.from_file(ref_points_file)
    >>> roi = BoundingBox(98.57726618, 38.52546262, 99.41100273, 39.13802703, crs=4326)

    initialize HyP3

    >>> ds_hyp3 = HyP3(hyp3_dir)

    using HyP3 crs and res as the output crs and res of GACOS dataset

    >>> gacos = GACOS(home_dir, crs=ds_hyp3.crs, res=ds_hyp3.res, nodata=np.nan)

    using reference points, roi and HyP3 pairs to generate gacos pair files

    >>> gacos.to_pair_files(out_dir, ds_hyp3.pairs, ref_points, roi)

    """

    #: This expression is used to find the GACOS files.
    pattern = "*.ztd.tif"

    @classmethod
    def parse_dates(cls, paths: list[Path]) -> pd.DatetimeIndex:
        """Parse dates from the paths of GACOS files."""
        dates_str = [Path(i).stem.split(".")[0] for i in paths]
        return pd.to_datetime(dates_str, format="%Y%m%d")

    def to_pair_files(
        self,
        out_dir: PathLike,
        pairs: Pairs,
        ref_points: Points,
        roi: BoundingBox | None = None,
        overwrite: bool = False,
        prefix: str = "GACOS",
    ) -> None:
        """Generate aps-pair files for given pairs and reference points.

        Parameters
        ----------
        out_dir : str or Path
            path to the directory to save the aps-pair files
        pairs : Pairs
            pairs to generate aps pair files
        ref_points : Points
            reference points which values are subtracted for all aps pair files
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be used.
        overwrite : bool, optional
            if True, overwrite existing files, default: False
        prefix : str, optional
            prefix of the aps-pair files, default: "GACOS"

        """
        super().to_pair_files(out_dir, pairs, ref_points, roi, overwrite, prefix)


class GACOSPairs(ApsPairs):
    """A dataset manages the data of GACOS pairs."""

    #: This expression is used to find the GACOSPairs files.
    pattern = "*.tif"
