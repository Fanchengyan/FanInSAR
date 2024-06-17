from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Optional, Sequence

from faninsar._core.logger import setup_logger
from faninsar.datasets import GeoDataset
from faninsar.query import BoundingBox

logger = setup_logger(
    log_name="FanInSAR.samplers.batch", log_format="%(levelname)s - %(message)s"
)


class RowSampler:
    """A sampler samples data from a dataset in a row-wise manner.

    This class is used to sample data from a dataset. The dataset is
    represented by a bounding box, and the sampler is used to sample
    data in the bounding box. The result of sampling is an iterator
    that yields data from the dataset.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        roi: Optional[BoundingBox | Sequence[float]] = None,
        patch_size: Optional[int] = None,
        patch_num: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize a sampler.

        Parameters
        ----------
        dataset : GeoDataset
            The dataset needs to be sampled.
        roi : BoundingBox or Sequence, optional
            The the region of interest bounding box. If not provided, the
            bounding box of the dataset will be used.
        patch_size : int, optional
            The size of the patch to be sampled for row-wise sampling in
            pixels. if not provided, the patch_num will be used.

            .. note::

                patch_size is a tuple of (height, width) in pixels. But in this
                class, only the height is used. The width is set to the width of
                the roi.
        patch_num : int, optional
            The number of patches to be sampled for row-wise sampling. If patch_size
            is provided, this parameter will be ignored.
        verbose : bool, optional
            Whether to print verbose information. Default is False.
        """
        self.dataset = dataset
        self.verbose = verbose

        if roi is not None:
            self.dataset.roi = roi

        profile = dataset.get_profile("roi")
        height = profile["height"]

        if patch_size is not None:
            patch_size = int(patch_size)
            patch_num = math.ceil(height / patch_size)
        else:
            if patch_num is None:
                raise ValueError("Either patch_size or patch_num must be provided.")
            if patch_num > height:
                logger.warning(
                    f"patch_num ({patch_num}) is larger than the height ({height })\n"
                    "of the dataset. The patch_num will be set to the height of the dataset.\n"
                    "If this cannot meet your requirement, please try to choose other Sampler."
                )
                patch_num = height
            patch_num = int(patch_num)
            patch_size = math.floor(height / patch_num)

        self.patch_num = patch_num
        self.patch_size = patch_size

    def __iter__(self) -> Iterator:
        roi = self.dataset.roi
        patch_size = self.patch_size
        patch_num = self.patch_num

        patch_bboxs = []
        for i in range(patch_num):
            bottom = i * patch_size + roi.bottom
            top = bottom + patch_size
            patch_bboxs.append([roi.left, bottom, roi.right, top])
        # make last patch top equal to roi top
        patch_bboxs[-1][3] = roi.top

        for patch_bbox in patch_bboxs:
            yield BoundingBox(*patch_bbox)

    def __len__(self) -> int:
        return self.patch_num
