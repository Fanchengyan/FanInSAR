from __future__ import annotations

import abc
import math
from collections.abc import Iterator
from typing import Any, Optional, Sequence

import numpy as np

from faninsar._core.logger import setup_logger
from faninsar.datasets import GeoDataset
from faninsar.query import BoundingBox

logger = setup_logger(
    log_name="FanInSAR.samplers.batch", log_format="%(levelname)s - %(message)s"
)


class PatchSampler(abc.ABC):
    """Abstract base class for patch samplers."""

    _boxes: np.ndarray
    _length: int
    _shape: tuple[int]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: Any) -> BoundingBox | np.ndarray:
        """Get the bounding boxes/patches of at the given index.

        Parameters
        ----------
        index : Any
            The index of the patch. It can be any form of index that can be used
            to index a numpy array. See numpy indexing for more information:
            https://numpy.org/doc/stable/user/basics.indexing.html

        Returns
        -------
        BoundingBox | np.ndarray
            The bounding box of the patch at the given index.
        """
        return self._boxes[index]

    @property
    def boxes(self) -> np.ndarray:
        """The bounding boxes/patches of the sampler used to sample dataset."""
        return self._boxes

    @property
    def shape(self) -> tuple[int]:
        """The shape of the patch sampler."""
        return self._shape


class RowSampler(PatchSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        roi: BoundingBox | None = None,
        row_num: int | None = None,
        height: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize a RowSampler.

        Parameters
        ----------
        dataset : GeoDataset
            The dataset needs to be sampled.
        roi : BoundingBox or Sequence, optional
            The the region of interest bounding box. If not provided, the
            bounding box of the dataset will be used.
        row_num : int, optional
            The number of patches to be sampled for row-wise sampling. If height
            is provided, this parameter will be ignored.
        height : int, optional
            The height (in pixels) of the patch to be sampled for row-wise sampling .
            if not provided, the row_num will be used.
        verbose : bool, optional
            Whether to print verbose information. The verbose of the dataset will
            be set to this value. Default is True.
        """
        self.dataset = dataset
        self.res = dataset.res

        self.dataset.verbose = verbose
        if roi is not None:
            self.dataset.roi = roi

        profile = dataset.get_profile("roi")
        ds_height = profile["height"]

        if height is not None:
            height = int(height)
            row_num = math.ceil(ds_height / height)
        else:
            if row_num is None:
                raise ValueError("Either height or row_num must be provided.")
            if row_num > height:
                logger.warning(
                    f"row_num ({row_num}) is larger than the height ({ds_height})\n"
                    "of the dataset. The row_num will be set to the height of the dataset.\n"
                    "If this cannot meet your requirement, please try to choose other Sampler."
                )
                row_num = ds_height
            row_num = int(row_num)
            height = math.floor(ds_height / row_num)

        self.row_num = row_num
        self.height = height

        self._length = row_num
        self._shape = (row_num,)
        self._boxes = self._gen_patch_boxes()

    def _gen_patch_boxes(self) -> None:
        roi = self.dataset.roi
        patch_boxes = []
        for i in range(self.row_num):
            bottom = (i * self.height) * self.res[1] + roi.bottom
            top = self.height * self.res[1] + bottom
            patch_boxes.append([roi.left, bottom, roi.right, top])
        # make last patch top equal to roi top
        patch_boxes[-1][3] = roi.top
        patch_boxes = np.asarray(patch_boxes, dtype=np.object_)

    def __iter__(self) -> Iterator:
        roi = self.dataset.roi
        height = self.height
        row_num = self.row_num

        patch_boxes = []
        for i in range(row_num):
            bottom = (i * height) * self.res + roi.bottom
            top = height * self.res + bottom
            patch_boxes.append([roi.left, bottom, roi.right, top])
        # make last patch top equal to roi top
        patch_boxes[-1][3] = roi.top

        for patch_bbox in patch_boxes:
            yield BoundingBox(*patch_bbox, crs=self.dataset.crs)


class ColSampler(PatchSampler):
    """A sampler samples data from a dataset in a col-wise manner.

    This class is used to sample data from a dataset. The dataset is
    represented by a bounding box, and the sampler is used to sample
    data in the bounding box. The result of sampling is an iterator
    that yields data from the dataset.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        roi: BoundingBox | None = None,
        col_num: int | None = None,
        width: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize a ColSampler.

        Parameters
        ----------
        dataset : GeoDataset
            The dataset needs to be sampled.
        roi : BoundingBox or Sequence, optional
            The the region of interest bounding box. If not provided, the
            bounding box of the dataset will be used.
        col_num : int, optional
            The number of patches to be sampled for row-wise sampling. If width
            is provided, this parameter will be ignored.
        width : int, optional
            The width (in pixel) of the patch to be sampled for col-wise sampling.
            if not provided, the col_num will be used.
        verbose : bool, optional
            Whether to print verbose information. The verbose of the dataset will
            be set to this value. Default is True.
        """
        self.dataset = dataset
        self.res = dataset.res[1]

        self.dataset.verbose = verbose
        if roi is not None:
            self.dataset.roi = roi

        profile = dataset.get_profile("roi")
        width = profile["width"]

        if width is not None:
            width = int(width)
            col_num = math.ceil(width / width)
        else:
            if col_num is None:
                raise ValueError("Either width or col_num must be provided.")
            if col_num > width:
                logger.warning(
                    f"col_num ({col_num}) is larger than the width ({width })\n"
                    "of the dataset. The col_num will be set to the width of the dataset.\n"
                    "If this cannot meet your requirement, please try to choose other Sampler."
                )
                col_num = width
            col_num = int(col_num)
            width = math.floor(width / col_num)

        self.col_num = col_num
        self.width = width

    def __iter__(self) -> Iterator:
        roi = self.dataset.roi
        width = self.width
        col_num = self.col_num

        patch_boxes = []
        for i in range(col_num):
            bottom = (i * width) * self.res + roi.bottom
            top = width * self.res + bottom
            patch_boxes.append([roi.left, bottom, roi.right, top])
        # make last patch top equal to roi top
        patch_boxes[-1][3] = roi.top

        for patch_bbox in patch_boxes:
            yield BoundingBox(*patch_bbox, crs=self.dataset.crs)

    def __len__(self) -> int:
        return self.col_num


class RowColSampler(PatchSampler):
    """A sampler samples data from a dataset in a row-col-wise manner.

    This class is used to sample data from a dataset. The dataset is
    represented by a bounding box, and the sampler is used to sample
    data in the bounding box. The result of sampling is an iterator
    that yields data from the dataset.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        roi: BoundingBox | None = None,
        height: int | None = None,
        width: int | None = None,
        row_num: int | None = None,
        col_num: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize a RowColSampler.

        Parameters
        ----------
        dataset : GeoDataset
            The dataset to be sampled.
        roi : BoundingBox or Sequence, optional
            The region of interest bounding box. If not provided, the
            bounding box of the dataset will be used.
        height : int, optional
            The height (in pixels) for each patch. If row_num is provided, this
            parameter will be ignored.
        width : int, optional
            The width (in pixels) for each patch. If col_num is provided, this
            parameter will be ignored.
        row_num : int, optional
            The number of rows to be sampled for row-col-wise sampling. If height
            is provided, this parameter will be ignored.
        col_num : int, optional
            The number of columns to be sampled for row-col-wise sampling. If width
            is provided, this parameter will be ignored.
        verbose : bool, optional
            Whether to print verbose information. The verbose of the dataset will
            be set to this value. Default is True.
        """
        self.dataset = dataset
        self.res = dataset.res

        self.dataset.verbose = verbose
        if roi is not None:
            self.dataset.roi = roi

        profile = dataset.get_profile("roi")
        ds_height = profile["height"]
        ds_width = profile["width"]

        # row direction
        if height is not None:
            height = int(height)
            row_num = math.ceil(ds_height / height)
        else:
            if row_num is None:
                raise ValueError("Either height or row_num must be provided.")
            if row_num > ds_height:
                logger.warning(
                    f"row_num ({row_num}) is larger than the height ({ds_height})\n"
                    "of the dataset. The row_num will be set to the height of the dataset.\n"
                    "If this cannot meet your requirement, please try to choose other Sampler."
                )
                row_num = ds_height
            row_num = int(row_num)
            height = math.floor(ds_height / row_num)

        # col direction
        if width is not None:
            width = int(width)
            col_num = math.ceil(ds_width / width)
        else:
            if col_num is None:
                raise ValueError("Either width or col_num must be provided.")
            if col_num > ds_width:
                logger.warning(
                    f"col_num ({col_num}) is larger than the width ({ds_width})\n"
                    "of the dataset. The col_num will be set to the width of the dataset.\n"
                    "If this cannot meet your requirement, please try to choose other Sampler."
                )
                col_num = width
            col_num = int(col_num)
            width = math.floor(ds_width / col_num)

        self.row_num = row_num
        self.height = height
        self.col_num = col_num
        self.width = width

        self._shape = (row_num, col_num)
        self._length = row_num * col_num
        self._boxes = self._gen_patch_boxes()

    def _gen_patch_boxes(self) -> None:
        roi = self.dataset.roi
        patch_boxes = []
        for i in range(self.row_num):
            patch_boxes_row = []
            for j in range(self.col_num):
                left = (j * self.width) * self.res[0] + roi.left
                right = self.width * self.res[0] + left
                bottom = (i * self.height) * self.res[1] + roi.bottom
                top = self.height * self.res[1] + bottom

                if j == self.col_num - 1:
                    right = roi.right
                if i == self.row_num - 1:
                    top = roi.top
                bbox = BoundingBox(left, bottom, right, top, crs=self.dataset.crs)
                patch_boxes_row.append(bbox)
            patch_boxes.append(patch_boxes_row)
        patch_boxes = np.asarray(patch_boxes, dtype=np.object_)
        return patch_boxes

    def __iter__(self) -> Iterator:
        for i in range(self.row_num):
            for j in range(self.col_num):
                patch_bbox = self.boxes[i, j]
                yield patch_bbox
