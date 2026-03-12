"""A module defines the grid samplers for sampling data from a dataset."""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import torch
from torch.utils.data import Sampler

from faninsar._core.device import parse_device
from faninsar.logging import setup_logger
from faninsar.query import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from torch.utils.data import DataLoader

    from faninsar.datasets import GeoDataset
    from faninsar.typing import DeviceLike

logger = setup_logger(__name__)


def _normalize_device(device: DeviceLike | None) -> torch.device | None:
    """Normalize a device specifier into a torch.device instance.

    Parameters
    ----------
    device : DeviceLike or None
        Device specifier accepted by torch. If None, no device normalization
        is performed and None is returned.

    Returns
    -------
    torch.device or None
        Normalized device when provided, otherwise None.

    """
    if device is None:
        return None
    return parse_device(device)


@overload
def _to_tensor(value: Any, device: None) -> Any: ...
@overload
def _to_tensor(
    value: np.ndarray, device: torch.device
) -> np.ndarray | torch.Tensor: ...
@overload
def _to_tensor(value: np.generic, device: torch.device) -> torch.Tensor: ...
@overload
def _to_tensor(value: dict[Any, Any], device: torch.device) -> dict[Any, Any]: ...
@overload
def _to_tensor(value: list[Any], device: torch.device) -> list[Any]: ...
@overload
def _to_tensor(value: tuple[Any, ...], device: torch.device) -> tuple[Any, ...]: ...


def _to_tensor(  # noqa: PLR0911
    value: np.ndarray | np.generic | dict[Any, Any] | list[Any] | tuple[Any, ...],
    device: torch.device | None,
) -> Any | np.ndarray | torch.Tensor | dict[Any, Any] | list[Any] | tuple[Any, ...]:
    """Recursively convert numpy arrays to torch tensors.

    Parameters
    ----------
    value : Any
        Input value to be converted when it contains numpy arrays.
    device : torch.device or None
        Target torch device.

    Returns
    -------
    Any
        Converted structure with numpy arrays turned into torch tensors.

    """
    if isinstance(value, np.ndarray):
        if value.dtype == np.object_:
            return value
        return torch.as_tensor(value, device=device)
    if isinstance(value, np.generic):
        return torch.as_tensor(value, device=device)
    if isinstance(value, dict):
        return {key: _to_tensor(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_tensor(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_tensor(item, device) for item in value)

    return value


def identity_collate(batch: list[Any]) -> Any | list[Any]:
    """Return batch as-is; unbox single-item batches for convenience.

    Parameters
    ----------
    batch : list[Any]
        Samples returned by the dataset.

    Returns
    -------
    Any
        A single sample when batch size is 1, otherwise the original list.

    """
    if len(batch) == 1:
        return batch[0]
    return batch


def tensor_collate(
    batch: list[Any], device: torch.device | None = None
) -> Any | list[Any]:
    """Convert numpy arrays to torch tensors and unbox single-item batches.

    Parameters
    ----------
    batch : list[Any]
        Samples returned by the dataset.
    device : torch.device or None, optional
        Target device for torch tensors. If None, returns the input unchanged.

    Returns
    -------
    Any | list[Any]
        A single converted sample when batch size is 1, otherwise a list of
        converted samples.

    """
    if len(batch) == 1:
        return _to_tensor(batch[0], device)
    return [_to_tensor(item, device) for item in batch]


class GridSampler(Sampler):
    """Abstract base class for grid samplers."""

    _boxes: np.ndarray
    _length: int
    _shape: tuple[int]
    _indexes: int | Iterable[int] | np.ndarray | None
    _dataset: GeoDataset
    _device: torch.device | None

    def __len__(self) -> int:
        """Return the length of the grid sampler."""
        return self._length

    def __str__(self) -> str:
        """Return the string representation of the grid sampler."""
        return f"{self.__class__.__name__}[boxes={self.shape}]"

    def __repr__(self) -> str:
        """Return the string representation of the grid sampler."""
        return self.__str__()

    def __getitem__(self, index: int | slice) -> BoundingBox | np.ndarray:
        """Get the bounding boxes/grids of at the given index.

        Parameters
        ----------
        index : Any
            The index of the grid. It can be any form of index that can be used
            to index a numpy array. See numpy indexing for more information:
            https://numpy.org/doc/stable/user/basics.indexing.html

        Returns
        -------
        BoundingBox | np.ndarray
            The bounding box of the grid at the given index.

        """
        return self._boxes[index]

    @property
    def boxes(self) -> np.ndarray:
        """The bounding boxes/grids of the sampler used to sample dataset."""
        return self._boxes

    @property
    def shape(self) -> tuple[int]:
        """The shape of the grid sampler."""
        return self._shape

    @property
    def indexes(self) -> int | Iterable[int] | np.ndarray | None:
        """File indexes to query. None means all valid files."""
        return self._indexes

    @property
    def dataset(self) -> GeoDataset:
        """The dataset to be sampled."""
        return self._dataset

    @property
    def device(self) -> torch.device | None:
        """Torch device for tensor conversion."""
        return self._device

    def _yield_box(
        self, bbox: BoundingBox
    ) -> BoundingBox | tuple[BoundingBox, int | Iterable[int] | np.ndarray]:
        """Yield a bbox, or (bbox, indexes) when file indexes are set."""
        if self._indexes is not None:
            return bbox, self._indexes
        return bbox

    def to_dataloader(
        self,
        *,
        num_workers: int = 1,
        prefetch_factor: int | None = None,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        **kwargs: Any,
    ) -> DataLoader:
        """Create a PyTorch DataLoader for this sampler.

        Parameters
        ----------
        num_workers : int, optional
            Number of worker processes. Default is 1. This overlaps I/O with
            compute for large InSAR stacks. If your disk is fast (SSD) and RAM is
            sufficient, you may try ``num_workers=2`` to further reduce waiting.
            Use 0 when you want deterministic single-process behavior or if
            multi-process overhead outweighs I/O latency.
        prefetch_factor : int or None, optional
            Number of samples to prefetch per worker. If None, uses PyTorch
            default behavior.
        collate_fn : Callable, optional
            Collate function used to merge samples. If None, uses
            :func:`tensor_collate` when ``device`` is set on the sampler,
            otherwise :func:`identity_collate` to avoid default tensor stacking.
        **kwargs : Any
            Additional keyword arguments forwarded to ``torch.utils.data.DataLoader``.
            The following keys are not allowed here because they are managed by
            the sampler: ``dataset``, ``batch_size``, ``sampler``, ``batch_sampler``,
            ``shuffle``.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader instance.

        """
        from torch.utils.data import DataLoader

        if collate_fn is None:
            if self.device is None:
                collate_fn = identity_collate
            else:
                collate_fn = partial(tensor_collate, device=self.device)

        kwargs["prefetch_factor"] = prefetch_factor

        forbidden_kwargs = {
            "dataset",
            "sampler",
            "batch_sampler",
            "shuffle",
            "batch_size",
        }
        invalid_kwargs = forbidden_kwargs.intersection(kwargs)
        if invalid_kwargs:
            msg = (
                "The following DataLoader arguments are managed by FanInSAR "
                f"and will be ignored: {sorted(invalid_kwargs)}. "
                "Try building a torch DataLoader manually if truly needed."
            )
            logger.warning(msg)
            kwargs = {k: v for k, v in kwargs.items() if k not in invalid_kwargs}

        return DataLoader(
            self.dataset,
            batch_size=1,
            sampler=self,
            num_workers=num_workers,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
            **kwargs,
        )


class RowSampler(GridSampler):
    """A sampler samples data from a dataset in a row-wise manner."""

    def __init__(
        self,
        dataset: GeoDataset,
        roi: BoundingBox | None = None,
        row_num: int | None = None,
        height: int | None = None,
        indexes: int | Iterable[int] | np.ndarray | None = None,
        device: DeviceLike | None = None,
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
            The number of grids to be sampled for row-wise sampling. If height
            is provided, this parameter will be ignored.
        height : int, optional
            The height (in pixels) of the grid to be sampled for row-wise sampling .
            if not provided, the row_num will be used.
        indexes : int | Iterable[int] | np.ndarray | None, optional
            Indexes of files to query. If None, all valid files are queried
            when using with ``dataset[bbox]``. Default is None.
        device : DeviceLike or None, optional
            Torch device for tensor conversion when using ``to_dataloader``.
        verbose : bool, optional
            Whether to print verbose information. The verbose of the dataset will
            be set to this value. Default is True.

        """
        self._dataset = dataset
        self.res = dataset.res
        self._indexes = indexes
        self._device = _normalize_device(device)

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
                msg = "Either height or row_num must be provided."
                logger.error(msg)
                raise ValueError(msg)
            if row_num > ds_height:
                msg = (
                    f"row_num ({row_num}) is larger than the height ({ds_height})\n"
                    "of the dataset. The row_num will be set to the height of the "
                    "dataset.\n If this cannot meet your requirement, please "
                    "try to choose other Sampler."
                )
                logger.warning(msg, stacklevel=2)
                row_num = ds_height
            row_num = int(row_num)
            height = math.floor(ds_height / row_num)

        self.row_num = row_num
        self.height = height

        self._length = row_num
        self._shape = (row_num,)
        self._boxes = self._gen_patch_boxes()

    def _gen_patch_boxes(self) -> np.ndarray:
        roi = self.dataset.roi
        patch_boxes = []
        for i in range(self.row_num):
            bottom = (i * self.height) * self.res[1] + roi.bottom
            top = self.height * self.res[1] + bottom
            if i == self.row_num - 1:
                top = roi.top
            patch_boxes.append(
                BoundingBox(roi.left, bottom, roi.right, top, crs=self.dataset.crs),
            )

        return np.asarray(patch_boxes, dtype=np.object_)

    def __iter__(self) -> Iterator:
        """Iterate over the bounding boxes of the grids."""
        for i in range(self.row_num):
            yield self._yield_box(self.boxes[i])


class ColSampler(GridSampler):
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
        indexes: int | Iterable[int] | np.ndarray | None = None,
        device: DeviceLike | None = None,
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
            The number of grids to be sampled for row-wise sampling. If width
            is provided, this parameter will be ignored.
        width : int, optional
            The width (in pixel) of the grid to be sampled for col-wise sampling.
            if not provided, the col_num will be used.
        indexes : int | Iterable[int] | np.ndarray | None, optional
            Indexes of files to query. If None, all valid files are queried
            when using with ``dataset[bbox]``. Default is None.
        device : DeviceLike or None, optional
            Torch device for tensor conversion when using ``to_dataloader``.
        verbose : bool, optional
            Whether to print verbose information. The verbose of the dataset will
            be set to this value. Default is True.

        """
        self._dataset = dataset
        self.res = dataset.res[1]
        self._indexes = indexes
        self._device = _normalize_device(device)

        self.dataset.verbose = verbose
        if roi is not None:
            self.dataset.roi = roi

        profile = dataset.get_profile("roi")
        ds_width = profile["width"]

        if width is not None:
            width = int(width)
            col_num = math.ceil(ds_width / width)
        else:
            if col_num is None:
                msg = "Either width or col_num must be provided."
                logger.error(msg)
                raise ValueError(msg)
            if col_num > ds_width:
                msg = (
                    f"col_num ({col_num}) is larger than the width ({ds_width})\n"
                    "of the dataset. The col_num will be set to the width of the "
                    "dataset.\n If this cannot meet your requirement, please "
                    "try to choose other Sampler."
                )
                logger.warning(msg)
                col_num = width
            col_num = int(col_num)
            width = math.floor(ds_width / col_num)

        self.col_num = col_num
        self.width = width

    def __iter__(self) -> Iterator:
        """Iterate over the bounding boxes of the grids."""
        roi = self.dataset.roi
        width = self.width
        col_num = self.col_num

        patch_boxes = []
        for i in range(col_num):
            bottom = (i * width) * self.res + roi.bottom
            top = width * self.res + bottom
            patch_boxes.append([roi.left, bottom, roi.right, top])
        # make last grid top equal to roi top
        patch_boxes[-1][3] = roi.top

        for patch_bbox in patch_boxes:
            yield self._yield_box(BoundingBox(*patch_bbox, crs=self.dataset.crs))

    def __len__(self) -> int:
        """Return the length of the grid sampler."""
        return self.col_num


class RowColSampler(GridSampler):
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
        indexes: int | Iterable[int] | np.ndarray | None = None,
        device: DeviceLike | None = None,
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
            The height (in pixels) for each grid. If row_num is provided, this
            parameter will be ignored.
        width : int, optional
            The width (in pixels) for each grid. If col_num is provided, this
            parameter will be ignored.
        row_num : int, optional
            The number of rows to be sampled for row-col-wise sampling. If height
            is provided, this parameter will be ignored.
        col_num : int, optional
            The number of columns to be sampled for row-col-wise sampling. If width
            is provided, this parameter will be ignored.
        indexes : int | Iterable[int] | np.ndarray | None, optional
            Indexes of files to query. If None, all valid files are queried
            when using with ``dataset[bbox]``. Default is None.
        device : DeviceLike or None, optional
            Torch device for tensor conversion when using ``to_dataloader``.
        verbose : bool, optional
            Whether to print verbose information. The verbose of the dataset will
            be set to this value. Default is True.

        """
        self._dataset = dataset
        self.res = dataset.res
        self._indexes = indexes
        self._device = _normalize_device(device)

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
                msg = "Either height or row_num must be provided."
                logger.error(msg)
                raise ValueError(msg)
            if row_num > ds_height:
                msg = (
                    f"row_num ({row_num}) is larger than the height ({ds_height})\n"
                    "of the dataset. The row_num will be set to the height of the"
                    " dataset.\n If this cannot meet your requirement, please try"
                    " to choose other Sampler.",
                )
                logger.warning(msg)
                row_num = ds_height
            row_num = int(row_num)
            height = math.floor(ds_height / row_num)

        # col direction
        if width is not None:
            width = int(width)
            col_num = math.ceil(ds_width / width)
        else:
            if col_num is None:
                msg = "Either width or col_num must be provided."
                logger.error(msg)
                raise ValueError(msg)
            if col_num > ds_width:
                msg = (
                    f"col_num ({col_num}) is larger than the width ({ds_width})\n"
                    "of the dataset. The col_num will be set to the width of the"
                    " dataset.\n If this cannot meet your requirement, please try"
                    " to choose other Sampler.",
                )
                logger.warning(msg)
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
        return np.asarray(patch_boxes, dtype=np.object_)

    def __iter__(self) -> Iterator:
        """Iterate over the bounding boxes of the grids."""
        for i in range(self.row_num):
            for j in range(self.col_num):
                yield self._yield_box(self.boxes[i, j])
