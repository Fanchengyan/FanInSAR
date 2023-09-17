import abc
from collections.abc import Iterator
from typing import Optional, Union

from rtree.index import Index, Property

from faninsar.datasets import BoundingBox, GeoDataset


class Sampler(abc.ABC):
    '''Abstract base class for samplers.

    This class is used to sample data from a dataset. The dataset is
    represented by a bounding box, and the sampler is used to sample
    data in the bounding box. The result of sampling is an iterator
    that yields data from the dataset.
    '''

    def __init__(
        self,
        dataset: GeoDataset,
        bbox: Optional[BoundingBox] = None,
    ) -> None:
        '''Initialize a sampler.
        
        Parameters
        ----------
        dataset : GeoDataset
            The dataset needs to be sampled.
        bbox : BoundingBox, optional
            The bounding box of the dataset. If not provided, the
            bounding box of the dataset will be used.
        '''
        if bbox is None:
            bbox = dataset.bbox
