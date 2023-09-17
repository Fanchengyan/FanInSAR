import abc
from collections.abc import Sequence
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
from rasterio.crs import CRS

from faninsar.datasets.base import BoundingBox, RasterDataset


class HyP3(RasterDataset):
    '''A dataset manages the data of Hyp3 product.

    `Hyp3 <https://hyp3-docs.asf.alaska.edu/>`_ is a service for processing 
    Synthetic Aperture Radar (SAR) imagery. This class is used to manage the
    data of Hyp3 product.    
    '''

    def __init__(self, bbox: BoundingBox) -> None:
        '''Initialize a Hyp3 dataset.

        Parameters
        ----------
        bbox : BoundingBox
            The bounding box of the dataset.
        '''
        super().__init__(bbox)
