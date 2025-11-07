import odc.geo  # ensure odc is loaded when using datasets

from .aps import ApsDataset, ApsPairs
from .base import GeoDataset, PairDataset, RasterDataset
from .gacos import GACOS, GACOSPairs
from .geobox import GeoBox
from .hierarchical import HierarchicalDataset
from .hyp3 import HyP3S1, HyP3S1Burst
from .ifg import CoherenceDataset, InterferogramDataset
from .licsar import LiCSAR
from .xarray_dataset import XarrayDataset

available = [
    "ApsDataset",
    "ApsPairs",
    "GeoDataset",
    "PairDataset",
    "RasterDataset",
    "GACOS",
    "GACOSPairs",
    "HyP3",
    "InterferogramDataset",
    "LiCSAR",
    "HierarchicalDataset",
    "XarrayDataset",
    "GeoBox",
]

MAPPING_PAIRS = {
    "hyp3s1": HyP3S1,
    "hyp3s1burst": HyP3S1Burst,
    "licsar": LiCSAR,
}


def get_dataset(name: str) -> RasterDataset:
    """Get a dataset object from a string name.

    Parameters
    ----------
    name : str
        Name of the dataset to return.

    Returns
    -------
    dataset : Dataset
        Dataset class.

    """
    if name.lower() in MAPPING_PAIRS:
        return MAPPING_PAIRS[name.lower()]
    msg = f"Dataset {name} not found. Available datasets are: {available}"
    raise ValueError(msg)
