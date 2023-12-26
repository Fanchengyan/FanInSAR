from .base import ApsDataset, ApsPairs, GeoDataset, PairDataset, RasterDataset
from .gacos import GACOS, GACOSPairs
from .hyp3 import HyP3
from .ifg import InterferogramDataset
from .licsar import LiCSAR

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
]

MAPPING_PAIRS = {
    "hyp3": HyP3,
    "licsar": LiCSAR,
}


def get_dataset(name):
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
    if name.lower() in MAPPING_PAIRS.keys():
        return MAPPING_PAIRS[name.lower()]
    else:
        raise ValueError(f"Dataset {name} not found.")
