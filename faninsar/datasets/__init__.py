from typing import Any

import odc.geo.xr  # ensure odc is loaded when using datasets

from .aps import ApsDataset, ApsPairs
from .base._base_common import PairParser
from .base.geo import GeoDataset
from .base.hierarchical import HierarchicalMixin
from .base.pair import PairDataset
from .base.raster import RasterDataset
from .base.timeseries import TimeSeriesDataset
from .frame import FrameGeometry, FrameInterferogramCollection
from .gacos import GACOS, GACOSPairs
from .geogrid import GeoGrid

# Backwards-compat alias — deprecated, use GeoGrid.
from .geogrid import GeoGrid as GeoBox
from .hierarchical import HierarchicalDataset
from .ifg import CoherenceDataset, InterferogramDataset, StackInterferogramDataset
from .network import (
    ExternalNetworkLayoutError,
    GAMMANetwork,
    GMTSARNetwork,
    IncompleteNetworkError,
    IncompleteNetworkProductError,
    ISCE2Network,
    ISCE3Network,
    LegacyNetworkLayoutError,
    Network,
    NetworkAnalysisError,
    NetworkConstructionError,
    NetworkCurrentError,
    NetworkGenerationError,
    NetworkManifestError,
    NetworkPathError,
    SNAPNetwork,
    UnknownNetworkIndexTypeError,
)
from .xarray_dataset import XarrayDataset


def __getattr__(name: str) -> Any:
    """Lazy loaders rehomed under :mod:`faninsar.io.datasets`."""
    if name == "HyP3S1":
        from faninsar.io.datasets.hyp3 import HyP3S1

        return HyP3S1
    if name == "LiCSAR":
        from faninsar.io.datasets.licsar import LiCSAR

        return LiCSAR
    if name == "ARIA":
        from faninsar.io.datasets.aria import ARIA

        return ARIA
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
