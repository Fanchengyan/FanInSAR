"""Reusable physical dataset readers and dataset base classes."""

from __future__ import annotations

from typing import Any

import odc.geo.xr  # register odc's xarray accessors

from .aps import ApsDataset, ApsPairs
from .base._base_common import PairParser
from .base.geo import GeoDataset
from .base.hierarchical import HierarchicalMixin
from .base.pair import PairDataset
from .base.raster import RasterDataset
from .base.timeseries import TimeSeriesDataset
from .gacos import GACOS, GACOSPairs
from .geogrid import GeoGrid
from .geogrid import GeoGrid as GeoBox
from .hierarchical import HierarchicalDataset
from .ifg import CoherenceDataset, InterferogramDataset, StackInterferogramDataset
from .xarray_dataset import XarrayDataset, XarrayDataSpec


def __getattr__(name: str) -> Any:
    """Load optional provider-specific dataset adapters on demand."""
    if name == "HyP3S1":
        from faninsar.data._adapters.hyp3 import HyP3S1

        return HyP3S1
    if name == "LiCSAR":
        from faninsar.data._adapters.licsar import LiCSAR

        return LiCSAR
    if name == "ARIA":
        from faninsar.data._adapters.aria import ARIA

        return ARIA
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


__all__ = [
    "ARIA",
    "GACOS",
    "ApsDataset",
    "ApsPairs",
    "CoherenceDataset",
    "GACOSPairs",
    "GeoBox",
    "GeoDataset",
    "GeoGrid",
    "HierarchicalDataset",
    "HierarchicalMixin",
    "InterferogramDataset",
    "PairDataset",
    "PairParser",
    "RasterDataset",
    "StackInterferogramDataset",
    "TimeSeriesDataset",
    "XarrayDataSpec",
    "XarrayDataset",
]
