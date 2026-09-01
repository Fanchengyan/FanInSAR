"""FanInSAR - A fancy InSAR time series library.

This package provides tools for InSAR time series analysis in a Pythonic,
fast, and flexible way.
"""

from __future__ import annotations

from typing import Any

# dev versions should have "dev" in them, stable should not.
# doc/conf.py makes use of this to set the version drop-down.
# eg: "0.1.dev0", "0.1"
__version__ = "0.1.dev0"

# Compatibility exports still used by tests and internal modules that have not
# yet migrated to the curated surface. These are **not** in ``__all__``.
from faninsar._core import (
    SAR,
    DateManager,
    DaySpan,
    Frequency,
    GeoDataFormatConverter,
    PairsFactory,
    PhaseDeformationConverter,
    Profile,
    Sentinel1,
    TripletLoop,
    TripletLoops,
    Wavelength,
    array2kmz,
    bounds_from_xy,
    cuda_available,
    dataarray2kmz,
    geoinfo_from_xy,
    gpu_available,
    gradient_magnitude,
    load_meta,
    load_metas,
    match_to_raster,
    mps_available,
    multi_look,
    parse_device,
    save_colorbar,
    transform_from_xy,
    xy_from_profile,
)
from faninsar._public import Acquisition, Acquisitions, Interferogram, Pair, Pairs
from faninsar._public import __all__ as _public_all
from faninsar.missions.base import list_missions, register
from faninsar.timeseries.invert import NSBAS, SBAS, invert

# Network is loaded lazily: importing its Dataset-backed implementation while
# this package is still initializing would make the existing Dataset imports
# observe a partially initialized ``faninsar`` module.
_NETWORK_EXPORTS = (
    "Network",
    "NISARStack",
    "S1Stack",
    "Stack",
)
__all__ = [  # noqa: PLE0604
    "Acquisition", "Acquisitions", "Baselines", "Interferogram", "Pair",
    "Pairs", "Loops", "NSBAS", "SBAS", "Stage", "invert", "list_missions",
    "open_stac", "open_zarr", "register", *_NETWORK_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """Lazily expose the path-based Network public seam and its errors."""
    if name in {
        "GAMMANetwork",
        "GMTSARNetwork",
        "ISCE2Network",
        "ISCE3Network",
        "IncompleteNetworkError",
        "IncompleteNetworkProductError",
        "LegacyLayoutError",
        "LegacyNetworkLayoutError",
        "Network",
        "NetworkAnalysisError",
        "NetworkConstructionError",
        "NetworkLayoutError",
        "NetworkPathError",
        "NISARStack",
        "SNAPNetwork",
        "S1Stack",
        "Stack",
    }:
        from faninsar.datasets.network import (
            GAMMANetwork,
            GMTSARNetwork,
            IncompleteNetworkError,
            IncompleteNetworkProductError,
            ISCE2Network,
            ISCE3Network,
            LegacyLayoutError,
            LegacyNetworkLayoutError,
            Network,
            NetworkAnalysisError,
            NetworkConstructionError,
            NetworkLayoutError,
            NetworkPathError,
            SNAPNetwork,
        )
        from faninsar.processing.stack import NISARStack, S1Stack, Stack

        values = {
            "GAMMANetwork": GAMMANetwork,
            "GMTSARNetwork": GMTSARNetwork,
            "ISCE2Network": ISCE2Network,
            "ISCE3Network": ISCE3Network,
            "IncompleteNetworkError": IncompleteNetworkError,
            "IncompleteNetworkProductError": IncompleteNetworkProductError,
            "LegacyLayoutError": LegacyLayoutError,
            "LegacyNetworkLayoutError": LegacyNetworkLayoutError,
            "Network": Network,
            "NetworkAnalysisError": NetworkAnalysisError,
            "NetworkConstructionError": NetworkConstructionError,
            "NetworkLayoutError": NetworkLayoutError,
            "NetworkPathError": NetworkPathError,
            "NISARStack": NISARStack,
            "SNAPNetwork": SNAPNetwork,
            "S1Stack": S1Stack,
            "Stack": Stack,
        }
        return values[name]
    if name in {
        "TimeSeries",
        "Orbit",
        "StackConfig",
        "Points",
        "BoundingBox",
        "Polygons",
        "DEM",
    }:
        from faninsar.processing.contracts.products import OrbitMetadata
        from faninsar.processing.dem.api import DEM
        from faninsar.processing.stack.config import StackConfig
        from faninsar.processing.timeseries.inversion import TimeSeriesResult
        from faninsar.query import BoundingBox, Points, Polygons

        values = {
            "TimeSeries": TimeSeriesResult,
            "Orbit": OrbitMetadata,
            "StackConfig": StackConfig,
            "Points": Points,
            "BoundingBox": BoundingBox,
            "Polygons": Polygons,
            "DEM": DEM,
        }
        return values[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
