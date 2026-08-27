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
from faninsar._public import *  # noqa: F403
from faninsar._public import __all__ as _public_all

# Network is loaded lazily: importing its Dataset-backed implementation while
# this package is still initializing would make the existing Dataset imports
# observe a partially initialized ``faninsar`` module.
__all__ = [*_public_all, "Network"]  # noqa: F405, PLE0604


def __getattr__(name: str) -> Any:
    """Lazily expose the path-based Network public seam and its errors."""
    if name in {
        "LegacyLayoutError",
        "LegacyNetworkLayoutError",
        "Network",
        "NetworkConstructionError",
        "NetworkLayoutError",
        "NetworkPathError",
    }:
        from faninsar.datasets.network import (
            LegacyLayoutError,
            LegacyNetworkLayoutError,
            Network,
            NetworkConstructionError,
            NetworkLayoutError,
            NetworkPathError,
        )

        values = {
            "LegacyLayoutError": LegacyLayoutError,
            "LegacyNetworkLayoutError": LegacyNetworkLayoutError,
            "Network": Network,
            "NetworkConstructionError": NetworkConstructionError,
            "NetworkLayoutError": NetworkLayoutError,
            "NetworkPathError": NetworkPathError,
        }
        return values[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
