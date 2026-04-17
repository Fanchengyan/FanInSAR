"""FanInSAR - A fancy InSAR time series library.

This package provides tools for InSAR time series analysis in a Pythonic,
fast, and flexible way.
"""

from __future__ import annotations

# dev versions should have "dev" in them, stable should not.
# doc/conf.py makes use of this to set the version drop-down.
# eg: "0.1.dev0", "0.1"
__version__ = "0.1.dev0"

from ._core import (
    SAR,
    Acquisition,
    Baselines,
    DateManager,
    DaySpan,
    Frequency,
    GeoDataFormatConverter,
    Loop,
    Loops,
    Pair,
    Pairs,
    PairsFactory,
    PhaseDeformationConverter,
    Profile,
    Sentinel1,
    TripletLoop,
    TripletLoops,
    Wavelength,
    array2kml,
    array2kmz,
    bounds_from_xy,
    cuda_available,
    dataarray2kml,
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
    write_geoinfo_into_ds,
    write_geoinfo_into_nc,
    xy_from_profile,
)
