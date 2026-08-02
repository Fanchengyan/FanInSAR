"""Internal utilities (device, geo helpers, render). Domain nouns live in ``faninsar.core``."""

from __future__ import annotations

from . import accessors as accessors
from .alg import gradient_magnitude
from .device import cuda_available, gpu_available, mps_available, parse_device
from .file_tools import load_meta, load_metas, strip_str
from .geo import (
    GeoDataFormatConverter,
    GeoGrid,
    GeoGridMixin,
    Profile,
    array2kmz,
    bounds_from_xy,
    dataarray2kmz,
    format_bounds_and_crs,
    geoinfo_from_xy,
    match_to_raster,
    save_colorbar,
    transform_from_xy,
    xy_from_profile,
)

# Domain nouns re-exported lazily to avoid circular imports with faninsar.core
# (core.pairs imports faninsar._core.render during core package init).


def __getattr__(name: str):
    """Lazy re-export of domain nouns from :mod:`faninsar.core`."""
    nouns = {
        "SAR",
        "Acquisition",
        "Baselines",
        "DateManager",
        "DaySpan",
        "Frequency",
        "Loop",
        "Loops",
        "Pair",
        "Pairs",
        "PairsFactory",
        "PhaseDeformationConverter",
        "Sentinel1",
        "TripletLoop",
        "TripletLoops",
        "Wavelength",
        "multi_look",
    }
    if name in nouns:
        import faninsar.core as _core_domain

        return getattr(_core_domain, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GeoDataFormatConverter",
    "GeoGrid",
    "GeoGridMixin",
    "Profile",
    "accessors",
    "array2kmz",
    "bounds_from_xy",
    "cuda_available",
    "dataarray2kmz",
    "format_bounds_and_crs",
    "geoinfo_from_xy",
    "gpu_available",
    "gradient_magnitude",
    "load_meta",
    "load_metas",
    "match_to_raster",
    "mps_available",
    "parse_device",
    "save_colorbar",
    "strip_str",
    "transform_from_xy",
    "xy_from_profile",
]
