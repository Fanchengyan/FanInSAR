"""FanInSAR - A fancy InSAR time series library.

This package provides tools for InSAR time series analysis in a Pythonic,
fast, and flexible way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# dev versions should have "dev" in them, stable should not.
# doc/conf.py makes use of this to set the version drop-down.
# eg: "0.1.dev0", "0.1"
__version__ = "0.1.dev0"

if not TYPE_CHECKING:
    # Lazy imports to avoid requiring heavy dependencies when only using submodules
    def __getattr__(name: str) -> object:
        """Lazy import attributes to avoid importing torch and other heavy dependencies."""
        # First check if it's from _core
        from . import _core

        if hasattr(_core, name):
            return getattr(_core, name)

        # Check cmaps
        if name == "cmaps":
            from .cmaps import cmaps

            return cmaps

        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    def __dir__() -> list[str]:
        """Return list of available attributes."""
        return [
            "__version__",
            "SAR",
            "Acquisition",
            "Baselines",
            "DateManager",
            "DaySpan",
            "Frequency",
            "GeoDataFormatConverter",
            "Loop",
            "Loops",
            "Pair",
            "Pairs",
            "PairsFactory",
            "PhaseDeformationConverter",
            "Profile",
            "Sentinel1",
            "TripletLoop",
            "TripletLoops",
            "Wavelength",
            "array2kml",
            "array2kmz",
            "cuda_available",
            "geoinfo_from_latlon",
            "gpu_available",
            "gradient_magnitude",
            "latlon_from_profile",
            "load_meta",
            "load_metas",
            "match_to_raster",
            "mps_available",
            "multi_look",
            "parse_device",
            "save_colorbar",
            "transform_from_latlon",
            "write_geoinfo_into_ds",
            "write_geoinfo_into_nc",
            "cmaps",
        ]

else:
    # Type checking imports
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
        cuda_available,
        geoinfo_from_latlon,
        gpu_available,
        gradient_magnitude,
        latlon_from_profile,
        load_meta,
        load_metas,
        match_to_raster,
        mps_available,
        multi_look,
        parse_device,
        save_colorbar,
        transform_from_latlon,
        write_geoinfo_into_ds,
        write_geoinfo_into_nc,
    )
    from .cmaps import cmaps
