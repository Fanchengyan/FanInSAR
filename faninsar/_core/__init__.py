from __future__ import annotations

from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    # Lazy imports to avoid requiring heavy dependencies when only using submodules
    def __getattr__(name: str) -> object:
        """Lazy import attributes to avoid importing torch and other heavy dependencies."""
        if name == "gradient_magnitude":
            from .alg import gradient_magnitude

            return gradient_magnitude
        if name in ("cuda_available", "gpu_available", "mps_available", "parse_device"):
            from .device import (
                cuda_available,
                gpu_available,
                mps_available,
                parse_device,
            )

            return locals()[name]
        if name in ("load_meta", "load_metas", "strip_str"):
            from .file_tools import load_meta, load_metas, strip_str

            return locals()[name]
        if name in (
            "GeoDataFormatConverter",
            "Profile",
            "array2kml",
            "array2kmz",
            "geoinfo_from_latlon",
            "latlon_from_profile",
            "match_to_raster",
            "save_colorbar",
            "transform_from_latlon",
            "write_geoinfo_into_ds",
            "write_geoinfo_into_nc",
        ):
            from .geo.geo_tools import (
                GeoDataFormatConverter,
                Profile,
                array2kml,
                array2kmz,
                geoinfo_from_latlon,
                latlon_from_profile,
                match_to_raster,
                save_colorbar,
                transform_from_latlon,
                write_geoinfo_into_ds,
                write_geoinfo_into_nc,
            )

            return locals()[name]
        if name in (
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
        ):
            from .sar import (
                SAR,
                Acquisition,
                Baselines,
                DateManager,
                DaySpan,
                Frequency,
                Loop,
                Loops,
                Pair,
                Pairs,
                PairsFactory,
                PhaseDeformationConverter,
                Sentinel1,
                TripletLoop,
                TripletLoops,
                Wavelength,
                multi_look,
            )

            return locals()[name]
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    def __dir__() -> list[str]:
        """Return list of available attributes."""
        return [
            "gradient_magnitude",
            "cuda_available",
            "gpu_available",
            "mps_available",
            "parse_device",
            "load_meta",
            "load_metas",
            "strip_str",
            "GeoDataFormatConverter",
            "Profile",
            "array2kml",
            "array2kmz",
            "geoinfo_from_latlon",
            "latlon_from_profile",
            "match_to_raster",
            "save_colorbar",
            "transform_from_latlon",
            "write_geoinfo_into_ds",
            "write_geoinfo_into_nc",
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
        ]

else:
    # Type checking imports
    from .alg import gradient_magnitude
    from .device import cuda_available, gpu_available, mps_available, parse_device
    from .file_tools import load_meta, load_metas, strip_str
    from .geo.geo_tools import (
        GeoDataFormatConverter,
        Profile,
        array2kml,
        array2kmz,
        geoinfo_from_latlon,
        latlon_from_profile,
        match_to_raster,
        save_colorbar,
        transform_from_latlon,
        write_geoinfo_into_ds,
        write_geoinfo_into_nc,
    )
    from .sar import (
        SAR,
        Acquisition,
        Baselines,
        DateManager,
        DaySpan,
        Frequency,
        Loop,
        Loops,
        Pair,
        Pairs,
        PairsFactory,
        PhaseDeformationConverter,
        Sentinel1,
        TripletLoop,
        TripletLoops,
        Wavelength,
        multi_look,
    )
