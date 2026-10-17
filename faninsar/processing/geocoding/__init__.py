"""Geocoding and geographic-grid processing stages.

One engine, one entry point: :class:`Geocoder` composes the geographic
lookup table with :class:`RadarCoordinates` and the kind-driven kernels in
:mod:`.geo_resample`, and ``Stack.geocode`` is the user-facing entry point
on top of it (PROPOSAL-0033 keeps geocoding a processing-domain capability
that Stack composes). Retired duplicates — ``processing.mosaicking
.geocode_raster``, ``processing.geocoding.geocode``, and the S1
``stage_geocode`` stage — were deleted without shims: a repository-wide
search at HEAD showed no production callers, and their capability is
superseded by ``Stack.geocode`` for on-demand entry and by
``Geocoder``/``sample`` for in-engine multi-layer flows.
"""

from __future__ import annotations

from .coordinates import RadarCoordinates
from .geo_lut import Geo2RdrLUT, build_geo2rdr_lut
from .geo_modes import coregister_geocoded_slcs, coregister_geocoded_slcs_chunked
from .geo_resample import GeocodeKind, resample_complex_at_coordinates, sample
from .geocoder import GeocodedRaster, Geocoder
from .product import GeocodeableProduct, RadarDomainProduct

__all__ = [
    "Geo2RdrLUT",
    "GeocodeKind",
    "GeocodeableProduct",
    "GeocodedRaster",
    "Geocoder",
    "RadarCoordinates",
    "RadarDomainProduct",
    "build_geo2rdr_lut",
    "coregister_geocoded_slcs",
    "coregister_geocoded_slcs_chunked",
    "resample_complex_at_coordinates",
    "sample",
]
