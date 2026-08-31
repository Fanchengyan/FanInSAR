DEM materialization
===================

FanInSAR uses one public DEM facade. Construct a source recipe or a local
raster, then materialize it on the grid owned by the consuming stage::

   from faninsar.processing.dem import DEM

   source = DEM.from_source("glo30:pc", cache_dir=".cache/dem")
   dem = source.to_raster(stack.grid, vertical_datum="ellipsoidal")

``DEM.from_constant`` is useful for a zero-height ellipsoid in geometry
tests. ``RasterDEM`` exposes ``array``, ``crs``, ``transform``, ``bounds``,
``height`` and ``width`` and can be passed to NumPy directly. ``to_raster``
performs the single fixed P0032 6x6 terrain resampling step; coordinate datum
conversion uses WGS84 pixel centres and is not a second raster resample.

Source providers and EGM2008 are fetched lazily into explicit cache
locations. Construction performs no network I/O. Unsupported product or
provider selections fail before discovery.
