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
locations. Construction, ``list``, ``get``, catalog inspection, and source
``plan`` perform no network I/O. Planetary Computer ``glo30:pc`` and
``glo90:pc`` plans are immutable STAC descriptors; search and asset signing
start only at ``to_raster``. Unsupported product or provider selections fail
before discovery.

For the provider-neutral catalog API and general CMR/STAC discovery, see
:doc:`remote`. DEM selection remains on the same explicit product/provider
registry; a provider outage raises an error and never silently changes the
requested product or provider.

The default target datum is WGS84 ellipsoidal height. The source-to-target
graph fetches only the geoid models it needs: same-datum conversion is a
no-op, EGM96 and EGM2008 conversions use their respective model, and a
conversion between EGM96 and EGM2008 uses both through the ellipsoid. The
fixed EGM2008 resource is ``egm2008-2_5``; it is not bundled in the wheel.
``Fetch`` checks ``FANINSAR_GEOID_CACHE`` (or
``~/.cache/faninsar/geoid``), downloads a missing model, validates its size
and digest, and then loads it. With ``PROJ_NETWORK=OFF`` a missing model is a
typed offline error, never a silent fallback.

``RasterDEM.save(path)`` writes a new ``.tif``/``.tiff`` only and refuses to
overwrite an existing path. The write is ordinary and non-atomic. Provider
outages, ambiguous local datum metadata, invalid grids, cache configuration,
resource budgets, and overwrite attempts raise explicit errors; credentials
and signed URLs are not included in diagnostics.

For ``Stack(grid="auto")``, an explicit ``GridSpec`` wins. Otherwise the
configured ``roi`` wins, followed by the deterministic union of selected
acquisition/swath/burst footprints. The centre chooses UTM in ordinary
latitudes and UPS in polar latitudes. Cross-zone, UTM/UPS-boundary,
antimeridian, and projected extents above 1,000 km warn and continue. An
explicit seam-crossing ROI fails before provider I/O; provide an explicit
``GridSpec`` for such a run.
