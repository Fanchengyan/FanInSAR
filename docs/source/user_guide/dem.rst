DEM materialization
===================

FanInSAR uses one public DEM facade. Construct a source recipe or a local
raster. ``DEM.to_raster`` materializes a derived raster on an explicit
``GridSpec``; that export is optional and is not the default Stack working
DEM::

   from faninsar.processing.geometry import DEM

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

Default Stack DEM
-----------------

Acquisition planning, Stack preparation, registration, and geometry share one
managed raster, ``work_dir/dem.tif``. It is EPSG:4326, north-up, with WGS84
ellipsoidal heights in metres. Spacing follows the DEM source and is
independent of ``Stack.grid``. Projected output grids transform query centres
to lon/lat and sample this raster with the established interpolation kernel;
``stack.dem`` therefore need not match the output-grid shape or CRS. New
workspaces do not create ``dem_wgs84.tif`` or ``dem/rgv_stack.tif``. Incoming
source or raster data is normalized to ellipsoidal heights when the managed
DEM is created; a projected or orthometric file is not merely relabelled.

Reuse and coverage
------------------

An existing ``dem.tif`` with no explicit ROI is reused as-is. FanInSAR does
not inspect coverage, hash pixels, track size or mtime, or detect an external
same-range replacement. Reopening or moving the workspace still reuses that
file.

An explicit ROI is compared with the raster bounds in the same CRS. If the
ROI is covered, the file is left untouched. If it is not, or if a later
processing step actually needs coordinates outside the usable DEM, the same
``dem.tif`` is rewritten to the previous extent union the required extent,
including interpolation support, and the current step is retried. Prior
coverage is kept. The file is created when explicit preparation or the first
real DEM request supplies an extent; no ROI does not trigger a global
download.

A same-range replacement of the DEM file is not detected; affected results
and views must be rebuilt explicitly. A failed expansion of the managed DEM
keeps the previous ``dem.tif``. An external raster that cannot be extended is
left unchanged and does not silently change source.

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
outages, missing land coverage, ambiguous local datum metadata, invalid
grids, cache configuration, resource budgets, and overwrite attempts raise
explicit errors; credentials and signed URLs are not included in
diagnostics. Unpublished Copernicus all-water cells may be filled with 0 m
under the existing automatic source rule; uncovered land is not filled with
0 m.

Stack output grid
-----------------

For ``Stack(grid="utm_ups")``, an explicit ``GridSpec`` wins. Otherwise the
configured ``roi`` wins, followed by the deterministic union of selected
acquisition/swath/burst footprints. The centre chooses UTM in ordinary
latitudes and UPS in polar latitudes. Cross-zone, UTM/UPS-boundary,
antimeridian, and projected extents above 1,000 km warn and continue. An
explicit seam-crossing ROI fails before provider I/O; provide an explicit
``GridSpec`` for such a run. This output grid does not force a second
persisted DEM.
