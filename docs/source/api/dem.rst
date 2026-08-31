Unified DEM API
===============

The public DEM boundary has three categories: ``SourceDEM``, ``RasterDEM``
and ``ConstantDEM``.  Construct them through ``DEM.from_source``,
``DEM.from_raster`` or ``DEM.from_constant``.  Construction is offline;
provider and geoid I/O starts only when ``to_raster(grid=...)`` is called.

``to_raster`` evaluates terrain once on the target ``GridSpec`` using the
fixed ISCE/P0032 six-sample biquintic rule.  If a vertical datum conversion is
requested, it is applied pointwise at the same target centres.  Supported
datums are ``ellipsoidal``, ``egm96`` and ``egm2008``.

The public categories are deliberately small. ``SourceDEM`` is an offline
recipe and requires ``cache_dir`` at materialization; ``RasterDEM`` owns a
read-only NumPy array plus CRS, transform, bounds, shape, nodata, datum, and
provenance; ``ConstantDEM`` is an in-memory finite field. ``DEM.from_*``
factories validate selection and finite values without opening a socket.
``RasterDEM.to_raster`` performs the one source-to-target terrain warp. It
does not stage an EPSG:4326 raster, and datum conversion is pointwise after
sampling at target pixel centres. Source and target datum identities,
resampling rule, precision, boundary, longitude wrapping, nodata, and
projection runtime identity are included in the resulting identity.

EGM2008 uses the pinned ``egm2008-2_5`` resource.  ``Fetch`` checks the cache
at ``FANINSAR_GEOID_CACHE`` (or ``~/.cache/faninsar/geoid``), downloads only
on a miss, validates the artifact, and then loads it.  Ellipsoidal-only
processing never downloads a geoid model.  The 1-minute ``egm2008-1`` model
is not used.

Stack accepts ``grid=\"auto\"`` and chooses UTM or UPS from the ROI centre;
an explicit ``GridSpec`` always wins.  Cross-zone, polar-boundary,
antimeridian, and large-extent automatic ROIs emit warnings and continue.
ROI precedence is explicit ``StackConfig.roi`` (or a ``resolve_grid(roi=...)``
override), then the deterministic union of selected acquisition, swath, and
burst footprints. An explicit seam-crossing ROI fails before planning;
automatic seam cases warn and continue. Resource shape/overflow/byte guards
still fail before fetch or allocation.

``RasterDEM.save`` is the only public persistence operation. It requires a
GeoTIFF suffix and refuses overwrite; there is no atomic-write or locking
promise. Provider planning is immutable and zero-network, while STAC search,
asset signing, geoid fetching, and source bytes all begin at the
materialization I/O boundary. Errors are fail-closed and logged before they
are raised.

API
---

.. autosummary::
   :toctree: generated

   faninsar.processing.dem.DEM
   faninsar.processing.dem.SourceDEM
   faninsar.processing.dem.RasterDEM
   faninsar.processing.dem.ConstantDEM
   faninsar.processing.dem.GridSpec
   faninsar.processing.dem.Fetch
   faninsar.processing.dem.convert_heights
