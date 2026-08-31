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

EGM2008 uses the pinned ``egm2008-2_5`` resource.  ``Fetch`` checks the cache
at ``FANINSAR_GEOID_CACHE`` (or ``~/.cache/faninsar/geoid``), downloads only
on a miss, validates the artifact, and then loads it.  Ellipsoidal-only
processing never downloads a geoid model.  The 1-minute ``egm2008-1`` model
is not used.

Stack accepts ``grid=\"auto\"`` and chooses UTM or UPS from the ROI centre;
an explicit ``GridSpec`` always wins.  Cross-zone, polar-boundary,
antimeridian, and large-extent automatic ROIs emit warnings and continue.

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
