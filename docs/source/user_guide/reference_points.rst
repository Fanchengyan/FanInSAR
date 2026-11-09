Reference points
================

One ``Stack.set_ref_points`` call configures the geographic points used by SBAS
inversion and by layer plots. The original unwrap rasters are never overwritten.
The subtracted value is the arithmetic mean of valid samples; that mean does
**not** independently align disconnected unwrap components.

Set, inspect, reset, and clear::

   from faninsar.data.query import Points

   stack.set_ref_points(Points([[100.12, 31.45], [100.18, 31.50]]))
   stack.ref_points
   stack.ref_policy
   stack.set_ref_points(stack.ref_points, policy="per_image")
   stack.set_ref_points(None)

CRS-less ``Points`` are stored as WGS84 longitude, latitude. Projected CRS is
kept as given and converted to WGS84 only when sampling radar products. Heights
come from the Stack DEM; a missing DEM is an error, not height 0.

Policies
--------

``common`` (the default, including when ``policy`` is omitted) uses the nonempty
intersection of valid points across every image and warns once when requested
points were excluded. ``per_image`` uses each image's own nonempty subset and
warns once if those subsets differ. All-invalid samples raise.

Inversion and plotting
----------------------

``stack.invert_timeseries()`` inherits the saved points. Pass ``ref_points=None``
to disable for one call, or pass explicit ``Points`` to override locations while
keeping the Stack policy. Applied offsets are stored in
``result.metadata["reference"]``. Changing or clearing points invalidates
time-series reuse; coregistration and IFG caches stay valid.

``layer.plot()`` inherits the same defaults when the layer is Stack-bound.
Inherit on a standalone layer is a no-op. Explicit geographic points work on a
standalone geographic layer. Radar layers need geometry context.

See :meth:`faninsar.stack.session.Stack.set_ref_points` and
:meth:`faninsar.network.layers.ProductLayer.plot`.
