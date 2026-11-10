Scene burst inspection
======================

``stack.scenes.bursts`` is a read-only collection of per-acquisition Sentinel-1
scene units. Constructing it does not open annotation, read SLC samples, prepare
inputs, or make network requests.

Collection vs scalar demotion
-----------------------------

A date **list** keeps a scene collection, so ``.bursts`` stays available::

   stack.scenes.bursts
   stack.scenes.sel(dates=["20200110", "20200116"]).bursts
   stack.scenes.sel(track="128D").bursts

A **scalar** date demotes the view to that scene's layers. Local burst tags
remain the existing string tuple; this is not the observation collection::

   stack.scenes["20200110"].layers            # SceneLayers
   stack.scenes.sel(dates="20200110")         # SceneLayers
   stack.scenes["20200110"].layers.bursts     # tuple[str, ...], e.g. ("f0_IW1_b0", ...)
   stack.scenes["20200110"].layers.sel(bursts=["f0_IW1_b0"], swaths=["IW1"])

``scenes.sel(burst=integer)`` is still the optional absolute-ID **date filter**.
It does not select a local unit tag. Missing official IDs do not block
inspection.

Handles
-------

Each handle exposes ``date``, ``track``, local ``burst`` tag, and
``absolute_burst_id`` (``None`` when the catalog/annotation did not provide an
official ESA identifier). Internal ``faninsar:burst_id`` / ANX keys are never
substituted.

Geographic coverage is not specified on this page. Geometry resolution and
persistence are owned outside this inspection surface and are not yet a
stable public API. This page does not promise a local ``rdr2geo`` four-corner
solve, an annotation bbox, or any other fallback when geometry is missing.
Missing geometry is a null row or an omitted plot, not an implemented
substitute.

Plot labels
-----------

Collection and individual ``.plot()`` draw WGS84 longitude/latitude polygons
when plottable geometry is present. Each observation keeps these two lines::

   Burst: f0_IW1_b0
   Absolute burst ID: None

The official integer replaces ``None`` when known. The plot does not draw a
legend and does not color or style outlines by date. Every GeoDataFrame row is
retained. Drawing display-deduplicates identical geometry and identical
``Burst`` / ``Absolute burst ID`` text; remaining identities are a right-hand
callout column with leaders to a representative centroid.
