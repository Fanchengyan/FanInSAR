.. _query:

geospatial Queries
==================

.. currentmodule:: faninsar.query

Geospatial Query in FanInSAR is used to retrieve or sample values from GeoDataset. Query includes:

.. csv-table::
   :header: "Query Type", "Description"

   :class:`Points`, "A collection of points, can be used to sample multiple pixel values from GeoDataset."
   :class:`BoundingBox`, "A bounding box, can be used to sample rectangular region values from GeoDataset."
   :class:`Polygons`, "A collection of polygons, can be used to sample multiple pixel values from GeoDataset."
   :class:`GeoQuery`, "A combination of :class:`Points`, :class:`BoundingBox`, and :class:`Polygons`. It is highly recommended if you want to sample values using multiple query types simultaneously from a GeoDataset."




Points query
^^^^^^^^^^^^

.. autoclass:: Points
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

Bounding Box query
^^^^^^^^^^^^^^^^^^

.. autoclass:: BoundingBox
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

Polygons query
^^^^^^^^^^^^^^

.. autoclass:: Polygons
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

GeoQuery
^^^^^^^^

.. autoclass:: GeoQuery
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance: