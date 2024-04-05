geospatial Queries
==================

Geospatial Query in FanInSAR is used to sample values from GeoDataset. Query includes:

* :class:`.BoundingBox` : a bounding box, can be used to sample rectangular region values from GeoDataset.
* :class:`.Points` : a collection of points, can be used to sample multiple pixel values from GeoDataset.
* :class:`.GeoQuery` : **a combination** of :class:`.BoundingBox` and :class:`.Points`, highly recommended if you want to sample multiple rectangular region and pixel values from GeoDataset simultaneously.


Bounding Box query
^^^^^^^^^^^^^^^^^^

.. autoclass:: faninsar.query.BoundingBox
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

Points query
^^^^^^^^^^^^

.. autoclass:: faninsar.query.Points
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

Polygons query
^^^^^^^^^^^^^^

.. autoclass:: faninsar.query.Polygons
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

GeoQuery
^^^^^^^^

.. autoclass:: faninsar.query.GeoQuery
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance: