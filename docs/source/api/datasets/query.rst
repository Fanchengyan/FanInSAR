geospatial Queries
==================

Geospatial Query in FanInSAR is used to sample values from GeoDataset. Query includes:

* **Rectangular/BoundingBox query**: :class:`faninsar.datasets.query.BoundingBox`  
* **Point query**: :class:`faninsar.datasets.query.Point`, :class:`faninsar.datasets.query.Points` 
* **a combination of BoundingBox and Point**: :class:`faninsar.datasets.query.GeoQuery`.


Bounding Box query
^^^^^^^^^^^^^^^^^^

.. autoclass:: faninsar.datasets.query.BoundingBox
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

Point query
^^^^^^^^^^^

.. autoclass:: faninsar.datasets.query.Point
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

Points query
^^^^^^^^^^^^

.. autoclass:: faninsar.datasets.query.Points
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

GeoQuery
^^^^^^^^

.. autoclass:: faninsar.datasets.query.GeoQuery
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance: