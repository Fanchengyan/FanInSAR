.. _query_result:

Query Results
=============

.. currentmodule:: faninsar.query.result

This module defines classes that store the results of queries. It includes:


.. csv-table::
   :header: "Type of Query Result", "Description"

   :class:`QueryResult`, "A combined result of the :class:`PointsResult`, :class:`BBoxesResult`, and :class:`PolygonsResult` queries. This class is the default return type of the :ref:`query` results for the datasets."
   :class:`BaseResult`, "Base class for the result of the queries."
   :class:`PointsResult`, "A class to manage the result of :class:`~faninsar.query.Points` query."
   :class:`BBoxesResult`, "A class to manage the result of :class:`~faninsar.query.BoundingBox` query."
   :class:`PolygonsResult`, "A class to manage the result of :class:`~faninsar.query.Polygons` query."





Base Result
^^^^^^^^^^^

.. autoclass:: BaseResult
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

Points Result
^^^^^^^^^^^^^

.. autoclass:: PointsResult
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:
   :inherited-members:

BBoxes Result
^^^^^^^^^^^^^

.. autoclass:: BBoxesResult
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:
   :inherited-members:

Polygons Result
^^^^^^^^^^^^^^^

.. autoclass:: PolygonsResult
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:
   :inherited-members:

Query Result
^^^^^^^^^^^^

.. autoclass:: QueryResult
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:
   :inherited-members: