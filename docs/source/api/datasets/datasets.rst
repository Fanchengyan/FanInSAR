geospatial Datasets
===================

Geospatial Datasets are used to process large amounts of geospatial data files that have common information. These files are combined to form a complete dataset.

The following datasets are available:

* basic geospatial Datasets:
   * :class:`.GeoDataset`
   * :class:`.RasterDataset` : base Datasets for all Raster-like productions.
* InSAR related Datasets:
   * :class:`.ApsDataset` : base Datasets for all APS (Atmospheric Phase Screen) productions. Child classes:
      * :class:`.GACOS`
   * :class:`.PairDataset` : base Datasets for all Pair-like productions. Child classes:
      * :class:`.InterferogramDataset`
      * :class:`.GACOSPairs` 
   * :class:`.InterferogramDataset` : base Datasets for all Interferogram productions. Child classes (well known productions):
      * :class:`.HyP3S1`, :class:`.HyP3S1Burst`,
      * :class:`.LiCSAR`


GeoDataset
----------

.. autoclass:: faninsar.datasets.GeoDataset
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:


RasterDataset
-------------

.. autoclass:: faninsar.datasets.RasterDataset
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

PairDataset
-----------

.. autoclass:: faninsar.datasets.PairDataset
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:


InterferogramDataset
--------------------

.. autoclass:: faninsar.datasets.InterferogramDataset
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:


HyP3S1
------

.. autoclass:: faninsar.datasets.HyP3S1
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

HyP3S1Burst
-----------

.. autoclass:: faninsar.datasets.HyP3S1Burst
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

LiCSAR
------

.. autoclass:: faninsar.datasets.LiCSAR
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

ApsDataset
----------

.. autoclass:: faninsar.datasets.ApsDataset
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

GACOS
-----

.. autoclass:: faninsar.datasets.GACOS
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

GACOSPairs
----------

.. autoclass:: faninsar.datasets.GACOSPairs
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:
