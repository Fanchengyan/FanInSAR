.. _geo_datasets:

========
Datasets
========

Datasets in ``faninsar.data.datasets`` provide reusable access to raster and
hierarchical scientific products with common geospatial information.

The following datasets are available:

* basic geospatial Datasets:
   * :class:`.GeoDataset`
   * :class:`.RasterDataset` : base Datasets for all Raster-like productions.
* InSAR related Datasets:
   * :class:`.ApsDataset` : base Datasets for all APS (Atmospheric Phase Screen) productions. Child classes:
      * :class:`.GACOS`
   * :class:`.PairDataset` : base Datasets for all Pair-like productions. Child classes:
      * :class:`.InterferogramDataset`, :class:`.CoherenceDataset`
      * :class:`.GACOSPairs`
   * :class:`.InterferogramDataset` : base Datasets for all Interferogram productions. Child classes (well known productions):
      * :class:`.HyP3S1`, :class:`.HyP3S1Burst`,
      * :class:`.LiCSAR`


Raster Based Datasets
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   faninsar.data.datasets.GeoDataset
   faninsar.data.datasets.RasterDataset
   faninsar.data.datasets.PairDataset
   faninsar.data.datasets.InterferogramDataset
   faninsar.data.datasets.CoherenceDataset
   faninsar.data.datasets.HyP3S1
   faninsar.data.datasets.LiCSAR
   faninsar.data.datasets.ApsDataset
   faninsar.data.datasets.GACOS
   faninsar.data.datasets.GACOSPairs

Hierarchical Datasets
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   faninsar.data.datasets.XarrayDataset
   faninsar.data.datasets.HierarchicalDataset
