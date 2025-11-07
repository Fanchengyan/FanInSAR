.. _geo_datasets:

========
Datasets
========

Datasets in FanInSAR are used to process large amounts of raster files that have common geospatial information. These files are combined to form a complete dataset.

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

   faninsar.datasets.GeoDataset
   faninsar.datasets.RasterDataset
   faninsar.datasets.PairDataset
   faninsar.datasets.InterferogramDataset
   faninsar.datasets.CoherenceDataset
   faninsar.datasets.HyP3S1
   faninsar.datasets.HyP3S1Burst
   faninsar.datasets.LiCSAR
   faninsar.datasets.ApsDataset
   faninsar.datasets.GACOS
   faninsar.datasets.GACOSPairs

Hierarchical Datasets
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   faninsar.datasets.XarrayDataset
   faninsar.datasets.HierarchicalDataset
