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

Frame-Level Products
~~~~~~~~~~~~~~~~~~~~

Standardized geocoded raster InSAR frames: geometry + interferogram
collections + time series, with COG/Zarr/STAC round-tripping and MintPy
interoperability.

.. autosummary::
   :toctree: generated/

   faninsar.datasets.frame.Frame
   faninsar.datasets.frame.FrameGeometry
   faninsar.datasets.frame.FrameInterferogramCollection
   faninsar.datasets.frame.FrameTimeSeries
   faninsar.datasets.frame.RemoteFrame
   faninsar.datasets.frame.build_mintpy

Inversion Pipeline
~~~~~~~~~~~~~~~~~~

NSBAS time-series inversion orchestrated via Dask spatial tiling and
(optional) GPU acceleration.

.. autosummary::
   :toctree: generated/

   faninsar.pipeline.InversionPipeline
