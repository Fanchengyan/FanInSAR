.. _custom_datasets:

===============
Custom Datasets
===============

In this tutorial, we will discuss how to customize your own datasets in ``FanInSAR``.

Choosing a base class
---------------------

To customize your own dataset, you need to choose a base class to inherit from. In ``FanInSAR``, the base class for datasets are as follows:

- :class:`~faninsar.datasets.RasterDataset`: The base class for raster datasets, just like the files with the extension of ``.tif`` / ``.tiff``.
- :class:`~faninsar.datasets.InterferogramDataset`: The base class for interferogram datasets.


.. - :class:`~faninsar.datasets.HierarchicalDataset`: The base class for hierarchical datasets, just like the files with the extension of ``.h5`` (hdf5) or ``.nc`` (netcdf).
