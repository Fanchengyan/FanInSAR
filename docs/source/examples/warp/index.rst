.. _example_warp_header:

.. _warping_upon_loading:

====================
Warping upon loading
====================


In FanInSAR, datasets can be initialized with specific geospatial properties,
and their rasters are automatically warped to uniform properties upon data
loading. This feature allows for seamless resampling, reprojection,
or alignment to a reference raster by simply specifying the desired
coordinate reference system (CRS), resolution, extent, and resampling
method during dataset initialization, without the need to consider the
geospatial properties of the underlying original raster files.


.. base-gallery::
    :tooltip:

    align
    reproject
    resample
