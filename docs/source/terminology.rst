.. _terminology:

Terminology
===========

.. glossary::

    Acquisition
        The date of SAR acquisition or image, expressed as a ``datetime.datetime`` object.

    Pair/Pairs
        A pair is a combination of two SAR acquisitions in acquisition order.

        .. warning::
            In FanInSAR, a ``Pair`` must be in acquisition order, which means that the first acquisition should be earlier than the second acquisition. For example, a pair of (2018-01-01, 2018-02-01) is valid, but a pair of (2018-02-01, 2018-01-01) is invalid. If an invalid pair is provided, FanInSAR may yield unexpected results.

    Loop/Loops
        A loop contains a list of edge pairs and one diagonal pair. See paper [1]_ for more details.

    TripletLoop/TripletLoops
        triplet loop is a special loop that contains three pairs. See paper [1]_ for more details.

    CRS
        A coordinate reference system (CRS) is a coordinate-based local, regional, or global system used to locate geographical entities. In FanInSAR, the CRS is handled by the ``rasterio`` and ``pyproj`` packages. A valid CRS input for FanInSAR can be any type supported by the :meth:`pyproj.crs.CRS.from_user_input` method.

.. [1] : Fan, Chengyan, Lin Liu, Zhuoyi Zhao and Cuicui Mu. “Pronounced Underestimation of Surface Deformation due to Unwrapping Errors over Tibetan Plateau Permafrost by Sentinel-1 InSAR and Its Correction”.
