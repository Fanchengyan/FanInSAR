.. _terminology:

Terminology
===========

.. glossary::

    Acquisition
        The date of a SAR acquisition/image, and is expressed as ``datetime.datetime`` object.

    Pair
        A pair is a combination of two SAR acquisitions in acquisition order.

        .. note::

            In FanInSAR, a ``Pair`` must be in acquisition order, or in other words, 
            the first acquisition should be earlier than the second acquisition. 
            For example, a pair of (2018-01-01, 2018-02-01) is valid, but a pair 
            of (2018-02-01, 2018-01-01) is invalid.
            If you do not follow this rule, the results may be unexpected or even wrong.

    Pairs
        A collection of pairs. 

    Loop
        A loop contains a list of edge pairs and one diagonal pair. See paper [1]_ for more details.

    Loops
        A collection of loops.

    TripletLoop
        triplet loop is a special loop that contains three pairs. See paper [1]_ for more details.

    TripletLoops
        A collection of triplet loops.

    SBASNetwork
        A collection of loops and pairs.

    CRS
        A coordinate reference system (CRS) is a coordinate-based local, regional or global system used to locate geographical entities. In FanInSAR, the CRS is handled by the ``rasterio`` package.

.. [1] : Fan, Chengyan, Lin Liu, Zhuoyi Zhao and Cuicui Mu. “Unwrapping errors matter: Pronounced underestimation of surface deformation over Tibetan Plateau permafrost by Sentinel-1 InSAR and its correction”.




