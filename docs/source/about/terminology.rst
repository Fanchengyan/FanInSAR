.. _terminology:

Terminology
===========

.. glossary::

    Acquisition
        The date of SAR acquisition or image, expressed as a ``datetime.datetime`` object.

    Pair/Pairs
        A pair is a combination of two SAR acquisitions in acquisition order.

        .. note::
            In FanInSAR, pairs will be ordered automatically.

    Loop/Loops
        A loop contains a list of edge pairs and one diagonal pair. See paper [1]_ for more details.

    TripletLoop/TripletLoops
        triplet loop is a special loop that contains three pairs. See paper [1]_ for more details.

    CRS
        A coordinate reference system (CRS) is a coordinate-based local, regional,
        or global system used to locate geographical entities. In FanInSAR, the
        CRS is handled by the ``rasterio`` and ``pyproj`` packages. A valid CRS
        input for FanInSAR can be any type supported by the :meth:`pyproj.crs.CRS.from_user_input` method.

.. [1] : Fan, C., Liu, L., Zhao, Z., Mu, C., 2025. Pronounced underestimation of surface deformation due to unwrapping errors over tibetan plateau permafrost by sentinel-1 InSAR: identification and correction. J. Geophys. Res.: Earth Surf. 130, e2024JF007854. https://doi.org/10.1029/2024JF007854
