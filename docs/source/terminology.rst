.. _terminology:

Terminology
===========

.. glossary::

    Acquisition
        A single SAR acquisition, and is expressed as ``datetime.datetime`` object.

    Pair
        A pair is a combination of two SAR acquisitions.

    Pairs
        A collection of pairs. 

    Loop
        A loop 

    Loops
        A collection of loops.

    SBASNetwork
        A collection of loops and pairs.

    CRS
        A coordinate reference system (CRS) is a coordinate-based local, regional or global system used to locate geographical entities. In FanInSAR, the CRS is handled by the ``rasterio`` package.