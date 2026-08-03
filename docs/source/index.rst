====================================
Welcome to FanInSAR's documentation!
====================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11398347.svg
    :target: https://doi.org/10.5281/zenodo.11398347
    :alt: DOI

.. image:: https://readthedocs.org/projects/faninsar/badge/?version=latest
    :target: https://faninsar.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

Introduction
------------

**FanInSAR** is a :strong:`Fan`\ cy :strong:`In`\ terferometric :strong:`S`\ ynthetic :strong:`A`\ perture :strong:`R`\ adar (InSAR) time series analysis library written in Python. It aims to provide a foundational library for the development of InSAR algorithms, facilitating efficient processing of InSAR time series data by offering a Pythonic, fast, and flexible approach.





Why FanInSAR?
-------------

Most existing community InSAR software adopts a workflow-oriented approach. While this lowers the entry barrier for new users, it often compromises flexibility and extensibility. For algorithm researchers, integrating new methods into these rigid workflows can be challenging, highlighting the need for a more adaptable framework for InSAR time series analysis.

**FanInSAR** is designed to bridge this gap. It serves as a foundational library for InSAR time series processing, offering a flexible and extensible framework tailored for algorithm researchers and developers. FanInSAR is not a complete end-to-end InSAR processing system; rather, it provides building blocks for creating custom workflows. Its high-level API abstracts the complexity of the processing pipeline and hides low-level implementation details, allowing users to focus on developing and testing new algorithms. For researchers aiming to rapidly prototype and deploy their own InSAR methods, FanInSAR offers a fast and efficient starting point.

Highlight Features
------------------

- **Pythonic**: FanInSAR is written in Python and provides a user-friendly API. For example, a series of well-known InSAR datasets are provided in the form of Python classes; loading data from ``HyP3`` or ``LiCSAR`` products is as simple as providing the corresponding home directory. Sampling values from an interferometric dataset is as easy as calling the ``query()`` method by passing the spatial (Points, BoundingBox, Polygons) and temporal (Pairs) queries. The warping process during sampling (such as reprojecting and resampling) is automatically handled by the library.
- **Fast**: The core computation in FanInSAR is implemented using ``PyTorch``, a high-performance deep learning library. This allows for efficient processing on both CPU and GPU, enabling faster execution.
- **Flexible**: FanInSAR is designed to be flexible, allowing for customization and extension. Users can easily inherit classes or customize the processing pipeline for their specific needs.


.. grid:: 1 2 2 2
    :gutter: 5
    :class-container: sd-text-center
    :padding: 4


    .. grid-item-card:: Quick Overview
        :img-top: /_static/doc_index/index_get_start.svg
        :class-card: intro-card
        :shadow: md

        Learn how to use **FanInSAR** and discover its capabilities with a quick overview.

        +++

        .. button-ref:: quick_overview
            :ref-type: ref
            :click-parent:
            :color: primary
            :expand:

            To the Quick Start

    .. grid-item-card:: Examples Gallery
        :img-top: /_static/doc_index/index_example.svg
        :class-card: intro-card
        :shadow: md

        Explore the examples to see how **FanInSAR** can be used in practice.

        +++

        .. button-ref:: gallery_header
            :ref-type: ref
            :click-parent:
            :color: primary
            :expand:

            To the Examples Gallery

    .. grid-item-card:: API Reference
        :img-top: /_static/doc_index/index_api.svg
        :class-card: intro-card
        :shadow: md

        Dive into the **FanInSAR** API Reference to learn more about the available classes and methods.

        +++

        .. button-ref:: api
            :ref-type: ref
            :click-parent:
            :color: primary
            :expand:

            To the API Reference

    .. grid-item-card::  Contributing Guide
        :img-top: /_static/doc_index/index_contribute.svg
        :class-card: intro-card
        :shadow: md

        Do you want to contribute to **FanInSAR**? Check out the contributing guide.

        +++

        .. button-ref:: contributing
            :ref-type: ref
            :click-parent:
            :color: primary
            :expand:

            To the Contributing Guide


.. note::

    1. FanInSAR is under active development and is currently in the alpha stage. Its API may change in the future until it reaches a stable version.
    2. If you have any questions, suggestions, or issues, please feel free to open an issue or discussion on our GitHub repository at `GitHub Issues <https://github.com/Fanchengyan/FanInSAR/issues>`_ or `GitHub Discussions <https://github.com/Fanchengyan/FanInSAR/discussions>`_.




.. toctree::
    :maxdepth: 2
    :hidden:

    install
    user_guide/index
    examples/index
    api/index
    Contributing <contributing/index>
    About <about/index>
