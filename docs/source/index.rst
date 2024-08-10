====================================
Welcome to FanInSAR's documentation!
====================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11398347.svg
  :target: https://doi.org/10.5281/zenodo.11398347
  :alt: DOI

.. image:: https://readthedocs.org/projects/faninsar/badge/?version=latest
    :target: https://faninsar.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

Introduction
------------

FanInSAR is a Fancy Interferometric Synthetic Aperture Radar (InSAR) time series analysis library written in Python. It aims to provide a foundational library for the development of InSAR algorithms, facilitating efficient processing of InSAR time series data by offering a Pythonic, fast, and flexible approach. FanInSAR’s high-level API abstracts the complex processing pipeline and conceals the low-level programming details, enabling users to focus on algorithm development. For researchers and developers aiming to rapidly implement their own InSAR algorithms, FanInSAR offers a quick start for their projects.


Highlight Features
------------------

- **Pythonic**: FanInSAR is written in Python and provides a user-friendly API. The API is designed to be simple and intuitive, by abstracting the complex processing pipeline and concealing the low-level programming details, which allows users to focus on algorithm development. For example, loading data from ``HyP3`` or ``LiCSAR`` products is as simple as providing the corresponding home directory. Filtering interferometric pairs can be performed by a time slice, similar to the ``pandas`` package.
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
            :color: secondary
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
            :color: secondary
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
            :color: secondary
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
            :color: secondary
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
    auto_examples/index
    api/index
    Contributing <contributing/index>
    About <about/index>
