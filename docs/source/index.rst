====================================
Welcome to FanInSAR's documentation!
====================================

.. image:: https://readthedocs.org/projects/faninsar/badge/?version=latest
    :target: https://faninsar.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Introduction
------------
FanInSAR is a fancy InSAR post-processing library written in Python. It is specifically designed to assist in the efficient processing of InSAR data, offering a Pythonic, fast, and flexible approach. If you are looking to implement your own InSAR algorithm, FanInSAR is highly recommended.

Highlight Features
------------------

- **Fast**: The core computation in FanInSAR is implemented using ``PyTorch``, a high-performance deep learning library. This allows for efficient processing on both CPU and GPU, enabling faster execution.
- **Pythonic**: FanInSAR is written in Python and provides a user-friendly API. The API is designed to be simple and intuitive, making it easy for InSAR users to work with. For example, loading data from ``HyP3`` or ``LiCSAR`` products is as simple as providing the corresponding home directory. Filtering interferometric pairs can be performed by a time slice, similar to the ``pandas`` package. The complex processing pipeline is abstracted away, allowing users to focus on algorithm development.
- **Flexible**: FanInSAR is designed to be flexible, allowing for customization and extension. Users can easily inherit classes or customize the processing pipeline to suit their specific needs.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   install
   User Guide <user_guide/index>
   API Reference <api/index>
   terminology



