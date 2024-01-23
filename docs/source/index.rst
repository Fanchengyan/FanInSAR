====================================
Welcome to FanInSAR's documentation!
====================================

.. image:: https://readthedocs.org/projects/faninsar/badge/?version=latest
    :target: https://faninsar.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Introduction
------------
FanInSAR is a fancy InSAR processing library written in Python. It is designed to help you process InSAR data in a pythonic, fast, and flexible way. If you want to implement your InSAR algorithm, FanInSAR is a highly recommended choice.

Highlight Features
------------------

- **Fast**: The heavy computation core in FanInSAR is written in ``Pytorch``, a high-performance deep learning library. You can easily run FanInSAR on GPU to speed up the processing.
- **Pythonic**: FanInSAR is written in Python, and the API is designed to be simple and intuitive. We design the class structure to be high-level and easy to use for the InSAR user. For example, you can easily load the ``HyP3`` product and process it with a few lines of code. 
- **Flexible**: FanInSAR is designed to be flexible. You can easily inherit the class or customize the processing pipeline to meet your needs. 


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   install
   User Guide <user_guide/index>
   FanInSAR API Reference <api/index>
   terminology



