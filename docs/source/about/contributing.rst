.. _contributing:

==================
Contributing Guide
==================


We appreciate your help in improving this document and our library!

Please `open an issue <https://github.com/Fanchengyan/FanInSAR/issues>`_
if you face any problems or have suggestions for improvements. We are always happy to help.


If you are interested in contributing code or documentation, we strongly
recommend that you install a development version of FanInSAR in a
development environment. If you are unfamiliar with the git/github workflow,
please see Github's guide to `contributing to projects
<https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`_.

This guide assumes familiarity with the Github workflow and focuses on aspects
specific to contributing to FanInSAR.


Get Latest Source Code
----------------------

You can get the latest development source code from our `Github repository
<https://github.com/Fanchengyan/FanInSAR>`_. Fork the repository and clone the forked repository to your local machine:

.. code-block:: bash

    git clone https://github.com/<your github user name>/FanInSAR


Install Dependencies
--------------------

FanInSAR uses `uv <https://docs.astral.sh/uv/>`_ to create and manage its
development environment. From the repository root, install the project and its
development dependencies with:

.. code-block:: bash

    uv sync --dev


Run Tests
---------

Check that you are all set by running the tests:

.. code-block:: bash

    uv run pytest

The test configuration measures coverage for :mod:`faninsar`. Run a focused
test by passing its path after ``pytest``, for example
``uv run pytest tests/_core/sar/test_pairs.py``.

Some shells inherit ``PROJ_DATA`` or the legacy ``PROJ_LIB`` from a Conda or
system installation. Those variables can point Rasterio and pyproj at an
incompatible ``proj.db`` while tests run in the uv environment. The root test
configuration checks each inherited override in an isolated Python process
before test modules are imported. It preserves paths that let both pyproj and
Rasterio resolve EPSG:4326, and removes only incompatible or unresponsive
overrides so the geospatial wheels can use their matching bundled PROJ data.
For non-test commands, unset the variables in the shell if PROJ reports a
database layout-version mismatch.

Install pre-commit hooks
------------------------

pre-commit hooks check for things like spelling and formatting in contributed
code and documentation. To set up pre-commit hooks:

.. code-block:: bash

    uv run pre-commit install

This will install the pre-commit hooks in your local repository. You can run the hooks manually with:

.. code-block:: bash

    uv run pre-commit run --all-files

Testing
-------

All code contributions should be tested. We use the `pytest
<https://docs.pytest.org/>`_ testing framework to build test
pages. Tests can be found in :file:`faninsar/tests`.



Build the documentation
-----------------------

If you are contributing to the documentation, you can build the docs locally to see how your changes will look.

To build the docs, run:


.. code-block:: bash

    cd docs
    make html

After building the docs, you can view them by opening :file:`_build/html/index.html` in your browser.

To clean up the build files and generated galleries, run:

.. code-block:: bash

    make clean


Contributing
------------

When contributing to FanInSAR, please follow the `Contributor Covenant
<https://www.contributor-covenant.org/version/2/0/code_of_conduct/>`_ in all
your interactions with the project.
