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


Create a Dedicated Environment
------------------------------

We strongly recommend that you create a virtual environment for developing FanInSAR to isolate it from other Python installations on your system.

Create a new virtual environment using `conda <https://docs.conda.io/en/latest/>`_:

.. code-block:: bash

    conda create -n faninsar python=3.10


Activate the environment:

.. code-block:: bash

    conda activate faninsar


Install Dependencies
--------------------

Most of the FanInSAR dependencies are listed in :file:`pyproject.toml` and can be
installed from those files:

.. code-block:: bash

    python -m pip install ".[dev]"

FanInSAR requires that `setuptools
<https://setuptools.pypa.io/en/latest/setuptools.html>`_ is installed. It is
usually packaged with python, but if necessary can be installed using ``pip``:

.. code-block:: bash

    python -m pip install setuptools



Install for Development
-----------------------

Editable installs means that the environment Python will always use the most
recently changed version of your code. To install Sphinx Gallery in editable
mode, ensure you are in the sphinx-gallery directory

.. code-block:: bash

    cd FanInSAR

Then install using the editable flag:

.. code-block:: bash

    python -m pip install -e .


Run Tests
---------

Check that you are all set by running the tests:

.. code-block:: bash

    python -m pytest

Install pre-commit hooks
------------------------

pre-commit hooks check for things like spelling and formatting in contributed
code and documentation. To set up pre-commit hooks:

.. code-block:: bash

    pre-commit install

This will install the pre-commit hooks in your local repository. You can run the hooks manually with:

.. code-block:: bash

    pre-commit run --all-files

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
