Stack and Network
=================

FanInSAR has two workflow boundaries. A concrete Stack turns supported mission
source products into a completed generation. A Network opens one immutable
generation for interferogram and time-series analysis.

Use a supported mission Stack directly:

.. code-block:: python

   from faninsar.missions import NISARStack, S1Stack

   s1_stack = S1Stack.from_safes(safe_paths, work_dir=work_dir)
   nisar_stack = NISARStack.from_rslc(rslc_paths, work_dir=work_dir)

For a live Stack, use its Network directly after unwrapping:

.. code-block:: python

   s1_stack.unwrap()
   time_series = s1_stack.network.invert()

   # A model class selects NSBAS; omitting it selects SBAS.
   from faninsar.timeseries import LinearModel

   nsbas_result = s1_stack.network.invert(model=LinearModel)

Open an already processed generation through Network only when it was
persisted elsewhere:

.. code-block:: python

   from faninsar import Network

   network = Network.open(generation_path)
   interferograms = network.interferograms
   time_series = network.invert()

``faninsar.stack.Stack`` is the developer extension contract; it is not a root
constructor. ``Interferogram`` is available from ``faninsar.network`` and
``TimeSeries`` from ``faninsar.timeseries``.

API
---

.. autosummary::
   :toctree: generated/

   faninsar.Network
   faninsar.missions.S1Stack
   faninsar.missions.NISARStack
   faninsar.network.Interferogram
   faninsar.timeseries.TimeSeries
