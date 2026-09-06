"""FanInSAR's small science-oriented public surface.

This module is the sole root API definition.  Keep the whitelist intentionally
small: workflow implementations and persistence/runtime details belong to
their owning subpackages.
"""

from __future__ import annotations

from faninsar.core.acquisition import Acquisition, Acquisitions
from faninsar.core.baseline import Baselines
from faninsar.core.pair import Pair, Pairs
from faninsar.core.sar_property import Frequency, Wavelength

# Import grouped facades only after core values exist. Dataset modules use
# these values in their type-level imports during package initialization.
from faninsar import data, network, remote, stack  # isort: skip
from faninsar.network import Network

__version__ = "0.1.dev0"

# Hard cap: the root surface is deliberately limited to the stable scientific
# values and grouped workflow facades below.
__all__ = [
    "Acquisition",
    "Acquisitions",
    "Baselines",
    "Frequency",
    "Network",
    "Pair",
    "Pairs",
    "Wavelength",
    "data",
    "network",
    "remote",
    "stack",
]
