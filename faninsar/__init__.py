"""FanInSAR's small science-oriented public surface."""

from __future__ import annotations

from faninsar.core.acquisition import Acquisition, Acquisitions
from faninsar.core.baseline import Baselines
from faninsar.core.pair import Pair, Pairs
from faninsar.core.sar_property import Frequency, Wavelength

# Import the grouped facades only after core values exist. Dataset modules use
# these values in their type-level imports during package initialization.
from faninsar import data, network, remote, stack
from faninsar.network import Network

__version__ = "0.1.dev0"

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
