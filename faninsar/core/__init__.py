"""Domain nouns: pairs, frame façade, physical types.

``core`` must not import ``missions``, ``processing``, ``timeseries``,
``io``, or ``compute`` (lint-enforced).
"""

from __future__ import annotations

from faninsar.core.acquisition import (
    Acquisition,
    AcquisitionRecord,
    Acquisitions,
)
from faninsar.core.baseline import Baselines
from faninsar.core.constants import SAR, Frequency, Wavelength
from faninsar.core.dates import DateManager, DaySpan
from faninsar.core.loops import Loop, Loops, TripletLoop, TripletLoops
from faninsar.core.pair import Pair, Pairs, PairsFactory
from faninsar.core.physical import (
    PHYSICAL_TYPE_MEMBERS,
    SEQ_TRANSITIONS,
    PhysicalType,
    is_complete_lattice,
)
from faninsar.core.sar_missions import Sentinel1
from faninsar.core.sar_tools import PhaseDeformationConverter, multi_look

__all__ = [
    "PHYSICAL_TYPE_MEMBERS",
    "SAR",
    "SEQ_TRANSITIONS",
    "Acquisition",
    "AcquisitionRecord",
    "Acquisitions",
    "Baselines",
    "DateManager",
    "DaySpan",
    "Frequency",
    "Loop",
    "Loops",
    "Pair",
    "Pairs",
    "PairsFactory",
    "PhaseDeformationConverter",
    "PhysicalType",
    "Sentinel1",
    "TripletLoop",
    "TripletLoops",
    "Wavelength",
    "is_complete_lattice",
    "multi_look",
]
