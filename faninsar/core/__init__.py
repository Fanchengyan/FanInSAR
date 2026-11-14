"""Reusable scientific values: acquisitions, pairs, baselines, and SAR types.

``core`` must not import ``missions``, ``processing``, ``timeseries``,
``io``, or runtime backends (lint-enforced).
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
from faninsar.core.parameters import Parameter
from faninsar.core.physical import (
    PHYSICAL_TYPE_MEMBERS,
    SEQ_TRANSITIONS,
    PhysicalType,
    is_complete_lattice,
)
from faninsar.core.sar_missions import Sentinel1
from faninsar.core.sar_tools import (
    Looks,
    PhaseDeformationConverter,
    coerce_looks,
    multi_look,
)

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
    "Looks",
    "Loop",
    "Loops",
    "Pair",
    "Pairs",
    "PairsFactory",
    "Parameter",
    "PhaseDeformationConverter",
    "PhysicalType",
    "Sentinel1",
    "TripletLoop",
    "TripletLoops",
    "Wavelength",
    "coerce_looks",
    "is_complete_lattice",
    "multi_look",
]
