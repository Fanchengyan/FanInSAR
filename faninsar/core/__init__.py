"""Domain nouns: pairs, frame façade, physical types.

``core`` must not import ``missions``, ``processing``, ``timeseries``,
``io``, or ``compute`` (lint-enforced).
"""

from __future__ import annotations

from faninsar.core.acquisition import (
    Acquisition,
    AcquisitionRecord,
    Acquisitions,
    DateManager,
    DaySpan,
)
from faninsar.core.baseline import Baselines
from faninsar.core.interferogram import Interferogram
from faninsar.core.loops import Loop, Loops, TripletLoop, TripletLoops
from faninsar.core.network import (
    AcquisitionKey,
    AssetKind,
    AssetTransform,
    AssetTransformOperation,
    Network,
    NetworkProduct,
    NetworkProductIndex,
    NetworkProductKey,
    NetworkProductRecord,
    PhaseConvention,
)
from faninsar.core.pairs import Pair, Pairs, PairsFactory
from faninsar.core.physical import (
    PHYSICAL_TYPE_MEMBERS,
    SEQ_TRANSITIONS,
    PhysicalType,
    is_complete_lattice,
)
from faninsar.core.sar_missions import SAR, Sentinel1
from faninsar.core.sar_property import Frequency, Wavelength
from faninsar.core.sar_tools import PhaseDeformationConverter, multi_look

__all__ = [
    "PHYSICAL_TYPE_MEMBERS",
    "SAR",
    "SEQ_TRANSITIONS",
    "Acquisition",
    "AcquisitionKey",
    "AcquisitionRecord",
    "Acquisitions",
    "AssetKind",
    "AssetTransform",
    "AssetTransformOperation",
    "Baselines",
    "DateManager",
    "DaySpan",
    "Frequency",
    "Interferogram",
    "Loop",
    "Loops",
    "Network",
    "NetworkProduct",
    "NetworkProductIndex",
    "NetworkProductKey",
    "NetworkProductRecord",
    "Pair",
    "Pairs",
    "PairsFactory",
    "PhaseConvention",
    "PhaseDeformationConverter",
    "PhysicalType",
    "Sentinel1",
    "TripletLoop",
    "TripletLoops",
    "Wavelength",
    "is_complete_lattice",
    "multi_look",
]
