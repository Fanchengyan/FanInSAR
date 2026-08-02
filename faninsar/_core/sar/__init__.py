"""Deprecated path — import from :mod:`faninsar.core` instead.

This module re-exports domain nouns during the greenfield rehome. Callers
should migrate to ``faninsar.core``; this package will be removed.
"""

from __future__ import annotations

from faninsar.core import (
    SAR,
    Acquisition,
    Baselines,
    DateManager,
    DaySpan,
    Frequency,
    Loop,
    Loops,
    Pair,
    Pairs,
    PairsFactory,
    PhaseDeformationConverter,
    Sentinel1,
    TripletLoop,
    TripletLoops,
    Wavelength,
    multi_look,
)

__all__ = [
    "SAR",
    "Acquisition",
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
    "Sentinel1",
    "TripletLoop",
    "TripletLoops",
    "Wavelength",
    "multi_look",
]
