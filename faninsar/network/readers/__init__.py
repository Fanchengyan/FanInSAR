"""Readers for persisted Network product collections.

The reader package contains the concrete file-backed implementations used by
``Network.open``. It is separate from the public Network workflow and does not
expose a second product hierarchy.
"""

from __future__ import annotations

from .geometry import NetworkGeometry
from .interferogram import InterferogramCollection
from .metadata import (
    GEOMETRY_ASSETS,
    INTERFEROGRAM_ASSETS,
    GeometryAssetName,
    InterferogramAssetName,
)
from .stack import StackInterferogramCollection
from .timeseries import NetworkTimeSeries

__all__ = [
    "GEOMETRY_ASSETS",
    "INTERFEROGRAM_ASSETS",
    "GeometryAssetName",
    "InterferogramAssetName",
    "InterferogramCollection",
    "NetworkGeometry",
    "NetworkTimeSeries",
    "StackInterferogramCollection",
]
