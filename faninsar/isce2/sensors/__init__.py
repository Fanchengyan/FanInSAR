"""ISCE2 sensor-specific implementations.

This module provides sensor abstraction for different SAR satellites.
"""

from faninsar.isce2.sensors.base import BaseSensor
from faninsar.isce2.sensors.sentinel1 import Sentinel1Sensor

__all__ = ["BaseSensor", "Sentinel1Sensor"]
