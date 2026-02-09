"""faninsar.isce2 - ISCE2 Stack Sentinel Processing.

A modern Python package for managing ISCE2 InSAR processing workflows.
Provides centralized path management, command execution, and workflow
orchestration for Sentinel-1 stack processing.
"""

__version__ = "0.1.0"

from faninsar.isce2.command_manager import (
    Command,
    TopsStackCommands,
)
from faninsar.isce2.path_manager import (
    Multilook,
    PathManager,
)
from faninsar.isce2.sensors.base import BaseSensor
from faninsar.isce2.sensors.sentinel1 import Sentinel1Sensor
from faninsar.isce2.tops_workflow import (
    TopsStackWorkflow,
    WorkflowConfig,
)
from faninsar.isce2.workflows.base import BaseWorkflow
from faninsar.isce2.workflows.interferogram_stack import InterferogramStack
from faninsar.isce2.workflows.slc_stack import SLCStack

__all__ = [
    "BaseSensor",
    "BaseWorkflow",
    "Command",
    "InterferogramStack",
    "Multilook",
    "PathManager",
    "SLCStack",
    "Sentinel1Sensor",
    "TopsStackCommands",
    "TopsStackWorkflow",
    "WorkflowConfig",
    "__version__",
]
