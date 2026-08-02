"""Product contracts and stage tokens for the processing spine."""

from __future__ import annotations

from faninsar.processing.contracts.ifg import Interferogram, InterferogramStack
from faninsar.processing.contracts.products import (
    ArrayDescriptor,
    ArrayRepresentation,
    CalibrationState,
    CarrierState,
    ComplexInterferogram,
    CoregistrationState,
    FlatteningState,
    OrbitMetadata,
    OrbitStateVector,
    PairProduct,
    SLCProduct,
    StackProduct,
    UnwrapResult,
)
from faninsar.processing.contracts.stage import Stage, StageNode
from faninsar.processing.contracts.tokens import ArrayToken, assert_token

__all__ = [
    "ArrayDescriptor",
    "ArrayRepresentation",
    "ArrayToken",
    "CalibrationState",
    "CarrierState",
    "ComplexInterferogram",
    "CoregistrationState",
    "FlatteningState",
    "Interferogram",
    "InterferogramStack",
    "OrbitMetadata",
    "OrbitStateVector",
    "PairProduct",
    "SLCProduct",
    "StackProduct",
    "Stage",
    "StageNode",
    "UnwrapResult",
    "assert_token",
]
