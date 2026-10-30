"""Mission-neutral SAR processing domain contracts."""

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.io.storage.arrays import (
    ChunkAccessLog,
    ChunkedZarrArrayStore,
    InMemoryArrayStore,
)
from faninsar.io.storage.provenance import (
    ProcessingEvent,
    ProvenanceRecord,
    SoftwareIdentity,
)
from faninsar.missions.protocols import (
    DopplerCentroidPolynomial,
    SLCReader,
    SLCReadResult,
    ValidSampleMask,
    require_critical_metadata,
)
from faninsar.processing.geometry.coordinates import (
    ArrayDescriptor,
    ArrayRepresentation,
    CoordinateSystem,
    GeoGrid,
    OffsetField,
    RadarGrid,
    TransformDirection,
    TransformLUT,
)

from .coregistration.resampling import lanczos_resample
from .errors import (
    GridMismatchError,
    InvalidProcessingStateError,
    ProcessingContractError,
)
from .interferometry.products import (
    CalibrationState,
    CarrierState,
    ComplexInterferogram,
    CoregistrationState,
    FlatteningState,
    PairProduct,
    UnwrapResult,
)
from .runtime.backends import ArrayReader, ArrayWriter
from .runtime.resources import (
    ResourceAdmissionError,
    ResourceAdmissionLedger,
    ResourceBudget,
    ResourceEstimate,
    ResourceReservation,
    ResourceUsage,
)
from .slc.products import SLCProduct, StackProduct

__all__ = [
    "ArrayDescriptor",
    "ArrayReader",
    "ArrayRepresentation",
    "ArrayWriter",
    "CalibrationState",
    "CarrierState",
    "ChunkAccessLog",
    "ChunkedZarrArrayStore",
    "ComplexInterferogram",
    "CoordinateSystem",
    "CoregistrationState",
    "DopplerCentroidPolynomial",
    "FlatteningState",
    "GeoGrid",
    "GridMismatchError",
    "InMemoryArrayStore",
    "InvalidProcessingStateError",
    "OffsetField",
    "OrbitMetadata",
    "OrbitStateVector",
    "PairProduct",
    "ProcessingContractError",
    "ProcessingEvent",
    "ProvenanceRecord",
    "RadarGrid",
    "ResourceAdmissionError",
    "ResourceAdmissionLedger",
    "ResourceBudget",
    "ResourceEstimate",
    "ResourceReservation",
    "ResourceUsage",
    "SLCProduct",
    "SLCReadResult",
    "SLCReader",
    "SoftwareIdentity",
    "StackProduct",
    "TransformDirection",
    "TransformLUT",
    "UnwrapResult",
    "ValidSampleMask",
    "lanczos_resample",
    "require_critical_metadata",
]
