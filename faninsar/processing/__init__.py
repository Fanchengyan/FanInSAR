"""Mission-neutral SAR processing domain contracts."""

from .backends import ArrayReader, ArrayWriter
from .contracts import (
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
from .coordinates import (
    CoordinateSystem,
    GeoGrid,
    OffsetField,
    RadarGrid,
    TransformDirection,
    TransformLUT,
)
from .errors import (
    GridMismatchError,
    InvalidProcessingStateError,
    ProcessingContractError,
)
from .provenance import ProcessingEvent, ProvenanceRecord, SoftwareIdentity
from .readers import (
    DopplerCentroidPolynomial,
    MissingCriticalMetadataError,
    SLCReader,
    SLCReadResult,
    ValidSampleMask,
    require_critical_metadata,
)
from .resampling import lanczos_resample
from .storage import ChunkAccessLog, ChunkedZarrArrayStore, InMemoryArrayStore
from .synthetic_slc import SyntheticSLCReader, SyntheticSLCSpec, write_synthetic_slc

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
    "MissingCriticalMetadataError",
    "OffsetField",
    "OrbitMetadata",
    "OrbitStateVector",
    "PairProduct",
    "ProcessingContractError",
    "ProcessingEvent",
    "ProvenanceRecord",
    "RadarGrid",
    "SLCProduct",
    "SLCReadResult",
    "SLCReader",
    "SoftwareIdentity",
    "StackProduct",
    "SyntheticSLCReader",
    "SyntheticSLCSpec",
    "TransformDirection",
    "TransformLUT",
    "UnwrapResult",
    "ValidSampleMask",
    "lanczos_resample",
    "require_critical_metadata",
    "write_synthetic_slc",
]
