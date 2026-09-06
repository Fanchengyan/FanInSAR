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

from .coordinates import (
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
    PairConfigurationMigrationError,
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
from .readers import (
    DopplerCentroidPolynomial,
    MissingCriticalMetadataError,
    SLCReader,
    SLCReadResult,
    ValidSampleMask,
    require_critical_metadata,
)
from .runtime.backends import ArrayReader, ArrayWriter
from .runtime.resources import (
    ProcessTreeAdmission,
    ProcessTreeMemoryWatchdog,
    ProcessTreeSampler,
    ProcessTreeSamplingError,
    ProcessTreeSnapshot,
    ResourceAdmissionError,
    ResourceAdmissionLedger,
    ResourceBudget,
    ResourceEstimate,
    ResourceReservation,
    ResourceUsage,
    bootstrap_worker_runtime,
)
from .runtime.source_snapshots import (
    ImmutableSourceSnapshot,
    SourceSnapshotEntry,
    snapshot_local_source,
)
from .slc.products import SLCProduct, StackProduct
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
    "ImmutableSourceSnapshot",
    "InMemoryArrayStore",
    "InvalidProcessingStateError",
    "MissingCriticalMetadataError",
    "OffsetField",
    "OrbitMetadata",
    "OrbitStateVector",
    "PairConfigurationMigrationError",
    "PairProduct",
    "ProcessTreeAdmission",
    "ProcessTreeMemoryWatchdog",
    "ProcessTreeSampler",
    "ProcessTreeSamplingError",
    "ProcessTreeSnapshot",
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
    "SourceSnapshotEntry",
    "StackProduct",
    "SyntheticSLCReader",
    "SyntheticSLCSpec",
    "TransformDirection",
    "TransformLUT",
    "UnwrapResult",
    "ValidSampleMask",
    "bootstrap_worker_runtime",
    "lanczos_resample",
    "require_critical_metadata",
    "snapshot_local_source",
    "write_synthetic_slc",
]
