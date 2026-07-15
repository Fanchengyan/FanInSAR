"""Immutable public SAR product contracts and processing state machine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from .coordinates import (
    ArrayDescriptor,
    ArrayRepresentation,
    CoordinateSystem,
    ProcessingGrid,
    Vector3,
)
from .errors import reject_grid_mismatch, reject_invalid_state

if TYPE_CHECKING:
    from datetime import datetime


class CarrierState(StrEnum):
    """TOPS carrier handling state."""

    PRESENT = "present"
    DERAMPED = "deramped"
    RESTORED = "restored"


class CoregistrationState(StrEnum):
    """Registration state relative to the stack reference."""

    NOT_REGISTERED = "not_registered"
    REGISTERED = "registered"


class FlatteningState(StrEnum):
    """Interferometric reference-phase removal state."""

    NOT_APPLIED = "not_applied"
    APPLIED = "applied"


class CalibrationState(StrEnum):
    """Radiometric calibration state."""

    RAW_DN = "raw_dn"
    CALIBRATED = "calibrated"


@dataclass(frozen=True, slots=True)
class OrbitStateVector:
    """Time-tagged Cartesian orbit position and velocity."""

    time: datetime
    position_m: Vector3
    velocity_m_s: Vector3

    def __post_init__(self) -> None:
        """Validate that the orbit epoch is timezone-aware."""
        if self.time.tzinfo is None:
            reject_invalid_state("orbit state-vector time must include a timezone")


@dataclass(frozen=True, slots=True)
class OrbitMetadata:
    """Ordered orbit state vectors and their reference frame."""

    reference_frame: str
    source: str
    vectors: tuple[OrbitStateVector, ...]

    def __post_init__(self) -> None:
        """Validate required identity and strictly ordered orbit epochs."""
        if not self.reference_frame or not self.source or not self.vectors:
            reject_invalid_state("orbit frame, source, and vectors are required")
        times = tuple(vector.time for vector in self.vectors)
        if times != tuple(sorted(times)) or len(times) != len(set(times)):
            reject_invalid_state("orbit state-vector times must be unique and ordered")


@dataclass(frozen=True, slots=True)
class SLCProduct:
    """Mission-neutral single-look complex product metadata."""

    acquisition_id: str
    grid: ProcessingGrid
    samples: ArrayDescriptor
    orbit: OrbitMetadata
    carrier: CarrierState
    coregistration: CoregistrationState
    calibration: CalibrationState
    deramp_application_id: str | None = None

    def __post_init__(self) -> None:
        """Validate complex samples, grid identity, and carrier state."""
        if not self.acquisition_id:
            reject_invalid_state("SLC acquisition ID must not be empty")
        if self.samples.representation is not ArrayRepresentation.COMPLEX:
            reject_invalid_state("SLC samples must retain their complex representation")
        if self.samples.shape != self.grid.shape:
            reject_grid_mismatch("SLC samples and metadata must use the same grid")
        if self.carrier is CarrierState.DERAMPED and not self.deramp_application_id:
            reject_invalid_state("deramped SLC requires an application ID")
        if self.carrier is CarrierState.PRESENT and self.deramp_application_id:
            reject_invalid_state(
                "carrier-present SLC cannot have a deramp application ID"
            )

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Return the coordinate system of the product grid."""
        return self.grid.coordinate_system

    def deramp(self, application_id: str) -> SLCProduct:
        """Return a product marked as deramped exactly once."""
        if self.carrier is CarrierState.DERAMPED:
            reject_invalid_state("SLC is already deramped")
        if not application_id:
            reject_invalid_state("deramp application ID must not be empty")
        return replace(
            self,
            carrier=CarrierState.DERAMPED,
            deramp_application_id=application_id,
        )


@dataclass(frozen=True, slots=True)
class ComplexInterferogram:
    """Complex interferogram formed from registered SLC products."""

    pair_id: str
    primary_id: str
    secondary_id: str
    grid: ProcessingGrid
    samples: ArrayDescriptor
    flattening: FlatteningState

    @classmethod
    def form(
        cls,
        primary: SLCProduct,
        secondary: SLCProduct,
        *,
        uri: str | None = None,
    ) -> ComplexInterferogram:
        """Create metadata for an interferogram after validating its inputs."""
        if (
            primary.coregistration is not CoregistrationState.REGISTERED
            or secondary.coregistration is not CoregistrationState.REGISTERED
        ):
            reject_invalid_state("interferogram inputs must both be registered")
        if primary.grid != secondary.grid:
            reject_grid_mismatch("interferogram inputs must use the same grid")
        pair_id = f"{primary.acquisition_id}_{secondary.acquisition_id}"
        return cls(
            pair_id=pair_id,
            primary_id=primary.acquisition_id,
            secondary_id=secondary.acquisition_id,
            grid=primary.grid,
            samples=ArrayDescriptor(
                uri=uri or f"memory://{pair_id}/complex",
                shape=primary.grid.shape,
                dtype="complex64",
                representation=ArrayRepresentation.COMPLEX,
            ),
            flattening=FlatteningState.NOT_APPLIED,
        )

    def __post_init__(self) -> None:
        """Validate complex representation and grid identity."""
        if self.samples.representation is not ArrayRepresentation.COMPLEX:
            reject_invalid_state("interferogram samples must remain complex")
        if self.samples.shape != self.grid.shape:
            reject_grid_mismatch("interferogram samples must match their grid")


@dataclass(frozen=True, slots=True)
class UnwrapResult:
    """Unwrapped phase and algorithm identity for one pair."""

    pair_id: str
    grid: ProcessingGrid
    phase: ArrayDescriptor
    method: str

    def __post_init__(self) -> None:
        """Validate unwrapped phase representation and grid identity."""
        if self.phase.representation is not ArrayRepresentation.PHASE:
            reject_invalid_state("unwrap output must be a phase array")
        if self.phase.shape != self.grid.shape:
            reject_grid_mismatch("unwrap phase must match its grid")
        if not self.method:
            reject_invalid_state("unwrap method must not be empty")


@dataclass(frozen=True, slots=True)
class PairProduct:
    """Interferogram and optional unwrapped result for one acquisition pair."""

    interferogram: ComplexInterferogram
    unwrap: UnwrapResult | None = None

    def __post_init__(self) -> None:
        """Validate that unwrapping belongs to the same pair and grid."""
        if self.unwrap is not None and (
            self.unwrap.pair_id != self.interferogram.pair_id
            or self.unwrap.grid != self.interferogram.grid
        ):
            reject_invalid_state("unwrap result must match its interferogram")


@dataclass(frozen=True, slots=True)
class StackProduct:
    """Non-empty collection of uniquely identified pair products."""

    stack_id: str
    pairs: tuple[PairProduct, ...]

    def __post_init__(self) -> None:
        """Validate stack identity, membership, and pair uniqueness."""
        pair_ids = tuple(pair.interferogram.pair_id for pair in self.pairs)
        if not self.stack_id or not pair_ids:
            reject_invalid_state("stack ID and at least one pair are required")
        if len(pair_ids) != len(set(pair_ids)):
            reject_invalid_state("stack pair IDs must be unique")


__all__ = [
    "ArrayDescriptor",
    "ArrayRepresentation",
    "CalibrationState",
    "CarrierState",
    "ComplexInterferogram",
    "CoregistrationState",
    "FlatteningState",
    "OrbitMetadata",
    "OrbitStateVector",
    "PairProduct",
    "SLCProduct",
    "StackProduct",
    "UnwrapResult",
]
