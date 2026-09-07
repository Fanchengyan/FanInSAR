"""Single-look complex product values and stack membership."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from faninsar.processing.errors import reject_grid_mismatch, reject_invalid_state
from faninsar.processing.geometry.coordinates import (
    ArrayDescriptor,
    ArrayRepresentation,
    CoordinateSystem,
    ProcessingGrid,
)
from faninsar.processing.interferometry.products import (
    CalibrationState,
    CarrierState,
    CoregistrationState,
)

if TYPE_CHECKING:
    from faninsar.core.orbit import OrbitMetadata
    from faninsar.processing.interferometry.products import PairProduct


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


__all__ = ["SLCProduct", "StackProduct"]
