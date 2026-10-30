"""Shared scientific contracts for prepared geometry reuse.

The Pair, geo, and Stack orchestration layers exchange these immutable
records instead of importing each other's mutable workflow state.  The
module owns the residual/phase lineage types consumed by the S1 production
pipeline; it deliberately keeps no lease, token, or provider state machine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from faninsar.processing.mosaicking.grid import GeoGridSpec

ArtifactDomain = Literal["radar", "geo"]
ResidualPolicy = Literal["measure_pair", "consume_solution", "geometry_only"]


class ResidualStatus(StrEnum):
    """Validity state for a measured or consumed residual."""

    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"


class PhaseCarrier(StrEnum):
    """Carrier state tracked by :class:`PhaseState`."""

    PRESENT = "present"
    DERAMPED = "deramped"
    RESTORED = "restored"


class GeometricPhase(StrEnum):
    """Geometric phase state tracked by :class:`PhaseState`."""

    NONE = "none"
    FLAT_REMOVED = "flat_removed"
    TOPO_REMOVED = "topo_removed"


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        reject_invalid_state(f"{name} must be a non-empty string")


def _require_digest(value: str, name: str) -> None:
    _require_text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        reject_invalid_state(f"{name} must be a lowercase SHA-256 digest")


def _canonical_json(value: object) -> bytes:
    """Serialize an identity projection using one deterministic encoding."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"identity value is not canonicalizable: {error}")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def geo_grid_identity(grid: GeoGridSpec) -> str:
    """Return the canonical identity of a geographic output grid.

    Parameters
    ----------
    grid : GeoGridSpec
        Fully resolved CRS, transform, shape, resolution, and bounding box.

    Returns
    -------
    str
        Lowercase SHA-256 identity used to bind a prepared LUT to one exact
        geographic grid.

    """
    return _digest(
        {
            "crs": str(grid.crs),
            "transform": [float(value) for value in grid.transform],
            "width": int(grid.width),
            "height": int(grid.height),
            "resolution_m": [float(value) for value in grid.resolution_m],
            "bbox": [float(value) for value in grid.bbox],
        }
    )


@dataclass(frozen=True, slots=True)
class PreparedGeometryArrayPayload:
    """Validated read-only dense controls loaded from a prepared generation."""

    range_offset_px: np.ndarray
    azimuth_offset_px: np.ndarray
    coverage: np.ndarray
    uncertainty_px: np.ndarray
    source_shape: tuple[int, int]
    crop_bounds: tuple[int, int, int, int]
    control_spacing: int

    def __post_init__(self) -> None:
        """Validate the immutable field domain and canonical dtypes."""
        if len(self.source_shape) != 2 or any(
            not isinstance(value, int) or value <= 0 for value in self.source_shape
        ):
            reject_invalid_state("prepared geometry source shape is invalid")
        if len(self.crop_bounds) != 4 or any(
            not isinstance(value, int) or value < 0 for value in self.crop_bounds
        ):
            reject_invalid_state("prepared geometry crop bounds are invalid")
        row0, row1, col0, col1 = self.crop_bounds
        if not (row0 < row1 <= self.source_shape[0]):
            reject_invalid_state("prepared geometry row crop is outside the source")
        if not (col0 < col1 <= self.source_shape[1]):
            reject_invalid_state("prepared geometry column crop is outside the source")
        expected_shape = (row1 - row0, col1 - col0)
        arrays = (
            self.range_offset_px,
            self.azimuth_offset_px,
            self.coverage,
            self.uncertainty_px,
        )
        if any(array.shape != expected_shape for array in arrays):
            reject_invalid_state("prepared geometry payload shape does not match crop")
        if (
            self.range_offset_px.dtype != np.float32
            or self.azimuth_offset_px.dtype != np.float32
            or self.uncertainty_px.dtype != np.float32
            or self.coverage.dtype != np.bool_
        ):
            reject_invalid_state("prepared geometry payload dtypes are not canonical")
        if self.control_spacing < 1:
            reject_invalid_state("prepared geometry control spacing must be positive")
        for array in arrays:
            array.setflags(write=False)


@dataclass(frozen=True, slots=True)
class PreparedLutArrayPayload:
    """Validated read-only geo2rdr LUT arrays from one prepared view.

    The arrays are a cropped view of the canonical geographic grid.  The
    crop origin is carried explicitly so consumers cannot reopen a fixed-name
    memmap with a guessed shape or silently shift the LUT.
    """

    az_full: np.ndarray
    rg_full: np.ndarray
    valid: np.ndarray
    height_full: np.ndarray
    full_radar_shape: tuple[int, int]
    geo_grid_shape: tuple[int, int]
    crop_bounds: tuple[int, int, int, int]
    height_m: float

    def __post_init__(self) -> None:
        """Validate LUT dtypes, shape, and crop metadata."""
        if len(self.full_radar_shape) != 2 or any(
            not isinstance(value, int) or value <= 0 for value in self.full_radar_shape
        ):
            reject_invalid_state("prepared LUT radar shape is invalid")
        if len(self.geo_grid_shape) != 2 or any(
            not isinstance(value, int) or value <= 0 for value in self.geo_grid_shape
        ):
            reject_invalid_state("prepared LUT geographic grid shape is invalid")
        if len(self.crop_bounds) != 4 or any(
            not isinstance(value, int) or value < 0 for value in self.crop_bounds
        ):
            reject_invalid_state("prepared LUT crop bounds are invalid")
        row0, row1, col0, col1 = self.crop_bounds
        if not (
            row0 < row1 <= self.geo_grid_shape[0]
            and col0 < col1 <= self.geo_grid_shape[1]
        ):
            reject_invalid_state("prepared LUT crop is outside its geographic grid")
        expected_shape = (row1 - row0, col1 - col0)
        arrays = (self.az_full, self.rg_full, self.valid, self.height_full)
        if any(array.shape != expected_shape for array in arrays):
            reject_invalid_state("prepared LUT array shape does not match its crop")
        if (
            self.az_full.dtype != np.float64
            or self.rg_full.dtype != np.float64
            or self.height_full.dtype != np.float64
            or self.valid.dtype != np.bool_
        ):
            reject_invalid_state("prepared LUT arrays do not use canonical dtypes")
        if not isinstance(self.height_m, (float, int)) or not np.isfinite(
            self.height_m
        ):
            reject_invalid_state("prepared LUT mean height must be finite")
        for array in arrays:
            array.setflags(write=False)


@dataclass(frozen=True, slots=True)
class ResidualObservation:
    """One Ampcor or ESD observation with explicit validity."""

    observation_id: str
    kind: Literal["range", "azimuth"]
    status: ResidualStatus
    value: float | None
    units: str
    sign_convention: str
    estimator: str
    sample_count: int
    quality: float | None
    uncertainty: float | None
    source_fingerprint: str
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        """Validate residual status and observation provenance."""
        _require_text(self.observation_id, "observation_id")
        if self.kind not in ("range", "azimuth"):
            reject_invalid_state("residual observation kind must be range or azimuth")
        if not isinstance(self.status, ResidualStatus):
            reject_invalid_state("invalid residual observation status")
        if self.status is ResidualStatus.VALID and self.value is None:
            reject_invalid_state("valid residual observations require a value")
        if self.status is not ResidualStatus.VALID and self.value is not None:
            reject_invalid_state("invalid residual observations cannot carry a value")
        if self.sample_count < 0:
            reject_invalid_state("sample_count must be non-negative")
        for name in (
            "units",
            "sign_convention",
            "estimator",
            "source_fingerprint",
            "configuration_fingerprint",
        ):
            _require_text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class PairResidualSolution:
    """Common range/Ampcor and azimuth/ESD solution for one Pair."""

    solution_id: str
    range_observation_ids: tuple[str, ...]
    azimuth_observation_ids: tuple[str, ...]
    range_value: float | None
    azimuth_value: float | None
    status: ResidualStatus
    force_id: str
    configuration_fingerprint: str
    application_id: str | None = None

    def __post_init__(self) -> None:
        """Validate common Pair residual values and force identity."""
        _require_text(self.solution_id, "solution_id")
        _require_text(self.force_id, "force_id")
        _require_text(self.configuration_fingerprint, "configuration_fingerprint")
        if self.status is ResidualStatus.VALID and (
            self.range_value is None or self.azimuth_value is None
        ):
            reject_invalid_state("valid pair solutions require both residual values")
        if self.status is not ResidualStatus.VALID and (
            self.range_value is not None or self.azimuth_value is not None
        ):
            reject_invalid_state("invalid pair solutions cannot carry residual values")


@dataclass(frozen=True, slots=True)
class ResidualSolution:
    """Network or Pair residual lineage consumed by a prepared view."""

    solution_id: str
    kind: Literal["pair", "network"]
    parent_observation_ids: tuple[str, ...]
    payload_digest: str
    application_id: str
    status: ResidualStatus

    def __post_init__(self) -> None:
        """Validate solution lineage and application identity."""
        _require_text(self.solution_id, "solution_id")
        if self.kind not in ("pair", "network"):
            reject_invalid_state("solution kind must be pair or network")
        _require_digest(self.payload_digest, "payload_digest")
        _require_text(self.application_id, "application_id")
        if self.status is ResidualStatus.VALID and not self.parent_observation_ids:
            reject_invalid_state("valid solutions require observation lineage")


@dataclass(frozen=True, slots=True)
class AppliedSolution:
    """Exactly-once residual application boundary."""

    kind: Literal["pair", "network"]
    solution_payload_digest: str
    application_id: str
    domain: ArtifactDomain
    parent_observation_ids: tuple[str, ...]
    transition_digest: str

    def __post_init__(self) -> None:
        """Validate the exactly-once applied-solution transition."""
        if self.kind not in ("pair", "network"):
            reject_invalid_state("applied solution kind must be pair or network")
        _require_digest(self.solution_payload_digest, "solution_payload_digest")
        _require_text(self.application_id, "application_id")
        _require_digest(self.transition_digest, "transition_digest")


@dataclass(frozen=True, slots=True)
class PhaseTransitionRecord:
    """Auditable state transition for carrier, residual, or phase changes."""

    operation_id: str
    input_state_digest: str
    output_state_digest: str
    input_payload_digest: str
    output_payload_digest: str

    def __post_init__(self) -> None:
        """Validate exactly-once solution application lineage."""
        _require_text(self.operation_id, "operation_id")
        for name in (
            "input_state_digest",
            "output_state_digest",
            "input_payload_digest",
            "output_payload_digest",
        ):
            _require_digest(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class PhaseState:
    """Immutable phase state machine that rejects duplicate applications."""

    carrier: PhaseCarrier
    registration_model: Literal["reference_relative", "network_relative"]
    geometric_phase: GeometricPhase
    phase_model_id: str
    phase_lineage_id: str
    residual_solution_id: str | None
    residual_application_id: str | None
    transition_history: tuple[PhaseTransitionRecord, ...] = ()
    transition_digest: str = field(default="0" * 64)

    def __post_init__(self) -> None:
        """Validate phase state and residual lineage."""
        if self.registration_model not in (
            "reference_relative",
            "network_relative",
        ):
            reject_invalid_state("unsupported registration model")
        for name in ("phase_model_id", "phase_lineage_id"):
            _require_text(getattr(self, name), name)
        if self.residual_application_id and not self.residual_solution_id:
            reject_invalid_state("residual application requires a solution ID")
        _require_digest(self.transition_digest, "transition_digest")

    def apply_solution(
        self,
        solution: ResidualSolution,
        *,
        payload_digest: str,
        operation_id: str,
    ) -> PhaseState:
        """Apply one residual solution and reject duplicate or conflicting use."""
        _require_digest(payload_digest, "payload_digest")
        _require_text(operation_id, "operation_id")
        if self.residual_application_id is not None:
            if self.residual_solution_id == solution.solution_id:
                reject_invalid_state("residual solution was already applied")
            reject_invalid_state("a different residual solution is already applied")
        if solution.status is not ResidualStatus.VALID:
            reject_invalid_state("invalid residual solution cannot be applied")
        transition = PhaseTransitionRecord(
            operation_id=operation_id,
            input_state_digest=self.transition_digest,
            output_state_digest=_digest((self.phase_lineage_id, solution.solution_id)),
            input_payload_digest=payload_digest,
            output_payload_digest=solution.payload_digest,
        )
        history = (*self.transition_history, transition)
        return PhaseState(
            carrier=self.carrier,
            registration_model=self.registration_model,
            geometric_phase=self.geometric_phase,
            phase_model_id=self.phase_model_id,
            phase_lineage_id=self.phase_lineage_id,
            residual_solution_id=solution.solution_id,
            residual_application_id=solution.application_id,
            transition_history=history,
            transition_digest=_digest(
                [
                    {
                        "operation_id": item.operation_id,
                        "input_state_digest": item.input_state_digest,
                        "output_state_digest": item.output_state_digest,
                        "input_payload_digest": item.input_payload_digest,
                        "output_payload_digest": item.output_payload_digest,
                    }
                    for item in history
                ]
            ),
        )


__all__ = [
    "AppliedSolution",
    "ArtifactDomain",
    "GeometricPhase",
    "PairResidualSolution",
    "PhaseCarrier",
    "PhaseState",
    "PhaseTransitionRecord",
    "PreparedGeometryArrayPayload",
    "PreparedLutArrayPayload",
    "ResidualObservation",
    "ResidualSolution",
    "ResidualStatus",
    "geo_grid_identity",
]
