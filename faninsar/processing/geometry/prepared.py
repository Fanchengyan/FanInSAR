"""Neutral contracts for exact prepared geometry reuse.

The Pair, geo, and Stack orchestration layers exchange these immutable records
instead of importing each other's mutable workflow state.  The module is the
single owner of the ``prepared_geometry_provider.v1`` schema and of the V/K/I
identity projections used by Stack result manifests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np

from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from faninsar.processing.mosaicking.grid import GeoGridSpec

PROVIDER_SCHEMA = "prepared_geometry_provider.v1"
ViewKind = Literal["pair_window", "stack_scene_view"]
ArtifactDomain = Literal["radar", "geo"]
ResidualPolicy = Literal["measure_pair", "consume_solution", "geometry_only"]


class SourceRole(StrEnum):
    """Allowed ordered roles in a prepared source pair."""

    REFERENCE = "reference"
    SECONDARY = "secondary"


class ViewState(StrEnum):
    """Lifecycle state of a materialized view."""

    FINAL = "FINAL"
    DISCARDED = "DISCARDED"


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


def _require_nonnegative(value: int, name: str) -> None:
    if not isinstance(value, int) or value < 0:
        reject_invalid_state(f"{name} must be a non-negative integer")


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
class SourceDescriptor:
    """Immutable, role-bearing source snapshot metadata.

    Parameters
    ----------
    source_id
        Stable source descriptor identifier.
    role
        Explicit reference or secondary role.  Role is part of the identity.
    burst_key
        Canonical structural burst key shared by Stack and Pair.
    source_manifest_digest
        Digest of the immutable source snapshot and all auxiliary inputs.
    snapshot_device, snapshot_inode
        Device and inode of the opened immutable snapshot root.
    shape
        Native ``(rows, columns)`` shape.
    row_origin, col_origin
        Native global pixel origin.
    polarization
        Polarization identity; it is never inferred from a product alias.
    annotation_digest, orbit_digest, calibration_digest
        Digests of the parsed source auxiliaries.

    """

    source_id: str
    role: SourceRole
    burst_key: tuple[str, ...]
    source_manifest_digest: str
    snapshot_device: int
    snapshot_inode: int
    shape: tuple[int, int]
    row_origin: int
    col_origin: int
    polarization: str
    annotation_digest: str
    orbit_digest: str
    calibration_digest: str

    def __post_init__(self) -> None:
        """Validate immutable source identity fields."""
        _require_text(self.source_id, "source_id")
        if not isinstance(self.role, SourceRole):
            reject_invalid_state("source role must be reference or secondary")
        if not self.burst_key or any(not item for item in self.burst_key):
            reject_invalid_state("burst_key must contain non-empty fields")
        _require_digest(self.source_manifest_digest, "source_manifest_digest")
        _require_digest(self.annotation_digest, "annotation_digest")
        _require_digest(self.orbit_digest, "orbit_digest")
        _require_digest(self.calibration_digest, "calibration_digest")
        _require_nonnegative(self.snapshot_device, "snapshot_device")
        _require_nonnegative(self.snapshot_inode, "snapshot_inode")
        if len(self.shape) != 2 or any(size <= 0 for size in self.shape):
            reject_invalid_state("source shape must contain two positive dimensions")
        _require_text(self.polarization, "polarization")


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Authenticated finite resource budget for one preparation run."""

    profile_id: str
    profile_authority_id: str
    profile_digest: str
    administrator_ceiling_digest: str
    cgroup_id: str
    limit_digest: str
    max_dimensions: int
    max_files: int
    max_chunks: int
    max_encoded_bytes: int
    max_decoded_bytes: int
    max_temporary_bytes: int
    max_codec_expansion: int
    max_manifest_bytes: int
    max_workers: int
    max_processes: int
    disk_reserve_bytes: int
    max_wall_time_seconds: int
    max_rss_bytes: int
    max_device_bytes: int

    def __post_init__(self) -> None:
        """Validate positive finite resource limits and profile digests."""
        for name in (
            "profile_id",
            "profile_authority_id",
            "cgroup_id",
        ):
            _require_text(getattr(self, name), name)
        for name in (
            "profile_digest",
            "administrator_ceiling_digest",
            "limit_digest",
        ):
            _require_digest(getattr(self, name), name)
        for name in (
            "max_dimensions",
            "max_files",
            "max_chunks",
            "max_encoded_bytes",
            "max_decoded_bytes",
            "max_temporary_bytes",
            "max_codec_expansion",
            "max_manifest_bytes",
            "max_workers",
            "max_processes",
            "disk_reserve_bytes",
            "max_wall_time_seconds",
            "max_rss_bytes",
            "max_device_bytes",
        ):
            value = getattr(self, name)
            if value <= 0:
                reject_invalid_state(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class CoregistrationPolicy:
    """Numerical and domain policy shared by Pair, geo, and Stack."""

    artifact_domain: ArtifactDomain
    measurement_domain: Literal["radar"]
    residual_policy: ResidualPolicy
    control_spacing: int
    solver_id: str
    tolerance_id: str
    interpolation_id: str
    kernel_id: str
    dtype_id: str

    def __post_init__(self) -> None:
        """Validate the domain and unchanged numerical policy identifiers."""
        if self.artifact_domain not in ("radar", "geo"):
            reject_invalid_state("artifact_domain must be radar or geo")
        if self.measurement_domain != "radar":
            reject_invalid_state("measurement_domain is fixed to radar in v1")
        if self.residual_policy not in (
            "measure_pair",
            "consume_solution",
            "geometry_only",
        ):
            reject_invalid_state("unsupported residual policy")
        if self.control_spacing <= 0:
            reject_invalid_state("control_spacing must be positive")
        for name in (
            "solver_id",
            "tolerance_id",
            "interpolation_id",
            "kernel_id",
            "dtype_id",
        ):
            _require_text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class PreparedIdentity:
    """Canonical identity for one immutable prepared parent generation."""

    provider_schema: Literal["prepared_geometry_provider.v1"]
    parent_generation_id: str
    domain: ArtifactDomain
    ordered_source_ids: tuple[str, str]
    common_domain_digest: str
    policy_digest: str
    schema_code_backend_device_digest: str
    payload_manifest_digest: str

    def __post_init__(self) -> None:
        """Validate parent identity schema and payload digests."""
        if self.provider_schema != PROVIDER_SCHEMA:
            reject_invalid_state("unsupported prepared geometry provider schema")
        _require_text(self.parent_generation_id, "parent_generation_id")
        if self.domain not in ("radar", "geo"):
            reject_invalid_state("identity domain must be radar or geo")
        if len(self.ordered_source_ids) != 2 or any(
            not source_id for source_id in self.ordered_source_ids
        ):
            reject_invalid_state("identity must contain reference and secondary IDs")
        for name in (
            "common_domain_digest",
            "policy_digest",
            "schema_code_backend_device_digest",
            "payload_manifest_digest",
        ):
            _require_digest(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class SceneViewRequest:
    """Exact child view request over a canonical prepared parent."""

    view_id: str
    parent_id: str
    pair_id: str
    attempt_id: str
    view_kind: ViewKind
    ordered_burst_keys: tuple[tuple[str, ...], ...]
    normalized_roi: str
    normalized_crs: str
    crop_bounds: tuple[int, int, int, int]
    halo: tuple[int, int, int, int]
    output_origin: tuple[int, int]
    output_shape: tuple[int, int]
    residual_window_digest: str
    source_mask_identity: str
    artifact_generation_ids: tuple[str, ...] = ()
    ifg_request_identity: str | None = None
    pair_result_key_identity: str | None = None
    no_data_policy_id: str | None = None
    overlap_policy_id: str | None = None
    multilook_identity: str | None = None
    filter_identity: str | None = None

    def __post_init__(self) -> None:
        """Validate discriminated Pair/Stack view fields."""
        _require_digest(self.view_id, "view_id")
        _require_text(self.parent_id, "parent_id")
        _require_text(self.pair_id, "pair_id")
        _require_text(self.attempt_id, "attempt_id")
        if self.view_kind not in ("pair_window", "stack_scene_view"):
            reject_invalid_state("unsupported scene view kind")
        if not self.ordered_burst_keys:
            reject_invalid_state("scene view needs at least one burst key")
        if len(self.crop_bounds) != 4 or any(value < 0 for value in self.crop_bounds):
            reject_invalid_state("crop bounds must be four non-negative values")
        if len(self.halo) != 4 or any(value < 0 for value in self.halo):
            reject_invalid_state("halo must be four non-negative values")
        if len(self.output_origin) != 2 or len(self.output_shape) != 2:
            reject_invalid_state("view origin and shape must be two-dimensional")
        if any(value <= 0 for value in self.output_shape):
            reject_invalid_state("view shape must be positive")
        _require_digest(self.residual_window_digest, "residual_window_digest")
        _require_text(self.source_mask_identity, "source_mask_identity")
        is_stack = self.view_kind == "stack_scene_view"
        if is_stack and (
            not self.artifact_generation_ids
            or not self.no_data_policy_id
            or not self.overlap_policy_id
            or not self.multilook_identity
            or not self.filter_identity
        ):
            reject_invalid_state("stack scene views require Stack child identities")
        if not is_stack and self.artifact_generation_ids:
            reject_invalid_state("pair views cannot carry Stack artifact IDs")


@dataclass(frozen=True, slots=True)
class PreparedSceneHandle:
    """Opaque parent-generation handle returned by ``prepare_scene``."""

    handle_id: str
    provider_parent_generation_id: str
    identity: PreparedIdentity
    expected_burst_manifest_digest: str
    owner_capability_digest: str


@dataclass(frozen=True, slots=True)
class PreparedGeometryHandle:
    """Read-only handle for raw and canonical post-fill controls."""

    handle_id: str
    parent_id: str
    burst_key: tuple[str, ...]
    ordered_source_roles: tuple[SourceRole, SourceRole]
    raw_payload_digest: str
    post_fill_payload_digest: str
    validity_mask_digest: str
    read_only_capability_digest: str


@dataclass(frozen=True, slots=True)
class PreparedGeometryArrayPayload:
    """Validated read-only dense controls loaded from a provider generation."""

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
class PreparedLutHandle:
    """Read-only handle for an exact geo LUT child artifact."""

    handle_id: str
    parent_id: str
    view_id: str
    geo_grid_identity: str
    payload_digest: str
    read_only_capability_digest: str


@dataclass(frozen=True, slots=True)
class PreparedViewHandle:
    """Read-only handle for a materialized Pair or Stack view."""

    handle_id: str
    request: SceneViewRequest
    state: ViewState
    payload_digest: str
    read_only_capability_digest: str


@dataclass(frozen=True, slots=True)
class ProviderLeaseToken:
    """Capability-bound generation lease passed to every provider method."""

    owner_nonce: str
    worker_nonce: str
    parent_generation_id: str
    generation_root_device: int
    generation_root_inode: int
    source_snapshot_identities: tuple[tuple[int, int], ...]
    owner_uid: int
    host_boot_id: str
    pid_start: str
    expiry_epoch_seconds: int
    heartbeat_epoch_seconds: int
    capability_digest: str
    resource_digest: str
    state: Literal["PINNED", "ABORTED", "CLOSED"]

    def __post_init__(self) -> None:
        """Validate lease capability, inode, and lifecycle fields."""
        for name in (
            "owner_nonce",
            "worker_nonce",
            "parent_generation_id",
            "host_boot_id",
            "pid_start",
        ):
            _require_text(getattr(self, name), name)
        for name in ("capability_digest", "resource_digest"):
            _require_digest(getattr(self, name), name)
        if self.state not in ("PINNED", "ABORTED", "CLOSED"):
            reject_invalid_state("unsupported provider lease state")
        if self.expiry_epoch_seconds < self.heartbeat_epoch_seconds:
            reject_invalid_state("lease expiry cannot precede heartbeat")


@dataclass(frozen=True, slots=True)
class WorkerAttestation:
    """Pre-import worker identity and resource-cap attestation."""

    token_digest: str
    provider_schema: Literal["prepared_geometry_provider.v1"]
    resolved_backend: str
    physical_device_identity: str
    visible_device_mapping: str
    numerical_environment_digest: str
    bootstrap_code_digest: str
    monitor_lease_digest: str
    pid_start: str
    limits_digest: str

    def __post_init__(self) -> None:
        """Validate the pre-import worker attestation fields."""
        for name in (
            "token_digest",
            "numerical_environment_digest",
            "bootstrap_code_digest",
            "monitor_lease_digest",
            "limits_digest",
        ):
            _require_digest(getattr(self, name), name)
        if self.provider_schema != PROVIDER_SCHEMA:
            reject_invalid_state("worker provider schema mismatch")
        for name in (
            "resolved_backend",
            "physical_device_identity",
            "visible_device_mapping",
            "pid_start",
        ):
            _require_text(getattr(self, name), name)


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
    """Exactly-once residual application boundary shared by P18 and P19."""

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


@dataclass(frozen=True, slots=True)
class ProviderQualificationReceipt:
    """P18 receipt consumed by P19 correctness and activation gates."""

    provider_schema: Literal["prepared_geometry_provider.v1"]
    provider_contract_digest: str
    provider_parent_generation_id: str
    prepared_identity_digest: str
    expected_unit_manifest_digest: str
    source_snapshot_digest: str
    limits_profile_digest: str
    p18_acceptance_event_id: str
    p18_verification_event_id: str

    def __post_init__(self) -> None:
        """Validate the immutable P18 qualification receipt."""
        if self.provider_schema != PROVIDER_SCHEMA:
            reject_invalid_state("receipt provider schema mismatch")
        for name in (
            "provider_contract_digest",
            "prepared_identity_digest",
            "expected_unit_manifest_digest",
            "source_snapshot_digest",
            "limits_profile_digest",
        ):
            _require_digest(getattr(self, name), name)
        for name in (
            "provider_parent_generation_id",
            "p18_acceptance_event_id",
            "p18_verification_event_id",
        ):
            _require_text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ActivationToken:
    """Post-publication capability for a P19 ``scene_artifact_v1`` run.

    The coordinator-issued signature is represented by ``issuer_record_digest``
    in this product layer.  Stack entry points still validate the complete
    subject, parent, event, namespace, and fence tuple before reading any
    scene artifact; a copied binding without the matching token is rejected.
    """

    intent_id: str
    parent_id: str
    parent_manifest_digest: str
    root_device: int
    root_inode: int
    namespace: str
    mode: Literal["qualified", "reference"]
    domain: ArtifactDomain
    policy_identity: str
    code_identity: str
    schema: str
    provider_receipt_digest: str
    threshold_configuration_hash: str
    qualification_evidence_digest: str
    p19_qualified_event_ids: tuple[str, ...]
    p18_stack_gate_event_id: str | None
    fence_epoch: int
    issuer_record_digest: str

    def __post_init__(self) -> None:
        """Validate the immutable activation subject and event lineage."""
        for name in (
            "intent_id",
            "parent_id",
            "namespace",
            "policy_identity",
            "code_identity",
            "schema",
        ):
            _require_text(getattr(self, name), name)
        for name in (
            "parent_manifest_digest",
            "provider_receipt_digest",
            "threshold_configuration_hash",
            "qualification_evidence_digest",
            "issuer_record_digest",
        ):
            _require_digest(getattr(self, name), name)
        if self.mode not in ("qualified", "reference"):
            reject_invalid_state("unsupported activation token mode")
        if self.domain not in ("radar", "geo"):
            reject_invalid_state("unsupported activation token domain")
        if self.root_device < 0 or self.root_inode < 0:
            reject_invalid_state("activation token root identity is invalid")
        _require_nonnegative(self.fence_epoch, "fence_epoch")
        if any(not event_id for event_id in self.p19_qualified_event_ids):
            reject_invalid_state("activation event IDs must be non-empty")
        if self.mode == "qualified":
            if not self.p19_qualified_event_ids or not self.p18_stack_gate_event_id:
                reject_invalid_state(
                    "qualified activation requires P19 and P18 gate events"
                )
        elif self.p18_stack_gate_event_id is not None:
            reject_invalid_state("reference activation cannot carry P18 Stack gate")

    def digest(self) -> str:
        """Return the canonical digest bound into the activation record."""
        return _digest(
            {
                "intent_id": self.intent_id,
                "parent_id": self.parent_id,
                "parent_manifest_digest": self.parent_manifest_digest,
                "root_device": self.root_device,
                "root_inode": self.root_inode,
                "namespace": self.namespace,
                "mode": self.mode,
                "domain": self.domain,
                "policy_identity": self.policy_identity,
                "code_identity": self.code_identity,
                "schema": self.schema,
                "provider_receipt_digest": self.provider_receipt_digest,
                "threshold_configuration_hash": self.threshold_configuration_hash,
                "qualification_evidence_digest": self.qualification_evidence_digest,
                "p19_qualified_event_ids": self.p19_qualified_event_ids,
                "p18_stack_gate_event_id": self.p18_stack_gate_event_id,
                "fence_epoch": self.fence_epoch,
                "issuer_record_digest": self.issuer_record_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class StackActivationBinding:
    """Acyclic P18/P19 activation binding."""

    provider_parent_generation_id: str
    qualification_receipt_digest: str
    p19_correctness_event_id: str
    activation_mode: Literal["reference", "qualified"]
    p19_qualified_event_id: str | None
    p18_stack_gate_event_id: str | None
    stack_generation_id: str
    p19_qualified_event_ids: tuple[str, ...] = ()
    activation_token_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate the acyclic P18/P19 activation binding."""
        _require_text(
            self.provider_parent_generation_id, "provider_parent_generation_id"
        )
        _require_digest(
            self.qualification_receipt_digest, "qualification_receipt_digest"
        )
        _require_text(self.p19_correctness_event_id, "p19_correctness_event_id")
        _require_text(self.stack_generation_id, "stack_generation_id")
        if self.activation_mode not in ("reference", "qualified"):
            reject_invalid_state("unsupported Stack activation mode")
        if self.activation_mode == "qualified" and (
            not self.p19_qualified_event_id
            or not self.p18_stack_gate_event_id
            or not self.p19_qualified_event_ids
            or self.p19_qualified_event_id not in self.p19_qualified_event_ids
            or self.activation_token_digest is None
        ):
            reject_invalid_state(
                "qualified Stack activation requires typed token and gate events"
            )
        if self.activation_token_digest is not None:
            _require_digest(self.activation_token_digest, "activation_token_digest")


@runtime_checkable
class PreparedGeometryProvider(Protocol):
    """Versioned provider consumed by Pair, geo, and Stack orchestration."""

    def prepare_scene(
        self,
        *,
        reference: SourceDescriptor,
        secondary: SourceDescriptor,
        policy: CoregistrationPolicy,
        geo_grid: GeoGridSpec | None,
        limits: ResourceLimits,
        operation_id: str,
    ) -> PreparedSceneHandle:
        """Prepare one immutable common-domain parent generation."""
        ...

    def provider_schema(self) -> Literal["prepared_geometry_provider.v1"]:
        """Return the exact neutral provider schema identifier."""
        ...

    def bootstrap_worker(
        self, token: ProviderLeaseToken, limits: ResourceLimits
    ) -> WorkerAttestation:
        """Attest caps and resolved device before numerical imports."""
        ...

    def open_prepared_geometry(
        self, handle_id: str, token: ProviderLeaseToken
    ) -> PreparedGeometryHandle:
        """Open read-only raw and post-fill control payloads."""
        ...

    def read_prepared_geometry(
        self,
        geometry_handle: PreparedGeometryHandle,
        token: ProviderLeaseToken,
    ) -> PreparedGeometryArrayPayload:
        """Read validated primitive dense controls from a pinned generation."""
        ...

    def materialize_view(
        self,
        handle_id: str,
        request: SceneViewRequest,
        token: ProviderLeaseToken,
    ) -> PreparedViewHandle:
        """Materialize one exact child view without solving or refilling."""
        ...

    def open_prepared_lut(
        self,
        handle_id: str,
        request: SceneViewRequest,
        geo_grid: GeoGridSpec,
        token: ProviderLeaseToken,
    ) -> PreparedLutHandle:
        """Open one identity-matched read-only geo LUT view."""
        ...

    def read_prepared_lut(
        self,
        lut_handle: PreparedLutHandle,
        token: ProviderLeaseToken,
    ) -> PreparedLutArrayPayload:
        """Read validated primitive arrays from one prepared geo LUT."""
        ...

    def identity(self, handle_id: str, token: ProviderLeaseToken) -> PreparedIdentity:
        """Return the validated parent identity for a prepared handle."""
        ...

    def pin_generation(
        self, parent_generation_id: str, token: ProviderLeaseToken
    ) -> None:
        """Pin a generation before any payload is opened."""
        ...

    def unpin_generation(
        self, parent_generation_id: str, token: ProviderLeaseToken
    ) -> None:
        """Release one worker's generation pin."""
        ...

    def abort_generation(
        self, parent_generation_id: str, token: ProviderLeaseToken
    ) -> None:
        """Abort and quarantine an incomplete generation."""
        ...

    def close_generation(
        self, parent_generation_id: str, token: ProviderLeaseToken
    ) -> None:
        """Close a generation after all worker pins are released."""
        ...


class NeutralIdentityProjector(Protocol):
    """Identity projection interface shared by P18 and P19."""

    def view_id(self, request: SceneViewRequest) -> str:
        """Derive and validate V, the canonical view identity."""
        ...

    def pair_result_key_identity(
        self,
        pair_id: str,
        multilook_identity: str,
        view_request_identity: str,
        filter_identity: str,
        output_format_identity: str,
        output_identity: str,
    ) -> str:
        """Derive K, the canonical Pair result identity."""
        ...

    def ifg_request_identity(
        self,
        parent_id: str,
        ordered_pair_ids: tuple[str, ...],
        ordered_multilook_identities: tuple[str, ...],
        filter_identity: str,
        output_format_identity: str,
        domain: ArtifactDomain,
        grid_identity: str,
        ordered_view_identities: tuple[str, ...],
        ordered_pair_result_identities: tuple[str, ...],
        time_series_requested: bool,
    ) -> str:
        """Derive I, the canonical Stack interferogram request identity."""
        ...


class _IdentityProjector:
    """Default deterministic implementation of the neutral projections."""

    @staticmethod
    def view_id(request: SceneViewRequest) -> str:
        fields = {
            "view_kind": request.view_kind,
            "parent_id": request.parent_id,
            "pair_id": request.pair_id,
            "attempt_id": request.attempt_id,
            "ordered_burst_keys": request.ordered_burst_keys,
            "normalized_roi": request.normalized_roi,
            "normalized_crs": request.normalized_crs,
            "crop_bounds": request.crop_bounds,
            "halo": request.halo,
            "output_origin": request.output_origin,
            "output_shape": request.output_shape,
            "residual_window_digest": request.residual_window_digest,
            "source_mask_identity": request.source_mask_identity,
            "artifact_generation_ids": request.artifact_generation_ids,
            "no_data_policy_id": request.no_data_policy_id,
            "overlap_policy_id": request.overlap_policy_id,
            "multilook_identity": request.multilook_identity,
            "filter_identity": request.filter_identity,
        }
        result = _digest(fields)
        if request.view_id != result:
            reject_invalid_state("SceneViewRequest.view_id does not match its fields")
        return result

    @staticmethod
    def pair_result_key_identity(
        pair_id: str,
        multilook_identity: str,
        view_request_identity: str,
        filter_identity: str,
        output_format_identity: str,
        output_identity: str,
    ) -> str:
        return _digest(
            {
                "pair_id": pair_id,
                "multilook_identity": multilook_identity,
                "view_request_identity": view_request_identity,
                "filter_identity": filter_identity,
                "output_format_identity": output_format_identity,
                "output_identity": output_identity,
            }
        )

    @staticmethod
    def ifg_request_identity(
        parent_id: str,
        ordered_pair_ids: tuple[str, ...],
        ordered_multilook_identities: tuple[str, ...],
        filter_identity: str,
        output_format_identity: str,
        domain: ArtifactDomain,
        grid_identity: str,
        ordered_view_identities: tuple[str, ...],
        ordered_pair_result_identities: tuple[str, ...],
        time_series_requested: bool,
    ) -> str:
        return _digest(
            {
                "parent_id": parent_id,
                "ordered_pair_ids": ordered_pair_ids,
                "ordered_multilook_identities": ordered_multilook_identities,
                "filter_identity": filter_identity,
                "output_format_identity": output_format_identity,
                "domain": domain,
                "grid_identity": grid_identity,
                "ordered_view_identities": ordered_view_identities,
                "ordered_pair_result_identities": ordered_pair_result_identities,
                "time_series_requested": time_series_requested,
            }
        )


def get_neutral_identity_projector() -> NeutralIdentityProjector:
    """Return the sole V/K/I identity projector for P18 and P19."""
    return _IdentityProjector()


__all__ = [
    "PROVIDER_SCHEMA",
    "ActivationToken",
    "AppliedSolution",
    "ArtifactDomain",
    "CoregistrationPolicy",
    "GeometricPhase",
    "NeutralIdentityProjector",
    "PairResidualSolution",
    "PhaseCarrier",
    "PhaseState",
    "PhaseTransitionRecord",
    "PreparedGeometryArrayPayload",
    "PreparedGeometryHandle",
    "PreparedGeometryProvider",
    "PreparedIdentity",
    "PreparedLutArrayPayload",
    "PreparedLutHandle",
    "PreparedSceneHandle",
    "PreparedViewHandle",
    "ProviderLeaseToken",
    "ProviderQualificationReceipt",
    "ResidualObservation",
    "ResidualSolution",
    "ResidualStatus",
    "ResourceLimits",
    "SceneViewRequest",
    "SourceDescriptor",
    "SourceRole",
    "StackActivationBinding",
    "ViewKind",
    "ViewState",
    "WorkerAttestation",
    "geo_grid_identity",
    "get_neutral_identity_projector",
]
