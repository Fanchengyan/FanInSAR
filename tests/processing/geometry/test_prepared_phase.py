"""Public PhaseState residual-application contract (PROPOSAL-0082)."""

from __future__ import annotations

import hashlib
import inspect

import pytest
from affine import Affine

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.grids import GridSpec
from faninsar.processing.geometry.prepared import (
    GeometricPhase,
    PhaseCarrier,
    PhaseState,
    PhaseTransitionRecord,
    ResidualSolution,
    ResidualStatus,
    geo_grid_identity,
)

_LEGACY_DIGEST = "ab" * 32
_EMPTY_HISTORY_DIGEST = "0" * 64


def _solution(
    *,
    solution_id: str = "sol-1",
    application_id: str = "app-1",
    payload_digest: str | None = None,
) -> ResidualSolution:
    return ResidualSolution(
        solution_id=solution_id,
        kind="pair",
        parent_observation_ids=("obs-1",),
        payload_digest=payload_digest,
        application_id=application_id,
        status=ResidualStatus.VALID,
    )


def _phase_state() -> PhaseState:
    return PhaseState(
        carrier=PhaseCarrier.DERAMPED,
        registration_model="network_relative",
        geometric_phase=GeometricPhase.NONE,
        phase_model_id="p0082.operation-lineage.v1",
        phase_lineage_id="pair-1",
        residual_solution_id=None,
        residual_application_id=None,
    )


def test_apply_solution_accepts_optional_legacy_payload_digest_kwarg() -> None:
    """NISAR may still pass the old digest keyword; it is stored inertly."""
    parameters = inspect.signature(PhaseState.apply_solution).parameters
    assert "payload_digest" in parameters
    assert parameters["payload_digest"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["payload_digest"].default is None

    applied = _phase_state().apply_solution(
        _solution(payload_digest=_LEGACY_DIGEST),
        payload_digest=_LEGACY_DIGEST,
        operation_id="apply-carrier",
    )
    assert applied.residual_solution_id == "sol-1"
    assert applied.residual_application_id == "app-1"
    history = applied.transition_history
    assert len(history) == 1
    assert history[0].operation_id == "apply-carrier"
    assert history[0].input_payload_digest == _LEGACY_DIGEST
    assert history[0].output_payload_digest == _LEGACY_DIGEST


def test_apply_solution_does_not_require_a_payload_digest() -> None:
    """New S1 applications identify the solution, not a payload checksum."""
    applied = _phase_state().apply_solution(
        _solution(),
        operation_id="apply-carrier",
    )
    assert applied.residual_solution_id == "sol-1"
    history = applied.transition_history[0]
    assert history.input_payload_digest is None
    assert history.output_payload_digest is None


def test_apply_solution_rejects_duplicate_use() -> None:
    """Exactly-once residual application remains a public PhaseState guard."""
    solution = _solution()
    applied = _phase_state().apply_solution(solution, operation_id="first")
    with pytest.raises(InvalidProcessingStateError, match="already applied"):
        applied.apply_solution(solution, operation_id="again")


def test_apply_solution_rejects_a_conflicting_second_solution() -> None:
    """A different residual cannot replace one that is already applied."""
    applied = _phase_state().apply_solution(_solution(), operation_id="first")
    other = _solution(solution_id="sol-2", application_id="app-2")
    with pytest.raises(InvalidProcessingStateError, match="already applied"):
        applied.apply_solution(other, operation_id="second")


def test_legacy_phase_records_are_accepted_inertly() -> None:
    """Stored payload digest fields remain readable and are not rehashed."""
    record = PhaseTransitionRecord(
        operation_id="legacy-op",
        input_state_digest=_EMPTY_HISTORY_DIGEST,
        output_state_digest=hashlib.sha256(b"legacy-out").hexdigest(),
        input_payload_digest=_LEGACY_DIGEST,
        output_payload_digest=_LEGACY_DIGEST,
    )
    solution = ResidualSolution(
        solution_id="legacy-sol",
        kind="network",
        parent_observation_ids=("legacy-obs",),
        payload_digest=_LEGACY_DIGEST,
        application_id="legacy-app",
        status=ResidualStatus.VALID,
    )
    state = PhaseState(
        carrier=PhaseCarrier.DERAMPED,
        registration_model="reference_relative",
        geometric_phase=GeometricPhase.FLAT_REMOVED,
        phase_model_id="p18-p19.scientific-lineage.v1",
        phase_lineage_id="legacy-pair",
        residual_solution_id="legacy-sol",
        residual_application_id="legacy-app",
        transition_history=(record,),
        transition_digest=hashlib.sha256(b"legacy-history").hexdigest(),
    )
    assert record.input_payload_digest == _LEGACY_DIGEST
    assert solution.payload_digest == _LEGACY_DIGEST
    assert state.transition_history[0] is record


def test_geo_grid_identity_remains_a_metadata_hash() -> None:
    """Native qualification identity hashes are not part of this deletion."""
    grid = GridSpec(
        crs="EPSG:4326",
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 12.0),
        width=4,
        height=3,
    )
    identity = geo_grid_identity(grid)
    assert len(identity) == 64
    assert identity == geo_grid_identity(grid)
    shifted = GridSpec(
        crs="EPSG:4326",
        transform=Affine(1.0, 0.0, 1.0, 0.0, -1.0, 12.0),
        width=4,
        height=3,
    )
    assert geo_grid_identity(shifted) != identity
