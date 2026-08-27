# ruff: noqa: D100, D103

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from faninsar.processing.contracts import (
    CoregistrationPolicy,
    PairResidualSolution,
    PhaseCarrier,
    PhaseState,
    PreparedIdentity,
    ResidualSolution,
    ResidualStatus,
    SceneViewRequest,
    SourceDescriptor,
    SourceRole,
    get_neutral_identity_projector,
)
from faninsar.processing.errors import InvalidProcessingStateError


def digest(value: object) -> str:
    """Return the test SHA-256 digest using the contract's JSON shape."""
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def source(role: SourceRole) -> SourceDescriptor:
    return SourceDescriptor(
        source_id=f"{role.value}-source",
        role=role,
        burst_key=("IW1", "0", "20240101T000000", "20240101T000001", "4x4"),
        source_manifest_digest=digest("source"),
        snapshot_device=1,
        snapshot_inode=2,
        shape=(4, 4),
        row_origin=10,
        col_origin=20,
        polarization="VV",
        annotation_digest=digest("annotation"),
        orbit_digest=digest("orbit"),
        calibration_digest=digest("calibration"),
    )


def test_source_roles_are_explicit_and_identity_rejects_bad_schema() -> None:
    assert source(SourceRole.REFERENCE).role is SourceRole.REFERENCE
    with pytest.raises(InvalidProcessingStateError):
        PreparedIdentity(
            provider_schema="prepared_geometry_provider.v0",  # type: ignore[arg-type]
            parent_generation_id="generation",
            domain="radar",
            ordered_source_ids=("reference", "secondary"),
            common_domain_digest=digest("common"),
            policy_digest=digest("policy"),
            schema_code_backend_device_digest=digest("runtime"),
            payload_manifest_digest=digest("payload"),
        )


def test_scene_view_projection_is_stable_and_cross_reference_is_checked() -> None:
    projector = get_neutral_identity_projector()
    fields = {
        "parent_id": "parent",
        "pair_id": "pair",
        "attempt_id": "attempt",
        "view_kind": "pair_window",
        "ordered_burst_keys": (source(SourceRole.REFERENCE).burst_key,),
        "normalized_roi": "roi-v1",
        "normalized_crs": "EPSG:32649",
        "crop_bounds": (0, 4, 0, 4),
        "halo": (1, 1, 1, 1),
        "output_origin": (0, 0),
        "output_shape": (4, 4),
        "residual_window_digest": digest("window"),
        "source_mask_identity": "mask-v1",
    }
    provisional = SceneViewRequest(view_id="0" * 64, **fields)
    view_id = digest(
        {
            "view_kind": provisional.view_kind,
            "parent_id": provisional.parent_id,
            "pair_id": provisional.pair_id,
            "attempt_id": provisional.attempt_id,
            "ordered_burst_keys": provisional.ordered_burst_keys,
            "normalized_roi": provisional.normalized_roi,
            "normalized_crs": provisional.normalized_crs,
            "crop_bounds": provisional.crop_bounds,
            "halo": provisional.halo,
            "output_origin": provisional.output_origin,
            "output_shape": provisional.output_shape,
            "residual_window_digest": provisional.residual_window_digest,
            "source_mask_identity": provisional.source_mask_identity,
            "artifact_generation_ids": (),
            "no_data_policy_id": None,
            "overlap_policy_id": None,
            "multilook_identity": None,
            "filter_identity": None,
        }
    )
    request = SceneViewRequest(view_id=view_id, **fields)
    assert projector.view_id(request) == view_id
    with pytest.raises(InvalidProcessingStateError):
        projector.view_id(SceneViewRequest(view_id="1" * 64, **fields))


def test_phase_state_rejects_duplicate_residual_application() -> None:
    solution = ResidualSolution(
        solution_id="solution",
        kind="pair",
        parent_observation_ids=("ampcor", "esd"),
        payload_digest=digest("solution"),
        application_id="application",
        status=ResidualStatus.VALID,
    )
    state = PhaseState(
        carrier=PhaseCarrier.DERAMPED,
        registration_model="reference_relative",
        geometric_phase="none",  # type: ignore[arg-type]
        phase_model_id="phase-model",
        phase_lineage_id="lineage",
        residual_solution_id=None,
        residual_application_id=None,
    )
    with pytest.raises(InvalidProcessingStateError, match="registration model"):
        replace(state, registration_model="master_relative")
    applied = state.apply_solution(
        solution,
        payload_digest=digest("input"),
        operation_id="apply-1",
    )
    assert applied.residual_solution_id == "solution"
    with pytest.raises(InvalidProcessingStateError):
        applied.apply_solution(
            solution,
            payload_digest=digest("input"),
            operation_id="apply-2",
        )


def test_valid_pair_solution_requires_both_range_and_azimuth() -> None:
    with pytest.raises(InvalidProcessingStateError):
        PairResidualSolution(
            solution_id="solution",
            range_observation_ids=("ampcor",),
            azimuth_observation_ids=("esd",),
            range_value=0.0,
            azimuth_value=None,
            status=ResidualStatus.VALID,
            force_id="force",
            configuration_fingerprint="config",
        )


def test_policy_keeps_measurement_domain_on_radar() -> None:
    policy = CoregistrationPolicy(
        artifact_domain="geo",
        measurement_domain="radar",
        residual_policy="geometry_only",
        control_spacing=8,
        solver_id="geo2rdr-v1",
        tolerance_id="default",
        interpolation_id="rgi-linear-v1",
        kernel_id="lanczos-a4",
        dtype_id="float64-controls-float32-dense",
    )
    assert policy.artifact_domain == "geo"
