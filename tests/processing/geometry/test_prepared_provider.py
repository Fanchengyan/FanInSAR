"""Tests for the concrete local prepared geometry provider."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.processing.contracts import (
    CoregistrationPolicy,
    PreparedIdentity,
    ResourceLimits,
    SceneViewRequest,
    SourceDescriptor,
    SourceRole,
    geo_grid_identity,
)
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.prepared_provider import (
    LocalPreparedGeometryProvider,
    PreparedGeometryArrayPayload,
    PreparedLutArrayPayload,
    PreparedScenePayload,
)

if TYPE_CHECKING:
    from pathlib import Path


def _digest(value: object) -> str:
    """Return a canonical test digest."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _source(role: SourceRole) -> SourceDescriptor:
    """Return a small role-bearing immutable source descriptor."""
    return SourceDescriptor(
        source_id=f"{role.value}-source",
        role=role,
        burst_key=("IW1", "0", "20240101T000000", "20240101T000001", "4x4"),
        source_manifest_digest=_digest(role.value),
        snapshot_device=1,
        snapshot_inode=2 if role is SourceRole.REFERENCE else 3,
        shape=(4, 4),
        row_origin=0,
        col_origin=0,
        polarization="VV",
        annotation_digest=_digest("annotation"),
        orbit_digest=_digest("orbit"),
        calibration_digest=_digest("calibration"),
    )


def _limits() -> ResourceLimits:
    """Return a valid bounded provider resource profile."""
    return ResourceLimits(
        profile_id="test-profile",
        profile_authority_id="test-authority",
        profile_digest=_digest("profile"),
        administrator_ceiling_digest=_digest("ceiling"),
        cgroup_id="test-cgroup",
        limit_digest=_digest("limits"),
        max_dimensions=2,
        max_files=16,
        max_chunks=16,
        max_encoded_bytes=1024,
        max_decoded_bytes=1024,
        max_temporary_bytes=1024,
        max_codec_expansion=4,
        max_manifest_bytes=1024,
        max_workers=1,
        max_processes=2,
        disk_reserve_bytes=1,
        max_wall_time_seconds=60,
        max_rss_bytes=1024 * 1024 * 1024,
        max_device_bytes=1024,
    )


def _policy() -> CoregistrationPolicy:
    """Return a policy that keeps the numerical interpolation identity fixed."""
    return CoregistrationPolicy(
        artifact_domain="radar",
        measurement_domain="radar",
        residual_policy="geometry_only",
        control_spacing=8,
        solver_id="test-solver-v1",
        tolerance_id="test-tolerance-v1",
        interpolation_id="rgi-linear-v1",
        kernel_id="lanczos-a4",
        dtype_id="float64-controls-float32-dense",
    )


def test_local_provider_publishes_once_and_requires_a_live_lease(
    tmp_path: Path,
) -> None:
    """The callback runs once and all reads are bound to the active lease."""
    payloads = {
        "raw.bin": b"raw-controls",
        "post.bin": b"post-fill-controls",
        "mask.bin": b"validity-mask",
    }
    payload_manifest = [
        {
            "name": name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(payloads.items())
    ]
    callback_calls = 0

    def prepare(**_: object) -> PreparedScenePayload:
        nonlocal callback_calls
        callback_calls += 1
        return PreparedScenePayload(
            identity=PreparedIdentity(
                provider_schema="prepared_geometry_provider.v1",
                parent_generation_id="generation-1",
                domain="radar",
                ordered_source_ids=("reference-source", "secondary-source"),
                common_domain_digest=_digest("common"),
                policy_digest=_digest("policy"),
                schema_code_backend_device_digest=_digest("runtime"),
                payload_manifest_digest=_digest(payload_manifest),
            ),
            metadata={
                "expected_burst_manifest": [
                    ["IW1", "0", "20240101T000000", "20240101T000001", "4x4"]
                ],
                "geometry": {
                    "burst-0": {
                        "burst_key": [
                            "IW1",
                            "0",
                            "20240101T000000",
                            "20240101T000001",
                            "4x4",
                        ],
                        "raw_payload": "raw.bin",
                        "post_fill_payload": "post.bin",
                        "validity_payload": "mask.bin",
                    }
                },
            },
            payloads=payloads,
        )

    provider = LocalPreparedGeometryProvider(tmp_path / "prepared", prepare)
    handle = provider.prepare_scene(
        reference=_source(SourceRole.REFERENCE),
        secondary=_source(SourceRole.SECONDARY),
        policy=_policy(),
        geo_grid=None,
        limits=_limits(),
        operation_id="prepare-1",
    )
    assert callback_calls == 1
    token = provider.issue_lease(handle, _limits())
    attestation = provider.bootstrap_worker(token, _limits())
    assert attestation.provider_schema == "prepared_geometry_provider.v1"
    geometry = provider.open_prepared_geometry(handle.handle_id, token)
    assert geometry.burst_key[0] == "IW1"
    assert provider.read_payload(handle.handle_id, "raw.bin", token) == b"raw-controls"
    provider.unpin_generation(handle.provider_parent_generation_id, token)
    with pytest.raises(InvalidProcessingStateError):
        provider.read_payload(handle.handle_id, "raw.bin", token)
    provider.close_generation(handle.provider_parent_generation_id, token)


def test_local_provider_worker_constructor_attaches_serialized_lease(
    tmp_path: Path,
) -> None:
    """A worker provider can read only after validating and attaching a token."""
    payloads = {"controls.bin": b"worker-controls"}
    payload_manifest = [
        {
            "name": name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(payloads.items())
    ]

    def prepare(**_: object) -> PreparedScenePayload:
        return PreparedScenePayload(
            identity=PreparedIdentity(
                provider_schema="prepared_geometry_provider.v1",
                parent_generation_id="generation-worker",
                domain="radar",
                ordered_source_ids=("reference-source", "secondary-source"),
                common_domain_digest=_digest("common"),
                policy_digest=_digest("policy"),
                schema_code_backend_device_digest=_digest("runtime"),
                payload_manifest_digest=_digest(payload_manifest),
            ),
            metadata={
                "expected_burst_manifest": [
                    list(_source(SourceRole.REFERENCE).burst_key)
                ]
            },
            payloads=payloads,
        )

    limits = _limits()
    owner = LocalPreparedGeometryProvider(tmp_path / "prepared", prepare)
    handle = owner.prepare_scene(
        reference=_source(SourceRole.REFERENCE),
        secondary=_source(SourceRole.SECONDARY),
        policy=_policy(),
        geo_grid=None,
        limits=limits,
        operation_id="prepare-worker",
    )
    token = owner.issue_lease(handle, limits)

    worker = LocalPreparedGeometryProvider.from_worker_generation(
        tmp_path / "prepared", token, limits
    )
    assert worker.read_payload(handle.handle_id, "controls.bin", token) == (
        b"worker-controls"
    )
    with pytest.raises(InvalidProcessingStateError):
        worker.prepare_scene(
            reference=_source(SourceRole.REFERENCE),
            secondary=_source(SourceRole.SECONDARY),
            policy=_policy(),
            geo_grid=None,
            limits=limits,
            operation_id="worker-must-not-prepare",
        )

    worker.unpin_generation(handle.provider_parent_generation_id, token)
    worker.close_generation(handle.provider_parent_generation_id, token)
    owner.unpin_generation(handle.provider_parent_generation_id, token)
    owner.close_generation(handle.provider_parent_generation_id, token)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capability_digest", "0" * 64),
        ("expiry_epoch_seconds", 0),
        ("generation_root_inode", 0),
    ],
)
def test_local_provider_worker_constructor_rejects_tampered_token(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Worker attachment fails closed for stale or altered token fields."""
    payloads = {"controls.bin": b"worker-controls"}
    payload_manifest = [
        {
            "name": "controls.bin",
            "size": len(payloads["controls.bin"]),
            "sha256": hashlib.sha256(payloads["controls.bin"]).hexdigest(),
        }
    ]

    def prepare(**_: object) -> PreparedScenePayload:
        return PreparedScenePayload(
            identity=PreparedIdentity(
                provider_schema="prepared_geometry_provider.v1",
                parent_generation_id="generation-worker-tamper",
                domain="radar",
                ordered_source_ids=("reference-source", "secondary-source"),
                common_domain_digest=_digest("common"),
                policy_digest=_digest("policy"),
                schema_code_backend_device_digest=_digest("runtime"),
                payload_manifest_digest=_digest(payload_manifest),
            ),
            metadata={
                "expected_burst_manifest": [
                    list(_source(SourceRole.REFERENCE).burst_key)
                ]
            },
            payloads=payloads,
        )

    limits = _limits()
    owner = LocalPreparedGeometryProvider(tmp_path / "prepared", prepare)
    handle = owner.prepare_scene(
        reference=_source(SourceRole.REFERENCE),
        secondary=_source(SourceRole.SECONDARY),
        policy=_policy(),
        geo_grid=None,
        limits=limits,
        operation_id="prepare-worker-tamper",
    )
    token = owner.issue_lease(handle, limits)
    if field == "expiry_epoch_seconds":
        tampered = replace(token, heartbeat_epoch_seconds=0, expiry_epoch_seconds=0)
    else:
        tampered = replace(token, **{field: value})
    with pytest.raises(InvalidProcessingStateError):
        LocalPreparedGeometryProvider.from_worker_generation(
            tmp_path / "prepared", tampered, limits
        )
    owner.unpin_generation(handle.provider_parent_generation_id, token)
    owner.close_generation(handle.provider_parent_generation_id, token)


def test_local_provider_rejects_role_swap_before_callback(tmp_path: Path) -> None:
    """Role mismatch fails before invoking the numerical preparation callback."""
    called = False

    def prepare(**_: object) -> PreparedScenePayload:
        nonlocal called
        called = True
        raise AssertionError

    provider = LocalPreparedGeometryProvider(tmp_path / "prepared", prepare)
    with pytest.raises(InvalidProcessingStateError):
        provider.prepare_scene(
            reference=_source(SourceRole.SECONDARY),
            secondary=_source(SourceRole.REFERENCE),
            policy=_policy(),
            geo_grid=None,
            limits=_limits(),
            operation_id="prepare-1",
        )
    assert not called


def test_local_provider_decodes_geometry_arrays_read_only(tmp_path: Path) -> None:
    """Provider payload reads return canonical arrays without pickle support."""
    shape = (4, 4)
    range_offset = np.full(shape, 0.25, dtype=np.float32)
    azimuth_offset = np.full(shape, -0.125, dtype=np.float32)
    uncertainty = np.zeros(shape, dtype=np.float32)
    coverage = np.ones(shape, dtype=bool)
    post_buffer = io.BytesIO()
    np.savez(
        post_buffer,
        range_offset_px=range_offset,
        azimuth_offset_px=azimuth_offset,
        uncertainty_px=uncertainty,
    )
    mask_buffer = io.BytesIO()
    np.save(mask_buffer, coverage, allow_pickle=False)
    payloads = {
        "raw.bin": b"raw-controls",
        "post.npz": post_buffer.getvalue(),
        "mask.npy": mask_buffer.getvalue(),
    }
    payload_manifest = [
        {
            "name": name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(payloads.items())
    ]

    def prepare(**_: object) -> PreparedScenePayload:
        return PreparedScenePayload(
            identity=PreparedIdentity(
                provider_schema="prepared_geometry_provider.v1",
                parent_generation_id="generation-arrays",
                domain="radar",
                ordered_source_ids=("reference-source", "secondary-source"),
                common_domain_digest=_digest("common"),
                policy_digest=_digest("policy"),
                schema_code_backend_device_digest=_digest("runtime"),
                payload_manifest_digest=_digest(payload_manifest),
            ),
            metadata={
                "expected_burst_manifest": [
                    list(_source(SourceRole.REFERENCE).burst_key)
                ],
                "geometry": {
                    "burst-0": {
                        "burst_key": list(_source(SourceRole.REFERENCE).burst_key),
                        "raw_payload": "raw.bin",
                        "post_fill_payload": "post.npz",
                        "validity_payload": "mask.npy",
                        "source_shape": list(shape),
                        "crop_bounds": [0, 4, 0, 4],
                        "control_spacing": 8,
                    }
                },
            },
            payloads=payloads,
        )

    provider = LocalPreparedGeometryProvider(tmp_path / "prepared", prepare)
    handle = provider.prepare_scene(
        reference=_source(SourceRole.REFERENCE),
        secondary=_source(SourceRole.SECONDARY),
        policy=_policy(),
        geo_grid=None,
        limits=_limits(),
        operation_id="prepare-arrays",
    )
    token = provider.issue_lease(handle, _limits())
    geometry_handle = provider.open_prepared_geometry(handle.handle_id, token)
    arrays = provider.read_prepared_geometry(geometry_handle, token)
    assert isinstance(arrays, PreparedGeometryArrayPayload)
    np.testing.assert_array_equal(arrays.range_offset_px, range_offset)
    np.testing.assert_array_equal(arrays.azimuth_offset_px, azimuth_offset)
    np.testing.assert_array_equal(arrays.coverage, coverage)
    assert not arrays.range_offset_px.flags.writeable
    assert not arrays.coverage.flags.writeable
    provider.unpin_generation(handle.provider_parent_generation_id, token)
    provider.close_generation(handle.provider_parent_generation_id, token)


def test_local_provider_decodes_geo_lut_with_explicit_crop_origin(
    tmp_path: Path,
) -> None:
    """A prepared LUT read preserves grid identity and crop coordinates."""
    from faninsar.processing.merge.grid import GeoGridSpec
    from faninsar.processing.pipeline.production import read_prepared_lut

    grid = GeoGridSpec(
        crs="EPSG:32633",
        transform=(100.0, 10.0, 0.0, 200.0, 0.0, -10.0),
        width=6,
        height=4,
        resolution_m=(10.0, 10.0),
    )
    crop_bounds = (1, 3, 2, 5)
    shape = (2, 3)
    arrays = {
        "az_full": np.arange(6, dtype=np.float64).reshape(shape),
        "rg_full": np.arange(6, dtype=np.float64).reshape(shape) + 10.0,
        "valid": np.ones(shape, dtype=bool),
        "height_full": np.full(shape, 42.0, dtype=np.float64),
    }
    lut_buffer = io.BytesIO()
    np.savez(lut_buffer, **arrays)
    payloads = {
        "lut.npz": lut_buffer.getvalue(),
    }
    payload_manifest = [
        {
            "name": name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(payloads.items())
    ]
    view_id = _digest("lut-view")

    def prepare(**_: object) -> PreparedScenePayload:
        return PreparedScenePayload(
            identity=PreparedIdentity(
                provider_schema="prepared_geometry_provider.v1",
                parent_generation_id="generation-lut",
                domain="geo",
                ordered_source_ids=("reference-source", "secondary-source"),
                common_domain_digest=_digest("common"),
                policy_digest=_digest("policy"),
                schema_code_backend_device_digest=_digest("runtime"),
                payload_manifest_digest=_digest(payload_manifest),
            ),
            metadata={
                "expected_burst_manifest": [
                    list(_source(SourceRole.REFERENCE).burst_key)
                ],
                "luts": {
                    view_id: {
                        "payload": "lut.npz",
                        "geo_grid_identity": geo_grid_identity(grid),
                        "full_radar_shape": [8, 9],
                        "geo_grid_shape": [4, 6],
                        "crop_bounds": list(crop_bounds),
                        "height_m": 42.0,
                    }
                },
            },
            payloads=payloads,
        )

    provider = LocalPreparedGeometryProvider(tmp_path / "prepared", prepare)
    handle = provider.prepare_scene(
        reference=_source(SourceRole.REFERENCE),
        secondary=_source(SourceRole.SECONDARY),
        policy=_policy(),
        geo_grid=grid,
        limits=_limits(),
        operation_id="prepare-lut",
    )
    token = provider.issue_lease(handle, _limits())
    request = SceneViewRequest(
        view_id=view_id,
        parent_id=handle.handle_id,
        pair_id="pair-lut",
        attempt_id="attempt-1",
        view_kind="pair_window",
        ordered_burst_keys=(_source(SourceRole.REFERENCE).burst_key,),
        normalized_roi="roi",
        normalized_crs=grid.crs,
        crop_bounds=crop_bounds,
        halo=(0, 0, 0, 0),
        output_origin=(1, 2),
        output_shape=shape,
        residual_window_digest=_digest("window"),
        source_mask_identity="mask",
    )
    lut_handle = provider.open_prepared_lut(
        handle.handle_id,
        request,
        grid,
        token,
    )
    payload = provider.read_prepared_lut(lut_handle, token)
    assert isinstance(payload, PreparedLutArrayPayload)
    np.testing.assert_array_equal(payload.az_full, arrays["az_full"])
    assert payload.crop_bounds == crop_bounds
    assert not payload.az_full.flags.writeable
    lut = read_prepared_lut(provider, lut_handle, token)
    assert lut.shape == shape
    assert (lut.row0, lut.col0) == (1, 2)
    np.testing.assert_array_equal(lut.valid, arrays["valid"])
    provider.unpin_generation(handle.provider_parent_generation_id, token)
    provider.close_generation(handle.provider_parent_generation_id, token)


def test_local_provider_rejects_geo_lut_grid_identity_mismatch(tmp_path: Path) -> None:
    """A LUT cannot be opened for a different CRS or grid shape."""
    from faninsar.processing.merge.grid import GeoGridSpec

    grid = GeoGridSpec(
        crs="EPSG:32633",
        transform=(100.0, 10.0, 0.0, 200.0, 0.0, -10.0),
        width=2,
        height=2,
        resolution_m=(10.0, 10.0),
    )
    other_grid = GeoGridSpec(
        crs="EPSG:32633",
        transform=(100.0, 10.0, 0.0, 200.0, 0.0, -10.0),
        width=3,
        height=2,
        resolution_m=(10.0, 10.0),
    )
    payloads = {"lut.bin": b"not-an-array"}
    manifest = [
        {
            "name": "lut.bin",
            "size": len(payloads["lut.bin"]),
            "sha256": hashlib.sha256(payloads["lut.bin"]).hexdigest(),
        }
    ]
    view_id = _digest("mismatch-view")

    def prepare(**_: object) -> PreparedScenePayload:
        return PreparedScenePayload(
            identity=PreparedIdentity(
                provider_schema="prepared_geometry_provider.v1",
                parent_generation_id="generation-mismatch",
                domain="geo",
                ordered_source_ids=("reference-source", "secondary-source"),
                common_domain_digest=_digest("common"),
                policy_digest=_digest("policy"),
                schema_code_backend_device_digest=_digest("runtime"),
                payload_manifest_digest=_digest(manifest),
            ),
            metadata={
                "expected_burst_manifest": [
                    list(_source(SourceRole.REFERENCE).burst_key)
                ],
                "luts": {
                    view_id: {
                        "payload": "lut.bin",
                        "geo_grid_identity": geo_grid_identity(grid),
                        "full_radar_shape": [2, 2],
                        "geo_grid_shape": [2, 2],
                        "crop_bounds": [0, 2, 0, 2],
                        "height_m": 0.0,
                    }
                },
            },
            payloads=payloads,
        )

    provider = LocalPreparedGeometryProvider(tmp_path / "prepared", prepare)
    handle = provider.prepare_scene(
        reference=_source(SourceRole.REFERENCE),
        secondary=_source(SourceRole.SECONDARY),
        policy=_policy(),
        geo_grid=grid,
        limits=_limits(),
        operation_id="prepare-mismatch",
    )
    token = provider.issue_lease(handle, _limits())
    request = SceneViewRequest(
        view_id=view_id,
        parent_id=handle.handle_id,
        pair_id="pair-mismatch",
        attempt_id="attempt-1",
        view_kind="pair_window",
        ordered_burst_keys=(_source(SourceRole.REFERENCE).burst_key,),
        normalized_roi="roi",
        normalized_crs=other_grid.crs,
        crop_bounds=(0, 2, 0, 2),
        halo=(0, 0, 0, 0),
        output_origin=(0, 0),
        output_shape=(2, 2),
        residual_window_digest=_digest("window"),
        source_mask_identity="mask",
    )
    with pytest.raises(InvalidProcessingStateError):
        provider.open_prepared_lut(handle.handle_id, request, other_grid, token)
