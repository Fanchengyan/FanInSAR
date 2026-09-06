"""Local provider adapter backed by the crash-safe prepared generation store.

The adapter intentionally receives the numerical preparation callback from the
orchestration layer.  That keeps the existing geometry solver unchanged while
making publication, identity, pinning, and read-back policy executable now.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import psutil

from faninsar.logging import setup_logger
from faninsar.processing.contracts.prepared_geometry import (
    PROVIDER_SCHEMA,
    CoregistrationPolicy,
    PreparedGeometryArrayPayload,
    PreparedGeometryHandle,
    PreparedIdentity,
    PreparedLutArrayPayload,
    PreparedLutHandle,
    PreparedSceneHandle,
    PreparedViewHandle,
    ProviderLeaseToken,
    ResourceLimits,
    SceneViewRequest,
    SourceDescriptor,
    SourceRole,
    ViewState,
    WorkerAttestation,
    geo_grid_identity,
)
from faninsar.processing.errors import InvalidProcessingStateError, reject_invalid_state
from faninsar.processing.geometry.prepared_store import (
    PreparedGenerationLease,
    PreparedGenerationStore,
)

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.processing.mosaicking.grid import GeoGridSpec

logger = setup_logger(__name__)


def _canonical_json(value: object) -> bytes:
    """Serialize provider metadata with one stable encoding."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"provider metadata is not canonicalizable: {error}")


def _digest(value: object) -> str:
    """Return a canonical SHA-256 digest."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _payload_manifest_digest(payloads: Mapping[str, bytes]) -> str:
    """Hash payload names, lengths, and bytes before publication."""
    entries = [
        {
            "name": name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(payloads.items())
    ]
    return _digest(entries)


@dataclass(frozen=True, slots=True)
class PreparedScenePayload:
    """Numerical callback result consumed by :class:`LocalPreparedGeometryProvider`.

    ``metadata`` must contain ``expected_burst_manifest`` and may contain
    ``geometry``, ``views``, and ``luts`` binding tables.  Payload bytes are
    opaque to the store; their dtype/shape validation remains the numerical
    consumer's responsibility before this callback returns.
    """

    identity: PreparedIdentity
    metadata: Mapping[str, Any]
    payloads: Mapping[str, bytes]


class ScenePreparationCallback(Protocol):
    """Callable boundary for the existing exact geometry solver."""

    def __call__(
        self,
        *,
        reference: SourceDescriptor,
        secondary: SourceDescriptor,
        policy: CoregistrationPolicy,
        geo_grid: GeoGridSpec | None,
        limits: ResourceLimits,
        operation_id: str,
    ) -> PreparedScenePayload:
        """Produce canonical controls/LUT payloads without publishing them."""
        ...


class LocalPreparedGeometryProvider:
    """Concrete local ``prepared_geometry_provider.v1`` implementation.

    Parameters
    ----------
    root : pathlib.Path or str
        Caller-owned ``work_dir/prepared`` store root.
    prepare_callback : ScenePreparationCallback or None
        Existing numerical preparation function.  It must return canonical
        full-domain controls and any child metadata before this provider
        publishes the generation.  Worker-only providers leave it unset.
    max_payload_bytes : int, optional
        Bound enforced by the generation store before write and read.

    """

    def __init__(
        self,
        root: str | Path,
        prepare_callback: ScenePreparationCallback | None = None,
        *,
        max_payload_bytes: int = 512 * 1024 * 1024,
        worker_token: ProviderLeaseToken | None = None,
        worker_limits: ResourceLimits | None = None,
    ) -> None:
        """Create a provider with an injected numerical preparation seam.

        A worker process must not receive the numerical preparation callback.
        It can instead pass ``worker_token`` and ``worker_limits`` to attach
        the serialized capability issued by the owner process.  Attaching is
        performed only after the token's root identity, resource binding,
        expiry, and manifest capability have been checked.

        Parameters
        ----------
        root : pathlib.Path or str
            Caller-owned ``work_dir/prepared`` store root.
        prepare_callback : ScenePreparationCallback or None, optional
            Numerical preparation seam used by the owner process.  It is not
            required for a worker-only provider.
        max_payload_bytes : int, optional
            Bound enforced by the generation store before write and read.
        worker_token : ProviderLeaseToken or None, optional
            Serialized owner-issued lease capability for a worker process.
        worker_limits : ResourceLimits or None, optional
            Resource profile bound to ``worker_token``.

        Raises
        ------
        InvalidProcessingStateError
            If only one worker attachment argument is supplied, or if the
            serialized lease fails validation.

        """
        self.store = PreparedGenerationStore(
            root,
            max_payload_bytes=max_payload_bytes,
        )
        self.prepare_callback = prepare_callback
        self._leases: dict[tuple[str, str], PreparedGenerationLease] = {}
        self._tokens: dict[tuple[str, str], ProviderLeaseToken] = {}
        if (worker_token is None) != (worker_limits is None):
            reject_invalid_state(
                "worker_token and worker_limits must be supplied together"
            )
        if worker_token is not None and worker_limits is not None:
            self.attach_worker_lease(worker_token, worker_limits)

    @classmethod
    def from_worker_generation(
        cls,
        root: str | Path,
        token: ProviderLeaseToken,
        limits: ResourceLimits,
        *,
        max_payload_bytes: int = 512 * 1024 * 1024,
    ) -> LocalPreparedGeometryProvider:
        """Construct a read-only provider attached to an owner-issued lease.

        This constructor is intended for a worker process that receives only
        the prepared store root and a serialized :class:`ProviderLeaseToken`.
        The worker never republishes a generation or invokes the numerical
        preparation callback.

        Parameters
        ----------
        root : pathlib.Path or str
            The exact prepared store root used by the owner process.
        token : ProviderLeaseToken
            Owner-issued worker capability.
        limits : ResourceLimits
            Resource profile bound to the capability.
        max_payload_bytes : int, optional
            Bound enforced by the generation store before read-back.

        Returns
        -------
        LocalPreparedGeometryProvider
            Worker-side provider with one attached generation lease.

        """
        return cls(
            root,
            prepare_callback=None,
            max_payload_bytes=max_payload_bytes,
            worker_token=token,
            worker_limits=limits,
        )

    def attach_worker_lease(
        self,
        token: ProviderLeaseToken,
        limits: ResourceLimits,
    ) -> None:
        """Attach and validate one serialized owner-issued worker lease.

        ``PreparedGenerationStore.attach`` only recreates the local in-memory
        pin.  This method is the provider capability boundary: it validates
        the token before attachment and then verifies the immutable manifest
        against the token before making the lease available to provider
        methods.  A failed post-attach check releases the local pin again.

        Parameters
        ----------
        token : ProviderLeaseToken
            Serialized lease capability issued by :meth:`issue_lease`.
        limits : ResourceLimits
            Exact resource profile used when the token was issued.

        Raises
        ------
        InvalidProcessingStateError
            If the token is stale, belongs to another root or resource
            profile, or does not match the generation manifest.

        """
        if not isinstance(token, ProviderLeaseToken):
            reject_invalid_state("worker lease token has an invalid type")
        if not isinstance(limits, ResourceLimits):
            reject_invalid_state("worker lease limits have an invalid type")
        if token.state != "PINNED":
            reject_invalid_state("worker lease token is not pinned")
        if token.resource_digest != limits.limit_digest:
            reject_invalid_state("worker limits do not match the provider lease")
        if token.owner_uid != os.getuid():
            reject_invalid_state("worker lease owner is not the current user")
        if token.host_boot_id != str(psutil.boot_time()):
            reject_invalid_state("worker lease belongs to a different host boot")
        now = int(time.time())
        if token.expiry_epoch_seconds <= now:
            reject_invalid_state("worker lease token has expired")
        if token.heartbeat_epoch_seconds > now:
            reject_invalid_state("worker lease heartbeat is in the future")
        if (
            isinstance(token.generation_root_device, bool)
            or not isinstance(token.generation_root_device, int)
            or isinstance(token.generation_root_inode, bool)
            or not isinstance(token.generation_root_inode, int)
        ):
            reject_invalid_state("worker lease root identity is invalid")
        root_stat = self.store.root.stat()
        if (
            token.generation_root_device != root_stat.st_dev
            or token.generation_root_inode != root_stat.st_ino
        ):
            reject_invalid_state("worker lease root identity does not match the store")
        if not isinstance(token.source_snapshot_identities, (list, tuple)):
            reject_invalid_state("worker lease source snapshots are invalid")
        if any(
            not isinstance(identity, (list, tuple))
            or len(identity) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in identity
            )
            for identity in token.source_snapshot_identities
        ):
            reject_invalid_state("worker lease source snapshots are invalid")
        if not token.source_snapshot_identities:
            reject_invalid_state("worker lease source snapshots are missing")

        lease = PreparedGenerationLease(
            token.parent_generation_id,
            token.worker_nonce,
        )
        self.store.attach(lease)
        try:
            reader = self.store.open(lease)
            manifest = reader.manifest
            manifest_digest = hashlib.sha256(
                _canonical_json(manifest)
            ).hexdigest()
            expected_capability = _digest(
                {
                    "generation_id": token.parent_generation_id,
                    "lease_nonce": lease.nonce,
                    "root_device": root_stat.st_dev,
                    "root_inode": root_stat.st_ino,
                    "manifest_digest": manifest_digest,
                }
            )
            if token.capability_digest != expected_capability:
                reject_invalid_state(
                    "worker lease manifest capability does not match"
                )
            if manifest.get("generation_id") != token.parent_generation_id:
                reject_invalid_state("worker lease generation does not match manifest")
            if manifest.get("provider_schema") != PROVIDER_SCHEMA:
                reject_invalid_state("worker lease provider schema does not match")
            metadata = _manifest_metadata(manifest)
            source_identities = _source_snapshot_identities(metadata)
            if source_identities != token.source_snapshot_identities:
                reject_invalid_state(
                    "worker lease source snapshots do not match the generation"
                )
            identity = manifest.get("identity")
            if not isinstance(identity, Mapping):
                reject_invalid_state("worker lease generation identity is missing")
            manifest_identity = _identity_from_json(identity)
            if (
                manifest_identity.parent_generation_id != token.parent_generation_id
                or manifest_identity.provider_schema != PROVIDER_SCHEMA
            ):
                reject_invalid_state(
                    "worker lease identity does not match the generation"
                )
        except Exception:
            self.store.unpin(lease)
            raise
        key = (token.parent_generation_id, token.worker_nonce)
        self._leases[key] = lease
        self._tokens[key] = token

    def attach_worker(
        self,
        token: ProviderLeaseToken,
        limits: ResourceLimits,
    ) -> None:
        """Alias for :meth:`attach_worker_lease` used by worker initializers."""
        self.attach_worker_lease(token, limits)

    def provider_schema(self) -> str:
        """Return the sole neutral provider schema identifier."""
        return PROVIDER_SCHEMA

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
        """Run the injected solver once and publish one immutable generation."""
        if self.prepare_callback is None:
            reject_invalid_state(
                "worker-only prepared provider cannot prepare a new scene"
            )
        if reference.role is not SourceRole.REFERENCE:
            reject_invalid_state("prepare_scene reference argument has the wrong role")
        if secondary.role is not SourceRole.SECONDARY:
            reject_invalid_state("prepare_scene secondary argument has the wrong role")
        if reference.burst_key != secondary.burst_key:
            reject_invalid_state("reference and secondary burst keys must match")
        payload = self.prepare_callback(
            reference=reference,
            secondary=secondary,
            policy=policy,
            geo_grid=geo_grid,
            limits=limits,
            operation_id=operation_id,
        )
        if not isinstance(payload, PreparedScenePayload):
            reject_invalid_state(
                "prepared callback must return a PreparedScenePayload record"
            )
        actual_digest = _payload_manifest_digest(payload.payloads)
        if payload.identity.payload_manifest_digest != actual_digest:
            reject_invalid_state(
                "prepared identity payload digest does not match callback payload bytes"
            )
        expected_manifest = payload.metadata.get("expected_burst_manifest")
        if not isinstance(expected_manifest, (list, tuple)) or not expected_manifest:
            reject_invalid_state(
                "prepared payload must contain expected_burst_manifest"
            )
        metadata = dict(payload.metadata)
        metadata.setdefault(
            "source_snapshot_identities",
            [
                [reference.snapshot_device, reference.snapshot_inode],
                [secondary.snapshot_device, secondary.snapshot_inode],
            ],
        )
        self.store.stage(
            payload.identity.parent_generation_id,
            identity=payload.identity,
            metadata=metadata,
            payloads=payload.payloads,
        )
        record = self.store.publish(payload.identity.parent_generation_id)
        return PreparedSceneHandle(
            handle_id=payload.identity.parent_generation_id,
            provider_parent_generation_id=payload.identity.parent_generation_id,
            identity=payload.identity,
            expected_burst_manifest_digest=_digest(expected_manifest),
            owner_capability_digest=_digest(
                {
                    "generation_id": payload.identity.parent_generation_id,
                    "manifest_digest": record.manifest_digest,
                }
            ),
        )

    def issue_lease(
        self,
        handle: PreparedSceneHandle,
        limits: ResourceLimits,
    ) -> ProviderLeaseToken:
        """Issue an owner/worker-bound pin token for local orchestration."""
        lease = self.store.pin(handle.provider_parent_generation_id)
        now = int(time.time())
        root_stat = self.store.root.stat()
        process = psutil.Process(os.getpid())
        reader = self.store.open(lease)
        manifest = reader.manifest
        manifest_digest = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        expected_owner_capability = _digest(
            {
                "generation_id": handle.provider_parent_generation_id,
                "manifest_digest": manifest_digest,
            }
        )
        if handle.owner_capability_digest != expected_owner_capability:
            self.store.unpin(lease)
            reject_invalid_state("prepared manifest changed before lease issuance")
        try:
            source_snapshot_identities = _source_snapshot_identities(
                _manifest_metadata(manifest)
            )
        except InvalidProcessingStateError:
            self.store.unpin(lease)
            raise
        capability = _digest(
            {
                "generation_id": handle.provider_parent_generation_id,
                "lease_nonce": lease.nonce,
                "root_device": root_stat.st_dev,
                "root_inode": root_stat.st_ino,
                "manifest_digest": manifest_digest,
            }
        )
        token = ProviderLeaseToken(
            owner_nonce=secrets.token_hex(16),
            worker_nonce=lease.nonce,
            parent_generation_id=handle.provider_parent_generation_id,
            generation_root_device=root_stat.st_dev,
            generation_root_inode=root_stat.st_ino,
            source_snapshot_identities=tuple(
                sorted(
                    (int(identity[0]), int(identity[1]))
                    for identity in source_snapshot_identities
                    if isinstance(identity, (list, tuple)) and len(identity) == 2
                )
            ),
            owner_uid=os.getuid(),
            host_boot_id=str(psutil.boot_time()),
            pid_start=str(process.create_time()),
            expiry_epoch_seconds=now + 3600,
            heartbeat_epoch_seconds=now,
            capability_digest=capability,
            resource_digest=limits.limit_digest,
            state="PINNED",
        )
        key = (token.parent_generation_id, token.worker_nonce)
        self._leases[key] = lease
        self._tokens[key] = token
        return token

    def bootstrap_worker(
        self,
        token: ProviderLeaseToken,
        limits: ResourceLimits,
    ) -> WorkerAttestation:
        """Attest the resolved local backend before numerical worker imports."""
        self._require_token(token)
        if token.resource_digest != limits.limit_digest:
            reject_invalid_state("worker limits do not match the provider lease")
        return WorkerAttestation(
            token_digest=_digest(asdict(token)),
            provider_schema=PROVIDER_SCHEMA,
            resolved_backend="python-reference",
            physical_device_identity="cpu",
            visible_device_mapping="cpu",
            numerical_environment_digest=_digest({"backend": "python-reference"}),
            bootstrap_code_digest=_digest("local-prepared-provider-v1"),
            monitor_lease_digest=_digest(token.capability_digest),
            pid_start=token.pid_start,
            limits_digest=limits.limit_digest,
        )

    def open_prepared_geometry(
        self,
        handle_id: str,
        token: ProviderLeaseToken,
    ) -> PreparedGeometryHandle:
        """Open one metadata-bound geometry payload read-only."""
        reader, manifest = self._open_parent(token.parent_generation_id, token)
        metadata = _manifest_metadata(manifest)
        binding, output_handle_id = self._geometry_binding(
            token.parent_generation_id,
            handle_id,
            metadata,
        )
        return self._geometry_handle(
            reader,
            token.parent_generation_id,
            binding,
            output_handle_id,
        )

    def read_prepared_geometry(
        self,
        geometry_handle: PreparedGeometryHandle,
        token: ProviderLeaseToken,
    ) -> PreparedGeometryArrayPayload:
        """Read one validated dense geometry field from a pinned generation.

        Parameters
        ----------
        geometry_handle : PreparedGeometryHandle
            Handle returned by :meth:`open_prepared_geometry`.
        token : ProviderLeaseToken
            Live capability for the parent generation.

        Returns
        -------
        PreparedGeometryArrayPayload
            Read-only float32 controls, boolean coverage, crop metadata, and
            control spacing.

        Raises
        ------
        InvalidProcessingStateError
            If the handle is stale, the generation binding is incomplete, or
            the payload is not a bounded primitive NumPy array.

        """
        reader, manifest = self._open_parent(token.parent_generation_id, token)
        if geometry_handle.parent_id != token.parent_generation_id:
            reject_invalid_state("geometry handle parent does not match the lease")
        binding, output_handle_id = self._geometry_binding(
            token.parent_generation_id,
            geometry_handle.handle_id,
            _manifest_metadata(manifest),
        )
        if output_handle_id != geometry_handle.handle_id:
            reject_invalid_state("geometry handle identity is not canonical")
        post_name = binding.get("post_fill_payload")
        mask_name = binding.get("validity_payload")
        if not isinstance(post_name, str) or not isinstance(mask_name, str):
            reject_invalid_state("geometry binding payload names are incomplete")
        source_shape = _validated_int_tuple(binding.get("source_shape"), 2)
        crop_bounds = _validated_int_tuple(binding.get("crop_bounds"), 4)
        control_spacing = binding.get("control_spacing")
        if not isinstance(control_spacing, int) or control_spacing < 1:
            reject_invalid_state("geometry binding control spacing is invalid")
        post_payload = reader.read_payload(post_name)
        mask_payload = reader.read_payload(mask_name)
        try:
            with np.load(io.BytesIO(post_payload), allow_pickle=False) as archive:
                expected_names = {
                    "range_offset_px",
                    "azimuth_offset_px",
                    "uncertainty_px",
                }
                if set(archive.files) != expected_names:
                    reject_invalid_state(
                        "post-fill geometry payload has unexpected array names"
                    )
                range_offset_px = np.asarray(archive["range_offset_px"])
                azimuth_offset_px = np.asarray(archive["azimuth_offset_px"])
                uncertainty_px = np.asarray(archive["uncertainty_px"])
            coverage = np.asarray(
                np.load(io.BytesIO(mask_payload), allow_pickle=False)
            )
        except (OSError, ValueError, TypeError) as error:
            reject_invalid_state(
                "prepared geometry payload is not valid NumPy data: " + str(error)
            )
        return PreparedGeometryArrayPayload(
            range_offset_px=range_offset_px,
            azimuth_offset_px=azimuth_offset_px,
            coverage=coverage,
            uncertainty_px=uncertainty_px,
            source_shape=source_shape,
            crop_bounds=crop_bounds,
            control_spacing=control_spacing,
        )

    def geometry_handle_ids(
        self,
        parent_handle_id: str,
        token: ProviderLeaseToken,
    ) -> tuple[str, ...]:
        """Return deterministic child geometry handle IDs for a parent."""
        _, manifest = self._open_parent(parent_handle_id, token)
        metadata = _manifest_metadata(manifest)
        bindings = metadata.get("geometry")
        if not isinstance(bindings, Mapping):
            reject_invalid_state("prepared generation has no geometry bindings")
        return tuple(
            f"{parent_handle_id}::geometry::{key}" for key in sorted(bindings)
        )

    def materialize_view(
        self,
        handle_id: str,
        request: SceneViewRequest,
        token: ProviderLeaseToken,
    ) -> PreparedViewHandle:
        """Open one exact pre-materialized child view without refilling."""
        reader, manifest = self._open_parent(handle_id, token)
        metadata = _manifest_metadata(manifest)
        if request.parent_id != handle_id:
            reject_invalid_state(
                "scene view parent does not match the provider handle"
            )
        bindings = metadata.get("views")
        binding = (
            bindings.get(request.view_id) if isinstance(bindings, Mapping) else None
        )
        if not isinstance(binding, Mapping):
            reject_invalid_state(
                "requested scene view is not present in the generation"
            )
        payload_name = binding.get("payload")
        if not isinstance(payload_name, str):
            reject_invalid_state("scene view binding has no payload")
        payload = reader.read_payload(payload_name)
        try:
            state = ViewState(binding.get("state", ViewState.FINAL))
        except ValueError as error:
            reject_invalid_state(f"scene view state is invalid: {error}")
        return PreparedViewHandle(
            handle_id=request.view_id,
            request=request,
            state=state,
            payload_digest=hashlib.sha256(payload).hexdigest(),
            read_only_capability_digest=_digest(
                {"generation": handle_id, "view": request.view_id}
            ),
        )

    def open_prepared_lut(
        self,
        handle_id: str,
        request: SceneViewRequest,
        geo_grid: GeoGridSpec,
        token: ProviderLeaseToken,
    ) -> PreparedLutHandle:
        """Open an identity-matched precomputed LUT payload."""
        reader, manifest = self._open_parent(handle_id, token)
        metadata = _manifest_metadata(manifest)
        bindings = metadata.get("luts")
        binding = (
            bindings.get(request.view_id) if isinstance(bindings, Mapping) else None
        )
        if not isinstance(binding, Mapping):
            reject_invalid_state("requested geo LUT is not present in the generation")
        payload_name = binding.get("payload")
        grid_identity = binding.get("geo_grid_identity")
        if not isinstance(payload_name, str) or not isinstance(grid_identity, str):
            reject_invalid_state("geo LUT binding is incomplete")
        if grid_identity != geo_grid_identity(geo_grid):
            reject_invalid_state(
                "requested geographic grid does not match the prepared LUT"
            )
        payload = reader.read_payload(payload_name)
        return PreparedLutHandle(
            handle_id=f"{handle_id}::lut::{request.view_id}",
            parent_id=handle_id,
            view_id=request.view_id,
            geo_grid_identity=grid_identity,
            payload_digest=hashlib.sha256(payload).hexdigest(),
            read_only_capability_digest=_digest(
                {
                    "generation": handle_id,
                    "view": request.view_id,
                    "grid": grid_identity,
                }
            ),
        )

    def read_prepared_lut(
        self,
        lut_handle: PreparedLutHandle,
        token: ProviderLeaseToken,
    ) -> PreparedLutArrayPayload:
        """Read and validate one provider-owned geo2rdr LUT payload.

        The reader accepts only the immutable payload selected by
        ``open_prepared_lut``.  It never reopens a caller path or infers a
        crop shape from a fixed filename.
        """
        reader, manifest = self._open_parent(lut_handle.parent_id, token)
        expected_handle = f"{lut_handle.parent_id}::lut::{lut_handle.view_id}"
        if lut_handle.handle_id != expected_handle:
            reject_invalid_state("geo LUT handle identity is not canonical")
        metadata = _manifest_metadata(manifest)
        bindings = metadata.get("luts")
        binding = (
            bindings.get(lut_handle.view_id) if isinstance(bindings, Mapping) else None
        )
        if not isinstance(binding, Mapping):
            reject_invalid_state("geo LUT binding is missing")
        payload_name = binding.get("payload")
        grid_identity = binding.get("geo_grid_identity")
        if not isinstance(payload_name, str) or not isinstance(grid_identity, str):
            reject_invalid_state("geo LUT binding is incomplete")
        if grid_identity != lut_handle.geo_grid_identity:
            reject_invalid_state("geo LUT handle grid identity does not match manifest")
        payload = reader.read_payload(payload_name)
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                expected_names = {"az_full", "rg_full", "valid", "height_full"}
                if set(archive.files) != expected_names:
                    reject_invalid_state(
                        "geo LUT payload has unexpected array names"
                    )
                az_full = np.asarray(archive["az_full"])
                rg_full = np.asarray(archive["rg_full"])
                valid = np.asarray(archive["valid"])
                height_full = np.asarray(archive["height_full"])
        except (OSError, ValueError, TypeError) as error:
            reject_invalid_state(
                "geo LUT payload is not valid NumPy data: " + str(error)
            )

        full_radar_shape = _validated_int_tuple(binding.get("full_radar_shape"), 2)
        geo_grid_shape = _validated_int_tuple(binding.get("geo_grid_shape"), 2)
        crop_bounds = _validated_int_tuple(binding.get("crop_bounds"), 4)
        height_m = binding.get("height_m")
        if not isinstance(height_m, (int, float)) or not np.isfinite(height_m):
            reject_invalid_state("geo LUT binding mean height is invalid")
        return PreparedLutArrayPayload(
            az_full=az_full,
            rg_full=rg_full,
            valid=valid,
            height_full=height_full,
            full_radar_shape=full_radar_shape,
            geo_grid_shape=geo_grid_shape,
            crop_bounds=crop_bounds,
            height_m=float(height_m),
        )

    def identity(self, handle_id: str, token: ProviderLeaseToken) -> PreparedIdentity:
        """Return the manifest-bound parent identity."""
        reader, manifest = self._open_parent(handle_id, token)
        del reader
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping):
            reject_invalid_state("prepared generation identity is missing")
        return _identity_from_json(identity)

    def pin_generation(
        self,
        parent_generation_id: str,
        token: ProviderLeaseToken,
    ) -> None:
        """Pin a generation for a token that has not yet been opened."""
        self._require_token(token, parent_generation_id=parent_generation_id)

    def unpin_generation(
        self,
        parent_generation_id: str,
        token: ProviderLeaseToken,
    ) -> None:
        """Release one worker pin and invalidate its capability."""
        lease = self._require_token(token, parent_generation_id=parent_generation_id)
        self.store.unpin(lease)
        key = (parent_generation_id, token.worker_nonce)
        self._leases.pop(key, None)

    def abort_generation(
        self,
        parent_generation_id: str,
        token: ProviderLeaseToken,
    ) -> None:
        """Quarantine a generation after releasing its pin."""
        lease = self._require_token(token, parent_generation_id=parent_generation_id)
        self.store.unpin(lease)
        self.store.abort(parent_generation_id)
        key = (parent_generation_id, token.worker_nonce)
        self._leases.pop(key, None)
        self._tokens.pop(key, None)

    def close_generation(
        self,
        parent_generation_id: str,
        token: ProviderLeaseToken,
    ) -> None:
        """Close a generation after all worker pins have been released."""
        self._require_token(
            token,
            parent_generation_id=parent_generation_id,
            require_pin=False,
        )
        self.store.close(parent_generation_id)
        key = (parent_generation_id, token.worker_nonce)
        self._leases.pop(key, None)
        self._tokens.pop(key, None)

    def read_payload(
        self,
        parent_generation_id: str,
        payload_name: str,
        token: ProviderLeaseToken,
    ) -> bytes:
        """Read an opaque payload after provider lease validation."""
        reader, _ = self._open_parent(parent_generation_id, token)
        return reader.read_payload(payload_name)

    def _geometry_binding(
        self,
        parent_id: str,
        handle_id: str,
        metadata: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], str]:
        """Resolve a parent or child geometry handle against one manifest."""
        bindings = metadata.get("geometry")
        if not isinstance(bindings, Mapping):
            reject_invalid_state("prepared generation has no geometry bindings")
        prefix = f"{parent_id}::geometry::"
        if handle_id in (parent_id, f"{parent_id}::geometry"):
            if len(bindings) != 1:
                reject_invalid_state(
                    "open_prepared_geometry requires a child handle for multi-burst "
                    "generations"
                )
            binding = next(iter(bindings.values()))
            output_handle_id = f"{parent_id}::geometry"
        elif handle_id.startswith(prefix):
            child_key = handle_id[len(prefix) :]
            binding = bindings.get(child_key)
            output_handle_id = handle_id
        else:
            reject_invalid_state("geometry handle is not bound to the provider parent")
        if not isinstance(binding, Mapping):
            reject_invalid_state("geometry binding must be an object")
        return binding, output_handle_id

    def _open_parent(
        self,
        handle_id: str,
        token: ProviderLeaseToken,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Open a pinned parent and return its validated manifest."""
        lease = self._require_token(token, parent_generation_id=handle_id)
        reader = self.store.open(lease)
        manifest = reader.manifest
        manifest_digest = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        expected_capability = _digest(
            {
                "generation_id": handle_id,
                "lease_nonce": lease.nonce,
                "root_device": self.store.root.stat().st_dev,
                "root_inode": self.store.root.stat().st_ino,
                "manifest_digest": manifest_digest,
            }
        )
        if token.capability_digest != expected_capability:
            reject_invalid_state("provider lease manifest capability does not match")
        metadata = _manifest_metadata(manifest)
        expected_identities = _source_snapshot_identities(metadata)
        if expected_identities != token.source_snapshot_identities:
            reject_invalid_state(
                "provider lease source snapshots do not match the generation"
            )
        return reader, manifest

    def _require_token(
        self,
        token: ProviderLeaseToken,
        *,
        parent_generation_id: str | None = None,
        require_pin: bool = True,
    ) -> PreparedGenerationLease:
        """Validate capability, root identity, and active pin state."""
        if not isinstance(token, ProviderLeaseToken):
            reject_invalid_state("provider lease token has an invalid type")
        expected_parent = parent_generation_id or token.parent_generation_id
        key = (expected_parent, token.worker_nonce)
        lease = self._leases.get(key)
        if self._tokens.get(key) != token or (require_pin and lease is None):
            reject_invalid_state("provider lease is unknown, stale, or already closed")
        if token.state != "PINNED":
            reject_invalid_state("provider lease is not pinned")
        root_stat = self.store.root.stat()
        if (
            token.parent_generation_id != expected_parent
            or token.generation_root_device != root_stat.st_dev
            or token.generation_root_inode != root_stat.st_ino
        ):
            reject_invalid_state(
                "provider lease root identity does not match the store"
            )
        if lease is None and require_pin:
            reject_invalid_state("provider lease has no active pin")
        return lease

    def _geometry_handle(
        self,
        reader: Any,
        parent_id: str,
        binding: Mapping[str, Any],
        handle_id: str,
    ) -> PreparedGeometryHandle:
        """Validate one geometry binding and build its read-only handle."""
        raw_name = binding.get("raw_payload")
        post_name = binding.get("post_fill_payload")
        mask_name = binding.get("validity_payload")
        burst_key = binding.get("burst_key")
        if not all(
            isinstance(value, str) for value in (raw_name, post_name, mask_name)
        ):
            reject_invalid_state("geometry binding payload names are incomplete")
        if not isinstance(burst_key, (list, tuple)) or not burst_key:
            reject_invalid_state("geometry binding burst key is missing")
        raw = reader.read_payload(raw_name)
        post_fill = reader.read_payload(post_name)
        mask = reader.read_payload(mask_name)
        return PreparedGeometryHandle(
            handle_id=handle_id,
            parent_id=parent_id,
            burst_key=tuple(str(value) for value in burst_key),
            ordered_source_roles=(SourceRole.REFERENCE, SourceRole.SECONDARY),
            raw_payload_digest=hashlib.sha256(raw).hexdigest(),
            post_fill_payload_digest=hashlib.sha256(post_fill).hexdigest(),
            validity_mask_digest=hashlib.sha256(mask).hexdigest(),
            read_only_capability_digest=_digest(
                {"generation": parent_id, "raw": raw_name, "post_fill": post_name}
            ),
        )


def _identity_from_json(value: Mapping[str, Any]) -> PreparedIdentity:
    """Reconstruct and validate a manifest-bound prepared identity."""
    try:
        return PreparedIdentity(
            provider_schema=value["provider_schema"],
            parent_generation_id=value["parent_generation_id"],
            domain=value["domain"],
            ordered_source_ids=tuple(value["ordered_source_ids"]),
            common_domain_digest=value["common_domain_digest"],
            policy_digest=value["policy_digest"],
            schema_code_backend_device_digest=value[
                "schema_code_backend_device_digest"
            ],
            payload_manifest_digest=value["payload_manifest_digest"],
        )
    except (KeyError, TypeError, ValueError) as error:
        reject_invalid_state(f"prepared identity manifest is invalid: {error}")


def _manifest_metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return validated provider metadata from a generation manifest."""
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        reject_invalid_state("prepared generation metadata is missing")
    return metadata


def _source_snapshot_identities(
    metadata: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Return canonical source snapshot device/inode identities.

    Parameters
    ----------
    metadata : mapping
        Prepared-generation metadata containing
        ``source_snapshot_identities``.

    Returns
    -------
    tuple of tuple of int
        Sorted ``(st_dev, st_ino)`` identities.

    Raises
    ------
    InvalidProcessingStateError
        If the metadata is absent or contains malformed identities.

    """
    values = metadata.get("source_snapshot_identities")
    if not isinstance(values, (list, tuple)) or not values:
        reject_invalid_state("source snapshot identities are missing from the manifest")
    if any(
        not isinstance(identity, (list, tuple))
        or len(identity) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in identity
        )
        for identity in values
    ):
        reject_invalid_state("source snapshot identities are incomplete")
    return tuple(sorted((int(identity[0]), int(identity[1])) for identity in values))


def _validated_int_tuple(value: object, size: int) -> tuple[int, ...]:
    """Validate a fixed-size tuple of non-negative integer metadata."""
    if not isinstance(value, (list, tuple)) or len(value) != size:
        reject_invalid_state(f"geometry metadata must contain {size} integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        reject_invalid_state("geometry metadata must contain integers")
    result = tuple(int(item) for item in value)
    if any(item < 0 for item in result):
        reject_invalid_state("geometry metadata integers must be non-negative")
    return result


__all__ = [
    "LocalPreparedGeometryProvider",
    "PreparedGeometryArrayPayload",
    "PreparedScenePayload",
    "ScenePreparationCallback",
]
