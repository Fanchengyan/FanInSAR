"""Immutable manifest-bound storage for Stack ionosphere artifacts.

Ionosphere estimate stores live in their own namespace tree at
``<ion_root>/ml_<az>x<rg>/<pair>/`` while correction generations are
published inside the corresponding IFG pair directory (PROPOSAL-0036).
They reuse the same transactional generation machinery as IFG and unwrap
artifacts under the dedicated ``"ion"`` / ``"ion_correction"``
namespaces, so qualified, degraded, and differently configured
ionosphere generations can never silently resume or mix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.stack.artifact_transaction import (
    ArtifactResourceLimits,
    GenerationLease,
    commit_generation,
    open_current_generation,
    stage_generation,
)
from faninsar.stack.ifg_store import (
    _array_descriptor,
    _atomic_manifest,
    _atomic_save,
    _canonical_json,
    _digest_bytes,
    _is_sha256,
    _read_payloads,
    _store_root,
    _validate_json_mapping,
    _validate_payload_table,
    _validate_shape,
    _verified_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

logger = setup_logger(__name__)

ION_ARTIFACT_SCHEMA = "stack_ion_artifact_v1"
ION_CORRECTION_SCHEMA = "stack_ion_correction_artifact_v1"
ION_FILENAMES = {
    "ionosphere_phase": "ionosphere_phase.npy",
    "nondispersive_phase": "nondispersive_phase.npy",
    "weight": "weight.npy",
}
ION_CORRECTION_FILENAMES = {
    "corrected_unwrapped_phase": "corrected_unwrapped_phase.npy",
}


@dataclass(frozen=True, slots=True)
class IonosphereArtifact:
    """Validated arrays from one complete ionosphere generation.

    Attributes
    ----------
    ionosphere_phase : numpy.ndarray
        Dispersive ionospheric phase screen in radians. The screen models
        ``ion[primary] - ion[secondary]``; corrections subtract it from the
        pair's unwrapped interferometric phase.
    nondispersive_phase : numpy.ndarray
        Non-dispersive component retained as a diagnostic layer.
    weight : numpy.ndarray
        Final per-pixel inverse-variance weights in ``[0, 1]``; larger is
        more confident. NaN marks unconstrained pixels.

    """

    ionosphere_phase: np.ndarray
    nondispersive_phase: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True, slots=True)
class IonosphereArtifactStore:
    """Read-only validated access for one Stack pair ionosphere artifact.

    Attributes
    ----------
    root : pathlib.Path
        Validated artifact directory.
    pair : tuple[str, str]
        Primary and secondary acquisition identifiers.
    looks : tuple[int, int]
        Applied azimuth and range look factors of the ion grid.
    domain : str
        Coordinate domain of the source scene generation.
    wavelength_m : float | None
        Radar wavelength echoed from the source scenes.
    grid_identity : str
        Canonical source scene coordinate-grid identity.
    method_name : str
        Estimation method recorded at publication.
    method_parameters : dict[str, object]
        Canonical JSON echo of the full lane configuration.
    degraded : bool
        Whether this generation was published with the degraded flag.
    degradation_reason : str | None
        Caller-supplied reason code required when ``degraded`` is true.
    runtime_fingerprint : str
        Canonical Stack runtime fingerprint bound at publication.
    devices : tuple[str, ...]
        Resolved device list used by the estimation lane.
    source_manifest_digests : dict[str, str]
        Named source scene manifest digests.
    shape : tuple[int, int]
        Shared ion-grid shape.
    manifest_digest : str
        SHA-256 digest of the canonical manifest.

    """

    root: Path
    generation_id: str
    generation_root: Path
    pair: tuple[str, str]
    looks: tuple[int, int]
    domain: str
    wavelength_m: float | None
    grid_identity: str
    method_name: str
    method_parameters: dict[str, Any]
    degraded: bool
    degradation_reason: str | None
    runtime_fingerprint: str
    devices: tuple[str, ...]
    source_manifest_digests: dict[str, str]
    shape: tuple[int, int]
    manifest_digest: str
    _payloads: dict[str, dict[str, Any]]
    _lease: GenerationLease

    @classmethod
    def open(cls, root: str | Path) -> Self:
        """Open and validate a complete ionosphere generation."""
        path = _store_root(root, create=False)
        direct_payloads = [path / filename for filename in ION_FILENAMES.values()]
        if any(payload.exists() for payload in direct_payloads):
            reject_invalid_state(
                "direct ionosphere payload layout is not supported; ionosphere "
                "artifacts are transactional generations"
            )
        opened = open_current_generation(path, "ion")
        try:
            manifest = _verified_manifest(
                opened.path / "ion_manifest.json", ION_ARTIFACT_SCHEMA
            )
            if manifest.get("manifest_digest") != opened.manifest_digest:
                reject_invalid_state("ion CURRENT digest does not match its generation")
        except Exception:
            opened.lease.close()
            raise
        raw_pair = manifest.get("pair")
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or any(not isinstance(value, str) or not value for value in raw_pair)
        ):
            reject_invalid_state("ion artifact pair must contain two non-empty ids")
        raw_looks = manifest.get("looks")
        if (
            not isinstance(raw_looks, list)
            or len(raw_looks) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in raw_looks
            )
        ):
            reject_invalid_state("ion artifact looks must be two positive integers")
        domain = manifest.get("domain")
        if domain not in {"radar", "geo"}:
            reject_invalid_state("ion artifact domain is invalid")
        raw_wavelength = manifest.get("wavelength_m")
        wavelength_m = None if raw_wavelength is None else float(raw_wavelength)
        if wavelength_m is not None and (
            not np.isfinite(wavelength_m) or wavelength_m <= 0.0
        ):
            reject_invalid_state("ion artifact wavelength_m is invalid")
        grid_identity = manifest.get("grid_identity")
        if not _is_sha256(grid_identity):
            reject_invalid_state("ion artifact grid_identity is invalid")
        raw_method = manifest.get("method")
        if (
            not isinstance(raw_method, dict)
            or not isinstance(raw_method.get("name"), str)
            or not raw_method["name"]
        ):
            reject_invalid_state("ion artifact method metadata is invalid")
        method_parameters = _validate_json_mapping(
            raw_method.get("parameters"), "ion method parameters"
        )
        degraded = manifest.get("degraded")
        if not isinstance(degraded, dict) or not isinstance(degraded.get("flag"), bool):
            reject_invalid_state("ion artifact degraded metadata is invalid")
        degradation_reason = degraded.get("reason")
        if degraded["flag"] and (
            not isinstance(degradation_reason, str) or not degradation_reason
        ):
            reject_invalid_state("degraded ion artifacts require a degradation reason")
        if not degraded["flag"] and degradation_reason is not None:
            reject_invalid_state(
                "qualified ion artifacts must not carry a degradation reason"
            )
        runtime_fingerprint = manifest.get("runtime_fingerprint")
        if not _is_sha256(runtime_fingerprint):
            reject_invalid_state("ion artifact runtime fingerprint is invalid")
        raw_devices = manifest.get("devices")
        if (
            not isinstance(raw_devices, list)
            or not raw_devices
            or any(not isinstance(value, str) or not value for value in raw_devices)
        ):
            reject_invalid_state("ion artifact device list is invalid")
        sources = manifest.get("source_manifest_digests")
        if (
            not isinstance(sources, dict)
            or not sources
            or any(
                not isinstance(name, str) or not name or not _is_sha256(digest)
                for name, digest in sources.items()
            )
        ):
            reject_invalid_state("ion artifact source manifest digests are invalid")
        shape = _validate_shape(manifest.get("shape"), "ion shape")
        payloads = _validate_payload_table(
            opened.path, manifest.get("payloads"), ION_FILENAMES, shape
        )
        return cls(
            root=path,
            generation_id=opened.generation_id,
            generation_root=opened.path,
            pair=(raw_pair[0], raw_pair[1]),
            looks=(raw_looks[0], raw_looks[1]),
            domain=str(domain),
            wavelength_m=wavelength_m,
            grid_identity=str(grid_identity),
            method_name=str(raw_method["name"]),
            method_parameters=method_parameters,
            degraded=bool(degraded["flag"]),
            degradation_reason=(
                None if degradation_reason is None else str(degradation_reason)
            ),
            runtime_fingerprint=str(runtime_fingerprint),
            devices=tuple(str(value) for value in raw_devices),
            source_manifest_digests={str(k): str(v) for k, v in sources.items()},
            shape=shape,
            manifest_digest=str(manifest["manifest_digest"]),
            _payloads=payloads,
            _lease=opened.lease,
        )

    def close(self) -> None:
        """Release this store's durable reader pin."""
        self._lease.close()

    def __enter__(self) -> Self:
        """Return this pinned store as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the generation pin on context-manager exit."""
        self.close()

    def read(self) -> IonosphereArtifact:
        """Read all ionosphere layers after full payload revalidation."""
        arrays = _read_payloads(self.generation_root, self._payloads)
        if any(arrays[name].dtype.kind != "f" for name in ION_FILENAMES):
            reject_invalid_state("ion artifact layer dtypes are incompatible")
        return IonosphereArtifact(
            ionosphere_phase=arrays["ionosphere_phase"],
            nondispersive_phase=arrays["nondispersive_phase"],
            weight=arrays["weight"],
        )


def write_ionosphere_artifact(
    root: str | Path,
    *,
    pair: tuple[str, str],
    looks: tuple[int, int],
    domain: str,
    wavelength_m: float | None,
    grid_identity: str,
    method_name: str,
    method_parameters: Mapping[str, Any],
    degraded: bool,
    degradation_reason: str | None,
    runtime_fingerprint: str,
    devices: Sequence[str],
    source_manifest_digests: Mapping[str, str],
    ionosphere_phase: np.ndarray,
    nondispersive_phase: np.ndarray,
    weight: np.ndarray,
    resource_limits: ArtifactResourceLimits | None = None,
    replace_existing: bool = False,
) -> IonosphereArtifactStore:
    """Atomically persist one complete ionosphere generation.

    Parameters
    ----------
    root : str or pathlib.Path
        Destination artifact directory (``ml_<az>x<rg>/<pair>`` under the
        ion artifact root).
    pair : tuple[str, str]
        Primary and secondary acquisition identifiers.
    looks : tuple[int, int]
        Applied azimuth and range look factors of the ion grid.
    domain : {"radar", "geo"}
        Coordinate domain of the source scene generation.
    wavelength_m : float, optional
        Radar wavelength echoed from the source scenes.
    grid_identity : str
        Canonical source scene coordinate-grid identity.
    method_name : str
        Estimation method name (``"ionosphere_split_spectrum"``).
    method_parameters : mapping
        Canonical JSON echo of the full lane configuration.
    degraded : bool
        Whether this generation is marked degraded.
    degradation_reason : str, optional
        Required non-empty reason when ``degraded`` is true; forbidden
        otherwise.
    runtime_fingerprint : str
        Canonical Stack runtime fingerprint (SHA-256).
    devices : sequence of str
        Resolved device list used by the estimation lane.
    source_manifest_digests : mapping[str, str]
        Named SHA-256 digests of every consumed scene manifest.
    ionosphere_phase, nondispersive_phase, weight : numpy.ndarray
        Matching two-dimensional float layers.
    resource_limits : ArtifactResourceLimits, optional
        Transactional generation resource bounds (leases, staging size).
    replace_existing : bool, optional
        Republish over an already-published generation when true; the
        default refuses to overwrite a current generation.

    Returns
    -------
    IonosphereArtifactStore
        Reopened, validated artifact store.

    """
    arrays = {
        "ionosphere_phase": np.asarray(ionosphere_phase),
        "nondispersive_phase": np.asarray(nondispersive_phase),
        "weight": np.asarray(weight),
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        reject_invalid_state("ion artifact layers must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 2 or any(size <= 0 for size in shape):
        reject_invalid_state("ion artifact layers must be non-empty 2-D arrays")
    if any(arrays[name].dtype.kind != "f" for name in arrays):
        reject_invalid_state("ion artifact layer dtypes must be floating point")
    devices_tuple = tuple(str(value) for value in devices)
    if (
        len(pair) != 2
        or any(not isinstance(value, str) or not value for value in pair)
        or len(looks) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in looks
        )
        or domain not in {"radar", "geo"}
        or not isinstance(method_name, str)
        or not method_name
        or not devices_tuple
        or any(not value for value in devices_tuple)
    ):
        reject_invalid_state("ion artifact processing metadata is invalid")
    if wavelength_m is not None and (
        not np.isfinite(wavelength_m) or wavelength_m <= 0.0
    ):
        reject_invalid_state("ion artifact wavelength_m is invalid")
    if not _is_sha256(grid_identity):
        reject_invalid_state("ion artifact grid_identity is invalid")
    if not _is_sha256(runtime_fingerprint):
        reject_invalid_state("ion artifact runtime fingerprint is invalid")
    if degraded and (not isinstance(degradation_reason, str) or not degradation_reason):
        reject_invalid_state("degraded ion artifacts require a degradation reason")
    if not degraded and degradation_reason is not None:
        reject_invalid_state(
            "qualified ion artifacts must not carry a degradation reason"
        )
    parameters = dict(method_parameters)
    _canonical_json(parameters)
    sources = dict(source_manifest_digests)
    if not sources or any(
        not isinstance(name, str) or not name or not _is_sha256(digest)
        for name, digest in sources.items()
    ):
        reject_invalid_state("ion artifact source manifest digests are invalid")

    path = _store_root(root, create=True)
    estimated_bytes = sum(int(array.nbytes) + 1024 for array in arrays.values())
    with stage_generation(
        path,
        "ion",
        final_bytes=estimated_bytes,
        temporary_bytes=estimated_bytes,
        file_count=len(arrays) + 1,
        dimensions=(int(shape[0]), int(shape[1])),
        limits=resource_limits,
    ) as (generation_id, staging):
        if not replace_existing and (
            (path / "ION_CURRENT").exists() or (path / "ion_manifest.json").exists()
        ):
            reject_invalid_state("ion artifact generation is already published")
        descriptors: dict[str, dict[str, Any]] = {}
        for name, filename in ION_FILENAMES.items():
            payload_path = staging / filename
            _atomic_save(payload_path, arrays[name])
            descriptors[name] = _array_descriptor(payload_path, arrays[name])
        unsigned: dict[str, Any] = {
            "schema_version": ION_ARTIFACT_SCHEMA,
            "status": "complete",
            "generation_id": generation_id,
            "pair": list(pair),
            "looks": list(looks),
            "domain": domain,
            "wavelength_m": wavelength_m,
            "grid_identity": grid_identity,
            "method": {"name": method_name, "parameters": parameters},
            "degraded": {"flag": bool(degraded), "reason": degradation_reason},
            "runtime_fingerprint": runtime_fingerprint,
            "devices": list(devices_tuple),
            "source_manifest_digests": sources,
            "shape": [int(shape[0]), int(shape[1])],
            "payloads": descriptors,
        }
        manifest = {
            **unsigned,
            "manifest_digest": _digest_bytes(_canonical_json(unsigned)),
        }
        _atomic_manifest(staging / "ion_manifest.json", manifest)
        commit_generation(
            path,
            "ion",
            generation_id,
            staging,
            manifest_digest=str(manifest["manifest_digest"]),
            compatibility_manifest=manifest,
        )
    logger.info("Published ion artifact %s for pair %s", path, pair)
    return IonosphereArtifactStore.open(path)


def write_ion_correction_artifact(
    root: str | Path,
    *,
    unwrapped_phase: np.ndarray,
    ion_manifest_digest: str,
    ifg_manifest_digest: str,
    degraded_consumed: bool,
    resource_limits: ArtifactResourceLimits | None = None,
    replace_existing: bool = False,
) -> None:
    """Atomically persist ion-corrected unwrapped phase for one pair.

    The correction generation is published under the ``ion_correction``
    namespace of the owning IFG pair directory, distinct from the
    date-level reconciled products. Upstream IFG, unwrap, and ion layers
    are never rewritten.

    Parameters
    ----------
    root : str or pathlib.Path
        Owning complete IFG pair artifact directory.
    unwrapped_phase : numpy.ndarray
        Ion-corrected unwrapped phase with the IFG unwrap grid shape.
    ion_manifest_digest : str
        Digest of the consumed ion generation manifest.
    ifg_manifest_digest : str
        Digest of the owning IFG generation manifest.
    degraded_consumed : bool
        Whether a degraded ion generation was consumed under an explicit
        ``allow_degraded`` admission.
    resource_limits : ArtifactResourceLimits, optional
        Hard publication size, file-count, and free-space limits.
    replace_existing : bool, optional
        Publish a new correction generation when true.

    """
    from faninsar.stack.ifg_store import InterferogramArtifactStore

    store = InterferogramArtifactStore.open(root)
    try:
        phase = np.asarray(unwrapped_phase)
        if phase.shape != store.shape or phase.dtype.kind != "f":
            reject_invalid_state(
                "corrected phase must match the IFG shape as a float array"
            )
        if (
            not _is_sha256(ion_manifest_digest)
            or not _is_sha256(ifg_manifest_digest)
            or ifg_manifest_digest != store.manifest_digest
        ):
            reject_invalid_state("ion correction digest binding is invalid")
        arrays = {"corrected_unwrapped_phase": phase.astype(np.float32)}
        estimated_bytes = sum(int(array.nbytes) + 1024 for array in arrays.values())
        with stage_generation(
            store.root,
            "ion_correction",
            final_bytes=estimated_bytes,
            temporary_bytes=estimated_bytes,
            file_count=len(arrays) + 1,
            dimensions=store.shape,
            limits=resource_limits,
        ) as (generation_id, staging):
            if not replace_existing and (
                (store.root / "ION_CORRECTION_CURRENT").exists()
                or (store.root / "ion_correction_manifest.json").exists()
            ):
                reject_invalid_state("ion correction generation is already published")
            descriptors: dict[str, dict[str, Any]] = {}
            for name, filename in ION_CORRECTION_FILENAMES.items():
                payload_path = staging / filename
                _atomic_save(payload_path, arrays[name])
                descriptors[name] = _array_descriptor(payload_path, arrays[name])
            unsigned: dict[str, Any] = {
                "schema_version": ION_CORRECTION_SCHEMA,
                "status": "complete",
                "generation_id": generation_id,
                "ifg_generation_id": store.generation_id,
                "ifg_manifest_digest": store.manifest_digest,
                "ion_manifest_digest": ion_manifest_digest,
                "degraded_consumed": bool(degraded_consumed),
                "shape": list(store.shape),
                "payloads": descriptors,
            }
            manifest = {
                **unsigned,
                "manifest_digest": _digest_bytes(_canonical_json(unsigned)),
            }
            _atomic_manifest(staging / "ion_correction_manifest.json", manifest)
            commit_generation(
                store.root,
                "ion_correction",
                generation_id,
                staging,
                manifest_digest=str(manifest["manifest_digest"]),
                compatibility_manifest=manifest,
            )
    finally:
        store.close()
    logger.info("Published ion correction artifact %s", root)


def read_ion_correction_artifact(root: str | Path) -> np.ndarray:
    """Read the hash-validated ion-corrected unwrapped phase of one pair."""
    path = _store_root(root, create=False)
    opened = open_current_generation(path, "ion_correction")
    try:
        manifest = _verified_manifest(
            opened.path / "ion_correction_manifest.json", ION_CORRECTION_SCHEMA
        )
        if manifest.get("manifest_digest") != opened.manifest_digest:
            reject_invalid_state(
                "ion correction CURRENT digest does not match its generation"
            )
        descriptors = _validate_payload_table(
            opened.path,
            manifest.get("payloads"),
            ION_CORRECTION_FILENAMES,
            _validate_shape(manifest.get("shape"), "ion correction shape"),
        )
        arrays = _read_payloads(opened.path, descriptors)
    finally:
        opened.lease.close()
    return arrays["corrected_unwrapped_phase"]


__all__ = [
    "ION_ARTIFACT_SCHEMA",
    "ION_CORRECTION_SCHEMA",
    "IonosphereArtifact",
    "IonosphereArtifactStore",
    "read_ion_correction_artifact",
    "write_ion_correction_artifact",
    "write_ionosphere_artifact",
]
