"""Immutable manifest-bound storage for Stack interferogram artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.stack.artifact_transaction import (
    ArtifactResourceLimits,
    GenerationLease,
    commit_generation,
    open_current_generation,
    stage_generation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = setup_logger(__name__)

IFG_ARTIFACT_SCHEMA = "stack_ifg_artifact_v1"
UNWRAP_ARTIFACT_SCHEMA = "stack_unwrap_artifact_v1"
_MAX_MANIFEST_BYTES = 1024 * 1024
_IFG_FILENAMES = {
    "complex_ifg": "complex_ifg.npy",
    "coherence": "coherence.npy",
    "wrapped_phase": "wrapped_phase.npy",
    "amplitude": "amplitude.npy",
}
_REQUIRED_IFG_FILENAMES = {
    name: filename for name, filename in _IFG_FILENAMES.items() if name != "coherence"
}
_VALID_MASK_FILENAME = "valid_mask.npy"
_PHASE_SCREEN_MANIFEST_FIELDS = frozenset(
    {
        "flatten_stage",
        "phase_screen_model",
        "phase_screen_digests",
        "phase_screen_domain",
        "phase_screen_grid_identity",
    }
)
_UNWRAP_FILENAMES = {
    "unwrapped_phase": "unwrapped_phase.npy",
    "connected_components": "connected_components.npy",
}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    """Serialize a manifest frame deterministically."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"artifact metadata is not canonical JSON: {error}")


def _digest_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(payload).hexdigest()


def _digest_file(path: Path) -> str:
    """Hash one artifact payload without loading it into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        reject_invalid_state(f"artifact payload cannot be hashed: {error}")
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    """Return whether a value is a canonical lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_symlink_components(path: Path) -> None:
    """Reject existing symbolic-link components in an artifact path."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                reject_invalid_state(
                    f"artifact store path contains a symbolic link: {current}"
                )
        except OSError as error:
            reject_invalid_state(f"artifact store path cannot be inspected: {error}")


def _store_root(root: str | Path, *, create: bool) -> Path:
    """Resolve and validate one caller-owned artifact directory."""
    path = Path(root)
    _reject_symlink_components(path)
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            reject_invalid_state(f"artifact store root is unsafe: {path}")
    elif create:
        try:
            # Keep the managed root private even when the caller's umask is
            # permissive (for example, a shared Linux group umask of 0002).
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.chmod(0o700)
        except OSError as error:
            reject_invalid_state(f"artifact store root cannot be created: {error}")
        _reject_symlink_components(path)
    else:
        reject_invalid_state(f"artifact store root is missing: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    """Read one bounded, regular, non-symlink JSON manifest."""
    if not path.is_file() or path.is_symlink():
        reject_invalid_state(f"artifact manifest missing or unsafe: {path}")
    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            reject_invalid_state(f"artifact manifest exceeds size limit: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        reject_invalid_state(f"artifact manifest cannot be read: {error}")
    if not isinstance(value, dict):
        reject_invalid_state("artifact manifest must be a JSON object")
    return value


def _verified_manifest(path: Path, schema: str) -> dict[str, Any]:
    """Read a complete manifest and verify its self-digest."""
    manifest = _read_json(path)
    if manifest.get("schema_version") != schema:
        reject_invalid_state("unsupported artifact manifest schema")
    if manifest.get("status") != "complete":
        reject_invalid_state("artifact manifest is not complete")
    expected = manifest.get("manifest_digest")
    unsigned = dict(manifest)
    unsigned.pop("manifest_digest", None)
    actual = _digest_bytes(_canonical_json(unsigned))
    if expected != actual:
        reject_invalid_state("artifact manifest digest mismatch")
    return manifest


def _atomic_save(path: Path, array: np.ndarray) -> None:
    """Durably publish one NumPy payload through a same-directory rename."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except OSError as error:
        reject_invalid_state(f"artifact payload publication failed: {error}")
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Publish a manifest last, after all referenced payloads are durable."""
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        reject_invalid_state(f"artifact manifest publication failed: {error}")
    finally:
        temporary.unlink(missing_ok=True)


def _array_descriptor(path: Path, array: np.ndarray) -> dict[str, Any]:
    """Build immutable metadata for one already-published array."""
    return {
        "file": path.name,
        "shape": [int(size) for size in array.shape],
        "dtype": array.dtype.str,
        "sha256": _digest_file(path),
    }


def _validate_shape(value: object, field: str) -> tuple[int, int]:
    """Decode a positive two-dimensional manifest shape."""
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in value
        )
    ):
        reject_invalid_state(f"artifact {field} must be a positive 2-D shape")
    return int(value[0]), int(value[1])


def _validate_payload_table(
    root: Path,
    raw_payloads: object,
    expected_filenames: Mapping[str, str],
    expected_shape: tuple[int, int],
) -> dict[str, dict[str, Any]]:
    """Validate payload descriptors without trusting manifest paths."""
    if not isinstance(raw_payloads, dict) or set(raw_payloads) != set(
        expected_filenames
    ):
        reject_invalid_state("artifact payload table is incomplete")
    payloads: dict[str, dict[str, Any]] = {}
    for name, filename in expected_filenames.items():
        raw = raw_payloads[name]
        if not isinstance(raw, dict) or raw.get("file") != filename:
            reject_invalid_state(f"artifact payload descriptor is invalid: {name}")
        if _validate_shape(raw.get("shape"), f"{name} shape") != expected_shape:
            reject_invalid_state(f"artifact payload shape mismatch: {name}")
        try:
            dtype = np.dtype(raw.get("dtype"))
        except (TypeError, ValueError) as error:
            reject_invalid_state(f"artifact payload dtype is invalid: {error}")
        if dtype.hasobject or not _is_sha256(raw.get("sha256")):
            reject_invalid_state(f"artifact payload metadata is unsafe: {name}")
        path = root / filename
        _reject_symlink_components(path)
        if not path.is_file() or path.is_symlink():
            reject_invalid_state(f"artifact payload missing or unsafe: {path}")
        payloads[name] = dict(raw)
    return payloads


def _read_payloads(
    root: Path,
    descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Hash, load, and revalidate every payload in a generation."""
    arrays: dict[str, np.ndarray] = {}
    for name, descriptor in descriptors.items():
        path = root / str(descriptor["file"])
        if _digest_file(path) != descriptor["sha256"]:
            reject_invalid_state(f"artifact payload digest mismatch: {name}")
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as error:
            reject_invalid_state(f"artifact payload cannot be loaded: {error}")
        expected_shape = tuple(int(size) for size in descriptor["shape"])
        if array.shape != expected_shape or array.dtype.str != descriptor["dtype"]:
            reject_invalid_state(f"artifact payload shape or dtype mismatch: {name}")
        arrays[name] = array
    return arrays


def _validate_json_mapping(value: object, field: str) -> dict[str, Any]:
    """Validate string-keyed metadata by canonical JSON round-trip."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        reject_invalid_state(f"artifact {field} must be an object")
    _canonical_json(value)
    return dict(value)


def _validate_phase_screen_digests(value: object) -> dict[str, dict[str, str]]:
    """Validate the role- and unit-keyed phase-screen digest table."""
    if not isinstance(value, dict) or any(
        role not in {"primary", "secondary"} or not isinstance(digests, dict)
        for role, digests in value.items()
    ):
        reject_invalid_state(
            "artifact phase_screen_digests must map primary/secondary to unit digests"
        )
    result: dict[str, dict[str, str]] = {}
    for role, raw_digests in value.items():
        if any(
            not isinstance(tag, str) or not tag or not _is_sha256(digest)
            for tag, digest in raw_digests.items()
        ):
            reject_invalid_state("artifact phase-screen digests are invalid")
        result[str(role)] = {
            str(tag): str(digest) for tag, digest in raw_digests.items()
        }
    return result


@dataclass(frozen=True, slots=True)
class InterferogramArtifact:
    """Validated arrays from one complete interferogram generation.

    Attributes
    ----------
    complex_ifg, wrapped_phase, amplitude : numpy.ndarray
        Hash-validated product layers sharing one grid.
    coherence : numpy.ndarray or None
        Optional hash-validated coherence layer.  A generation created without
        this layer is valid and represents unavailable coherence explicitly.

    """

    complex_ifg: np.ndarray
    coherence: np.ndarray | None
    wrapped_phase: np.ndarray
    amplitude: np.ndarray
    valid_mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class UnwrappedArtifact:
    """Validated unwrapped arrays bound to one interferogram generation.

    Attributes
    ----------
    unwrapped_phase, connected_components : numpy.ndarray
        Hash-validated unwrap result layers.
    method : str
        Unwrapping method name recorded at publication.
    method_parameters : dict[str, object]
        Canonical JSON method configuration.
    ifg_manifest_digest : str
        Exact source IFG manifest digest.

    """

    unwrapped_phase: np.ndarray
    connected_components: np.ndarray
    method: str
    method_parameters: dict[str, Any]
    ifg_manifest_digest: str


@dataclass(frozen=True, slots=True)
class InterferogramArtifactStore:
    """Read-only metadata and validated access for one Stack pair artifact.

    Attributes
    ----------
    root : pathlib.Path
        Validated artifact directory.
    pair : tuple[str, str]
        Reference and secondary acquisition identifiers.
    looks : tuple[int, int]
        Applied azimuth and range look factors.
    filter_name : str
        Applied filter name.
    filter_parameters : dict[str, object]
        Canonical JSON filter configuration.
    source_manifest_digests : dict[str, str]
        Named source scene manifest digests.
    flatten_stage : str
        Stage at which flattening was applied (or deferred).
    phase_screen_model : str, optional
        Phase-screen model/convention, when a screen was applied.
    phase_screen_digests : dict[str, dict[str, str]]
        Exact phase-screen payload digests keyed by source role and scene unit.
    phase_screen_domain : str
        Coordinate domain bound to the phase screens.
    phase_screen_grid_identity : str
        Coordinate-grid identity bound to the phase screens.
    shape : tuple[int, int]
        Shared output grid shape.
    manifest_digest : str
        SHA-256 digest of the canonical base manifest.

    """

    root: Path
    generation_id: str
    generation_root: Path
    pair: tuple[str, str]
    domain: str
    wavelength_m: float | None
    grid_identity: str
    looks: tuple[int, int]
    filter_name: str
    filter_parameters: dict[str, Any]
    source_manifest_digests: dict[str, str]
    shape: tuple[int, int]
    manifest_digest: str
    _payloads: dict[str, dict[str, Any]]
    _lease: GenerationLease
    # Keep the metadata fields optional for callers that constructed this
    # dataclass directly before IFG phase lineage was added.  ``open`` always
    # resolves them to concrete manifest-bound values.
    flatten_stage: str = "coregistration"
    phase_screen_model: str | None = None
    phase_screen_digests: dict[str, dict[str, str]] = field(default_factory=dict)
    phase_screen_domain: str = "radar"
    phase_screen_grid_identity: str = ""

    @classmethod
    def open(cls, root: str | Path) -> Self:
        """Open and validate a complete Stack interferogram generation.

        Parameters
        ----------
        root : str or pathlib.Path
            Artifact directory containing ``manifest.json``.

        Returns
        -------
        InterferogramArtifactStore
            Validated, read-only store metadata.

        """
        path = _store_root(root, create=False)
        direct_payloads = [
            path / filename
            for filename in (*_IFG_FILENAMES.values(), _VALID_MASK_FILENAME)
        ]
        if any(payload.exists() for payload in direct_payloads):
            reject_invalid_state(
                "legacy direct IFG payload layout is quarantined and cannot be read"
            )
        opened = open_current_generation(path, "ifg")
        try:
            manifest = _verified_manifest(
                opened.path / "manifest.json", IFG_ARTIFACT_SCHEMA
            )
            if manifest.get("manifest_digest") != opened.manifest_digest:
                reject_invalid_state("IFG CURRENT digest does not match its generation")
            compatibility_manifest = _verified_manifest(
                path / "manifest.json", IFG_ARTIFACT_SCHEMA
            )
            if compatibility_manifest != manifest:
                reject_invalid_state(
                    "IFG compatibility manifest does not match CURRENT"
                )
        except Exception:
            opened.lease.close()
            raise
        raw_pair = manifest.get("pair")
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or any(not isinstance(value, str) or not value for value in raw_pair)
        ):
            reject_invalid_state("artifact pair must contain two non-empty ids")
        raw_looks = manifest.get("looks")
        if (
            not isinstance(raw_looks, list)
            or len(raw_looks) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in raw_looks
            )
        ):
            reject_invalid_state("artifact looks must contain two positive integers")
        domain = manifest.get("domain", "radar")
        if domain not in {"radar", "geo"}:
            reject_invalid_state("artifact domain is invalid")
        raw_wavelength = manifest.get("wavelength_m")
        wavelength_m = None if raw_wavelength is None else float(raw_wavelength)
        if wavelength_m is not None and (
            not np.isfinite(wavelength_m) or wavelength_m <= 0.0
        ):
            reject_invalid_state("artifact wavelength_m is invalid")
        grid_identity = manifest.get("grid_identity")
        if not _is_sha256(grid_identity):
            reject_invalid_state("artifact grid_identity is invalid")
        raw_filter = manifest.get("filter")
        if (
            not isinstance(raw_filter, dict)
            or not isinstance(raw_filter.get("name"), str)
            or not raw_filter["name"]
        ):
            reject_invalid_state("artifact filter metadata is invalid")
        filter_parameters = _validate_json_mapping(
            raw_filter.get("parameters"), "filter parameters"
        )
        sources = manifest.get("source_manifest_digests")
        if (
            not isinstance(sources, dict)
            or not sources
            or any(
                not isinstance(name, str) or not name or not _is_sha256(digest)
                for name, digest in sources.items()
            )
        ):
            reject_invalid_state("artifact source manifest digests are invalid")
        missing_phase_fields = _PHASE_SCREEN_MANIFEST_FIELDS.difference(manifest)
        if missing_phase_fields:
            reject_invalid_state(
                "IFG manifest is missing explicit phase-screen lineage fields: "
                + ", ".join(sorted(missing_phase_fields))
            )
        flatten_stage = manifest["flatten_stage"]
        if not isinstance(flatten_stage, str) or flatten_stage not in {
            "coregistration",
            "interferogram",
        }:
            reject_invalid_state("artifact flatten_stage is invalid")
        phase_screen_model = manifest.get("phase_screen_model")
        if not (
            phase_screen_model is None or phase_screen_model == "nisar_ellipsoidal_v1"
        ):
            reject_invalid_state("artifact phase_screen_model is invalid")
        phase_screen_digests = _validate_phase_screen_digests(
            manifest["phase_screen_digests"]
        )
        phase_screen_domain = manifest["phase_screen_domain"]
        if not isinstance(phase_screen_domain, str) or phase_screen_domain not in {
            "radar",
            "geo",
        }:
            reject_invalid_state("artifact phase_screen_domain is invalid")
        phase_screen_grid_identity = manifest["phase_screen_grid_identity"]
        if not _is_sha256(phase_screen_grid_identity):
            reject_invalid_state("artifact phase_screen_grid_identity is invalid")
        if phase_screen_domain != domain or phase_screen_grid_identity != grid_identity:
            reject_invalid_state("artifact phase-screen grid binding differs")
        if flatten_stage == "coregistration" and (
            phase_screen_model is not None or phase_screen_digests
        ):
            reject_invalid_state(
                "coregistration IFG artifacts must not carry a phase screen"
            )
        if flatten_stage == "interferogram" and (
            phase_screen_model != "nisar_ellipsoidal_v1"
            or not phase_screen_digests
            or not any(phase_screen_digests.values())
        ):
            reject_invalid_state(
                "interferogram IFG artifacts require non-empty NISAR "
                "phase-screen lineage"
            )
        if phase_screen_model is None and phase_screen_digests:
            reject_invalid_state("phase-screen digests require a phase-screen model")
        shape = _validate_shape(manifest.get("shape"), "shape")
        raw_payloads = manifest["payloads"]
        if not isinstance(raw_payloads, dict):
            reject_invalid_state("artifact payload table is incomplete")
        payloads = _validate_payload_table(
            opened.path,
            {
                name: raw_payloads[name]
                for name in _REQUIRED_IFG_FILENAMES
                if name in raw_payloads
            },
            _REQUIRED_IFG_FILENAMES,
            shape,
        )
        if "coherence" in raw_payloads:
            payloads.update(
                _validate_payload_table(
                    opened.path,
                    {"coherence": raw_payloads["coherence"]},
                    {"coherence": _IFG_FILENAMES["coherence"]},
                    shape,
                )
            )
        if "valid_mask" in raw_payloads:
            payloads.update(
                _validate_payload_table(
                    opened.path,
                    {"valid_mask": raw_payloads["valid_mask"]},
                    {"valid_mask": _VALID_MASK_FILENAME},
                    shape,
                )
            )
        digest = str(manifest["manifest_digest"])
        return cls(
            root=path,
            generation_id=opened.generation_id,
            generation_root=opened.path,
            pair=(raw_pair[0], raw_pair[1]),
            domain=str(domain),
            wavelength_m=wavelength_m,
            grid_identity=str(grid_identity),
            looks=(raw_looks[0], raw_looks[1]),
            filter_name=str(raw_filter["name"]),
            filter_parameters=filter_parameters,
            source_manifest_digests={str(k): str(v) for k, v in sources.items()},
            flatten_stage=str(flatten_stage),
            phase_screen_model=(
                None if phase_screen_model is None else str(phase_screen_model)
            ),
            phase_screen_digests=phase_screen_digests,
            phase_screen_domain=str(phase_screen_domain),
            phase_screen_grid_identity=str(phase_screen_grid_identity),
            shape=shape,
            manifest_digest=digest,
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

    def read(self) -> InterferogramArtifact:
        """Read all IFG layers after validating hashes, shapes, and dtypes."""
        arrays = _read_payloads(self.generation_root, self._payloads)
        if arrays["complex_ifg"].dtype.kind != "c" or any(
            arrays[name].dtype.kind != "f"
            for name in ("wrapped_phase", "amplitude")
        ) or (
            arrays.get("coherence") is not None
            and arrays["coherence"].dtype.kind != "f"
        ):
            reject_invalid_state("artifact IFG layer dtypes are incompatible")
        valid_mask = arrays.get("valid_mask")
        if valid_mask is not None and valid_mask.dtype != np.bool_:
            reject_invalid_state("artifact IFG valid_mask dtype is incompatible")
        return InterferogramArtifact(
            complex_ifg=arrays["complex_ifg"],
            coherence=arrays.get("coherence"),
            wrapped_phase=arrays["wrapped_phase"],
            amplitude=arrays["amplitude"],
            valid_mask=valid_mask,
        )

    def read_unwrapped(self) -> UnwrappedArtifact:
        """Read the complete unwrap generation bound to this exact IFG."""
        if any(
            (self.root / filename).exists() for filename in _UNWRAP_FILENAMES.values()
        ):
            reject_invalid_state(
                "legacy direct unwrap payload layout is quarantined and cannot be read"
            )
        opened = open_current_generation(self.root, "unwrap")
        try:
            manifest = _verified_manifest(
                opened.path / "unwrap_manifest.json", UNWRAP_ARTIFACT_SCHEMA
            )
            if manifest.get("manifest_digest") != opened.manifest_digest:
                reject_invalid_state(
                    "unwrap CURRENT digest does not match its generation"
                )
            compatibility_manifest = _verified_manifest(
                self.root / "unwrap_manifest.json", UNWRAP_ARTIFACT_SCHEMA
            )
            if compatibility_manifest != manifest:
                reject_invalid_state(
                    "unwrap compatibility manifest does not match CURRENT"
                )
        except Exception:
            opened.lease.close()
            raise
        if manifest.get("ifg_manifest_digest") != self.manifest_digest:
            reject_invalid_state("unwrap artifact is bound to a different IFG")
        if _validate_shape(manifest.get("shape"), "unwrap shape") != self.shape:
            reject_invalid_state("unwrap artifact shape differs from its IFG")
        raw_method = manifest.get("method")
        if (
            not isinstance(raw_method, dict)
            or not isinstance(raw_method.get("name"), str)
            or not raw_method["name"]
        ):
            reject_invalid_state("unwrap method metadata is invalid")
        parameters = _validate_json_mapping(
            raw_method.get("parameters"), "unwrap method parameters"
        )
        descriptors = _validate_payload_table(
            opened.path,
            manifest.get("payloads"),
            _UNWRAP_FILENAMES,
            self.shape,
        )
        arrays = _read_payloads(opened.path, descriptors)
        opened.lease.close()
        if (
            arrays["unwrapped_phase"].dtype.kind != "f"
            or arrays["connected_components"].dtype.kind not in "iu"
        ):
            reject_invalid_state("unwrap artifact layer dtypes are incompatible")
        return UnwrappedArtifact(
            unwrapped_phase=arrays["unwrapped_phase"],
            connected_components=arrays["connected_components"],
            method=str(raw_method["name"]),
            method_parameters=parameters,
            ifg_manifest_digest=self.manifest_digest,
        )


def write_ifg_artifact(
    root: str | Path,
    *,
    pair: tuple[str, str],
    looks: tuple[int, int],
    domain: str = "radar",
    wavelength_m: float | None = None,
    grid_identity: str | None = None,
    filter_name: str,
    filter_parameters: Mapping[str, Any],
    source_manifest_digests: Mapping[str, str],
    flatten_stage: str = "coregistration",
    phase_screen_model: str | None = None,
    phase_screen_digests: Mapping[str, Mapping[str, str]] | None = None,
    phase_screen_domain: str | None = None,
    phase_screen_grid_identity: str | None = None,
    complex_ifg: np.ndarray,
    coherence: np.ndarray | None,
    wrapped_phase: np.ndarray,
    amplitude: np.ndarray,
    valid_mask: np.ndarray | None = None,
    resource_limits: ArtifactResourceLimits | None = None,
    replace_existing: bool = False,
) -> InterferogramArtifactStore:
    """Atomically persist one complete derived interferogram generation.

    Payloads are made durable first and ``manifest.json`` is published last.
    A visible manifest therefore always names a complete generation.

    Parameters
    ----------
    root : str or pathlib.Path
        Destination artifact directory.
    pair : tuple[str, str]
        Reference and secondary acquisition identifiers.
    looks : tuple[int, int]
        Applied azimuth and range look factors.
    domain : {"radar", "geo"}, optional
        Common output coordinate domain.
    wavelength_m : float, optional
        Radar wavelength used for phase-to-displacement conversion.
    grid_identity : str, optional
        Canonical source scene coordinate-grid identity. When omitted, a
        deterministic compatibility identity is derived from domain and shape.
    filter_name : str
        Applied filter name, or ``"none"``.
    filter_parameters : mapping
        Canonical JSON parameters for the filter.
    source_manifest_digests : mapping[str, str]
        Named SHA-256 digests of every source scene manifest.
    flatten_stage : {"coregistration", "interferogram"}, optional
        Stage at which flattening was applied (or deferred).
    phase_screen_model : {None, "nisar_ellipsoidal_v1"}, optional
        Phase-screen model/convention.
    phase_screen_digests : mapping[str, mapping[str, str]], optional
        Exact source-role and scene-unit phase-screen payload digests.
    phase_screen_domain, phase_screen_grid_identity : str, optional
        Coordinate-domain and grid identity binding for the phase screen.
        Defaults to the IFG domain and grid identity.
    resource_limits : ArtifactResourceLimits, optional
        Hard publication size, file-count, and free-space limits.
    replace_existing : bool, optional
        Publish a new immutable generation and advance ``CURRENT`` when true.
    complex_ifg, wrapped_phase, amplitude : numpy.ndarray
        Matching two-dimensional required IFG product layers.
    coherence : numpy.ndarray, optional
        Optional matching coherence layer.  ``None`` preserves the explicit
        absence of coherence for a downstream spatial unwrapper.
    valid_mask : numpy.ndarray, optional
        Boolean authoritative support mask. If omitted, finite complex support
        is used when publishing the artifact.

    Returns
    -------
    InterferogramArtifactStore
        Reopened, validated artifact store.

    """
    arrays = {
        "complex_ifg": np.asarray(complex_ifg),
        "wrapped_phase": np.asarray(wrapped_phase),
        "amplitude": np.asarray(amplitude),
    }
    if coherence is not None:
        arrays["coherence"] = np.asarray(coherence)
    if valid_mask is not None:
        arrays["valid_mask"] = np.asarray(valid_mask)
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        reject_invalid_state("IFG artifact layers must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 2 or any(size <= 0 for size in shape):
        reject_invalid_state("IFG artifact layers must be non-empty 2-D arrays")
    if arrays["complex_ifg"].dtype.kind != "c" or any(
        arrays[name].dtype.kind != "f"
        for name in ("wrapped_phase", "amplitude")
    ) or (
        coherence is not None and arrays["coherence"].dtype.kind != "f"
    ):
        reject_invalid_state("IFG artifact layer dtypes are incompatible")
    if valid_mask is not None and arrays["valid_mask"].dtype != np.bool_:
        reject_invalid_state("IFG artifact valid_mask must be boolean")
    if (
        len(pair) != 2
        or any(not isinstance(value, str) or not value for value in pair)
        or len(looks) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in looks
        )
        or not isinstance(filter_name, str)
        or not filter_name
        or domain not in {"radar", "geo"}
        or not isinstance(flatten_stage, str)
        or flatten_stage not in {"coregistration", "interferogram"}
    ):
        reject_invalid_state("IFG artifact processing metadata is invalid")
    if wavelength_m is not None and (
        not np.isfinite(wavelength_m) or wavelength_m <= 0.0
    ):
        reject_invalid_state("IFG artifact wavelength_m is invalid")
    if domain == "geo" and grid_identity is None:
        reject_invalid_state("Geo IFG publication requires an explicit grid_identity")
    resolved_grid_identity = grid_identity or _digest_bytes(
        _canonical_json(
            {
                "domain": domain,
                "shape": [int(shape[0]), int(shape[1])],
            }
        )
    )
    if not _is_sha256(resolved_grid_identity):
        reject_invalid_state("IFG artifact grid_identity is invalid")
    resolved_phase_screen_domain = (
        domain if phase_screen_domain is None else phase_screen_domain
    )
    resolved_phase_screen_grid_identity = (
        resolved_grid_identity
        if phase_screen_grid_identity is None
        else phase_screen_grid_identity
    )
    if not isinstance(resolved_phase_screen_domain, str) or (
        resolved_phase_screen_domain not in {"radar", "geo"}
    ):
        reject_invalid_state("IFG phase_screen_domain is invalid")
    if resolved_phase_screen_domain != domain:
        reject_invalid_state("IFG phase-screen domain differs from IFG domain")
    if not _is_sha256(resolved_phase_screen_grid_identity):
        reject_invalid_state("IFG phase_screen_grid_identity is invalid")
    if resolved_phase_screen_grid_identity != resolved_grid_identity:
        reject_invalid_state("IFG phase-screen grid differs from IFG grid")
    screen_digests = _validate_phase_screen_digests(phase_screen_digests or {})
    if not (phase_screen_model is None or phase_screen_model == "nisar_ellipsoidal_v1"):
        reject_invalid_state("IFG phase_screen_model is invalid")
    if flatten_stage == "coregistration" and (
        phase_screen_model is not None or screen_digests
    ):
        reject_invalid_state(
            "coregistration IFG artifacts must not carry a phase screen"
        )
    if flatten_stage == "interferogram" and (
        phase_screen_model != "nisar_ellipsoidal_v1"
        or not screen_digests
        or not any(screen_digests.values())
    ):
        reject_invalid_state(
            "interferogram IFG artifacts require non-empty NISAR phase-screen lineage"
        )
    if phase_screen_model is None and screen_digests:
        reject_invalid_state("phase-screen digests require a phase-screen model")
    parameters = dict(filter_parameters)
    _canonical_json(parameters)
    sources = dict(source_manifest_digests)
    if not sources or any(
        not isinstance(name, str) or not name or not _is_sha256(digest)
        for name, digest in sources.items()
    ):
        reject_invalid_state("IFG artifact source manifest digests are invalid")

    path = _store_root(root, create=True)
    estimated_bytes = sum(int(array.nbytes) + 1024 for array in arrays.values())
    with stage_generation(
        path,
        "ifg",
        final_bytes=estimated_bytes,
        temporary_bytes=estimated_bytes,
        file_count=len(arrays) + 1,
        dimensions=(int(shape[0]), int(shape[1])),
        limits=resource_limits,
    ) as (generation_id, staging):
        if not replace_existing and (
            (path / "CURRENT").exists() or (path / "manifest.json").exists()
        ):
            reject_invalid_state("IFG artifact generation is already published")
        descriptors: dict[str, dict[str, Any]] = {}
        for name, filename in _IFG_FILENAMES.items():
            if name not in arrays:
                continue
            payload_path = staging / filename
            _atomic_save(payload_path, arrays[name])
            descriptors[name] = _array_descriptor(payload_path, arrays[name])
        if valid_mask is not None:
            payload_path = staging / _VALID_MASK_FILENAME
            _atomic_save(payload_path, arrays["valid_mask"])
            descriptors["valid_mask"] = _array_descriptor(
                payload_path, arrays["valid_mask"]
            )
        unsigned: dict[str, Any] = {
            "schema_version": IFG_ARTIFACT_SCHEMA,
            "status": "complete",
            "generation_id": generation_id,
            "pair": list(pair),
            "looks": list(looks),
            "domain": domain,
            "wavelength_m": wavelength_m,
            "grid_identity": resolved_grid_identity,
            "flatten_stage": flatten_stage,
            "phase_screen_model": phase_screen_model,
            "phase_screen_digests": screen_digests,
            "phase_screen_domain": resolved_phase_screen_domain,
            "phase_screen_grid_identity": resolved_phase_screen_grid_identity,
            "filter": {"name": filter_name, "parameters": parameters},
            "source_manifest_digests": sources,
            "shape": [int(shape[0]), int(shape[1])],
            "payloads": descriptors,
        }
        manifest = {
            **unsigned,
            "manifest_digest": _digest_bytes(_canonical_json(unsigned)),
        }
        _atomic_manifest(staging / "manifest.json", manifest)
        commit_generation(
            path,
            "ifg",
            generation_id,
            staging,
            manifest_digest=str(manifest["manifest_digest"]),
            compatibility_manifest=manifest,
        )
    logger.info("Published IFG artifact %s for pair %s", path, pair)
    return InterferogramArtifactStore.open(path)


def write_unwrapped_artifact(
    root: str | Path,
    *,
    unwrapped_phase: np.ndarray,
    connected_components: np.ndarray,
    method: str,
    method_parameters: Mapping[str, Any],
    ifg_manifest_digest: str,
    resource_limits: ArtifactResourceLimits | None = None,
    replace_existing: bool = False,
) -> None:
    """Atomically persist unwrap layers bound to one exact IFG generation.

    Parameters
    ----------
    root : str or pathlib.Path
        Existing complete IFG artifact directory.
    unwrapped_phase : numpy.ndarray
        Floating-point unwrapped phase in radians.
    connected_components : numpy.ndarray
        Integer connected-component labels.
    method : str
        Unwrapping backend or method name.
    method_parameters : mapping
        Canonical JSON parameters for the unwrapping method.
    ifg_manifest_digest : str
        Digest returned by :class:`InterferogramArtifactStore`.
    resource_limits : ArtifactResourceLimits, optional
        Hard publication size, file-count, and free-space limits.
    replace_existing : bool, optional
        Publish a new immutable unwrap generation when true.

    """
    store = InterferogramArtifactStore.open(root)
    if ifg_manifest_digest != store.manifest_digest:
        reject_invalid_state("unwrap artifact IFG digest binding is invalid")
    phase = np.asarray(unwrapped_phase)
    components = np.asarray(connected_components)
    if (
        phase.shape != store.shape
        or components.shape != store.shape
        or phase.dtype.kind != "f"
        or components.dtype.kind not in "iu"
    ):
        reject_invalid_state("unwrap layers must match the IFG shape and dtypes")
    if not isinstance(method, str) or not method:
        reject_invalid_state("unwrap method must not be empty")
    parameters = dict(method_parameters)
    _canonical_json(parameters)
    arrays = {
        "unwrapped_phase": phase,
        "connected_components": components,
    }
    estimated_bytes = sum(int(array.nbytes) + 1024 for array in arrays.values())
    with stage_generation(
        store.root,
        "unwrap",
        final_bytes=estimated_bytes,
        temporary_bytes=estimated_bytes,
        file_count=len(arrays) + 1,
        dimensions=store.shape,
        limits=resource_limits,
    ) as (generation_id, staging):
        if not replace_existing and (
            (store.root / "UNWRAP_CURRENT").exists()
            or (store.root / "unwrap_manifest.json").exists()
        ):
            reject_invalid_state("unwrap artifact generation is already published")
        descriptors: dict[str, dict[str, Any]] = {}
        for name, filename in _UNWRAP_FILENAMES.items():
            payload_path = staging / filename
            _atomic_save(payload_path, arrays[name])
            descriptors[name] = _array_descriptor(payload_path, arrays[name])
        unsigned: dict[str, Any] = {
            "schema_version": UNWRAP_ARTIFACT_SCHEMA,
            "status": "complete",
            "generation_id": generation_id,
            "ifg_generation_id": store.generation_id,
            "ifg_manifest_digest": store.manifest_digest,
            "shape": list(store.shape),
            "method": {"name": method, "parameters": parameters},
            "payloads": descriptors,
        }
        manifest = {
            **unsigned,
            "manifest_digest": _digest_bytes(_canonical_json(unsigned)),
        }
        _atomic_manifest(staging / "unwrap_manifest.json", manifest)
        commit_generation(
            store.root,
            "unwrap",
            generation_id,
            staging,
            manifest_digest=str(manifest["manifest_digest"]),
            compatibility_manifest=manifest,
        )
    store.close()
    logger.info("Published unwrap artifact %s with method %s", store.root, method)


__all__ = [
    "IFG_ARTIFACT_SCHEMA",
    "UNWRAP_ARTIFACT_SCHEMA",
    "ArtifactResourceLimits",
    "InterferogramArtifact",
    "InterferogramArtifactStore",
    "UnwrappedArtifact",
    "write_ifg_artifact",
    "write_unwrapped_artifact",
]
