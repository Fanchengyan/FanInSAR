"""Atomic parent generations for complete Stack-derived result sets."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.stack.artifact_transaction import (
    GenerationLease,
    canonical_json,
    commit_generation,
    open_current_generation,
    sha256_bytes,
    stage_generation,
)
from faninsar.processing.timeseries.inversion import open_timeseries_zarr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.processing.stack.ifg_store import InterferogramArtifactStore

logger = setup_logger(__name__)

STACK_GENERATION_SCHEMA = "stack_result_generation_v1"
UNWRAP_GENERATION_SCHEMA = "stack_unwrap_generation_v1"
_MAX_MANIFEST_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PairGenerationBinding:
    """Immutable IFG and unwrap generation identities for one Stack pair.

    Attributes
    ----------
    pair_id : str
        Canonical ``primary_secondary`` pair identifier.
    artifact_root : pathlib.Path
        Pair transaction root inside the Stack work directory.
    ifg_generation_id, unwrap_generation_id : str
        Exact immutable child generation identifiers.
    ifg_manifest_digest, unwrap_manifest_digest : str
        SHA-256 identities of the bound child manifests.

    """

    pair_id: str
    artifact_root: Path
    ifg_generation_id: str
    ifg_manifest_digest: str
    unwrap_generation_id: str
    unwrap_manifest_digest: str


@dataclass(frozen=True, slots=True)
class StackResultGeneration:
    """Pinned, validated parent generation for a complete Stack result set.

    Attributes
    ----------
    root, generation_root : pathlib.Path
        Stack transaction root and selected immutable generation directory.
    generation_id, manifest_digest : str
        Parent generation and canonical manifest identities.
    pair_ids : tuple[str, ...]
        Exact configured pair network in Stack order.
    pairs : tuple[PairGenerationBinding, ...]
        Immutable IFG and unwrap bindings for every expected pair.
    timeseries_root : pathlib.Path
        Bound time-series transaction root.
    timeseries_generation_id, timeseries_manifest_digest : str
        Exact immutable SBAS child identity.

    """

    root: Path
    generation_id: str
    generation_root: Path
    manifest_digest: str
    pair_ids: tuple[str, ...]
    pairs: tuple[PairGenerationBinding, ...]
    timeseries_root: Path
    timeseries_generation_id: str
    timeseries_manifest_digest: str
    _lease: GenerationLease

    def close(self) -> None:
        """Release the parent generation reader pin."""
        self._lease.close()

    def __enter__(self) -> Self:
        """Return this pinned generation as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the parent generation reader pin."""
        self.close()


@dataclass(frozen=True, slots=True)
class UnwrapResultGeneration:
    """One complete Stack-root spatial unwrap result generation.

    Parameters
    ----------
    root : pathlib.Path
        Stack transaction root.
    generation_id, generation_root, manifest_digest : object
        Immutable generation identity and selected directory.
    pair_ids : tuple[str, ...]
        Complete ordered Pair universe captured at call entry.
    products : dict[str, dict[str, numpy.ndarray]]
        Eager named result layers keyed by canonical Pair id.

    Notes
    -----
    Payloads are decoded from the immutable generation selected by
    ``UNWRAP_CURRENT``.  Normal reads validate metadata and shape but do not
    recompute payload SHA-256 digests.

    """

    root: Path
    generation_id: str
    generation_root: Path
    manifest_digest: str
    pair_ids: tuple[str, ...]
    products: dict[str, dict[str, np.ndarray]]
    _lease: GenerationLease

    def close(self) -> None:
        """Release the generation reader lease."""
        self._lease.close()

    def __enter__(self) -> Self:
        """Return this generation as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the generation reader lease."""
        self.close()


def _read_manifest(path: Path, schema: str) -> dict[str, Any]:
    """Read and self-verify one bounded immutable child manifest."""
    try:
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size > _MAX_MANIFEST_BYTES
        ):
            reject_invalid_state(
                f"Stack generation manifest is missing or unsafe: {path}"
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        reject_invalid_state(f"Stack generation manifest cannot be read: {error}")
    if not isinstance(manifest, dict):
        reject_invalid_state("Stack generation manifest must be an object")
    unsigned = dict(manifest)
    expected_digest = unsigned.pop("manifest_digest", None)
    if (
        manifest.get("schema_version") != schema
        or manifest.get("status") != "complete"
        or expected_digest != sha256_bytes(canonical_json(unsigned))
    ):
        reject_invalid_state("Stack generation manifest is incomplete or tampered")
    return manifest


def _is_lower_hex(value: object, length: int) -> bool:
    """Return whether a value is fixed-length canonical lowercase hex."""
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_symlink_components(path: Path) -> None:
    """Reject symbolic links in every existing component of a child root."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.exists() and current.is_symlink():
            reject_invalid_state(
                f"Stack child artifact path contains a symbolic link: {current}"
            )


def _relative_artifact_root(stack_root: Path, artifact_root: Path) -> str:
    """Return one safe artifact root relative to the Stack transaction root."""
    try:
        relative = artifact_root.absolute().relative_to(stack_root.absolute())
    except ValueError:
        reject_invalid_state("Stack child artifacts must be inside the Stack work root")
    if not relative.parts or ".." in relative.parts:
        reject_invalid_state("Stack child artifact path is unsafe")
    return relative.as_posix()


def _decode_relative_root(stack_root: Path, value: object) -> Path:
    """Decode a manifest-relative child root without allowing traversal."""
    if not isinstance(value, str) or not value:
        reject_invalid_state("Stack child artifact root must be a relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        reject_invalid_state("Stack child artifact root is unsafe")
    path = stack_root / relative
    _reject_symlink_components(path)
    if not path.is_dir() or path.is_symlink():
        reject_invalid_state("Stack child artifact root is missing or unsafe")
    return path


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write and durably flush a new generation manifest."""
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json(manifest) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_array(path: Path, value: object) -> np.ndarray:
    """Write one non-object result array and return an owned copy."""
    array = np.asarray(value)
    if array.dtype.hasobject or array.ndim == 0 or array.ndim > 2:
        reject_invalid_state("Stack unwrap result layers must be 1-D or 2-D arrays")
    try:
        with path.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError) as error:
        reject_invalid_state(f"Stack unwrap result cannot be written: {error}")
    return array.copy()


def _unwrap_payload_descriptor(path: Path, array: np.ndarray) -> dict[str, Any]:
    """Describe one published layer without a read-time content hash."""
    return {
        "file": path.name,
        "shape": [int(size) for size in array.shape],
        "dtype": array.dtype.str,
    }


def publish_unwrap_generation(
    stack_root: str | Path,
    *,
    pair_ids: Sequence[str],
    products: Mapping[str, Mapping[str, object]],
) -> UnwrapResultGeneration:
    """Atomically publish one complete Stack-root unwrap result set.

    Parameters
    ----------
    stack_root : str or pathlib.Path
        Stack work directory owning ``UNWRAP_CURRENT``.
    pair_ids : sequence of str
        Complete ordered Pair universe frozen by the Stack call.
    products : mapping[str, mapping[str, object]]
        Result layers for every Pair, stored below canonical Pair directories.

    Returns
    -------
    UnwrapResultGeneration
        The newly committed generation reopened through ``UNWRAP_CURRENT``.

    Raises
    ------
    InvalidProcessingStateError
        If Pair coverage or result layer declarations are invalid.

    Notes
    -----
    The root manifest records shape and dtype only for result payloads.
    Publication computes the manifest identity once; normal reads do not
    recalculate payload SHA-256 digests.

    """
    root = Path(stack_root)
    ordered_pairs = tuple(pair_ids)
    if (
        not ordered_pairs
        or any(not isinstance(pair_id, str) or not pair_id for pair_id in ordered_pairs)
        or len(set(ordered_pairs)) != len(ordered_pairs)
    ):
        reject_invalid_state("Stack unwrap generation requires unique Pair ids")
    if set(products) != set(ordered_pairs):
        reject_invalid_state("Stack unwrap generation Pair results are incomplete")

    normalized: dict[str, dict[str, np.ndarray]] = {}
    final_bytes = 1024
    file_count = 1
    for pair_id in ordered_pairs:
        raw_layers = products[pair_id]
        if not isinstance(raw_layers, Mapping) or not raw_layers:
            reject_invalid_state(f"Stack unwrap result is empty: {pair_id}")
        layers: dict[str, np.ndarray] = {}
        for name, value in raw_layers.items():
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                reject_invalid_state("Stack unwrap result layer name is invalid")
            array = np.asarray(value)
            if array.dtype.hasobject or array.ndim == 0 or array.ndim > 2:
                reject_invalid_state(
                    "Stack unwrap result layer shape or dtype is invalid"
                )
            final_bytes += int(array.nbytes) + 1024
            file_count += 1
            layers[name] = array
        normalized[pair_id] = layers

    with stage_generation(
        root,
        "unwrap",
        final_bytes=final_bytes,
        temporary_bytes=final_bytes,
        file_count=file_count,
    ) as (generation_id, staging):
        manifest_pairs: list[dict[str, Any]] = []
        for pair_id in ordered_pairs:
            pair_root = staging / pair_id
            pair_root.mkdir(mode=0o700)
            payloads: dict[str, dict[str, Any]] = {}
            for name, value in normalized[pair_id].items():
                payload_path = pair_root / f"{name}.npy"
                array = _write_array(payload_path, value)
                payloads[name] = _unwrap_payload_descriptor(payload_path, array)
            manifest_pairs.append({"pair_id": pair_id, "payloads": payloads})
        unsigned: dict[str, Any] = {
            "schema_version": UNWRAP_GENERATION_SCHEMA,
            "status": "complete",
            "generation_id": generation_id,
            "pair_ids": list(ordered_pairs),
            "pairs": manifest_pairs,
        }
        manifest = {
            **unsigned,
            "manifest_digest": sha256_bytes(canonical_json(unsigned)),
        }
        _write_manifest(staging / "unwrap_manifest.json", manifest)
        commit_generation(
            root,
            "unwrap",
            generation_id,
            staging,
            manifest_digest=str(manifest["manifest_digest"]),
            compatibility_manifest=manifest,
        )
    return open_unwrap_generation(root)


def open_unwrap_generation(stack_root: str | Path) -> UnwrapResultGeneration:
    """Open and decode the complete generation selected by ``UNWRAP_CURRENT``.

    Parameters
    ----------
    stack_root : str or pathlib.Path
        Stack work directory containing a committed ``UNWRAP_CURRENT``.

    Returns
    -------
    UnwrapResultGeneration
        Pinned immutable root generation.  Call :meth:`close` when finished.

    Raises
    ------
    InvalidProcessingStateError
        If the root manifest, Pair coverage, or payload shape/dtype is invalid.

    """
    root = Path(stack_root)
    opened = open_current_generation(root, "unwrap")
    try:
        manifest = _read_manifest(
            opened.path / "unwrap_manifest.json", UNWRAP_GENERATION_SCHEMA
        )
        compatibility = _read_manifest(
            root / "unwrap_manifest.json", UNWRAP_GENERATION_SCHEMA
        )
        if (
            manifest != compatibility
            or manifest.get("generation_id") != opened.generation_id
            or manifest.get("manifest_digest") != opened.manifest_digest
        ):
            reject_invalid_state("Stack unwrap CURRENT and manifest identities differ")
        raw_pair_ids = manifest.get("pair_ids")
        if not isinstance(raw_pair_ids, list) or any(
            not isinstance(pair_id, str) or not pair_id for pair_id in raw_pair_ids
        ):
            reject_invalid_state("Stack unwrap generation Pair ids are invalid")
        pair_ids = tuple(raw_pair_ids)
        if not pair_ids or len(set(pair_ids)) != len(pair_ids):
            reject_invalid_state("Stack unwrap generation Pair ids are not unique")
        raw_pairs = manifest.get("pairs")
        if not isinstance(raw_pairs, list) or len(raw_pairs) != len(pair_ids):
            reject_invalid_state("Stack unwrap generation Pair results are incomplete")
        products: dict[str, dict[str, np.ndarray]] = {}
        for pair_id, raw_pair in zip(pair_ids, raw_pairs, strict=True):
            if not isinstance(raw_pair, dict) or raw_pair.get("pair_id") != pair_id:
                reject_invalid_state("Stack unwrap generation Pair order is invalid")
            raw_payloads = raw_pair.get("payloads")
            if not isinstance(raw_payloads, dict) or not raw_payloads:
                reject_invalid_state("Stack unwrap generation payload table is empty")
            pair_root = opened.path / pair_id
            layers: dict[str, np.ndarray] = {}
            for name, descriptor in raw_payloads.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(descriptor, dict)
                    or descriptor.get("file") != f"{name}.npy"
                    or not isinstance(descriptor.get("shape"), list)
                    or not isinstance(descriptor.get("dtype"), str)
                ):
                    reject_invalid_state("Stack unwrap payload descriptor is invalid")
                try:
                    expected_dtype = np.dtype(descriptor["dtype"])
                    expected_shape = tuple(int(size) for size in descriptor["shape"])
                except (TypeError, ValueError):
                    reject_invalid_state("Stack unwrap payload descriptor is invalid")
                if expected_dtype.hasobject or not expected_shape or any(
                    size <= 0 for size in expected_shape
                ):
                    reject_invalid_state(
                        "Stack unwrap payload shape or dtype is invalid"
                    )
                try:
                    array = np.load(
                        pair_root / str(descriptor["file"]), allow_pickle=False
                    )
                except (OSError, ValueError, EOFError) as error:
                    reject_invalid_state(
                        f"Stack unwrap payload cannot be read: {error}"
                    )
                if array.shape != expected_shape or array.dtype != expected_dtype:
                    reject_invalid_state(
                        "Stack unwrap payload shape or dtype mismatch"
                    )
                layers[name] = np.asarray(array)
            products[pair_id] = layers
    except Exception:
        opened.lease.close()
        raise
    return UnwrapResultGeneration(
        root=root,
        generation_id=opened.generation_id,
        generation_root=opened.path,
        manifest_digest=opened.manifest_digest,
        pair_ids=pair_ids,
        products=products,
        _lease=opened.lease,
    )


def _timeseries_pair_ids(generation_root: Path) -> tuple[str, ...]:
    """Read the pair identity embedded in one hash-validated Zarr generation."""
    import zarr

    try:
        group = zarr.open_group(str(generation_root), mode="r")
        raw_pair_ids = group.attrs["pair_ids"]
    except (KeyError, OSError, ValueError) as error:
        reject_invalid_state(f"time-series pair identity cannot be read: {error}")
    if not isinstance(raw_pair_ids, list) or any(
        not isinstance(pair_id, str) or not pair_id for pair_id in raw_pair_ids
    ):
        reject_invalid_state("time-series pair identity is invalid")
    return tuple(raw_pair_ids)


def _pair_binding_payload(
    stack_root: Path,
    store: InterferogramArtifactStore,
    expected_pair_id: str,
) -> dict[str, str]:
    """Validate and encode one exact IFG/unwrap child-generation binding."""
    actual_pair_id = f"{store.pair[0]}_{store.pair[1]}"
    if actual_pair_id != expected_pair_id:
        reject_invalid_state("Stack IFG pair order differs from the requested network")
    if not _is_lower_hex(store.generation_id, 32) or not _is_lower_hex(
        store.manifest_digest, 64
    ):
        reject_invalid_state("Stack IFG generation identity is invalid")
    store.read()
    store.read_unwrapped()
    ifg_manifest = _read_manifest(
        store.generation_root / "manifest.json", "stack_ifg_artifact_v1"
    )
    unwrap_manifest = _read_manifest(
        store.root / "unwrap_manifest.json", "stack_unwrap_artifact_v1"
    )
    if (
        ifg_manifest.get("generation_id") != store.generation_id
        or ifg_manifest.get("manifest_digest") != store.manifest_digest
        or unwrap_manifest.get("ifg_generation_id") != store.generation_id
        or unwrap_manifest.get("ifg_manifest_digest") != store.manifest_digest
    ):
        reject_invalid_state("Stack unwrap generation is not bound to its exact IFG")
    unwrap_generation_id = unwrap_manifest.get("generation_id")
    unwrap_manifest_digest = unwrap_manifest.get("manifest_digest")
    if not isinstance(unwrap_generation_id, str) or not isinstance(
        unwrap_manifest_digest, str
    ):
        reject_invalid_state("Stack unwrap generation identity is invalid")
    if not _is_lower_hex(unwrap_generation_id, 32) or not _is_lower_hex(
        unwrap_manifest_digest, 64
    ):
        reject_invalid_state("Stack unwrap generation identity is invalid")
    return {
        "pair_id": actual_pair_id,
        "artifact_root": _relative_artifact_root(stack_root, store.root),
        "ifg_generation_id": store.generation_id,
        "ifg_manifest_digest": store.manifest_digest,
        "unwrap_generation_id": unwrap_generation_id,
        "unwrap_manifest_digest": unwrap_manifest_digest,
    }


def publish_stack_generation(
    stack_root: str | Path,
    *,
    expected_pair_ids: Sequence[str],
    stores: Sequence[InterferogramArtifactStore],
    timeseries_root: str | Path,
) -> StackResultGeneration:
    """Atomically publish one parent binding a complete derived Stack result set.

    Parameters
    ----------
    stack_root : str or pathlib.Path
        Stack work directory that owns every derived child artifact.
    expected_pair_ids : sequence of str
        Exact ordered pair network required for completeness.
    stores : sequence of InterferogramArtifactStore
        Pinned, validated current IFG generations for the pair network.
    timeseries_root : str or pathlib.Path
        Immutable time-series transaction root produced from the same network.

    Returns
    -------
    StackResultGeneration
        Reopened and validated current parent generation.

    """
    root = Path(stack_root)
    pair_ids = tuple(expected_pair_ids)
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        reject_invalid_state("Stack generation requires unique expected pairs")
    if len(stores) != len(pair_ids):
        reject_invalid_state("Stack generation IFG set is incomplete")
    pair_payloads = [
        _pair_binding_payload(root, store, expected_pair_id)
        for store, expected_pair_id in zip(stores, pair_ids, strict=True)
    ]
    timeseries_path = Path(timeseries_root)
    relative_timeseries_root = _relative_artifact_root(root, timeseries_path)
    with open_timeseries_zarr(timeseries_path) as timeseries_store:
        timeseries_pair_ids = _timeseries_pair_ids(timeseries_store.path)
        if len(timeseries_pair_ids) != len(pair_ids) or set(timeseries_pair_ids) != set(
            pair_ids
        ):
            reject_invalid_state(
                "time-series generation does not match the exact Stack pair network"
            )
        timeseries_generation_id = timeseries_store.generation_id
        timeseries_manifest_digest = timeseries_store.manifest_digest
        if not _is_lower_hex(timeseries_generation_id, 32) or not _is_lower_hex(
            timeseries_manifest_digest, 64
        ):
            reject_invalid_state("Stack time-series generation identity is invalid")
    with stage_generation(
        root,
        "stack",
        final_bytes=_MAX_MANIFEST_BYTES,
        temporary_bytes=_MAX_MANIFEST_BYTES,
        file_count=1,
    ) as (generation_id, staging):
        unsigned: dict[str, Any] = {
            "schema_version": STACK_GENERATION_SCHEMA,
            "status": "complete",
            "generation_id": generation_id,
            "pair_ids": list(pair_ids),
            "pairs": pair_payloads,
            "timeseries": {
                "artifact_root": relative_timeseries_root,
                "generation_id": timeseries_generation_id,
                "manifest_digest": timeseries_manifest_digest,
            },
        }
        manifest = {
            **unsigned,
            "manifest_digest": sha256_bytes(canonical_json(unsigned)),
        }
        _write_manifest(staging / "stack_manifest.json", manifest)
        commit_generation(
            root,
            "stack",
            generation_id,
            staging,
            manifest_digest=str(manifest["manifest_digest"]),
            compatibility_manifest=manifest,
        )
    logger.info("Published complete Stack result generation %s", generation_id)
    return open_stack_generation(root)


def _decode_pair_bindings(
    stack_root: Path,
    raw_bindings: object,
    pair_ids: tuple[str, ...],
) -> tuple[PairGenerationBinding, ...]:
    """Decode and validate every immutable pair child named by a parent."""
    from faninsar.processing.stack.ifg_store import InterferogramArtifactStore

    if not isinstance(raw_bindings, list) or len(raw_bindings) != len(pair_ids):
        reject_invalid_state("Stack generation pair bindings are incomplete")
    bindings: list[PairGenerationBinding] = []
    for expected_pair_id, raw in zip(pair_ids, raw_bindings, strict=True):
        if not isinstance(raw, dict) or raw.get("pair_id") != expected_pair_id:
            reject_invalid_state("Stack generation pair binding order is invalid")
        artifact_root = _decode_relative_root(stack_root, raw.get("artifact_root"))
        ifg_generation_id = raw.get("ifg_generation_id")
        ifg_manifest_digest = raw.get("ifg_manifest_digest")
        unwrap_generation_id = raw.get("unwrap_generation_id")
        unwrap_manifest_digest = raw.get("unwrap_manifest_digest")
        if not (
            _is_lower_hex(ifg_generation_id, 32)
            and _is_lower_hex(ifg_manifest_digest, 64)
            and _is_lower_hex(unwrap_generation_id, 32)
            and _is_lower_hex(unwrap_manifest_digest, 64)
        ):
            reject_invalid_state("Stack pair generation identity is invalid")
        ifg_manifest = _read_manifest(
            artifact_root
            / ".ifg_generations"
            / str(ifg_generation_id)
            / "manifest.json",
            "stack_ifg_artifact_v1",
        )
        unwrap_manifest = _read_manifest(
            artifact_root
            / ".unwrap_generations"
            / str(unwrap_generation_id)
            / "unwrap_manifest.json",
            "stack_unwrap_artifact_v1",
        )
        if (
            ifg_manifest.get("generation_id") != ifg_generation_id
            or ifg_manifest.get("manifest_digest") != ifg_manifest_digest
            or unwrap_manifest.get("generation_id") != unwrap_generation_id
            or unwrap_manifest.get("manifest_digest") != unwrap_manifest_digest
            or unwrap_manifest.get("ifg_generation_id") != ifg_generation_id
            or unwrap_manifest.get("ifg_manifest_digest") != ifg_manifest_digest
        ):
            reject_invalid_state("Stack pair child generation binding is invalid")
        with InterferogramArtifactStore.open(artifact_root) as store:
            if (
                store.generation_id != ifg_generation_id
                or store.manifest_digest != ifg_manifest_digest
            ):
                reject_invalid_state(
                    "Stack pair CURRENT differs from the bound IFG generation"
                )
            store.read()
            store.read_unwrapped()
        bindings.append(
            PairGenerationBinding(
                pair_id=expected_pair_id,
                artifact_root=artifact_root,
                ifg_generation_id=str(ifg_generation_id),
                ifg_manifest_digest=str(ifg_manifest_digest),
                unwrap_generation_id=str(unwrap_generation_id),
                unwrap_manifest_digest=str(unwrap_manifest_digest),
            )
        )
    return tuple(bindings)


def open_stack_generation(stack_root: str | Path) -> StackResultGeneration:
    """Open and validate the current complete Stack parent and every child.

    Parameters
    ----------
    stack_root : str or pathlib.Path
        Stack work directory containing ``STACK_CURRENT``.

    Returns
    -------
    StackResultGeneration
        Pinned current parent after all referenced child payloads validate.

    """
    root = Path(stack_root)
    opened = open_current_generation(root, "stack")
    try:
        if not _is_lower_hex(opened.generation_id, 32) or not _is_lower_hex(
            opened.manifest_digest, 64
        ):
            reject_invalid_state("Stack parent generation identity is invalid")
        manifest = _read_manifest(
            opened.path / "stack_manifest.json", STACK_GENERATION_SCHEMA
        )
        compatibility = _read_manifest(
            opened.root / "stack_manifest.json", STACK_GENERATION_SCHEMA
        )
        if (
            manifest != compatibility
            or manifest.get("generation_id") != opened.generation_id
            or manifest.get("manifest_digest") != opened.manifest_digest
        ):
            reject_invalid_state("Stack CURRENT and parent manifest identities differ")
        raw_pair_ids = manifest.get("pair_ids")
        if not isinstance(raw_pair_ids, list) or any(
            not isinstance(pair_id, str) or not pair_id for pair_id in raw_pair_ids
        ):
            reject_invalid_state("Stack generation pair network is invalid")
        pair_ids = tuple(raw_pair_ids)
        if not pair_ids or len(set(pair_ids)) != len(pair_ids):
            reject_invalid_state("Stack generation pair network is not unique")
        pairs = _decode_pair_bindings(opened.root, manifest.get("pairs"), pair_ids)
        raw_timeseries = manifest.get("timeseries")
        if not isinstance(raw_timeseries, dict):
            reject_invalid_state("Stack generation time-series binding is missing")
        timeseries_root = _decode_relative_root(
            opened.root, raw_timeseries.get("artifact_root")
        )
        generation_id = raw_timeseries.get("generation_id")
        manifest_digest = raw_timeseries.get("manifest_digest")
        if not _is_lower_hex(generation_id, 32) or not _is_lower_hex(
            manifest_digest, 64
        ):
            reject_invalid_state("Stack time-series generation identity is invalid")
        timeseries_manifest = _read_manifest(
            timeseries_root
            / ".timeseries_generations"
            / generation_id
            / "artifact_manifest.json",
            "stack_timeseries_zarr_v1",
        )
        if (
            timeseries_manifest.get("generation_id") != generation_id
            or timeseries_manifest.get("manifest_digest") != manifest_digest
        ):
            reject_invalid_state(
                "Stack time-series child generation binding is invalid"
            )
        with open_timeseries_zarr(timeseries_root) as timeseries_store:
            if (
                timeseries_store.generation_id != generation_id
                or timeseries_store.manifest_digest != manifest_digest
            ):
                reject_invalid_state(
                    "Stack time-series CURRENT differs from its bound generation"
                )
            timeseries_pair_ids = _timeseries_pair_ids(timeseries_store.path)
            if len(timeseries_pair_ids) != len(pair_ids) or set(
                timeseries_pair_ids
            ) != set(pair_ids):
                reject_invalid_state("Stack time-series pair network is invalid")
    except Exception:
        opened.lease.close()
        raise
    return StackResultGeneration(
        root=opened.root,
        generation_id=opened.generation_id,
        generation_root=opened.path,
        manifest_digest=opened.manifest_digest,
        pair_ids=pair_ids,
        pairs=pairs,
        timeseries_root=timeseries_root,
        timeseries_generation_id=generation_id,
        timeseries_manifest_digest=manifest_digest,
        _lease=opened.lease,
    )


__all__ = [
    "STACK_GENERATION_SCHEMA",
    "UNWRAP_GENERATION_SCHEMA",
    "PairGenerationBinding",
    "StackResultGeneration",
    "UnwrapResultGeneration",
    "open_stack_generation",
    "open_unwrap_generation",
    "publish_stack_generation",
    "publish_unwrap_generation",
]
