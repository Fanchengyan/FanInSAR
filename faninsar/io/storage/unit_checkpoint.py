"""Durable mission-unit checkpoints on the product publication kernel (P0088).

A checkpoint is one ordinary FanInSAR product directory: staged Zarr payloads
plus one ``item.json``. It records the stable unit locator and a small caller
owned specification over scientific metadata, never a payload content hash.
Payload publication and replacement reuse :class:`ProductWriter`, so a crash
before ``item.json`` leaves no committed unit and a failed replacement keeps
the previous good unit.

The payload contract mirrors
:meth:`faninsar.io.storage.scene_store.SceneWriter.add_unit`.
A loaded checkpoint can be replayed into a fresh scene writer without
recomputing the unchanged arrays; the resulting scene manifest is the normal
FanInSAR scene product.
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.io.storage._zarr_stac import (
    inspect_zarr_array_metadata,
    read_zarr_array,
    require_basename,
)
from faninsar.io.storage.product import (
    ITEM_NAME,
    Product,
    ProductWriter,
    _read_product_item,
    open_product,
)
from faninsar.processing.errors import (
    InvalidProcessingStateError,
    reject_invalid_state,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.io.storage.scene_store import SceneWriter

UNIT_CHECKPOINT_KIND = "unit-checkpoint"
UNIT_CHECKPOINT_SCHEMA_VERSION = 1

_SLC_ASSET = "slc"
_PHASE_ASSET = "phase"
_VALID_ASSET = "valid"
_SLC_HREF = f"{_SLC_ASSET}.zarr"
_PHASE_HREF = f"{_PHASE_ASSET}.zarr"
_VALID_HREF = f"{_VALID_ASSET}.zarr"


def _canonical_mapping(value: object, field_name: str) -> dict[str, Any]:
    """Return one JSON mapping with canonical text ordering."""
    if not isinstance(value, Mapping):
        reject_invalid_state(f"unit checkpoint {field_name} must be a mapping")
    try:
        text = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        reject_invalid_state(
            f"unit checkpoint {field_name} is not strict JSON: {error}"
        )
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        reject_invalid_state(f"unit checkpoint {field_name} must be a JSON mapping")
    return parsed


def _canonical_specification(value: object) -> tuple[dict[str, Any], str]:
    """Return the normalized specification and its stable small-metadata id."""
    payload = _canonical_mapping(value, "specification")
    text = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return payload, digest


def _require_2d_dtype(value: object, dtype: object, field_name: str) -> np.ndarray:
    """Return ``value`` when it is a positive 2-D array of ``dtype``."""
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or any(int(size) <= 0 for size in value.shape)
        or value.dtype != np.dtype(dtype)
    ):
        reject_invalid_state(
            f"unit checkpoint {field_name} must be a 2-D {np.dtype(dtype)} array"
        )
    return value


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    """Return an integral scalar at or above ``minimum``."""
    if isinstance(value, bool):
        reject_invalid_state(f"unit checkpoint {field_name} must be an integer")
    try:
        resolved = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        reject_invalid_state(f"unit checkpoint {field_name} must be an integer")
    if resolved < minimum:
        reject_invalid_state(f"unit checkpoint {field_name} must be at least {minimum}")
    return resolved


def _write_payload(path: Path, value: np.ndarray) -> None:
    """Write one checkpoint payload with every declared chunk on disk.

    Zarr omits chunks whose samples all equal the fill value by default. An
    all-``False`` validity tile, or an all-zero phase screen, would then look
    exactly like a truncated payload to the structural load check. Checkpoint
    payloads therefore materialize fill chunks too, so an absent chunk always
    means corruption and never a legitimate sparse tile.
    """
    import zarr

    chunks = tuple(max(1, min(int(size), 256)) for size in value.shape)
    array = zarr.create_array(
        store=str(path),
        shape=value.shape,
        chunks=chunks,
        dtype=value.dtype,
        config={"write_empty_chunks": True},
    )
    array[...] = value
    array.attrs["faninsar:format"] = "stac-zarr"


def _validate_stored_chunks(path: Path, label: str) -> None:
    """Reject a Zarr payload whose declared chunks are absent or empty.

    This is a structural publication check, not a payload hash or sample scan.
    It only inspects local chunk files when the store exposes them.
    """
    import zarr

    try:
        array = zarr.open_array(store=str(path), mode="r")
        metadata = array.metadata
        encoding = getattr(metadata, "chunk_key_encoding", None)
        grid = getattr(metadata, "chunk_grid", None)
        root = getattr(array.store, "root", None)
        if encoding is None or grid is None or root is None:
            return
        chunk_root = Path(root)
        shape = tuple(int(size) for size in array.shape)
        for coordinates in grid.all_chunk_coords(shape):
            key = encoding.encode_chunk_key(coordinates)
            chunk = chunk_root / key
            if chunk.is_symlink():
                reject_invalid_state(
                    f"unit checkpoint {label} chunk is a symbolic link"
                )
            if not chunk.is_file():
                reject_invalid_state(
                    f"unit checkpoint {label} is missing chunk {key!r}"
                )
            if chunk.stat().st_size == 0:
                reject_invalid_state(
                    f"unit checkpoint {label} has a truncated chunk {key!r}"
                )
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        reject_invalid_state(f"unit checkpoint {label} cannot be validated: {error}")


def _validate_stored_array(
    path: Path,
    *,
    label: str,
    shape: tuple[int, int],
    dtype: object,
    declared_dtype: object,
) -> None:
    """Validate one committed Zarr asset's declared and stored structure."""
    actual_shape, actual_dtype = inspect_zarr_array_metadata(path)
    if actual_shape != shape:
        reject_invalid_state(
            f"unit checkpoint {label} shape {actual_shape} differs from {shape}"
        )
    expected_dtype = np.dtype(dtype)
    if actual_dtype != expected_dtype:
        reject_invalid_state(
            f"unit checkpoint {label} dtype {actual_dtype} "
            f"differs from {expected_dtype}"
        )
    if declared_dtype is not None:
        try:
            declared = np.dtype(declared_dtype)
        except TypeError:
            reject_invalid_state(f"unit checkpoint {label} declares an invalid dtype")
        if declared != expected_dtype:
            reject_invalid_state(
                f"unit checkpoint {label} declares dtype {declared}, "
                f"expected {expected_dtype}"
            )
    _validate_stored_chunks(path, label)


@dataclass(frozen=True, slots=True)
class UnitPayloads:
    """One mission unit's scene arrays and bounded placement metadata.

    The field set mirrors :meth:`faninsar.io.storage.scene_store.SceneWriter.add_unit`
    so a checkpoint can be replayed without reinterpretation.

    Parameters
    ----------
    slc : numpy.ndarray
        Two-dimensional ``complex64`` aligned unit payload.
    row_origin, col_origin : int
        Non-negative placement on the owner's common grid.
    phase : numpy.ndarray, optional
        Finite ``float32`` phase screen with the payload shape.
    valid : numpy.ndarray, optional
        Boolean validity mask with the payload shape.
    burst_id : str, optional
        Physical burst identifier (a single path basename).
    absolute_burst_id : int, optional
        Official absolute burst identifier.
    scientific_lineage : sequence of mapping, optional
        Small upstream source records consumed by this unit.
    phase_state : mapping, optional
        Flattening state recorded by the scene product.

    """

    slc: np.ndarray
    row_origin: int
    col_origin: int
    phase: np.ndarray | None = None
    valid: np.ndarray | None = None
    burst_id: str | None = None
    absolute_burst_id: int | None = None
    scientific_lineage: Sequence[Mapping[str, str]] = ()
    phase_state: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate the unit payload contract before publication."""
        slc = _require_2d_dtype(self.slc, np.complex64, "slc")
        if self.phase is not None:
            phase = _require_2d_dtype(self.phase, np.float32, "phase")
            if phase.shape != slc.shape:
                reject_invalid_state("unit checkpoint phase must match the slc shape")
            if not np.all(np.isfinite(phase)):
                reject_invalid_state("unit checkpoint phase must be finite")
        if self.valid is not None:
            valid = _require_2d_dtype(self.valid, np.bool_, "valid")
            if valid.shape != slc.shape:
                reject_invalid_state("unit checkpoint valid must match the slc shape")
        object.__setattr__(self, "row_origin", _integer(self.row_origin, "row_origin"))
        object.__setattr__(self, "col_origin", _integer(self.col_origin, "col_origin"))
        if self.burst_id is not None:
            object.__setattr__(
                self,
                "burst_id",
                require_basename(self.burst_id, "unit checkpoint burst_id"),
            )
        if self.absolute_burst_id is not None:
            object.__setattr__(
                self,
                "absolute_burst_id",
                _integer(self.absolute_burst_id, "absolute_burst_id"),
            )
        lineage: list[dict[str, str]] = []
        for item in self.scientific_lineage:
            if not isinstance(item, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in item.items()
            ):
                reject_invalid_state(
                    "unit checkpoint scientific_lineage must be string mappings"
                )
            lineage.append(dict(item))
        object.__setattr__(self, "scientific_lineage", tuple(lineage))
        if self.phase_state is not None:
            object.__setattr__(
                self,
                "phase_state",
                _canonical_mapping(self.phase_state, "phase_state"),
            )


@dataclass(frozen=True, slots=True)
class UnitCheckpoint:
    """One validated committed unit receipt and its payload locators.

    Attributes
    ----------
    unit_id : str
        Stable unit locator (also the checkpoint directory name).
    path : pathlib.Path
        Committed checkpoint product directory.
    write_id : str
        Checkpoint publication id; an exact dependency reference.
    specification : mapping
        Caller-owned small metadata that must match for reuse.
    specification_id : str
        Canonical digest of ``specification``.
    properties : mapping
        Extra receipt metadata supplied at commit.
    slc_path, phase_path, valid_path : pathlib.Path or None
        Committed Zarr assets.
    rows, cols, row_origin, col_origin : int
        Stored unit shape and common-grid placement.
    burst_id, absolute_burst_id : optional
        Stored physical burst identity.
    scientific_lineage : tuple of mapping
        Stored upstream source records.
    phase_state : mapping or None
        Stored flattening state.

    """

    unit_id: str
    path: Path
    write_id: str
    specification: Mapping[str, Any]
    specification_id: str
    properties: Mapping[str, Any]
    slc_path: Path
    phase_path: Path | None
    valid_path: Path | None
    rows: int
    cols: int
    row_origin: int
    col_origin: int
    burst_id: str | None
    absolute_burst_id: int | None
    scientific_lineage: tuple[dict[str, str], ...] = field(default=())
    phase_state: Mapping[str, Any] | None = field(default=None)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the stored unit shape."""
        return (self.rows, self.cols)

    def _read_payload(self, path: Path, label: str) -> np.ndarray:
        """Read one committed payload, reporting decode corruption as state.

        A non-empty but corrupt compressed chunk survives the structural
        publication check, so the decoder can raise a raw ``RuntimeError``.
        Recovery callers key on :class:`InvalidProcessingStateError`, so that
        failure is converted here instead of escaping as a crash.
        """
        try:
            return read_zarr_array(path)
        except InvalidProcessingStateError:
            raise
        except RuntimeError as error:
            reject_invalid_state(
                f"unit checkpoint {label} payload cannot be decoded: {error}"
            )

    def read_slc(self) -> np.ndarray:
        """Return the committed unit payload.

        Raises
        ------
        InvalidProcessingStateError
            If the stored payload cannot be read or decoded.

        """
        return self._read_payload(self.slc_path, "slc")

    def read_phase(self) -> np.ndarray | None:
        """Return the committed phase screen, or ``None`` when absent.

        Raises
        ------
        InvalidProcessingStateError
            If the stored phase screen cannot be read or decoded.

        """
        if self.phase_path is None:
            return None
        return self._read_payload(self.phase_path, "phase")

    def read_valid(self) -> np.ndarray | None:
        """Return the committed validity mask, or ``None`` when absent.

        Raises
        ------
        InvalidProcessingStateError
            If the stored validity mask cannot be read or decoded.

        """
        if self.valid_path is None:
            return None
        return self._read_payload(self.valid_path, "valid")

    def add_to(self, writer: SceneWriter, *, tag: str | None = None) -> None:
        """Stage this committed unit on ``writer`` without recomputation.

        Parameters
        ----------
        writer : SceneWriter
            Target scene writer for the same scientific unit.
        tag : str, optional
            Unit asset name; defaults to :attr:`unit_id`.

        """
        resolved_tag = (
            self.unit_id if tag is None else require_basename(tag, "scene tag")
        )
        writer.add_unit(
            tag=resolved_tag,
            payload=self.read_slc(),
            row_origin=self.row_origin,
            col_origin=self.col_origin,
            scientific_lineage=self.scientific_lineage,
            phase_state=self.phase_state,
            phase_screen=self.read_phase(),
            valid_mask=self.read_valid(),
            burst_id=self.burst_id,
            absolute_burst_id=self.absolute_burst_id,
        )


class UnitCheckpointStore:
    """Commit and reopen mission-unit checkpoints under one local root.

    Parameters
    ----------
    root : path-like
        Directory holding one committed checkpoint directory per unit id.
        Missing parents are created.

    """

    def __init__(self, root: str | Path) -> None:
        """Create the checkpoint root without touching existing units."""
        resolved = Path(root)
        if resolved.is_symlink():
            reject_invalid_state("unit checkpoint root must not be a symbolic link")
        if resolved.exists() and not resolved.is_dir():
            reject_invalid_state("unit checkpoint root must be a directory")
        resolved.mkdir(parents=True, exist_ok=True)
        self.root = resolved

    def _unit_target(self, unit_id: object) -> tuple[str, Path]:
        """Return a safe unit id and its checkpoint directory."""
        resolved = require_basename(unit_id, "unit checkpoint unit_id")
        if resolved.startswith("."):
            reject_invalid_state("unit checkpoint unit_id must not be hidden")
        target = self.root / resolved
        if target.is_symlink():
            reject_invalid_state("unit checkpoint unit must not be a symbolic link")
        if target.exists() and not target.is_dir():
            reject_invalid_state("unit checkpoint unit must be a directory")
        return resolved, target

    @staticmethod
    def _require_replaceable(target: Path, expected_unit_id: str) -> None:
        """Refuse to replace a unit path holding another kind or locator.

        This reads the persisted item only; payload structure is not
        inspected here so a damaged checkpoint of this same unit can still be
        repaired by a new commit. Case-differing ids alias onto one directory
        on case-insensitive filesystems, so the stored locator is compared
        exactly before any overwrite.
        """
        item_path = target / ITEM_NAME
        if not (item_path.is_file() or item_path.is_symlink()):
            return
        _, item = _read_product_item(target)
        properties = item["properties"]
        kind = properties["faninsar:kind"]
        if kind != UNIT_CHECKPOINT_KIND:
            reject_invalid_state(
                f"unit checkpoint will not overwrite {target}: existing product "
                f"kind is {kind!r}, expected {UNIT_CHECKPOINT_KIND!r}"
            )
        stored_unit_id = properties.get("faninsar:unit_id")
        if stored_unit_id != expected_unit_id:
            reject_invalid_state(
                f"unit checkpoint will not overwrite {target}: existing unit "
                f"{stored_unit_id!r} differs from requested {expected_unit_id!r}"
            )

    def load(
        self, unit_id: str, specification: Mapping[str, object]
    ) -> UnitCheckpoint | None:
        """Return one validated committed unit, or ``None`` when not reusable.

        ``None`` means the unit was never committed, or its stored
        specification differs from ``specification``. A committed checkpoint
        whose item or assets are structurally corrupt is rejected instead of
        being silently recomputed.

        Parameters
        ----------
        unit_id : str
            Stable unit locator.
        specification : mapping
            Small metadata that must match the committed specification.

        Returns
        -------
        UnitCheckpoint or None
            The validated receipt when reusable.

        """
        resolved_id, target = self._unit_target(unit_id)
        _, requested_id = _canonical_specification(specification)
        if not (target / ITEM_NAME).is_file():
            return None
        product = open_product(target, kind=UNIT_CHECKPOINT_KIND)
        stored_id = product.properties.get("faninsar:specification_id")
        if isinstance(stored_id, str) and stored_id != requested_id:
            return None
        return self._validated_record(product, expected_unit_id=resolved_id)

    def commit(
        self,
        unit_id: str,
        specification: Mapping[str, object],
        *,
        payloads: UnitPayloads,
        properties: Mapping[str, object] | None = None,
    ) -> UnitCheckpoint:
        """Publish one unit checkpoint atomically and return its receipt.

        An existing checkpoint for ``unit_id`` is replaced only after the new
        item is durable; a failure leaves the previous good unit visible. A
        path holding a different product kind or a different stored unit
        locator is refused rather than overwritten.

        Parameters
        ----------
        unit_id : str
            Stable unit locator and checkpoint directory name.
        specification : mapping
            Small scientific metadata compared exactly on reuse. Include
            reference-unit publication identities, not payload hashes.
        payloads : UnitPayloads
            Stored scene arrays and placement metadata.
        properties : mapping, optional
            Extra JSON receipt metadata.

        Returns
        -------
        UnitCheckpoint
            The validated committed receipt.

        """
        resolved_id, target = self._unit_target(unit_id)
        self._require_replaceable(target, resolved_id)
        stored_specification, specification_id = _canonical_specification(specification)
        stored_properties = _canonical_mapping(properties or {}, "properties")
        rows, cols = (int(payloads.slc.shape[0]), int(payloads.slc.shape[1]))
        with ProductWriter(target, kind=UNIT_CHECKPOINT_KIND, overwrite=True) as writer:
            _write_payload(writer.staging / _SLC_HREF, payloads.slc)
            slc_fields: dict[str, object] = {
                "faninsar:role": "slc",
                "faninsar:dtype": str(payloads.slc.dtype),
                "faninsar:rows": rows,
                "faninsar:cols": cols,
                "faninsar:row_origin": payloads.row_origin,
                "faninsar:col_origin": payloads.col_origin,
                "faninsar:lineage": [
                    dict(item) for item in payloads.scientific_lineage
                ],
                "faninsar:phase_state": (
                    None if payloads.phase_state is None else dict(payloads.phase_state)
                ),
            }
            if payloads.burst_id is not None:
                slc_fields["faninsar:burst_id"] = payloads.burst_id
            if payloads.absolute_burst_id is not None:
                slc_fields["faninsar:absolute_burst_id"] = payloads.absolute_burst_id
            if payloads.phase is not None:
                _write_payload(writer.staging / _PHASE_HREF, payloads.phase)
                writer.add_asset(
                    _PHASE_ASSET,
                    _PHASE_HREF,
                    **{
                        "faninsar:role": "phase",
                        "faninsar:dtype": str(payloads.phase.dtype),
                        "faninsar:rows": rows,
                        "faninsar:cols": cols,
                    },
                )
            if payloads.valid is not None:
                _write_payload(writer.staging / _VALID_HREF, payloads.valid)
                writer.add_asset(
                    _VALID_ASSET,
                    _VALID_HREF,
                    **{
                        "faninsar:role": "valid_mask",
                        "faninsar:dtype": str(payloads.valid.dtype),
                        "faninsar:rows": rows,
                        "faninsar:cols": cols,
                    },
                )
            writer.add_asset(_SLC_ASSET, _SLC_HREF, **slc_fields)
            product = writer.commit(
                {
                    "faninsar:unit_id": resolved_id,
                    "faninsar:unit_checkpoint_schema": UNIT_CHECKPOINT_SCHEMA_VERSION,
                    "faninsar:specification": stored_specification,
                    "faninsar:specification_id": specification_id,
                    "faninsar:unit_properties": stored_properties,
                }
            )
        return self._validated_record(product, expected_unit_id=resolved_id)

    def _validated_record(
        self, product: Product, *, expected_unit_id: str
    ) -> UnitCheckpoint:
        """Validate one committed item and all payload structure."""
        properties = product.properties
        schema = properties.get("faninsar:unit_checkpoint_schema")
        if schema != UNIT_CHECKPOINT_SCHEMA_VERSION:
            reject_invalid_state(
                f"unit checkpoint schema {schema!r} is unsupported; "
                f"expected {UNIT_CHECKPOINT_SCHEMA_VERSION}"
            )
        if properties.get("faninsar:unit_id") != expected_unit_id:
            reject_invalid_state("unit checkpoint item id does not match its locator")
        raw_specification = properties.get("faninsar:specification")
        stored_specification, specification_id = _canonical_specification(
            raw_specification
        )
        if properties.get("faninsar:specification_id") != specification_id:
            reject_invalid_state("unit checkpoint specification id is inconsistent")
        raw_properties = properties.get("faninsar:unit_properties", {})
        if not isinstance(raw_properties, dict):
            reject_invalid_state("unit checkpoint properties must be a mapping")

        assets = product.assets
        slc_asset = assets.get(_SLC_ASSET)
        if not isinstance(slc_asset, dict) or slc_asset.get("faninsar:role") != "slc":
            reject_invalid_state("unit checkpoint has no slc payload asset")
        rows = _integer(slc_asset.get("faninsar:rows"), "rows", minimum=1)
        cols = _integer(slc_asset.get("faninsar:cols"), "cols", minimum=1)
        row_origin = _integer(slc_asset.get("faninsar:row_origin"), "row_origin")
        col_origin = _integer(slc_asset.get("faninsar:col_origin"), "col_origin")
        shape = (rows, cols)
        slc_path = product.asset_path(_SLC_ASSET)
        _validate_stored_array(
            slc_path,
            label="slc",
            shape=shape,
            dtype=np.complex64,
            declared_dtype=slc_asset.get("faninsar:dtype"),
        )

        phase_path = self._optional_asset(
            product,
            _PHASE_ASSET,
            role="phase",
            shape=shape,
            dtype=np.float32,
        )
        valid_path = self._optional_asset(
            product,
            _VALID_ASSET,
            role="valid_mask",
            shape=shape,
            dtype=np.bool_,
        )

        raw_lineage = slc_asset.get("faninsar:lineage", [])
        if not isinstance(raw_lineage, list) or not all(
            isinstance(item, dict)
            and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in item.items()
            )
            for item in raw_lineage
        ):
            reject_invalid_state("unit checkpoint lineage must be string mappings")
        raw_burst = slc_asset.get("faninsar:burst_id")
        burst_id = (
            None
            if raw_burst is None
            else require_basename(raw_burst, "unit checkpoint burst_id")
        )
        raw_absolute = slc_asset.get("faninsar:absolute_burst_id")
        absolute_burst_id = (
            None
            if raw_absolute is None
            else _integer(raw_absolute, "absolute_burst_id")
        )
        raw_phase_state = slc_asset.get("faninsar:phase_state")
        if raw_phase_state is not None and not isinstance(raw_phase_state, dict):
            reject_invalid_state("unit checkpoint phase_state must be a mapping")
        return UnitCheckpoint(
            unit_id=expected_unit_id,
            path=product.path,
            write_id=product.write_id,
            specification=stored_specification,
            specification_id=specification_id,
            properties=dict(raw_properties),
            slc_path=slc_path,
            phase_path=phase_path,
            valid_path=valid_path,
            rows=rows,
            cols=cols,
            row_origin=row_origin,
            col_origin=col_origin,
            burst_id=burst_id,
            absolute_burst_id=absolute_burst_id,
            scientific_lineage=tuple(dict(item) for item in raw_lineage),
            phase_state=(None if raw_phase_state is None else dict(raw_phase_state)),
        )

    @staticmethod
    def _optional_asset(
        product: Product,
        name: str,
        *,
        role: str,
        shape: tuple[int, int],
        dtype: object,
    ) -> Path | None:
        """Validate one optional Zarr asset and return its path."""
        asset = product.assets.get(name)
        if asset is None:
            return None
        if not isinstance(asset, dict) or asset.get("faninsar:role") != role:
            reject_invalid_state(f"unit checkpoint asset {name!r} has the wrong role")
        path = product.asset_path(name)
        _validate_stored_array(
            path,
            label=name,
            shape=shape,
            dtype=dtype,
            declared_dtype=asset.get("faninsar:dtype"),
        )
        return path


__all__ = [
    "UNIT_CHECKPOINT_KIND",
    "UNIT_CHECKPOINT_SCHEMA_VERSION",
    "UnitCheckpoint",
    "UnitCheckpointStore",
    "UnitPayloads",
]
