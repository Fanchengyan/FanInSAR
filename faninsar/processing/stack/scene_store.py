"""Manifest-bound master-aligned SLC scene storage for Stack formation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from faninsar.processing.errors import reject_invalid_state

SCENE_SCHEMA = "scene_artifact_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SceneUnit:
    """One immutable master-grid aligned SLC unit."""

    tag: str
    reference_path: Path
    secondary_path: Path
    shape: tuple[int, int]
    row_origin: int
    col_origin: int
    payload_digest: str

    def __post_init__(self) -> None:
        """Validate the unit's bounded shape and digest."""
        if not self.tag or len(self.shape) != 2 or any(
            size <= 0 for size in self.shape
        ):
            reject_invalid_state("scene unit has an invalid tag or shape")
        if len(self.payload_digest) != 64:
            reject_invalid_state("scene unit payload digest is invalid")


@dataclass(frozen=True, slots=True)
class CoregisteredSceneStore:
    """Read-only manifest-bound store for one Stack date."""

    root: Path
    date_id: str
    master_id: str
    domain: str
    units: tuple[SceneUnit, ...]
    manifest_digest: str

    @classmethod
    def open(cls, root: str | Path) -> CoregisteredSceneStore:
        """Open and validate a complete scene artifact generation.

        Parameters
        ----------
        root
            Caller-owned scene store directory containing ``manifest.json``.

        Returns
        -------
        CoregisteredSceneStore
            Immutable read-only store metadata.

        """
        path = Path(root)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            reject_invalid_state(f"scene manifest missing or unsafe: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            reject_invalid_state(f"scene manifest cannot be read: {error}")
        if manifest.get("schema_version") != SCENE_SCHEMA:
            reject_invalid_state("unsupported scene artifact schema")
        units: list[SceneUnit] = []
        for raw in manifest.get("units", []):
            reference = path / str(raw["reference_file"])
            secondary = path / str(raw["secondary_file"])
            for payload in (reference, secondary):
                if not payload.is_file() or payload.is_symlink():
                    reject_invalid_state(f"scene payload missing or unsafe: {payload}")
            shape = (int(raw["rows"]), int(raw["cols"]))
            units.append(
                SceneUnit(
                    tag=str(raw["tag"]),
                    reference_path=reference,
                    secondary_path=secondary,
                    shape=shape,
                    row_origin=int(raw["row_origin"]),
                    col_origin=int(raw["col_origin"]),
                    payload_digest=str(raw["payload_digest"]),
                )
            )
        if not units:
            reject_invalid_state("scene store has no complete units")
        expected_digest = str(manifest.get("manifest_digest", ""))
        unsigned = dict(manifest)
        unsigned.pop("manifest_digest", None)
        actual_digest = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if expected_digest != actual_digest:
            reject_invalid_state("scene manifest digest mismatch")
        return cls(
            root=path,
            date_id=str(manifest["date_id"]),
            master_id=str(manifest["master_id"]),
            domain=str(manifest["domain"]),
            units=tuple(units),
            manifest_digest=actual_digest,
        )

    def unit_map(self) -> dict[str, SceneUnit]:
        """Return units keyed by their stable burst tag."""
        tags = [unit.tag for unit in self.units]
        if len(tags) != len(set(tags)):
            reject_invalid_state("scene store contains duplicate unit tags")
        return {unit.tag: unit for unit in self.units}

    def read(self, tag: str) -> tuple[np.ndarray, np.ndarray, SceneUnit]:
        """Read one unit after validating both payload bytes and shape."""
        unit = self.unit_map().get(tag)
        if unit is None:
            reject_invalid_state(f"scene unit {tag!r} is not in the manifest")
        payload_digest = hashlib.sha256(
            (_sha256(unit.reference_path) + _sha256(unit.secondary_path)).encode()
        ).hexdigest()
        if payload_digest != unit.payload_digest:
            reject_invalid_state(f"scene unit {tag!r} payload digest mismatch")
        reference = np.load(unit.reference_path, allow_pickle=False)
        secondary = np.load(unit.secondary_path, allow_pickle=False)
        if reference.shape != unit.shape or secondary.shape != unit.shape:
            reject_invalid_state(f"scene unit {tag!r} shape mismatch")
        if reference.dtype != np.complex64 or secondary.dtype != np.complex64:
            reject_invalid_state(f"scene unit {tag!r} dtype mismatch")
        return reference, secondary, unit


def write_scene_unit(
    root: str | Path,
    *,
    date_id: str,
    master_id: str,
    domain: str,
    tag: str,
    reference: np.ndarray,
    secondary: np.ndarray,
    row_origin: int,
    col_origin: int,
) -> None:
    """Atomically add one aligned unit and publish a complete manifest."""
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    if reference.shape != secondary.shape or reference.dtype != np.complex64:
        reject_invalid_state("aligned scene arrays must be matching complex64 arrays")
    ref_path = path / f"{tag}.reference.npy"
    sec_path = path / f"{tag}.secondary.npy"
    for target, array in ((ref_path, reference), (sec_path, secondary)):
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(array, dtype=np.complex64), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    manifest_path = path / "manifest.json"
    existing: dict[str, object] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = [
        item for item in existing.get("units", []) if item.get("tag") != tag
    ]
    units.append(
        {
            "tag": tag,
            "reference_file": ref_path.name,
            "secondary_file": sec_path.name,
            "rows": int(reference.shape[0]),
            "cols": int(reference.shape[1]),
            "row_origin": int(row_origin),
            "col_origin": int(col_origin),
            "payload_digest": hashlib.sha256(
                (_sha256(ref_path) + _sha256(sec_path)).encode()
            ).hexdigest(),
        }
    )
    unsigned = {
        "schema_version": SCENE_SCHEMA,
        "status": "complete",
        "date_id": date_id,
        "master_id": master_id,
        "domain": domain,
        "units": sorted(units, key=lambda item: str(item["tag"])),
    }
    manifest = {
        **unsigned,
        "manifest_digest": hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)


def copy_reference_units(source: str | Path, target: str | Path) -> None:
    """Copy reference payloads into the master store without source aliases."""
    source_store = CoregisteredSceneStore.open(source)
    destination = Path(target)
    for unit in source_store.units:
        reference, _, _ = source_store.read(unit.tag)
        write_scene_unit(
            destination,
            date_id=source_store.master_id,
            master_id=source_store.master_id,
            domain=source_store.domain,
            tag=unit.tag,
            reference=reference,
            secondary=reference,
            row_origin=unit.row_origin,
            col_origin=unit.col_origin,
        )


def form_scene_interferograms(
    reference_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    *,
    reference_role: str = "reference",
    secondary_role: str = "secondary",
) -> dict[str, np.ndarray]:
    """Form IFGs from aligned scene payloads only, never from source SAFE paths."""
    if reference_store.domain != secondary_store.domain:
        reject_invalid_state("scene artifact domains do not match")
    reference_units = reference_store.unit_map()
    secondary_units = secondary_store.unit_map()
    if set(reference_units) != set(secondary_units):
        reject_invalid_state("scene artifact unit manifests differ")
    outputs: dict[str, np.ndarray] = {}
    for tag in sorted(reference_units):
        reference_ref, reference_sec, reference_unit = reference_store.read(tag)
        secondary_ref, secondary_sec, secondary_unit = secondary_store.read(tag)
        if reference_unit.shape != secondary_unit.shape:
            reject_invalid_state(f"scene unit {tag!r} shape differs across dates")
        if reference_role == "reference":
            reference = reference_ref
        elif reference_role == "secondary":
            reference = reference_sec
        else:
            reject_invalid_state("unsupported reference scene role")
        if secondary_role == "reference":
            secondary = secondary_ref
        elif secondary_role == "secondary":
            secondary = secondary_sec
        else:
            reject_invalid_state("unsupported secondary scene role")
        outputs[tag] = (reference * np.conj(secondary)).astype(np.complex64)
    return outputs


__all__ = [
    "SCENE_SCHEMA",
    "CoregisteredSceneStore",
    "SceneUnit",
    "copy_reference_units",
    "form_scene_interferograms",
    "write_scene_unit",
]
