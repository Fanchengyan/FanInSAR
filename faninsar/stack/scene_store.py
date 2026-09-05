"""Manifest-bound Reference-aligned SLC scene storage for Stack formation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.interferometry.pair import (
    InterferogramProduct,
    form_interferogram,
    goldstein_filter,
    mask_invalid_looks,
)

SCENE_SCHEMA = "scene_artifact_v1"

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch

    from faninsar.processing.interferometry.phase_filter import (
        PhaseFilter,
        PhaseFilterResult,
    )


def _apply_phase_filter_contract(
    phase_filter: PhaseFilter,
    interferogram: torch.Tensor,
    valid_mask: torch.Tensor,
) -> PhaseFilterResult:
    """Apply a runtime filter and enforce the formation-result contract.

    Parameters
    ----------
    phase_filter : PhaseFilter
        Trusted runtime strategy selected at the public Stack formation
        boundary.
    interferogram : torch.Tensor
        Two-dimensional complex IFG on the selected Stack device.
    valid_mask : torch.Tensor
        Boolean finite support mask on the same device as ``interferogram``.

    Returns
    -------
    PhaseFilterResult
        Contract-conforming filtered tensor and support mask.

    Raises
    ------
    InvalidProcessingStateError
        If a strategy returns a result with incompatible shape, dtype, device,
        support, or finite-value semantics.

    Notes
    -----
    Custom filters are trusted Python execution objects, but their output is
    still validated here because this is the one boundary that converts their
    result into a durable Stack artifact. No implicit cast, device transfer,
    or mask widening is permitted.

    """
    import torch

    from faninsar.processing.interferometry.phase_filter import PhaseFilterResult

    result = phase_filter.apply(interferogram, valid_mask=valid_mask)
    if not isinstance(result, PhaseFilterResult):
        reject_invalid_state("phase filter must return a PhaseFilterResult")
    output = result.interferogram
    output_mask = result.valid_mask
    if not isinstance(output, torch.Tensor):
        reject_invalid_state("phase filter interferogram result must be a Tensor")
    if output.shape != interferogram.shape:
        reject_invalid_state("phase filter result shape must match its input")
    if not output.is_complex() or output.dtype != interferogram.dtype:
        reject_invalid_state(
            "phase filter result must preserve the input complex dtype"
        )
    if output.device != interferogram.device:
        reject_invalid_state("phase filter result must remain on the input device")
    if not isinstance(output_mask, torch.Tensor):
        reject_invalid_state("phase filter valid_mask result must be a Tensor")
    if output_mask.shape != valid_mask.shape:
        reject_invalid_state("phase filter valid_mask shape must match its input")
    if output_mask.dtype != torch.bool:
        reject_invalid_state("phase filter valid_mask result must have bool dtype")
    if output_mask.device != valid_mask.device:
        reject_invalid_state("phase filter valid_mask must remain on the input device")
    if bool(torch.any(output_mask & ~valid_mask)):
        reject_invalid_state("phase filter valid_mask cannot expand input support")
    valid_values = output[output_mask]
    if not bool(
        torch.all(torch.isfinite(valid_values.real) & torch.isfinite(valid_values.imag))
    ):
        reject_invalid_state("phase filter valid output samples must be finite")
    return result


def scene_grid_identity(
    domain: str,
    grid_shape: tuple[int, int],
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Return a canonical identity for one scene coordinate grid.

    Parameters
    ----------
    domain : str
        Coordinate domain name.
    grid_shape : tuple[int, int]
        Global row and column dimensions.
    metadata : mapping, optional
        Domain-specific coordinate metadata such as CRS and affine transform.

    Returns
    -------
    str
        Lowercase SHA-256 identity.

    """
    payload = {
        "domain": domain,
        "shape": [int(grid_shape[0]), int(grid_shape[1])],
        "metadata": dict(metadata or {}),
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"scene grid metadata is not canonical JSON: {error}")
    return hashlib.sha256(encoded).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    """Reject a path whose existing components contain symbolic links.

    Parameters
    ----------
    path : pathlib.Path
        Path to validate. Missing leaf components are allowed so callers can
        create a new store or payload after validation.

    Raises
    ------
    InvalidProcessingStateError
        If an existing path component is a symbolic link or cannot be
        inspected safely.

    """
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                reject_invalid_state(
                    f"scene store path contains a symbolic link: {current}"
                )
        except OSError as error:
            reject_invalid_state(f"scene store path cannot be inspected: {error}")


def _validated_store_root(root: str | Path, *, create: bool) -> Path:
    """Validate a caller-owned scene store root before reading or writing."""
    path = Path(root)
    _reject_symlink_components(path)
    if path.exists():
        if not path.is_dir():
            reject_invalid_state(f"scene store root is not a directory: {path}")
    elif create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            reject_invalid_state(f"scene store root cannot be created: {error}")
        _reject_symlink_components(path)
        if not path.is_dir() or path.is_symlink():
            reject_invalid_state(f"scene store root is unsafe: {path}")
    else:
        reject_invalid_state(f"scene store root is missing: {path}")
    return path


def _safe_payload_path(root: Path, value: object, field: str) -> Path:
    """Resolve a manifest payload filename without allowing path traversal."""
    if not isinstance(value, str):
        reject_invalid_state(f"scene manifest {field} must be a filename")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != value
        or value in {"", ".", ".."}
    ):
        reject_invalid_state(f"scene manifest {field} must be a basename")
    path = root / candidate
    _reject_symlink_components(path)
    return path


def _require_basename(value: object, field: str) -> str:
    """Validate a generated filename component before joining it to a root."""
    if not isinstance(value, str):
        reject_invalid_state(f"scene {field} must be a filename component")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != value
        or value in {"", ".", ".."}
    ):
        reject_invalid_state(f"scene {field} must be a basename")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SceneUnit:
    """One immutable Reference-grid aligned SLC unit."""

    tag: str
    primary_path: Path
    secondary_path: Path
    shape: tuple[int, int]
    row_origin: int
    col_origin: int
    payload_digest: str
    scientific_lineage: tuple[dict[str, str], ...] = ()
    phase_state: dict[str, object] | None = None
    phase_screen_path: Path | None = None
    phase_screen_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate the unit's bounded shape and digest."""
        if (
            not self.tag
            or len(self.shape) != 2
            or any(size <= 0 for size in self.shape)
        ):
            reject_invalid_state("scene unit has an invalid tag or shape")
        if len(self.payload_digest) != 64:
            reject_invalid_state("scene unit payload digest is invalid")


@dataclass(frozen=True, slots=True)
class CoregisteredSceneStore:
    """Read-only manifest-bound store for one Stack date."""

    root: Path
    date_id: str
    reference_id: str
    domain: str
    units: tuple[SceneUnit, ...]
    manifest_digest: str
    grid_shape: tuple[int, int]
    wavelength_m: float | None
    grid_identity: str
    flatten_stage: str = "coregistration"

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
        path = _validated_store_root(root, create=False)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            reject_invalid_state(f"scene manifest missing or unsafe: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            reject_invalid_state(f"scene manifest cannot be read: {error}")
        if manifest.get("schema_version") != SCENE_SCHEMA:
            reject_invalid_state("unsupported scene artifact schema")
        if "master_id" in manifest or "master" in manifest:
            reject_invalid_state(
                "legacy scene manifest uses master terminology; rebuild with "
                "reference_id"
            )
        if "reference_id" not in manifest:
            reject_invalid_state("scene manifest reference_id is missing")
        raw_units = manifest.get("units")
        if not isinstance(raw_units, list):
            reject_invalid_state("scene manifest units must be a list")
        units: list[SceneUnit] = []
        for raw in raw_units:
            if not isinstance(raw, dict):
                reject_invalid_state("scene manifest unit must be an object")
            primary = _safe_payload_path(path, raw.get("primary_file"), "primary_file")
            secondary = _safe_payload_path(
                path, raw.get("secondary_file"), "secondary_file"
            )
            for payload in (primary, secondary):
                if not payload.is_file() or payload.is_symlink():
                    reject_invalid_state(f"scene payload missing or unsafe: {payload}")
            shape = (int(raw["rows"]), int(raw["cols"]))
            units.append(
                SceneUnit(
                    tag=str(raw["tag"]),
                    primary_path=primary,
                    secondary_path=secondary,
                    shape=shape,
                    row_origin=int(raw["row_origin"]),
                    col_origin=int(raw["col_origin"]),
                    payload_digest=str(raw["payload_digest"]),
                    scientific_lineage=tuple(
                        item
                        for item in raw.get("scientific_lineage", [])
                        if isinstance(item, dict)
                    ),
                    phase_state=(
                        dict(raw["phase_state"])
                        if isinstance(raw.get("phase_state"), dict)
                        else None
                    ),
                    phase_screen_path=(
                        _safe_payload_path(
                            path,
                            raw["phase_screen_file"],
                            "phase_screen_file",
                        )
                        if raw.get("phase_screen_file") is not None
                        else None
                    ),
                    phase_screen_digest=(
                        str(raw["phase_screen_digest"])
                        if raw.get("phase_screen_digest") is not None
                        else None
                    ),
                )
            )
            unit = units[-1]
            if unit.phase_screen_path is not None:
                if (
                    not unit.phase_screen_path.is_file()
                    or unit.phase_screen_path.is_symlink()
                    or unit.phase_screen_digest is None
                    or _sha256(unit.phase_screen_path) != unit.phase_screen_digest
                ):
                    reject_invalid_state(
                        f"scene phase screen missing or digest mismatch: {unit.tag}"
                    )
                try:
                    screen = np.load(unit.phase_screen_path, allow_pickle=False)
                except (OSError, ValueError) as error:
                    reject_invalid_state(
                        f"scene phase screen cannot be read for {unit.tag!r}: {error}"
                    )
                if screen.shape != unit.shape or screen.dtype != np.float32:
                    reject_invalid_state(
                        f"scene phase screen shape or dtype mismatch for {unit.tag!r}"
                    )
            elif unit.phase_screen_digest is not None:
                reject_invalid_state(
                    f"scene phase screen digest has no payload for {unit.tag!r}"
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
        raw_grid_shape = manifest.get("grid_shape")
        if raw_grid_shape is None:
            grid_shape = (
                max(unit.row_origin + unit.shape[0] for unit in units),
                max(unit.col_origin + unit.shape[1] for unit in units),
            )
        elif (
            not isinstance(raw_grid_shape, list)
            or len(raw_grid_shape) != 2
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
                for size in raw_grid_shape
            )
        ):
            reject_invalid_state("scene manifest grid_shape is invalid")
        else:
            grid_shape = (int(raw_grid_shape[0]), int(raw_grid_shape[1]))
        if any(
            unit.row_origin < 0
            or unit.col_origin < 0
            or unit.row_origin + unit.shape[0] > grid_shape[0]
            or unit.col_origin + unit.shape[1] > grid_shape[1]
            for unit in units
        ):
            reject_invalid_state("scene unit lies outside the declared common grid")
        unit_stages = {
            str((unit.phase_state or {}).get("flatten_stage", "coregistration"))
            for unit in units
        }
        manifest_stage = str(
            manifest.get("flatten_stage", next(iter(unit_stages), "coregistration"))
        )
        if manifest_stage not in {"coregistration", "interferogram"}:
            reject_invalid_state("scene manifest flatten_stage is unsupported")
        if unit_stages != {manifest_stage}:
            reject_invalid_state("scene units mix flattening stages")
        raw_wavelength = manifest.get("wavelength_m")
        wavelength_m = None if raw_wavelength is None else float(raw_wavelength)
        if wavelength_m is not None and (
            not np.isfinite(wavelength_m) or wavelength_m <= 0.0
        ):
            reject_invalid_state("scene manifest wavelength_m is invalid")
        domain = str(manifest["domain"])
        raw_grid_identity = manifest.get("grid_identity")
        if domain == "geo" and raw_grid_identity is None:
            reject_invalid_state(
                "Geo scene manifest requires an explicit grid_identity"
            )
        grid_identity = str(
            raw_grid_identity or scene_grid_identity(domain, grid_shape)
        )
        if len(grid_identity) != 64 or any(
            character not in "0123456789abcdef" for character in grid_identity
        ):
            reject_invalid_state("scene manifest grid_identity is invalid")
        return cls(
            root=path,
            date_id=str(manifest["date_id"]),
            reference_id=str(manifest["reference_id"]),
            domain=domain,
            units=tuple(units),
            manifest_digest=actual_digest,
            grid_shape=grid_shape,
            wavelength_m=wavelength_m,
            grid_identity=grid_identity,
            flatten_stage=manifest_stage,
        )

    def read_phase_screen(self, tag: str) -> np.ndarray | None:
        """Read one persisted range-offset phase screen, if present."""
        unit = self.unit_map().get(tag)
        if unit is None:
            reject_invalid_state(f"scene unit {tag!r} is not in the manifest")
        if unit.phase_screen_path is None:
            return None
        screen = np.load(unit.phase_screen_path, allow_pickle=False)
        if screen.shape != unit.shape or screen.dtype != np.float32:
            reject_invalid_state(f"scene phase screen shape mismatch for {tag!r}")
        if _sha256(unit.phase_screen_path) != unit.phase_screen_digest:
            reject_invalid_state(f"scene phase screen digest mismatch for {tag!r}")
        return np.asarray(screen, dtype=np.float32)

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
            (_sha256(unit.primary_path) + _sha256(unit.secondary_path)).encode()
        ).hexdigest()
        if payload_digest != unit.payload_digest:
            reject_invalid_state(f"scene unit {tag!r} payload digest mismatch")
        primary = np.load(unit.primary_path, allow_pickle=False)
        secondary = np.load(unit.secondary_path, allow_pickle=False)
        if primary.shape != unit.shape or secondary.shape != unit.shape:
            reject_invalid_state(f"scene unit {tag!r} shape mismatch")
        if primary.dtype != np.complex64 or secondary.dtype != np.complex64:
            reject_invalid_state(f"scene unit {tag!r} dtype mismatch")
        return primary, secondary, unit


def write_scene_unit(
    root: str | Path,
    *,
    date_id: str,
    reference_id: str,
    domain: str,
    tag: str,
    primary: np.ndarray,
    secondary: np.ndarray,
    row_origin: int,
    col_origin: int,
    grid_shape: tuple[int, int] | None = None,
    wavelength_m: float | None = None,
    grid_identity: str | None = None,
    scientific_lineage: Sequence[Mapping[str, str]] | None = None,
    phase_state: Mapping[str, object] | None = None,
    phase_screen: np.ndarray | None = None,
    validate_existing_payloads: bool = True,
) -> None:
    """Atomically add one aligned unit and publish a complete manifest.

    Parameters
    ----------
    root : path-like
        Scene-store directory.
    date_id, reference_id : str
        Secondary acquisition and Stack Reference identifiers.
    domain : str
        Coordinate domain, either radar or geographic.
    tag : str
        Stable unit identifier within the store.
    primary, secondary : numpy.ndarray
        Matching complex64 scene arrays.
    row_origin, col_origin : int
        Unit origin within the declared common grid.
    grid_shape : tuple of int, optional
        Common scene dimensions. Defaults to the unit extent.
    wavelength_m : float, optional
        Radar wavelength in metres.
    grid_identity : str, optional
        Canonical coordinate-grid SHA-256.
    scientific_lineage : sequence of mappings, optional
        Source and processing lineage attached to the unit.
    phase_state : mapping, optional
        Phase-correction state attached to the unit.
    phase_screen : numpy.ndarray, optional
        Exact finite float32 phase screen in radians for the unit.
    validate_existing_payloads : bool, optional
        Re-read all previously published payload bytes before appending. A
        streaming producer that has already validated its resume tiles may set
        this to ``False`` to avoid quadratic I/O while retaining manifest
        validation. The default preserves the public fail-closed behavior.

    """
    path = _validated_store_root(root, create=True)
    _require_basename(tag, "tag")
    if (
        primary.ndim != 2
        or secondary.ndim != 2
        or primary.shape != secondary.shape
        or any(size <= 0 for size in primary.shape)
        or primary.dtype != np.complex64
        or secondary.dtype != np.complex64
    ):
        reject_invalid_state("aligned scene arrays must be matching complex64 arrays")
    if phase_screen is not None and (
        phase_screen.ndim != 2
        or phase_screen.shape != primary.shape
        or phase_screen.dtype != np.float32
        or not np.all(np.isfinite(phase_screen))
    ):
        reject_invalid_state("phase screen must be finite matching float32 array")
    resolved_grid_shape = grid_shape or (
        int(row_origin) + int(primary.shape[0]),
        int(col_origin) + int(primary.shape[1]),
    )
    if (
        len(resolved_grid_shape) != 2
        or any(size <= 0 for size in resolved_grid_shape)
        or row_origin < 0
        or col_origin < 0
        or row_origin + primary.shape[0] > resolved_grid_shape[0]
        or col_origin + primary.shape[1] > resolved_grid_shape[1]
    ):
        reject_invalid_state("scene unit lies outside the declared common grid")
    if wavelength_m is not None and (
        not np.isfinite(wavelength_m) or wavelength_m <= 0.0
    ):
        reject_invalid_state("scene wavelength_m must be finite and positive")
    if domain == "geo" and grid_identity is None:
        reject_invalid_state("Geo scene publication requires an explicit grid_identity")
    resolved_grid_identity = grid_identity or scene_grid_identity(
        domain,
        resolved_grid_shape,
    )
    if len(resolved_grid_identity) != 64 or any(
        character not in "0123456789abcdef" for character in resolved_grid_identity
    ):
        reject_invalid_state("scene grid_identity must be a lowercase SHA-256")
    manifest_path = path / "manifest.json"
    existing: dict[str, object] = {}
    if manifest_path.is_file():
        existing_store = CoregisteredSceneStore.open(path)
        if validate_existing_payloads:
            for existing_unit in existing_store.units:
                existing_store.read(existing_unit.tag)
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("date_id") != date_id
            or existing.get("reference_id") != reference_id
            or existing.get("domain") != domain
        ):
            reject_invalid_state("scene units must share date, Reference, and domain")
        existing_grid_shape = existing.get("grid_shape")
        if existing_grid_shape is not None and existing_grid_shape != list(
            resolved_grid_shape
        ):
            reject_invalid_state("scene units must share one common grid shape")
        if existing.get("wavelength_m") != wavelength_m:
            reject_invalid_state("scene units must share one radar wavelength")
        existing_grid_identity = existing.get("grid_identity")
        if existing_grid_identity is not None and (
            existing_grid_identity != resolved_grid_identity
        ):
            reject_invalid_state("scene units must share one coordinate grid")
    primary_path = path / f"{tag}.primary.npy"
    sec_path = path / f"{tag}.secondary.npy"
    for target, array in ((primary_path, primary), (sec_path, secondary)):
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(array, dtype=np.complex64), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    phase_path: Path | None = None
    phase_digest: str | None = None
    if phase_screen is not None:
        phase_path = path / f"{tag}.phase.npy"
        temporary = phase_path.with_suffix(phase_path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.save(
                stream,
                np.asarray(phase_screen, dtype=np.float32),
                allow_pickle=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(phase_path)
        phase_digest = _sha256(phase_path)
    else:
        (path / f"{tag}.phase.npy").unlink(missing_ok=True)
    units = [item for item in existing.get("units", []) if item.get("tag") != tag]
    unit_record = {
        "tag": tag,
        "primary_file": primary_path.name,
        "secondary_file": sec_path.name,
        "rows": int(primary.shape[0]),
        "cols": int(primary.shape[1]),
        "row_origin": int(row_origin),
        "col_origin": int(col_origin),
        "payload_digest": hashlib.sha256(
            (_sha256(primary_path) + _sha256(sec_path)).encode()
        ).hexdigest(),
        "scientific_lineage": [dict(item) for item in (scientific_lineage or ())],
        "phase_state": dict(phase_state) if phase_state is not None else None,
    }
    if phase_path is not None:
        unit_record["phase_screen_file"] = phase_path.name
        unit_record["phase_screen_digest"] = phase_digest
    units.append(unit_record)
    flatten_stage = str((phase_state or {}).get("flatten_stage", "coregistration"))
    if flatten_stage not in {"coregistration", "interferogram"}:
        reject_invalid_state("scene flatten_stage is unsupported")
    unsigned = {
        "schema_version": SCENE_SCHEMA,
        "status": "complete",
        "date_id": date_id,
        "reference_id": reference_id,
        "domain": domain,
        "grid_shape": list(resolved_grid_shape),
        "wavelength_m": wavelength_m,
        "grid_identity": resolved_grid_identity,
        "flatten_stage": flatten_stage,
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
    """Copy Reference payloads into the Reference store without source aliases."""
    source_store = CoregisteredSceneStore.open(source)
    destination = Path(target)
    if (destination / "manifest.json").is_file():
        existing_store = CoregisteredSceneStore.open(destination)
        for existing_unit in existing_store.units:
            existing_store.read(existing_unit.tag)
    for unit in source_store.units:
        primary, _, _ = source_store.read(unit.tag)
        write_scene_unit(
            destination,
            date_id=source_store.reference_id,
            reference_id=source_store.reference_id,
            domain=source_store.domain,
            tag=unit.tag,
            primary=primary,
            secondary=primary,
            row_origin=unit.row_origin,
            col_origin=unit.col_origin,
            grid_shape=source_store.grid_shape,
            wavelength_m=source_store.wavelength_m,
            grid_identity=source_store.grid_identity,
            scientific_lineage=unit.scientific_lineage,
            phase_state=unit.phase_state,
            phase_screen=None,
            validate_existing_payloads=False,
        )


def _tag_swath(tag: str) -> str:
    """Extract the stable swath component from a production burst tag."""
    parts = tag.split("_")
    if len(parts) == 2 and parts[1].startswith("b"):
        return parts[0]
    if len(parts) != 3 or not parts[0].startswith("f") or not parts[2].startswith("b"):
        reject_invalid_state(f"scene unit tag has no stable swath identity: {tag!r}")
    return parts[1]


def _validate_nisar_geo_ownership(store: CoregisteredSceneStore) -> None:
    """Validate deterministic, non-overlapping NISAR Geo tile ownership."""
    if store.domain != "geo" or not any(
        unit.tag.startswith("NISAR_b") for unit in store.units
    ):
        return
    policies = {(unit.phase_state or {}).get("coverage_policy") for unit in store.units}
    if policies == {None}:
        # Backward-compatible bounded NISAR generations predate tiled ownership.
        return
    if policies != {"joint_first_valid_row_major_v1"}:
        reject_invalid_state("NISAR Geo scene mixes ownership contracts")
    pair_owned = np.zeros(store.grid_shape, dtype=bool)
    for unit in sorted(store.units, key=lambda item: item.tag):
        primary, secondary, _ = store.read(unit.tag)
        row_slice = slice(unit.row_origin, unit.row_origin + unit.shape[0])
        col_slice = slice(unit.col_origin, unit.col_origin + unit.shape[1])
        primary_valid = (
            np.isfinite(primary.real)
            & np.isfinite(primary.imag)
            & (np.abs(primary) > 0.0)
        )
        secondary_valid = (
            np.isfinite(secondary.real)
            & np.isfinite(secondary.imag)
            & (np.abs(secondary) > 0.0)
        )
        if not np.array_equal(primary_valid, secondary_valid):
            reject_invalid_state(
                f"NISAR Geo scene unit {unit.tag!r} has asymmetric pair ownership"
            )
        declared_count = (unit.phase_state or {}).get("pair_valid_pixels")
        if declared_count != int(np.sum(primary_valid)):
            reject_invalid_state(
                f"NISAR Geo scene unit {unit.tag!r} pair coverage count differs"
            )
        if np.any(pair_owned[row_slice, col_slice] & primary_valid):
            reject_invalid_state("NISAR Geo scene tiles have overlapping ownership")
        pair_owned[row_slice, col_slice] |= primary_valid


def form_merged_scene_interferogram(
    primary_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    *,
    primary_role: str = "primary",
    secondary_role: str = "secondary",
    multilook: tuple[int, int] = (1, 1),
    goldstein_alpha: float = 0.0,
    device: str = "auto",
    dask_client: object | None = None,
    flatten_stage: str = "coregistration",
    coherence_window: tuple[int, int] | None = None,
    phase_filter: PhaseFilter | None = None,
) -> InterferogramProduct:
    """Form one common-grid complex IFG from all persisted scene units.

    The accumulator is the persisted-scene equivalent of the production Pair
    burst merge. Radar swaths are accumulated independently and then composed
    in swath order; geographic units share one common accumulator. Burst
    overlap is resolved before spatial unwrapping and no interpolation kernel
    or feathering is introduced.

    Parameters
    ----------
    primary_store, secondary_store : CoregisteredSceneStore
        Persisted Reference-aligned scene generations on one common grid.
    primary_role, secondary_role : {"primary", "secondary"}, optional
        Payload role selected from each generation.
    multilook : tuple[int, int], optional
        Azimuth and range boxcar look factors aligned to global origins.
    goldstein_alpha : float, optional
        Goldstein filter exponent. Zero disables filtering.
    flatten_stage : {"coregistration", "interferogram"}, optional
        Stage at which the NISAR range-offset phase screen is applied.
    device : str, optional
        Numerical device policy for the qualified Goldstein stage.
    dask_client : object, optional
        Explicitly trusted Dask client for qualified remote Goldstein work.
    coherence_window : tuple[int, int] or None, optional
        Coherence MLE support passed to the pair kernel. ``None`` computes
        direct MLE in each multilook block.
    phase_filter : PhaseFilter or None, optional
        Runtime filter applied after multilooking. ``None`` leaves the
        multilooked complex interferogram unchanged.

    Returns
    -------
    InterferogramProduct
        One merged common-grid interferogram and its quality layers.

    """
    if primary_store.domain != secondary_store.domain:
        reject_invalid_state("scene artifact domains do not match")
    if primary_store.reference_id != secondary_store.reference_id:
        reject_invalid_state("scene artifacts use different References")
    if primary_store.grid_shape != secondary_store.grid_shape:
        reject_invalid_state("scene artifact common grid shapes do not match")
    if primary_store.wavelength_m != secondary_store.wavelength_m:
        reject_invalid_state("scene artifact radar wavelengths do not match")
    if primary_store.grid_identity != secondary_store.grid_identity:
        reject_invalid_state("scene artifact coordinate grids do not match")
    if flatten_stage not in {"coregistration", "interferogram"}:
        reject_invalid_state("unsupported interferogram flattening stage")
    if (
        primary_store.flatten_stage != flatten_stage
        or secondary_store.flatten_stage != flatten_stage
    ):
        reject_invalid_state("scene artifacts mix flattening stages")
    _validate_nisar_geo_ownership(primary_store)
    _validate_nisar_geo_ownership(secondary_store)
    primary_units = primary_store.unit_map()
    secondary_units = secondary_store.unit_map()
    if set(primary_units) != set(secondary_units):
        reject_invalid_state("scene artifact unit manifests differ")
    az_looks, rg_looks = multilook
    if az_looks < 1 or rg_looks < 1:
        reject_invalid_state("multilook factors must be >= 1")
    out_rows = primary_store.grid_shape[0] // az_looks
    out_cols = primary_store.grid_shape[1] // rg_looks
    if out_rows < 1 or out_cols < 1:
        reject_invalid_state("multilook factors exceed the common scene grid")
    if len(primary_units) == 1:
        tag = next(iter(primary_units))
        primary_ref, primary_sec, primary_unit = primary_store.read(tag)
        secondary_ref, secondary_sec, secondary_unit = secondary_store.read(tag)
        if (
            primary_unit.shape == secondary_unit.shape
            and primary_unit.row_origin == secondary_unit.row_origin
            and primary_unit.col_origin == secondary_unit.col_origin
            and primary_unit.row_origin == 0
            and primary_unit.col_origin == 0
            and primary_unit.shape == primary_store.grid_shape
        ):
            valid_roles = {"primary", "secondary"}
            if primary_role not in valid_roles or secondary_role not in valid_roles:
                reject_invalid_state("unsupported scene payload role")
            primary = primary_ref if primary_role == "primary" else primary_sec
            secondary = secondary_ref if secondary_role == "primary" else secondary_sec
            if flatten_stage == "interferogram":
                primary_screen = (
                    primary_store.read_phase_screen(tag)
                    if primary_role == "secondary"
                    else None
                )
                secondary_screen = (
                    secondary_store.read_phase_screen(tag)
                    if secondary_role == "secondary"
                    else None
                )
                if primary_screen is None:
                    primary_screen = np.zeros(primary.shape, dtype=np.float32)
                if secondary_screen is None:
                    secondary_screen = np.zeros(secondary.shape, dtype=np.float32)
                screen = secondary_screen - primary_screen
                # Multiplying the secondary by exp(+j screen) makes the
                # resulting native IFG carry the required exp(-j screen)
                # before the Pair kernel performs multilooking.
                secondary = secondary * np.exp(1j * screen)
            product = form_interferogram(
                primary,
                secondary,
                multilook=multilook,
                coherence_window=coherence_window,
            )
            if phase_filter is not None:
                import torch

                filter_device = device
                if filter_device in {"", "auto"}:
                    filter_device = "cuda" if torch.cuda.is_available() else "cpu"
                tensor = torch.as_tensor(product.complex_ifg, device=filter_device)
                support = torch.as_tensor(
                    product.valid_mask, dtype=torch.bool, device=filter_device
                )
                filtered = _apply_phase_filter_contract(
                    phase_filter, tensor, support
                )
                result_ifg = filtered.interferogram.detach().cpu().numpy()
                result_mask = filtered.valid_mask.detach().cpu().numpy()
                result_ifg[~result_mask] = np.nan + 1j * np.nan
                _, result_coh, result_phase = mask_invalid_looks(
                    result_ifg, product.coherence
                )
                assert result_coh is not None
                return InterferogramProduct(
                    complex_ifg=result_ifg,
                    coherence=result_coh,
                    wrapped_phase=result_phase,
                    amplitude=np.abs(result_ifg).astype(np.float32),
                    valid_mask=result_mask,
                )
            if goldstein_alpha <= 0.0:
                return product
            from faninsar.backends.dask_gpu import should_accelerate

            if (
                dask_client is not None or device.lower() == "cuda"
            ) and should_accelerate(device, dask_client, kernel="goldstein_filter"):
                from faninsar.backends.dask_gpu import run_goldstein_filter

                filtered = run_goldstein_filter(
                    product.complex_ifg,
                    alpha=goldstein_alpha,
                    window=32,
                    device=device,
                    client=dask_client,
                )
            else:
                filtered = goldstein_filter(product.complex_ifg, alpha=goldstein_alpha)
            filtered, coherence, wrapped = mask_invalid_looks(
                filtered,
                product.coherence,
            )
            assert coherence is not None
            return InterferogramProduct(
                complex_ifg=filtered,
                coherence=coherence,
                wrapped_phase=wrapped,
                amplitude=np.abs(filtered).astype(np.float32),
            )

    groups = (
        {"geo": sorted(primary_units)}
        if primary_store.domain == "geo"
        else {
            swath: sorted(
                (tag for tag in primary_units if _tag_swath(tag) == swath),
                key=lambda tag: primary_units[tag].row_origin,
                reverse=True,
            )
            for swath in sorted({_tag_swath(tag) for tag in primary_units})
        }
    )
    group_products: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    looks_per_window = az_looks * rg_looks
    for group, tags in groups.items():
        ifg_acc = np.zeros((out_rows, out_cols), dtype=np.complex128)
        primary_power_acc = np.zeros((out_rows, out_cols), dtype=np.float64)
        secondary_power_acc = np.zeros((out_rows, out_cols), dtype=np.float64)
        claimed = np.zeros((out_rows, out_cols), dtype=np.int32)
        for tag in tags:
            primary_ref, primary_sec, primary_unit = primary_store.read(tag)
            secondary_ref, secondary_sec, secondary_unit = secondary_store.read(tag)
            if (
                primary_unit.shape != secondary_unit.shape
                or primary_unit.row_origin != secondary_unit.row_origin
                or primary_unit.col_origin != secondary_unit.col_origin
            ):
                reject_invalid_state(
                    f"scene unit {tag!r} grid placement differs across dates"
                )
            primary = primary_ref if primary_role == "primary" else primary_sec
            secondary = secondary_ref if secondary_role == "primary" else secondary_sec
            valid_roles = {"primary", "secondary"}
            if primary_role not in valid_roles or secondary_role not in valid_roles:
                reject_invalid_state("unsupported scene payload role")
            ifg = primary * np.conjugate(secondary)
            if flatten_stage == "interferogram":
                primary_screen = (
                    primary_store.read_phase_screen(tag)
                    if primary_role == "secondary"
                    else None
                )
                secondary_screen = (
                    secondary_store.read_phase_screen(tag)
                    if secondary_role == "secondary"
                    else None
                )
                if primary_screen is None:
                    primary_screen = np.zeros(primary.shape, dtype=np.float32)
                if secondary_screen is None:
                    secondary_screen = np.zeros(secondary.shape, dtype=np.float32)
                ifg *= np.exp(-1j * (secondary_screen - primary_screen))
            primary_power = primary.real**2 + primary.imag**2
            secondary_power = secondary.real**2 + secondary.imag**2
            valid = np.isfinite(ifg.real) & np.isfinite(ifg.imag) & (np.abs(ifg) > 0)
            rows = primary_unit.row_origin + np.arange(primary_unit.shape[0])
            cols = primary_unit.col_origin + np.arange(primary_unit.shape[1])
            output_rows = rows[:, None] // az_looks
            output_cols = cols[None, :] // rg_looks
            in_bounds = (
                (output_rows >= 0)
                & (output_rows < out_rows)
                & (output_cols >= 0)
                & (output_cols < out_cols)
                & valid
            )
            row_indices, col_indices = np.broadcast_arrays(output_rows, output_cols)
            valid_rows = row_indices[in_bounds]
            valid_cols = col_indices[in_bounds]
            free = claimed[valid_rows, valid_cols] < looks_per_window
            free_rows = valid_rows[free]
            free_cols = valid_cols[free]
            np.add.at(
                ifg_acc,
                (free_rows, free_cols),
                ifg[in_bounds][free].astype(np.complex128),
            )
            np.add.at(
                primary_power_acc,
                (free_rows, free_cols),
                primary_power[in_bounds][free],
            )
            np.add.at(
                secondary_power_acc,
                (free_rows, free_cols),
                secondary_power[in_bounds][free],
            )
            np.add.at(claimed, (free_rows, free_cols), 1)
        has_data = claimed > 0
        denominator = np.where(has_data, claimed, 1)
        group_ifg = np.where(has_data, ifg_acc / denominator, 0).astype(np.complex64)
        coherence_denominator = np.sqrt(
            np.maximum(
                (primary_power_acc / denominator) * (secondary_power_acc / denominator),
                1e-30,
            )
        )
        group_coherence = np.clip(
            np.abs(group_ifg) / coherence_denominator, 0.0, 1.0
        ).astype(np.float32)
        group_coherence[~has_data] = np.nan
        group_products[group] = group_ifg, group_coherence

    merged_ifg = np.zeros((out_rows, out_cols), dtype=np.complex64)
    coherence = np.full((out_rows, out_cols), np.nan, dtype=np.float32)
    for group in sorted(group_products):
        group_ifg, group_coherence = group_products[group]
        valid = np.abs(group_ifg) > 0
        merged_ifg[valid] = group_ifg[valid]
        coherence[valid] = group_coherence[valid]
    if phase_filter is not None:
        import torch

        filter_device = device
        if filter_device in {"", "auto"}:
            filter_device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.as_tensor(merged_ifg, device=filter_device)
        support = torch.as_tensor(
            np.isfinite(merged_ifg.real) & np.isfinite(merged_ifg.imag),
            dtype=torch.bool,
            device=filter_device,
        )
        filtered = _apply_phase_filter_contract(phase_filter, tensor, support)
        merged_ifg = filtered.interferogram.detach().cpu().numpy()
        filtered_mask = filtered.valid_mask.detach().cpu().numpy()
        merged_ifg[~filtered_mask] = np.nan + 1j * np.nan
    elif goldstein_alpha > 0.0:
        from faninsar.backends.dask_gpu import should_accelerate

        if (dask_client is not None or device.lower() == "cuda") and should_accelerate(
            device, dask_client, kernel="goldstein_filter"
        ):
            from faninsar.backends.dask_gpu import run_goldstein_filter

            merged_ifg = run_goldstein_filter(
                merged_ifg,
                alpha=goldstein_alpha,
                window=32,
                device=device,
                client=dask_client,
            )
        else:
            merged_ifg = goldstein_filter(merged_ifg, alpha=goldstein_alpha)
    merged_ifg, coherence_out, wrapped = mask_invalid_looks(merged_ifg, coherence)
    if phase_filter is not None:
        valid_mask = filtered_mask & np.isfinite(merged_ifg.real) & np.isfinite(
            merged_ifg.imag
        )
    else:
        valid_mask = np.isfinite(merged_ifg.real) & np.isfinite(merged_ifg.imag)
    assert coherence_out is not None
    return InterferogramProduct(
        complex_ifg=merged_ifg,
        coherence=coherence_out,
        wrapped_phase=wrapped,
        amplitude=np.abs(merged_ifg).astype(np.float32),
        valid_mask=valid_mask,
    )


def form_scene_interferograms(
    primary_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    *,
    primary_role: str = "primary",
    secondary_role: str = "secondary",
    multilook: tuple[int, int] = (1, 1),
    goldstein_alpha: float = 0.0,
    device: str = "auto",
    dask_client: object | None = None,
) -> dict[str, np.ndarray]:
    """Form derived IFGs from aligned scene payloads only.

    Parameters
    ----------
    primary_store, secondary_store : CoregisteredSceneStore
        Persisted Reference-aligned scene generations.
    primary_role, secondary_role : str, optional
        Payload role to consume from each generation.
    multilook : tuple[int, int], optional
        Azimuth and range looks applied through the Pair interferogram kernel.
    goldstein_alpha : float, optional
        Goldstein filter exponent. Zero disables filtering.
    device : str, optional
        Numerical device policy for the qualified Goldstein stage.
    dask_client : object, optional
        Explicitly trusted Dask client for qualified remote Goldstein work.

    Returns
    -------
    dict[str, numpy.ndarray]
        Multilooked and optionally filtered complex interferograms by burst tag.

    """
    if primary_store.domain != secondary_store.domain:
        reject_invalid_state("scene artifact domains do not match")
    primary_units = primary_store.unit_map()
    secondary_units = secondary_store.unit_map()
    if set(primary_units) != set(secondary_units):
        reject_invalid_state("scene artifact unit manifests differ")
    outputs: dict[str, np.ndarray] = {}
    for tag in sorted(primary_units):
        primary_ref, primary_sec, primary_unit = primary_store.read(tag)
        secondary_ref, secondary_sec, secondary_unit = secondary_store.read(tag)
        if primary_unit.shape != secondary_unit.shape:
            reject_invalid_state(f"scene unit {tag!r} shape differs across dates")
        if primary_role == "primary":
            primary = primary_ref
        elif primary_role == "secondary":
            primary = primary_sec
        else:
            reject_invalid_state("unsupported primary scene role")
        if secondary_role == "primary":
            secondary = secondary_ref
        elif secondary_role == "secondary":
            secondary = secondary_sec
        else:
            reject_invalid_state("unsupported secondary scene role")
        complex_ifg = form_interferogram(
            primary,
            secondary,
            multilook=multilook,
        ).complex_ifg
        if goldstein_alpha > 0.0:
            from faninsar.backends.dask_gpu import should_accelerate

            if (
                dask_client is not None or device.lower() == "cuda"
            ) and should_accelerate(device, dask_client, kernel="goldstein_filter"):
                from faninsar.backends.dask_gpu import run_goldstein_filter

                complex_ifg = run_goldstein_filter(
                    complex_ifg,
                    alpha=goldstein_alpha,
                    window=32,
                    device=device,
                    client=dask_client,
                )
            else:
                complex_ifg = goldstein_filter(complex_ifg, alpha=goldstein_alpha)
        outputs[tag] = np.asarray(complex_ifg, dtype=np.complex64)
    return outputs


__all__ = [
    "SCENE_SCHEMA",
    "CoregisteredSceneStore",
    "SceneUnit",
    "copy_reference_units",
    "form_merged_scene_interferogram",
    "form_scene_interferograms",
    "scene_grid_identity",
    "write_scene_unit",
]
