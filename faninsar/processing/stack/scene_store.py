"""Manifest-bound master-aligned SLC scene storage for Stack formation."""

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
    from collections.abc import Mapping


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
    master_id: str
    domain: str
    units: tuple[SceneUnit, ...]
    manifest_digest: str
    grid_shape: tuple[int, int]
    wavelength_m: float | None
    grid_identity: str

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
        raw_units = manifest.get("units")
        if not isinstance(raw_units, list):
            reject_invalid_state("scene manifest units must be a list")
        units: list[SceneUnit] = []
        for raw in raw_units:
            if not isinstance(raw, dict):
                reject_invalid_state("scene manifest unit must be an object")
            reference = _safe_payload_path(
                path, raw.get("reference_file"), "reference_file"
            )
            secondary = _safe_payload_path(
                path, raw.get("secondary_file"), "secondary_file"
            )
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
            master_id=str(manifest["master_id"]),
            domain=domain,
            units=tuple(units),
            manifest_digest=actual_digest,
            grid_shape=grid_shape,
            wavelength_m=wavelength_m,
            grid_identity=grid_identity,
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
    grid_shape: tuple[int, int] | None = None,
    wavelength_m: float | None = None,
    grid_identity: str | None = None,
) -> None:
    """Atomically add one aligned unit and publish a complete manifest."""
    path = _validated_store_root(root, create=True)
    _require_basename(tag, "tag")
    if (
        reference.ndim != 2
        or secondary.ndim != 2
        or reference.shape != secondary.shape
        or any(size <= 0 for size in reference.shape)
        or reference.dtype != np.complex64
        or secondary.dtype != np.complex64
    ):
        reject_invalid_state("aligned scene arrays must be matching complex64 arrays")
    resolved_grid_shape = grid_shape or (
        int(row_origin) + int(reference.shape[0]),
        int(col_origin) + int(reference.shape[1]),
    )
    if (
        len(resolved_grid_shape) != 2
        or any(size <= 0 for size in resolved_grid_shape)
        or row_origin < 0
        or col_origin < 0
        or row_origin + reference.shape[0] > resolved_grid_shape[0]
        or col_origin + reference.shape[1] > resolved_grid_shape[1]
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
        for existing_unit in existing_store.units:
            existing_store.read(existing_unit.tag)
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("date_id") != date_id
            or existing.get("master_id") != master_id
            or existing.get("domain") != domain
        ):
            reject_invalid_state("scene units must share date, master, and domain")
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
    ref_path = path / f"{tag}.reference.npy"
    sec_path = path / f"{tag}.secondary.npy"
    for target, array in ((ref_path, reference), (sec_path, secondary)):
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(array, dtype=np.complex64), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    units = [item for item in existing.get("units", []) if item.get("tag") != tag]
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
        "grid_shape": list(resolved_grid_shape),
        "wavelength_m": wavelength_m,
        "grid_identity": resolved_grid_identity,
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
            grid_shape=source_store.grid_shape,
            wavelength_m=source_store.wavelength_m,
            grid_identity=source_store.grid_identity,
        )


def _tag_swath(tag: str) -> str:
    """Extract the stable swath component from a production burst tag."""
    parts = tag.split("_")
    if len(parts) == 2 and parts[1].startswith("b"):
        return parts[0]
    if len(parts) != 3 or not parts[0].startswith("f") or not parts[2].startswith("b"):
        reject_invalid_state(f"scene unit tag has no stable swath identity: {tag!r}")
    return parts[1]


def form_merged_scene_interferogram(
    reference_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    *,
    reference_role: str = "reference",
    secondary_role: str = "secondary",
    multilook: tuple[int, int] = (1, 1),
    goldstein_alpha: float = 0.0,
) -> InterferogramProduct:
    """Form one common-grid complex IFG from all persisted scene units.

    The accumulator is the persisted-scene equivalent of the production Pair
    burst merge. Radar swaths are accumulated independently and then composed
    in swath order; geographic units share one common accumulator. Burst
    overlap is resolved before spatial unwrapping and no interpolation kernel
    or feathering is introduced.

    Parameters
    ----------
    reference_store, secondary_store : CoregisteredSceneStore
        Persisted master-aligned scene generations on one common grid.
    reference_role, secondary_role : {"reference", "secondary"}, optional
        Payload role selected from each generation.
    multilook : tuple[int, int], optional
        Azimuth and range boxcar look factors aligned to global origins.
    goldstein_alpha : float, optional
        Goldstein filter exponent. Zero disables filtering.

    Returns
    -------
    InterferogramProduct
        One merged common-grid interferogram and its quality layers.

    """
    if reference_store.domain != secondary_store.domain:
        reject_invalid_state("scene artifact domains do not match")
    if reference_store.master_id != secondary_store.master_id:
        reject_invalid_state("scene artifacts use different alignment masters")
    if reference_store.grid_shape != secondary_store.grid_shape:
        reject_invalid_state("scene artifact common grid shapes do not match")
    if reference_store.wavelength_m != secondary_store.wavelength_m:
        reject_invalid_state("scene artifact radar wavelengths do not match")
    if reference_store.grid_identity != secondary_store.grid_identity:
        reject_invalid_state("scene artifact coordinate grids do not match")
    reference_units = reference_store.unit_map()
    secondary_units = secondary_store.unit_map()
    if set(reference_units) != set(secondary_units):
        reject_invalid_state("scene artifact unit manifests differ")
    az_looks, rg_looks = multilook
    if az_looks < 1 or rg_looks < 1:
        reject_invalid_state("multilook factors must be >= 1")
    out_rows = reference_store.grid_shape[0] // az_looks
    out_cols = reference_store.grid_shape[1] // rg_looks
    if out_rows < 1 or out_cols < 1:
        reject_invalid_state("multilook factors exceed the common scene grid")
    if len(reference_units) == 1:
        tag = next(iter(reference_units))
        reference_ref, reference_sec, reference_unit = reference_store.read(tag)
        secondary_ref, secondary_sec, secondary_unit = secondary_store.read(tag)
        if (
            reference_unit.shape == secondary_unit.shape
            and reference_unit.row_origin == secondary_unit.row_origin
            and reference_unit.col_origin == secondary_unit.col_origin
            and reference_unit.row_origin == 0
            and reference_unit.col_origin == 0
            and reference_unit.shape == reference_store.grid_shape
        ):
            valid_roles = {"reference", "secondary"}
            if reference_role not in valid_roles or secondary_role not in valid_roles:
                reject_invalid_state("unsupported scene payload role")
            reference = (
                reference_ref if reference_role == "reference" else reference_sec
            )
            secondary = (
                secondary_ref if secondary_role == "reference" else secondary_sec
            )
            product = form_interferogram(
                reference,
                secondary,
                multilook=multilook,
            )
            if goldstein_alpha <= 0.0:
                return product
            filtered = goldstein_filter(
                product.complex_ifg,
                alpha=goldstein_alpha,
            )
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
        {"geo": sorted(reference_units)}
        if reference_store.domain == "geo"
        else {
            swath: sorted(
                (tag for tag in reference_units if _tag_swath(tag) == swath),
                key=lambda tag: reference_units[tag].row_origin,
                reverse=True,
            )
            for swath in sorted({_tag_swath(tag) for tag in reference_units})
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
            reference_ref, reference_sec, reference_unit = reference_store.read(tag)
            secondary_ref, secondary_sec, secondary_unit = secondary_store.read(tag)
            if (
                reference_unit.shape != secondary_unit.shape
                or reference_unit.row_origin != secondary_unit.row_origin
                or reference_unit.col_origin != secondary_unit.col_origin
            ):
                reject_invalid_state(
                    f"scene unit {tag!r} grid placement differs across dates"
                )
            reference = (
                reference_ref if reference_role == "reference" else reference_sec
            )
            secondary = (
                secondary_ref if secondary_role == "reference" else secondary_sec
            )
            valid_roles = {"reference", "secondary"}
            if reference_role not in valid_roles or secondary_role not in valid_roles:
                reject_invalid_state("unsupported scene payload role")
            ifg = reference * np.conjugate(secondary)
            primary_power = reference.real**2 + reference.imag**2
            secondary_power = secondary.real**2 + secondary.imag**2
            valid = np.isfinite(ifg.real) & np.isfinite(ifg.imag) & (np.abs(ifg) > 0)
            rows = reference_unit.row_origin + np.arange(reference_unit.shape[0])
            cols = reference_unit.col_origin + np.arange(reference_unit.shape[1])
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
    if goldstein_alpha > 0.0:
        merged_ifg = goldstein_filter(merged_ifg, alpha=goldstein_alpha)
    merged_ifg, coherence_out, wrapped = mask_invalid_looks(merged_ifg, coherence)
    assert coherence_out is not None
    return InterferogramProduct(
        complex_ifg=merged_ifg,
        coherence=coherence_out,
        wrapped_phase=wrapped,
        amplitude=np.abs(merged_ifg).astype(np.float32),
    )


def form_scene_interferograms(
    reference_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    *,
    reference_role: str = "reference",
    secondary_role: str = "secondary",
    multilook: tuple[int, int] = (1, 1),
    goldstein_alpha: float = 0.0,
) -> dict[str, np.ndarray]:
    """Form derived IFGs from aligned scene payloads only.

    Parameters
    ----------
    reference_store, secondary_store : CoregisteredSceneStore
        Persisted master-aligned scene generations.
    reference_role, secondary_role : str, optional
        Payload role to consume from each generation.
    multilook : tuple[int, int], optional
        Azimuth and range looks applied through the Pair interferogram kernel.
    goldstein_alpha : float, optional
        Goldstein filter exponent. Zero disables filtering.

    Returns
    -------
    dict[str, numpy.ndarray]
        Multilooked and optionally filtered complex interferograms by burst tag.

    """
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
        complex_ifg = form_interferogram(
            reference,
            secondary,
            multilook=multilook,
        ).complex_ifg
        if goldstein_alpha > 0.0:
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
