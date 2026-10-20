"""Reference-aligned SLC scene products for Stack formation (PROPOSAL-0059)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np

from faninsar.io.storage._zarr_stac import (
    read_zarr_array,
    validate_spatial_metadata,
    write_zarr_array,
)
from faninsar.io.storage.product import ProductWriter, open_product
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.interferometry.pair import (
    InterferogramProduct,
    _sliding_coherence,
    form_interferogram,
    goldstein_filter,
    mask_invalid_looks,
)

SCENE_KIND = "scene"

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


def _scene_role(date_id: str, reference_id: str) -> str:
    """Return the persisted role of one acquisition product."""
    return "reference" if date_id == reference_id else "secondary"


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


@dataclass(frozen=True, slots=True)
class SceneUnit:
    """One Reference-grid aligned SLC unit (burst or tile) of one acquisition."""

    tag: str
    payload_path: Path
    shape: tuple[int, int]
    row_origin: int
    col_origin: int
    scientific_lineage: tuple[dict[str, str], ...] = ()
    phase_state: dict[str, object] | None = None
    phase_screen_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate the unit's bounded shape."""
        if (
            not self.tag
            or len(self.shape) != 2
            or any(size <= 0 for size in self.shape)
        ):
            reject_invalid_state("scene unit has an invalid tag or shape")


@dataclass(frozen=True, slots=True)
class CoregisteredSceneStore:
    """One committed scene product: an acquisition on the Reference grid.

    The Stack Reference is one product (``faninsar:role == "reference"``);
    every other acquisition is one secondary product that binds the
    Reference ``write_id`` it was aligned against.
    """

    root: Path
    date_id: str
    reference_id: str
    domain: str
    units: tuple[SceneUnit, ...]
    write_id: str
    grid_shape: tuple[int, int]
    wavelength_m: float | None
    grid_identity: str
    crs: str | None = None
    transform: tuple[float, ...] | None = None
    flatten_stage: str = "coregistration"
    inputs: dict[str, str] = field(default_factory=dict)
    properties: dict[str, object] = field(default_factory=dict, repr=False)

    @property
    def manifest_digest(self) -> str:
        """Return the commit identity used by downstream lineage records."""
        return self.write_id

    @classmethod
    def open(cls, root: str | Path) -> CoregisteredSceneStore:
        """Open one committed scene product.

        Parameters
        ----------
        root : path-like
            Scene product directory containing ``item.json``.

        Returns
        -------
        CoregisteredSceneStore
            Read-only view of the product metadata.

        Raises
        ------
        InvalidProcessingStateError
            If the directory is not a committed scene product or its metadata
            is inconsistent.

        """
        product = open_product(root, kind=SCENE_KIND)
        properties = dict(product.properties)
        date_id = str(properties["faninsar:date"])
        reference_id = str(properties["faninsar:reference"])
        role = properties.get("faninsar:role")
        if role != _scene_role(date_id, reference_id):
            reject_invalid_state("scene product role does not match its date")
        domain = properties.get("faninsar:domain")
        if domain not in {"radar", "geo"}:
            reject_invalid_state("scene product domain is invalid")
        raw_shape = properties.get("faninsar:shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 2:
            reject_invalid_state("scene product grid shape is invalid")
        grid_shape = (int(raw_shape[0]), int(raw_shape[1]))
        units: list[SceneUnit] = []
        for name, asset in sorted(product.assets.items()):
            if asset.get("faninsar:role") != "slc":
                continue
            phase_name = asset.get("faninsar:phase_screen")
            unit = SceneUnit(
                tag=str(name),
                payload_path=product.asset_path(name),
                shape=(int(asset["faninsar:rows"]), int(asset["faninsar:cols"])),
                row_origin=int(asset["faninsar:row_origin"]),
                col_origin=int(asset["faninsar:col_origin"]),
                scientific_lineage=tuple(
                    item
                    for item in asset.get("faninsar:lineage", [])
                    if isinstance(item, dict)
                ),
                phase_state=(
                    dict(asset["faninsar:phase_state"])
                    if isinstance(asset.get("faninsar:phase_state"), dict)
                    else None
                ),
                phase_screen_path=(
                    product.asset_path(str(phase_name)) if phase_name else None
                ),
            )
            if (
                unit.row_origin < 0
                or unit.col_origin < 0
                or unit.row_origin + unit.shape[0] > grid_shape[0]
                or unit.col_origin + unit.shape[1] > grid_shape[1]
            ):
                reject_invalid_state("scene unit lies outside the declared common grid")
            units.append(unit)
        if not units:
            reject_invalid_state("scene product has no units")
        flatten_stage = str(properties.get("faninsar:flatten_stage", "coregistration"))
        if flatten_stage not in {"coregistration", "interferogram"}:
            reject_invalid_state("scene product flatten_stage is unsupported")
        raw_wavelength = properties.get("faninsar:wavelength_m")
        wavelength_m = None if raw_wavelength is None else float(raw_wavelength)
        _, _, grid_identity, crs, transform = validate_spatial_metadata(
            domain,
            grid_shape,
            str(properties.get("faninsar:grid_identity")),
            crs=properties.get("proj:code"),
            transform=properties.get("proj:transform"),
        )
        return cls(
            root=Path(root),
            date_id=date_id,
            reference_id=reference_id,
            domain=domain,
            units=tuple(units),
            write_id=product.write_id,
            grid_shape=grid_shape,
            wavelength_m=wavelength_m,
            grid_identity=grid_identity,
            crs=crs,
            transform=transform,
            flatten_stage=flatten_stage,
            inputs=product.inputs,
            properties=properties,
        )

    @property
    def role(self) -> str:
        """Return ``"reference"`` for the Reference product, else ``"secondary"``."""
        return _scene_role(self.date_id, self.reference_id)

    @property
    def is_reference(self) -> bool:
        """Return whether this product holds the Stack Reference acquisition."""
        return self.date_id == self.reference_id

    @property
    def radar_geometry(self) -> dict[str, object] | None:
        """Return the persisted Reference frame geometry record, if any."""
        value = self.properties.get("faninsar:radar_geometry")
        return dict(value) if isinstance(value, dict) else None

    def stale_inputs(self, current: Mapping[str, object]) -> list[str]:
        """Return the bound upstream names whose ``write_id`` changed.

        Parameters
        ----------
        current : mapping
            Current upstream products or write ids by binding name.

        """
        stale = []
        for name, write_id in sorted(self.inputs.items()):
            upstream = current.get(name)
            upstream_id = getattr(upstream, "write_id", upstream)
            if upstream_id != write_id:
                stale.append(name)
        return stale

    def read_phase_screen(self, tag: str) -> np.ndarray | None:
        """Read one persisted range-offset phase screen, if present."""
        unit = self.unit_map().get(tag)
        if unit is None:
            reject_invalid_state(f"scene unit {tag!r} is not in the product")
        if unit.phase_screen_path is None:
            return None
        screen = read_zarr_array(unit.phase_screen_path)
        if screen.shape != unit.shape or screen.dtype != np.float32:
            reject_invalid_state(f"scene phase screen shape mismatch for {tag!r}")
        return np.asarray(screen, dtype=np.float32)

    def unit_map(self) -> dict[str, SceneUnit]:
        """Return units keyed by their stable burst tag."""
        return {unit.tag: unit for unit in self.units}

    def read(self, tag: str) -> tuple[np.ndarray, SceneUnit]:
        """Read one unit payload and validate its shape and dtype.

        Parameters
        ----------
        tag : str
            Stable unit identifier within the product.

        Returns
        -------
        tuple[numpy.ndarray, SceneUnit]
            The complex64 SLC payload of this acquisition and its unit record.

        Raises
        ------
        InvalidProcessingStateError
            If the unit is unknown or its shape or dtype differs from the item.

        """
        unit = self.unit_map().get(tag)
        if unit is None:
            reject_invalid_state(f"scene unit {tag!r} is not in the product")
        payload = read_zarr_array(unit.payload_path)
        if payload.shape != unit.shape:
            reject_invalid_state(f"scene unit {tag!r} shape mismatch")
        if payload.dtype != np.complex64:
            reject_invalid_state(f"scene unit {tag!r} dtype mismatch")
        return payload, unit


@dataclass(slots=True)
class _PendingUnit:
    """One staged unit awaiting product commit."""

    tag: str
    rows: int
    cols: int
    row_origin: int
    col_origin: int
    lineage: list[dict[str, str]]
    phase_state: dict[str, object] | None
    phase_screen: bool


class SceneWriter:
    """Stage the units of one scene product and commit it once.

    Every unit of an acquisition is written into one staging directory; the
    product (one ``item.json``) is published by :meth:`commit`. A crash
    before commit leaves no scene product, so a partially coregistered date
    is never mistaken for a complete one.

    Parameters
    ----------
    root : path-like
        Final scene product directory, ``<work_dir>/coreg/<date>``.
    date_id, reference_id : str
        Acquisition and Stack Reference identifiers.
    domain : {"radar", "geo"}
        Coordinate domain.
    grid_shape : tuple[int, int]
        Common scene grid dimensions.
    wavelength_m : float, optional
        Radar wavelength in metres.
    grid_identity : str, optional
        Canonical grid SHA-256; required for geo, derived for radar.
    crs, transform : optional
        Geographic CRS and affine transform (geo only).
    overwrite : bool, optional
        Replace an existing product on commit.

    """

    def __init__(
        self,
        root: str | Path,
        *,
        date_id: str,
        reference_id: str,
        domain: str,
        grid_shape: tuple[int, int],
        wavelength_m: float | None = None,
        grid_identity: str | None = None,
        crs: object = None,
        transform: object = None,
        overwrite: bool = False,
    ) -> None:
        """Validate the product header and open its staging directory."""
        if domain not in {"radar", "geo"}:
            reject_invalid_state("scene domain must be radar or geo")
        if domain == "geo" and grid_identity is None:
            reject_invalid_state(
                "Geo scene publication requires an explicit grid_identity"
            )
        if wavelength_m is not None and (
            not np.isfinite(wavelength_m) or wavelength_m <= 0.0
        ):
            reject_invalid_state("scene wavelength_m must be finite and positive")
        shape = (int(grid_shape[0]), int(grid_shape[1]))
        if any(size <= 0 for size in shape):
            reject_invalid_state("scene grid shape must be positive")
        identity = grid_identity or scene_grid_identity(domain, shape)
        _, _, identity, resolved_crs, resolved_transform = validate_spatial_metadata(
            domain, shape, identity, crs=crs, transform=transform
        )
        self.date_id = str(date_id)
        self.reference_id = str(reference_id)
        self.domain = domain
        self.grid_shape = shape
        self.wavelength_m = None if wavelength_m is None else float(wavelength_m)
        self.grid_identity = identity
        self.crs = resolved_crs
        self.transform = resolved_transform
        self._writer = ProductWriter(root, kind=SCENE_KIND, overwrite=overwrite)
        self._units: dict[str, _PendingUnit] = {}

    @property
    def staging(self) -> Path:
        """Return the staging directory (for scratch files owned by the unit)."""
        return self._writer.staging

    def add_unit(
        self,
        *,
        tag: str,
        payload: np.ndarray,
        row_origin: int,
        col_origin: int,
        scientific_lineage: Sequence[Mapping[str, str]] | None = None,
        phase_state: Mapping[str, object] | None = None,
        phase_screen: np.ndarray | None = None,
    ) -> None:
        """Stage one aligned unit payload."""
        _require_basename(tag, "tag")
        if tag in self._units:
            reject_invalid_state(f"scene unit {tag!r} staged twice")
        if (
            payload.ndim != 2
            or any(size <= 0 for size in payload.shape)
            or payload.dtype != np.complex64
        ):
            reject_invalid_state("aligned scene payload must be a 2-D complex64 array")
        if (
            row_origin < 0
            or col_origin < 0
            or row_origin + payload.shape[0] > self.grid_shape[0]
            or col_origin + payload.shape[1] > self.grid_shape[1]
        ):
            reject_invalid_state("scene unit lies outside the declared common grid")
        if phase_screen is not None and (
            phase_screen.ndim != 2
            or phase_screen.shape != payload.shape
            or phase_screen.dtype != np.float32
            or not np.all(np.isfinite(phase_screen))
        ):
            reject_invalid_state("phase screen must be finite matching float32 array")
        write_zarr_array(self.staging / f"{tag}.slc.zarr", payload)
        if phase_screen is not None:
            write_zarr_array(self.staging / f"{tag}.phase.zarr", phase_screen)
        self._units[tag] = _PendingUnit(
            tag=tag,
            rows=int(payload.shape[0]),
            cols=int(payload.shape[1]),
            row_origin=int(row_origin),
            col_origin=int(col_origin),
            lineage=[dict(item) for item in (scientific_lineage or ())],
            phase_state=dict(phase_state) if phase_state is not None else None,
            phase_screen=phase_screen is not None,
        )

    def commit(
        self,
        *,
        inputs: Mapping[str, object] | None = None,
        properties: Mapping[str, object] | None = None,
    ) -> CoregisteredSceneStore:
        """Publish the scene product and return it.

        Parameters
        ----------
        inputs : mapping, optional
            Upstream products or write ids (secondaries bind ``reference``
            and, when NESD constants were applied, ``misreg``).
        properties : mapping, optional
            Extra ``faninsar:`` properties (radar geometry, coregistration
            record).

        """
        if not self._units:
            self._writer.abort()
            reject_invalid_state("scene product has no units")
        stages = {
            str((unit.phase_state or {}).get("flatten_stage", "coregistration"))
            for unit in self._units.values()
        }
        if len(stages) != 1:
            self._writer.abort()
            reject_invalid_state("scene units mix flattening stages")
        flatten_stage = next(iter(stages))
        if flatten_stage not in {"coregistration", "interferogram"}:
            self._writer.abort()
            reject_invalid_state("scene flatten_stage is unsupported")
        for tag, unit in sorted(self._units.items()):
            fields: dict[str, object] = {
                "faninsar:role": "slc",
                "faninsar:rows": unit.rows,
                "faninsar:cols": unit.cols,
                "faninsar:row_origin": unit.row_origin,
                "faninsar:col_origin": unit.col_origin,
                "faninsar:lineage": unit.lineage,
                "faninsar:phase_state": unit.phase_state,
            }
            if unit.phase_screen:
                fields["faninsar:phase_screen"] = f"{tag}.phase"
                self._writer.add_asset(
                    f"{tag}.phase", f"{tag}.phase.zarr", **{"faninsar:role": "phase"}
                )
            self._writer.add_asset(tag, f"{tag}.slc.zarr", **fields)
        header: dict[str, object] = {
            "faninsar:date": self.date_id,
            "faninsar:reference": self.reference_id,
            "faninsar:role": _scene_role(self.date_id, self.reference_id),
            "faninsar:domain": self.domain,
            "faninsar:shape": list(self.grid_shape),
            "faninsar:wavelength_m": self.wavelength_m,
            "faninsar:grid_identity": self.grid_identity,
            "faninsar:flatten_stage": flatten_stage,
            **dict(properties or {}),
        }
        if self.crs is not None:
            header["proj:code"] = self.crs
            header["proj:transform"] = list(self.transform or ())
        product = self._writer.commit(header, inputs=inputs)  # type: ignore[arg-type]
        return CoregisteredSceneStore.open(product.path)

    def abort(self) -> None:
        """Discard the staged product."""
        self._writer.abort()

    def __enter__(self) -> Self:
        """Return the writer."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Abort an uncommitted product when the block raised."""
        self._writer.__exit__(exc_type, exc, traceback)


def write_scene(
    root: str | Path,
    *,
    date_id: str,
    reference_id: str,
    domain: str,
    units: Sequence[Mapping[str, object]],
    grid_shape: tuple[int, int] | None = None,
    wavelength_m: float | None = None,
    grid_identity: str | None = None,
    crs: object = None,
    transform: object = None,
    inputs: Mapping[str, object] | None = None,
    properties: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> CoregisteredSceneStore:
    """Write a complete scene product from in-memory units in one call.

    Parameters
    ----------
    root : path-like
        Final scene product directory.
    date_id, reference_id : str
        Acquisition and Stack Reference identifiers.
    domain : {"radar", "geo"}
        Coordinate domain.
    units : sequence of mapping
        Each mapping holds the :meth:`SceneWriter.add_unit` keyword arguments
        (``tag``, ``payload``, ``row_origin``, ``col_origin`` and optional
        lineage, phase state, and phase screen).
    grid_shape : tuple[int, int], optional
        Common grid; defaults to the union extent of the units.
    wavelength_m : float, optional
        Radar wavelength in metres.
    grid_identity : str, optional
        Canonical grid SHA-256 (required for geo).
    crs, transform : optional
        Geographic CRS and affine transform (geo only).
    inputs : mapping, optional
        Upstream products or write ids.
    properties : mapping, optional
        Extra item properties.
    overwrite : bool, optional
        Replace an existing product.

    Returns
    -------
    CoregisteredSceneStore
        The committed scene product.

    """
    if grid_shape is None:
        grid_shape = (
            max(int(u["row_origin"]) + int(u["payload"].shape[0]) for u in units),  # type: ignore[union-attr]
            max(int(u["col_origin"]) + int(u["payload"].shape[1]) for u in units),  # type: ignore[union-attr]
        )
    with SceneWriter(
        root,
        date_id=date_id,
        reference_id=reference_id,
        domain=domain,
        grid_shape=grid_shape,
        wavelength_m=wavelength_m,
        grid_identity=grid_identity,
        crs=crs,
        transform=transform,
        overwrite=overwrite,
    ) as writer:
        for unit in units:
            writer.add_unit(**unit)  # type: ignore[arg-type]
        return writer.commit(inputs=inputs, properties=properties)


def validate_reference_binding(
    reference_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
) -> None:
    """Validate that a secondary product was aligned against this Reference.

    Raises
    ------
    InvalidProcessingStateError
        If the products disagree on Reference, grid, domain, or unit
        placement, or the secondary binds another Reference ``write_id``.

    """
    if not reference_store.is_reference:
        reject_invalid_state("scene Reference binding requires a Reference product")
    if secondary_store.is_reference:
        reject_invalid_state("scene Reference binding requires a secondary product")
    _validate_store_compatibility(reference_store, secondary_store)
    if secondary_store.inputs.get("reference") != reference_store.write_id:
        reject_invalid_state(
            f"scene {secondary_store.date_id} was aligned against another "
            "Reference; recoregister it with overwrite=True"
        )
    reference_units = reference_store.unit_map()
    for tag, unit in secondary_store.unit_map().items():
        reference_unit = reference_units.get(tag)
        if reference_unit is None:
            reject_invalid_state(f"secondary scene unit {tag!r} has no Reference unit")
        if (
            unit.shape != reference_unit.shape
            or unit.row_origin != reference_unit.row_origin
            or unit.col_origin != reference_unit.col_origin
        ):
            reject_invalid_state(
                f"scene unit {tag!r} grid placement differs from the Reference"
            )


def _validate_store_compatibility(
    first: CoregisteredSceneStore,
    second: CoregisteredSceneStore,
) -> None:
    """Reject scene products that do not share one Reference grid."""
    if first.domain != second.domain:
        reject_invalid_state("scene artifact domains do not match")
    if first.reference_id != second.reference_id:
        reject_invalid_state("scene artifacts use different References")
    if first.grid_shape != second.grid_shape:
        reject_invalid_state("scene artifact common grid shapes do not match")
    if first.wavelength_m != second.wavelength_m:
        reject_invalid_state("scene artifact radar wavelengths do not match")
    if first.grid_identity != second.grid_identity:
        reject_invalid_state("scene artifact coordinate grids do not match")


def _tag_swath(tag: str) -> str:
    """Extract the stable swath component from a production burst tag."""
    parts = tag.split("_")
    if len(parts) == 2 and parts[1].startswith("b"):
        return parts[0]
    if len(parts) != 3 or not parts[0].startswith("f") or not parts[2].startswith("b"):
        reject_invalid_state(f"scene unit tag has no stable swath identity: {tag!r}")
    return parts[1]


NISAR_GEO_COVERAGE_POLICY = "reference_first_valid_row_major_v1"


def valid_samples(values: np.ndarray) -> np.ndarray:
    """Return the finite, non-zero support mask of one complex payload."""
    return np.isfinite(values.real) & np.isfinite(values.imag) & (np.abs(values) > 0)


def _requires_nisar_geo_ownership(store: CoregisteredSceneStore) -> bool:
    """Return whether a store carries tiled NISAR Geo ownership metadata."""
    if store.domain != "geo" or not any(
        unit.tag.startswith("NISAR_b") for unit in store.units
    ):
        return False
    policies = {(unit.phase_state or {}).get("coverage_policy") for unit in store.units}
    if policies == {None}:
        # Bounded single-window NISAR generations have no tiled ownership.
        return False
    if policies != {NISAR_GEO_COVERAGE_POLICY}:
        reject_invalid_state("NISAR Geo scene mixes ownership contracts")
    return True


def _validate_nisar_geo_ownership(
    store: CoregisteredSceneStore,
    payloads: Mapping[str, np.ndarray],
) -> None:
    """Validate deterministic, non-overlapping NISAR Geo tile ownership.

    Ownership is decided by the Reference alone: each Reference sample is
    owned by the first tile (in tag order) that covers it with valid data, and
    a secondary tile only carries samples its Reference tile owns. Both
    therefore satisfy the same non-overlap and declared-count invariants.
    """
    owned = np.zeros(store.grid_shape, dtype=bool)
    for unit in sorted(store.units, key=lambda item: item.tag):
        valid = valid_samples(payloads[unit.tag])
        declared_count = (unit.phase_state or {}).get("valid_pixels")
        if declared_count != int(np.sum(valid)):
            reject_invalid_state(
                f"NISAR Geo scene unit {unit.tag!r} coverage count differs"
            )
        row_slice = slice(unit.row_origin, unit.row_origin + unit.shape[0])
        col_slice = slice(unit.col_origin, unit.col_origin + unit.shape[1])
        if np.any(owned[row_slice, col_slice] & valid):
            reject_invalid_state("NISAR Geo scene tiles have overlapping ownership")
        owned[row_slice, col_slice] |= valid


def _validate_pair_binding(
    primary_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
) -> list[str]:
    """Validate that two acquisition products share one Reference.

    Returns
    -------
    list[str]
        Sorted unit tags present in both products. A unit one date does not
        cover is left blank in the interferogram.

    """
    _validate_store_compatibility(primary_store, secondary_store)
    primary_units = primary_store.unit_map()
    secondary_units = secondary_store.unit_map()
    common = sorted(set(primary_units) & set(secondary_units))
    if not common:
        reject_invalid_state("scene products share no unit")
    for tag in common:
        primary_unit = primary_units[tag]
        secondary_unit = secondary_units[tag]
        if (
            primary_unit.shape != secondary_unit.shape
            or primary_unit.row_origin != secondary_unit.row_origin
            or primary_unit.col_origin != secondary_unit.col_origin
        ):
            reject_invalid_state(
                f"scene unit {tag!r} grid placement differs across dates"
            )
    bound = {
        store.write_id if store.is_reference else store.inputs.get("reference")
        for store in (primary_store, secondary_store)
    }
    if len(bound) != 1:
        reject_invalid_state(
            "scene products were aligned against different Reference products"
        )
    return common


def _phase_screen(store: CoregisteredSceneStore, tag: str) -> np.ndarray:
    """Return one unit's range-offset phase screen, zero for the Reference."""
    unit = store.unit_map()[tag]
    screen = None if store.is_reference else store.read_phase_screen(tag)
    if screen is None:
        return np.zeros(unit.shape, dtype=np.float32)
    return screen


def validate_scene_ownership(store: CoregisteredSceneStore) -> None:
    """Validate tiled NISAR Geo ownership of one store, if it declares any.

    Parameters
    ----------
    store : CoregisteredSceneStore
        Acquisition store to validate.

    Raises
    ------
    InvalidProcessingStateError
        If tiles overlap or their declared coverage counts differ.

    """
    if _requires_nisar_geo_ownership(store):
        _validate_nisar_geo_ownership(
            store,
            {tag: store.read(tag)[0] for tag in sorted(store.unit_map())},
        )


def form_merged_scene_interferogram(
    primary_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    *,
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
        Persisted acquisition stores of the pair's primary and secondary
        dates. Either may be the single Stack Reference store; both must be
        bound to the same Reference payloads.
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
    common = _validate_pair_binding(primary_store, secondary_store)
    if flatten_stage not in {"coregistration", "interferogram"}:
        reject_invalid_state("unsupported interferogram flattening stage")
    if (
        primary_store.flatten_stage != flatten_stage
        or secondary_store.flatten_stage != flatten_stage
    ):
        reject_invalid_state("scene artifacts mix flattening stages")
    validate_scene_ownership(primary_store)
    validate_scene_ownership(secondary_store)
    all_units = primary_store.unit_map()
    primary_units = {tag: all_units[tag] for tag in common}
    az_looks, rg_looks = multilook
    if az_looks < 1 or rg_looks < 1:
        reject_invalid_state("multilook factors must be >= 1")
    out_rows = primary_store.grid_shape[0] // az_looks
    out_cols = primary_store.grid_shape[1] // rg_looks
    if out_rows < 1 or out_cols < 1:
        reject_invalid_state("multilook factors exceed the common scene grid")
    if len(primary_units) == 1:
        tag = next(iter(primary_units))
        primary_unit = primary_units[tag]
        if (
            primary_unit.row_origin == 0
            and primary_unit.col_origin == 0
            and primary_unit.shape == primary_store.grid_shape
        ):
            primary, _ = primary_store.read(tag)
            secondary, _ = secondary_store.read(tag)
            if flatten_stage == "interferogram":
                screen = _phase_screen(secondary_store, tag) - _phase_screen(
                    primary_store, tag
                )
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
                filtered = _apply_phase_filter_contract(phase_filter, tensor, support)
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
            from faninsar.processing.runtime.dask_gpu import should_accelerate

            if (
                dask_client is not None or device.lower() == "cuda"
            ) and should_accelerate(device, dask_client, kernel="goldstein_filter"):
                from faninsar.processing.runtime.dask_gpu import run_goldstein_filter

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
        coherence_acc = np.zeros((out_rows, out_cols), dtype=np.float64)
        coherence_count = np.zeros((out_rows, out_cols), dtype=np.int32)
        claimed = np.zeros((out_rows, out_cols), dtype=np.int32)
        for tag in tags:
            primary, primary_unit = primary_store.read(tag)
            secondary, _ = secondary_store.read(tag)
            ifg = primary * np.conjugate(secondary)
            if flatten_stage == "interferogram":
                ifg *= np.exp(
                    -1j
                    * (
                        _phase_screen(secondary_store, tag)
                        - _phase_screen(primary_store, tag)
                    )
                )
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
            if coherence_window is not None:
                unit_coherence = _sliding_coherence(
                    primary, secondary, coherence_window
                )
                finite_coh = np.isfinite(unit_coherence)
                coh_values = np.where(finite_coh, unit_coherence, 0.0)
                np.add.at(
                    coherence_acc,
                    (free_rows, free_cols),
                    coh_values[in_bounds][free],
                )
                np.add.at(
                    coherence_count,
                    (free_rows, free_cols),
                    finite_coh[in_bounds][free].astype(np.int32),
                )
        has_data = claimed > 0
        denominator = np.where(has_data, claimed, 1)
        group_ifg = np.where(has_data, ifg_acc / denominator, 0).astype(np.complex64)
        if coherence_window is None:
            coherence_denominator = np.sqrt(
                np.maximum(
                    (primary_power_acc / denominator)
                    * (secondary_power_acc / denominator),
                    1e-30,
                )
            )
            group_coherence = np.clip(
                np.abs(group_ifg) / coherence_denominator, 0.0, 1.0
            ).astype(np.float32)
        else:
            # Documented window semantics: a stride-one local coherence MLE
            # accumulated per fine sample and averaged over each look block,
            # never a degenerate single-sample MLE under (1, 1) looks.
            group_coherence = np.where(
                coherence_count > 0,
                coherence_acc / np.maximum(coherence_count, 1),
                np.nan,
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
        from faninsar.processing.runtime.dask_gpu import should_accelerate

        if (dask_client is not None or device.lower() == "cuda") and should_accelerate(
            device, dask_client, kernel="goldstein_filter"
        ):
            from faninsar.processing.runtime.dask_gpu import run_goldstein_filter

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
        valid_mask = (
            filtered_mask & np.isfinite(merged_ifg.real) & np.isfinite(merged_ifg.imag)
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
    multilook: tuple[int, int] = (1, 1),
    goldstein_alpha: float = 0.0,
    device: str = "auto",
    dask_client: object | None = None,
) -> dict[str, np.ndarray]:
    """Form derived IFGs from aligned scene payloads only.

    Parameters
    ----------
    primary_store, secondary_store : CoregisteredSceneStore
        Persisted acquisition stores bound to the same Reference payloads.
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
    outputs: dict[str, np.ndarray] = {}
    for tag in _validate_pair_binding(primary_store, secondary_store):
        primary, _ = primary_store.read(tag)
        secondary, _ = secondary_store.read(tag)
        complex_ifg = form_interferogram(
            primary,
            secondary,
            multilook=multilook,
        ).complex_ifg
        if goldstein_alpha > 0.0:
            from faninsar.processing.runtime.dask_gpu import should_accelerate

            if (
                dask_client is not None or device.lower() == "cuda"
            ) and should_accelerate(device, dask_client, kernel="goldstein_filter"):
                from faninsar.processing.runtime.dask_gpu import run_goldstein_filter

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
    "NISAR_GEO_COVERAGE_POLICY",
    "SCENE_KIND",
    "CoregisteredSceneStore",
    "SceneUnit",
    "SceneWriter",
    "form_merged_scene_interferogram",
    "form_scene_interferograms",
    "scene_grid_identity",
    "valid_samples",
    "validate_reference_binding",
    "validate_scene_ownership",
    "write_scene",
]
