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
    require_basename,
    validate_spatial_metadata,
    write_zarr_array,
)
from faninsar.io.storage.product import ProductWriter, open_product
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.interferometry.pair import (
    InterferogramProduct,
    form_interferogram,
    form_interferogram_from_fields,
    goldstein_filter,
    mask_invalid_looks,
)
from faninsar.processing.mosaicking import (
    BurstWindow,
    LayoutSourceWindow,
    MergedWindow,
    MergeLayoutWindow,
    MergeScope,
    MergeSeam,
    build_merge_layout,
    merge_bursts,
)
from faninsar.processing.mosaicking.windows import intersect_windows, window_slices

SCENE_KIND = "scene"
MERGED_SCENE_UNIT_TAG = "merged"

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch

    from faninsar.io.storage.product import Product
    from faninsar.network.layers import ProductLayer
    from faninsar.network.products import AssetKind
    from faninsar.processing.geocoding.product import GeocodeSession
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
    return require_basename(value, f"scene {field}")


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
    burst_id: str | None = None
    valid_mask_path: Path | None = None

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
    representation: str = "burst"
    layout_id: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    properties: dict[str, object] = field(default_factory=dict, repr=False)
    _session: GeocodeSession | None = field(default=None, repr=False, compare=False)
    _upstream: tuple[Product, ...] = field(default=(), repr=False, compare=False)

    @property
    def layers(self) -> SceneLayers:
        """Return the lazy raster-layer view over this scene's units.

        The view is bound to the Stack session when the store was obtained
        through :attr:`Stack.scenes`; a store opened directly has
        ``_session=None`` and its ``.geocode()`` reports the standard
        "not bound to a Stack session" error.
        """
        return SceneLayers(
            self,
            _session=self._session,
            _upstream=self._upstream,
        )

    def select_units(self, tags: frozenset[str]) -> CoregisteredSceneStore:
        """Return a view of this store narrowed to a unit subset.

        The returned store shares this product's metadata and payloads but
        exposes only the selected units; ``read``, ``mosaic_slc``, and the
        selection-aware extent helpers all honour the subset.
        """
        known = self.unit_map()
        unknown = sorted(set(tags) - set(known))
        if unknown:
            reject_invalid_state(f"scene units not in the product: {unknown!r}")
        units = tuple(unit for unit in self.units if unit.tag in tags)
        from dataclasses import replace

        return replace(self, units=units)

    def selected_units(self) -> tuple[SceneUnit, ...]:
        """Return the units in this (possibly selected) view."""
        return self.units

    def selected_origin(self) -> tuple[int, int]:
        """Return the ``(row, col)`` offset of the union bounding box."""
        if not self.units:
            return (0, 0)
        row = min(unit.row_origin for unit in self.units)
        col = min(unit.col_origin for unit in self.units)
        return (row, col)

    def selected_extent(self) -> tuple[int, int]:
        """Return the ``(rows, cols)`` of the union bounding box."""
        if not self.units:
            return self.grid_shape
        row0, col0 = self.selected_origin()
        row1 = max(unit.row_origin + unit.shape[0] for unit in self.units)
        col1 = max(unit.col_origin + unit.shape[1] for unit in self.units)
        return (row1 - row0, col1 - col0)

    def mosaic_slc(self, *, full: bool = False) -> np.ndarray:
        """Mosaic the selected units' SLC payloads onto one frame.

        Units composite in ``SceneUnit`` order: a later unit overwrites an
        earlier one only on samples where the earlier is invalid
        (``not valid_samples``), so a valid edge sample keeps the first
        committed value. ``full=True`` writes onto the common grid; the
        default ``False`` writes onto the selected units' union bounding
        box.
        """
        shape = self.grid_shape if full else self.selected_extent()
        origin = (0, 0) if full else self.selected_origin()
        out = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex64)
        for unit in self.units:
            payload, _ = self.read(unit.tag)
            r0 = unit.row_origin - origin[0]
            c0 = unit.col_origin - origin[1]
            win = out[r0 : r0 + unit.shape[0], c0 : c0 + unit.shape[1]]
            invalid = ~valid_samples(win)
            win[invalid] = payload[invalid]
        return out

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
            raw_burst_id = asset.get("faninsar:burst_id")
            burst_id = (
                require_basename(raw_burst_id, "scene burst_id")
                if raw_burst_id is not None
                else None
            )
            valid_name = f"{name}.valid"
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
                burst_id=burst_id,
                valid_mask_path=(
                    product.asset_path(valid_name)
                    if valid_name in product.assets
                    else None
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
        representation = str(properties.get("faninsar:representation", "burst"))
        if representation not in {"burst", "merged"}:
            reject_invalid_state("scene representation must be burst or merged")
        layout_id = str(
            properties.get("faninsar:layout_id") or product.inputs.get("layout") or ""
        )
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
            representation=representation,
            layout_id=layout_id,
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
    def baseline(self) -> dict[str, float]:
        """Return the scene's baseline relative to the Stack Reference.

        Reads the persisted ``faninsar:baseline_parallel_m`` and
        ``faninsar:baseline_perpendicular_m`` product properties written at
        coregistration (signed perpendicular, metres). The Reference is its
        own baseline origin, so a product with ``is_reference`` and absent
        fields reads ``0.0``; a secondary committed before the fields existed
        reads ``NaN``. Only real ``int``/``float`` values are accepted —
        ``bool``, strings, and other types read as ``NaN``/``0.0`` under the
        same absent-field rules.

        Returns
        -------
        dict
            ``{"parallel_m": float, "perpendicular_m": float}``.

        """
        fallback = 0.0 if self.is_reference else float("nan")
        parallel = self.properties.get("faninsar:baseline_parallel_m")
        perpendicular = self.properties.get("faninsar:baseline_perpendicular_m")
        return {
            "parallel_m": (
                float(parallel)
                if isinstance(parallel, int | float) and not isinstance(parallel, bool)
                else fallback
            ),
            "perpendicular_m": (
                float(perpendicular)
                if isinstance(perpendicular, int | float)
                and not isinstance(perpendicular, bool)
                else fallback
            ),
        }

    def __repr__(self) -> str:
        """Return a concise one-line scene-product summary."""
        return (
            f"CoregisteredSceneStore(date={self.date_id!r}, role={self.role!r}, "
            f"units={len(self.units)}, shape={self.grid_shape}, "
            f"domain={self.domain!r})"
        )

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

    def read_phase_screen(
        self, tag: str, window: tuple[int, int, int, int] | None = None
    ) -> np.ndarray | None:
        """Read one persisted range-offset phase screen, if present.

        ``window`` is local to the unit payload, matching :meth:`read`.
        """
        unit = self.unit_map().get(tag)
        if unit is None:
            reject_invalid_state(f"scene unit {tag!r} is not in the product")
        if unit.phase_screen_path is None:
            return None
        screen = (
            read_zarr_array(unit.phase_screen_path)
            if window is None
            else read_zarr_array(unit.phase_screen_path, window=window)
        )
        expected = _payload_window_shape(unit.shape, window)
        if screen.shape != expected or screen.dtype != np.float32:
            reject_invalid_state(f"scene phase screen shape mismatch for {tag!r}")
        return np.asarray(screen, dtype=np.float32)

    def read_valid_mask(
        self, tag: str, window: tuple[int, int, int, int] | None = None
    ) -> np.ndarray | None:
        """Read one unit's explicit boolean support, if persisted.

        No-ROI products omit this asset; callers then keep the legacy
        amplitude-inferred support used by NISAR non-overlap checks.
        """
        unit = self.unit_map().get(tag)
        if unit is None:
            reject_invalid_state(f"scene unit {tag!r} is not in the product")
        if unit.valid_mask_path is None:
            return None
        mask = (
            read_zarr_array(unit.valid_mask_path)
            if window is None
            else read_zarr_array(unit.valid_mask_path, window=window)
        )
        expected = _payload_window_shape(unit.shape, window)
        if mask.shape != expected or mask.dtype != np.bool_:
            reject_invalid_state(f"scene valid_mask shape mismatch for {tag!r}")
        return np.asarray(mask, dtype=bool)

    def unit_map(self) -> dict[str, SceneUnit]:
        """Return units keyed by their stable burst tag."""
        return {unit.tag: unit for unit in self.units}

    def read(
        self, tag: str, window: tuple[int, int, int, int] | None = None
    ) -> tuple[np.ndarray, SceneUnit]:
        """Read one unit payload and validate its shape and dtype.

        Parameters
        ----------
        tag : str
            Stable unit identifier within the product.
        window : tuple of int, optional
            Local half-open ``(r0, r1, c0, c1)`` in the unit payload. Placement
            on the parent grid remains on the returned :class:`SceneUnit`.

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
        payload = (
            read_zarr_array(unit.payload_path)
            if window is None
            else read_zarr_array(unit.payload_path, window=window)
        )
        expected = _payload_window_shape(unit.shape, window)
        if payload.shape != expected:
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
    burst_id: str | None = None
    valid_mask: bool = False


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
        valid_mask: np.ndarray | None = None,
        burst_id: str | None = None,
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
        if valid_mask is not None and (
            valid_mask.ndim != 2
            or valid_mask.shape != payload.shape
            or valid_mask.dtype != np.bool_
        ):
            reject_invalid_state(
                "scene valid_mask must be a boolean array matching payload"
            )
        resolved_burst_id = (
            require_basename(burst_id, "scene burst_id")
            if burst_id is not None
            else None
        )
        write_zarr_array(self.staging / f"{tag}.slc.zarr", payload)
        if phase_screen is not None:
            write_zarr_array(self.staging / f"{tag}.phase.zarr", phase_screen)
        if valid_mask is not None:
            write_zarr_array(self.staging / f"{tag}.valid.zarr", np.asarray(valid_mask))
        self._units[tag] = _PendingUnit(
            tag=tag,
            rows=int(payload.shape[0]),
            cols=int(payload.shape[1]),
            row_origin=int(row_origin),
            col_origin=int(col_origin),
            lineage=[dict(item) for item in (scientific_lineage or ())],
            phase_state=dict(phase_state) if phase_state is not None else None,
            phase_screen=phase_screen is not None,
            burst_id=resolved_burst_id,
            valid_mask=valid_mask is not None,
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
        extra_properties = dict(properties or {})
        representation = extra_properties.get("faninsar:representation", "burst")
        if representation not in {"burst", "merged"}:
            self._writer.abort()
            reject_invalid_state("scene representation must be burst or merged")
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
            if unit.burst_id is not None:
                fields["faninsar:burst_id"] = unit.burst_id
            if unit.phase_screen:
                fields["faninsar:phase_screen"] = f"{tag}.phase"
                self._writer.add_asset(
                    f"{tag}.phase", f"{tag}.phase.zarr", **{"faninsar:role": "phase"}
                )
            if unit.valid_mask:
                self._writer.add_asset(
                    f"{tag}.valid",
                    f"{tag}.valid.zarr",
                    **{"faninsar:role": "valid_mask"},
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
            **extra_properties,
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


def _payload_window_shape(
    shape: tuple[int, int],
    window: tuple[int, int, int, int] | None,
) -> tuple[int, int]:
    """Return the 2-D shape of a local payload window."""
    if window is None:
        return shape
    return (window[1] - window[0], window[3] - window[2])


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


def _pair_unit_support(
    primary_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    tag: str,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Return joint explicit support, or ``None`` when both units omit a mask."""
    primary_mask = primary_store.read_valid_mask(tag)
    secondary_mask = secondary_store.read_valid_mask(tag)
    if primary_mask is None and secondary_mask is None:
        return None
    support = np.ones(shape, dtype=bool)
    if primary_mask is not None:
        support &= primary_mask
    if secondary_mask is not None:
        support &= secondary_mask
    return support


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


def _phase_screen(
    store: CoregisteredSceneStore,
    tag: str,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Return one unit's range-offset phase screen, zero for the Reference."""
    unit = store.unit_map()[tag]
    screen = None if store.is_reference else store.read_phase_screen(tag, window=window)
    if screen is None:
        return np.zeros(_payload_window_shape(unit.shape, window), dtype=np.float32)
    return screen


def _unit_burst_id(unit: SceneUnit) -> str:
    """Return the physical burst identity, falling back to the unit tag."""
    return unit.burst_id or unit.tag


def _unit_parent_window(unit: SceneUnit) -> tuple[int, int, int, int]:
    """Return the unit's half-open window on the parent scene grid."""
    return (
        int(unit.row_origin),
        int(unit.row_origin + unit.shape[0]),
        int(unit.col_origin),
        int(unit.col_origin + unit.shape[1]),
    )


def _scene_merge_scope(store: CoregisteredSceneStore) -> MergeScope:
    """Build the merge identity from persisted scene metadata."""
    properties = store.properties
    track = properties.get("faninsar:track_id")
    channel = properties.get("faninsar:channel")
    polygon = properties.get("faninsar:polygon_identity")
    return MergeScope(
        domain=store.domain,  # type: ignore[arg-type]
        track_id=track if isinstance(track, str) and track else "stack",
        channel=channel if isinstance(channel, str) and channel else "slc",
        reference_id=store.reference_id,
        grid_identity=store.grid_identity,
        polygon_identity=(polygon if isinstance(polygon, str) and polygon else "none"),
    )


def _scene_units_overlap(units: Sequence[SceneUnit]) -> bool:
    """Return whether any two scene units share parent-grid samples."""
    ordered = tuple(units)
    for index, left in enumerate(ordered):
        left_window = _unit_parent_window(left)
        for right in ordered[index + 1 :]:
            if intersect_windows(left_window, _unit_parent_window(right)) is not None:
                return True
    return False


def scene_units_overlap(store: CoregisteredSceneStore) -> bool:
    """Return whether any two units of ``store`` share parent-grid samples.

    Parameters
    ----------
    store : CoregisteredSceneStore
        Scene product to inspect.

    Returns
    -------
    bool
        ``True`` when any pair of units intersects on the parent grid.

    """
    return _scene_units_overlap(store.units)


def _crop_merge_layout(
    layout: MergeLayoutWindow, window: tuple[int, int, int, int]
) -> MergeLayoutWindow | None:
    """Return ``layout`` restricted to ``window``, or ``None`` if empty."""
    overlap = intersect_windows(layout.window, _as_merge_window(window))
    if overlap is None:
        return None
    rows, cols = window_slices(layout.window, overlap)
    support = np.asarray(layout.scientific_support[rows, cols], dtype=bool)
    if not np.any(support):
        return None
    return MergeLayoutWindow(
        scope=layout.scope,
        window=overlap,
        source_ids=layout.source_ids,
        owner=np.asarray(layout.owner[rows, cols]),
        scientific_support=support,
        seams=layout.seams,
        source_windows=layout.source_windows,
    )


def _complete_look_indices(
    native_window: tuple[int, int, int, int],
    multilook: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return ``(first_i, last_i, first_j, last_j)`` complete global looks."""
    az_looks, rg_looks = multilook
    row0, row1, col0, col1 = native_window
    return (
        (row0 + az_looks - 1) // az_looks,
        row1 // az_looks,
        (col0 + rg_looks - 1) // rg_looks,
        col1 // rg_looks,
    )


def _local_read_window(
    unit: SceneUnit, parent_window: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Return the unit-local window covering ``parent_window``, if any."""
    overlap = intersect_windows(_unit_parent_window(unit), parent_window)
    if overlap is None:
        return None
    return (
        overlap[0] - unit.row_origin,
        overlap[1] - unit.row_origin,
        overlap[2] - unit.col_origin,
        overlap[3] - unit.col_origin,
    )


def _scene_parent_window(
    store: CoregisteredSceneStore,
) -> tuple[int, int, int, int]:
    """Return the full parent-grid window of one scene product."""
    return (
        0,
        int(store.grid_shape[0]),
        0,
        int(store.grid_shape[1]),
    )


def _as_merge_window(
    window: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return ``window`` as Python ints for the merge-window contract."""
    return (int(window[0]), int(window[1]), int(window[2]), int(window[3]))


def _store_unit_support(
    store: CoregisteredSceneStore,
    tag: str,
    payload: np.ndarray,
    local_window: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Return explicit or inferred native support for one unit window."""
    mask = store.read_valid_mask(tag, window=local_window)
    finite = np.isfinite(payload.real) & np.isfinite(payload.imag)
    if mask is None:
        return np.asarray(valid_samples(payload) & finite, dtype=bool)
    return np.asarray(mask & finite, dtype=bool)


def _read_unit_on_layout(
    store: CoregisteredSceneStore,
    unit: SceneUnit,
    layout_window: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """Read one unit's layout overlap as a window-local payload and mask."""
    local_window = _local_read_window(unit, layout_window)
    if local_window is None:
        return None
    overlap = intersect_windows(_unit_parent_window(unit), layout_window)
    if overlap is None:
        return None
    read_window = (
        None
        if local_window == (0, int(unit.shape[0]), 0, int(unit.shape[1]))
        else local_window
    )
    payload, _ = store.read(unit.tag, window=read_window)
    support = _store_unit_support(store, unit.tag, payload, read_window)
    return _as_merge_window(overlap), payload, support


def _expand_native_window(
    window: tuple[int, int, int, int],
    *,
    halo: tuple[int, int],
    clip: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Expand ``window`` by ``halo`` and clip it to ``clip``."""
    row0, row1, col0, col1 = window
    halo_az, halo_rg = halo
    expanded = (
        int(row0) - int(halo_az),
        int(row1) + int(halo_az),
        int(col0) - int(halo_rg),
        int(col1) + int(halo_rg),
    )
    overlap = intersect_windows(_as_merge_window(expanded), _as_merge_window(clip))
    if overlap is None:
        reject_invalid_state("native tile does not intersect the merge layout window")
    return overlap


def _validate_consumed_layout(
    layout: MergeLayoutWindow,
    store: CoregisteredSceneStore,
) -> None:
    """Reject a layout that does not belong to this scene grid."""
    if not isinstance(layout, MergeLayoutWindow):
        reject_invalid_state(
            "form_merged_scene_interferogram requires a MergeLayoutWindow"
        )
    if layout.scope.domain != store.domain:
        reject_invalid_state(
            "committed merge_layout domain does not match the scene product"
        )
    if layout.scope.grid_identity != store.grid_identity:
        reject_invalid_state(
            "committed merge_layout grid_identity does not match the scene product"
        )
    if layout.scope.reference_id != store.reference_id:
        reject_invalid_state(
            "committed merge_layout reference_id does not match the scene product"
        )


def _pair_burst_windows(
    primary_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    tags: Sequence[str],
    *,
    layout: MergeLayoutWindow,
    flatten_stage: str,
) -> list[BurstWindow]:
    """Read unit-local pair fields that intersect ``layout.window``."""
    source_ids = set(layout.source_ids)
    primary_units = primary_store.unit_map()
    products: list[BurstWindow] = []
    for tag in tags:
        unit = primary_units[tag]
        burst_id = _unit_burst_id(unit)
        if burst_id not in source_ids:
            continue
        local_window = _local_read_window(unit, layout.window)
        if local_window is None:
            continue
        read_window = (
            None
            if local_window == (0, int(unit.shape[0]), 0, int(unit.shape[1]))
            else local_window
        )
        primary, _ = primary_store.read(tag, window=read_window)
        secondary, _ = secondary_store.read(tag, window=read_window)
        if flatten_stage == "interferogram":
            screen = _phase_screen(
                secondary_store, tag, window=read_window
            ) - _phase_screen(primary_store, tag, window=read_window)
            secondary = secondary * np.exp(1j * screen)
        support = _unit_scientific_support(
            primary_store,
            secondary_store,
            tag,
            primary,
            secondary,
            read_window,
        )
        overlap = intersect_windows(_unit_parent_window(unit), layout.window)
        if overlap is None:
            continue
        products.append(
            BurstWindow(
                scope=layout.scope,
                burst_id=burst_id,
                acquisitions=(primary_store.date_id, secondary_store.date_id),
                write_ids=(primary_store.write_id, secondary_store.write_id),
                phase_convention="wrapped_primary_conj_secondary",
                window=_as_merge_window(overlap),
                complex=np.asarray(
                    primary * np.conjugate(secondary), dtype=np.complex64
                ),
                valid_mask=support,
                primary_power=np.asarray(
                    primary.real**2 + primary.imag**2, dtype=np.float32
                ),
                secondary_power=np.asarray(
                    secondary.real**2 + secondary.imag**2, dtype=np.float32
                ),
            )
        )
    return products


def build_scene_merge_layout(
    store: CoregisteredSceneStore,
    *,
    window: tuple[int, int, int, int] | None = None,
    seams: Sequence[MergeSeam] = (),
    owner: np.ndarray | None = None,
) -> MergeLayoutWindow:
    """Assemble a merge layout from scene support without inventing cuts.

    Overlapping units require caller-supplied physical ``seams`` or a finished
    ``owner`` raster. This helper never infers midpoints from ROI-cropped
    ``row_origin``/``shape``. ``scope.reference_id`` is the acquisition key
    and is not a product write id.

    Parameters
    ----------
    store : CoregisteredSceneStore
        Typically the Stack Reference scene.
    window : tuple of int, optional
        Layout window. Default is the full parent scene grid.
    seams : sequence of MergeSeam, optional
        Physical cuts from mission parent geometry. Required when units
        overlap and ``owner`` is omitted.
    owner : numpy.ndarray, optional
        Finished preferred-owner labels on ``window``.

    Returns
    -------
    MergeLayoutWindow
        Owner map for :func:`~faninsar.processing.mosaicking.merge_bursts`.

    Raises
    ------
    InvalidProcessingStateError
        If no unit intersects the window, or overlapping units are supplied
        without physical seams or an owner raster.

    """
    layout_window = (
        _as_merge_window(window) if window is not None else _scene_parent_window(store)
    )
    if _scene_units_overlap(store.units) and not seams and owner is None:
        reject_invalid_state(
            "overlapping scene units require a committed physical merge_layout; "
            "do not infer seams from ROI-cropped origins"
        )
    sources: list[LayoutSourceWindow] = []
    for unit in store.units:
        loaded = _read_unit_on_layout(store, unit, layout_window)
        if loaded is None:
            continue
        overlap, _, support = loaded
        sources.append(
            LayoutSourceWindow(
                burst_id=_unit_burst_id(unit),
                window=overlap,
                scientific_support=support,
                swath=_tag_swath(unit.tag),
            )
        )
    if not sources:
        reject_invalid_state("scene product has no unit inside the merge window")
    return build_merge_layout(
        sources,
        scope=_scene_merge_scope(store),
        seams=seams,
        owner=owner,
        window=layout_window,
    )


def merge_scene_slc(
    store: CoregisteredSceneStore,
    *,
    layout: MergeLayoutWindow,
) -> MergedWindow:
    """Select one scene's burst SLCs onto ``layout`` with the midpoint kernel.

    Parameters
    ----------
    store : CoregisteredSceneStore
        Burst scene product to select from. Inputs remain unmodified.
    layout : MergeLayoutWindow
        Preferred-owner map, typically from the Reference scene.

    Returns
    -------
    MergedWindow
        Layout-window SLC samples and explicit support.

    Raises
    ------
    InvalidProcessingStateError
        If the scene has no burst that intersects the layout.

    """
    source_ids = set(layout.source_ids)
    products: list[BurstWindow] = []
    for unit in store.units:
        burst_id = _unit_burst_id(unit)
        if burst_id not in source_ids:
            continue
        loaded = _read_unit_on_layout(store, unit, layout.window)
        if loaded is None:
            continue
        overlap, payload, support = loaded
        products.append(
            BurstWindow(
                scope=layout.scope,
                burst_id=burst_id,
                acquisitions=(store.date_id,),
                write_ids=(store.write_id,),
                phase_convention="slc",
                window=overlap,
                complex=np.asarray(payload, dtype=np.complex64),
                valid_mask=support,
            )
        )
    if not products:
        reject_invalid_state("scene product has no burst window on the merge layout")
    return merge_bursts(products, layout=layout)


def _unit_scientific_support(
    primary_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    tag: str,
    primary: np.ndarray,
    secondary: np.ndarray,
    local_window: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Return explicit or inferred native support for one unit window."""
    primary_mask = primary_store.read_valid_mask(tag, window=local_window)
    secondary_mask = secondary_store.read_valid_mask(tag, window=local_window)
    if primary_mask is None and secondary_mask is None:
        support = valid_samples(primary) & valid_samples(secondary)
    else:
        support = np.ones(primary.shape, dtype=bool)
        if primary_mask is not None:
            support &= primary_mask
        if secondary_mask is not None:
            support &= secondary_mask
    finite = (
        np.isfinite(primary.real)
        & np.isfinite(primary.imag)
        & np.isfinite(secondary.real)
        & np.isfinite(secondary.imag)
    )
    return np.asarray(support & finite, dtype=bool)


def _finish_looked_interferogram(
    product: InterferogramProduct,
    *,
    goldstein_alpha: float,
    device: str,
    dask_client: object | None,
    phase_filter: PhaseFilter | None,
) -> InterferogramProduct:
    """Apply the runtime filter to an already-looked selected-field product."""
    if phase_filter is not None:
        import torch

        filter_device = device
        if filter_device in {"", "auto"}:
            filter_device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.as_tensor(product.complex, device=filter_device)
        filter_support = torch.as_tensor(
            product.valid_mask, dtype=torch.bool, device=filter_device
        )
        filtered = _apply_phase_filter_contract(phase_filter, tensor, filter_support)
        result_ifg = filtered.interferogram.detach().cpu().numpy()
        result_mask = filtered.valid_mask.detach().cpu().numpy()
        result_ifg[~result_mask] = np.nan + 1j * np.nan
        coherence = np.asarray(product.coherence, dtype=np.float32)
        coherence[~result_mask] = np.nan
        wrapped = np.angle(result_ifg).astype(np.float32)
        wrapped[~result_mask] = np.nan
        return InterferogramProduct(
            complex=result_ifg,
            coherence=coherence,
            wrapped_phase=wrapped,
            amplitude=np.abs(result_ifg).astype(np.float32),
            valid_mask=result_mask,
        )
    if goldstein_alpha <= 0.0:
        return product
    from faninsar.processing.runtime.dask_gpu import should_accelerate

    if (dask_client is not None or device.lower() == "cuda") and should_accelerate(
        device, dask_client, kernel="goldstein_filter"
    ):
        from faninsar.processing.runtime.dask_gpu import run_goldstein_filter

        filtered = run_goldstein_filter(
            product.complex,
            alpha=goldstein_alpha,
            window=32,
            device=device,
            client=dask_client,
        )
    else:
        filtered = goldstein_filter(product.complex, alpha=goldstein_alpha)
    support = np.asarray(product.valid_mask, dtype=bool)
    filtered = np.asarray(filtered)
    filtered[~support] = np.nan + 1j * np.nan
    coherence = np.asarray(product.coherence, dtype=np.float32)
    wrapped = np.angle(filtered).astype(np.float32)
    wrapped[~support] = np.nan
    return InterferogramProduct(
        complex=filtered,
        coherence=coherence,
        wrapped_phase=wrapped,
        amplitude=np.abs(filtered).astype(np.float32),
        valid_mask=support,
    )


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


def _form_unit_pair_interferogram(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    support: np.ndarray | None,
    flatten_screen: np.ndarray | None,
    multilook: tuple[int, int],
    goldstein_alpha: float,
    device: str,
    dask_client: object | None,
    coherence_window: tuple[int, int] | None,
    phase_filter: PhaseFilter | None,
) -> InterferogramProduct:
    """Form one unit-local interferogram through the Pair kernel."""
    if support is not None:
        primary = primary.copy()
        secondary = secondary.copy()
        invalid = ~support
        primary[invalid] = np.nan + 1j * np.nan
        secondary[invalid] = np.nan + 1j * np.nan
    if flatten_screen is not None:
        # Multiplying the secondary by exp(+j screen) makes the resulting
        # native IFG carry the required exp(-j screen) before multilooking.
        secondary = secondary * np.exp(1j * flatten_screen)
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
        tensor = torch.as_tensor(product.complex, device=filter_device)
        filter_support = torch.as_tensor(
            product.valid_mask, dtype=torch.bool, device=filter_device
        )
        filtered = _apply_phase_filter_contract(phase_filter, tensor, filter_support)
        result_ifg = filtered.interferogram.detach().cpu().numpy()
        result_mask = filtered.valid_mask.detach().cpu().numpy()
        result_ifg[~result_mask] = np.nan + 1j * np.nan
        _, result_coh, result_phase = mask_invalid_looks(result_ifg, product.coherence)
        assert result_coh is not None
        return InterferogramProduct(
            complex=result_ifg,
            coherence=result_coh,
            wrapped_phase=result_phase,
            amplitude=np.abs(result_ifg).astype(np.float32),
            valid_mask=result_mask,
        )
    if goldstein_alpha <= 0.0:
        return product
    from faninsar.processing.runtime.dask_gpu import should_accelerate

    if (dask_client is not None or device.lower() == "cuda") and should_accelerate(
        device, dask_client, kernel="goldstein_filter"
    ):
        from faninsar.processing.runtime.dask_gpu import run_goldstein_filter

        filtered = run_goldstein_filter(
            product.complex,
            alpha=goldstein_alpha,
            window=32,
            device=device,
            client=dask_client,
        )
    else:
        filtered = goldstein_filter(product.complex, alpha=goldstein_alpha)
    filtered, coherence, wrapped = mask_invalid_looks(filtered, product.coherence)
    assert coherence is not None
    return InterferogramProduct(
        complex=filtered,
        coherence=coherence,
        wrapped_phase=wrapped,
        amplitude=np.abs(filtered).astype(np.float32),
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
    layout: MergeLayoutWindow | None = None,
    native_tile: int = 512,
) -> InterferogramProduct:
    """Form one looked IFG from selected native pair fields.

    Native per-burst cross-products and endpoint powers are selected with
    :func:`~faninsar.processing.mosaicking.merge_bursts` on the committed
    Reference :class:`~faninsar.processing.mosaicking.windows.MergeLayoutWindow`,
    then looked by
    :func:`~faninsar.processing.interferometry.pair.form_interferogram_from_fields`.
    Overlapping coverage requires that layout (or explicit seams when building
    a generic non-production helper). This function never infers physical cuts
    from ROI-cropped ``row_origin``/``shape``. Bounded native tiles cover
    complete looks plus the coherence halo; the returned object is the assembled
    looked interferogram. Explicit ``valid_mask`` assets keep supported
    zero-amplitude samples, including two-sample cancellation looks.

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
        Runtime filter applied after the assembled looked interferogram.
        ``None`` leaves the multilooked complex interferogram unchanged.
    layout : MergeLayoutWindow, optional
        Committed Reference owner map. ``None`` is allowed only for
        non-overlapping unique coverage (generic NISAR).
    native_tile : int, optional
        Maximum native pixel extent of one formation tile before the
        coherence halo. Default ``512``.

    Returns
    -------
    InterferogramProduct
        Looked interferogram covering the complete looks in ``layout.window``.

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
    az_looks, rg_looks = multilook
    if (
        type(az_looks) is not int
        or type(rg_looks) is not int
        or az_looks < 1
        or rg_looks < 1
    ):
        reject_invalid_state("multilook factors must be >= 1")
    if type(native_tile) is not int or native_tile < 1:
        reject_invalid_state("native_tile must be a positive int")
    if layout is None:
        layout = build_scene_merge_layout(primary_store)
    else:
        _validate_consumed_layout(layout, primary_store)
    first_i, last_i, first_j, last_j = _complete_look_indices(layout.window, multilook)
    n_i = last_i - first_i
    n_j = last_j - first_j
    if n_i < 1 or n_j < 1:
        reject_invalid_state("multilook factors exceed the merge layout window")
    halo = (0, 0)
    if coherence_window is not None:
        halo = (int(coherence_window[0]) // 2, int(coherence_window[1]) // 2)
    look_tile_az = max(1, native_tile // az_looks)
    look_tile_rg = max(1, native_tile // rg_looks)
    assembled_complex = np.full((n_i, n_j), np.nan + 1j * np.nan, dtype=np.complex64)
    assembled_coh = np.full((n_i, n_j), np.nan, dtype=np.float32)
    assembled_mask = np.zeros((n_i, n_j), dtype=bool)
    for i0 in range(first_i, last_i, look_tile_az):
        i1 = min(i0 + look_tile_az, last_i)
        for j0 in range(first_j, last_j, look_tile_rg):
            j1 = min(j0 + look_tile_rg, last_j)
            out_native = (
                int(i0 * az_looks),
                int(i1 * az_looks),
                int(j0 * rg_looks),
                int(j1 * rg_looks),
            )
            read_native = _expand_native_window(
                out_native, halo=halo, clip=layout.window
            )
            cropped = _crop_merge_layout(layout, read_native)
            if cropped is None:
                continue
            products = _pair_burst_windows(
                primary_store,
                secondary_store,
                common,
                layout=cropped,
                flatten_stage=flatten_stage,
            )
            if not products:
                continue
            selected = merge_bursts(products, layout=cropped)
            if selected.primary_power is None or selected.secondary_power is None:
                reject_invalid_state("selected merge window is missing endpoint powers")
            piece = form_interferogram_from_fields(
                selected.complex,
                selected.primary_power,
                selected.secondary_power,
                selected.valid_mask,
                native_origin=(int(cropped.window[0]), int(cropped.window[2])),
                output_window=_as_merge_window(out_native),
                multilook=multilook,
                coherence_window=coherence_window,
            )
            dest = (
                slice(i0 - first_i, i1 - first_i),
                slice(j0 - first_j, j1 - first_j),
            )
            assembled_complex[dest] = piece.complex
            assembled_coh[dest] = piece.coherence
            if piece.valid_mask is not None:
                assembled_mask[dest] = piece.valid_mask
    wrapped = np.angle(assembled_complex).astype(np.float32)
    amplitude = np.abs(assembled_complex).astype(np.float32)
    wrapped[~assembled_mask] = np.nan
    amplitude[~assembled_mask] = np.nan
    assembled_coh[~assembled_mask] = np.nan
    product = InterferogramProduct(
        complex=assembled_complex,
        coherence=assembled_coh,
        wrapped_phase=wrapped,
        amplitude=amplitude,
        valid_mask=assembled_mask,
    )
    return _finish_looked_interferogram(
        product,
        goldstein_alpha=goldstein_alpha,
        device=device,
        dask_client=dask_client,
        phase_filter=phase_filter,
    )


def form_burst_scene_interferogram(
    primary_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    tag: str,
    *,
    multilook: tuple[int, int] = (1, 1),
    goldstein_alpha: float = 0.0,
    device: str = "auto",
    dask_client: object | None = None,
    flatten_stage: str = "coregistration",
    coherence_window: tuple[int, int] | None = None,
    phase_filter: PhaseFilter | None = None,
) -> InterferogramProduct:
    """Form one burst-local 2-D interferogram from a shared scene unit.

    Unlike :func:`form_merged_scene_interferogram`, the result stays on the
    unit payload after looks; parent-grid placement is the caller's metadata.

    Parameters
    ----------
    primary_store, secondary_store : CoregisteredSceneStore
        Persisted acquisition stores bound to the same Reference payloads.
    tag : str
        Shared unit identifier.
    multilook : tuple[int, int], optional
        Azimuth and range boxcar look factors on the unit payload.
    goldstein_alpha : float, optional
        Goldstein filter exponent. Zero disables filtering.
    device : str, optional
        Numerical device policy for the qualified Goldstein stage.
    dask_client : object, optional
        Explicitly trusted Dask client for qualified remote Goldstein work.
    flatten_stage : {"coregistration", "interferogram"}, optional
        Stage at which the NISAR range-offset phase screen is applied.
    coherence_window : tuple[int, int] or None, optional
        Coherence MLE support passed to the pair kernel.
    phase_filter : PhaseFilter or None, optional
        Runtime filter applied after multilooking.

    Returns
    -------
    InterferogramProduct
        Unit-local interferogram and quality layers.

    """
    common = _validate_pair_binding(primary_store, secondary_store)
    if tag not in common:
        reject_invalid_state(f"scene unit {tag!r} is not shared by both dates")
    if flatten_stage not in {"coregistration", "interferogram"}:
        reject_invalid_state("unsupported interferogram flattening stage")
    if (
        primary_store.flatten_stage != flatten_stage
        or secondary_store.flatten_stage != flatten_stage
    ):
        reject_invalid_state("scene artifacts mix flattening stages")
    az_looks, rg_looks = multilook
    if az_looks < 1 or rg_looks < 1:
        reject_invalid_state("multilook factors must be >= 1")
    primary, primary_unit = primary_store.read(tag)
    secondary, _ = secondary_store.read(tag)
    if primary_unit.shape[0] // az_looks < 1 or primary_unit.shape[1] // rg_looks < 1:
        reject_invalid_state("multilook factors exceed the burst unit")
    flatten_screen = None
    if flatten_stage == "interferogram":
        flatten_screen = _phase_screen(secondary_store, tag) - _phase_screen(
            primary_store, tag
        )
    return _form_unit_pair_interferogram(
        primary,
        secondary,
        support=_pair_unit_support(primary_store, secondary_store, tag, primary.shape),
        flatten_screen=flatten_screen,
        multilook=multilook,
        goldstein_alpha=goldstein_alpha,
        device=device,
        dask_client=dask_client,
        coherence_window=coherence_window,
        phase_filter=phase_filter,
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
        ).complex
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


@dataclass(frozen=True, slots=True)
class SceneLayers:
    """Lazy raster-layer view over one coregistered scene product.

    Holds the owning :class:`CoregisteredSceneStore`, the current unit
    selection, and the optional Stack session/upstream binding injected by
    :attr:`Stack.scenes`. Build a narrowed view with :meth:`sel`; the
    default view covers every unit (union bounding box for ``slc`` reads,
    common grid for :meth:`full`).

    Attributes
    ----------
    _store : CoregisteredSceneStore
        Committed scene product backing this view.
    _selection : frozenset[str] | None
        Selected unit tags; ``None`` selects all units.
    _session : GeocodeSession | None
        Stack session injected when the store was obtained through
        :attr:`Stack.scenes`; ``None`` on a directly opened store.
    _upstream : tuple[Product, ...]
        Upstream products (Reference) bound for ``validate_reference``.

    """

    _store: CoregisteredSceneStore
    _selection: frozenset[str] | None = None
    _full: bool = False
    _session: GeocodeSession | None = field(default=None, repr=False, compare=False)
    _upstream: tuple[Product, ...] = field(default=(), repr=False, compare=False)

    # ---- unit vocabulary ------------------------------------------------

    @property
    def units(self) -> tuple[str, ...]:
        """Return all unit tags in mission-neutral vocabulary."""
        return tuple(unit.tag for unit in self._store.units)

    @property
    def unit_kind(self) -> str:
        """Classify the unit vocabulary of this product.

        Ordered first-match-wins: ``NISAR_b*`` → ``"tile"``; a two- or
        three-part tag whose last segment starts with ``b`` and whose head
        is ``f``-prefixed or a bare swath → ``"burst"``; anything else →
        ``"unit"``.
        """
        tags = self.units
        if not tags:
            return "unit"
        if any(tag.startswith("NISAR_b") for tag in tags):
            return "tile"
        for tag in tags:
            parts = tag.split("_")
            if (
                len(parts) == 3
                and parts[0].startswith("f")
                and parts[2].startswith("b")
            ):
                return "burst"
            if len(parts) == 2 and parts[1].startswith("b"):
                return "burst"
        return "unit"

    @property
    def bursts(self) -> tuple[str, ...]:
        """Return unit tags when this product is burst-organized (S1)."""
        if self.unit_kind != "burst":
            reject_invalid_state(
                f"scene product units are {self.unit_kind}s, not bursts; "
                "use `units` or `tiles`"
            )
        return self.units

    @property
    def tiles(self) -> tuple[str, ...]:
        """Return unit tags when this product is tile-organized (NISAR)."""
        if self.unit_kind != "tile":
            reject_invalid_state(
                f"scene product units are {self.unit_kind}s, not tiles; "
                "use `units` or `bursts`"
            )
        return self.units

    @property
    def swaths(self) -> tuple[str, ...]:
        """Return the distinct swath names of a burst-organized product."""
        if self.unit_kind != "burst":
            return ()
        return tuple(sorted({_tag_swath(tag) for tag in self.units}))

    # ---- selection ------------------------------------------------------

    def sel(
        self,
        *,
        units: tuple[str, ...] | list[str] | None = None,
        bursts: tuple[str, ...] | list[str] | None = None,
        tiles: tuple[str, ...] | list[str] | None = None,
        swaths: tuple[str, ...] | list[str] | None = None,
    ) -> SceneLayers:
        """Return a view narrowed to a unit subset.

        At most one of ``units=``/``bursts=``/``tiles=`` names the selected
        tags; ``swaths=`` is honoured only on a ``burst``-kind product and
        may be used alone or to expand a tag selection with the named
        swaths' units. The default read window is the selected units'
        union bounding box; :meth:`full` returns a view pinned to the
        common grid. Mirrors the xarray ``sel`` dims-as-kwargs vocabulary
        used by :meth:`StackInterferogramCollection.sel`.
        """
        provided = [
            name
            for name, value in (
                ("units", units),
                ("bursts", bursts),
                ("tiles", tiles),
            )
            if value is not None
        ]
        if len(provided) > 1:
            reject_invalid_state(
                "sel() accepts at most one of units=, bursts=, or tiles="
            )
        kind = self.unit_kind
        if provided:
            name = provided[0]
            value = {"units": units, "bursts": bursts, "tiles": tiles}[name]
            expected = {"burst": "bursts", "tile": "tiles", "unit": "units"}[kind]
            if name not in {expected, "units"}:
                reject_invalid_state(
                    f"scene product units are {kind}s; use {expected}= "
                    f"(or the neutral units=), not {name}="
                )
            tags = frozenset(str(tag) for tag in value or ())
        else:
            tags = frozenset()
        if swaths is not None:
            if kind != "burst":
                reject_invalid_state(
                    "swaths= is only valid on burst-organized (Sentinel-1) "
                    "scene products"
                )
            swath_set = {str(s) for s in swaths}
            swath_tags = {tag for tag in self.units if _tag_swath(tag) in swath_set}
            tags = frozenset(set(tags) | swath_tags) if tags else frozenset(swath_tags)
        if not tags:
            reject_invalid_state("sel() produced an empty unit selection")
        unknown = sorted(set(tags) - set(self.units))
        if unknown:
            reject_invalid_state(f"scene units not in the product: {unknown!r}")
        from dataclasses import replace

        return replace(self, _selection=tags)

    def full(self) -> SceneLayers:
        """Return a view pinned to the full common grid.

        The returned view reads the selected units onto the common-grid
        frame (``extent=None``) instead of their union bounding box.
        """
        from dataclasses import replace

        return replace(self, _full=True)

    @property
    def _active_store(self) -> CoregisteredSceneStore:
        """Return the store narrowed to the current selection."""
        if self._selection is None:
            return self._store
        return self._store.select_units(self._selection)

    def _layer(
        self,
        name: str,
        asset_kind: AssetKind,
        *,
        _store: CoregisteredSceneStore | None = None,
        _full_grid: bool = False,
    ) -> ProductLayer:
        """Build one lazy :class:`ProductLayer` bound to this view."""
        from faninsar.io.storage.product import open_product
        from faninsar.network.layers import ProductLayer

        store = _store if _store is not None else self._active_store
        full = self._full or _full_grid
        extent = None if full else store.selected_extent()
        origin = (0, 0) if full else store.selected_origin()
        product = open_product(store.root, kind=SCENE_KIND)
        return ProductLayer(
            product,
            name,
            asset_kind,
            store.grid_shape,
            multilook=(1, 1),
            origin=origin,
            crs=store.crs,
            extent=extent,
            _scene_selection=self._selection,
            _upstream=self._upstream,
            _session=self._session,
        )

    # ---- raster layers ---------------------------------------------------

    @property
    def slc(self) -> ProductLayer:
        """Return the virtual mosaic of the selected units' SLC payloads."""
        from faninsar.network.products import AssetKind

        return self._layer("slc", AssetKind.SLC)

    @property
    def amplitude(self) -> ProductLayer:
        """Return ``|slc|`` — the magnitude of the complex mosaic."""
        from faninsar.network.products import AssetKind

        return self._layer("amplitude", AssetKind.AMPLITUDE)

    @property
    def intensity(self) -> ProductLayer:
        """Return ``|slc|²`` — the squared magnitude of the complex mosaic."""
        from faninsar.network.products import AssetKind

        return self._layer("intensity", AssetKind.AMPLITUDE)

    @property
    def phase(self) -> ProductLayer:
        """Return ``angle(slc)`` — the raw SLC phase in ``[-pi, pi]``."""
        from faninsar.network.products import AssetKind

        return self._layer("phase", AssetKind.WRAPPED_PHASE)

    # ---- Reference-bound geometry layers (PROPOSAL-0066) -----------------

    def _geometry_layer(self, name: str) -> ProductLayer:
        """Return one committed Reference geometry layer, materializing lazily.

        The six radar-domain geometry fields are committed on the Reference
        scene product, so they are only available on the Stack Reference.
        The first access runs the rdr2geo solve once and commits all six
        payloads; later reads open the persisted zarr directly.
        """
        from faninsar.io.storage.scene_geometry import (
            SCENE_GEOMETRY_ASSETS,
            commit_scene_geometry,
        )
        from faninsar.network.products import AssetKind

        if name not in SCENE_GEOMETRY_ASSETS:
            reject_invalid_state(f"unknown scene geometry layer {name!r}")
        if not self._store.is_reference:
            reject_invalid_state(
                f"geometry layer {name!r} is bound to the Stack Reference; "
                f"scene {self._store.date_id!r} is a secondary"
            )
        store = self._store
        if name not in open_product(store.root, kind=SCENE_KIND).assets:
            if store._session is None:
                reject_invalid_state(
                    f"geometry layer {name!r} is not committed and this scene "
                    "is not bound to a Stack session; obtain it via "
                    "stack.scenes to materialize the geometry assets"
                )
            store = commit_scene_geometry(store)
        asset_kind = AssetKind.LABEL if name == "geom_flags" else AssetKind.DISPLACEMENT
        return self._layer(name, asset_kind, _store=store, _full_grid=True)

    def _los_layer(self, name: str) -> ProductLayer:
        """Return one virtual LOS unit-vector layer derived at read time."""
        from faninsar.io.storage.scene_geometry import VIRTUAL_LOS_LAYERS
        from faninsar.network.products import AssetKind

        if name not in VIRTUAL_LOS_LAYERS:
            reject_invalid_state(f"unknown LOS layer {name!r}")
        if not self._store.is_reference:
            reject_invalid_state(
                f"LOS layer {name!r} is bound to the Stack Reference; "
                f"scene {self._store.date_id!r} is a secondary"
            )
        return self._layer(name, AssetKind.DISPLACEMENT, _full_grid=True)

    @property
    def latitude(self) -> ProductLayer:
        """Return the committed Reference geodetic latitude field (degrees)."""
        return self._geometry_layer("latitude")

    @property
    def longitude(self) -> ProductLayer:
        """Return the committed Reference geodetic longitude field (degrees)."""
        return self._geometry_layer("longitude")

    @property
    def height(self) -> ProductLayer:
        """Return the committed Reference ellipsoidal height field (metres)."""
        return self._geometry_layer("height")

    @property
    def incidence(self) -> ProductLayer:
        """Return the committed Reference incidence-angle field (degrees)."""
        return self._geometry_layer("incidence")

    @property
    def azimuth(self) -> ProductLayer:
        """Return the committed Reference look-azimuth field (degrees)."""
        return self._geometry_layer("azimuth")

    @property
    def geom_flags(self) -> ProductLayer:
        """Return the committed layover/shadow bitmask (bit0 layover, bit1 shadow)."""
        return self._geometry_layer("geom_flags")

    @property
    def los_east(self) -> ProductLayer:
        """Return the virtual east LOS unit-vector component (ground to sat)."""
        return self._los_layer("los_east")

    @property
    def los_north(self) -> ProductLayer:
        """Return the virtual north LOS unit-vector component (ground to sat)."""
        return self._los_layer("los_north")

    @property
    def los_up(self) -> ProductLayer:
        """Return the virtual up LOS unit-vector component (ground to sat)."""
        return self._los_layer("los_up")

    def __repr__(self) -> str:
        """Return a concise summary of this scene layer view."""
        selected = (
            len(self._selection)
            if self._selection is not None
            else len(self._store.units)
        )
        frame = "full" if self._full else "bbox"
        return (
            f"SceneLayers(date={self._store.date_id!r}, kind={self.unit_kind!r}, "
            f"units={selected}/{len(self._store.units)}, frame={frame!r}, "
            f"layers=['slc', 'amplitude', 'intensity', 'phase'])"
        )


__all__ = [
    "MERGED_SCENE_UNIT_TAG",
    "NISAR_GEO_COVERAGE_POLICY",
    "SCENE_KIND",
    "CoregisteredSceneStore",
    "SceneLayers",
    "SceneUnit",
    "SceneWriter",
    "build_scene_merge_layout",
    "form_burst_scene_interferogram",
    "form_merged_scene_interferogram",
    "form_scene_interferograms",
    "merge_scene_slc",
    "scene_grid_identity",
    "scene_units_overlap",
    "valid_samples",
    "validate_reference_binding",
    "validate_scene_ownership",
    "write_scene",
]
