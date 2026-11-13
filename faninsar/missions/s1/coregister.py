"""Sentinel-1 Stack coregistration with the Reference computed once (P0059).

:func:`prepare_s1_reference` reads and deramps the Reference bursts exactly
once per Stack, commits the Reference scene product (radar: the cropped
Reference SLC; geo: the geocoded Reference SLC), and returns a
metadata-only :class:`S1ReferenceContext`. :func:`coregister_s1_secondary`
then processes one secondary date against that context; Reference pixels
are never re-read from SAFE nor re-deramped per date.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.io.storage.scene_store import (
    CoregisteredSceneStore,
    SceneWriter,
    scene_grid_identity,
)
from faninsar.logging import setup_logger
from faninsar.missions.s1.measure import _resolve_common_bursts, _resolve_swaths
from faninsar.missions.s1.processing import (
    SPEED_OF_LIGHT_M_S,
    BurstSelection,
    ProductionPairState,
    _as_frame_paths,
    _common_burst_pairs,
    _first_swath,
    _normalize_burst_selection,
    _phase_state_manifest,
    _roi_burst_window,
    _scene_id,
    _select_bursts_by_roi,
    _swath_range_offsets,
    _union_swath_names,
    build_reference_geo_lut,
    load_production_scene,
    reference_window_crop,
    stage_coregister,
    stage_deramp,
)
from faninsar.missions.s1.safe import open_acquisition_products
from faninsar.processing.coregistration.tops.deramp import deramp, reramp
from faninsar.processing.errors import (
    InvalidProcessingStateError,
    reject_invalid_state,
)
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.record import radar_geometry_record

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.core.orbit import OrbitMetadata
    from faninsar.data.query import BoundingBox, Polygons
    from faninsar.io.storage.product import Product
    from faninsar.io.storage.unit_checkpoint import (
        UnitCheckpoint,
        UnitCheckpointStore,
    )
    from faninsar.missions.s1.processing import ProductionScene
    from faninsar.missions.s1.types import S1Product
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
    from faninsar.processing.geometry import DEM
    from faninsar.processing.mosaicking.grid import GeoGridSpec

logger = setup_logger(__name__)

S1_SLC_VALIDITY_REVISION = "mask-aware-lanczos-v1"
"""Scientific validity revision of S1 complex SLC resampling (P0078)."""

_S1_SLC_VALIDITY_REVISION_KEY = "faninsar:slc_validity_revision"

_S1_COREGISTRATION_RECIPE: dict[str, str] = {
    "deramp": "s1-tops-deramp",
    "reramp": "s1-tops-reramp",
    "coregistration": "s1-tops-coregister",
    "flatten_stage": "coregistration",
}
"""Declared result-affecting recipe names for unit checkpoint specifications."""

_UNIT_SPEC_SCHEMA = 1

__all__ = [
    "S1_SLC_VALIDITY_REVISION",
    "S1ReferenceContext",
    "S1ReferenceUnit",
    "coregister_s1_scene",
    "coregister_s1_secondary",
    "prepare_s1_reference",
]


@dataclass(slots=True)
class S1ReferenceUnit:
    """Metadata of one Reference burst unit; holds no Reference pixels.

    Attributes
    ----------
    tag : str
        Unit tag ``f<frame>_<swath>_b<burst>``.
    frame_index, swath, burst_index : int, str, int
        Burst identity within the Reference frames.
    scene : ProductionScene
        Reference burst geometry, carrier, and placement with zero-stride
        placeholder samples.
    native_shape : tuple[int, int]
        Full native burst shape.
    roi_window : tuple[int, int, int, int] or None
        Radar ROI window on the native burst (radar ROI runs only).
    crop : tuple[int, int, int, int] or None
        Pair-independent Reference crop of ``roi_window``.
    row_origin, col_origin : int
        Placement of the persisted unit on the scene grid.
    parent_row0, parent_col0 : int
        Full-burst origin on the parent radar lattice, before ROI cropping.
    lut : Geo2RdrLUT or None
        Reference geo2rdr LUT (geo runs only).
    scientific_support : numpy.ndarray or None
        Destination geometry/ROI/LUT admissibility of the published window.
        Per-date echo validity is combined only at publication and is never
        written back onto this shared plan.
    publication_id : str or None
        Committed local reference-unit checkpoint write id. Secondaries bind
        this unit-local publication instead of the whole Reference scene id.
    specification_id : str or None
        Canonical id of the local reference-unit scientific specification.
    support_key : str or None
        Canonical window/geometry signature of this unit's actual support.

    """

    tag: str
    frame_index: int
    swath: str
    burst_index: int
    scene: ProductionScene
    native_shape: tuple[int, int]
    roi_window: tuple[int, int, int, int] | None = None
    crop: tuple[int, int, int, int] | None = None
    row_origin: int = 0
    col_origin: int = 0
    parent_row0: int = 0
    parent_col0: int = 0
    lut: Geo2RdrLUT | None = None
    scientific_support: np.ndarray | None = None
    publication_id: str | None = None
    specification_id: str | None = None
    support_key: str | None = None


@dataclass(slots=True)
class S1ReferenceContext:
    """Everything a secondary date needs from the once-computed Reference.

    Attributes
    ----------
    reference_id : str
        Reference acquisition identifier.
    product : CoregisteredSceneStore
        Committed Reference scene product.
    coregistration_grid : {"radar", "geo"}
        Product grid.
    grid_shape : tuple[int, int]
        Scene grid shape.
    units : dict[str, S1ReferenceUnit]
        Reference units by tag, in placement order.
    geo_grid : GeoGridSpec or None
        Geographic grid for geo runs.
    roi : BoundingBox or Polygons or None
        Apply-window ROI.
    dem : DEM
        Height sampler.
    polarizations : tuple of str or None
        The single channel published on this Reference layout. One
        ``MergeScope.channel`` cannot carry two polarizations.
    checkpoint_root : pathlib.Path or None
        Local root of committed reference-unit checkpoints.
    integrity_root : pathlib.Path or None
        Strict integrity baselines owned by the Stack workspace.
    verification : {"fast", "strict"}
        Checkpoint reuse policy. ``strict`` rereads committed unit payloads
        against their baselines and is never the default.
    dem_identity : str or None
        Cheap authoritative DEM descriptor that participated in the units.
    roi_identity : str or None
        Canonical ROI geometry identity that participated in the units.
    reference_orbit_path, reference_aux_cal_path : pathlib.Path or None
        Explicit Reference orbit/AUX_CAL inputs.

    """

    reference_id: str
    product: CoregisteredSceneStore
    coregistration_grid: Literal["radar", "geo"]
    grid_shape: tuple[int, int]
    units: dict[str, S1ReferenceUnit] = field(default_factory=dict)
    geo_grid: GeoGridSpec | None = None
    roi: BoundingBox | Polygons | None = None
    dem: DEM | None = None
    polarizations: tuple[str, ...] | None = None
    checkpoint_root: Path | None = None
    integrity_root: Path | None = None
    verification: Literal["fast", "strict"] = "fast"
    dem_identity: str | None = None
    roi_identity: str | None = None
    reference_orbit_path: Path | None = None
    reference_aux_cal_path: Path | None = None

    @property
    def write_id(self) -> str:
        """Return the Reference product write id secondaries bind."""
        return self.product.write_id


def _selected_polarizations(
    polarizations: Sequence[str] | str | None,
) -> tuple[str, ...] | None:
    """Return requested channels, or ``None`` when the caller kept every channel."""
    if polarizations is None:
        return None
    if isinstance(polarizations, str):
        if polarizations.strip().lower() == "all":
            return None
        tokens: Sequence[str] = (polarizations,)
    else:
        tokens = polarizations
    selected: list[str] = []
    for value in tokens:
        for part in str(value).replace(" ", "").split("+"):
            token = part.strip().upper()
            if token and token not in selected:
                selected.append(token)
    return tuple(selected) or None


def _present_polarizations(products: Sequence[S1Product]) -> tuple[str, ...]:
    """Return unique polarizations present on opened SAFE products."""
    return tuple(
        dict.fromkeys(
            str(swath.polarization).upper()
            for product in products
            for swath in product.swaths
        )
    )


def _restrict_product_polarizations(
    product: S1Product, selected: tuple[str, ...]
) -> S1Product:
    """Return ``product`` carrying only ``selected`` measurement swaths."""
    wanted = set(selected)
    filtered = tuple(
        swath for swath in product.swaths if str(swath.polarization).upper() in wanted
    )
    if not filtered:
        reject_invalid_state(
            f"requested polarization {selected[0]} is not present on "
            f"{product.source_path}; available={product.polarizations}"
        )
    polarizations = tuple(
        dict.fromkeys(str(swath.polarization).upper() for swath in filtered)
    )
    if filtered == product.swaths and product.polarizations == polarizations:
        return product
    return replace(product, swaths=filtered, polarizations=polarizations)


def _require_single_coreg_polarization(
    polarizations: Sequence[str] | str | None,
    products: Sequence[S1Product],
) -> tuple[str, ...]:
    """Return the one channel this layout may publish.

    Dual-pol SAFE annotations sort VH before VV. Opening every present
    channel, or requesting several, would otherwise publish the first
    annotation rather than the requested measurement.
    """
    present = _present_polarizations(products)
    requested = _selected_polarizations(polarizations)
    if requested is not None:
        if len(requested) > 1:
            reject_invalid_state(
                "coregistration publishes one channel per layout; "
                f"requested polarizations {requested} are ambiguous"
            )
        if requested[0] not in present:
            reject_invalid_state(
                f"requested polarization {requested[0]} is not present; "
                f"available={present}"
            )
        return requested
    if len(present) == 1:
        return present
    if not present:
        reject_invalid_state("SAFE product has no polarizations")
    return reject_invalid_state(
        "coregistration publishes one channel per layout; "
        f"SAFE polarizations {present} are ambiguous — request a single channel"
    )


def _coreg_open_filter(
    polarizations: Sequence[str] | str | None,
) -> tuple[str, ...] | None:
    """Return the SAFE open filter, rejecting multi-channel requests first."""
    requested = _selected_polarizations(polarizations)
    if requested is not None and len(requested) > 1:
        reject_invalid_state(
            "coregistration publishes one channel per layout; "
            f"requested polarizations {requested} are ambiguous"
        )
    return requested


def _open_coreg_products(
    paths: Sequence[Path],
    polarizations: Sequence[str] | str | None,
) -> tuple[list[S1Product], tuple[str, ...]]:
    """Open SAFE products and restrict them to one coregistration channel."""
    products = open_acquisition_products(
        paths, polarizations=_coreg_open_filter(polarizations)
    )
    selected = _require_single_coreg_polarization(polarizations, products)
    restricted = [
        _restrict_product_polarizations(product, selected) for product in products
    ]
    return restricted, selected


def _require_checkpoint_policy(
    checkpoint_root: Path | None,
    verification: str,
    integrity_root: Path | None,
) -> None:
    """Validate the opt-in checkpoint/verification combination."""
    if verification not in {"fast", "strict"}:
        reject_invalid_state(
            f"verification must be 'fast' or 'strict', got {verification!r}"
        )
    if verification == "strict" and (checkpoint_root is None or integrity_root is None):
        reject_invalid_state(
            "strict checkpoint verification requires both checkpoint_root and "
            "integrity_root"
        )


def _checkpoint_token(value: object) -> str:
    """Return a safe single-component checkpoint locator token."""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "unit"


def _file_identity(value: object) -> dict[str, object] | None:
    """Return a small path/stat identity for one explicitly consumed file.

    SAFE directories are deliberately not stat'ed: extending a shared canvas
    directory changes its mtime without changing the consumed burst bytes and
    would invalidate every local unit. Only the actual annotation, measurement,
    orbit, and AUX_CAL inputs named here are recorded.
    """
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    if "://" in text:
        return {"path": text}
    path = Path(text).expanduser().resolve(strict=False)
    identity: dict[str, object] = {"path": str(path)}
    try:
        stat = path.stat()
    except OSError:
        return identity
    identity["size"] = int(stat.st_size)
    identity["mtime_ns"] = int(stat.st_mtime_ns)
    return identity


def _finite_float_or_none(value: object) -> float | None:
    """Return a finite float or ``None`` for absent/non-finite values."""
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None


def _source_identity(
    product: S1Product,
    *,
    swath: object | None = None,
    burst: object | None = None,
) -> dict[str, object]:
    """Return the physical acquisition/burst identity consumed by one unit."""
    identity: dict[str, object] = {
        "product_id": str(product.product_id),
        "source_path": str(
            Path(product.source_path).expanduser().resolve(strict=False)
        ),
        "mission_id": str(product.mission_id),
        "product_type": str(product.product_type),
        "mode": str(product.mode),
        "absolute_orbit": int(product.absolute_orbit),
        "polarizations": [str(value) for value in product.polarizations],
        "ipf_version": getattr(product, "ipf_version", None),
    }
    if swath is not None:
        identity["swath"] = {
            "name": str(getattr(swath, "swath", "")),
            "polarization": str(getattr(swath, "polarization", "")),
            "annotation": _file_identity(getattr(swath, "annotation_path", None)),
            "measurement": _file_identity(getattr(swath, "measurement_path", None)),
        }
    if burst is not None:
        identity["burst"] = {
            "index": int(getattr(burst, "index", -1)),
            "absolute_burst_id": getattr(burst, "absolute_burst_id", None),
            "azimuth_anx_time_s": _finite_float_or_none(
                getattr(burst, "azimuth_anx_time_s", None)
            ),
            "lines": int(getattr(burst, "lines", 0)),
            "samples": int(getattr(burst, "samples", 0)),
        }
    return identity


def _source_token(product: S1Product) -> str:
    """Return a stable checkpoint token for one physical source acquisition."""
    raw = getattr(product, "product_id", None)
    if not raw:
        raw = Path(str(getattr(product, "source_path", ""))).name
    return _checkpoint_token(raw)


def _roi_identity(roi: BoundingBox | Polygons | None) -> str:
    """Return a canonical ROI geometry identity, never a mutable list index."""
    if roi is None:
        return "none"
    from faninsar.data.query import BoundingBox

    if isinstance(roi, BoundingBox):
        crs = roi.crs.to_string() if roi.crs else None
        payload: dict[str, object] = {
            "kind": "bbox",
            "crs": crs,
            "bounds": [float(value) for value in roi],
        }
    else:
        geometry = roi.to_desired().geodataframe.to_crs(4326)
        members = sorted(
            member.wkb.hex() for member in geometry.geometry if member is not None
        )
        payload = {
            "kind": "polygons",
            "crs": "EPSG:4326",
            "wkb": members,
            "all_touched": bool(getattr(roi, "all_touched", False)),
            "pad": bool(getattr(roi, "pad", False)),
        }
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _absolute_placement(
    unit: S1ReferenceUnit, coregistration_grid: str
) -> tuple[int, int]:
    """Return placement independent of the current scene's growing origin."""
    if coregistration_grid == "radar":
        crop = unit.crop
        row = int(unit.parent_row0) + (int(crop[0]) if crop is not None else 0)
        col = int(unit.parent_col0) + (int(crop[2]) if crop is not None else 0)
        return row, col
    return int(unit.row_origin), int(unit.col_origin)


def _lut_geometry_identity(unit: S1ReferenceUnit) -> dict[str, object] | None:
    """Return the cheap LUT grid bounds, never LUT payload contents."""
    lut = unit.lut
    if lut is None:
        return None
    return {
        "row0": int(lut.row0),
        "col0": int(lut.col0),
        "shape": [int(lut.shape[0]), int(lut.shape[1])],
    }


def _geo_unit_crop_footprint(
    unit: S1ReferenceUnit,
    *,
    geo_grid: GeoGridSpec | None,
) -> object | None:
    """Return an exact EPSG:4326 Geo window polygon, or ``None``.

    A north-up affine window is an exact rectangle only in its own CRS. A
    nonlinear projection to EPSG:4326 could bow past corner edges, so this
    helper is used only for already-geographic EPSG:4326 grids; all other
    Geo grids fall back to the canonical full ROI identity.
    """
    from affine import Affine
    from shapely.geometry import box

    lut = unit.lut
    if lut is None or geo_grid is None or str(geo_grid.crs) != "EPSG:4326":
        return None
    affine = geo_grid.transform
    if not isinstance(affine, Affine):
        affine = Affine(*tuple(affine)[:6])  # type: ignore[arg-type]
    row0, col0 = int(lut.row0), int(lut.col0)
    rows, cols = int(lut.shape[0]), int(lut.shape[1])
    x0, y0 = affine * (col0, row0)
    x1, y1 = affine * (col0 + cols, row0 + rows)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _unit_local_roi_identity(
    unit: S1ReferenceUnit,
    *,
    coregistration_grid: str,
    roi: BoundingBox | Polygons | None,
    geo_grid: GeoGridSpec | None,
) -> str:
    """Return a conservative local ROI identity for one published unit.

    Radar has no certified cheap footprint that bounds the nonlinear
    rdr2geo mapping; a sampled hull can miss support, so radar keeps the full
    canonical ROI identity and deliberately recomputes on any ROI change.
    Geo clips to the exact affine window only for an EPSG:4326 grid.
    """
    if roi is None:
        return "none"
    if coregistration_grid != "geo":
        return _roi_identity(roi)
    from shapely import normalize

    from faninsar.missions.s1.roi_support import roi_lonlat_geometry

    footprint = _geo_unit_crop_footprint(unit, geo_grid=geo_grid)
    if footprint is None or footprint.is_empty:
        return _roi_identity(roi)
    local = normalize(roi_lonlat_geometry(roi).intersection(footprint))
    if local.is_empty:
        return "empty"
    return json.dumps(
        {
            "kind": local.geom_type,
            "wkb": local.wkb.hex(),
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reference_support_metadata(
    unit: S1ReferenceUnit,
    *,
    coregistration_grid: str,
    roi: BoundingBox | Polygons | None,
    geo_grid: GeoGridSpec | None,
    geo_grid_identity: str | None,
) -> dict[str, object]:
    """Return the canonical support window/geometry metadata for one unit."""
    crop = unit.crop
    roi_window = unit.roi_window
    return {
        "domain": coregistration_grid,
        "roi": _unit_local_roi_identity(
            unit,
            coregistration_grid=coregistration_grid,
            roi=roi,
            geo_grid=geo_grid,
        ),
        "native_shape": [int(value) for value in unit.native_shape],
        "roi_window": (
            None if roi_window is None else [int(value) for value in roi_window]
        ),
        "crop": None if crop is None else [int(value) for value in crop],
        "payload_shape": [
            int(value) for value in _persisted_unit_shape(unit, coregistration_grid)
        ],
        "absolute_placement": [
            int(value) for value in _absolute_placement(unit, coregistration_grid)
        ],
        "geo_grid_identity": geo_grid_identity,
        "lut": _lut_geometry_identity(unit),
    }


def _support_key(support: Mapping[str, object]) -> str:
    """Return a short canonical digest of small support/geometry metadata."""
    text = json.dumps(
        dict(support),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _reference_unit_locator(
    product: S1Product, unit: S1ReferenceUnit, support_key: str
) -> str:
    """Return the stable checkpoint locator of one physical reference burst."""
    return "_".join(
        (
            "reference",
            _source_token(product),
            _checkpoint_token(unit.swath),
            f"b{int(unit.burst_index)}",
            support_key,
        )
    )


def _secondary_unit_locator(
    date_id: str,
    product: S1Product,
    unit: S1ReferenceUnit,
    secondary_burst_index: int,
    support_key: str,
) -> str:
    """Return the stable checkpoint locator of one secondary physical burst."""
    return "_".join(
        (
            _checkpoint_token(date_id),
            _source_token(product),
            _checkpoint_token(unit.swath),
            f"b{int(secondary_burst_index)}",
            support_key,
        )
    )


def _reference_unit_specification(
    unit: S1ReferenceUnit,
    *,
    reference_id: str,
    source: Mapping[str, object],
    orbit: Mapping[str, object] | None,
    aux_cal: Mapping[str, object] | None,
    dem_identity: str,
    coregistration_grid: str,
    support_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Return the small scientific specification of one reference burst unit."""
    from faninsar.missions.s1.merge_layout import physical_burst_id

    return {
        "schema": _UNIT_SPEC_SCHEMA,
        "stage": "s1_reference_unit",
        "reference_id": reference_id,
        "source": dict(source),
        "physical_burst_id": physical_burst_id(unit),
        "coregistration_grid": coregistration_grid,
        "support": dict(support_metadata),
        "support_key": _support_key(support_metadata),
        "orbit": None if orbit is None else dict(orbit),
        "aux_cal": None if aux_cal is None else dict(aux_cal),
        "dem": dem_identity,
        "slc_validity_revision": S1_SLC_VALIDITY_REVISION,
    }


def _secondary_unit_specification(
    context: S1ReferenceContext,
    unit: S1ReferenceUnit,
    *,
    tag: str,
    source: Mapping[str, object],
    secondary_source: Mapping[str, object],
    secondary_burst_index: int,
    secondary_orbit: Mapping[str, object] | None,
    secondary_aux_cal: Mapping[str, object] | None,
    dem_identity: str,
    support_metadata: Mapping[str, object],
    geo_grid_identity: str | None,
    misreg_az_px: float,
    misreg_rg_px: float,
) -> dict[str, object]:
    """Return the small scientific specification of one secondary burst unit."""
    from faninsar.missions.s1.merge_layout import physical_burst_id

    support = dict(support_metadata)
    support_key = unit.support_key or _support_key(support)
    publication_id = unit.publication_id or (f"scene:{context.product.write_id}:{tag}")
    return {
        "schema": _UNIT_SPEC_SCHEMA,
        "stage": "s1_coregister_secondary",
        "coregistration_grid": context.coregistration_grid,
        "reference": {
            "reference_id": context.reference_id,
            "unit_publication_id": publication_id,
            "unit_specification_id": unit.specification_id,
            "physical_burst_id": physical_burst_id(unit),
            "source": dict(source),
            "support": support,
            "support_key": support_key,
        },
        "secondary": {
            "source": dict(secondary_source),
            "burst_index": int(secondary_burst_index),
        },
        "orbit": {
            "reference": _file_identity(context.reference_orbit_path),
            "secondary": None if secondary_orbit is None else dict(secondary_orbit),
        },
        "aux_cal": {
            "reference": _file_identity(context.reference_aux_cal_path),
            "secondary": None if secondary_aux_cal is None else dict(secondary_aux_cal),
        },
        "dem": dem_identity,
        "roi": support.get("roi", "none"),
        "geo_grid_identity": geo_grid_identity,
        "correction": {
            "misreg_az_px": float(misreg_az_px),
            "misreg_rg_px": float(misreg_rg_px),
        },
        "phase_recipe": {
            **_S1_COREGISTRATION_RECIPE,
            "slc_validity_revision": S1_SLC_VALIDITY_REVISION,
        },
    }


def _verify_committed_checkpoint(
    checkpoint: UnitCheckpoint,
    *,
    integrity_root: Path,
) -> bool:
    """Return whether strict bytes match this checkpoint's own baseline."""
    from faninsar.stack.integrity import verify_integrity

    verdict = verify_integrity(
        integrity_root,
        checkpoint.path,
        strict=True,
        write_id=checkpoint.write_id,
    )
    return verdict.status == "verified"


def _establish_committed_checkpoint(
    checkpoint: UnitCheckpoint,
    *,
    integrity_root: Path,
) -> None:
    """Record a strict baseline immediately after a new checkpoint commit."""
    from faninsar.stack.integrity import establish_integrity

    establish_integrity(integrity_root, checkpoint.path, write_id=checkpoint.write_id)


def _load_reusable_checkpoint(
    store: UnitCheckpointStore,
    unit_id: str,
    specification: Mapping[str, object],
) -> UnitCheckpoint | None:
    """Load one committed unit, treating a typed structural rejection as a miss.

    Checkpoints published before fill chunks were materialized, or whose
    stored chunk disappeared, cannot be replayed and are recomputed. The
    replacement is published only by the caller's atomic commit, which still
    refuses a foreign product kind or unit locator at the same path; a decode
    failure during replay stays an error instead of becoming a silent reuse.
    """
    try:
        return store.load(unit_id, specification)
    except InvalidProcessingStateError as error:
        logger.warning(
            "%s: committed checkpoint is not reusable (%s); recomputing this unit",
            unit_id,
            error,
        )
        return None


def _replay_scene_unit(
    writer: SceneWriter,
    *,
    tag: str,
    checkpoint: UnitCheckpoint,
    row_origin: int,
    col_origin: int,
) -> None:
    """Replay committed arrays with current placement, phase, and mask."""
    writer.add_unit(
        tag=tag,
        payload=checkpoint.read_slc(),
        row_origin=int(row_origin),
        col_origin=int(col_origin),
        scientific_lineage=checkpoint.scientific_lineage,
        phase_state=checkpoint.phase_state,
        phase_screen=checkpoint.read_phase(),
        valid_mask=checkpoint.read_valid(),
        burst_id=checkpoint.burst_id,
        absolute_burst_id=checkpoint.absolute_burst_id,
    )


def _replay_reference_unit(
    writer: SceneWriter,
    unit: S1ReferenceUnit,
    checkpoint: UnitCheckpoint,
    *,
    coregistration_grid: str,
    scene_row0: int,
    scene_col0: int,
) -> None:
    """Replay one reference unit on the current scene origin."""
    if coregistration_grid == "radar":
        row, col = _absolute_placement(unit, coregistration_grid)
        row -= int(scene_row0)
        col -= int(scene_col0)
    else:
        row, col = int(unit.row_origin), int(unit.col_origin)
    _replay_scene_unit(
        writer,
        tag=unit.tag,
        checkpoint=checkpoint,
        row_origin=row,
        col_origin=col,
    )


def _replay_secondary_unit(
    writer: SceneWriter,
    *,
    tag: str,
    checkpoint: UnitCheckpoint,
    row_origin: int,
    col_origin: int,
) -> None:
    """Replay one secondary unit at its current Reference placement."""
    _replay_scene_unit(
        writer,
        tag=tag,
        checkpoint=checkpoint,
        row_origin=row_origin,
        col_origin=col_origin,
    )


def prepare_s1_reference(
    reference_path: Path | Sequence[Path],
    *,
    output_scene: Path,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    roi: BoundingBox | Polygons | None = None,
    dem: DEM,
    coregistration_grid: Literal["radar", "geo"] = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_lut_cache_dir: Path | None = None,
    orbit_path: Path | None = None,
    aux_cal_path: Path | None = None,
    polarizations: Sequence[str] | str | None = None,
    device: str = "auto",
    overwrite: bool = False,
    checkpoint_root: Path | None = None,
    verification: Literal["fast", "strict"] = "fast",
    integrity_root: Path | None = None,
) -> S1ReferenceContext:
    """Compute the Reference scene once and return its context.

    When ``output_scene`` already holds a committed Reference product and
    ``overwrite`` is false, Reference pixels are not read: only burst
    metadata, carriers, geometry, and (geo) LUTs are loaded.

    Parameters
    ----------
    reference_path : path or sequence of path
        Reference SAFE product(s), one per frame.
    output_scene : pathlib.Path
        Reference scene product directory, ``<work_dir>/coreg/<date>``.
    swaths : tuple of str, optional
        Sub-swaths; ``None`` selects all (or the ROI footprint set).
    bursts : BurstSelection, optional
        Per-swath burst selection.
    roi : BoundingBox or Polygons, optional
        Apply-window crop; selects bursts when ``bursts`` is omitted.
    dem : DEM
        Height sampler for geometry.
    coregistration_grid : {"radar", "geo"}, optional
        Product grid.
    geo_grid : GeoGridSpec, optional
        Required for geo coregistration.
    geo_lut_cache_dir : pathlib.Path, optional
        Shared content-addressed LUT cache (geo).
    orbit_path : pathlib.Path, optional
        Precise orbit of the Reference.
    aux_cal_path : pathlib.Path, optional
        AUX_CAL file of the Reference; required for IPF 002.36 products.
    polarizations : sequence of str or str, optional
        The one channel this layout publishes. Dual-pol SAFEs list VH
        before VV; ``None`` or ``"all"`` is allowed only when the product
        already carries a single polarization. Requesting several channels
        is rejected rather than processing the first annotation.
    device : str, optional
        Numerical device.
    overwrite : bool, optional
        Recompute and replace an existing Reference product.
    checkpoint_root : pathlib.Path, optional
        Root for committed per-physical-burst Reference units. Each unit is
        keyed by its own source, burst, and actual support/geometry metadata;
        unchanged units are replayed instead of recomputed.
    verification : {"fast", "strict"}, optional
        Checkpoint reuse policy. ``fast`` performs no payload hashing.
    integrity_root : pathlib.Path, optional
        Strict baseline root; required when ``verification="strict"``.

    Returns
    -------
    S1ReferenceContext
        Metadata-only Reference context.

    Raises
    ------
    InvalidProcessingStateError
        If the selection is empty, the configuration is invalid, an
        existing Reference product does not match the current selection,
        the requested polarizations are ambiguous for one layout, or strict
        verification is requested without its checkpoint/baseline roots.

    """
    _require_checkpoint_policy(checkpoint_root, verification, integrity_root)
    if coregistration_grid not in {"radar", "geo"}:
        reject_invalid_state("coregistration_grid must be 'radar' or 'geo'")
    if coregistration_grid == "geo" and geo_grid is None:
        reject_invalid_state("geo coregistration requires geo_grid")
    primary_paths = _as_frame_paths(reference_path, "reference_path")
    primary_products, selected_polarizations = _open_coreg_products(
        primary_paths, polarizations
    )
    ref_orbits = [orbit_path] * len(primary_products)
    swath_tuple, resolved = _resolve_apply_selection(
        primary_products,
        primary_products,
        primary_paths=primary_paths,
        swaths=swaths,
        bursts=bursts,
        roi=roi,
        ref_orbits=ref_orbits,
        dem=dem,
    )
    range_offsets = _swath_range_offsets(swath_tuple, primary_products)
    units_by_swath, frame_rows, frame_cols = _unit_placement(
        primary_products, resolved, swath_tuple, range_offsets
    )
    reference_id = _scene_id(primary_paths[0])
    output_scene = Path(output_scene)
    reuse = (output_scene / "item.json").is_file() and not overwrite
    replace_stale_validity = False
    if reuse:
        existing = CoregisteredSceneStore.open(output_scene)
        if not _has_current_slc_validity_revision(existing):
            logger.info(
                "Reference %s lacks current SLC validity revision %s; recomputing",
                reference_id,
                S1_SLC_VALIDITY_REVISION,
            )
            reuse = False
            replace_stale_validity = True
    scene_row0 = 0
    scene_col0 = 0
    if coregistration_grid == "geo":
        grid_shape = (int(geo_grid.shape[0]), int(geo_grid.shape[1]))  # type: ignore[union-attr]
        header = _geo_header(geo_grid)  # type: ignore[arg-type]
    elif roi is not None:
        # Pass 1: resolve every unit's Reference crop without reading pixels so
        # the scene grid is bounded by the union of the ROI crops rather than
        # the nominal burst placement.
        units, grid_shape, scene_row0, scene_col0 = _radar_roi_units(
            primary_products,
            primary_paths,
            ref_orbits,
            units_by_swath,
            range_offsets,
            swath_tuple,
            roi=roi,
            dem=dem,
            device=device,
        )
        _require_selected_crop_dem_coverage(units, dem, device=device)
        header = {}
    else:
        grid_shape = (int(frame_rows), int(frame_cols))
        header = {}
    work_root = output_scene.parent / f".{output_scene.name}.work"
    writer: SceneWriter | None = None
    roi_units_resolved = coregistration_grid == "radar" and roi is not None
    if not roi_units_resolved:
        units = {}
    ordered = (
        (f"f{frame_index}_{swath}_b{burst_index}", frame_index)
        for swath in swath_tuple
        for frame_index, burst_index, _azimuth_offset in units_by_swath[swath]
    )
    store = None
    dem_identity: str | None = None
    roi_identity: str | None = None
    geo_grid_id: str | None = None
    if checkpoint_root is not None:
        from faninsar.io.storage.unit_checkpoint import UnitCheckpointStore

        store = UnitCheckpointStore(checkpoint_root)
        from faninsar.processing.geometry.api import dem_content_identity

        dem_identity = dem_content_identity(dem)
        roi_identity = _roi_identity(roi)
        if coregistration_grid == "geo":
            from faninsar.processing.geometry.prepared import (
                geo_grid_identity as _geo_grid_identity,
            )

            geo_grid_id = _geo_grid_identity(geo_grid)  # type: ignore[arg-type]
    pending_replays: list[tuple[S1ReferenceUnit, UnitCheckpoint]] = []

    def _start_writer(scene: ProductionScene) -> SceneWriter:
        """Create the Reference writer and replay any already-validated units."""
        nonlocal writer
        if writer is not None:
            return writer
        writer = SceneWriter(
            output_scene,
            date_id=reference_id,
            reference_id=reference_id,
            domain=coregistration_grid,
            grid_shape=grid_shape,
            wavelength_m=float(scene.geometry.wavelength_m),
            overwrite=overwrite or replace_stale_validity or reuse,
            **header,
        )
        for pending_unit, pending in pending_replays:
            _replay_reference_unit(
                writer,
                pending_unit,
                pending,
                coregistration_grid=coregistration_grid,
                scene_row0=scene_row0,
                scene_col0=scene_col0,
            )
        pending_replays.clear()
        return writer

    try:
        for tag, frame_index in ordered:
            if roi_units_resolved and tag not in units:
                continue
            swath = tag.split("_")[1]
            burst_index = int(tag.rsplit("_b", 1)[1])
            scene = load_production_scene(
                primary_paths[min(frame_index, len(primary_paths) - 1)],
                swath=swath,
                burst_index=burst_index,
                dem=dem,
                orbit_path=ref_orbits[frame_index],
                aux_cal_path=aux_cal_path,
                coregistration_grid=coregistration_grid,
                full_range=True,
                read_samples=not reuse,
                product=primary_products[frame_index],
            )
            if store is None and writer is None and not reuse:
                writer = _start_writer(scene)
            unit = units.get(tag)
            parent_row0 = int(
                next(
                    azimuth_offset
                    for f_i, b_i, azimuth_offset in units_by_swath[swath]
                    if f_i == frame_index and b_i == burst_index
                )
            )
            parent_col0 = int(range_offsets[swath])
            if unit is None:
                unit = S1ReferenceUnit(
                    tag=tag,
                    frame_index=frame_index,
                    swath=swath,
                    burst_index=burst_index,
                    scene=scene,
                    native_shape=(
                        int(scene.array.samples.shape[0]),
                        int(scene.array.samples.shape[1]),
                    ),
                    row_origin=int(parent_row0 - scene_row0),
                    col_origin=int(parent_col0 - scene_col0),
                    parent_row0=parent_row0,
                    parent_col0=parent_col0,
                )
            else:
                unit.parent_row0 = parent_row0
                unit.parent_col0 = parent_col0
            unit.scene = scene
            if store is not None:
                if coregistration_grid == "radar":
                    _prepare_radar_unit(
                        unit, roi=roi, dem=dem, compute=False, device=device
                    )
                else:
                    _prepare_geo_unit(
                        unit,
                        dem=dem,
                        geo_grid=geo_grid,  # type: ignore[arg-type]
                        geo_lut_cache_dir=geo_lut_cache_dir,
                        work_dir=work_root / tag,
                        roi=roi,
                        compute=False,
                        device=device,
                    )
                    if unit.lut is None:
                        logger.warning("%s: empty geocoded footprint; skipped", tag)
                        del scene
                        continue
                support = _reference_support_metadata(
                    unit,
                    coregistration_grid=coregistration_grid,
                    roi=roi,
                    geo_grid=geo_grid,
                    geo_grid_identity=geo_grid_id,
                )
                unit.support_key = _support_key(support)
                source_product = primary_products[frame_index]
                unit_id = _reference_unit_locator(
                    source_product, unit, unit.support_key
                )
                specification = _reference_unit_specification(
                    unit,
                    reference_id=reference_id,
                    source=_source_identity(
                        source_product,
                        swath=unit.scene.swath,
                        burst=unit.scene.burst,
                    ),
                    orbit=_file_identity(ref_orbits[frame_index]),
                    aux_cal=_file_identity(aux_cal_path),
                    dem_identity=dem_identity or "none",
                    coregistration_grid=coregistration_grid,
                    support_metadata=support,
                )
                committed = _load_reusable_checkpoint(store, unit_id, specification)
                if (
                    committed is not None
                    and verification == "strict"
                    and not _verify_committed_checkpoint(
                        committed,
                        integrity_root=integrity_root,  # type: ignore[arg-type]
                    )
                ):
                    logger.warning(
                        "%s: strict integrity did not verify committed checkpoint; "
                        "recomputing this unit",
                        unit_id,
                    )
                    committed = None
                if committed is not None:
                    unit.publication_id = committed.write_id
                    unit.specification_id = committed.specification_id
                    unit.scene = _without_pixels(scene)
                    units[tag] = unit
                    if writer is not None:
                        _replay_reference_unit(
                            writer,
                            unit,
                            committed,
                            coregistration_grid=coregistration_grid,
                            scene_row0=scene_row0,
                            scene_col0=scene_col0,
                        )
                    elif not reuse:
                        _replay_reference_unit(
                            _start_writer(scene),
                            unit,
                            committed,
                            coregistration_grid=coregistration_grid,
                            scene_row0=scene_row0,
                            scene_col0=scene_col0,
                        )
                    else:
                        pending_replays.append((unit, committed))
                    del scene
                    continue
                # A checkpoint miss only authorizes recomputing this unit.
                if reuse:
                    scene = load_production_scene(
                        primary_paths[min(frame_index, len(primary_paths) - 1)],
                        swath=swath,
                        burst_index=burst_index,
                        dem=dem,
                        orbit_path=ref_orbits[frame_index],
                        aux_cal_path=aux_cal_path,
                        coregistration_grid=coregistration_grid,
                        full_range=True,
                        read_samples=True,
                        product=primary_products[frame_index],
                    )
                    unit.scene = scene
                if coregistration_grid == "radar":
                    prepared = _prepare_radar_unit(
                        unit, roi=roi, dem=dem, compute=True, device=device
                    )
                else:
                    prepared = _prepare_geo_unit(
                        unit,
                        dem=dem,
                        geo_grid=geo_grid,  # type: ignore[arg-type]
                        geo_lut_cache_dir=geo_lut_cache_dir,
                        work_dir=work_root / tag,
                        roi=roi,
                        compute=True,
                        device=device,
                    )
                if prepared is None:
                    logger.warning("%s: empty geocoded footprint; skipped", tag)
                    del scene
                    continue
                payload, published = prepared
                from faninsar.io.storage.unit_checkpoint import UnitPayloads
                from faninsar.missions.s1.merge_layout import physical_burst_id
                from faninsar.missions.s1.scene_bursts import (
                    official_absolute_burst_id,
                )

                burst_id = physical_burst_id(unit)
                absolute_burst_id = official_absolute_burst_id(
                    getattr(unit.scene, "burst", None)
                )
                lineage = [
                    {
                        "stage": "s1_reference_burst",
                        "source": str(primary_paths[frame_index]),
                    }
                ]
                row_origin, col_origin = _absolute_placement(unit, coregistration_grid)
                checkpoint = store.commit(
                    unit_id,
                    specification,
                    payloads=UnitPayloads(
                        slc=np.asarray(payload, dtype=np.complex64),
                        row_origin=row_origin,
                        col_origin=col_origin,
                        valid=(
                            None
                            if published is None
                            else np.asarray(published, dtype=bool)
                        ),
                        burst_id=burst_id,
                        absolute_burst_id=absolute_burst_id,
                        scientific_lineage=lineage,
                        phase_state=None,
                    ),
                )
                if verification == "strict":
                    _establish_committed_checkpoint(
                        checkpoint,
                        integrity_root=integrity_root,  # type: ignore[arg-type]
                    )
                unit.publication_id = checkpoint.write_id
                unit.specification_id = checkpoint.specification_id
                unit.scene = _without_pixels(scene)
                units[tag] = unit
                _stage_scene_unit(
                    _start_writer(scene),
                    tag=tag,
                    payload=payload,
                    row_origin=unit.row_origin,
                    col_origin=unit.col_origin,
                    valid_mask=published,
                    burst_id=burst_id,
                    scientific_lineage=lineage,
                    absolute_burst_id=absolute_burst_id,
                )
                del payload, prepared, scene
                continue
            if coregistration_grid == "radar":
                payload = _prepare_radar_unit(
                    unit, roi=roi, dem=dem, compute=not reuse, device=device
                )
            else:
                payload = _prepare_geo_unit(
                    unit,
                    dem=dem,
                    geo_grid=geo_grid,  # type: ignore[arg-type]
                    geo_lut_cache_dir=geo_lut_cache_dir,
                    work_dir=work_root / tag,
                    roi=roi,
                    compute=not reuse,
                    device=device,
                )
                if unit.lut is None:
                    logger.warning("%s: empty geocoded footprint; skipped", tag)
                    continue
            unit.scene = _without_pixels(scene)
            units[tag] = unit
            if writer is not None and payload is not None:
                from faninsar.missions.s1.merge_layout import physical_burst_id

                samples, published = payload
                from faninsar.missions.s1.scene_bursts import official_absolute_burst_id

                _stage_scene_unit(
                    writer,
                    tag=tag,
                    payload=samples,
                    row_origin=unit.row_origin,
                    col_origin=unit.col_origin,
                    valid_mask=published,
                    burst_id=physical_burst_id(unit),
                    scientific_lineage=[
                        {
                            "stage": "s1_reference_burst",
                            "source": str(primary_paths[frame_index]),
                        }
                    ],
                    absolute_burst_id=official_absolute_burst_id(
                        getattr(unit.scene, "burst", None)
                    ),
                )
            del payload, scene
        if not units:
            reject_invalid_state("Reference selection produced no scene units")
        if writer is not None:
            _stage_reference_footprint(writer, units)
            properties: dict[str, object] = {}
            if coregistration_grid == "radar":
                properties["faninsar:radar_geometry"] = frame_geometry_record(
                    primary_products,
                    resolved,
                    swath_tuple,
                    range_offsets,
                    grid_shape,
                    scene_offset=(scene_row0, scene_col0),
                    orbit=next(iter(units.values())).scene.swath.orbit,
                )
            # The Reference is its own baseline origin (PROPOSAL-0065).
            properties["faninsar:baseline_parallel_m"] = 0.0
            properties["faninsar:baseline_perpendicular_m"] = 0.0
            properties[_S1_SLC_VALIDITY_REVISION_KEY] = S1_SLC_VALIDITY_REVISION
            if store is not None:
                properties["faninsar:reference_unit_publications"] = {
                    tag: unit.publication_id for tag, unit in units.items()
                }
                properties["faninsar:reference_unit_specifications"] = {
                    tag: unit.specification_id for tag, unit in units.items()
                }
            from faninsar.missions.s1.merge_layout import (
                build_s1_reference_layout,
                publish_reference_merge_layout,
                track_identity,
            )

            layout = build_s1_reference_layout(
                units,
                domain=coregistration_grid,
                reference_id=reference_id,
                track_id=track_identity(primary_products[0]),
                channel=selected_polarizations[0].lower(),
                grid_identity=writer.grid_identity,
                roi=roi,
                scene_offset=(scene_row0, scene_col0),
            )
            layout_product = publish_reference_merge_layout(
                output_scene,
                layout,
                overwrite=overwrite or replace_stale_validity or reuse,
            )
            properties["faninsar:layout_id"] = layout_product.write_id
            product = writer.commit(
                properties=properties,
                inputs={"layout": layout_product.write_id},
            )
        else:
            from faninsar.missions.s1.merge_layout import physical_burst_id

            product = CoregisteredSceneStore.open(output_scene)
            _check_reused_reference(
                product,
                reference_id,
                units,
                coregistration_grid,
                grid_shape=grid_shape,
                grid_identity=(
                    str(header["grid_identity"]) if "grid_identity" in header else None
                ),
                expected_publications=(
                    None
                    if store is None
                    else {tag: unit.publication_id for tag, unit in units.items()}
                ),
                expected_specifications=(
                    None
                    if store is None
                    else {tag: unit.specification_id for tag, unit in units.items()}
                ),
                expected_burst_ids={
                    tag: physical_burst_id(unit) for tag, unit in units.items()
                },
            )
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    logger.info(
        "Reference %s %s once: %d units",
        reference_id,
        "reused" if reuse and writer is None else "computed",
        len(units),
    )
    return S1ReferenceContext(
        reference_id=reference_id,
        product=product,
        coregistration_grid=coregistration_grid,
        grid_shape=grid_shape,
        units=units,
        geo_grid=geo_grid,
        roi=roi,
        dem=dem,
        polarizations=selected_polarizations,
        checkpoint_root=checkpoint_root,
        integrity_root=integrity_root,
        verification=verification,
        dem_identity=dem_identity,
        roi_identity=roi_identity,
        reference_orbit_path=orbit_path,
        reference_aux_cal_path=aux_cal_path,
    )


def _radar_roi_units(
    primary_products: Sequence[S1Product],
    primary_paths: Sequence[Path],
    ref_orbits: Sequence[Path | None],
    units_by_swath: Mapping[str, list[tuple[int, int, int]]],
    range_offsets: Mapping[str, int],
    swath_tuple: tuple[str, ...],
    *,
    roi: BoundingBox | Polygons,
    dem: DEM,
    device: str,
) -> tuple[dict[str, S1ReferenceUnit], tuple[int, int], int, int]:
    """Resolve every Reference unit's ROI crop without reading burst pixels.

    Returns the metadata-only units keyed by tag, the ROI-bounded scene grid
    shape, and the scene-grid origin ``(row0, col0)`` in absolute placement
    coordinates.  Units whose ROI window resolves to ``None`` keep the full
    burst crop; the scene grid then still covers every unit.

    """
    units: dict[str, S1ReferenceUnit] = {}
    scene_r0 = scene_c0 = np.iinfo(np.int64).max
    scene_r1 = scene_c1 = 0
    for swath in swath_tuple:
        for frame_index, burst_index, azimuth_offset in units_by_swath[swath]:
            tag = f"f{frame_index}_{swath}_b{burst_index}"
            scene = load_production_scene(
                primary_paths[min(frame_index, len(primary_paths) - 1)],
                swath=swath,
                burst_index=burst_index,
                dem=dem,
                orbit_path=ref_orbits[frame_index],
                coregistration_grid="radar",
                full_range=True,
                read_samples=False,
                product=primary_products[frame_index],
            )
            native_shape = (
                int(scene.array.samples.shape[0]),
                int(scene.array.samples.shape[1]),
            )
            unit = S1ReferenceUnit(
                tag=tag,
                frame_index=frame_index,
                swath=swath,
                burst_index=burst_index,
                scene=scene,
                native_shape=native_shape,
                row_origin=int(azimuth_offset),
                col_origin=int(range_offsets[swath]),
                parent_row0=int(azimuth_offset),
                parent_col0=int(range_offsets[swath]),
            )
            unit.roi_window = _roi_burst_window(
                roi,
                scene.geometry,
                dem,
                native_shape,
                device=device,
                first_valid_sample=scene.burst.first_valid_sample,
                last_valid_sample=scene.burst.last_valid_sample,
            )
            if unit.roi_window is None:
                logger.info("%s: ROI does not intersect this burst; skipped", tag)
                continue
            unit.crop = reference_window_crop(unit.roi_window, native_shape)
            scene_r0 = min(scene_r0, unit.row_origin + unit.crop[0])
            scene_r1 = max(scene_r1, unit.row_origin + unit.crop[1])
            scene_c0 = min(scene_c0, unit.col_origin + unit.crop[2])
            scene_c1 = max(scene_c1, unit.col_origin + unit.crop[3])
            unit.scene = _without_pixels(scene)
            units[tag] = unit
    if not units or scene_r1 <= scene_r0 or scene_c1 <= scene_c0:
        reject_invalid_state("Reference ROI selection produced no scene units")
    for unit in units.values():
        unit.row_origin += unit.crop[0] - scene_r0  # type: ignore[index]
        unit.col_origin += unit.crop[2] - scene_c0  # type: ignore[index]
    grid_shape = (int(scene_r1 - scene_r0), int(scene_c1 - scene_c0))
    return units, grid_shape, int(scene_r0), int(scene_c0)


def _require_selected_crop_dem_coverage(
    units: Mapping[str, S1ReferenceUnit],
    dem: DEM,
    *,
    device: str,
) -> None:
    """Raise once if the union of selected crop DEM demands exceeds ``dem``.

    Constant and other non-raster samplers are skipped. Each unit uses the
    crop already resolved by :func:`_radar_roi_units` (full native burst when
    ``crop`` is unset). The union is the currently selected windows only.
    """
    from faninsar.processing.geometry.api import RasterDEM
    from faninsar.processing.geometry.managed_dem import (
        require_dem_coverage,
        union_bounds,
    )
    from faninsar.processing.geometry.prepare_production import (
        _rdr2geo_dem_required_bounds,
    )

    if not isinstance(dem, RasterDEM):
        return
    union: tuple[float, float, float, float] | None = None
    for unit in units.values():
        crop = unit.crop
        if crop is None:
            crop = (0, unit.native_shape[0], 0, unit.native_shape[1])
        row0, row1, col0, col1 = (int(value) for value in crop)
        if row1 <= row0 or col1 <= col0:
            reject_invalid_state("radar ROI crop must have positive shape")
        rows = np.arange(row0, row1, dtype=np.float64)[:, None]
        cols = np.arange(col0, col1, dtype=np.float64)[None, :]
        azimuth, slant = np.broadcast_arrays(rows, cols)
        demand = _rdr2geo_dem_required_bounds(
            unit.scene.geometry,
            azimuth,
            slant,
            dem,
            device=device,
            max_iter=20,
            range_tol_m=0.01,
            doppler_tol_hz=0.1,
        )
        if demand is None:
            continue
        union = demand if union is None else union_bounds(union, demand)
    if union is None:
        return
    require_dem_coverage(dem, union)


def _prepare_radar_unit(
    unit: S1ReferenceUnit,
    *,
    roi: BoundingBox | Polygons | None,
    dem: DEM,
    compute: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Resolve the Reference ROI crop and return the radar Reference payload.

    The payload is ``reramp(deramp(reference))`` on the Reference crop, the
    same pointwise values the per-pair path produced. Returns ``None`` when
    ``compute`` is false (the persisted product is reused).
    """
    scene = unit.scene
    if roi is not None and unit.crop is None:
        unit.roi_window = _roi_burst_window(
            roi,
            scene.geometry,
            dem,
            unit.native_shape,
            device=device,
            first_valid_sample=scene.burst.first_valid_sample,
            last_valid_sample=scene.burst.last_valid_sample,
        )
        if unit.roi_window is not None:
            unit.crop = reference_window_crop(unit.roi_window, unit.native_shape)
            unit.row_origin += unit.crop[0]
            unit.col_origin += unit.crop[2]
    crop = unit.crop
    if crop is None:
        crop = (0, unit.native_shape[0], 0, unit.native_shape[1])
    native_valid = _window_native_valid(scene.array.valid_mask, crop)
    payload: np.ndarray | None = None
    if compute:
        deramped = np.where(
            scene.array.valid_mask, deramp(scene.array.samples, scene.carrier), 0
        )
        row0, row1, col0, col1 = crop
        deramped = deramped[row0:row1, col0:col1]
        payload = np.asarray(
            reramp(
                deramped,
                scene.carrier,
                row0=row0,
                col0=col0,
                native_height=unit.native_shape[0],
            ),
            dtype=np.complex64,
        )
    shape = (int(crop[1] - crop[0]), int(crop[3] - crop[2]))
    if unit.scientific_support is None:
        if roi is not None:
            from faninsar.missions.s1.roi_support import radar_scientific_support

            unit.scientific_support = radar_scientific_support(
                scene.geometry,
                dem,
                crop=crop,
                native_valid=None,
                roi=roi,
                device=device,
            )
        else:
            unit.scientific_support = np.ones(shape, dtype=bool)
    if payload is None:
        return None
    published = _published_unit_mask(
        unit.scientific_support, native_valid, payload, tag=unit.tag
    )
    from faninsar.missions.s1.roi_support import apply_scientific_support

    payload = apply_scientific_support(payload, published)
    _warn_unsupported_inside_support(unit.tag, unit.scientific_support, published)
    return payload, published


def _prepare_geo_unit(
    unit: S1ReferenceUnit,
    *,
    dem: DEM,
    geo_grid: GeoGridSpec,
    geo_lut_cache_dir: Path | None,
    work_dir: Path,
    roi: BoundingBox | Polygons | None,
    compute: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the Reference LUT and (unless reusing) the geocoded Reference."""
    from faninsar.processing.geocoding.geo_modes import (
        geocode_reference_slc_chunked,
    )

    scene = unit.scene
    lut = build_reference_geo_lut(
        scene,
        radar_shape=unit.native_shape,
        dem=dem,
        geo_grid=geo_grid,
        storage_dir=work_dir / "lut",
        geo_lut_cache_dir=geo_lut_cache_dir,
        roi=roi,
        device=device,
    )
    if lut.shape[0] == 0 or lut.shape[1] == 0 or not np.any(lut.valid):
        return None
    lut = _materialize_lut(lut)
    unit.lut = lut
    unit.row_origin = int(lut.row0)
    unit.col_origin = int(lut.col0)
    payload: np.ndarray | None = None
    sampled_valid: np.ndarray | None = None
    if compute:
        native_mask = scene.array.valid_mask
        if native_mask is None:
            native_mask = np.ones(scene.array.samples.shape, dtype=bool)
        else:
            native_mask = np.asarray(native_mask, dtype=bool)
        deramped = np.where(native_mask, deramp(scene.array.samples, scene.carrier), 0)
        reference_geo, sampled_valid = geocode_reference_slc_chunked(
            deramped,
            reference_carrier=scene.carrier,
            reference_lut=lut,
            output_dir=work_dir / "slc",
            device=device,
            source_valid=native_mask,
        )
        payload = np.array(reference_geo, dtype=np.complex64)
        if sampled_valid is None:
            sampled_valid = np.ones(payload.shape, dtype=bool)
        else:
            sampled_valid = np.asarray(sampled_valid, dtype=bool)
    window = payload.shape if payload is not None else tuple(lut.valid.shape)
    if unit.scientific_support is None:
        if roi is not None:
            from faninsar.missions.s1.roi_support import geo_scientific_support

            unit.scientific_support = geo_scientific_support(
                geo_grid,
                row0=int(lut.row0),
                col0=int(lut.col0),
                shape=window,
                native_valid=np.asarray(lut.valid, dtype=bool),
                roi=roi,
            )
        else:
            unit.scientific_support = np.asarray(lut.valid, dtype=bool)
    if payload is None:
        return None
    published = _published_unit_mask(
        unit.scientific_support, sampled_valid, payload, tag=unit.tag
    )
    from faninsar.missions.s1.roi_support import apply_scientific_support

    payload = apply_scientific_support(payload, published)
    _warn_unsupported_inside_support(unit.tag, unit.scientific_support, published)
    return payload, published


def _materialize_lut(lut: Geo2RdrLUT) -> Geo2RdrLUT:
    """Return the LUT held in memory so its scratch directory can be removed."""
    return replace(
        lut,
        az_full=np.array(lut.az_full),
        rg_full=np.array(lut.rg_full),
        valid=np.array(lut.valid),
        height_full=None if lut.height_full is None else np.array(lut.height_full),
    )


def _without_pixels(scene: ProductionScene) -> ProductionScene:
    """Return ``scene`` with zero-stride placeholder samples (metadata only)."""
    shape = scene.array.samples.shape
    array = replace(
        scene.array,
        samples=np.broadcast_to(np.complex64(0), shape),
        valid_mask=np.broadcast_to(np.True_, shape),
    )
    return replace(scene, array=array)


def _persisted_unit_shape(
    unit: S1ReferenceUnit, coregistration_grid: str
) -> tuple[int, int]:
    """Return the published window shape for one Reference unit.

    Radar units persist the native burst, or the ROI crop of that burst.
    Geo units persist the LUT target window on the geographic grid, which
    is not the native radar burst shape.
    """
    if coregistration_grid == "geo":
        lut = unit.lut
        if lut is None:
            return reject_invalid_state(
                f"{unit.tag}: geo Reference unit is missing LUT"
            )
        return (int(lut.shape[0]), int(lut.shape[1]))
    if unit.crop is not None:
        return (
            int(unit.crop[1] - unit.crop[0]),
            int(unit.crop[3] - unit.crop[2]),
        )
    return unit.native_shape


def _has_current_slc_validity_revision(product: CoregisteredSceneStore) -> bool:
    """Return whether ``product`` was written with the current validity revision."""
    stored = product.properties.get(_S1_SLC_VALIDITY_REVISION_KEY)
    return stored == S1_SLC_VALIDITY_REVISION


def _check_reused_reference(
    product: CoregisteredSceneStore,
    reference_id: str,
    units: Mapping[str, S1ReferenceUnit],
    coregistration_grid: str,
    *,
    grid_shape: tuple[int, int],
    grid_identity: str | None = None,
    expected_publications: Mapping[str, str | None] | None = None,
    expected_specifications: Mapping[str, str | None] | None = None,
    expected_burst_ids: Mapping[str, str] | None = None,
) -> None:
    """Reject a persisted Reference that no longer matches the selection.

    Local unit publications are validated before a whole-scene shortcut is
    accepted: an existing ``item.json`` is never sufficient evidence by itself.
    """
    if not product.is_reference or product.reference_id != reference_id:
        reject_invalid_state(
            f"{product.root} is not the Reference scene of {reference_id}"
        )
    if product.domain != coregistration_grid:
        reject_invalid_state(
            f"persisted Reference is {product.domain!r}, requested "
            f"{coregistration_grid!r}; recoregister with overwrite=True"
        )
    if coregistration_grid == "geo" and tuple(product.grid_shape) != tuple(grid_shape):
        reject_invalid_state(
            f"persisted Reference grid {tuple(product.grid_shape)} differs from "
            f"requested {tuple(grid_shape)}; recoregister with overwrite=True"
        )
    if (
        coregistration_grid == "geo"
        and grid_identity is not None
        and product.grid_identity != grid_identity
    ):
        reject_invalid_state(
            "persisted Reference grid_identity differs from the current "
            "selection; recoregister with overwrite=True"
        )
    stored = product.unit_map()
    if set(stored) != set(units):
        reject_invalid_state(
            "persisted Reference units differ from the current selection; "
            "recoregister with overwrite=True"
        )
    if expected_publications is not None:
        stored_publications = product.properties.get(
            "faninsar:reference_unit_publications"
        )
        if not isinstance(stored_publications, Mapping):
            reject_invalid_state(
                "persisted Reference has no local unit publications; "
                "recoregister with overwrite=True"
            )
        if dict(stored_publications) != dict(expected_publications):
            reject_invalid_state(
                "persisted Reference local unit publications differ from the "
                "current selection; recoregister with overwrite=True"
            )
    if expected_specifications is not None:
        stored_specifications = product.properties.get(
            "faninsar:reference_unit_specifications"
        )
        if not isinstance(stored_specifications, Mapping):
            reject_invalid_state(
                "persisted Reference has no local unit specifications; "
                "recoregister with overwrite=True"
            )
        if dict(stored_specifications) != dict(expected_specifications):
            reject_invalid_state(
                "persisted Reference local unit specifications differ from the "
                "current selection; recoregister with overwrite=True"
            )
    for tag, unit in units.items():
        placed = stored[tag]
        if (placed.row_origin, placed.col_origin) != (unit.row_origin, unit.col_origin):
            reject_invalid_state(
                f"persisted Reference unit {tag!r} is placed differently from "
                "the current selection; recoregister with overwrite=True"
            )
        expected_shape = _persisted_unit_shape(unit, coregistration_grid)
        if tuple(placed.shape) != tuple(expected_shape):
            reject_invalid_state(
                f"persisted Reference unit {tag!r} has shape {tuple(placed.shape)}, "
                f"expected {tuple(expected_shape)}; recoregister with overwrite=True"
            )
        if expected_burst_ids is not None:
            expected_burst = expected_burst_ids.get(tag)
            if expected_burst is not None and placed.burst_id != expected_burst:
                reject_invalid_state(
                    f"persisted Reference unit {tag!r} burst id "
                    f"{placed.burst_id!r} differs from {expected_burst!r}; "
                    "recoregister with overwrite=True"
                )
        if unit.scientific_support is not None and tuple(
            unit.scientific_support.shape
        ) != tuple(placed.shape):
            reject_invalid_state(
                f"persisted Reference unit {tag!r} support shape "
                f"{tuple(unit.scientific_support.shape)} differs from payload "
                f"{tuple(placed.shape)}; recoregister with overwrite=True"
            )
    if not _has_current_slc_validity_revision(product):
        reject_invalid_state(
            "persisted Reference lacks the current SLC validity revision; "
            "recoregister with overwrite=True"
        )


def coregister_s1_secondary(
    context: S1ReferenceContext,
    secondary_path: Path | Sequence[Path],
    *,
    output_scene: Path,
    misreg_az_px: float,
    misreg_rg_px: float,
    secondary_orbit_path: Path | None = None,
    secondary_aux_cal_path: Path | None = None,
    polarizations: Sequence[str] | str | None = None,
    device: str = "auto",
    inputs: Mapping[str, Product | str] | None = None,
    properties: Mapping[str, object] | None = None,
    overwrite: bool = False,
    checkpoint_root: Path | None = None,
    verification: Literal["fast", "strict"] = "fast",
    integrity_root: Path | None = None,
) -> CoregisteredSceneStore:
    """Coregister one secondary date onto the once-prepared Reference.

    Only the secondary is read, deramped, and resampled. Reference units the
    secondary does not cover are omitted; samples outside the secondary
    support stay invalid. ESD and Ampcor are not measured.

    Parameters
    ----------
    context : S1ReferenceContext
        Result of :func:`prepare_s1_reference`.
    secondary_path : path or sequence of path
        Secondary SAFE product(s), one per Reference frame.
    output_scene : pathlib.Path
        Secondary scene product directory, ``<work_dir>/coreg/<date>``.
    misreg_az_px, misreg_rg_px : float
        Inverted date constants (0 for geometry-only). Must be finite.
    secondary_orbit_path : pathlib.Path, optional
        Precise orbit of the secondary.
    secondary_aux_cal_path : pathlib.Path, optional
        AUX_CAL file of the secondary; required for IPF 002.36 products.
    polarizations : sequence of str or str, optional
        Channel to open. Defaults to the Reference context channel so the
        secondary cannot silently load a different polarization.
    device : str, optional
        Numerical device.
    inputs : mapping, optional
        Extra upstream bindings such as ``misreg``; ``reference`` is always
        bound to the Reference product.
    properties : mapping, optional
        Extra item properties.
    overwrite : bool, optional
        Replace an existing secondary product.
    checkpoint_root : pathlib.Path, optional
        Root for committed per-burst scientific-unit checkpoints. Each unit
        binds its physical secondary source, orbit/AUX/DEM inputs, the local
        Reference unit publication, and the actual support/geometry window.
    verification : {"fast", "strict"}, optional
        Checkpoint reuse policy. ``fast`` performs no payload hashing.
    integrity_root : pathlib.Path, optional
        Strict baseline root; required when ``verification="strict"``.

    Returns
    -------
    CoregisteredSceneStore
        The committed secondary scene product.

    Raises
    ------
    InvalidProcessingStateError
        If the constants are non-finite, the date is the Reference, the
        secondary covers no Reference unit, or strict verification is
        requested without its checkpoint/baseline roots.

    """
    _require_checkpoint_policy(checkpoint_root, verification, integrity_root)
    if not np.isfinite(misreg_az_px) or not np.isfinite(misreg_rg_px):
        reject_invalid_state("misreg_az_px and misreg_rg_px must be finite")
    secondary_paths = _as_frame_paths(secondary_path, "secondary_path")
    date_id = _scene_id(secondary_paths[0])
    if date_id == context.reference_id:
        reject_invalid_state("coregister_s1_secondary requires a non-Reference date")
    secondary_products, _ = _open_coreg_products(
        secondary_paths,
        polarizations if polarizations is not None else context.polarizations,
    )
    reference = context.product
    reference_units = reference.unit_map()
    domain = context.coregistration_grid
    header = _geo_header(context.geo_grid) if domain == "geo" else {}  # type: ignore[arg-type]
    output_scene = Path(output_scene)
    work_root = output_scene.parent / f".{output_scene.name}.work"
    wrote = 0
    unit_baselines: list[tuple[float, float]] = []
    store = None
    dem_identity: str | None = None
    geo_grid_id: str | None = None
    if checkpoint_root is not None:
        from faninsar.io.storage.unit_checkpoint import UnitCheckpointStore
        from faninsar.processing.geometry.api import dem_content_identity

        store = UnitCheckpointStore(checkpoint_root)
        dem_identity = dem_content_identity(context.dem)
        if domain == "geo" and context.geo_grid is not None:
            from faninsar.processing.geometry.prepared import geo_grid_identity

            geo_grid_id = geo_grid_identity(context.geo_grid)
    matches = {
        tag: _secondary_burst_match(unit, secondary_products, tag, date_id)
        for tag, unit in context.units.items()
    }
    identities = [
        (id(product), context.units[tag].swath, index)
        for tag, match in matches.items()
        if match is not None
        for index, product in (match,)
    ]
    if len(set(identities)) != len(identities):
        reject_invalid_state("secondary burst is matched to multiple Reference units")
    try:
        with SceneWriter(
            output_scene,
            date_id=date_id,
            reference_id=context.reference_id,
            domain=domain,
            grid_shape=context.grid_shape,
            wavelength_m=reference.wavelength_m,
            overwrite=overwrite,
            **header,
        ) as writer:
            for tag, unit in context.units.items():
                match = matches[tag]
                if match is None:
                    continue
                secondary_burst_index, secondary_product = match
                placed = reference_units[tag]
                committed = None
                specification = None
                unit_id = ""
                if store is not None:
                    support = _reference_support_metadata(
                        unit,
                        coregistration_grid=domain,
                        roi=context.roi,
                        geo_grid=context.geo_grid,
                        geo_grid_identity=geo_grid_id,
                    )
                    support_key = unit.support_key or _support_key(support)
                    unit_id = _secondary_unit_locator(
                        date_id,
                        secondary_product,
                        unit,
                        secondary_burst_index,
                        support_key,
                    )
                    secondary_swath = secondary_product.find_swath(
                        unit.swath, unit.scene.swath.polarization
                    )
                    secondary_burst = None
                    if secondary_swath is not None and 0 <= secondary_burst_index < len(
                        secondary_swath.bursts
                    ):
                        secondary_burst = secondary_swath.bursts[secondary_burst_index]
                    specification = _secondary_unit_specification(
                        context,
                        unit,
                        tag=tag,
                        source=_source_identity(
                            unit.scene.product,
                            swath=unit.scene.swath,
                            burst=unit.scene.burst,
                        ),
                        secondary_source=_source_identity(
                            secondary_product,
                            swath=secondary_swath,
                            burst=secondary_burst,
                        ),
                        secondary_burst_index=secondary_burst_index,
                        secondary_orbit=_file_identity(secondary_orbit_path),
                        secondary_aux_cal=_file_identity(secondary_aux_cal_path),
                        dem_identity=dem_identity or "none",
                        support_metadata=support,
                        geo_grid_identity=geo_grid_id,
                        misreg_az_px=float(misreg_az_px),
                        misreg_rg_px=float(misreg_rg_px),
                    )
                    committed = _load_reusable_checkpoint(store, unit_id, specification)
                    if (
                        committed is not None
                        and verification == "strict"
                        and not _verify_committed_checkpoint(
                            committed,
                            integrity_root=integrity_root,  # type: ignore[arg-type]
                        )
                    ):
                        logger.warning(
                            "%s: strict integrity did not verify committed checkpoint; "
                            "recomputing this unit",
                            unit_id,
                        )
                        committed = None
                if committed is not None:
                    _replay_secondary_unit(
                        writer,
                        tag=tag,
                        checkpoint=committed,
                        row_origin=placed.row_origin,
                        col_origin=placed.col_origin,
                    )
                    parallel = committed.properties.get("baseline_parallel_m")
                    perpendicular = committed.properties.get("baseline_perpendicular_m")
                    if parallel is not None and perpendicular is not None:
                        unit_baselines.append((float(parallel), float(perpendicular)))
                    wrote += 1
                    logger.info("Reused committed scene unit %s", unit_id)
                    continue
                payload, published, state = _coregister_unit(
                    context,
                    unit,
                    secondary_product.source_path,
                    secondary_burst_index=secondary_burst_index,
                    reference=reference,
                    misreg_az_px=float(misreg_az_px),
                    misreg_rg_px=float(misreg_rg_px),
                    secondary_orbit_path=secondary_orbit_path,
                    secondary_aux_cal_path=secondary_aux_cal_path,
                    work_dir=work_root / tag,
                    device=device,
                    product=secondary_product,
                )
                if payload.shape != placed.shape:
                    reject_invalid_state(
                        f"{tag}: secondary shape {payload.shape} differs from "
                        f"the Reference {placed.shape}"
                    )
                if placed.burst_id is None:
                    reject_invalid_state(
                        f"{tag}: Reference unit is missing physical burst_id"
                    )

                from faninsar.missions.s1.scene_bursts import (
                    official_absolute_burst_id,
                )

                absolute_burst_id = official_absolute_burst_id(
                    getattr(state.secondary, "burst", None)
                )
                phase_state = _phase_state_manifest(state)
                baseline = _unit_scene_baseline(state)
                if store is not None:
                    from faninsar.io.storage.unit_checkpoint import UnitPayloads

                    assert specification is not None
                    checkpoint = store.commit(
                        unit_id,
                        specification,
                        payloads=UnitPayloads(
                            slc=np.asarray(payload, dtype=np.complex64),
                            row_origin=placed.row_origin,
                            col_origin=placed.col_origin,
                            valid=(
                                None
                                if published is None
                                else np.asarray(published, dtype=bool)
                            ),
                            burst_id=placed.burst_id,
                            absolute_burst_id=absolute_burst_id,
                            scientific_lineage=state.scientific_lineage,
                            phase_state=phase_state,
                        ),
                        properties={
                            "baseline_parallel_m": float(baseline[0]),
                            "baseline_perpendicular_m": float(baseline[1]),
                        },
                    )
                    if verification == "strict":
                        _establish_committed_checkpoint(
                            checkpoint,
                            integrity_root=integrity_root,  # type: ignore[arg-type]
                        )
                    _replay_secondary_unit(
                        writer,
                        tag=tag,
                        checkpoint=checkpoint,
                        row_origin=placed.row_origin,
                        col_origin=placed.col_origin,
                    )
                else:
                    _stage_scene_unit(
                        writer,
                        tag=tag,
                        payload=payload,
                        row_origin=placed.row_origin,
                        col_origin=placed.col_origin,
                        valid_mask=published,
                        burst_id=placed.burst_id,
                        scientific_lineage=state.scientific_lineage,
                        phase_state=phase_state,
                        absolute_burst_id=absolute_burst_id,
                    )
                unit_baselines.append(baseline)
                del payload, state
                shutil.rmtree(work_root / tag, ignore_errors=True)
                wrote += 1
            if wrote == 0:
                reject_invalid_state(f"secondary {date_id} covers no Reference unit")
            bindings: dict[str, Product | str] = dict(inputs or {})
            bindings["reference"] = reference.write_id
            record = {
                "faninsar:coregistration": {
                    "misreg_az_px": float(misreg_az_px),
                    "misreg_rg_px": float(misreg_rg_px),
                },
                **dict(properties or {}),
            }
            # Signed per-scene baseline against the Reference; the mean across
            # the scene's units (PROPOSAL-0065). Written after the caller
            # properties merge so a ``properties=`` value cannot shadow it.
            parallel_m, perpendicular_m = np.mean(
                np.asarray(unit_baselines, dtype=np.float64), axis=0
            )
            record["faninsar:baseline_parallel_m"] = float(parallel_m)
            record["faninsar:baseline_perpendicular_m"] = float(perpendicular_m)
            record[_S1_SLC_VALIDITY_REVISION_KEY] = S1_SLC_VALIDITY_REVISION
            product = writer.commit(inputs=bindings, properties=record)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    logger.info(
        "Coregistered %s onto Reference %s: %d/%d units",
        date_id,
        context.reference_id,
        wrote,
        len(context.units),
    )
    return product


def _unit_scene_baseline(state: ProductionPairState) -> tuple[float, float]:
    """Compute the scene baseline of one unit against the Reference.

    Mirrors :func:`stage_baseline`'s geocentric convention — each orbit is
    evaluated at *its own* scene's burst mid-azimuth time, and ``b =
    sec_pos - ref_pos`` is decomposed onto ``look = -ref_pos/|ref_pos|``.
    The perpendicular component is signed: projected onto the cross-track
    direction ``normalize(ref_velocity x look)`` (asf_search/ISCE2 sign
    convention), so it can be negative.

    Returns
    -------
    tuple[float, float]
        ``(parallel_m, perpendicular_m)`` of this unit's scene, metres.

    """
    from faninsar.processing.geometry.orbit import OrbitInterpolator

    mid_az = 0.5 * (state.primary.array.samples.shape[0] - 1)
    ref_time = state.primary.geometry.azimuth_time(mid_az)
    sec_time = state.secondary.geometry.azimuth_time(mid_az)
    state_ref = OrbitInterpolator.from_orbit(state.primary.swath.orbit).evaluate(
        ref_time
    )
    state_sec = OrbitInterpolator.from_orbit(state.secondary.swath.orbit).evaluate(
        sec_time
    )
    sat = np.asarray(state_ref.position_m, dtype=np.float64)
    look = -sat / np.linalg.norm(sat)
    baseline_vec = np.asarray(state_sec.position_m, dtype=np.float64) - sat
    parallel = float(np.dot(baseline_vec, look))
    perp_vec = baseline_vec - parallel * look
    velocity = np.asarray(state_ref.velocity_m_s, dtype=np.float64)
    cross_track = np.cross(velocity, look)
    cross_norm = np.linalg.norm(cross_track)
    if cross_norm > 0.0:
        cross_track = cross_track / cross_norm
        perpendicular = float(np.dot(perp_vec, cross_track))
    else:
        perpendicular = float(np.linalg.norm(perp_vec))
    return parallel, perpendicular


def _secondary_burst_match(
    unit: S1ReferenceUnit,
    secondary_products: Sequence[S1Product],
    tag: str,
    date_id: str,
) -> tuple[int, S1Product] | None:
    """Return ``(burst_index, product)`` for the unit in the secondary date.

    The reference ``unit.burst_index`` is matched to the secondary product's
    own burst grid by along-track position, so products clipped to different
    start lines (e.g. Sentinel-1C vs Sentinel-1D) still resolve.

    The secondary swath is searched across all products. Every candidate must
    belong to the Reference track and channel, and the ANX match must be
    unique across products. Frame ordering is not evidence of correspondence.
    """
    from faninsar.missions.s1.track_map import track_key_for_product

    reference_track = track_key_for_product(unit.scene.product)
    if reference_track is None:
        reject_invalid_state(f"{tag}: Reference track is unknown")
    matches: list[tuple[int, S1Product]] = []
    for product in secondary_products:
        track = track_key_for_product(product)
        if track is None:
            reject_invalid_state(f"{tag}: secondary track is unknown")
        if track != reference_track:
            reject_invalid_state(
                f"{tag}: secondary track {track} differs from "
                f"Reference {reference_track}"
            )
        candidate = product.find_swath(unit.swath, unit.scene.swath.polarization)
        if candidate is None:
            continue
        pairs = _common_burst_pairs(unit.scene.swath, candidate)
        secondary_index = pairs.get(unit.burst_index)
        if secondary_index is not None:
            matches.append((secondary_index, product))
    if len(matches) > 1:
        reject_invalid_state(f"{tag}: ambiguous secondary burst across products")
    if not matches:
        logger.info("%s: secondary %s does not cover this burst", tag, date_id)
        return None
    return matches[0]


def _coregister_unit(
    context: S1ReferenceContext,
    unit: S1ReferenceUnit,
    secondary_path: Path,
    *,
    secondary_burst_index: int,
    reference: CoregisteredSceneStore,
    misreg_az_px: float,
    misreg_rg_px: float,
    secondary_orbit_path: Path | None,
    work_dir: Path,
    device: str,
    product: S1Product | None = None,
    secondary_aux_cal_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, ProductionPairState]:
    """Resample one secondary burst onto its Reference unit."""
    domain = context.coregistration_grid
    secondary = load_production_scene(
        secondary_path,
        swath=unit.swath,
        burst_index=secondary_burst_index,
        dem=context.dem,
        orbit_path=secondary_orbit_path,
        aux_cal_path=secondary_aux_cal_path,
        coregistration_grid=domain,
        full_range=True,
        product=product,
    )
    state = ProductionPairState(
        pair_id=f"{context.reference_id}_{secondary.scene_id}_{unit.tag}",
        primary=unit.scene,
        secondary=secondary,
        dem=context.dem,  # type: ignore[arg-type]
        coregistration_grid=domain,
    )
    state = stage_deramp(state, device=device, secondary_only=True)
    if not reference.is_reference:
        reject_invalid_state(
            f"{unit.tag}: secondary coregistration requires the Reference store"
        )
    reference_geo_valid = None
    if domain == "geo":
        if unit.lut is None:
            reject_invalid_state(
                f"{unit.tag}: geo coregistration requires a Reference LUT"
            )
        reference_geo_valid = np.asarray(unit.lut.valid, dtype=bool)
    state = stage_coregister(
        state,
        misreg_az_px=misreg_az_px,
        misreg_rg_px=misreg_rg_px,
        device=device,
        coregistration_grid=domain,
        geo_grid=context.geo_grid,
        geo_work_dir=work_dir if domain == "geo" else None,
        roi=context.roi,
        roi_window=unit.roi_window,
        prepared_geo_lut=unit.lut,
        secondary_only=True,
        reference_shape=unit.native_shape,
        reference_geo_valid=reference_geo_valid,
    )
    if state.secondary_aligned is None:
        reject_invalid_state(f"{unit.tag}: coregistration produced no aligned SLC")
    payload = np.array(state.secondary_aligned, dtype=np.complex64)
    aligned_valid = state.secondary_aligned_valid
    if domain == "radar" and unit.crop is not None:
        payload = _crop_to_reference_window(state, unit.crop, payload, tag=unit.tag)
        if aligned_valid is not None:
            aligned_valid = _crop_to_reference_window(
                state, unit.crop, np.asarray(aligned_valid, dtype=bool), tag=unit.tag
            )
    geometry = unit.scientific_support
    if geometry is None:
        geometry = _compute_unit_support(context, unit, payload.shape)
        unit.scientific_support = geometry
    from faninsar.missions.s1.roi_support import apply_scientific_support

    published = _published_unit_mask(geometry, aligned_valid, payload, tag=unit.tag)
    payload = apply_scientific_support(payload, published)
    _warn_unsupported_inside_support(unit.tag, geometry, published)
    state.secondary_aligned = None
    state.secondary_aligned_valid = None
    return payload, published, state


def _crop_to_reference_window(
    state: ProductionPairState,
    crop: tuple[int, int, int, int],
    secondary: np.ndarray,
    *,
    tag: str,
) -> np.ndarray:
    """Slice a halo-cropped secondary to the Reference ROI crop."""
    origin = state.radar_roi_origin or (0, 0)
    row0 = crop[0] - origin[0]
    col0 = crop[2] - origin[1]
    row1 = row0 + crop[1] - crop[0]
    col1 = col0 + crop[3] - crop[2]
    if row0 < 0 or col0 < 0 or row1 > secondary.shape[0] or col1 > secondary.shape[1]:
        reject_invalid_state(
            f"{tag}: pair ROI crop does not contain the Reference ROI crop"
        )
    return np.ascontiguousarray(secondary[row0:row1, col0:col1])


def _published_unit_mask(
    geometry: np.ndarray,
    sampled: np.ndarray | None,
    payload: np.ndarray,
    *,
    tag: str,
) -> np.ndarray:
    """Combine geometry admissibility, sampled validity, and finite payload."""
    published = np.asarray(geometry, dtype=bool)
    if published.shape != payload.shape:
        reject_invalid_state(
            f"{tag}: geometry support {published.shape} does not match payload "
            f"{payload.shape}"
        )
    if sampled is not None:
        sampled_mask = np.asarray(sampled, dtype=bool)
        if sampled_mask.shape != published.shape:
            reject_invalid_state(
                f"{tag}: sampled validity {sampled_mask.shape} does not match "
                f"payload {payload.shape}"
            )
        published = published & sampled_mask
    return published & np.isfinite(payload.real) & np.isfinite(payload.imag)


def _warn_unsupported_inside_support(
    tag: str,
    geometry: np.ndarray,
    published: np.ndarray,
) -> None:
    """Warn once when requested support contains incomplete interpolation."""
    missing = np.asarray(geometry, dtype=bool) & ~np.asarray(published, dtype=bool)
    count = int(np.count_nonzero(missing))
    if count == 0:
        return
    support_count = int(np.count_nonzero(geometry))
    logger.warning(
        "%s: %d (%.4f of requested support) pixels lack interpolation support; "
        "wrote complex NaN",
        tag,
        count,
        count / max(support_count, 1),
    )


def _window_native_valid(
    valid_mask: np.ndarray | None,
    crop: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Return native validity on ``crop``, or ``None`` when it is unavailable."""
    if valid_mask is None:
        return None
    row0, row1, col0, col1 = crop
    native = np.asarray(valid_mask, dtype=bool)
    if native.ndim != 2:
        return None
    return native[row0:row1, col0:col1]


def _apply_unit_roi_support(
    context: S1ReferenceContext,
    unit: S1ReferenceUnit,
    payload: np.ndarray,
) -> np.ndarray:
    """Apply exact ROI support to a resampled payload before publication."""
    if context.roi is None:
        return payload
    from faninsar.missions.s1.roi_support import apply_scientific_support

    support = unit.scientific_support
    if support is None:
        support = _compute_unit_support(context, unit, payload.shape)
        unit.scientific_support = support
    if support.shape != payload.shape:
        reject_invalid_state(
            f"{unit.tag}: ROI support {support.shape} does not match payload "
            f"{payload.shape}"
        )
    return apply_scientific_support(payload, support)


def _compute_unit_support(
    context: S1ReferenceContext,
    unit: S1ReferenceUnit,
    shape: tuple[int, int],
) -> np.ndarray:
    """Compute exact ROI support for one published unit window."""
    from faninsar.missions.s1.roi_support import (
        geo_scientific_support,
        radar_scientific_support,
    )

    if context.coregistration_grid == "geo":
        if unit.lut is None or context.geo_grid is None:
            reject_invalid_state(
                f"{unit.tag}: geo ROI support requires a Reference LUT"
            )
        return geo_scientific_support(
            context.geo_grid,
            row0=int(unit.lut.row0),
            col0=int(unit.lut.col0),
            shape=shape,
            native_valid=np.asarray(unit.lut.valid, dtype=bool),
            roi=context.roi,
        )
    crop = unit.crop
    if crop is None:
        crop = (0, unit.native_shape[0], 0, unit.native_shape[1])
    return radar_scientific_support(
        unit.scene.geometry,
        context.dem,
        crop=crop,
        native_valid=None,
        roi=context.roi,
        device="auto",
    )


def _stage_scene_unit(
    writer: SceneWriter,
    *,
    tag: str,
    payload: np.ndarray,
    row_origin: int,
    col_origin: int,
    valid_mask: np.ndarray | None,
    burst_id: str,
    scientific_lineage: Sequence[Mapping[str, str]] | None = None,
    phase_state: Mapping[str, object] | None = None,
    absolute_burst_id: int | None = None,
) -> None:
    """Publish one unit with declared ``valid_mask`` and physical ``burst_id``."""
    mask = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    writer.add_unit(
        tag=tag,
        payload=payload,
        row_origin=row_origin,
        col_origin=col_origin,
        scientific_lineage=scientific_lineage,
        phase_state=phase_state,
        valid_mask=mask,
        burst_id=burst_id,
        absolute_burst_id=absolute_burst_id,
    )


def _stage_reference_footprint(
    writer: SceneWriter,
    units: Mapping[str, S1ReferenceUnit],
) -> None:
    """Persist burst footprints as one GeoJSON asset on the Reference scene."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    from faninsar.missions.s1.merge_layout import physical_burst_id
    from faninsar.missions.s1.scene_bursts import official_absolute_burst_id

    rows: list[dict[str, object]] = []
    for tag, unit in units.items():
        ring = getattr(getattr(unit.scene, "burst", None), "footprint", None)
        if not isinstance(ring, (list, tuple)) or len(ring) < 3:
            continue
        geometry = Polygon(ring)
        if geometry.is_empty:
            continue
        rows.append(
            {
                "tag": tag,
                "burst_id": physical_burst_id(unit),
                "absolute_burst_id": official_absolute_burst_id(
                    getattr(unit.scene, "burst", None)
                ),
                "geometry": geometry,
            }
        )
    if not rows:
        return
    writer.add_footprint(gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326"))


def coregister_s1_scene(
    reference_path: Path | Sequence[Path],
    secondary_path: Path | Sequence[Path],
    *,
    output_scene: Path,
    reference_scene: Path,
    swaths: tuple[str, ...] | None = None,
    bursts: BurstSelection | None = None,
    roi: BoundingBox | Polygons | None = None,
    dem: DEM,
    misreg_az_px: float,
    misreg_rg_px: float,
    coregistration_grid: Literal["radar", "geo"] = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_lut_cache_dir: Path | None = None,
    primary_orbit_path: Path | None = None,
    secondary_orbit_path: Path | None = None,
    primary_aux_cal_path: Path | None = None,
    secondary_aux_cal_path: Path | None = None,
    polarizations: Sequence[str] | str | None = None,
    device: str = "auto",
) -> CoregisteredSceneStore:
    """Coregister one secondary acquisition onto a Reference scene product.

    Convenience wrapper: prepares (or reuses) the Reference product at
    ``reference_scene`` and coregisters one secondary. Stacks call
    :func:`prepare_s1_reference` once and :func:`coregister_s1_secondary`
    per date instead.

    Parameters
    ----------
    reference_path, secondary_path : path or sequence of path
        SAFE products per date.
    output_scene, reference_scene : pathlib.Path
        Secondary and Reference scene product directories.
    swaths, bursts, roi, dem, coregistration_grid, geo_grid, geo_lut_cache_dir
        See :func:`prepare_s1_reference`.
    misreg_az_px, misreg_rg_px : float
        Inverted date constants (0 for geometry-only).
    primary_orbit_path, secondary_orbit_path : pathlib.Path, optional
        Precise orbits.
    primary_aux_cal_path, secondary_aux_cal_path : pathlib.Path, optional
        AUX_CAL files; required for IPF 002.36 products.
    polarizations : sequence of str or str, optional
        Single channel forwarded to Reference and secondary opens.
    device : str, optional
        Numerical device.

    Returns
    -------
    CoregisteredSceneStore
        The committed secondary scene product.

    """
    if not np.isfinite(misreg_az_px) or not np.isfinite(misreg_rg_px):
        reject_invalid_state("misreg_az_px and misreg_rg_px must be finite")
    context = prepare_s1_reference(
        reference_path,
        output_scene=reference_scene,
        swaths=swaths,
        bursts=bursts,
        roi=roi,
        dem=dem,
        coregistration_grid=coregistration_grid,
        geo_grid=geo_grid,
        geo_lut_cache_dir=geo_lut_cache_dir,
        orbit_path=primary_orbit_path,
        aux_cal_path=primary_aux_cal_path,
        polarizations=polarizations,
        device=device,
    )
    return coregister_s1_secondary(
        context,
        secondary_path,
        output_scene=output_scene,
        misreg_az_px=misreg_az_px,
        misreg_rg_px=misreg_rg_px,
        secondary_orbit_path=secondary_orbit_path,
        secondary_aux_cal_path=secondary_aux_cal_path,
        polarizations=polarizations,
        device=device,
    )


def frame_geometry_record(
    primary_products: Sequence[S1Product],
    resolved: Mapping[tuple[int, str], Sequence[int]],
    swath_tuple: tuple[str, ...],
    range_offsets: Mapping[str, int],
    grid_shape: tuple[int, int],
    *,
    scene_offset: tuple[int, int] = (0, 0),
    orbit: OrbitMetadata,
) -> dict[str, object]:
    """Return the merged Reference frame radar geometry as the units are placed.

    Follows ISCE2 ``topsStack`` (``mergeBursts``/``VRTManager``) and
    :func:`_unit_placement`: row 0 is the earliest selected burst sensing
    time with the first swath's azimuth interval; column 0 is the first
    swath's starting slant range with its range spacing; later swaths sit at
    integer column offsets. The orbit is trimmed to the frame span.

    Parameters
    ----------
    primary_products : sequence of S1Product
        Reference products, one per frame.
    resolved : mapping
        Selected burst indices per ``(frame, swath)``.
    swath_tuple : tuple of str
        Selected sub-swaths in range order.
    range_offsets : mapping
        Integer column offset of each sub-swath.
    grid_shape : tuple[int, int]
        Merged frame shape.
    scene_offset : tuple[int, int], optional
        Scene-grid ``(row0, col0)`` in absolute placement coordinates when the
        grid is bounded by an ROI crop rather than the burst placement.
    orbit : OrbitMetadata
        Reference orbit (precise when configured).

    Returns
    -------
    dict
        Strict-JSON record for
        :func:`~faninsar.processing.geometry.record.radar_geometry_from_record`.

    """
    swath0 = _first_swath(primary_products, swath_tuple[0])
    sensing_start = min(
        primary_products[frame_index].swath(swath).bursts[burst_index].azimuth_time
        for (frame_index, swath), indices in resolved.items()
        for burst_index in indices
    ) + timedelta(seconds=float(scene_offset[0]) * swath0.azimuth_time_interval_s)
    grid = RadarGrid(
        shape=(int(grid_shape[0]), int(grid_shape[1])),
        starting_slant_range_m=float(
            swath0.slant_range_time_s * SPEED_OF_LIGHT_M_S / 2.0
            + float(scene_offset[1]) * swath0.range_pixel_spacing_m
        ),
        range_spacing_m=float(swath0.range_pixel_spacing_m),
        sensing_start=sensing_start,
        azimuth_time_interval_s=float(swath0.azimuth_time_interval_s),
        wavelength_m=float(SPEED_OF_LIGHT_M_S / swath0.radar_frequency_hz),
        look_direction="right",
    )
    return radar_geometry_record(
        grid,
        orbit,
        swath_column_offsets={name: int(range_offsets[name]) for name in swath_tuple},
    )


def _geo_header(geo_grid: GeoGridSpec) -> dict[str, object]:
    """Return the SceneWriter geo header for ``geo_grid``."""
    return {
        "grid_identity": scene_grid_identity(
            "geo",
            geo_grid.shape,
            {
                "crs": geo_grid.crs,
                "transform": list(geo_grid.transform),
                "resolution_m": list(geo_grid.resolution_m),
            },
        ),
        "crs": geo_grid.crs,
        "transform": tuple(float(value) for value in tuple(geo_grid.transform)[:6]),
    }


def _resolve_apply_selection(
    primary_products: Sequence[S1Product],
    secondary_products: Sequence[S1Product],
    *,
    primary_paths: Sequence[Path],
    swaths: tuple[str, ...] | None,
    bursts: BurstSelection | None,
    roi: BoundingBox | Polygons | None,
    ref_orbits: Sequence[Path | None],
    dem: DEM,
) -> tuple[tuple[str, ...], dict[tuple[int, str], list[int]]]:
    """Resolve swaths and bursts; explicit selection wins over ROI re-select.

    ``roi`` still crops the apply window. NESD overlap is not required to
    fall inside that crop; the burst set is the one already resolved by
    the caller (or ROI footprints when the caller omitted ``bursts``).
    """
    same_acquisition = len(primary_products) == len(secondary_products) and all(
        left is right
        for left, right in zip(primary_products, secondary_products, strict=True)
    )
    if same_acquisition:
        swath_tuple = (
            _union_swath_names(primary_products) if swaths is None else tuple(swaths)
        )
        if bursts is None and swaths is None and roi is not None:
            resolved = _select_bursts_by_roi(
                roi, list(primary_paths), swath_tuple, orbits=ref_orbits, dem=dem
            )
        else:
            counts = {
                (frame, swath): len(found.bursts)
                if (found := product.find_swath(swath))
                else 0
                for frame, product in enumerate(primary_products)
                for swath in swath_tuple
            }
            resolved = _normalize_burst_selection(
                bursts, len(primary_products), swath_tuple, counts
            )
        kept = tuple(
            name
            for name in swath_tuple
            if any(
                indices for (frame, swath), indices in resolved.items() if swath == name
            )
        )
        if not kept:
            reject_invalid_state("Reference selection contains no bursts")
        return kept, resolved
    if bursts is not None or swaths is not None:
        swath_tuple = _resolve_swaths(primary_products, secondary_products, swaths)
        resolved, _ = _resolve_common_bursts(
            primary_products,
            secondary_products,
            swath_tuple,
            bursts,
        )
        return swath_tuple, resolved
    if roi is not None:
        swath_tuple = _union_swath_names(primary_products)
        resolved = _select_bursts_by_roi(
            roi,
            list(primary_paths),
            swath_tuple,
            orbits=ref_orbits,
            dem=dem,
        )
        all_common, _ = _resolve_common_bursts(
            primary_products, secondary_products, swath_tuple, None
        )
        common = {
            key: [index for index in indices if index in all_common.get(key, ())]
            for key, indices in resolved.items()
        }
        common = {key: indices for key, indices in common.items() if indices}
        if not any(common.values()):
            reject_invalid_state("ROI selection contains no common bursts")
        kept = tuple(
            swath
            for swath in swath_tuple
            if any(
                common.get((frame_index, swath))
                for frame_index in range(len(primary_products))
            )
        )
        return kept, common
    swath_tuple = _resolve_swaths(primary_products, secondary_products, swaths)
    resolved, _ = _resolve_common_bursts(
        primary_products,
        secondary_products,
        swath_tuple,
        bursts,
    )
    return swath_tuple, resolved


def _unit_placement(
    primary_products: Sequence[S1Product],
    resolved: dict[tuple[int, str], list[int]],
    swath_tuple: tuple[str, ...],
    range_offsets: dict[str, int],
) -> tuple[dict[str, list[tuple[int, int, int]]], int, int]:
    """Place selected bursts on one shared absolute radar grid."""
    from faninsar.missions.s1.merge_layout import (
        integer_lattice_offset,
        qualify_radar_lattice,
    )

    qualify_radar_lattice(primary_products, resolved, swath_tuple, range_offsets)
    primary_swath0 = _first_swath(primary_products, swath_tuple[0])
    dt = primary_swath0.azimuth_time_interval_s
    burst_lines = {
        swath: _first_swath(primary_products, swath).lines_per_burst
        for swath in swath_tuple
    }
    burst_width = {
        swath: _first_swath(primary_products, swath).samples_per_burst
        for swath in swath_tuple
    }
    azimuth_origin = min(
        primary_products[frame_index].swath(swath).bursts[burst_index].azimuth_time
        for (frame_index, swath), indices in resolved.items()
        for burst_index in indices
    )
    units_by_swath: dict[str, list[tuple[int, int, int]]] = {}
    for swath in swath_tuple:
        units: list[tuple[int, int, int]] = []
        for (frame_index, swath_key), indices in resolved.items():
            if swath_key != swath:
                continue
            swath_obj = primary_products[frame_index].swath(swath)
            for burst_index in indices:
                raw = (
                    swath_obj.bursts[burst_index].azimuth_time - azimuth_origin
                ).total_seconds() / dt
                azimuth_offset = integer_lattice_offset(
                    raw,
                    name=f"f{frame_index}_{swath}_b{burst_index} azimuth origin",
                )
                units.append((frame_index, burst_index, azimuth_offset))
        units.sort(key=lambda unit: unit[2], reverse=True)
        units_by_swath[swath] = units
    if not any(units_by_swath.values()):
        reject_invalid_state("selection contains no bursts")
    frame_rows = max(
        azimuth_offset + burst_lines[swath]
        for swath, units in units_by_swath.items()
        for _, _, azimuth_offset in units
    )
    frame_cols = max(range_offsets[swath] + burst_width[swath] for swath in swath_tuple)
    return units_by_swath, frame_rows, frame_cols
