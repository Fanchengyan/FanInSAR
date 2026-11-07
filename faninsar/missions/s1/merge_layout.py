"""Sentinel-1 parent-native midpoint layout for Reference publication (P0074).

Cuts are derived from Reference burst geometry on the qualified radar lattice
before ROI clipping. Same-swath seams are azimuth midpoints; cross-swath seams
are range midpoints. Geo owner labels are the same physical cuts evaluated at
each unit's LUT native coordinates, never a cropped Geo bounding-box midpoint.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from faninsar.io.storage.merge_layout import publish_merge_layout
from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.mosaicking.merge import build_merge_layout
from faninsar.processing.mosaicking.windows import (
    LayoutSourceWindow,
    MergeScope,
    MergeSeam,
    intersect_windows,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from faninsar.data.query import BoundingBox, Polygons
    from faninsar.io.storage.product import Product
    from faninsar.missions.s1.types import S1Product, S1Swath
    from faninsar.processing.mosaicking.windows import MergeLayoutWindow, MergeWindow

logger = setup_logger(__name__)

# Measured Sentinel-1 IW annotation residuals are ~3e-4 to 7.5e-4 px in azimuth
# and ~1e-11 px in range. Residuals at this scale are a common lattice.
LATTICE_ORIGIN_TOLERANCE_PX = 1.0e-3
_SPACING_RELATIVE_TOLERANCE = 1.0e-9


class _ReferenceUnit(Protocol):
    """Fields required to map one Reference burst onto the parent lattice."""

    tag: str
    swath: str
    native_shape: tuple[int, int]
    crop: tuple[int, int, int, int] | None
    row_origin: int
    col_origin: int
    parent_row0: int
    parent_col0: int
    scientific_support: np.ndarray | None
    lut: object | None
    scene: object


def integer_lattice_offset(raw: float, *, name: str) -> int:
    """Return a rounded lattice offset after qualifying the residual.

    Parameters
    ----------
    raw : float
        Offset in pixels on the shared radar lattice.
    name : str
        Identity used in the rejection message.

    Returns
    -------
    int
        Rounded offset admitted by :data:`LATTICE_ORIGIN_TOLERANCE_PX`.

    Raises
    ------
    InvalidProcessingStateError
        If ``raw`` is non-finite or the residual exceeds the annotation-scale
        tolerance (not a common lattice).

    """
    if not np.isfinite(raw):
        reject_invalid_state(f"{name} lattice offset is not finite")
    rounded = round(float(raw))
    residual = float(raw) - float(rounded)
    if abs(residual) > LATTICE_ORIGIN_TOLERANCE_PX:
        reject_invalid_state(
            f"{name} is not on the common radar lattice: offset={raw:.9f} px "
            f"rounded={rounded} residual={residual:.6e} px "
            f"(tolerance {LATTICE_ORIGIN_TOLERANCE_PX} px). "
            "Incompatible placement cannot be silently rounded."
        )
    return rounded


def qualify_radar_lattice(
    primary_products: Sequence[S1Product],
    resolved: Mapping[tuple[int, str], Sequence[int]],
    swath_tuple: tuple[str, ...],
    range_offsets: Mapping[str, int],
) -> None:
    """Reject radar placement that is not a qualified common integer lattice.

    Parameters
    ----------
    primary_products : sequence of S1Product
        Reference SAFE products.
    resolved : mapping
        Selected burst indices per ``(frame, swath)``.
    swath_tuple : tuple of str
        Selected sub-swaths in range order.
    range_offsets : mapping
        Integer column offsets already computed from slant range.

    Raises
    ------
    InvalidProcessingStateError
        If azimuth or range sampling disagrees across swaths, or a burst/swath
        origin residual exceeds :data:`LATTICE_ORIGIN_TOLERANCE_PX`.

    """
    swaths = [_first_swath(primary_products, name) for name in swath_tuple]
    _require_common_spacing(
        [(item.swath, float(item.azimuth_time_interval_s)) for item in swaths],
        name="azimuth time interval",
    )
    _require_common_spacing(
        [(item.swath, float(item.range_sampling_rate_hz)) for item in swaths],
        name="range sampling rate",
    )
    dt = float(swaths[0].azimuth_time_interval_s)
    if dt <= 0.0:
        reject_invalid_state("azimuth time interval must be positive")
    azimuth_origin = min(
        primary_products[frame_index].swath(swath).bursts[burst_index].azimuth_time
        for (frame_index, swath), indices in resolved.items()
        for burst_index in indices
    )
    for (frame_index, swath), indices in resolved.items():
        swath_obj = primary_products[frame_index].swath(swath)
        for burst_index in indices:
            burst = swath_obj.bursts[burst_index]
            raw = (burst.azimuth_time - azimuth_origin).total_seconds() / dt
            integer_lattice_offset(
                raw,
                name=f"f{frame_index}_{swath}_b{burst_index} azimuth origin",
            )
    first = swaths[0]
    rsr = float(first.range_sampling_rate_hz)
    for item in swaths:
        raw = (float(item.slant_range_time_s) - float(first.slant_range_time_s)) * rsr
        qualified = integer_lattice_offset(raw, name=f"swath {item.swath} range origin")
        provided = int(range_offsets[item.swath])
        if qualified != provided:
            reject_invalid_state(
                f"swath {item.swath} range offset {provided} disagrees with "
                f"qualified lattice origin {qualified}"
            )


def parent_midpoint(lo: int, hi: int) -> int:
    """Return the integer midpoint ``lo + (hi - lo) // 2`` of ``[lo, hi)``."""
    if hi <= lo:
        reject_invalid_state("parent overlap interval must be non-empty")
    return int(lo + (hi - lo) // 2)


def polygon_identity(roi: BoundingBox | Polygons | None) -> str:
    """Return a stable non-empty polygon key for :class:`MergeScope`."""
    if roi is None:
        return "full-frame"
    from faninsar.missions.s1.roi_support import roi_lonlat_geometry

    geometry = roi_lonlat_geometry(roi)
    wkt = str(getattr(geometry, "wkt", geometry))
    return hashlib.sha256(wkt.encode("utf-8")).hexdigest()


def track_identity(product: S1Product) -> str:
    """Return a non-empty track key from the Reference product."""
    return f"{product.mission_id}-{int(product.absolute_orbit)}"


def physical_burst_id(unit: _ReferenceUnit) -> str:
    """Return a date-stable physical burst identity, not a local ``b0`` tag.

    Prefers the ESA absolute burst id, then azimuth ANX time, then sensing
    time. The value is a single path basename so it can be stored as
    ``faninsar:burst_id``.
    """
    swath = str(unit.swath)
    burst = getattr(unit.scene, "burst", None)
    absolute = getattr(burst, "absolute_burst_id", None)
    if isinstance(absolute, int):
        return f"{swath}-{absolute}"
    anx = getattr(burst, "azimuth_anx_time_s", None)
    try:
        anx_value = float(anx) if anx is not None else None
    except (TypeError, ValueError):
        anx_value = None
    if anx_value is not None and np.isfinite(anx_value):
        return f"{swath}-anx{round(anx_value * 1_000_000.0)}"
    time = getattr(burst, "azimuth_time", None)
    stamp = getattr(time, "strftime", None)
    if callable(stamp):
        try:
            rendered = stamp("%Y%m%dT%H%M%S")
        except (TypeError, ValueError):
            rendered = None
        if rendered:
            return f"{swath}-t{rendered}"
    message = f"{unit.tag}: burst is missing a physical identity"
    reject_invalid_state(message)
    raise RuntimeError(message)


def build_s1_reference_layout(
    units: Mapping[str, _ReferenceUnit],
    *,
    domain: Literal["radar", "geo"],
    reference_id: str,
    track_id: str,
    channel: str,
    grid_identity: str,
    roi: BoundingBox | Polygons | None,
    scene_offset: tuple[int, int] = (0, 0),
) -> MergeLayoutWindow:
    """Assemble the ROI-local Reference layout from parent-native midpoints.

    Parameters
    ----------
    units : mapping
        Prepared Reference units keyed by tag.
    domain : {"radar", "geo"}
        Publication domain.
    reference_id, track_id, channel, grid_identity : str
        Merge scope identity.
    roi : BoundingBox or Polygons or None
        Scientific ROI used only for source support, not for the cut.
    scene_offset : tuple of int, optional
        Radar scene-grid origin ``(row0, col0)`` in parent coordinates.

    Returns
    -------
    MergeLayoutWindow
        Layout ready to publish before the Reference Item.

    """
    if not units:
        reject_invalid_state("Reference layout requires at least one unit")
    if domain not in {"radar", "geo"}:
        reject_invalid_state("merge layout domain must be 'radar' or 'geo'")
    parent_seams = _parent_seams(units)
    scope = MergeScope(
        domain=domain,
        track_id=str(track_id),
        channel=str(channel),
        reference_id=str(reference_id),
        grid_identity=str(grid_identity),
        polygon_identity=polygon_identity(roi),
    )
    if domain == "radar":
        sources = tuple(_radar_source_window(unit) for unit in units.values())
        seams = tuple(
            _scene_seam(seam, scene_offset=scene_offset, axis=seam.axis)
            for seam in parent_seams
        )
        layout = build_merge_layout(sources, scope=scope, seams=seams)
        logger.info(
            "radar merge layout %s seams=%d sources=%d window=%s",
            reference_id,
            len(layout.seams),
            len(layout.source_ids),
            layout.window,
        )
        return layout
    parent_layout = _parent_radar_layout(units, parent_seams, reference_id=reference_id)
    geo_sources = tuple(_geo_source_window(unit) for unit in units.values())
    owner = _geo_owner_from_parent(units, parent_layout, geo_sources)
    layout = build_merge_layout(geo_sources, scope=scope, owner=owner)
    logger.info(
        "geo merge layout %s sources=%d window=%s",
        reference_id,
        len(layout.source_ids),
        layout.window,
    )
    return layout


def publish_reference_merge_layout(
    output_scene: Path,
    layout: MergeLayoutWindow,
    *,
    overwrite: bool = False,
) -> Product:
    """Publish the layout product beside ``coreg/<date>`` before the Reference Item.

    Parameters
    ----------
    output_scene : pathlib.Path
        Reference scene directory, typically ``<work_dir>/coreg/<date>``.
    layout : MergeLayoutWindow
        Layout to persist.
    overwrite : bool, optional
        Replace an existing layout product.

    Returns
    -------
    Product
        The committed ``merge_layout`` product.

    """
    work_dir = Path(output_scene).parent
    root = work_dir.parent if work_dir.name == "coreg" else work_dir
    target = root / "layout" / layout.scope.reference_id
    return publish_merge_layout(target, layout, overwrite=overwrite)


def _first_swath(primary_products: Sequence[S1Product], name: str) -> S1Swath:
    """Return the first product swath matching ``name``."""
    from faninsar.missions.s1.processing import _first_swath as first_swath

    return first_swath(primary_products, name)


def _require_common_spacing(
    values: Sequence[tuple[str, float]],
    *,
    name: str,
) -> float:
    """Return the shared spacing or reject a mismatch."""
    if not values:
        reject_invalid_state(f"{name} is missing")
    _, reference = values[0]
    if not np.isfinite(reference) or reference <= 0.0:
        reject_invalid_state(f"{name} must be finite and positive")
    for label, value in values[1:]:
        if not np.isfinite(value) or value <= 0.0:
            reject_invalid_state(f"{name} for {label} must be finite and positive")
        scale = max(abs(reference), abs(value), 1.0)
        if abs(value - reference) / scale > _SPACING_RELATIVE_TOLERANCE:
            reject_invalid_state(
                f"{name} mismatch between {values[0][0]} ({reference}) and "
                f"{label} ({value})"
            )
    return float(reference)


def _valid_native_window(unit: _ReferenceUnit) -> tuple[int, int, int, int]:
    """Return the native valid rectangle, or the full burst when mask is absent."""
    height, width = (int(unit.native_shape[0]), int(unit.native_shape[1]))
    burst = getattr(unit.scene, "burst", None)
    first = getattr(burst, "first_valid_sample", None)
    last = getattr(burst, "last_valid_sample", None)
    if first is None or last is None:
        return 0, height, 0, width
    rows = [
        index
        for index, (start, stop) in enumerate(zip(first, last, strict=False))
        if int(stop) >= int(start) >= 0
    ]
    if not rows:
        return 0, height, 0, width
    row0 = int(rows[0])
    row1 = int(rows[-1]) + 1
    col0 = min(
        int(first[index])
        for index in range(row0, min(row1, len(first)))
        if int(last[index]) >= int(first[index])
    )
    col1 = (
        max(
            int(last[index])
            for index in range(row0, min(row1, len(last)))
            if int(last[index]) >= int(first[index])
        )
        + 1
    )
    return row0, row1, col0, col1


def _parent_window(unit: _ReferenceUnit) -> MergeWindow:
    """Return the parent-native valid window of ``unit``."""
    row0, row1, col0, col1 = _valid_native_window(unit)
    return (
        int(unit.parent_row0) + row0,
        int(unit.parent_row0) + row1,
        int(unit.parent_col0) + col0,
        int(unit.parent_col0) + col1,
    )


def _overlap(left: MergeWindow, right: MergeWindow) -> MergeWindow | None:
    """Return the overlapping half-open window, or ``None``."""
    return intersect_windows(left, right)


def _parent_seams(units: Mapping[str, _ReferenceUnit]) -> tuple[MergeSeam, ...]:
    """Return parent-native azimuth and range midpoint seams."""
    ordered = sorted(
        units.values(),
        key=lambda unit: (unit.swath, int(unit.parent_row0), unit.tag),
    )
    seams: list[MergeSeam] = []
    by_swath: dict[str, list[_ReferenceUnit]] = {}
    for unit in ordered:
        by_swath.setdefault(unit.swath, []).append(unit)
    for group in by_swath.values():
        for earlier, later in pairwise(group):
            overlap = _overlap(_parent_window(earlier), _parent_window(later))
            if overlap is None:
                continue
            seams.append(
                MergeSeam(
                    "azimuth",
                    parent_midpoint(overlap[0], overlap[1]),
                    physical_burst_id(earlier),
                    physical_burst_id(later),
                )
            )
    swath_names = list(by_swath)
    for left_name, right_name in pairwise(swath_names):
        for earlier in by_swath[left_name]:
            for later in by_swath[right_name]:
                overlap = _overlap(_parent_window(earlier), _parent_window(later))
                if overlap is None:
                    continue
                seams.append(
                    MergeSeam(
                        "range",
                        parent_midpoint(overlap[2], overlap[3]),
                        physical_burst_id(earlier),
                        physical_burst_id(later),
                    )
                )
    return tuple(seams)


def _unit_support(unit: _ReferenceUnit, shape: tuple[int, int]) -> np.ndarray:
    """Return published scientific support, defaulting to the full window."""
    if unit.scientific_support is None:
        return np.ones(shape, dtype=bool)
    support = np.asarray(unit.scientific_support, dtype=bool)
    if support.shape != shape:
        reject_invalid_state(
            f"{unit.tag}: layout support {support.shape} does not match window {shape}"
        )
    return support


def _radar_source_window(unit: _ReferenceUnit) -> LayoutSourceWindow:
    """Return the ROI-local radar source window on the scene grid."""
    crop = unit.crop
    if crop is None:
        height, width = (int(unit.native_shape[0]), int(unit.native_shape[1]))
    else:
        height = int(crop[1] - crop[0])
        width = int(crop[3] - crop[2])
    window = (
        int(unit.row_origin),
        int(unit.row_origin) + height,
        int(unit.col_origin),
        int(unit.col_origin) + width,
    )
    return LayoutSourceWindow(
        burst_id=physical_burst_id(unit),
        window=window,
        scientific_support=_unit_support(unit, (height, width)),
        swath=unit.swath,
    )


def _geo_source_window(unit: _ReferenceUnit) -> LayoutSourceWindow:
    """Return the Geo LUT window on the destination grid."""
    lut = unit.lut
    if lut is None:
        reject_invalid_state(f"{unit.tag}: geo layout requires a Reference LUT")
    height, width = (int(lut.valid.shape[0]), int(lut.valid.shape[1]))
    window = (
        int(lut.row0),
        int(lut.row0) + height,
        int(lut.col0),
        int(lut.col0) + width,
    )
    support = _unit_support(unit, (height, width)) & np.asarray(lut.valid, dtype=bool)
    return LayoutSourceWindow(
        burst_id=physical_burst_id(unit),
        window=window,
        scientific_support=support,
        swath=unit.swath,
    )


def _scene_seam(
    seam: MergeSeam,
    *,
    scene_offset: tuple[int, int],
    axis: str,
) -> MergeSeam:
    """Translate a parent-native seam into scene-grid coordinates."""
    shift = int(scene_offset[0] if axis == "azimuth" else scene_offset[1])
    return MergeSeam(
        seam.axis, int(seam.boundary) - shift, seam.earlier_id, seam.later_id
    )


def _parent_radar_layout(
    units: Mapping[str, _ReferenceUnit],
    seams: Sequence[MergeSeam],
    *,
    reference_id: str,
) -> MergeLayoutWindow:
    """Build a parent-native radar owner map used to label Geo pixels."""
    sources = []
    for unit in units.values():
        window = _parent_window(unit)
        shape = (window[1] - window[0], window[3] - window[2])
        sources.append(
            LayoutSourceWindow(
                burst_id=physical_burst_id(unit),
                window=window,
                scientific_support=np.ones(shape, dtype=bool),
                swath=unit.swath,
            )
        )
    scope = MergeScope(
        domain="radar",
        track_id="parent",
        channel="vv",
        reference_id=str(reference_id),
        grid_identity="a" * 64,
        polygon_identity="parent",
    )
    return build_merge_layout(sources, scope=scope, seams=seams)


def _geo_owner_from_parent(
    units: Mapping[str, _ReferenceUnit],
    parent_layout: MergeLayoutWindow,
    geo_sources: Sequence[LayoutSourceWindow],
) -> np.ndarray:
    """Paint Geo owner labels from parent-native LUT coordinates."""
    from faninsar.processing.mosaicking.windows import (
        merge_window_shape,
        window_slices,
    )

    layout_window = (
        min(source.window[0] for source in geo_sources),
        max(source.window[1] for source in geo_sources),
        min(source.window[2] for source in geo_sources),
        max(source.window[3] for source in geo_sources),
    )
    shape = merge_window_shape(layout_window)
    owner = np.zeros(shape, dtype=np.uint16)
    parent_window = parent_layout.window
    parent_owner = np.asarray(parent_layout.owner)
    source_index = {
        burst_id: index for index, burst_id in enumerate(parent_layout.source_ids)
    }
    lookup = {physical_burst_id(unit): unit for unit in units.values()}
    for source in geo_sources:
        unit = lookup[source.burst_id]
        lut = unit.lut
        label = np.uint16(source_index[source.burst_id] + 1)
        az = np.asarray(lut.az_full, dtype=np.float64)
        rg = np.asarray(lut.rg_full, dtype=np.float64)
        valid = np.asarray(source.scientific_support, dtype=bool)
        parent_row = np.rint(float(unit.parent_row0) + az).astype(np.int64)
        parent_col = np.rint(float(unit.parent_col0) + rg).astype(np.int64)
        local_row = parent_row - int(parent_window[0])
        local_col = parent_col - int(parent_window[2])
        in_parent = (
            valid
            & (local_row >= 0)
            & (local_row < parent_owner.shape[0])
            & (local_col >= 0)
            & (local_col < parent_owner.shape[1])
        )
        radar_label = np.zeros(az.shape, dtype=np.uint16)
        radar_label[in_parent] = parent_owner[
            local_row[in_parent], local_col[in_parent]
        ]
        mine = in_parent & (radar_label == label)
        dest = window_slices(layout_window, source.window)
        dest_owner = owner[dest]
        dest_owner[mine] = label
        owner[dest] = dest_owner
    return owner
