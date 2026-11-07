"""Midpoint hard-cut burst selection without alignment or blending."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.mosaicking.windows import (
    BurstWindow,
    LayoutSourceWindow,
    MergedWindow,
    MergeLayoutWindow,
    MergeScope,
    MergeSeam,
    MergeWindow,
    intersect_windows,
    merge_window_shape,
    window_slices,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = setup_logger(__name__)

_OPTIONAL_FIELDS = ("coherence", "primary_power", "secondary_power")


def _paste_valid(
    product: BurstWindow,
    layout: MergeLayoutWindow,
    *,
    extra: np.ndarray | None,
    out_complex: np.ndarray,
    out_valid: np.ndarray,
    out_source: np.ndarray,
    out_optional: dict[str, np.ndarray],
    label: int,
) -> None:
    """Copy one burst's valid samples into the layout window."""
    overlap = intersect_windows(product.window, layout.window)
    if overlap is None:
        reject_invalid_state(
            f"burst {product.burst_id!r} window {product.window} is incompatible "
            f"with layout window {layout.window}"
        )
    source = window_slices(product.window, overlap)
    dest = window_slices(layout.window, overlap)
    valid = np.asarray(product.valid_mask[source], dtype=bool)
    support = np.asarray(layout.scientific_support[dest], dtype=bool)
    mask = valid & support
    if extra is not None:
        mask = mask & np.asarray(extra[dest], dtype=bool)
    if not np.any(mask):
        return
    dest_complex = out_complex[dest]
    dest_complex[mask] = product.complex[source][mask]
    out_complex[dest] = dest_complex
    dest_valid = out_valid[dest]
    dest_valid[mask] = True
    out_valid[dest] = dest_valid
    dest_source = out_source[dest]
    dest_source[mask] = np.uint16(label)
    out_source[dest] = dest_source
    for field_name, destination in out_optional.items():
        values = getattr(product, field_name)
        dest_values = destination[dest]
        dest_values[mask] = values[source][mask]
        destination[dest] = dest_values


def merge_bursts(
    products: Sequence[BurstWindow],
    *,
    layout: MergeLayoutWindow,
) -> MergedWindow:
    """Select native samples by preferred midpoint owner, then VALID fallback.

    Parameters
    ----------
    products : sequence of BurstWindow
        Window-local burst samples. Arrays must cover each product window
        rather than a full parent canvas. Input order is ignored.
    layout : MergeLayoutWindow
        ROI-local preferred-owner map. Label ``0`` is unowned.

    Returns
    -------
    MergedWindow
        Selected complex samples, explicit support, and the actual source
        labels after fallback. Optional coherence and endpoint-power fields
        are copied from the same candidate when every product supplies them.

    Raises
    ------
    InvalidProcessingStateError
        If products are empty, duplicated, already merged, unwrapped, or
        disagree on scope, acquisitions, phase convention, or optional
        fields, or if a window does not intersect the layout.

    """
    if not products:
        reject_invalid_state("merge_bursts requires at least one BurstWindow")
    if not isinstance(layout, MergeLayoutWindow):
        reject_invalid_state("merge_bursts requires a MergeLayoutWindow")
    burst_ids = [product.burst_id for product in products]
    if len(set(burst_ids)) != len(burst_ids):
        reject_invalid_state("merge_bursts burst_id values must be unique")
    source_index = {burst_id: index for index, burst_id in enumerate(layout.source_ids)}
    acquisitions = products[0].acquisitions
    convention = products[0].phase_convention
    optional_present = {
        field_name: getattr(products[0], field_name) is not None
        for field_name in _OPTIONAL_FIELDS
    }
    by_id: dict[str, BurstWindow] = {}
    for product in products:
        if not isinstance(product, BurstWindow):
            reject_invalid_state("merge_bursts inputs must be BurstWindow values")
        if product.scope != layout.scope:
            reject_invalid_state("burst window scope does not match layout scope")
        if product.representation != "burst":
            reject_invalid_state("merge_bursts rejects merged representation inputs")
        if product.acquisitions != acquisitions:
            reject_invalid_state("burst window acquisitions must match within a merge")
        if product.phase_convention != convention:
            reject_invalid_state(
                "burst window phase_convention must match within a merge"
            )
        if product.burst_id not in source_index:
            reject_invalid_state(
                f"burst {product.burst_id!r} is not in layout source_ids"
            )
        for field_name, expected in optional_present.items():
            present = getattr(product, field_name) is not None
            if present != expected:
                reject_invalid_state(
                    "optional burst fields must be supplied by every product or none"
                )
        by_id[product.burst_id] = product

    shape = merge_window_shape(layout.window)
    out_complex = np.zeros(shape, dtype=np.complex64)
    out_valid = np.zeros(shape, dtype=bool)
    out_source = np.zeros(shape, dtype=np.uint16)
    out_optional: dict[str, np.ndarray] = {}
    if optional_present["coherence"]:
        out_optional["coherence"] = np.full(shape, np.nan, dtype=np.float32)
    if optional_present["primary_power"]:
        out_optional["primary_power"] = np.zeros(shape, dtype=np.float32)
    if optional_present["secondary_power"]:
        out_optional["secondary_power"] = np.zeros(shape, dtype=np.float32)

    for index in range(len(layout.source_ids) - 1, -1, -1):
        product = by_id.get(layout.source_ids[index])
        if product is None:
            continue
        _paste_valid(
            product,
            layout,
            extra=None,
            out_complex=out_complex,
            out_valid=out_valid,
            out_source=out_source,
            out_optional=out_optional,
            label=index + 1,
        )
    for index, burst_id in enumerate(layout.source_ids):
        product = by_id.get(burst_id)
        if product is None:
            continue
        preferred = layout.owner == np.uint16(index + 1)
        _paste_valid(
            product,
            layout,
            extra=preferred,
            out_complex=out_complex,
            out_valid=out_valid,
            out_source=out_source,
            out_optional=out_optional,
            label=index + 1,
        )
    out_valid &= layout.scientific_support
    out_source = np.where(out_valid, out_source, np.uint16(0))
    out_complex = np.where(out_valid, out_complex, np.complex64(0 + 0j))
    for field_name, values in out_optional.items():
        if field_name == "coherence":
            out_optional[field_name] = np.where(out_valid, values, np.float32(np.nan))
        else:
            out_optional[field_name] = np.where(out_valid, values, np.float32(0.0))
    logger.info(
        "merged %d burst windows on %s with %d valid samples",
        len(products),
        layout.window,
        int(out_valid.sum()),
    )
    return MergedWindow(
        scope=layout.scope,
        window=layout.window,
        complex=out_complex.astype(np.complex64, copy=False),
        valid_mask=out_valid,
        source=out_source.astype(np.uint16, copy=False),
        coherence=out_optional.get("coherence"),
        primary_power=out_optional.get("primary_power"),
        secondary_power=out_optional.get("secondary_power"),
    )


def _union_window(windows: Sequence[MergeWindow]) -> MergeWindow:
    """Return the bounding half-open window of ``windows``."""
    row0 = min(window[0] for window in windows)
    row1 = max(window[1] for window in windows)
    col0 = min(window[2] for window in windows)
    col1 = max(window[3] for window in windows)
    return row0, row1, col0, col1


def _source_support_on_layout(
    source: LayoutSourceWindow, layout_window: MergeWindow, shape: tuple[int, int]
) -> np.ndarray:
    """Paint one source's scientific support onto the layout window."""
    painted = np.zeros(shape, dtype=bool)
    overlap = intersect_windows(source.window, layout_window)
    if overlap is None:
        return painted
    source_slice = window_slices(source.window, overlap)
    dest = window_slices(layout_window, overlap)
    painted[dest] = np.asarray(source.scientific_support[source_slice], dtype=bool)
    return painted


def _apply_precomputed_seams(
    owner: np.ndarray,
    supports: dict[str, np.ndarray],
    source_index: dict[str, int],
    seams: Sequence[MergeSeam],
    layout_window: MergeWindow,
) -> np.ndarray:
    """Resolve overlaps using caller-supplied seams, never inferred Geo cuts."""
    rows = np.arange(layout_window[0], layout_window[1])[:, None]
    cols = np.arange(layout_window[2], layout_window[3])[None, :]
    result = owner.copy()
    for seam in seams:
        if seam.earlier_id not in source_index or seam.later_id not in source_index:
            reject_invalid_state("merge seam refers to a source_id that is not present")
        both = supports[seam.earlier_id] & supports[seam.later_id]
        if seam.axis == "azimuth":
            earlier_region = both & (rows < seam.boundary)
            later_region = both & (rows >= seam.boundary)
        else:
            earlier_region = both & (cols < seam.boundary)
            later_region = both & (cols >= seam.boundary)
        result = np.where(
            earlier_region, np.uint16(source_index[seam.earlier_id] + 1), result
        )
        result = np.where(
            later_region, np.uint16(source_index[seam.later_id] + 1), result
        )
    return result


def build_merge_layout(
    sources: Sequence[LayoutSourceWindow],
    *,
    scope: MergeScope,
    seams: Sequence[MergeSeam] = (),
    owner: np.ndarray | None = None,
    window: MergeWindow | None = None,
) -> MergeLayoutWindow:
    """Assemble an ROI-local owner map from source support and supplied cuts.

    This helper does **not** invent physical Geo midpoints from ROI-cropped
    bounding boxes. Overlapping coverage stays unowned unless the caller
    supplies ``seams`` or a finished ``owner`` raster. Mission-owned native
    geometry mapping belongs in ``faninsar.missions.s1.merge_layout``.

    Parameters
    ----------
    sources : sequence of LayoutSourceWindow
        ROI-local support windows. ``source_ids`` are ordered by swath,
        window origin, then burst identity, not caller iteration order.
    scope : MergeScope
        Shared merge identity.
    seams : sequence of MergeSeam, optional
        Precomputed midpoint cuts. Applied only where both sources have
        scientific support.
    owner : numpy.ndarray, optional
        Finished uint labels on ``window``. When supplied, seams are stored
        for persistence but do not rewrite the raster.
    window : tuple of int, optional
        Layout window. Default is the bounding window of ``sources``.

    Returns
    -------
    MergeLayoutWindow
        ROI-local owner and union scientific support.

    Raises
    ------
    InvalidProcessingStateError
        If sources are empty or duplicated, or supplied owner/window disagree.

    """
    if not sources:
        reject_invalid_state("build_merge_layout requires at least one source window")
    if not isinstance(scope, MergeScope):
        reject_invalid_state("build_merge_layout requires a MergeScope")
    burst_ids = [source.burst_id for source in sources]
    if len(set(burst_ids)) != len(burst_ids):
        reject_invalid_state("layout source burst_id values must be unique")
    ordered = tuple(
        sorted(sources, key=lambda item: (item.swath, item.window, item.burst_id))
    )
    source_ids = tuple(item.burst_id for item in ordered)
    layout_window = (
        window
        if window is not None
        else _union_window([item.window for item in ordered])
    )
    shape = merge_window_shape(layout_window)
    supports = {
        item.burst_id: _source_support_on_layout(item, layout_window, shape)
        for item in ordered
    }
    scientific_support = np.zeros(shape, dtype=bool)
    for support in supports.values():
        scientific_support |= support
    coverage = np.zeros(shape, dtype=np.uint16)
    unique = np.zeros(shape, dtype=np.uint16)
    for index, item in enumerate(ordered):
        label = np.uint16(index + 1)
        present = supports[item.burst_id]
        coverage = coverage + present.astype(np.uint16)
        unique = np.where(present, label, unique)
    unique = np.where(coverage == 1, unique, np.uint16(0))
    if owner is not None:
        labels = np.asarray(owner)
        if labels.shape != shape:
            reject_invalid_state("supplied layout owner must match the layout window")
        assembled = np.where(
            scientific_support, labels.astype(np.uint16, copy=False), np.uint16(0)
        )
    else:
        assembled = unique
        if seams:
            assembled = _apply_precomputed_seams(
                assembled,
                supports,
                {burst_id: index for index, burst_id in enumerate(source_ids)},
                tuple(seams),
                layout_window,
            )
        assembled = np.where(scientific_support, assembled, np.uint16(0))
    return MergeLayoutWindow(
        scope=scope,
        window=layout_window,
        source_ids=source_ids,
        owner=assembled.astype(np.uint16, copy=False),
        scientific_support=scientific_support,
        seams=tuple(seams),
        source_windows=ordered,
    )
