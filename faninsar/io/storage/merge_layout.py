"""Persist ROI-local merge layouts through the existing product protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.io.storage._zarr_stac import write_zarr_array
from faninsar.io.storage.product import Product, open_product, publish_product
from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.mosaicking.windows import (
    LayoutSourceWindow,
    MergeLayoutWindow,
    MergeScope,
    MergeSeam,
    MergeWindow,
    validate_merge_window,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = setup_logger(__name__)

LAYOUT_KIND = "merge_layout"


def _window_list(window: MergeWindow) -> list[int]:
    return [int(value) for value in window]


def _read_window(value: object, *, name: str) -> MergeWindow:
    if not isinstance(value, list) or len(value) != 4:
        reject_invalid_state(f"merge layout {name} must be a four-integer window")
    try:
        window = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        reject_invalid_state(f"merge layout {name} must be a four-integer window")
    return validate_merge_window(window, name=name)


def _compact_radar(layout: MergeLayoutWindow) -> bool:
    """Return whether seams plus per-source support reconstruct the owner."""
    if layout.scope.domain != "radar" or not layout.seams:
        return False
    if len(layout.source_windows) != len(layout.source_ids):
        return False
    return tuple(item.burst_id for item in layout.source_windows) == layout.source_ids


def _source_records(layout: MergeLayoutWindow) -> list[dict[str, object]]:
    records = []
    lookup = {item.burst_id: item for item in layout.source_windows}
    for burst_id in layout.source_ids:
        source = lookup.get(burst_id)
        records.append(
            {
                "burst_id": burst_id,
                "window": _window_list(source.window) if source is not None else None,
                "swath": source.swath if source is not None else None,
            }
        )
    return records


def _seam_records(layout: MergeLayoutWindow) -> list[dict[str, object]]:
    return [
        {
            "axis": seam.axis,
            "boundary": int(seam.boundary),
            "earlier_id": seam.earlier_id,
            "later_id": seam.later_id,
        }
        for seam in layout.seams
    ]


def publish_merge_layout(
    target: str | Path,
    layout: MergeLayoutWindow,
    *,
    inputs: Mapping[str, Product | str] | None = None,
    overwrite: bool = False,
) -> Product:
    """Publish one merge layout as a ``merge_layout`` product.

    Radar layouts with precomputed seams and per-source support omit
    ``owner.zarr`` and reconstruct labels on open. Geo layouts, and Radar
    layouts that cannot reconstruct irregular holes from compact metadata,
    persist ``owner.zarr``.

    Parameters
    ----------
    target : path-like
        Product directory, typically ``{work_dir}/layout/{reference_id}``.
    layout : MergeLayoutWindow
        ROI-local owner map to persist.
    inputs : mapping, optional
        Already-committed upstream write ids (grid/source). Do not bind the
        enclosing Reference write id before that product exists.
    overwrite : bool, optional
        Replace an existing layout product.

    Returns
    -------
    Product
        The published layout product.

    """
    if not isinstance(layout, MergeLayoutWindow):
        reject_invalid_state("publish_merge_layout requires a MergeLayoutWindow")
    compact = _compact_radar(layout)
    properties = {
        "faninsar:domain": layout.scope.domain,
        "faninsar:track_id": layout.scope.track_id,
        "faninsar:channel": layout.scope.channel,
        "faninsar:reference_id": layout.scope.reference_id,
        "faninsar:grid_identity": layout.scope.grid_identity,
        "faninsar:polygon_identity": layout.scope.polygon_identity,
        "faninsar:window": _window_list(layout.window),
        "faninsar:source_ids": list(layout.source_ids),
        "faninsar:source_windows": _source_records(layout),
        "faninsar:seams": _seam_records(layout),
        "faninsar:compact": compact,
    }

    def write_payloads(staging: Path) -> dict[str, dict[str, object]]:
        assets: dict[str, dict[str, object]] = {}
        write_zarr_array(staging / "scientific_support.zarr", layout.scientific_support)
        assets["scientific_support"] = {
            "href": "scientific_support.zarr",
            "roles": ["data"],
        }
        if compact:
            for index, source in enumerate(layout.source_windows):
                href = f"support-{index}.zarr"
                write_zarr_array(staging / href, source.scientific_support)
                assets[f"support-{index}"] = {"href": href, "roles": ["data"]}
        else:
            write_zarr_array(staging / "owner.zarr", layout.owner)
            assets["owner"] = {"href": "owner.zarr", "roles": ["data"]}
        return assets

    product = publish_product(
        target,
        kind=LAYOUT_KIND,
        properties=properties,
        write_payloads=write_payloads,
        inputs=inputs,
        overwrite=overwrite,
    )
    logger.info("published merge layout %s compact=%s", product.path, compact)
    return product


def _read_seams(value: object) -> tuple[MergeSeam, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        reject_invalid_state("merge layout seams must be a list")
    seams = []
    for item in value:
        if not isinstance(item, dict):
            reject_invalid_state("merge layout seam entries must be objects")
        axis = item.get("axis")
        if axis not in ("azimuth", "range"):
            reject_invalid_state("merge layout seam axis must be azimuth or range")
        seams.append(
            MergeSeam(
                axis=axis,
                boundary=int(item["boundary"]),
                earlier_id=str(item["earlier_id"]),
                later_id=str(item["later_id"]),
            )
        )
    return tuple(seams)


def _read_scope(properties: Mapping[str, object]) -> MergeScope:
    domain = properties.get("faninsar:domain")
    if domain not in ("radar", "geo"):
        reject_invalid_state("merge layout domain must be radar or geo")
    return MergeScope(
        domain=domain,
        track_id=str(properties["faninsar:track_id"]),
        channel=str(properties["faninsar:channel"]),
        reference_id=str(properties["faninsar:reference_id"]),
        grid_identity=str(properties["faninsar:grid_identity"]),
        polygon_identity=str(properties["faninsar:polygon_identity"]),
    )


def open_merge_layout(path: str | Path | Product) -> MergeLayoutWindow:
    """Reopen a published ``merge_layout`` product as a typed window.

    Parameters
    ----------
    path : path-like or Product
        Layout product directory or an already opened product.

    Returns
    -------
    MergeLayoutWindow
        Reconstructed ROI-local owner map.

    Raises
    ------
    InvalidProcessingStateError
        If the product is not a merge layout or required assets are missing.

    """
    product = (
        path if isinstance(path, Product) else open_product(path, kind=LAYOUT_KIND)
    )
    if product.kind != LAYOUT_KIND:
        reject_invalid_state(
            f"product {product.path} has kind {product.kind!r}, "
            f"expected {LAYOUT_KIND!r}"
        )
    properties = product.properties
    scope = _read_scope(properties)
    window = _read_window(properties.get("faninsar:window"), name="window")
    source_ids = properties.get("faninsar:source_ids")
    if not isinstance(source_ids, list) or not source_ids:
        reject_invalid_state("merge layout source_ids are missing")
    ids = tuple(str(item) for item in source_ids)
    seams = _read_seams(properties.get("faninsar:seams"))
    records = properties.get("faninsar:source_windows") or []
    source_windows: list[LayoutSourceWindow] = []
    if "owner" in product.assets:
        owner = np.asarray(product.read("owner"), dtype=np.uint16)
        support = np.asarray(product.read("scientific_support"), dtype=bool)
        if isinstance(records, list):
            for index, record in enumerate(records):
                if not isinstance(record, dict) or record.get("window") is None:
                    continue
                burst_id = str(record["burst_id"])
                source_window = _read_window(record["window"], name="source window")
                swath = str(record.get("swath") or "unknown")
                name = f"support-{index}"
                if name in product.assets:
                    source_support = np.asarray(product.read(name), dtype=bool)
                else:
                    source_support = np.ones(
                        (
                            source_window[1] - source_window[0],
                            source_window[3] - source_window[2],
                        ),
                        dtype=bool,
                    )
                source_windows.append(
                    LayoutSourceWindow(burst_id, source_window, source_support, swath)
                )
        return MergeLayoutWindow(
            scope=scope,
            window=window,
            source_ids=ids,
            owner=owner,
            scientific_support=support,
            seams=seams,
            source_windows=tuple(source_windows),
        )
    if scope.domain != "radar" or not seams:
        reject_invalid_state(
            "merge layout is missing owner.zarr and cannot reconstruct "
            "compact Radar seams"
        )
    if not isinstance(records, list) or len(records) != len(ids):
        reject_invalid_state("compact Radar layout requires per-source window metadata")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            reject_invalid_state("compact Radar source window metadata is invalid")
        name = f"support-{index}"
        if name not in product.assets:
            reject_invalid_state(
                "compact Radar layout requires per-source scientific support assets"
            )
        source_window = _read_window(record.get("window"), name="source window")
        source_windows.append(
            LayoutSourceWindow(
                burst_id=str(record.get("burst_id") or ids[index]),
                window=source_window,
                scientific_support=np.asarray(product.read(name), dtype=bool),
                swath=str(record.get("swath") or "unknown"),
            )
        )
    from faninsar.processing.mosaicking.merge import build_merge_layout

    rebuilt = build_merge_layout(
        source_windows, scope=scope, seams=seams, window=window
    )
    if rebuilt.source_ids != ids:
        reject_invalid_state(
            "compact Radar source_ids do not match reconstructed order"
        )
    return rebuilt
