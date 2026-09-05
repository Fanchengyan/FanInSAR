"""Persist pair products to Zarr and emit a minimal STAC item."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class PairProductArrays:
    """In-memory pair product layers ready for persistence."""

    pair_id: str
    complex_ifg: np.ndarray
    coherence: np.ndarray
    wrapped_phase: np.ndarray
    unwrapped_phase: np.ndarray
    connected_components: np.ndarray
    metadata: dict[str, Any]


def write_pair_zarr(
    product: PairProductArrays,
    store_path: str | Path,
) -> Path:
    """Write pair product arrays to a Zarr store.

    Parameters
    ----------
    product : PairProductArrays
        In-memory pair layers.
    store_path : str or pathlib.Path
        Destination Zarr directory.

    Returns
    -------
    pathlib.Path
        Path to the written store.

    """
    import zarr

    path = Path(store_path)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    layers = {
        "complex_ifg": product.complex_ifg,
        "coherence": product.coherence,
        "wrapped_phase": product.wrapped_phase,
        "unwrapped_phase": product.unwrapped_phase,
        "connected_components": product.connected_components,
    }
    for name, array in layers.items():
        root.create_array(name, data=np.asarray(array), overwrite=True)
    root.attrs.update(
        {
            "pair_id": product.pair_id,
            "created_utc": datetime.now(UTC).isoformat(),
            **{str(k): str(v) for k, v in product.metadata.items()},
        }
    )
    logger.info("Wrote pair Zarr product: %s", path)
    return path


def write_pair_stac_item(
    product: PairProductArrays,
    zarr_path: str | Path,
    item_path: str | Path,
    stac_item_id: str | None = None,
) -> Path:
    """Write a minimal STAC item describing the pair Zarr product.

    When the pair metadata carries a mask product (``mask_href`` plus the
    ``mask_product``/``mask_provider``/``mask_retrieved``/``mask_threshold``/
    ``mask_buffer_km`` provenance keys, PROPOSAL-0039), the buffered binary
    mask is registered as a ``mask`` STAC asset next to the Zarr store and
    the provenance keys surface as ``faninsar:mask_*`` item properties.

    Parameters
    ----------
    product : PairProductArrays
        Pair product metadata and arrays.
    zarr_path : str or pathlib.Path
        Path to the Zarr store asset.
    item_path : str or pathlib.Path
        Output STAC item JSON path.
    stac_item_id : str, optional
        Look-qualified item id override for sweep outputs.

    Returns
    -------
    pathlib.Path
        Path to the written STAC item.

    """
    zarr_path = Path(zarr_path)
    item_path = Path(item_path)
    item_id = stac_item_id if stac_item_id is not None else product.pair_id
    if not item_id:
        reject_invalid_state("pair_id is required for STAC item identity")
    height, width = product.wrapped_phase.shape
    assets: dict[str, dict[str, Any]] = {
        "zarr": {
            "href": str(zarr_path),
            "type": "application/vnd+zarr",
            "roles": ["data"],
            "title": "Pair product Zarr store",
        }
    }
    mask_href = product.metadata.get("mask_href")
    if mask_href:
        assets["mask"] = {
            "href": str(mask_href),
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "roles": ["mask", "data"],
            "title": "Buffered mask (1 = water/removed)",
        }
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "geometry": None,
        "bbox": None,
        "properties": {
            "datetime": datetime.now(UTC).isoformat(),
            "faninsar:pair_id": product.pair_id,
            "faninsar:shape": [height, width],
            "faninsar:unwrap_method": product.metadata.get("unwrap_method", "snaphu"),
            **{f"faninsar:{k}": v for k, v in product.metadata.items()},
        },
        "assets": assets,
        "links": [],
        "stac_extensions": [],
    }
    item_path.parent.mkdir(parents=True, exist_ok=True)
    item_path.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote pair STAC item: %s", item_path)
    return item_path
