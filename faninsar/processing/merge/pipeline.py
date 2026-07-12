"""Pipeline wiring for multi-burst and multi-path merge production."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.merge.mosaic import merge_burst_products

if TYPE_CHECKING:
    from faninsar.processing.merge.products import BurstGeoProduct, MosaicProduct

logger = setup_logger(__name__)

__all__ = [
    "run_frame_merge",
    "run_multi_burst_pair_merge",
    "run_multi_path_pair_merge",
    "write_mosaic_zarr",
]


def run_multi_burst_pair_merge(
    products: list[BurstGeoProduct],
    *,
    mode: Literal["complex_average", "phase_network"] = "phase_network",
    min_overlap_px: int = 500,
    min_edge_coherence: float = 0.15,
    path_policy: Literal["same_path_only", "allow_cross_path"] = "allow_cross_path",
    allow_unwrapped_merge: bool = False,
    allow_asc_desc_phase_link: bool = False,
) -> MosaicProduct:
    """Merge multiple bursts of the same pair (M2 mode) in memory.

    This is a thin wrapper over :func:`merge_burst_products` for the common
    case where the input products are already geocoded burst interferograms
    of a single pair.

    Parameters
    ----------
    products : list of BurstGeoProduct
        Geocoded burst products of one pair.
    mode : {"complex_average", "phase_network"}, optional
        Merge mode forwarded to :func:`merge_burst_products`.
    min_overlap_px : int, optional
        Minimum overlap pixels to form a phase edge.
    min_edge_coherence : float, optional
        Minimum edge coherence.
    path_policy : {"same_path_only", "allow_cross_path"}, optional
        Restrict phase edges to the same path or allow cross-path links.
    allow_unwrapped_merge : bool, optional
        Reject unwrapped inputs by default.
    allow_asc_desc_phase_link : bool, optional
        When ``False``, ascending/descending bursts share the grid but are
        not phase-linked.

    Returns
    -------
    MosaicProduct

    """
    logger.info(
        "run_multi_burst_pair_merge: %d bursts, mode=%s", len(products), mode
    )
    return merge_burst_products(
        products,
        mode=mode,
        min_overlap_px=min_overlap_px,
        min_edge_coherence=min_edge_coherence,
        path_policy=path_policy,
        allow_unwrapped_merge=allow_unwrapped_merge,
        allow_asc_desc_phase_link=allow_asc_desc_phase_link,
    )


def run_multi_path_pair_merge(
    products: list[BurstGeoProduct],
    *,
    min_overlap_px: int = 2000,
    min_edge_coherence: float = 0.25,
    allow_asc_desc_phase_link: bool = False,
    allow_unwrapped_merge: bool = False,
) -> MosaicProduct:
    """Merge bursts across multiple paths (same look direction by default).

    Parameters
    ----------
    products : list of BurstGeoProduct
        Geocoded burst products spanning multiple paths.
    min_overlap_px : int, optional
        Stricter overlap threshold for cross-path edges.
    min_edge_coherence : float, optional
        Stricter coherence threshold for cross-path edges.
    allow_asc_desc_phase_link : bool, optional
        When ``False`` (default), ascending and descending bursts share the
        grid but are not phase-linked.
    allow_unwrapped_merge : bool, optional
        Reject unwrapped inputs by default.

    Returns
    -------
    MosaicProduct

    """
    logger.info(
        "run_multi_path_pair_merge: %d bursts, asc/desc link=%s",
        len(products),
        allow_asc_desc_phase_link,
    )
    return merge_burst_products(
        products,
        mode="phase_network",
        min_overlap_px=min_overlap_px,
        min_edge_coherence=min_edge_coherence,
        path_policy="allow_cross_path",
        allow_unwrapped_merge=allow_unwrapped_merge,
        allow_asc_desc_phase_link=allow_asc_desc_phase_link,
    )


def run_frame_merge(
    products: list[BurstGeoProduct],
    *,
    output_dir: Path,
    merge_paths: bool = True,
    pair_id: str = "frame",
    min_overlap_px: int = 500,
    min_edge_coherence: float = 0.15,
) -> Path:
    """Orchestrate a full frame merge: per-path mosaic then optional cross-path.

    Parameters
    ----------
    products : list of BurstGeoProduct
        All geocoded burst products in the frame.
    output_dir : pathlib.Path
        Destination directory for Zarr products.
    merge_paths : bool, optional
        When ``True``, also attempt cross-path merge.
    pair_id : str, optional
        Identifier used in the output Zarr store name.
    min_overlap_px, min_edge_coherence
        Network thresholds.

    Returns
    -------
    pathlib.Path
        Path to the written Zarr store.

    """
    from faninsar.processing.merge.path_catalog import PathCatalog

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = PathCatalog.from_products(products)

    if merge_paths and len(catalog.paths) > 1:
        mosaic = run_multi_path_pair_merge(
            products,
            min_overlap_px=max(min_overlap_px, 2000),
            min_edge_coherence=max(min_edge_coherence, 0.25),
        )
    else:
        mosaic = run_multi_burst_pair_merge(
            products,
            min_overlap_px=min_overlap_px,
            min_edge_coherence=min_edge_coherence,
        )

    out_path = output_dir / f"{pair_id}_mosaic.zarr"
    return write_mosaic_zarr(mosaic, out_path, pair_id=pair_id)


def write_mosaic_zarr(
    mosaic: MosaicProduct,
    store_path: str | Path,
    *,
    pair_id: str,
) -> Path:
    """Write a :class:`MosaicProduct` to a Zarr store (plan §10 schema).

    Parameters
    ----------
    mosaic : MosaicProduct
        Mosaic to persist.
    store_path : str or pathlib.Path
        Destination Zarr directory.
    pair_id : str
        Pair identifier written into store attributes.

    Returns
    -------
    pathlib.Path
        Path to the written store.

    """
    import shutil

    import zarr

    path = Path(store_path)
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")

    root.create_array(
        "complex_ifg", data=np.asarray(mosaic.complex), overwrite=True
    )
    if mosaic.coherence is not None:
        root.create_array(
            "coherence", data=np.asarray(mosaic.coherence), overwrite=True
        )
    root.create_array(
        "weight_sum", data=np.asarray(mosaic.weight_sum), overwrite=True
    )
    root.create_array(
        "n_bursts", data=np.asarray(mosaic.n_bursts), overwrite=True
    )
    root.create_array(
        "component_id", data=np.asarray(mosaic.component_id), overwrite=True
    )

    grid_group = root.create_group("grid")
    grid_group.attrs.update(
        {
            "crs": mosaic.grid.crs,
            "transform": list(mosaic.grid.transform),
            "width": mosaic.grid.width,
            "height": mosaic.grid.height,
            "resolution_m": list(mosaic.grid.resolution_m),
            "bbox": list(mosaic.grid.bbox),
        }
    )

    if mosaic.network is not None:
        net_group = root.create_group("network")
        net_group.attrs.update(
            {
                "n_nodes": mosaic.network.n_nodes,
                "n_edges": mosaic.network.n_edges,
                "n_components": mosaic.network.n_components,
                "rms_residual": mosaic.network.rms_residual,
            }
        )

    root.attrs.update(
        {
            "pair_id": pair_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "merge_domain": "complex",
        }
    )
    logger.info("Wrote mosaic Zarr product: %s", path)
    return path
