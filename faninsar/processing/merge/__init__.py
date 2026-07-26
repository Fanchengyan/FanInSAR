"""Burst merge: per-burst geocode → phase network → weighted mosaic."""

from __future__ import annotations

from .geocode_raster import GeocodedComplex, geocode_complex_to_grid
from .grid import GeoGridSpec, build_geo_grid
from .methods import MergeMethod, merge_bursts
from .mosaic import merge_burst_products
from .overlap import (
    apply_feather,
    compute_feather,
    compute_hanning_weight,
    compute_weight_stack,
    overlap_mask,
)
from .path_catalog import PathCatalog
from .phase_network import (
    MergeGraph,
    MergeGraphStats,
    NetworkSolution,
    PhaseEdge,
    estimate_edge,
    estimate_edges,
    solve_network,
)
from .pipeline import (
    run_frame_merge,
    run_multi_burst_pair_merge,
    run_multi_path_pair_merge,
    write_mosaic_zarr,
)
from .products import BurstGeoProduct, MosaicProduct
from .unwrap_seam import (
    correct_unwrapped_seam_cycles,
    majority_cycle_align,
    reintegrate_highcoh_from_seed,
    reintegrate_unwrapped_along_range,
)

__all__ = [
    "BurstGeoProduct",
    "GeoGridSpec",
    "GeocodedComplex",
    "MergeGraph",
    "MergeGraphStats",
    "MosaicProduct",
    "NetworkSolution",
    "PathCatalog",
    "PhaseEdge",
    "apply_feather",
    "build_geo_grid",
    "compute_feather",
    "compute_hanning_weight",
    "compute_weight_stack",
    "correct_unwrapped_seam_cycles",
    "estimate_edge",
    "estimate_edges",
    "geocode_complex_to_grid",
    "majority_cycle_align",
    "merge_burst_products",
    "merge_bursts",
    "overlap_mask",
    "reintegrate_highcoh_from_seed",
    "reintegrate_unwrapped_along_range",
    "run_frame_merge",
    "run_multi_burst_pair_merge",
    "run_multi_path_pair_merge",
    "solve_network",
    "write_mosaic_zarr",
]
