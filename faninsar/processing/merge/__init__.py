"""Burst merge: per-burst geocode → phase network → weighted mosaic."""

from __future__ import annotations

from .geocode_raster import GeocodedComplex, geocode_complex_to_grid
from .grid import GeoGridSpec, build_geo_grid
from .mosaic import merge_burst_products
from .overlap import (
    apply_feather,
    compute_feather,
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
    "compute_weight_stack",
    "estimate_edge",
    "estimate_edges",
    "geocode_complex_to_grid",
    "merge_burst_products",
    "overlap_mask",
    "run_frame_merge",
    "run_multi_burst_pair_merge",
    "run_multi_path_pair_merge",
    "solve_network",
    "write_mosaic_zarr",
]
