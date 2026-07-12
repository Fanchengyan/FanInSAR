"""Weighted mosaic of geocoded burst products."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.merge.phase_network import (
    estimate_edges,
    solve_network,
)

if TYPE_CHECKING:
    from faninsar.processing.merge.products import BurstGeoProduct, MosaicProduct

logger = setup_logger(__name__)

__all__ = ["merge_burst_products"]


def merge_burst_products(
    products: list[BurstGeoProduct],
    *,
    mode: Literal["complex_average", "phase_network"] = "phase_network",
    min_overlap_px: int = 500,
    min_edge_coherence: float = 0.15,
    path_policy: Literal["same_path_only", "allow_cross_path"] = "allow_cross_path",
    allow_unwrapped_merge: bool = False,
    allow_asc_desc_phase_link: bool = False,
    reference_node: int = 0,
) -> MosaicProduct:
    """Merge geocoded burst products into a single weighted mosaic.

    Parameters
    ----------
    products : list of BurstGeoProduct
        Geocoded burst products on a common grid. All must share the same
        :class:`GeoGridSpec`.
    mode : {"complex_average", "phase_network"}, optional
        ``"complex_average"`` averages bursts without phase alignment;
        ``"phase_network"`` estimates overlap Δφ, solves a weighted LS
        adjustment, and aligns each burst before averaging.
    min_overlap_px : int, optional
        Minimum overlap pixels to form a phase edge.
    min_edge_coherence : float, optional
        Minimum edge coherence.
    path_policy : {"same_path_only", "allow_cross_path"}, optional
        Restrict phase edges to the same path or allow cross-path links.
    allow_unwrapped_merge : bool, optional
        When ``False`` (default), any product with
        ``phase_domain == "unwrapped"`` raises ``ValueError``. This
        enforces the timing rule: merge must occur in the complex domain,
        before unwrapping (see plan §4.3).
    allow_asc_desc_phase_link : bool, optional
        When ``False`` (default), ascending/descending bursts share the
        grid but are not phase-linked.
    reference_node : int, optional
        Node index pinned to phase 0 in its connected component.

    Returns
    -------
    MosaicProduct
        Weighted complex mosaic, coherence, weight_sum, n_bursts, and
        connected-component labels.

    Raises
    ------
    ValueError
        If ``products`` is empty, grids disagree, or an unwrapped product
        is supplied while ``allow_unwrapped_merge=False``.

    """
    if not products:
        msg = "merge_burst_products requires at least one product"
        raise ValueError(msg)

    if not allow_unwrapped_merge:
        unwrapped = [
            p.burst_id for p in products if p.phase_domain == "unwrapped"
        ]
        if unwrapped:
            msg = (
                "merge_burst_products received unwrapped-phase products "
                f"({unwrapped}). The timing rule (plan §4.3) requires merge "
                "in the complex domain before unwrapping. Pass "
                "allow_unwrapped_merge=True to opt in (not recommended; "
                "not part of the production DoD)."
            )
            raise ValueError(msg)

    grid = products[0].grid
    for p in products[1:]:
        if p.grid.shape != grid.shape or p.grid.crs != grid.crs:
            msg = "all products must share the same GeoGridSpec"
            raise ValueError(msg)

    shape = grid.shape
    n = len(products)

    # Phase alignment
    if mode == "phase_network":
        graph = estimate_edges(
            products,
            min_overlap_px=min_overlap_px,
            min_edge_coherence=min_edge_coherence,
            path_policy=path_policy,
            allow_asc_desc_phase_link=allow_asc_desc_phase_link,
        )
        # Failure-mode warnings (plan §12).
        if len(graph.edges) == 0 and n > 1:
            logger.warning(
                "merge_burst_products: no phase edges formed — "
                "all bursts are isolated (overlap < %d px or coherence < %.2f). "
                "Each burst becomes its own component; phases are not aligned.",
                min_overlap_px,
                min_edge_coherence,
            )
        solution = solve_network(graph, reference_node=reference_node)
        phi_hat = solution.phi_hat
        component_per_node = solution.component_id
        network_stats = solution.stats
        if network_stats.n_components > 1:
            logger.warning(
                "merge_burst_products: %d disconnected components — "
                "phases are aligned only within each component.",
                network_stats.n_components,
            )
        if network_stats.rms_residual > 0.1:
            logger.warning(
                "merge_burst_products: high network residual RMS=%.3e rad — "
                "overlap phase estimates may be inconsistent.",
                network_stats.rms_residual,
            )
    else:
        phi_hat = np.zeros(n, dtype=np.float64)
        component_per_node = np.zeros(n, dtype=np.int16)
        network_stats = None

    # Build per-pixel component id (max weight wins)
    complex_acc = np.zeros(shape, dtype=np.complex64)
    weight_sum = np.zeros(shape, dtype=np.float32)
    coh_acc = np.zeros(shape, dtype=np.float32)
    n_bursts = np.zeros(shape, dtype=np.uint8)
    component_id = np.zeros(shape, dtype=np.int16)
    max_weight_per_pixel = np.zeros(shape, dtype=np.float32)

    for k, p in enumerate(products):
        z_aligned = p.complex * np.exp(
            -1j * phi_hat[k], dtype=np.complex64
        ).astype(np.complex64)
        w = p.weight
        complex_acc += (w.astype(np.complex64) * z_aligned).astype(np.complex64)
        weight_sum += w
        if p.coherence is not None:
            coh_acc += w * p.coherence
        n_bursts += (w > 0).astype(np.uint8)
        # component label: pick the component of the strongest contributor
        stronger = w > max_weight_per_pixel
        component_id[stronger] = component_per_node[k] + 1  # 0 reserved for nodata
        max_weight_per_pixel[stronger] = w[stronger]

    # Normalize
    nodata = weight_sum <= 0.0
    complex_out = np.zeros(shape, dtype=np.complex64)
    coh_out = np.zeros(shape, dtype=np.float32)
    complex_out[~nodata] = complex_acc[~nodata] / weight_sum[~nodata]
    coh_out[~nodata] = coh_acc[~nodata] / weight_sum[~nodata]
    complex_out[nodata] = 0.0
    component_id[nodata] = 0

    from faninsar.processing.merge.products import MosaicProduct

    logger.info(
        "merge_burst_products: mode=%s n_bursts=%d nodata_px=%d",
        mode,
        n,
        int(nodata.sum()),
    )
    return MosaicProduct(
        grid=grid,
        complex=complex_out,
        coherence=coh_out,
        weight_sum=weight_sum,
        n_bursts=n_bursts,
        component_id=component_id,
        network=network_stats,
    )
