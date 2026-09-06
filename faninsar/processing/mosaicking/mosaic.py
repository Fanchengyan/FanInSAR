"""Weighted mosaic of geocoded burst products.

The primary entry point is now :func:`faninsar.processing.mosaicking.methods.merge_bursts`,
which exposes the seven reference-grounded merge strategies documented in
Waymark ``NOTE-0001`` and the global ``sar-burst-merge`` skill.
:func:`merge_burst_products` is retained as a
backward-compatible wrapper that maps the legacy ``mode`` argument onto a
:func:`merge_bursts` method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from faninsar.logging import setup_logger
from faninsar.processing.mosaicking.methods import MergeMethod, merge_bursts

if TYPE_CHECKING:
    from faninsar.processing.mosaicking.products import BurstGeoProduct, MosaicProduct

logger = setup_logger(__name__)

__all__ = ["merge_burst_products"]

#: Legacy ``mode`` → new ``MergeMethod`` mapping (see ``methods.py``).
_LEGACY_MODE_TO_METHOD: dict[str, MergeMethod] = {
    "phase_network": "faninsar_network",
    "complex_average": "complex_average",
}


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
        unwrapped = [p.burst_id for p in products if p.phase_domain == "unwrapped"]
        if unwrapped:
            msg = (
                "merge_burst_products received unwrapped-phase products "
                f"({unwrapped}). The timing rule requires merge in the "
                "complex domain before unwrapping. Pass "
                "allow_unwrapped_merge=True to opt in (not recommended)."
            )
            raise ValueError(msg)

    grid = products[0].grid
    for p in products[1:]:
        if p.grid.shape != grid.shape or p.grid.crs != grid.crs:
            msg = "all products must share the same GeoGridSpec"
            raise ValueError(msg)

    method = _LEGACY_MODE_TO_METHOD.get(mode, "insardev_ramp")
    if mode != "phase_network":
        logger.info(
            "merge_burst_products: legacy mode='%s' → method='%s'", mode, method
        )
    return merge_bursts(
        products,
        method=method,
        min_overlap_px=min_overlap_px,
        min_edge_coherence=min_edge_coherence,
        path_policy=path_policy,
        allow_asc_desc_phase_link=allow_asc_desc_phase_link,
        reference_node=reference_node,
    )
