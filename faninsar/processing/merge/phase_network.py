"""Burst overlap phase network: Δφ estimation and weighted LS adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.processing.merge.products import BurstGeoProduct

logger = setup_logger(__name__)

__all__ = [
    "MergeGraph",
    "MergeGraphStats",
    "NetworkSolution",
    "PhaseEdge",
    "estimate_edge",
    "estimate_edges",
    "solve_network",
]


@dataclass(frozen=True, slots=True)
class PhaseEdge:
    """One overlap edge between two burst products."""

    i: int
    j: int
    dphi_rad: float
    coherence: float
    overlap_px: int
    weight: float


@dataclass(frozen=True, slots=True)
class MergeGraph:
    """Burst overlap phase network."""

    nodes: tuple[BurstGeoProduct, ...]
    edges: tuple[PhaseEdge, ...]


@dataclass(frozen=True, slots=True)
class MergeGraphStats:
    """Summary statistics for a phase network."""

    n_nodes: int
    n_edges: int
    n_components: int
    rms_residual: float


@dataclass(frozen=True, slots=True)
class NetworkSolution:
    """Result of weighted least-squares phase adjustment."""

    phi_hat: np.ndarray
    component_id: np.ndarray
    edge_residuals: np.ndarray
    rms_residual: float
    stats: MergeGraphStats


def estimate_edge(
    product_i: BurstGeoProduct,
    product_j: BurstGeoProduct,
    *,
    min_overlap_px: int,
    min_edge_coherence: float,
    threshold: float = 0.0,
) -> PhaseEdge | None:
    r"""Estimate the overlap phase offset between two burst products.

    Computes

    .. math::

        \Delta\phi_{ij} = \arg\sum_{p\in\Omega} w_i w_j\, z_i z_j^*

    and the corresponding coherence and edge weight.

    Parameters
    ----------
    product_i, product_j : BurstGeoProduct
        Two geocoded burst products on the same grid.
    min_overlap_px : int
        Minimum number of overlapping pixels to form an edge.
    min_edge_coherence : float
        Minimum edge coherence to keep the edge.
    threshold : float, optional
        Weight threshold defining the overlap region.

    Returns
    -------
    PhaseEdge or None
        The estimated edge, or ``None`` if overlap/coherence insufficient.

    """
    w_i = product_i.weight
    w_j = product_j.weight
    overlap = (w_i > threshold) & (w_j > threshold)
    n_overlap = int(overlap.sum())
    if n_overlap < min_overlap_px:
        return None

    z_i = product_i.complex[overlap]
    z_j = product_j.complex[overlap]
    wi = w_i[overlap].astype(np.float64)
    wj = w_j[overlap].astype(np.float64)
    contrib = wi * wj * z_i * np.conj(z_j)
    c_sum = contrib.sum()
    dphi = float(np.angle(c_sum))
    denom = float((wi * wj * np.abs(z_i) * np.abs(z_j)).sum())
    coh = float(abs(c_sum) / denom) if denom > 0.0 else 0.0
    if coh < min_edge_coherence:
        return None
    weight = coh * n_overlap
    return PhaseEdge(
        i=0,
        j=1,
        dphi_rad=dphi,
        coherence=coh,
        overlap_px=n_overlap,
        weight=weight,
    )


def estimate_edges(
    products: list[BurstGeoProduct],
    *,
    min_overlap_px: int = 500,
    min_edge_coherence: float = 0.15,
    threshold: float = 0.0,
    path_policy: str = "allow_cross_path",
    allow_asc_desc_phase_link: bool = False,
) -> MergeGraph:
    """Build a phase network from a list of burst products.

    Parameters
    ----------
    products : list of BurstGeoProduct
        Geocoded burst products on the same grid.
    min_overlap_px : int, optional
        Minimum overlap pixels to form an edge.
    min_edge_coherence : float, optional
        Minimum edge coherence.
    threshold : float, optional
        Weight threshold defining the overlap region.
    path_policy : str, optional
        ``"same_path_only"`` only links bursts of the same ``path_id``;
        ``"allow_cross_path"`` also links different paths.
    allow_asc_desc_phase_link : bool, optional
        When ``False``, ascending/descending pairs are never linked by a
        phase edge (they may still share the grid).

    Returns
    -------
    MergeGraph
        The phase network with nodes and weighted edges.

    """
    nodes = tuple(products)
    edges: list[PhaseEdge] = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            pa, pb = nodes[a], nodes[b]
            if path_policy == "same_path_only" and pa.path_id != pb.path_id:
                continue
            if not allow_asc_desc_phase_link and pa.look_direction != pb.look_direction:
                continue
            edge = estimate_edge(
                pa,
                pb,
                min_overlap_px=min_overlap_px,
                min_edge_coherence=min_edge_coherence,
                threshold=threshold,
            )
            if edge is None:
                continue
            edges.append(
                PhaseEdge(
                    i=a,
                    j=b,
                    dphi_rad=edge.dphi_rad,
                    coherence=edge.coherence,
                    overlap_px=edge.overlap_px,
                    weight=edge.weight,
                )
            )
    logger.info("estimate_edges: %d nodes, %d edges", len(nodes), len(edges))
    return MergeGraph(nodes=nodes, edges=tuple(edges))


def _connected_components(n_nodes: int, edges: tuple[PhaseEdge, ...]) -> np.ndarray:
    """Return a component label (0-based) per node via union-find."""
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for e in edges:
        union(e.i, e.j)

    comp_ids: dict[int, int] = {}
    labels = np.zeros(n_nodes, dtype=np.int16)
    for k in range(n_nodes):
        root = find(k)
        if root not in comp_ids:
            comp_ids[root] = len(comp_ids)
        labels[k] = comp_ids[root]
    return labels


def solve_network(
    graph: MergeGraph,
    *,
    reference_node: int = 0,
) -> NetworkSolution:
    r"""Solve the weighted least-squares phase network adjustment.

    Minimizes

    .. math::

        \sum_{(i,j)\in E} W_{ij}(\phi_i - \phi_j - \Delta\phi_{ij})^2

    subject to ``phi[reference_node] = 0`` per connected component.

    Parameters
    ----------
    graph : MergeGraph
        Phase network.
    reference_node : int, optional
        Node index pinned to zero in its component.

    Returns
    -------
    NetworkSolution
        Per-node phase estimates, component labels, residuals, and RMS.

    """
    n = len(graph.nodes)
    component_id = _connected_components(n, graph.edges)
    phi_hat = np.zeros(n, dtype=np.float64)

    # Reference per component
    comp_refs: dict[int, int] = {}
    for k in range(n):
        c = int(component_id[k])
        if c not in comp_refs:
            ref = reference_node if int(component_id[reference_node]) == c else k
            comp_refs[c] = ref

    # Build sparse normal equations per component.
    # Unknowns: phi_k for k != ref_of_comp. We assemble A x = b with
    # one row per edge, weighted by sqrt(W).
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b: list[float] = []

    # Map node index -> column index (skip references)
    col_index: dict[int, int] = {}
    col = 0
    for k in range(n):
        c = int(component_id[k])
        if k == comp_refs[c]:
            continue
        col_index[k] = col
        col += 1
    n_unknowns = col

    for e in graph.edges:
        ci = int(component_id[e.i])
        cj = int(component_id[e.j])
        if ci != cj:
            continue
        sqrt_w = float(np.sqrt(max(e.weight, 1e-12)))
        row = len(b)
        ki = col_index.get(e.i)
        kj = col_index.get(e.j)
        if ki is None and kj is None:
            continue
        if ki is not None and kj is not None:
            rows.append(row)
            cols.append(ki)
            data.append(sqrt_w)
            rows.append(row)
            cols.append(kj)
            data.append(-sqrt_w)
            b.append(sqrt_w * e.dphi_rad)
        elif ki is not None:
            # j is the reference (phi_j = 0): phi_i = dphi
            rows.append(row)
            cols.append(ki)
            data.append(sqrt_w)
            b.append(sqrt_w * e.dphi_rad)
        else:  # kj is not None, i is the reference (phi_i = 0): -phi_j = dphi
            rows.append(row)
            cols.append(kj)
            data.append(-sqrt_w)
            b.append(sqrt_w * e.dphi_rad)

    if n_unknowns > 0 and rows:
        a_mat = csr_matrix((data, (rows, cols)), shape=(len(b), n_unknowns))
        b_arr = np.asarray(b, dtype=np.float64)
        sol = lsqr(a_mat, b_arr)[0]
        for k in range(n):
            c = int(component_id[k])
            if k == comp_refs[c]:
                phi_hat[k] = 0.0
            else:
                phi_hat[k] = float(sol[col_index[k]])
    else:
        # No edges or all references: phi_hat stays 0
        pass

    # Residuals
    residuals = np.array(
        [
            (phi_hat[e.i] - phi_hat[e.j] - e.dphi_rad)
            for e in graph.edges
            if int(component_id[e.i]) == int(component_id[e.j])
        ],
        dtype=np.float64,
    )
    if residuals.size:
        weights = np.array(
            [
                e.weight
                for e in graph.edges
                if int(component_id[e.i]) == int(component_id[e.j])
            ],
            dtype=np.float64,
        )
        rms = float(np.sqrt(np.sum(weights * residuals**2) / np.sum(weights)))
    else:
        rms = 0.0

    stats = MergeGraphStats(
        n_nodes=n,
        n_edges=len(graph.edges),
        n_components=int(component_id.max()) + 1 if n > 0 else 0,
        rms_residual=rms,
    )
    logger.info("solve_network: %d components, rms=%.3e rad", stats.n_components, rms)
    return NetworkSolution(
        phi_hat=phi_hat,
        component_id=component_id,
        edge_residuals=residuals,
        rms_residual=rms,
        stats=stats,
    )
