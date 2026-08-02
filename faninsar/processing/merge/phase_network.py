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
    """One overlap edge between two burst products.

    Models the relative phase as a constant plus optional range slope::

        φ_i(x) - φ_j(x) ≈ dphi_rad + range_slope_rad_per_px * (x - x_ref)

    ``x_ref`` is the frame-centre column ``0.5 * (width - 1)``.  The constant
    is converted from the overlap circular mean to this common reference so
    the mosaic screen ``phi_hat + slope * (x - x_ref)`` is consistent.
    The range slope captures residual cross-swath ramps that a pure constant
    cannot absorb.
    """

    i: int
    j: int
    dphi_rad: float
    coherence: float
    overlap_px: int
    weight: float
    range_slope_rad_per_px: float = 0.0


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
    """Result of weighted least-squares phase adjustment.

    Attributes
    ----------
    phi_hat : numpy.ndarray
        Per-node constant phase (rad), relative to the component reference.
    range_slope_hat : numpy.ndarray
        Per-node residual range slope (rad / column), relative to the
        component reference (pinned to 0).
    component_id : numpy.ndarray
        Connected-component label per node.
    edge_residuals : numpy.ndarray
        Constant-term residuals on each intra-component edge.
    rms_residual : float
        Weighted RMS of constant-term residuals.
    stats : MergeGraphStats
        Summary counts.

    """

    phi_hat: np.ndarray
    range_slope_hat: np.ndarray
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

    Computes a constant Δφ and a residual range slope from column-wise
    circular means of the complex product field:

    .. math::

        \Delta\phi_{ij}(x) \approx a + b\, x
        = \arg\sum_{p\in\Omega_x} w_i w_j\, z_i z_j^*

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
    denom = float((wi * wj * np.abs(z_i) * np.abs(z_j)).sum())
    coh = float(abs(c_sum) / denom) if denom > 0.0 else 0.0
    if coh < min_edge_coherence:
        return None

    # Constant from global circular mean (robust; no 2π unwrap at x=0).
    # This is the mean phase difference over the overlap footprint, i.e. the
    # value of Δφ near the overlap column centroid x0 — not the intercept at
    # the frame centre.  When a range slope is also estimated we convert the
    # constant to the common frame reference x_ref so that the mosaic screen
    #   Δφ(x) = dphi_ref + slope * (x - x_ref)
    # is consistent (otherwise nonzero slopes leave multi-radian seam jumps).
    dphi = float(np.angle(c_sum))
    # Column-wise means → unwrap → slope; then lift dphi to x_ref.
    _rows, cols = np.where(overlap)
    range_slope = 0.0
    flatness = 1.0
    x0_overlap: float | None = None
    if cols.size >= min_overlap_px:
        unique_cols = np.unique(cols)
        if unique_cols.size >= 4:
            col_centers: list[float] = []
            col_phases: list[float] = []
            for c in unique_cols:
                sel = cols == c
                if int(sel.sum()) < 2:
                    continue
                cc = contrib[sel].sum()
                if abs(cc) <= 0.0:
                    continue
                col_centers.append(float(c))
                col_phases.append(float(np.angle(cc)))
            if len(col_phases) >= 4:
                x = np.asarray(col_centers, dtype=np.float64)
                x0_overlap = float(np.mean(x))
                ph = np.unwrap(np.asarray(col_phases, dtype=np.float64))
                # ph ≈ slope * (x - x0) + c; circular mean ≈ c when centered.
                coef = np.polyfit(x - x0_overlap, ph, 1)
                range_slope = float(coef[0])
                resid = ph - np.polyval(coef, x - x0_overlap)
                resid_std = float(np.std(resid))
                flatness = 1.0 / (1.0 + resid_std)
                # Reject slopes that would integrate to multi-fringe across
                # the overlap (non-linear residual mis-modeled as a ramp).
                span_overlap = abs(range_slope) * float(np.ptp(x))
                if span_overlap > np.pi or resid_std > 1.0:
                    range_slope = 0.0
                    flatness *= 0.5
                    x0_overlap = None

    if range_slope != 0.0 and x0_overlap is not None:
        width = int(w_i.shape[1])
        x_ref = 0.5 * (width - 1)
        # Δφ(x) ≈ dphi + slope*(x - x0) → intercept at x_ref for mosaic apply.
        dphi = float(dphi + range_slope * (x_ref - x0_overlap))
        # Keep dphi in (-π, π] so LS constants stay well-conditioned.
        dphi = float(np.angle(np.exp(1j * dphi)))

    weight = coh * n_overlap * flatness
    return PhaseEdge(
        i=0,
        j=1,
        dphi_rad=dphi,
        coherence=coh,
        overlap_px=n_overlap,
        weight=weight,
        range_slope_rad_per_px=range_slope,
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
                    range_slope_rad_per_px=edge.range_slope_rad_per_px,
                )
            )
    logger.info("estimate_edges: %d nodes, %d edges", len(nodes), len(edges))
    return MergeGraph(nodes=nodes, edges=tuple(edges))


def _connected_components(n_nodes: int, edges: tuple[PhaseEdge, ...]) -> np.ndarray:
    """Union-find connected components over undirected edges."""
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

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

    Minimizes constant and range-slope residuals:

    .. math::

        \sum_{(i,j)\in E} W_{ij}\Big[
            (\phi_i - \phi_j - \Delta\phi_{ij})^2
            + (s_i - s_j - b_{ij})^2
        \Big]

    subject to ``phi[ref] = 0`` and ``s[ref] = 0`` per connected component.

    Parameters
    ----------
    graph : MergeGraph
        Phase network.
    reference_node : int, optional
        Node index pinned to zero in its component.

    Returns
    -------
    NetworkSolution
        Per-node constant phase, range slope, component labels, residuals.

    """
    n = len(graph.nodes)
    component_id = _connected_components(n, graph.edges)
    phi_hat = np.zeros(n, dtype=np.float64)
    range_slope_hat = np.zeros(n, dtype=np.float64)

    # Reference per component
    comp_refs: dict[int, int] = {}
    for k in range(n):
        c = int(component_id[k])
        if c not in comp_refs:
            ref = reference_node if int(component_id[reference_node]) == c else k
            comp_refs[c] = ref

    # Unknowns: for each non-ref node, (phi, slope) → 2 columns each.
    # Column layout: [phi_0, s_0, phi_1, s_1, ...] for non-ref nodes in order.
    col_index: dict[int, int] = {}
    col = 0
    for k in range(n):
        c = int(component_id[k])
        if k == comp_refs[c]:
            continue
        col_index[k] = col
        col += 2
    n_unknowns = col

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b: list[float] = []

    def _add_diff_eq(
        *,
        i: int,
        j: int,
        observed: float,
        sqrt_w: float,
        offset: int,
    ) -> None:
        """Add weighted difference equation for one unknown block offset."""
        row = len(b)
        ki = col_index.get(i)
        kj = col_index.get(j)
        if ki is None and kj is None:
            return
        if ki is not None and kj is not None:
            rows.append(row)
            cols.append(ki + offset)
            data.append(sqrt_w)
            rows.append(row)
            cols.append(kj + offset)
            data.append(-sqrt_w)
            b.append(sqrt_w * observed)
        elif ki is not None:
            rows.append(row)
            cols.append(ki + offset)
            data.append(sqrt_w)
            b.append(sqrt_w * observed)
        else:
            rows.append(row)
            cols.append(kj + offset)  # type: ignore[operator]
            data.append(-sqrt_w)
            b.append(sqrt_w * observed)

    for e in graph.edges:
        ci = int(component_id[e.i])
        cj = int(component_id[e.j])
        if ci != cj:
            continue
        sqrt_w = float(np.sqrt(max(e.weight, 1e-12)))
        # Constant equation: phi_i - phi_j = dphi
        _add_diff_eq(i=e.i, j=e.j, observed=e.dphi_rad, sqrt_w=sqrt_w, offset=0)
        # Slope equation: s_i - s_j = range_slope
        _add_diff_eq(
            i=e.i,
            j=e.j,
            observed=e.range_slope_rad_per_px,
            sqrt_w=sqrt_w,
            offset=1,
        )

    if n_unknowns > 0 and rows:
        a_mat = csr_matrix((data, (rows, cols)), shape=(len(b), n_unknowns))
        b_arr = np.asarray(b, dtype=np.float64)
        sol = lsqr(a_mat, b_arr)[0]
        for k in range(n):
            c = int(component_id[k])
            if k == comp_refs[c]:
                phi_hat[k] = 0.0
                range_slope_hat[k] = 0.0
            else:
                base = col_index[k]
                phi_hat[k] = float(sol[base])
                range_slope_hat[k] = float(sol[base + 1])

    # Residuals on constant term (backward compatible reporting)
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
    logger.info(
        "solve_network: %d components, rms=%.3e rad, max|slope|=%.3e rad/px",
        stats.n_components,
        rms,
        float(np.max(np.abs(range_slope_hat))) if n else 0.0,
    )
    return NetworkSolution(
        phi_hat=phi_hat,
        range_slope_hat=range_slope_hat,
        component_id=component_id,
        edge_residuals=residuals,
        rms_residual=rms,
        stats=stats,
    )
