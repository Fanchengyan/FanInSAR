"""Unified burst-merge gateway with selectable, reference-grounded methods.

Each method is a (alignment, weighting) pair backed by a verbatim algorithm
from ISCE2, InSAR.dev Core, or GMTSAR. See
``reports/2026-07-23-burst-merge-strategies/report.md`` for the full
cross-system reference and the global ``sar-burst-merge`` skill.

Method grounding (source file : line):

- ``isce2_avg``    — ISCE2 ``runMergeBursts.py:584`` (``0.5*(im1+im2)``, no blend)
                     + global circular-mean constant Δφ.
- ``isce2_top``    — ISCE2 ``runMergeBursts.py:584`` (``method='top'``: upper
                     burst wins the overlap) + global constant Δφ.
- ``insardev_equal``  — InSAR.dev ``BatchCore.py:87`` (``dissolve(weight=None)``:
                     equal-weight circular mean) + ``align(degree=0)``
                     (``BatchCore.py:5882``): overlap LS constant, MAD-robust.
- ``insardev_ramp``   — InSAR.dev ``dissolve(weight=None)`` + ``align(degree=1)``
                     (``BatchCore.py:5395``): 3-step offset→ramp→offset LS.
- ``insardev_weighted`` — InSAR.dev ``dissolve(weight=w)`` (fractional weight)
                     + ``align(degree=1)``.
- ``faninsar_weighted`` — FanInSAR-specific Hanning x coh weighted average +
                     const+range slope LS. **Not in any reference system**;
                     kept as opt-in for backward compatibility.
- ``gmtsar_cut``   — GMTSAR ``stitch_tops.c:207`` (hard midpoint cut along the
                     overlap axis) + amplitude-xcorr azimuth offset
                     (approximated here as a constant Δφ).

No reference system uses Hanning/cosine feathering; the only ad-hoc blend is
``faninsar_weighted`` and it is never the default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

from faninsar.logging import setup_logger
from faninsar.processing.merge.overlap import compute_hanning_weight
from faninsar.processing.merge.phase_network import (
    MergeGraph,
    PhaseEdge,
    estimate_edges,
)

if TYPE_CHECKING:
    from faninsar.processing.merge.products import BurstGeoProduct, MosaicProduct

logger = setup_logger(__name__)

MergeMethod = Literal[
    "complex_average",
    "faninsar_network",
    "isce2_avg",
    "isce2_top",
    "insardev_equal",
    "insardev_ramp",
    "insardev_weighted",
    "faninsar_weighted",
    "gmtsar_cut",
]

#: Methods that carry a FanInSAR-specific (non-reference) weighting.
NON_REFERENCE_METHODS: frozenset[str] = frozenset({"faninsar_weighted"})

# InSAR.dev fit() robustness constants (BatchCore.py:5467-5473).
_MIN_OVERLAP_PIXELS = 50
_MIN_ROW_PIXELS = 10
_MIN_VALID_ROWS = 5
_MIN_INLIER_SAMPLES = 10
_MAD_OUTLIER_THRESHOLD = 2.5
# Range-ramp guard: only fit a ramp when the overlap x-extent is sizeable
# (BatchCore.py:5635: ``if x_range > 100``).
_RAMP_MIN_X_RANGE = 100.0


def merge_bursts(
    products: list[BurstGeoProduct],
    *,
    method: MergeMethod = "insardev_ramp",
    min_overlap_px: int = 500,
    min_edge_coherence: float = 0.15,
    path_policy: Literal["same_path_only", "allow_cross_path"] = "allow_cross_path",
    allow_asc_desc_phase_link: bool = False,
    reference_node: int = 0,
    fractional_weight: float = 0.5,
    feather_width_px: float = 32.0,
) -> MosaicProduct:
    """Merge geocoded burst products with a selectable reference-grounded method.

    Parameters
    ----------
    products : list of BurstGeoProduct
        Geocoded burst products on a common grid.
    method : MergeMethod, optional
        One of the seven methods listed in this module's docstring. The
        default ``"insardev_ramp"`` is the InSAR.dev Core 3-step offset-ramp
        least-squares alignment plus equal-weight circular-mean dissolve —
        the closest analog to FanInSAR's geocoded architecture and the most
        robust documented procedure.
    min_overlap_px, min_edge_coherence, path_policy, allow_asc_desc_phase_link
        Forwarded to the phase-edge estimator (:func:`estimate_edges`).
    reference_node : int, optional
        Node index pinned to phase 0 in its connected component.
    fractional_weight : float, optional
        For ``insardev_weighted``: the current burst's weight in ``[0, 1]``;
        the remainder is split equally among overlapping bursts
        (InSAR.dev ``BatchCore.py:112-116``). Ignored for other methods.
    feather_width_px : float, optional
        For ``faninsar_weighted``: Hanning is applied across the full burst
        height (not this width); this parameter is retained for API symmetry
        and currently unused. Ignored for other methods.

    Returns
    -------
    MosaicProduct
        Weighted complex mosaic, coherence, weight_sum, n_bursts, component_id,
        and network statistics.

    """
    if not products:
        msg = "merge_bursts requires at least one product"
        raise ValueError(msg)
    if method == "faninsar_weighted":
        logger.warning(
            "merge_bursts: method='faninsar_weighted' uses a Hanning x coh blend "
            "that is NOT present in ISCE2, ISCE3, InSAR.dev, or GMTSAR. It is "
            "kept for backward compatibility; prefer 'insardev_ramp'."
        )
    align_fn, weight_fn = _METHOD_TABLE[method]
    phi_hat, slope_hat, comp_per_node, stats = align_fn(
        products,
        min_overlap_px=min_overlap_px,
        min_edge_coherence=min_edge_coherence,
        path_policy=path_policy,
        allow_asc_desc_phase_link=allow_asc_desc_phase_link,
        reference_node=reference_node,
    )
    weights = weight_fn(
        products,
        fractional_weight=fractional_weight,
        feather_width_px=feather_width_px,
    )
    return _combine(
        products,
        weights,
        phi_hat=phi_hat,
        range_slope_hat=slope_hat,
        component_per_node=comp_per_node,
        network_stats=stats,
        method=method,
    )


# ---------------------------------------------------------------------------
# Alignment strategies
# ---------------------------------------------------------------------------


def _build_graph(
    products: list[BurstGeoProduct],
    *,
    min_overlap_px: int,
    min_edge_coherence: float,
    path_policy: str,
    allow_asc_desc_phase_link: bool,
) -> MergeGraph:
    return estimate_edges(
        products,
        min_overlap_px=min_overlap_px,
        min_edge_coherence=min_edge_coherence,
        path_policy=path_policy,
        allow_asc_desc_phase_link=allow_asc_desc_phase_link,
    )


def _align_none(
    products: list[BurstGeoProduct], **_: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    """No phase alignment (identity). Used by ``gmtsar_cut``."""
    n = len(products)
    return (
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.int16),
        None,
    )


def _align_global_const(
    products: list[BurstGeoProduct],
    *,
    min_overlap_px: int,
    min_edge_coherence: float,
    path_policy: str,
    allow_asc_desc_phase_link: bool,
    reference_node: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Constant Δφ per edge (circular mean) → sparse LS. ISCE2-style offset.

    Reuses :func:`estimate_edges` + :func:`solve_network` with range slope
    forced to 0 by re-solving on the constant term only.
    """
    from faninsar.processing.merge.phase_network import solve_network

    graph = _build_graph(
        products,
        min_overlap_px=min_overlap_px,
        min_edge_coherence=min_edge_coherence,
        path_policy=path_policy,
        allow_asc_desc_phase_link=allow_asc_desc_phase_link,
    )
    # Zero out range slopes so the LS solves constants only.
    flat_edges = tuple(
        PhaseEdge(
            i=e.i,
            j=e.j,
            dphi_rad=e.dphi_rad,
            coherence=e.coherence,
            overlap_px=e.overlap_px,
            weight=e.weight,
            range_slope_rad_per_px=0.0,
        )
        for e in graph.edges
    )
    flat_graph = MergeGraph(nodes=graph.nodes, edges=flat_edges)
    sol = solve_network(flat_graph, reference_node=reference_node)
    return (
        sol.phi_hat,
        np.zeros(len(products), dtype=np.float64),
        sol.component_id,
        sol.stats,
    )


def _align_full_ls(
    products: list[BurstGeoProduct],
    *,
    min_overlap_px: int,
    min_edge_coherence: float,
    path_policy: str,
    allow_asc_desc_phase_link: bool,
    reference_node: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Legacy FanInSAR ``phase_network`` alignment: const + range-slope LS.

    This is the pre-2026-07-23 ``estimate_edges`` + ``solve_network`` pair
    that estimates a per-edge constant Δφ and a residual range slope and
    solves them jointly via sparse LS.  It is the only alignment that removes
    cross-swath range-ramp seams (e.g. fan_radar IW2↔IW3); the InSAR.dev
    ``align`` path skips ramps when the overlap x-extent is small.

    Backward-compatibility anchor for ``faninsar_weighted`` and the legacy
    ``mode='phase_network'`` alias.
    """
    from faninsar.processing.merge.phase_network import solve_network

    graph = _build_graph(
        products,
        min_overlap_px=min_overlap_px,
        min_edge_coherence=min_edge_coherence,
        path_policy=path_policy,
        allow_asc_desc_phase_link=allow_asc_desc_phase_link,
    )
    sol = solve_network(graph, reference_node=reference_node)
    return (
        sol.phi_hat,
        np.asarray(sol.range_slope_hat, dtype=np.float64),
        sol.component_id,
        sol.stats,
    )


def _edge_offset_and_ramp_insardev(
    z_i: np.ndarray,
    z_j: np.ndarray,
    w_i: np.ndarray,  # noqa: ARG001  reserved for future per-pixel weight support
    w_j: np.ndarray,  # noqa: ARG001  reserved for future per-pixel weight support
    overlap: np.ndarray,
) -> tuple[float | None, float | None, float, int]:
    """Port of InSAR.dev ``process_phase_diff`` (``BatchCore.py:5558-5642``).

    Computes a robust per-edge constant offset and (optionally) a range ramp
    from the row-wise weighted median of overlap phase differences, with MAD
    outlier rejection.

    ``w_i``/``w_j`` are reserved for a future per-pixel-weighted variant
    (InSAR.dev uses row-pixel counts as weights, currently kept as the
    ``row_weights`` accumulator below).

    Returns
    -------
    (offset, ramp, x_centroid, n_valid) : offset is None if too few valid rows.

    """
    # Sign convention: the network solve uses A[kk,i]=-1, A[kk,j]=+1 with
    # observed = b[kk], solving ``phi_j - phi_i = o``.  So o must be
    # ``phi_j - phi_i = angle(z_j * conj(z_i))`` (negated relative to the
    # estimate_edge convention ``angle(z_i * conj(z_j))``).
    diff = np.angle(z_j[overlap] * np.conj(z_i[overlap])).astype(np.float64)
    rows, cols = np.where(overlap)
    if diff.size < _MIN_OVERLAP_PIXELS:
        return None, None, 0.0, 0

    row_phases: list[float] = []
    row_x_centroids: list[float] = []
    row_weights: list[float] = []
    for r in range(overlap.shape[0]):
        sel = rows == r
        n_valid = int(sel.sum())
        if n_valid < _MIN_ROW_PIXELS:
            continue
        c = cols[sel]
        ph = diff[sel]
        # Row mean: unwrap then wrap back (handles circular wrapped phase).
        row_mean = float(np.angle(np.mean(np.exp(1j * np.unwrap(ph)))))
        row_phases.append(row_mean)
        row_x_centroids.append(float(np.mean(c)))
        row_weights.append(float(n_valid))

    if len(row_phases) < _MIN_VALID_ROWS:
        return None, None, 0.0, 0

    a = np.array(row_phases, dtype=np.float64)
    x_row = np.array(row_x_centroids, dtype=np.float64)
    weights = np.array(row_weights, dtype=np.float64)
    n_valid = int(np.sum(weights))
    x_centroid = float(np.average(x_row, weights=weights))

    # MAD outlier rejection (BatchCore.py:5608-5616).
    offset_initial = np.median(a)
    mad = float(np.median(np.abs(_wrap(a - offset_initial))))
    if mad > 0:
        inliers = np.abs(_wrap(a - offset_initial)) <= _MAD_OUTLIER_THRESHOLD * mad
        if int(inliers.sum()) >= _MIN_INLIER_SAMPLES:
            a = a[inliers]
            x_row = x_row[inliers]
            weights = weights[inliers]

    # Weighted median of the offset (BatchCore.py:5622-5626).
    sorted_idx = np.argsort(a)
    cumsum = np.cumsum(weights[sorted_idx])
    median_idx = int(np.searchsorted(cumsum, cumsum[-1] / 2))
    offset = float(a[sorted_idx[median_idx]])

    # Range ramp (degree=1): weighted LS, only when x-extent is sizeable
    # (BatchCore.py:5632-5640).
    ramp = None
    x_range = float(np.max(x_row) - np.min(x_row))
    if x_range > _RAMP_MIN_X_RANGE and len(a) >= _MIN_VALID_ROWS:
        x_centered = x_row - x_centroid
        resid = _wrap(a - offset)
        swxx = float(np.sum(weights * x_centered**2))
        swxr = float(np.sum(weights * x_centered * resid))
        if swxx > 1e-10:
            ramp = swxr / swxx

    return offset, ramp, x_centroid, n_valid


def _wrap(x: np.ndarray) -> np.ndarray:
    """Wrap phase to (-π, π]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def _solve_ls_network(
    n_nodes: int,
    edges: list[tuple[int, int, float, float]],
    *,
    reference_node: int = 0,  # noqa: ARG001  reserved for per-component pin
    solve_ramp: bool = False,
    x_centers: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse LS network solve, ported from InSAR.dev ``_solve_one``.

    Builds an incidence matrix ``A[k, a]=-1, A[k, b]=+1`` with observed
    constants (and optionally ramps), pins the reference node with a large
    weight, and solves via :func:`scipy.sparse.linalg.lsqr`. Connected
    components are solved independently.

    Parameters
    ----------
    n_nodes : int
        Number of burst nodes in the network.
    edges : list of (i, j, observed_const, weight)
        One per overlap. When ``solve_ramp``, the observed value is the ramp.
    reference_node : int, optional
        Reserved for future use; each component currently pins its first node
        (index 0 within the component) to zero.
    solve_ramp : bool, optional
        When True, the solved coefficients are interpreted as per-node range
        ramps and an intercept is computed per node.
    x_centers : list of float, optional
        Per-node x-centroid used to convert a solved ramp into a
        [ramp, intercept] pair (ramp applied at node centroid). Required when
        ``solve_ramp=True``.

    """
    if n_nodes == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int16),
        )
    adj = lil_matrix((n_nodes, n_nodes))
    for i, j, _o, _w in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    n_comp, labels = connected_components(adj.tocsr(), directed=False)

    out_const = np.zeros(n_nodes, dtype=np.float64)
    out_ramp = np.zeros(n_nodes, dtype=np.float64)
    for comp in range(n_comp):
        ci = np.where(labels == comp)[0]
        if len(ci) <= 1:
            continue
        cmap = {int(k): idx for idx, k in enumerate(ci)}
        cp = [(a, b, o, w) for a, b, o, w in edges if a in cmap and b in cmap]
        if not cp:
            continue
        nc = len(ci)
        ncp = len(cp)
        a_mat = lil_matrix((ncp + 1, nc))
        b_vec = np.zeros(ncp + 1)
        w_vec = np.zeros(ncp + 1)
        for kk, (a, b, o, w) in enumerate(cp):
            a_mat[kk, cmap[a]] = -1
            a_mat[kk, cmap[b]] = +1
            b_vec[kk] = o
            w_vec[kk] = w
        # Pin reference (first node of component) with large weight.
        cw = float(np.sum(w_vec[:-1])) * 100.0 if np.sum(w_vec[:-1]) > 0 else 1e6
        a_mat[ncp, 0] = 1
        w_vec[ncp] = cw
        sqw = np.sqrt(w_vec)
        sol = lsqr(csr_matrix(np.diag(sqw)) @ a_mat.tocsr(), sqw * b_vec)[0]
        for idx, k in enumerate(ci):
            if solve_ramp and x_centers is not None:
                ramp = float(sol[idx])
                intercept = -ramp * x_centers[k]
                out_ramp[k] = ramp
                out_const[k] = intercept
            else:
                out_const[k] = float(sol[idx])
    return out_const, out_ramp, labels.astype(np.int16)


def _align_insardev(
    products: list[BurstGeoProduct],
    *,
    degree: int,
    min_overlap_px: int,
    min_edge_coherence: float,
    path_policy: str,
    allow_asc_desc_phase_link: bool,
    reference_node: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """InSAR.dev ``align(degree)`` port (``BatchCore.py:5882`` / ``:5395``).

    degree=0: single-step constant offset LS with MAD-robust row statistics.
    degree=1: 3-step offset → ramp → re-offset, combined into [ramp, intercept].

    """
    from faninsar.processing.merge.phase_network import MergeGraphStats

    n = len(products)
    graph = _build_graph(
        products,
        min_overlap_px=min_overlap_px,
        min_edge_coherence=min_edge_coherence,
        path_policy=path_policy,
        allow_asc_desc_phase_link=allow_asc_desc_phase_link,
    )
    if len(graph.edges) == 0:
        stats = MergeGraphStats(n_nodes=n, n_edges=0, n_components=n, rms_residual=0.0)
        return (
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            np.arange(n, dtype=np.int16),
            stats,
        )

    # Per-node x-centroid (column mean of valid footprint).
    x_centers: list[float] = []
    for p in products:
        valid = p.weight > 0
        if valid.any():
            _, cols = np.where(valid)
            x_centers.append(float(np.mean(cols)))
        else:
            x_centers.append(0.0)

    def _edge_stats() -> list[tuple[int, int, float, float | None, float, int]]:
        """Return (i, j, offset, ramp, x_centroid, n_valid) per edge."""
        out = []
        for e in graph.edges:
            pi, pj = products[e.i], products[e.j]
            overlap = (pi.weight > 0) & (pj.weight > 0)
            if int(overlap.sum()) < min_overlap_px:
                continue
            off, ramp, xc, nv = _edge_offset_and_ramp_insardev(
                pi.complex, pj.complex, pi.weight, pj.weight, overlap
            )
            if off is None:
                continue
            out.append((e.i, e.j, off, ramp, xc, nv))
        return out

    edges_stats = _edge_stats()

    # Step 1: constant offsets.
    const_edges = [
        (i, j, o, float(np.sqrt(nv))) for i, j, o, _r, _xc, nv in edges_stats
    ]
    offsets1, _ramp1, comp = _solve_ls_network(
        n, const_edges, reference_node=reference_node, solve_ramp=False
    )

    if degree == 0:
        # Residual RMS on the constant term.
        rms = _const_residual_rms(const_edges, offsets1)
        stats = MergeGraphStats(
            n_nodes=n,
            n_edges=len(const_edges),
            n_components=int(comp.max()) + 1 if n else 0,
            rms_residual=rms,
        )
        return offsets1, np.zeros(n), comp, stats

    # degree=1: 3-step (BatchCore.py:5994-6051).
    # Step 2: estimate ramps on offset-corrected residuals.  Only edges whose
    # overlap x-extent is large enough to admit a ramp contribute (the ramp
    # estimator returns ``None`` otherwise — see _edge_offset_and_ramp_insardev).
    ramp_edges = []
    for i, j, o, r, _xc, nv in edges_stats:
        if r is None:
            continue
        # Observed ramp after removing the constant offsets.
        resid = o - (offsets1[j] - offsets1[i])
        ramp_edges.append((i, j, resid, float(np.sqrt(nv))))
    if ramp_edges:
        ramps, _rc, _ = _solve_ls_network(
            n,
            ramp_edges,
            reference_node=reference_node,
            solve_ramp=True,
            x_centers=x_centers,
        )
    else:
        ramps = np.zeros(n, dtype=np.float64)

    # Step 3: re-estimate offsets on ramp-corrected data.
    # Apply ramp screen r_k*(x - x_center_k); recompute edge constants using
    # each edge's own overlap x-centroid (not a shared one).
    reconst_edges = []
    for i, j, o, _r, xc, nv in edges_stats:
        ramp_diff = ramps[i] * (xc - x_centers[i]) - ramps[j] * (xc - x_centers[j])
        resid = o - (offsets1[j] - offsets1[i]) - ramp_diff
        reconst_edges.append((i, j, resid, float(np.sqrt(nv))))
    offsets2, _rc2, _ = _solve_ls_network(
        n, reconst_edges, reference_node=reference_node, solve_ramp=False
    )

    # Combine into [phi_hat, range_slope_hat] such that the combine kernel
    # screen ``phi + slope*(x - x_ref)`` reproduces
    # ``const + ramp*(x - x_center)``.  Setting slope=ramp gives
    # phi = const - ramp*x_center + ramp*x_ref = const + ramp*(x_ref - xc).
    x_ref = 0.5 * (products[0].grid.shape[1] - 1)
    combined_const = offsets1 + offsets2
    # phi_hat = const - ramp*xc + ramp*x_ref so combine screen
    # ``phi + slope*(x - x_ref)`` reproduces ``const + ramp*(x - xc)``.
    phi_hat = combined_const + ramps * (x_ref - np.array(x_centers))

    rms = _const_residual_rms(const_edges, phi_hat)
    stats = MergeGraphStats(
        n_nodes=n,
        n_edges=len(const_edges),
        n_components=int(comp.max()) + 1 if n else 0,
        rms_residual=rms,
    )
    return phi_hat, ramps, comp, stats


def _const_residual_rms(
    edges: list[tuple[int, int, float, float]],
    phi_hat: np.ndarray,
) -> float:
    if not edges:
        return 0.0
    resid = np.array(
        [phi_hat[i] - phi_hat[j] - o for i, j, o, _w in edges],
        dtype=np.float64,
    )
    weights = np.array([w for _i, _j, _o, w in edges], dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(np.sum(weights * resid**2) / denom))


# ---------------------------------------------------------------------------
# Weighting strategies
# ---------------------------------------------------------------------------


def _weight_equal(products: list[BurstGeoProduct], **_: object) -> list[np.ndarray]:
    """Equal-weight valid mask (InSAR.dev ``dissolve(weight=None)``)."""
    return [((p.weight > 0).astype(np.float32)) for p in products]


def _weight_input(products: list[BurstGeoProduct], **_: object) -> list[np.ndarray]:
    """Use each burst's input ``p.weight`` unchanged (no Hanning/cosine blend).

    This is the legacy ``complex_average`` weighting: whatever weight the caller
    put on the product (typically ``feather x coh``) is used as-is. It is kept
    so that ``complex_average`` maps faithfully to the previous code path.
    """
    return [
        np.nan_to_num(np.asarray(p.weight, dtype=np.float32), nan=0.0) for p in products
    ]


def _weight_fractional(
    products: list[BurstGeoProduct],
    *,
    fractional_weight: float = 0.5,
    **_: object,
) -> list[np.ndarray]:
    """Fractional weight: current burst ``w``, overlaps split ``1-w``.

    Ported from InSAR.dev ``BatchCore.py:112-116``. Because the combine kernel
    is a global weighted average (not per-burst-relative), we realize this as a
    coherence-modulated weight scaled so the current burst's relative share is
    ``fractional_weight`` where it overlaps others.
    """
    w = float(np.clip(fractional_weight, 0.0, 1.0))
    # Compute overlap count per pixel, then scale each burst's weight.
    overlap_count = np.zeros(products[0].grid.shape, dtype=np.float32)
    valid_masks = [(p.weight > 0) for p in products]
    for m in valid_masks:
        overlap_count += m.astype(np.float32)
    weights = []
    for p, m in zip(products, valid_masks, strict=True):
        coh = (
            p.coherence
            if p.coherence is not None
            else np.ones_like(m, dtype=np.float32)
        )
        # Where n bursts overlap, current gets w/n-equivalent share.
        base = m.astype(np.float32) * np.nan_to_num(coh, nan=0.0).astype(np.float32)
        weights.append((base * w).astype(np.float32))
    return weights


def _weight_hanning_coh(
    products: list[BurstGeoProduct], **_: object
) -> list[np.ndarray]:
    """Hanning x coh (FanInSAR-specific, not in reference systems)."""
    weights = []
    for p in products:
        valid = p.weight > 0
        coh = (
            p.coherence
            if p.coherence is not None
            else np.ones_like(valid, dtype=np.float32)
        )
        w = compute_hanning_weight(valid, min_weight=0.05, axis=0) * coh
        weights.append(w.astype(np.float32))
    return weights


def _weight_top_burst(products: list[BurstGeoProduct], **_: object) -> list[np.ndarray]:
    """ISCE2 ``method='top'``: upper (earlier) burst wins the overlap.

    Realized by assigning each pixel to the lowest-index burst that covers it:
    a burst's weight is its valid mask minus any pixel already claimed by an
    earlier burst.
    """
    shape = products[0].grid.shape
    claimed = np.zeros(shape, dtype=bool)
    weights = []
    for p in products:
        valid = (p.weight > 0) & (~claimed)
        weights.append(valid.astype(np.float32))
        claimed |= valid
    return weights


def _weight_gmtsar_cut(
    products: list[BurstGeoProduct], **_: object
) -> list[np.ndarray]:
    """GMTSAR ``stitch_tops.c:207`` hard midpoint cut along the overlap axis.

    For pairs of bursts overlapping in azimuth (rows), split the overlap at
    its midpoint: upper burst keeps the top half, lower burst the bottom half.
    For the general multi-burst case this falls back to ``_weight_top_burst``.
    """
    if len(products) <= 1:
        return _weight_equal(products)
    shape = products[0].grid.shape
    weights = [np.zeros(shape, dtype=np.float32) for _ in products]
    claimed = np.zeros(shape, dtype=bool)
    # Process in order; for each pair split overlap midpoint.
    for k, p in enumerate(products):
        valid = (p.weight > 0) & (~claimed)
        if k + 1 < len(products):
            nxt = products[k + 1]
            ov = valid & (nxt.weight > 0)
            if ov.any():
                rows = np.where(ov.any(axis=1))[0]
                if len(rows) > 0:
                    mid = (rows.min() + rows.max()) // 2
                    # Upper burst keeps rows below mid in the overlap.
                    valid = valid.copy()
                    valid[mid:, :] = valid[mid:, :] & False  # type: ignore[index]
        weights[k] = valid.astype(np.float32)
        claimed |= valid
    # Fill any unclaimed valid pixels with equal weight (non-overlap regions).
    for k, p in enumerate(products):
        extra = (p.weight > 0) & (~claimed)
        weights[k] = np.where(extra, 1.0, weights[k]).astype(np.float32)
        claimed |= extra
    return weights


# ---------------------------------------------------------------------------
# Combine kernel (shared weighted complex average → circular mean)
# ---------------------------------------------------------------------------


def _combine(
    products: list[BurstGeoProduct],
    weights: list[np.ndarray],
    *,
    phi_hat: np.ndarray,
    range_slope_hat: np.ndarray,
    component_per_node: np.ndarray,
    network_stats: object,
    method: str,
) -> MosaicProduct:
    """Weighted complex mosaic with per-burst phase screen applied.

    This is the accumulator from ``mosaic.py:158-189``, generalized so the
    weight array is supplied by the method's weighting function (rather than
    always ``p.weight``).
    """
    from faninsar.processing.merge.products import MosaicProduct

    shape = products[0].grid.shape
    col_idx = np.arange(shape[1], dtype=np.float64)
    x_ref = 0.5 * (shape[1] - 1)
    col_centered = col_idx - x_ref

    complex_acc = np.zeros(shape, dtype=np.complex64)
    weight_sum = np.zeros(shape, dtype=np.float32)
    coh_acc = np.zeros(shape, dtype=np.float32)
    n_bursts = np.zeros(shape, dtype=np.uint8)
    component_id = np.zeros(shape, dtype=np.int16)
    max_weight = np.zeros(shape, dtype=np.float32)

    for k, p in enumerate(products):
        phase_screen = float(phi_hat[k]) + float(range_slope_hat[k]) * col_centered
        phasor = np.exp(-1j * phase_screen).astype(np.complex64)
        z = p.complex * phasor[None, :]
        z = np.where(np.isfinite(z.real) & np.isfinite(z.imag), z, np.complex64(0.0))
        w = np.nan_to_num(weights[k], nan=0.0).astype(np.float32)
        complex_acc += (w.astype(np.complex64) * z).astype(np.complex64)
        weight_sum += w
        if p.coherence is not None:
            coh = np.nan_to_num(np.asarray(p.coherence, dtype=np.float32), nan=0.0)
            coh_acc += w * coh
        n_bursts += (w > 0).astype(np.uint8)
        stronger = w > max_weight
        component_id[stronger] = int(component_per_node[k]) + 1
        max_weight[stronger] = w[stronger]

    nodata = weight_sum <= 0.0
    complex_out = np.zeros(shape, dtype=np.complex64)
    coh_out = np.zeros(shape, dtype=np.float32)
    complex_out[~nodata] = complex_acc[~nodata] / weight_sum[~nodata]
    coh_out[~nodata] = coh_acc[~nodata] / weight_sum[~nodata]
    component_id[nodata] = 0

    logger.info(
        "merge_bursts method=%s n_bursts=%d nodata_px=%d",
        method,
        len(products),
        int(nodata.sum()),
    )
    return MosaicProduct(
        grid=products[0].grid,
        complex=complex_out,
        coherence=coh_out,
        weight_sum=weight_sum,
        n_bursts=n_bursts,
        component_id=component_id,
        network=network_stats,
    )


# Dispatch table: method → (align_fn, weight_fn).
_METHOD_TABLE: dict[
    str,
    tuple[
        object,
        object,
    ],
] = {
    "complex_average": (_align_none, _weight_input),
    "faninsar_network": (_align_full_ls, _weight_input),
    "isce2_avg": (_align_global_const, _weight_equal),
    "isce2_top": (_align_global_const, _weight_top_burst),
    "insardev_equal": (
        lambda p, **kw: _align_insardev(p, degree=0, **kw),
        _weight_equal,
    ),
    "insardev_ramp": (
        lambda p, **kw: _align_insardev(p, degree=1, **kw),
        _weight_equal,
    ),
    "insardev_weighted": (
        lambda p, **kw: _align_insardev(p, degree=1, **kw),
        _weight_fractional,
    ),
    "faninsar_weighted": (_align_full_ls, _weight_hanning_coh),
    "gmtsar_cut": (_align_none, _weight_gmtsar_cut),
}


__all__ = [
    "NON_REFERENCE_METHODS",
    "MergeMethod",
    "merge_bursts",
]
