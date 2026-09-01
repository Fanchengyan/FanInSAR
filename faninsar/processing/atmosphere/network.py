"""Date-level weighted least-squares inversion of ionospheric screens.

This module is the FanInSAR port of the alosStack ``ion_ls.py`` date-level
network step (PROPOSAL-0036). Per-pair ionospheric phase screens are
combined into per-acquisition screens with a fixed reference date whose
screen is identically zero, exactly like the upstream ``zro_date`` term.

Semantics preserved from the upstream implementation:

- The observation model for one pair is ``screen[pair] = ion[primary] -
  ion[secondary]``; the reference-date column is removed before solving.
- Window-size reciprocal weighting (``-ww``): each pair observation is
  weighted by ``1 / window`` where ``window`` is the smoothing window size
  used while filtering that pair's screen. Larger windows mean less
  confident screens.
- A ``matrix_rank``-style connectivity gate rejects networks whose
  observation matrix does not have full column rank before any pixel is
  solved.

Deliberate deviations (documented for reviewers):

- Excluded dates drop every pair touching them. The upstream active code
  only drops a pair when BOTH endpoints are excluded, contradicting its own
  CLI help text ("pairs involving these dates are excluded") and the
  commented-out list comprehension directly above it; the documented intent
  is implemented here.
- An optional robust screening loop (off with ``screening_iterations=0``,
  which reproduces exact upstream behaviour) iteratively down-weights the
  influence of pairwise residuals whose standardized magnitude exceeds a
  threshold (Huber-style IRLS). Pairs whose median relative weight falls
  below one half are reported in ``screened_pair_ids``; no pair is ever
  removed, so the network connectivity validated before the solve is
  preserved.
- Pixels whose normal-equation matrix is numerically singular, or that are
  not covered by enough valid pairs, are returned as NaN instead of
  raising, so one bad region cannot discard a whole stack.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = setup_logger(__name__)

_MAD_SCALE = 1.4826
_SINGULAR_LOGDET_DROP = 40.0


@dataclass(frozen=True, slots=True)
class IonosphereNetworkResult:
    """Per-acquisition ionospheric screens from one network inversion.

    Attributes
    ----------
    dates : tuple[str, ...]
        Acquisition identifiers, sorted ascending. ``screens`` rows align
        with this tuple.
    reference_date : str
        Date whose screen is identically zero (the ``zro_date`` analog).
    screens : numpy.ndarray
        ``(len(dates), rows, cols)`` float64 screens in radians. Unconstrained
        or singular pixels are NaN; the reference row is exactly zero.
    used_pair_ids : tuple[str, ...]
        Pair identifiers (``"primary_secondary"``) that survived exclusion
        and screening and entered the final solve.
    screened_pair_ids : tuple[str, ...]
        Pair identifiers whose median relative weight fell below one half
        under robust screening (strongly down-weighted observations).
    observation_rank : int
        Rank of the final observation matrix (equal to ``len(dates) - 1``
        for a connected network).
    screening_iterations : int
        Number of screening rounds actually executed.

    """

    dates: tuple[str, ...]
    reference_date: str
    screens: np.ndarray
    used_pair_ids: tuple[str, ...]
    screened_pair_ids: tuple[str, ...]
    observation_rank: int
    screening_iterations: int


def _pair_dates(pair_id: str) -> tuple[str, str]:
    """Split one ``primary_secondary`` identifier into its dates."""
    parts = pair_id.split("_")
    if len(parts) != 2 or not all(parts):
        message = f"pair identifier must be 'primary_secondary', got {pair_id!r}"
        raise ValueError(message)
    return parts[0], parts[1]


def _observation_matrix(
    pair_ids: tuple[str, ...],
    dates: tuple[str, ...],
    reference_date: str,
) -> torch.Tensor:
    """Build the upstream ``H0`` matrix with the reference column removed."""
    estimated = [date for date in dates if date != reference_date]
    column = {date: index for index, date in enumerate(estimated)}
    matrix = torch.zeros(len(pair_ids), len(estimated), dtype=torch.float64)
    for row, pair_id in enumerate(pair_ids):
        primary, secondary = _pair_dates(pair_id)
        if primary != reference_date:
            matrix[row, column[primary]] = 1.0
        if secondary != reference_date:
            matrix[row, column[secondary]] = -1.0
    return matrix


def _matrix_rank(matrix: torch.Tensor) -> int:
    """Torch equivalent of :func:`numpy.linalg.matrix_rank`."""
    singular_values = torch.linalg.svdvals(matrix)
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps
    tolerance = tolerance * float(singular_values.max())
    return int(torch.count_nonzero(singular_values > tolerance))


def _network_dates(pair_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the sorted unique dates covered by the given pairs."""
    dates = {date for pair_id in pair_ids for date in _pair_dates(pair_id)}
    return tuple(sorted(dates))


def invert_ionosphere_network(
    pair_screens: Mapping[str, np.ndarray],
    pair_windows: Mapping[str, np.ndarray] | None = None,
    *,
    reference_date: str | None = None,
    excluded_dates: Sequence[str] = (),
    excluded_pairs: Sequence[str] = (),
    screening_iterations: int = 6,
    screening_threshold: float = 3.0,
    device: str = "cpu",
) -> IonosphereNetworkResult:
    """Invert per-pair ionospheric screens into per-date screens.

    Parameters
    ----------
    pair_screens : mapping[str, numpy.ndarray]
        Per-pair ionospheric screens in radians keyed by
        ``"primary_secondary"`` identifiers, all sharing one 2-D grid.
    pair_windows : mapping[str, numpy.ndarray], optional
        Per-pair smoothing window sizes (the alosStack ``win`` layers).
        Each observation is weighted by ``1 / window``; NaN or
        non-positive window pixels exclude that pair at that pixel. When
        omitted every valid screen pixel carries weight one.
    reference_date : str, optional
        Date whose screen is pinned to zero. Defaults to the earliest
        date in the used network (the upstream ``zro_date`` default).
    excluded_dates : sequence of str, optional
        Every pair touching one of these dates is dropped before the
        connectivity gate (the documented upstream ``exc_date`` intent).
    excluded_pairs : sequence of str, optional
        Pair identifiers dropped before the connectivity gate (the
        upstream ``exc_pair``).
    screening_iterations : int, optional
        Maximum robust screening rounds. ``0`` reproduces exact alosStack
        behaviour (window weighting only).
    screening_threshold : float, optional
        Standardized residual magnitude at which a pair's influence
        begins to decay under Huber-style reweighting.
    device : str, optional
        Torch device hosting the solve. The result is always NumPy.

    Returns
    -------
    IonosphereNetworkResult
        Per-date screens plus the exact pair provenance of the solve.

    Raises
    ------
    ValueError
        On empty networks, mismatched grids, unknown exclusion or
        reference dates, or an observation matrix that does not have full
        column rank after exclusions and screening.

    """
    screens_input = {str(key): np.asarray(value) for key, value in pair_screens.items()}
    if not screens_input:
        message = "ionosphere network inversion requires at least one pair"
        raise ValueError(message)
    shapes = {array.shape for array in screens_input.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        message = "pair screens must be 2-D arrays sharing one shape"
        raise ValueError(message)

    excluded_pairs_set = {str(pair) for pair in excluded_pairs}
    for pair_id in excluded_pairs_set:
        if pair_id not in screens_input:
            message = f"excluded pair is not part of the network: {pair_id}"
            raise ValueError(message)
    excluded_dates_set = {str(date) for date in excluded_dates}
    network_dates = _network_dates(tuple(screens_input))
    unknown_dates = excluded_dates_set - set(network_dates)
    if unknown_dates:
        message = f"excluded dates are not part of the network: {sorted(unknown_dates)}"
        raise ValueError(message)

    selected = [
        pair_id
        for pair_id in screens_input
        if pair_id not in excluded_pairs_set
        and not (set(_pair_dates(pair_id)) & excluded_dates_set)
    ]
    if not selected:
        message = "pair selection removed every pair; the network cannot be inverted"
        raise ValueError(message)

    pair_ids_used = tuple(selected)
    dates = _network_dates(pair_ids_used)
    if reference_date is None:
        reference_date = dates[0]
    elif reference_date not in dates:
        message = f"reference date {reference_date!r} is not covered by the used pairs"
        raise ValueError(message)
    pair_ids_used, observation_rank = _require_connected(
        pair_ids_used, dates, reference_date
    )

    screens_out, screened, iterations = _solve_network(
        pair_ids_used,
        screens_input,
        pair_windows,
        dates,
        reference_date,
        iterations=screening_iterations,
        threshold=screening_threshold,
        device=device,
    )
    return IonosphereNetworkResult(
        dates=dates,
        reference_date=reference_date,
        screens=screens_out,
        used_pair_ids=pair_ids_used,
        screened_pair_ids=screened,
        observation_rank=observation_rank,
        screening_iterations=iterations,
    )


def _require_connected(
    pair_ids: tuple[str, ...],
    dates: tuple[str, ...],
    reference_date: str,
) -> tuple[tuple[str, ...], int]:
    """Enforce the upstream full-column-rank gate on the observation matrix."""
    matrix = _observation_matrix(pair_ids, dates, reference_date)
    rank = _matrix_rank(matrix)
    if rank < len(dates) - 1:
        message = (
            "dates to be estimated are not fully connected by the pairs used in "
            f"least squares: rank={rank}, dates={len(dates) - 1}"
        )
        raise ValueError(message)
    return pair_ids, rank


def _stack_pair_maps(
    pair_ids: tuple[str, ...],
    screens_input: Mapping[str, np.ndarray],
    pair_windows: Mapping[str, np.ndarray] | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack screens and reciprocal-window weights onto one device."""
    screen_rows = []
    weight_rows = []
    for pair_id in pair_ids:
        screen = torch.as_tensor(
            np.asarray(screens_input[pair_id], dtype=np.float64)
        ).reshape(1, -1)
        if pair_windows is None:
            weights = torch.ones_like(screen)
        else:
            window = torch.as_tensor(
                np.asarray(pair_windows[pair_id], dtype=np.float64)
            ).reshape(1, -1)
            valid = torch.isfinite(window) & (window > 0.0)
            weights = torch.where(valid, 1.0 / window.clamp(min=1e-30), torch.zeros(()))
        weights = weights * torch.isfinite(screen)
        screen_rows.append(screen)
        weight_rows.append(weights)
    screens = torch.cat(screen_rows, dim=0).to(device)
    weights = torch.cat(weight_rows, dim=0).to(device)
    return screens, weights


def _solve_pixels(
    design: torch.Tensor,
    screens: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Solve every pixel of one already-stacked flat network.

    Returns a ``(len(dates) - 1, npixels)`` solution with NaN at pixels
    whose normal equations are numerically singular.
    """
    normal = torch.einsum("pd,pe,pn->den", design, design, weights)
    rhs = torch.einsum("pd,pn->dn", design, screens * weights)
    sign, logabsdet = torch.linalg.slogdet(normal.movedim(-1, 0))
    solvable = (sign > 0) & (
        logabsdet > (float(logabsdet.max()) - _SINGULAR_LOGDET_DROP)
    )
    solution = torch.full(
        (design.shape[1], screens.shape[1]),
        float("nan"),
        dtype=torch.float64,
        device=screens.device,
    )
    if bool(solvable.any()):
        indices = torch.nonzero(solvable, as_tuple=False).squeeze(-1)
        selected_normal = normal.movedim(-1, 0)[indices]
        selected_rhs = rhs.movedim(-1, 0)[indices]
        solutions = torch.linalg.solve(selected_normal, selected_rhs)
        solution[:, indices] = solutions.T
    return solution


def _solve_network(
    pair_ids: tuple[str, ...],
    screens_input: Mapping[str, np.ndarray],
    pair_windows: Mapping[str, np.ndarray] | None,
    dates: tuple[str, ...],
    reference_date: str,
    *,
    iterations: int,
    threshold: float,
    device: str,
) -> tuple[np.ndarray, tuple[str, ...], int]:
    """Run the IRLS-screened batched solve and restore the image grid."""
    screens, weights = _stack_pair_maps(pair_ids, screens_input, pair_windows, device)
    design = _observation_matrix(pair_ids, dates, reference_date)
    effective = weights
    for _ in range(max(int(iterations), 0)):
        solution = _solve_pixels(design, screens, effective)
        model = design @ solution
        residual = (screens - model) * torch.sqrt(effective)
        # Standardize residuals across pairs at each pixel; observations
        # whose standardized residual exceeds the threshold lose influence
        # smoothly (Huber-style), never hard.
        median_pixel = residual.nanmedian(dim=0).values
        mad_pixel = (residual - median_pixel).abs().nanmedian(dim=0).values
        z = (residual - median_pixel).abs() / (_MAD_SCALE * mad_pixel + 1e-12)
        huber = 1.0 / (1.0 + (z / threshold) ** 2)
        effective = weights * torch.nan_to_num(huber, nan=0.0)
    solution = _solve_pixels(design, screens, effective)
    relative = torch.where(
        weights > 0.0,
        effective / weights.clamp(min=1e-300),
        torch.full((), float("nan")),
    )
    median_relative = relative.nanmedian(dim=1).values
    screened = tuple(
        pair_ids[row]
        for row in range(len(pair_ids))
        if torch.isfinite(median_relative[row]) and float(median_relative[row]) < 0.5
    )
    rows, cols = next(iter(screens_input.values())).shape
    estimated = solution.reshape(len(dates) - 1, rows, cols).cpu().numpy()
    reference_index = dates.index(reference_date)
    return (
        np.insert(estimated, reference_index, 0.0, axis=0),
        screened,
        max(int(iterations), 0),
    )


__all__ = ["IonosphereNetworkResult", "invert_ionosphere_network"]
