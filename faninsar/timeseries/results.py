"""Time-series inversion over unwrapped pair products (NSBAS/SBAS hook)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class TimeSeriesResult:
    """Phase time series and optional LOS displacement from SBAS.

    Attributes
    ----------
    pair_ids : tuple[str, ...]
        Pair identifiers aligned with ``residual_phase_pairs_rad``.
    dates : tuple[str, ...]
        Acquisition dates aligned with ``phase_cumulative_rad``.
    increments : numpy.ndarray
        Incremental phase in radians between consecutive dates. Prefer the
        explicit ``phase_increments_rad`` property in new code.
    residual_pairs : numpy.ndarray
        Pair residual phase in radians. Missing observations and pixels whose
        finite pair subnetwork is rank deficient are ``NaN``. Prefer the
        explicit ``residual_phase_pairs_rad`` property in new code.
    cumulative : numpy.ndarray
        Cumulative phase in radians, referenced to the first date. Prefer the
        explicit ``phase_cumulative_rad`` property in new code.
    displacement_increments_m : numpy.ndarray or None
        Incremental LOS displacement in metres when a wavelength was supplied.
    displacement_cumulative_m : numpy.ndarray or None
        Cumulative LOS displacement in metres when a wavelength was supplied.
    metadata : dict[str, Any]
        Product semantics and inversion diagnostics.

    Notes
    -----
    The legacy field names are retained so existing callers can still construct
    and consume the result. Their values have always been phase, not metres.

    """

    pair_ids: tuple[str, ...]
    dates: tuple[str, ...]
    increments: np.ndarray
    residual_pairs: np.ndarray
    cumulative: np.ndarray
    metadata: dict[str, Any]
    displacement_increments_m: np.ndarray | None = None
    displacement_cumulative_m: np.ndarray | None = None
    revision_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the optional source Network revision identity."""
        if self.revision_id is not None and (
            not isinstance(self.revision_id, str) or not self.revision_id.strip()
        ):
            message = "revision_id must be a non-empty string or None"
            raise ValueError(message)

    @property
    def phase_increments_rad(self) -> np.ndarray:
        """Return incremental phase explicitly identified as radians."""
        return self.increments

    @property
    def residual_phase_pairs_rad(self) -> np.ndarray:
        """Return pair residual phase explicitly identified as radians."""
        return self.residual_pairs

    @property
    def phase_cumulative_rad(self) -> np.ndarray:
        """Return cumulative phase explicitly identified as radians."""
        return self.cumulative


def _solve_connected_pixels(
    observations: np.ndarray,
    design_matrix: np.ndarray,
    *,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve only pixels whose finite pair subnetwork has full column rank.

    Parameters
    ----------
    observations : numpy.ndarray
        Pair phase matrix with shape ``(n_pairs, n_pixels)``.
    design_matrix : numpy.ndarray
        SBAS design matrix with shape ``(n_pairs, n_dates - 1)``.
    device : str or None
        Torch execution device.

    Returns
    -------
    increments : numpy.ndarray
        Phase increments with shape ``(n_dates - 1, n_pixels)``.
    residual : numpy.ndarray
        Pair residuals with shape ``(n_pairs, n_pixels)``.
    valid_pixel_mask : numpy.ndarray
        Boolean mask identifying pixels with a connected finite subnetwork.

    """
    import torch

    from faninsar.processing.runtime.device import parse_device

    n_pairs, n_pixels = observations.shape
    n_increments = design_matrix.shape[1]
    increments = np.full((n_increments, n_pixels), np.nan, dtype=np.float64)
    residual = np.full((n_pairs, n_pixels), np.nan, dtype=np.float64)
    valid_pixel_mask = np.zeros(n_pixels, dtype=bool)
    finite_by_pixel = np.isfinite(observations).T
    unique_masks, inverse = np.unique(finite_by_pixel, axis=0, return_inverse=True)
    torch_device = parse_device(device)

    for mask_index, finite_pairs in enumerate(unique_masks):
        pixel_indices = np.flatnonzero(inverse == mask_index)
        if pixel_indices.size == 0 or np.count_nonzero(finite_pairs) < n_increments:
            continue
        local_design = design_matrix[finite_pairs]
        if np.linalg.matrix_rank(local_design) != n_increments:
            continue

        local_observations = observations[finite_pairs][:, pixel_indices]
        design_tensor = torch.as_tensor(
            local_design,
            dtype=torch.float64,
            device=torch_device,
        )
        observation_tensor = torch.as_tensor(
            local_observations,
            dtype=torch.float64,
            device=torch_device,
        )
        solution = torch.linalg.lstsq(design_tensor, observation_tensor).solution
        local_increments = solution.detach().cpu().numpy()
        increments[:, pixel_indices] = local_increments
        predicted = local_design @ local_increments
        residual[np.ix_(finite_pairs, pixel_indices)] = local_observations - predicted
        valid_pixel_mask[pixel_indices] = True

    return increments, residual, valid_pixel_mask


def invert_unwrapped_pairs(
    pair_phases: dict[str, np.ndarray],
    *,
    device: str | None = "cpu",
    gamma: float = 1e-4,
    wavelength_m: float | None = None,
) -> TimeSeriesResult:
    """Invert a redundant unwrapped pair network with SBAS (no model term).

    Parameters
    ----------
    pair_phases : dict[str, numpy.ndarray]
        Mapping of pair id ``YYYYMMDD_YYYYMMDD`` to unwrapped phase arrays on a
        common grid.
    device : str, optional
        Torch device for :class:`~faninsar.timeseries.inversion.NSBASSolver`.
    gamma : float, optional
        Unused for pure SBAS (model is ``None``); retained for API stability.
    wavelength_m : float, optional
        Radar wavelength in metres. When supplied, phase is converted to LOS
        displacement using ``-wavelength_m / (4 * pi)`` for the
        ``primary * conj(secondary)`` interferogram convention.

    Returns
    -------
    TimeSeriesResult
        Incremental and cumulative phase, plus optional LOS displacement, on
        the pair grid. Pixels without a connected finite pair subnetwork are
        masked with ``NaN`` in every time-series epoch.

    """
    _ = gamma
    if not pair_phases:
        reject_invalid_state("pair_phases must not be empty")
    if wavelength_m is not None and (
        not np.isfinite(wavelength_m) or wavelength_m <= 0.0
    ):
        reject_invalid_state("wavelength_m must be finite and positive")
    pair_ids = tuple(sorted(pair_phases))
    shapes = {array.shape for array in pair_phases.values()}
    if len(shapes) != 1:
        reject_invalid_state("all pair phases must share the same grid shape")
    if len(next(iter(shapes))) != 2:
        reject_invalid_state("pair phase arrays must be two-dimensional")
    height, width = next(iter(shapes))
    n_pixels = height * width

    from faninsar import Pairs

    pairs = Pairs.from_names(list(pair_ids))
    unw = np.stack(
        [
            np.asarray(pair_phases[pid], dtype=np.float64).reshape(-1)
            for pid in pair_ids
        ],
        axis=0,
    )
    if unw.shape != (len(pair_ids), n_pixels):
        reject_invalid_state("failed to stack pair phases into (n_pair, n_pixel)")
    design_matrix = np.asarray(pairs.sbas_matrix(), dtype=np.float64)
    n_increments = len(pairs.dates) - 1
    if design_matrix.shape != (len(pair_ids), n_increments):
        reject_invalid_state("SBAS design matrix has an unexpected shape")
    if np.linalg.matrix_rank(design_matrix) != n_increments:
        reject_invalid_state("pair network is disconnected or rank deficient")

    solved_increments, solved_residual, valid_pixels = _solve_connected_pixels(
        unw,
        design_matrix,
        device=device,
    )
    increments = solved_increments.astype(np.float32).reshape(-1, height, width)
    residual = solved_residual.astype(np.float32).reshape(
        len(pair_ids),
        height,
        width,
    )
    cumulative = np.concatenate(
        [np.zeros((1, height, width), dtype=np.float32), np.cumsum(increments, axis=0)],
        axis=0,
    )
    cumulative.reshape(len(pairs.dates), -1)[:, ~valid_pixels] = np.nan
    displacement_increments = None
    displacement_cumulative = None
    if wavelength_m is not None:
        phase_to_displacement = -float(wavelength_m) / (4.0 * np.pi)
        displacement_increments = increments * phase_to_displacement
        displacement_cumulative = cumulative * phase_to_displacement
    dates = tuple(str(d.date()).replace("-", "") for d in pairs.dates)
    logger.info(
        "Inverted %s pairs over %sx%s grid (%s dates, %s/%s valid pixels)",
        len(pair_ids),
        height,
        width,
        len(dates),
        int(np.count_nonzero(valid_pixels)),
        n_pixels,
    )
    return TimeSeriesResult(
        pair_ids=pair_ids,
        dates=dates,
        increments=increments,
        residual_pairs=residual,
        cumulative=cumulative,
        metadata={
            "method": "sbas",
            "device": str(device),
            "n_pairs": len(pair_ids),
            "n_dates": len(dates),
            "n_valid_pixels": int(np.count_nonzero(valid_pixels)),
            "n_total_pixels": n_pixels,
            "phase_unit": "radian",
            "interferogram_convention": "primary_times_conjugate_secondary",
            "wavelength_m": wavelength_m,
            "displacement_unit": "metre" if wavelength_m is not None else None,
            "phase_to_displacement_sign": -1 if wavelength_m is not None else None,
        },
        displacement_increments_m=displacement_increments,
        displacement_cumulative_m=displacement_cumulative,
    )
