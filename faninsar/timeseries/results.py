"""Time-series inversion over unwrapped pair products (NSBAS/SBAS hook)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class SpatialMetadata:
    """Structured coordinate identity carried by a time-series result.

    Parameters
    ----------
    domain : {"radar", "geo"}
        Coordinate domain of the arrays.
    shape : tuple[int, int]
        Spatial ``(height, width)`` of every raster band.
    grid_identity : str
        Stable identity of the source coordinate grid.
    crs : str, optional
        CRS required for geographic products.
    transform : tuple[float, ...], optional
        Affine transform coefficients required for geographic products.
    radar_coordinates : dict[str, Any], optional
        Domain-specific radar coordinate metadata.

    """

    domain: Literal["radar", "geo"]
    shape: tuple[int, int]
    grid_identity: str
    crs: str | None = None
    transform: tuple[float, ...] | None = None
    radar_coordinates: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate the coordinate identity without guessing missing fields."""
        if self.domain not in {"radar", "geo"}:
            message = "spatial domain must be 'radar' or 'geo'"
            raise ValueError(message)
        if len(self.shape) != 2 or any(size <= 0 for size in self.shape):
            message = "spatial shape must contain two positive dimensions"
            raise ValueError(message)
        if not isinstance(self.grid_identity, str) or not self.grid_identity.strip():
            message = "spatial grid_identity must be a non-empty string"
            raise ValueError(message)
        if self.domain == "geo" and (
            not isinstance(self.crs, str)
            or not self.crs.strip()
            or self.transform is None
            or len(self.transform) != 6
        ):
            message = "geo spatial metadata requires CRS and six transform values"
            raise ValueError(message)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible structured representation."""
        payload: dict[str, Any] = {
            "domain": self.domain,
            "shape": list(self.shape),
            "grid_identity": self.grid_identity,
        }
        if self.crs is not None:
            payload["crs"] = self.crs
        if self.transform is not None:
            payload["transform"] = list(self.transform)
        if self.radar_coordinates is not None:
            payload["radar_coordinates"] = dict(self.radar_coordinates)
        return payload


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """Phase time series and optional LOS displacement from SBAS.

    Attributes
    ----------
    pair_ids : tuple[str, ...]
        Pair identifiers aligned with ``residual_phase_pairs_rad``.
    dates : tuple[str, ...]
        Acquisition dates aligned with ``phase_cumulative_rad``.
    phase_increments_rad : numpy.ndarray
        Incremental phase in radians between consecutive dates.
    residual_phase_pairs_rad : numpy.ndarray
        Pair residual phase in radians. Missing observations and pixels whose
        finite pair subnetwork is rank deficient are ``NaN``.
    phase_cumulative_rad : numpy.ndarray
        Cumulative phase in radians, referenced to the first date.
    displacement_increments_m : numpy.ndarray or None
        Incremental LOS displacement in metres when a wavelength was supplied.
    displacement_cumulative_m : numpy.ndarray or None
        Cumulative LOS displacement in metres when a wavelength was supplied.
    metadata : dict[str, Any]
        Product semantics and inversion diagnostics.
    canonical_path : pathlib.Path, optional
        Canonical STAC+Zarr product root when the result has been persisted.

    """

    pair_ids: tuple[str, ...]
    dates: tuple[str, ...]
    phase_increments_rad: np.ndarray
    residual_phase_pairs_rad: np.ndarray
    phase_cumulative_rad: np.ndarray
    metadata: dict[str, Any]
    displacement_increments_m: np.ndarray | None = None
    displacement_cumulative_m: np.ndarray | None = None
    revision_id: str | None = None
    spatial: SpatialMetadata | None = None
    canonical_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate the optional source Network revision identity."""
        if self.revision_id is not None and (
            not isinstance(self.revision_id, str) or not self.revision_id.strip()
        ):
            message = "revision_id must be a non-empty string or None"
            raise ValueError(message)
        if self.canonical_path is not None and not isinstance(
            self.canonical_path, Path
        ):
            message = "canonical_path must be a pathlib.Path or None"
            logger.error(message)
            raise ValueError(message)

    def export(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
        exporter: object | None = None,
    ) -> object:
        """Export this result through the time-series exporter seam.

        Parameters
        ----------
        path : str or pathlib.Path
            Complete destination product root.
        overwrite : bool, optional
            Replace an existing destination when ``True``.
        exporter : str, type, or callable, optional
            Registered exporter name, zero-argument exporter class, or a
            configured callable. ``None`` selects the default exporter.

        Returns
        -------
        object
            Exporter-defined result, normally the destination path.

        """
        from faninsar.timeseries.export import export_timeseries

        return export_timeseries(self, path, overwrite=overwrite, exporter=exporter)


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
) -> TimeSeries:
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
    TimeSeries
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
    return TimeSeries(
        pair_ids=pair_ids,
        dates=dates,
        phase_increments_rad=increments,
        residual_phase_pairs_rad=residual,
        phase_cumulative_rad=cumulative,
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
