"""SBAS inversion and the time-series product (PROPOSAL-0059).

A time-series product is a directory with one ``timeseries.zarr`` group and
an ``item.json`` (kind ``timeseries``) written through
:mod:`faninsar.io.storage.product`. Its inputs are the ``write_id`` of every
unwrap (or ionosphere-corrected unwrap) product it consumed, keyed by Pair.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.io.storage.product import commit_product, open_product
from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.timeseries.results import SpatialMetadata, TimeSeries

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = setup_logger(__name__)

TIMESERIES_KIND = "timeseries"
_ZARR_NAME = "timeseries.zarr"
_ARRAY_FIELDS = {
    "increments": "phase_increments_rad",
    "cumulative": "phase_cumulative_rad",
    "residual_pairs": "residual_phase_pairs_rad",
    "displacement_increments_m": "displacement_increments_m",
    "displacement_cumulative_m": "displacement_cumulative_m",
}


def _json_safe_metadata(value: Any) -> Any:
    """Return metadata with NumPy scalar values converted for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe_metadata(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def timeseries_properties(result: TimeSeries) -> dict[str, Any]:
    """Return the STAC properties describing one time series.

    Parameters
    ----------
    result : TimeSeries
        Scientific result.

    Returns
    -------
    dict
        Temporal extent, Pair and date axes, metadata, and spatial grid.

    """
    properties: dict[str, Any] = {
        "start_datetime": f"{result.dates[0][:4]}-{result.dates[0][4:6]}-"
        f"{result.dates[0][6:]}T00:00:00Z",
        "end_datetime": f"{result.dates[-1][:4]}-{result.dates[-1][4:6]}-"
        f"{result.dates[-1][6:]}T00:00:00Z",
        "faninsar:pair_ids": list(result.pair_ids),
        "faninsar:dates": list(result.dates),
    }
    metadata = _json_safe_metadata(dict(result.metadata))
    if metadata:
        properties["faninsar:metadata"] = metadata
    if result.spatial is not None:
        spatial = _json_safe_metadata(result.spatial.to_dict())
        properties["faninsar:domain"] = spatial["domain"]
        properties["faninsar:shape"] = spatial["shape"]
        properties["faninsar:grid_identity"] = spatial["grid_identity"]
        for key in ("crs", "transform", "radar_coordinates"):
            if key in spatial:
                properties[f"faninsar:{key}"] = spatial[key]
    return properties


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
        Torch device for :class:`~faninsar.timeseries.solver.NSBASSolver`.
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


def write_timeseries_product(
    result: TimeSeries,
    path: str | Path,
    *,
    inputs: Mapping[str, str] | None = None,
    properties: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Commit one time-series product.

    Parameters
    ----------
    result : TimeSeries
        Scientific result to persist.
    path : str or pathlib.Path
        Product directory; it holds ``timeseries.zarr`` and ``item.json``.
    inputs : mapping[str, str], optional
        Pair id to the ``write_id`` of the unwrap product it consumed.
    properties : mapping, optional
        Extra ``faninsar:`` Item properties (for example the phase source).
    overwrite : bool, optional
        Replace an existing product.

    Returns
    -------
    pathlib.Path
        The product directory.

    Raises
    ------
    InvalidProcessingStateError
        If the product exists and ``overwrite`` is false.

    """
    import zarr

    arrays = {
        name: np.asarray(getattr(result, field))
        for name, field in _ARRAY_FIELDS.items()
        if getattr(result, field) is not None
    }
    attrs: dict[str, Any] = {
        "pair_ids": list(result.pair_ids),
        "dates": list(result.dates),
        "metadata": _json_safe_metadata(result.metadata),
    }
    if result.spatial is not None:
        attrs["spatial"] = result.spatial.to_dict()

    def write(staging: Path) -> dict[str, dict[str, Any]]:
        group = zarr.open_group(str(staging / _ZARR_NAME), mode="w")
        for name, array in arrays.items():
            group.create_array(name, data=array, chunks=array.shape)
        group.attrs.update(attrs)
        return {"timeseries": {"href": _ZARR_NAME, "type": "application/vnd+zarr"}}

    product = commit_product(
        path,
        kind=TIMESERIES_KIND,
        properties={**timeseries_properties(result), **dict(properties or {})},
        write_payloads=write,
        inputs=dict(inputs or {}),
        overwrite=overwrite,
    )
    logger.info("Committed time-series product %s", product.path)
    return product.path


def read_timeseries(path: str | Path) -> TimeSeries:
    """Read one committed time-series product.

    Parameters
    ----------
    path : str or pathlib.Path
        Product directory written by :func:`write_timeseries_product`.

    Returns
    -------
    TimeSeries
        The persisted result with ``canonical_path`` set to ``path``.

    Raises
    ------
    InvalidProcessingStateError
        If the product is missing or its arrays are incomplete.

    """
    import zarr

    product = open_product(path, kind=TIMESERIES_KIND)
    group = zarr.open_group(str(product.asset_path("timeseries")), mode="r")
    attrs = dict(group.attrs)
    values: dict[str, np.ndarray] = {}
    for name, field in _ARRAY_FIELDS.items():
        if name in group:
            values[field] = np.asarray(group[name][...])
    missing = [
        field
        for field in (
            "phase_increments_rad",
            "phase_cumulative_rad",
            "residual_phase_pairs_rad",
        )
        if field not in values
    ]
    if missing:
        reject_invalid_state(f"time-series product is missing arrays: {missing}")
    spatial = attrs.get("spatial")
    return TimeSeries(
        pair_ids=tuple(str(value) for value in attrs.get("pair_ids", [])),
        dates=tuple(str(value) for value in attrs.get("dates", [])),
        metadata=dict(attrs.get("metadata", {})),
        spatial=(
            None
            if spatial is None
            else SpatialMetadata(
                domain=spatial["domain"],
                shape=tuple(int(size) for size in spatial["shape"]),
                grid_identity=spatial["grid_identity"],
                crs=spatial.get("crs"),
                transform=(
                    None
                    if spatial.get("transform") is None
                    else tuple(float(value) for value in spatial["transform"])
                ),
                radar_coordinates=spatial.get("radar_coordinates"),
            )
        ),
        canonical_path=product.path,
        **values,
    )


__all__ = [
    "TIMESERIES_KIND",
    "invert_unwrapped_pairs",
    "read_timeseries",
    "timeseries_properties",
    "write_timeseries_product",
]
