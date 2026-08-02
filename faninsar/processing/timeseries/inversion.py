"""Time-series inversion over unwrapped pair products (NSBAS/SBAS hook)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class TimeSeriesResult:
    """Displacement increments and residual diagnostics from NSBAS/SBAS."""

    pair_ids: tuple[str, ...]
    dates: tuple[str, ...]
    increments: np.ndarray
    residual_pairs: np.ndarray
    cumulative: np.ndarray
    metadata: dict[str, Any]


def invert_unwrapped_pairs(
    pair_phases: dict[str, np.ndarray],
    *,
    device: str | None = "cpu",
    gamma: float = 1e-4,
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

    Returns
    -------
    TimeSeriesResult
        Incremental and cumulative displacement on the pair grid.

    """
    _ = gamma
    if not pair_phases:
        reject_invalid_state("pair_phases must not be empty")
    pair_ids = tuple(sorted(pair_phases))
    shapes = {array.shape for array in pair_phases.values()}
    if len(shapes) != 1:
        reject_invalid_state("all pair phases must share the same grid shape")
    height, width = next(iter(shapes))
    n_pixels = height * width

    from faninsar import Pairs
    from faninsar.timeseries.solver import NSBASSolver

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

    solver = NSBASSolver(
        unw,
        pairs,
        model=None,
        device=device,
        verbose=False,
    )
    incs, _params, residual_pair, _residual_tsm = solver.inverse(return_numpy=True)
    increments = np.asarray(incs, dtype=np.float32).reshape(-1, height, width)
    residual = np.asarray(residual_pair, dtype=np.float32).reshape(
        len(pair_ids),
        height,
        width,
    )
    cumulative = np.concatenate(
        [np.zeros((1, height, width), dtype=np.float32), np.cumsum(increments, axis=0)],
        axis=0,
    )
    dates = tuple(str(d.date()).replace("-", "") for d in pairs.dates)
    logger.info(
        "Inverted %s pairs over %sx%s grid (%s dates)",
        len(pair_ids),
        height,
        width,
        len(dates),
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
        },
    )


def write_timeseries_zarr(result: TimeSeriesResult, store_path: str | Path) -> Path:
    """Persist time-series increments and cumulative displacement to Zarr."""
    import zarr

    path = Path(store_path)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    root.create_array("increments", data=result.increments, overwrite=True)
    root.create_array("cumulative", data=result.cumulative, overwrite=True)
    root.create_array("residual_pairs", data=result.residual_pairs, overwrite=True)
    root.attrs.update(
        {
            "pair_ids": list(result.pair_ids),
            "dates": list(result.dates),
            **{str(k): str(v) for k, v in result.metadata.items()},
        }
    )
    logger.info("Wrote time-series Zarr product: %s", path)
    return path
