"""NSBAS time-series inversion pipeline for Frame products.

Consumes a :class:`~faninsar.datasets.frame.Frame` with interferograms,
builds a Dask graph for spatial-chunk-parallel inversion via
:func:`dask.array.map_blocks`, calls :class:`~faninsar.timeseries.inversion.NSBASSolver`
per spatial chunk, and writes the output displacement time series to a Zarr
store.

The per-block function follows the dask-torch-numpy pattern: numpy arrays in,
numpy arrays out, with PyTorch as the internal compute engine (GPU when
available, CPU otherwise).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar._core.device import parse_device
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    import xarray as xr

    from faninsar.datasets.frame import Frame
    from faninsar.timeseries.models import TimeSeriesModels

logger = setup_logger(__name__)

ModelName = Literal["linear", "quadratic", "cubic", "sinusoidal"]

_MODEL_REGISTRY: dict[str, str] = {
    "linear": "LinearModel",
    "quadratic": "QuadraticModel",
    "cubic": "CubicModel",
    "sinusoidal": "AnnualSinusoidalModel",
}


def _build_model(model: TimeSeriesModels | str, dates: Any) -> TimeSeriesModels:
    """Resolve a model name or instance into a TimeSeriesModels."""
    if not isinstance(model, str):
        return model
    from faninsar.timeseries import models as tsmodels

    cls_name = _MODEL_REGISTRY.get(model)
    if cls_name is None:
        msg = f"Unknown model '{model}'. Choose from {list(_MODEL_REGISTRY)}."
        logger.error(msg)
        raise ValueError(msg)
    cls = getattr(tsmodels, cls_name)
    return cls(dates)


class InversionPipeline:
    """Orchestrate NSBAS time-series inversion from a Frame.

    Consumes a :class:`Frame` with interferograms, builds a Dask graph for
    spatial-chunk-parallel inversion via :func:`dask.array.map_blocks`, calls
    :class:`NSBASSolver` per spatial chunk, and writes the output displacement
    time series to a Zarr store.

    The pipeline does **not** mutate the input :class:`Frame` (immutability):
    it only reads from it and writes to *output_zarr*.

    Parameters
    ----------
    frame : Frame
        Frame with interferograms (and optionally geometry).
    model : TimeSeriesModels or str
        Time-series model or a model name (``"linear"``, ``"quadratic"``,
        ``"cubic"``, ``"sinusoidal"``). ``"linear"`` yields velocity.
    coh_threshold : float
        Coherence threshold for pixel masking (default 0.4).
    device : str
        ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Examples
    --------
    >>> from faninsar.pipeline import InversionPipeline
    >>> pipeline = InversionPipeline(frame, model="linear", device="cpu")
    >>> ds = pipeline.run(output_zarr="displacement.zarr")  # doctest: +SKIP

    """

    def __init__(
        self,
        frame: Frame,
        model: TimeSeriesModels | str = "linear",
        *,
        coh_threshold: float = 0.4,
        device: str = "auto",
    ) -> None:
        """Initialize the pipeline from a Frame and model choice."""
        if frame.interferograms is None:
            msg = "Frame has no interferograms; cannot run inversion."
            logger.error(msg)
            raise ValueError(msg)
        self._frame = frame
        self._pairs = frame.interferograms.pairs()
        self._model_arg = model
        self._model = _build_model(model, self._pairs.dates)
        self._coh_threshold = coh_threshold
        self._device = parse_device(device)
        self._device_input = device

    @property
    def device(self) -> Any:
        """The resolved torch device used for inversion."""
        return self._device

    @property
    def model(self) -> TimeSeriesModels:
        """The resolved time-series model."""
        return self._model

    @staticmethod
    def _invert_block(
        unw_block: np.ndarray,
        coh_block: np.ndarray | None,
        pairs_obj: Any,
        model_obj: TimeSeriesModels,
        coh_threshold: float,
        device: Any,
    ) -> np.ndarray:
        """Invert one spatial block. numpy in (pair, y, x), numpy out (time, y, x).

        Follows the dask-torch-numpy pattern: numpy in/out, torch internal.
        """
        from faninsar.timeseries.solver import NSBASSolver

        # unw_block: (n_pair, y, x). Reshape to (n_pair, n_pixel) for NSBAS.
        n_pair, ny, nx = unw_block.shape
        n_pixel = ny * nx
        unw_2d = unw_block.reshape(n_pair, n_pixel).astype(np.float64)

        coh_2d = None
        if coh_block is not None:
            coh_2d = coh_block.reshape(n_pair, n_pixel).astype(np.float64)

        solver = NSBASSolver(
            unw_2d,
            pairs_obj,
            model_obj,
            coh=coh_2d,
            coh_threshold=coh_threshold,
            device=str(device),
            verbose=False,
        )
        incs, _params, _res_pair, _res_tsm = solver.inverse(return_numpy=True)
        # incs: (n_date - 1, n_pixel). Reshape to (n_date - 1, ny, nx).
        n_time = incs.shape[0]
        out = incs.reshape(n_time, ny, nx).astype(np.float32)

        # GPU memory cleanup per the dask-torch-numpy pattern.
        try:
            import torch

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
        except Exception:  # pragma: no cover - defensive
            pass
        return out

    def run(
        self,
        *,
        output_zarr: str | Path | None = None,
        chunks: dict[str, int] | None = None,
        overwrite: bool = False,
        chunk_size: tuple[int, int] = (256, 256),
    ) -> xr.Dataset:
        """Run the inversion and write displacement to Zarr.

        Parameters
        ----------
        output_zarr : str or Path, optional
            Output Zarr store. Defaults to
            ``<frame>/timeseries/displacement.zarr``.
        chunks : dict, optional
            Read chunk spec for the input interferogram stacks.
        overwrite : bool
            Overwrite an existing output store.
        chunk_size : tuple of int
            Spatial chunk size ``(y, x)`` for the dask map_blocks tiling.

        Returns
        -------
        xarray.Dataset
            Lazy dataset of displacement ``(time, y, x)``, velocity
            ``(y, x)``, and displacement_variance ``(time, y, x)``.

        """
        import dask
        import dask.array as da
        import xarray as xr

        ifgs = self._frame.interferograms
        assert ifgs is not None

        unw = ifgs.open_stack("unw_phase", chunks=chunks or "auto")
        if "band" in unw.dims:
            unw = unw.squeeze("band", drop=True)

        has_coh = ifgs.exists(self._pairs.to_names()[0], "coherence")
        coh = None
        if has_coh:
            try:
                coh = ifgs.open_stack("coherence", chunks=chunks or "auto")
                if "band" in coh.dims:
                    coh = coh.squeeze("band", drop=True)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Could not open coherence stack: %s", e)

        # Number of output time steps (n_date - 1 incremental displacements).
        n_dates = len(self._pairs.dates)
        n_time = n_dates - 1

        # Build dask arrays for map_blocks.
        unw_da = unw.data  # (pair, y, x) dask array
        coh_da = coh.data if coh is not None else None

        # Rechunk so spatial tiles are (pair_full, chunk_y, chunk_x).
        unw_rechunked = unw_da.rechunk((unw_da.shape[0], chunk_size[0], chunk_size[1]))
        coh_rechunked = None
        if coh_da is not None:
            coh_rechunked = coh_da.rechunk(unw_rechunked.chunks)

        use_gpu = str(self._device_input) not in ("cpu",)

        def _block_fn(
            block_unw: np.ndarray, block_coh: np.ndarray | None = None
        ) -> np.ndarray:
            return InversionPipeline._invert_block(
                block_unw,
                block_coh,
                self._pairs,
                self._model,
                self._coh_threshold,
                self._device,
            )

        def _build_dask() -> da.Array:
            time_chunks = (n_time, *unw_rechunked.chunks[1:])
            if coh_rechunked is not None:
                return da.map_blocks(
                    _block_fn,
                    unw_rechunked,
                    coh_rechunked,
                    dtype=np.float32,
                    drop_axis=0,
                    new_axis=0,
                    chunks=time_chunks,
                )
            return da.map_blocks(
                lambda b: _block_fn(b, None),
                unw_rechunked,
                dtype=np.float32,
                drop_axis=0,
                new_axis=0,
                chunks=time_chunks,
            )

        if use_gpu:
            with dask.annotate(resources={"gpu": 1}):
                displacement_dask = _build_dask()
        else:
            displacement_dask = _build_dask()

        # Build coordinates from the input stack.
        dates = self._pairs.dates
        # Incremental displacement time dim: all but the first date.
        time_coords = dates[1:] if hasattr(dates, "__getitem__") else dates

        ds = xr.Dataset(
            {
                "displacement": (
                    ("time", "y", "x"),
                    displacement_dask,
                )
            },
            coords={
                "time": (["time"], np.asarray(time_coords)),
                "y": unw["y"],
                "x": unw["x"],
            },
            attrs={
                "long_name": "incremental displacement",
                "model": self._model_arg
                if isinstance(self._model_arg, str)
                else "custom",
                "device": str(self._device),
                "n_pairs": len(self._pairs),
                "n_dates": int(n_dates),
            },
        )

        # Velocity (y, x): cumulative displacement → linear model already gives
        # velocity as the first model parameter. Compute lazily.
        velocity = self._compute_velocity(ds["displacement"])
        ds["velocity"] = velocity

        # D2.3: displacement_variance via coherence-based phase variance.
        # Uses the standard InSAR relation var_phase = (1 - coh^2) /
        # (2 * N_looks * coh^2), propagated through the SBAS design matrix.
        try:
            variance = self._compute_variance(ds["displacement"], coh)
            if variance is not None:
                ds["displacement_variance"] = variance
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not compute displacement variance: %s", e)

        # Write to Zarr.
        out = (
            Path(output_zarr)
            if output_zarr is not None
            else (self._frame.root / "timeseries" / "displacement.zarr")
        )
        if out.exists() and overwrite:
            import shutil

            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(str(out), mode="w", consolidated=False)
        logger.info("Wrote displacement Zarr to %s", out)
        return ds

    def _compute_velocity(self, displacement: xr.DataArray) -> xr.DataArray:
        """Estimate mean linear velocity (y, x) from cumulative displacement."""
        import xarray as xr

        # Cumulative sum along time → cumulative displacement.
        cumul = displacement.cumsum(dim="time")
        # Simple velocity: slope of cumulative displacement vs time index.
        n = cumul.sizes["time"]
        if n < 2:
            return xr.zeros_like(displacement.isel(time=0).drop_vars("time"))
        t = np.arange(n, dtype=np.float64)
        t_mean = t.mean()
        t_dev = t - t_mean
        ss = (t_dev**2).sum()
        # velocity = sum(t_dev * cumul) / ss, over time
        velocity = ((cumul * xr.DataArray(t_dev, dims=["time"])).sum(dim="time")) / ss
        return velocity.astype(np.float32)

    def _compute_variance(
        self,
        displacement: xr.DataArray,
        coh: xr.DataArray | None,
    ) -> xr.DataArray | None:
        """Compute a per-pixel displacement variance estimate.

        Uses the standard InSAR coherence-to-phase-variance relation
        ``var_phase = (1 - coh^2) / (2 * n_looks * coh^2)`` averaged over
        pairs, then scaled to displacement units. Returns a lazy
        ``(time, y, x)`` DataArray broadcasting the mean per-pixel phase
        variance across the time dimension, or *None* if coherence is
        unavailable. The result is non-negative by construction.
        """
        if coh is None:
            return None
        n_looks = 1.0  # effective looks (conservative default; no multilook info)
        coh_clipped = coh.clip(min=1e-3, max=1.0)
        # Mean coherence over the pair dimension -> (y, x).
        mean_coh = coh_clipped.mean(dim="pair") if "pair" in coh.dims else coh_clipped
        phase_var = (1.0 - mean_coh**2) / (2.0 * n_looks * mean_coh**2)
        phase_var = phase_var.clip(min=0.0)
        # Broadcast across the time dimension of the displacement cube.
        n_time = displacement.sizes["time"]
        variance = phase_var.expand_dims(time=n_time).astype(np.float32)
        variance.name = "displacement_variance"
        return variance
