"""Frame-level time-series product (M2).

Represents the standardised ``frame/timeseries/`` folder produced by an
InSAR time-series inversion (NSBAS, MintPy, etc.). The folder holds:

- ``displacement.zarr``  — displacement time series (Zarr, lazy chunked reads)
- ``velocity.cog.tif``   — mean linear velocity (COG)
- ``velocity_std.cog.tif`` — velocity std-dev (optional)
- ``residual_rms.cog.tif``  — inversion residual RMS (optional)
- ``temporal_coherence.cog.tif`` — quality metric (optional)
- ``timeseries.json``      — product metadata

The implementation is deliberately minimal in M2: it knows how to *read* a
standardised folder and *lazily open* its assets. The actual inversion
pipeline (running NSBAS on a ``FrameInterferogramCollection`` to produce
these files) is out of scope for M2 and will be added later.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import xarray as xr

from faninsar.logging import setup_logger

from .exceptions import (
    MissingInterferogramAssetError,
)
from .metadata import load_json, save_json

logger = setup_logger(__name__)

TimeSeriesAssetName = Literal[
    "velocity",
    "velocity_std",
    "residual_rms",
    "temporal_coherence",
]

TIMESERIES_ASSETS: dict[str, str] = {
    "velocity": "velocity.cog.tif",
    "velocity_std": "velocity_std.cog.tif",
    "residual_rms": "residual_rms.cog.tif",
    "temporal_coherence": "temporal_coherence.cog.tif",
}


class FrameTimeSeries:
    """Standardised ``frame/timeseries/`` folder.

    Parameters
    ----------
    root : str or Path
        Path to the ``timeseries`` directory.

    """

    def __init__(self, root: str | Path) -> None:
        """Initialise from an existing ``timeseries`` directory."""
        self._root = Path(root)
        if not self._root.is_dir():
            msg = f"Time-series directory not found: {self._root}"
            raise FileNotFoundError(msg)

        self._meta_path = self._root / "timeseries.json"
        self._meta: dict[str, Any] | None = None
        if self._meta_path.exists():
            self._meta = load_json(self._meta_path)

    @property
    def root(self) -> Path:
        """Root directory of the time-series product."""
        return self._root

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Parsed ``timeseries.json`` content, or None if absent."""
        return self._meta

    def path(self, name: TimeSeriesAssetName) -> Path:
        """Return the on-disk path of a named asset."""
        return self._root / TIMESERIES_ASSETS[name]

    def exists(self, name: TimeSeriesAssetName) -> bool:
        """Return True if the named asset exists on disk."""
        return self.path(name).exists()

    def displacement_path(self) -> Path:
        """Return the path of ``displacement.zarr`` (may not exist)."""
        return self._root / "displacement.zarr"

    def has_displacement(self) -> bool:
        """Return True if ``displacement.zarr`` exists."""
        return self.displacement_path().exists()

    def open_displacement(self, chunks: Any = "auto") -> xr.Dataset:
        """Lazily open ``displacement.zarr`` as an :class:`xarray.Dataset`.

        Raises
        ------
        FileNotFoundError
            If ``displacement.zarr`` is missing.

        """
        p = self.displacement_path()
        if not p.exists():
            msg = f"displacement.zarr not found at {p}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        return xr.open_zarr(p, chunks=chunks)

    def open(
        self,
        name: TimeSeriesAssetName,
        *,
        masked: bool = True,
        chunks: Any = None,
    ) -> xr.DataArray:
        """Lazily open a scalar asset (velocity / velocity_std / ...) as a DataArray.

        Raises
        ------
        MissingInterferogramAssetError
            If the named asset does not exist.

        """
        p = self.path(name)
        if not p.exists():
            raise MissingInterferogramAssetError(name)
        import rioxarray  # noqa: F401  (registers .rio accessor)

        da = xr.open_dataarray(p, engine="rasterio", chunks=chunks)
        if masked and da.rio.encoded_nodata is not None:
            da = da.where(da != da.rio.encoded_nodata)
        return da

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of the time-series product."""
        meta = self._meta or {}
        return {
            "type": "FrameTimeSeries",
            "root": str(self._root),
            "has_displacement": self.has_displacement(),
            "velocity": self.exists("velocity"),
            "velocity_std": self.exists("velocity_std"),
            "residual_rms": self.exists("residual_rms"),
            "temporal_coherence": self.exists("temporal_coherence"),
            "pair_count": meta.get("pair_count"),
            "date_count": meta.get("date_count"),
            "reference_date": meta.get("reference_date"),
            "method": meta.get("method"),
            "version": meta.get("version"),
        }

    @classmethod
    def from_displacement(
        cls,
        out_dir: str | Path,
        displacement: xr.DataArray | xr.Dataset,
        *,
        velocity: Any = None,
        reference: Any = None,
        method: str = "unknown",
        overwrite: bool = False,
    ) -> FrameTimeSeries:
        """Write a synthetic time-series product from in-memory arrays.

        This is primarily a **test/seed helper** — it writes a
        ``displacement.zarr`` and (optionally) a ``velocity.cog.tif`` so a
        ``FrameTimeSeries`` can be exercised without running a real
        inversion. The real NSBAS-driven pipeline will live in
        ``faninsar.NSBAS`` and call a lower-level writer.

        Parameters
        ----------
        out_dir : str or Path
            Parent dir; ``<out_dir>/timeseries`` is created.
        displacement : xarray.DataArray or Dataset
            Displacement array with a time dimension. Written to
            ``displacement.zarr``.
        velocity : numpy.ndarray, optional
            Mean velocity 2-D array. Written to ``velocity.cog.tif`` if given.
        reference : Any, optional
            Reference GeoGrid/Profile for the velocity COG.
        method : str
            Inversion method label stored in metadata.
        overwrite : bool
            Overwrite existing files.

        """
        out_dir = Path(out_dir)
        ts_dir = out_dir / "timeseries"
        if ts_dir.exists() and not overwrite:
            msg = f"timeseries dir already exists: {ts_dir}"
            raise FileExistsError(msg)
        ts_dir.mkdir(parents=True, exist_ok=True)

        # displacement.zarr
        if isinstance(displacement, xr.DataArray):
            ds = displacement.to_dataset(name="displacement")
        else:
            ds = displacement
        zarr_path = ts_dir / "displacement.zarr"
        ds.to_zarr(str(zarr_path), mode="w")

        # velocity.cog.tif (optional)
        if velocity is not None and reference is not None:
            from .raster_io import write_cog

            write_cog(
                velocity,
                ts_dir / TIMESERIES_ASSETS["velocity"],
                reference,
                overwrite=True,
            )

        # timeseries.json metadata
        dates = (
            displacement["time"].values.tolist() if "time" in displacement.dims else []
        )
        meta = {
            "type": "FrameTimeSeries",
            "version": "0.1.0",
            "method": method,
            "pair_count": None,
            "date_count": len(dates),
            "reference_date": str(dates[0]) if len(dates) else None,
            "assets": {
                "displacement": {"href": "displacement.zarr"},
            },
        }
        if velocity is not None:
            meta["assets"]["velocity"] = {"href": TIMESERIES_ASSETS["velocity"]}
        save_json(meta, ts_dir / "timeseries.json")

        logger.info("FrameTimeSeries written to %s", ts_dir)
        return cls(ts_dir)

    def __repr__(self) -> str:
        """Return a short summary string."""
        n_assets = sum(self.exists(k) for k in TIMESERIES_ASSETS)
        return (
            f"FrameTimeSeries(root={self._root.name!r}, "
            f"displacement={self.has_displacement()}, assets={n_assets})"
        )


__all__ = ["FrameTimeSeries", "TimeSeriesAssetName"]
