# ruff: noqa: TRY003, EM101, EM102, D107, ARG002
"""Adapters for sampling the pinned EGM geoid resources."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from faninsar.logging import setup_logger

from .fetch import Fetch

logger = setup_logger(__name__)


class GeoidSampler:
    """Point sampler for a fetched geoid artifact."""

    def __init__(
        self, model: str, path: str | Path, *, fetch: Fetch | None = None
    ) -> None:
        self.model = model
        self.path = Path(path)
        self._sampler = self._build_sampler(fetch)

    def _build_sampler(self, fetch: Fetch | None) -> object:
        if self.model == "egm96":
            from faninsar.processing.geometry.egm96 import EGM96Geoid

            return EGM96Geoid(coefficient_path=self.path).sample
        if self.model == "egm2008":
            import rasterio
            from rasterio.transform import rowcol

            dataset = rasterio.open(self.path)

            def sample(
                latitude_deg: np.ndarray, longitude_deg: np.ndarray
            ) -> np.ndarray:
                lat, lon = np.broadcast_arrays(
                    np.asarray(latitude_deg, dtype=np.float64),
                    np.asarray(longitude_deg, dtype=np.float64),
                )
                if np.any(~np.isfinite(lat)) or np.any(~np.isfinite(lon)):
                    raise ValueError("geoid coordinates must be finite")
                if np.any((lat < -90.0) | (lat > 90.0)):
                    raise ValueError("geoid latitude must be within [-90, 90]")
                wrapped = ((lon + 180.0) % 360.0) - 180.0
                rows, cols = rowcol(dataset.transform, wrapped, lat)
                rows = np.asarray(rows, dtype=np.int64)
                cols = np.asarray(cols, dtype=np.int64)
                valid = (
                    (rows >= 0)
                    & (rows < dataset.height)
                    & (cols >= 0)
                    & (cols < dataset.width)
                )
                result = np.full(lat.shape, np.nan, dtype=np.float64)
                if np.any(valid):
                    window = rasterio.windows.Window(
                        int(cols[valid].min()),
                        int(rows[valid].min()),
                        int(cols[valid].max() - cols[valid].min() + 1),
                        int(rows[valid].max() - rows[valid].min() + 1),
                    )
                    values = dataset.read(1, window=window, masked=True)
                    local_rows = rows[valid] - int(window.row_off)
                    local_cols = cols[valid] - int(window.col_off)
                    picked = values[local_rows, local_cols]
                    result[valid] = np.asarray(picked.filled(np.nan), dtype=np.float64)
                return result

            return sample
        raise ValueError(f"unsupported geoid model: {self.model}")

    def sample(
        self, latitude_deg: np.ndarray, longitude_deg: np.ndarray
    ) -> np.ndarray:
        """Return undulation values in metres as float64."""
        return np.asarray(self._sampler(latitude_deg, longitude_deg), dtype=np.float64)


def load_geoid(model: str, *, fetch: Fetch | None = None) -> GeoidSampler:
    """Fetch lazily and construct a sampler for one registered geoid model."""
    resolver = fetch or Fetch()
    return GeoidSampler(model, resolver.fetch(model), fetch=resolver)


__all__ = ["GeoidSampler", "load_geoid"]
