"""Persist and reload coordinate transform LUTs with residual layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from faninsar.processing.geometry.transforms import TransformResult

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class TransformCacheKey:
    """Identity for a cached forward/inverse transform product."""

    product_id: str
    direction: str
    dem_identity: str
    orbit_source: str
    grid_shape: tuple[int, int]

    def as_path_stem(self) -> str:
        """Return a filesystem-safe stem for the cache directory name."""
        height, width = self.grid_shape
        return (
            f"{self.product_id}_{self.direction}_{height}x{width}_{self.dem_identity}"
        )


def write_transform_cache(
    root: str | Path,
    key: TransformCacheKey,
    result: TransformResult,
) -> Path:
    """Write transform arrays and residuals to a Zarr store.

    Parameters
    ----------
    root : str or pathlib.Path
        Parent directory that will contain one cache store per key.
    key : TransformCacheKey
        Cache identity used as the store directory name.
    result : TransformResult
        Dense transform arrays to persist.

    Returns
    -------
    pathlib.Path
        Path to the written Zarr store.

    """
    import zarr

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    store_path = root_path / f"{key.as_path_stem()}.zarr"
    if store_path.exists():
        import shutil

        shutil.rmtree(store_path)

    root_group = zarr.open_group(str(store_path), mode="w")
    for name in result.fields:
        values = getattr(result, name)
        if values.dtype == np.dtype(bool):
            values = values.astype(np.uint8)
        root_group.create_array(name, data=np.asarray(values), overwrite=True)
    root_group.attrs.update(
        {
            "product_id": key.product_id,
            "direction": key.direction,
            "dem_identity": key.dem_identity,
            "orbit_source": key.orbit_source,
            "grid_shape": list(key.grid_shape),
        }
    )
    logger.info("Wrote transform cache: %s", store_path)
    return store_path


def read_transform_cache(store_path: str | Path) -> tuple[TransformCacheKey, object]:
    """Load a previously written transform cache.

    Parameters
    ----------
    store_path : str or pathlib.Path
        Path to a Zarr transform cache store.

    Returns
    -------
    tuple
        Cache key and a namespace-like object with transform arrays.

    """
    import zarr

    from faninsar.processing.geometry.transforms import TransformResult

    path = Path(store_path)
    if not path.exists():
        message = f"transform cache not found: {path}"
        reject(message, FileNotFoundError)
    group = zarr.open_group(str(path), mode="r")
    attrs = dict(group.attrs)
    key = TransformCacheKey(
        product_id=str(attrs["product_id"]),
        direction=str(attrs["direction"]),
        dem_identity=str(attrs["dem_identity"]),
        orbit_source=str(attrs["orbit_source"]),
        grid_shape=(int(attrs["grid_shape"][0]), int(attrs["grid_shape"][1])),
    )
    direction = str(attrs["direction"])
    fields: dict[str, np.ndarray] = {}
    for name in TransformResult.invalid(1).fields:
        stored = np.asarray(group[name])
        if name in ("converged", "max_iter_exhausted", "boundary_rechecked"):
            stored = stored.astype(bool)
        fields[name] = stored
    result = TransformResult.from_arrays(
        fields,
        operation=direction,
        invalid_mask=~np.isfinite(fields["latitude_deg"]),
    )
    return key, result
