"""P0030 native-tile sampling must keep 128-row block edges finite."""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.io import MemoryFile

from faninsar.processing.geometry.providers.p0030_adapter import _sample_dataset


def _point_source(*, height: int = 200, width: int = 80) -> MemoryFile:
    """In-memory Point GLO-30-like raster whose value is the source row index."""
    spacing = 1.0 / 3600.0
    transform = Affine(spacing, 0.0, -91.0, 0.0, -spacing, 0.0)
    values = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, None],
        (height, width),
    ).copy()
    memory = MemoryFile()
    with memory.open(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.update_tags(AREA_OR_POINT="Point")
        dataset.write(values, 1)
    return memory


def test_sample_dataset_keeps_128_block_last_row_when_source_row_is_integer() -> None:
    """A 128-row block whose max source index is an integer must still sample.

    ``_sample_biquintic`` reads taps ``floor(row)-1 .. floor(row)+4``. The
    per-block read window used ``ceil(max)+3``, which is one pixel short when
    ``max`` is an integer — the last target row of every 128-row chunk became
    NaN even though the Copernicus tile was finite. That is the 1-row stripe
    baked into managed ``dem.tif`` mosaics.
    """
    spacing = 1.0 / 3600.0
    target = Affine(spacing, 0.0, -91.0 - 0.5 * spacing, 0.0, -spacing, 0.5 * spacing)
    height, width = 130, 40
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
    )
    xs, ys = target * (columns, rows)
    with _point_source() as memory, memory.open() as dataset:
        sampled = _sample_dataset(
            dataset,
            np.asarray(xs),
            np.asarray(ys),
            target_crs="EPSG:4326",
            source_crs="EPSG:4326",
            source_nodata=None,
        )
    # Target row r maps to source row r exactly; row 127 is the 128-block edge.
    interior = sampled[:, 2:-2]
    assert np.isfinite(interior[126]).all()
    assert np.isfinite(interior[127]).all()
    assert np.isfinite(interior[128]).all()
    np.testing.assert_allclose(interior[127], 127.0, atol=1e-4)
