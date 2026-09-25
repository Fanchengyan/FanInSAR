"""Regression: native geometry DEM view buckets must stay finite.

The private WGS84 execution view of a projected ``RasterDEM`` spans the
lon/lat bounding box, whose corner triangles fall outside the projected
footprint.  Sampling there is out-of-grid and returns NaN, but the native
kernels require a finite DEM bucket, so the view seam must fill void cells.
"""

from __future__ import annotations

import numpy as np

from faninsar.processing.geometry.api import RasterDEM
from faninsar.processing.geometry.grids import GridSpec
from faninsar.processing.geometry.prepare_production import (
    _public_raster_dem_native_arrays,
)


def test_projected_dem_view_bucket_is_finite() -> None:
    """Corner-triangle voids of a projected DEM must not reach the kernels."""
    size = 256
    grid = GridSpec(
        "EPSG:32633",
        (30.0, 0.0, 320067.0, 0.0, -30.0, 4284852.0),
        shape=(size, size),
    )
    dem = RasterDEM(
        array=np.full((size, size), 123.0, dtype=np.float32),
        grid=grid,
        vertical_datum="ellipsoidal",
    )
    values, metadata, bounds = _public_raster_dem_native_arrays(dem)
    assert values.shape == (size, size)
    assert np.isfinite(values).all()
    assert np.isfinite(metadata).all()
    assert np.isfinite(bounds).all()
