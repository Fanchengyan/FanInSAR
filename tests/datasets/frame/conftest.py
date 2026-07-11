"""Shared fixtures for faninsar.datasets.frame tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from faninsar.datasets.frame import FrameGeometry
from faninsar.datasets.frame.metadata import (
    build_interferograms_index,
    build_item_metadata,
    save_json,
)


def _write_tiff(
    path: Path,
    bounds: tuple[float, float, float, float],
    data: np.ndarray,
    nodata: float = -9999.0,
) -> None:
    """Write a single-band GeoTIFF."""
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    crs = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


_WGS84_CRS = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


@pytest.fixture
def frame_dir(tmp_path: Path) -> Path:
    """Create a geometry + interferogram frame from synthetic data."""
    bounds = (10.0, 45.0, 11.0, 46.0)
    shape = (10, 10)

    # Geometry
    inc = np.random.default_rng(42).uniform(20.0, 60.0, shape).astype(np.float32)
    inc_path = tmp_path / "incidence.tif"
    _write_tiff(inc_path, bounds, inc)

    FrameGeometry.from_rasters(
        out_dir=tmp_path / "frame",
        incidence=inc_path,
        overwrite=True,
    )

    # Interferograms
    ifgs_dir = tmp_path / "frame" / "interferograms"
    ifgs_dir.mkdir(parents=True, exist_ok=True)

    pair_names = ["20191115_20200314", "20200314_20200708"]
    for pname in pair_names:
        pair_dir = ifgs_dir / pname
        pair_dir.mkdir(parents=True, exist_ok=True)
        unw = np.random.default_rng(42).uniform(-3.0, 3.0, shape).astype(np.float32)
        coh = np.random.default_rng(42).uniform(0.0, 1.0, shape).astype(np.float32)
        _write_tiff(pair_dir / "unw_phase.cog.tif", bounds, unw)
        _write_tiff(pair_dir / "coherence.cog.tif", bounds, coh, nodata=0.0)

    for pname in pair_names:
        pair_dir = ifgs_dir / pname
        parts = pname.split("_")
        item_meta = build_item_metadata(
            pair_name=pname,
            reference_date=parts[0],
            secondary_date=parts[1],
            grid={
                "crs": _WGS84_CRS,
                "width": shape[1],
                "height": shape[0],
                "transform": [0.1, 0.0, 10.0, 0.0, -0.1, 46.0],
                "bounds": list(bounds),
                "resolution": [0.1, 0.1],
            },
            assets={
                "unw_phase": {
                    "href": "unw_phase.cog.tif",
                    "dtype": "float32",
                    "nodata": -9999.0,
                },
                "coherence": {
                    "href": "coherence.cog.tif",
                    "dtype": "float32",
                    "nodata": 0.0,
                },
            },
            geometry_href="../../geometry/geometry.json",
            temporal_baseline_days=120,
        )
        save_json(item_meta, pair_dir / "item.json")

    index_meta = build_interferograms_index(
        pair_count=2,
        pairs=pair_names,
        assets_by_pair={pname: ["unw_phase", "coherence"] for pname in pair_names},
        common_grid=True,
        geometry_href="../../geometry/geometry.json",
    )
    save_json(index_meta, ifgs_dir / "interferograms_index.json")

    return tmp_path / "frame"
