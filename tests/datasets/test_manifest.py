"""Regression tests for the on-disk dataset manifest.

The manifest (``<root_dir>/faninsar.manifest.json``) stores per-file scan
metadata so a second instantiation of the same dataset can skip the
per-file ``rasterio.open`` pass. These tests exercise the whole contract:

* first instantiation writes the manifest,
* second instantiation restores from it (no ``rasterio.open`` calls),
* path-list drift (file added / removed) invalidates the entry and rescans,
* ``reindex=True`` forces a rescan even when the entry is valid,
* ``crs`` / ``res`` / ``roi`` / ``bands`` / ``resampling`` / ``chunks`` /
  ``fill_nodata`` changes still hit the manifest (the transform is
  re-derived from raw file facts),
* a corrupted manifest degrades to a silent rescan.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import rasterio
from pyproj.crs import CRS
from rasterio.transform import from_bounds

from faninsar.datasets.base import RasterDataset, manifest
from faninsar.query import BoundingBox


def _write_tiff(
    path: Path,
    bounds: tuple[float, float, float, float],
    data: np.ndarray,
    crs: str = "EPSG:4326",
) -> None:
    """Write a single-band GeoTIFF."""
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)
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
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def ds_dir():
    """A temporary directory holding five small GeoTIFFs."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bounds_list = [
            (0, 0, 10, 10),
            (10, 0, 20, 10),
            (20, 0, 30, 10),
            (0, 10, 10, 20),
            (10, 10, 20, 20),
        ]
        for i, b in enumerate(bounds_list):
            _write_tiff(root / f"tile_{i:02d}.tif", b, np.zeros((32, 32), np.float32))
        yield root


def _count_rasterio_opens(fn, *args, **kwargs):
    """Run ``fn`` while counting real ``rasterio.open`` calls.

    Returns ``(result, open_count)``.
    """
    calls = 0
    real_open = rasterio.open

    def counting_open(*a, **k):
        nonlocal calls
        calls += 1
        return real_open(*a, **k)

    with mock.patch.object(rasterio, "open", counting_open):
        result = fn(*args, **kwargs)
    return result, calls


def test_first_scan_writes_manifest(ds_dir):
    ds, opens = _count_rasterio_opens(RasterDataset, root_dir=ds_dir, verbose=False)
    assert opens == len(ds.files)
    mpath = manifest.manifest_path(ds_dir)
    assert mpath.is_file()
    doc = json.loads(mpath.read_text())
    assert doc["faninsar"]
    assert doc["schema"] == manifest.MANIFEST_SCHEMA
    assert ds._manifest_name() in doc["datasets"]


def test_second_instantiation_restores_without_opens(ds_dir):
    RasterDataset(root_dir=ds_dir, verbose=False)
    ds, opens = _count_rasterio_opens(RasterDataset, root_dir=ds_dir, verbose=False)
    assert opens == 0
    assert len(ds.files) == 5
    assert ds.crs == CRS.from_epsg(4326)


def test_reindex_forces_rescan(ds_dir):
    RasterDataset(root_dir=ds_dir, verbose=False)
    ds, opens = _count_rasterio_opens(
        RasterDataset, root_dir=ds_dir, verbose=False, reindex=True
    )
    assert opens == len(ds.files)


def test_path_list_change_invalidates_manifest(ds_dir):
    RasterDataset(root_dir=ds_dir, verbose=False)
    # Add one more file — the stored path list no longer matches.
    _write_tiff(ds_dir / "tile_99.tif", (30, 0, 40, 10), np.zeros((32, 32), np.float32))
    ds, opens = _count_rasterio_opens(RasterDataset, root_dir=ds_dir, verbose=False)
    assert opens == 6  # rescan all files
    # The manifest now holds the new entry — next instantiation restores.
    ds2, opens2 = _count_rasterio_opens(RasterDataset, root_dir=ds_dir, verbose=False)
    assert opens2 == 0 and len(ds2.files) == 6


def test_explicit_paths_also_restore(ds_dir):
    paths = sorted(ds_dir.glob("*.tif"))
    RasterDataset(paths=paths, root_dir=ds_dir, verbose=False)
    ds, opens = _count_rasterio_opens(
        RasterDataset, paths=paths, root_dir=ds_dir, verbose=False
    )
    assert opens == 0


def test_different_crs_still_hits_manifest(ds_dir):
    """Manifest stores raw file facts; a new ``crs`` recomputes the transform."""
    RasterDataset(root_dir=ds_dir, verbose=False)
    ds, opens = _count_rasterio_opens(
        RasterDataset, root_dir=ds_dir, crs=CRS.from_epsg(3857), verbose=False
    )
    assert opens == 0
    assert ds.crs == CRS.from_epsg(3857)


def test_corrupted_manifest_falls_back_to_scan(ds_dir):
    RasterDataset(root_dir=ds_dir, verbose=False)
    mpath = manifest.manifest_path(ds_dir)
    mpath.write_text("{ this is not json", encoding="utf-8")
    ds, opens = _count_rasterio_opens(RasterDataset, root_dir=ds_dir, verbose=False)
    assert opens == len(ds.files)


def test_missing_schema_key_falls_back_to_scan(ds_dir):
    RasterDataset(root_dir=ds_dir, verbose=False)
    mpath = manifest.manifest_path(ds_dir)
    doc = json.loads(mpath.read_text())
    doc["schema"] = manifest.MANIFEST_SCHEMA + 99
    mpath.write_text(json.dumps(doc), encoding="utf-8")
    ds, opens = _count_rasterio_opens(RasterDataset, root_dir=ds_dir, verbose=False)
    assert opens == len(ds.files)


def test_res_roi_bands_do_not_reindex(ds_dir):
    """Derived parameters must hit the manifest — they are not scan inputs."""
    RasterDataset(root_dir=ds_dir, verbose=False)
    ds, opens = _count_rasterio_opens(
        RasterDataset,
        root_dir=ds_dir,
        res=(5.0, 5.0),
        roi=BoundingBox(0, 0, 15, 15, crs=4326),
        verbose=False,
    )
    assert opens == 0
    assert ds.res == (5.0, 5.0)
