"""Tests for publish (Hugging Face) helpers and RemoteFrame logic (M5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from faninsar.datasets.frame import (
    Frame,
    RemoteFrame,
    build_upload_plan,
    iter_frame_assets,
    publish_to_huggingface,
)


def _write_tiff(path: Path, bounds: tuple, arr: np.ndarray) -> None:
    crs = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    h, w = arr.shape
    transform = from_bounds(*bounds, w, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype=arr.dtype, crs=crs, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(arr, 1)


class TestIterFrameAssets:
    def test_skips_hidden_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.tif").write_bytes(b"x")
        (tmp_path / ".hidden").write_bytes(b"y")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.tif").write_bytes(b"z")
        files = list(iter_frame_assets(tmp_path))
        names = [p.name for p in files]
        assert "a.tif" in names
        assert "b.tif" in names
        assert ".hidden" not in names

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert list(iter_frame_assets(tmp_path)) == []


class TestBuildUploadPlan:
    def test_relative_paths(self, tmp_path: Path) -> None:
        (tmp_path / "geometry").mkdir()
        (tmp_path / "geometry" / "incidence.cog.tif").write_bytes(b"")
        plan = build_upload_plan(tmp_path)
        assert len(plan) == 1
        assert plan[0]["path_in_repo"] == "geometry/incidence.cog.tif"


class TestPublishToHuggingface:
    def test_dry_run_returns_plan(self, tmp_path: Path) -> None:
        (tmp_path / "geometry").mkdir()
        (tmp_path / "geometry" / "incidence.cog.tif").write_bytes(b"")
        result = publish_to_huggingface(tmp_path, "user/repo", dry_run=True)
        assert result["dry_run"] is True
        assert result["uploaded"] is False
        assert result["file_count"] == 1
        assert result["repo_id"] == "user/repo"

    def test_missing_frame_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            publish_to_huggingface(tmp_path / "nope", "user/repo", dry_run=True)

    def test_real_upload_without_hf_raises(self, tmp_path: Path) -> None:
        (tmp_path / "a.tif").write_bytes(b"")
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError):
                publish_to_huggingface(tmp_path, "user/repo", dry_run=False)
        else:
            pytest.skip("huggingface_hub is installed; cannot test ImportError path")


# ---- RemoteFrame logic tests ----
# Build a synthetic local STAC catalog on disk, serve it via the file:// URL,
# and verify RemoteFrame parses it correctly. (No HTTP server required.)

pystac = pytest.importorskip("pystac")


@pytest.fixture
def remote_catalog_url(tmp_path: Path) -> str:
    """Build a local STAC catalog with geometry + interferograms collections."""
    import datetime

    from pystac import Asset, Catalog, Collection, Extent, Item, MediaType
    from shapely.geometry import box as shapely_box

    bounds = (10.0, 45.0, 11.0, 46.0)
    arr = np.random.default_rng(42).uniform(20, 60, (4, 4)).astype(np.float32)
    geom_dir = tmp_path / "geometry"
    geom_dir.mkdir()
    _write_tiff(geom_dir / "incidence.cog.tif", bounds, arr)

    ifgs_dir = tmp_path / "interferograms"
    ifgs_dir.mkdir()
    pair_dir = ifgs_dir / "20191115_20200314"
    pair_dir.mkdir()
    _write_tiff(pair_dir / "unw_phase.cog.tif", bounds, arr)

    geom_shape = shapely_box(*bounds).__geo_interface__
    bbox = list(bounds)
    now = datetime.datetime.now(datetime.timezone.utc)
    spatial = pystac.SpatialExtent(bboxes=[bbox])
    temporal = pystac.TemporalExtent(intervals=[[now, now]])
    extent = Extent(spatial=spatial, temporal=temporal)

    geom_item = Item(
        id="geometry",
        geometry=geom_shape,
        bbox=bbox,
        datetime=now,
        properties={"frame:type": "FrameGeometry"},
    )
    geom_item.add_asset(
        "incidence",
        Asset(href=str(geom_dir / "incidence.cog.tif"), media_type=MediaType.GEOTIFF),
    )
    geom_col = Collection(id="geometry", description="geom", extent=extent)
    geom_col.add_item(geom_item)

    ifg_item = Item(
        id="20191115_20200314",
        geometry=geom_shape,
        bbox=bbox,
        datetime=now,
        properties={"frame:type": "FrameInterferogramItem"},
    )
    ifg_item.add_asset(
        "unw_phase",
        Asset(href=str(pair_dir / "unw_phase.cog.tif"), media_type=MediaType.GEOTIFF),
    )
    ifg_extent = Extent(spatial=spatial, temporal=temporal)
    ifg_col = Collection(id="interferograms", description="ifgs", extent=ifg_extent)
    ifg_col.add_item(ifg_item)

    catalog = Catalog(id="test", description="test")
    catalog.add_child(geom_col)
    catalog.add_child(ifg_col)

    catalog.normalize_hrefs(str(tmp_path / "stac"))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return (tmp_path / "stac" / "catalog.json").as_uri()


class TestRemoteFrame:
    def test_loads_remote_catalog(self, remote_catalog_url: str) -> None:
        rf = RemoteFrame(remote_catalog_url, anonymous=True)
        assert rf.geometry_item is not None
        assert len(rf.interferogram_items) == 1

    def test_summary(self, remote_catalog_url: str) -> None:
        rf = RemoteFrame(remote_catalog_url)
        s = rf.summary()
        assert "incidence" in s["geometry_asset_keys"]
        assert s["interferogram_pair_count"] == 1
        assert s["interferogram_pair_names"] == ["20191115_20200314"]

    def test_repr(self, remote_catalog_url: str) -> None:
        rf = RemoteFrame(remote_catalog_url)
        r = repr(rf)
        assert "RemoteFrame" in r
        assert "pairs=1" in r

    def test_missing_pystac_raises(self, monkeypatch, remote_catalog_url: str) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pystac":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pystac is required"):
            RemoteFrame(remote_catalog_url)

    def test_open_remote_returns_remote_frame(self, remote_catalog_url: str) -> None:
        rf = Frame.open_remote(remote_catalog_url)
        assert isinstance(rf, RemoteFrame)

    def test_asset_url_resolution(self, remote_catalog_url: str) -> None:
        rf = RemoteFrame(remote_catalog_url)
        urls = rf._item_asset_urls(rf.geometry_item)
        assert "incidence" in urls
        assert "://" in urls["incidence"]
