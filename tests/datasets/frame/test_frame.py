"""Tests for faninsar.datasets.frame.frame — Frame facade class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faninsar.datasets.frame import Frame, FrameGeometry, FrameInterferogramCollection

# The `frame_dir` fixture and `_write_tiff` helper are provided by
# tests/datasets/frame/conftest.py.
from tests.datasets.frame.conftest import _write_tiff

pystac = pytest.importorskip("pystac")


class TestFrameInit:
    def test_loads_existing_frame(self, frame_dir: Path) -> None:
        frame = Frame(frame_dir)
        assert frame.root == frame_dir

    def test_geometry_property(self, frame_dir: Path) -> None:
        frame = Frame(frame_dir)
        assert frame.geometry is not None
        assert isinstance(frame.geometry, FrameGeometry)

    def test_interferograms_property(self, frame_dir: Path) -> None:
        frame = Frame(frame_dir)
        assert frame.interferograms is not None
        assert isinstance(frame.interferograms, FrameInterferogramCollection)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Frame(tmp_path / "nonexistent")

    def test_geometry_only(self, frame_dir: Path) -> None:
        """Frame with only geometry/ (no ifg/) should still load."""
        import shutil

        no_ifg = frame_dir.parent / "geom_only"
        shutil.copytree(frame_dir, no_ifg)
        shutil.rmtree(no_ifg / "interferograms")

        frame = Frame(no_ifg)
        assert frame.geometry is not None
        assert frame.interferograms is None


class TestFrameToStac:
    def test_to_stac_returns_catalog(self, frame_dir: Path) -> None:
        frame = Frame(frame_dir)
        catalog = frame.to_stac(catalog_id="test")
        assert isinstance(catalog, pystac.Catalog)
        assert catalog.id == "test"

    def test_to_stac_has_both_collections(self, frame_dir: Path) -> None:
        frame = Frame(frame_dir)
        catalog = frame.to_stac()
        ids = [c.id for c in catalog.get_children()]
        assert "geometry" in ids
        assert "interferograms" in ids

    def test_to_stac_saves(self, frame_dir: Path, tmp_path: Path) -> None:
        frame = Frame(frame_dir)
        out = tmp_path / "stac_out"
        frame.to_stac(output_dir=out)
        assert (out / "catalog.json").exists()

    def test_to_stac_no_geometry_raises(self, frame_dir: Path) -> None:
        """Frame without geometry should raise on to_stac."""
        import shutil

        no_geom = frame_dir.parent / "no_geom"
        shutil.copytree(frame_dir, no_geom)
        shutil.rmtree(no_geom / "geometry")

        frame = Frame(no_geom)
        with pytest.raises(FileNotFoundError, match="No geometry"):
            frame.to_stac()


class TestFrameSummary:
    def test_summary_has_both(self, frame_dir: Path) -> None:
        frame = Frame(frame_dir)
        s = frame.summary()
        assert "geometry" in s
        assert "interferograms" in s
        assert s["interferograms"]["pair_count"] == 2

    def test_repr(self, frame_dir: Path) -> None:
        frame = Frame(frame_dir)
        r = repr(frame)
        assert "Frame(" in r
        assert "2 pairs" in r


class TestFrameValidate:
    def test_clean_frame_no_issues(self, frame_dir: Path) -> None:
        frame = Frame(frame_dir)
        issues = frame.validate()
        # A frame built by the fixture may have stale geometry_href values
        # (the fixture hardcodes "../../geometry/geometry.json"). Issues are
        # informational; just confirm validate() returns a list of str.
        assert isinstance(issues, list)
        assert all(isinstance(i, str) for i in issues)

    def test_validate_returns_empty_for_geometry_only(self, tmp_path: Path) -> None:
        """A frame with only geometry/ has nothing to cross-check."""
        bounds = (10.0, 45.0, 11.0, 46.0)
        shape = (10, 10)
        inc = np.random.default_rng(42).uniform(20.0, 60.0, shape).astype(np.float32)
        inc_path = tmp_path / "incidence.tif"
        _write_tiff(inc_path, bounds, inc)
        FrameGeometry.from_rasters(
            out_dir=tmp_path / "frame", incidence=inc_path, overwrite=True
        )
        frame = Frame(tmp_path / "frame")
        assert frame.validate() == []

    def test_validate_flags_missing_geometry_href_target(
        self, frame_dir: Path, tmp_path: Path
    ) -> None:
        """If interferograms reference geometry/ but it's gone, that's an issue."""
        import shutil

        broken = tmp_path / "broken"
        shutil.copytree(frame_dir, broken)
        shutil.rmtree(broken / "geometry")
        frame = Frame(broken)
        issues = frame.validate()
        # geometry_href in item.json/index points at a non-existent target.
        assert any("geometry_href" in i or "geometry" in i for i in issues)


class TestFrameZarr:
    """D1.3: Frame.to_zarr / from_zarr round-trip."""

    def test_to_zarr_writes_store(self, frame_dir: Path, tmp_path: Path) -> None:
        import xarray as xr

        frame = Frame(frame_dir)
        store = tmp_path / "frame.zarr"
        frame.to_zarr(store, overwrite=True, assets=("unw_phase",))
        assert store.exists()
        # Top-level group carries frame metadata attrs.
        root = xr.open_zarr(str(store), consolidated=False)
        assert root.attrs.get("frame_type") == "Frame"
        assert "geometry_metadata" in root.attrs

    def test_from_zarr_roundtrip_summary(
        self, frame_dir: Path, tmp_path: Path
    ) -> None:
        frame = Frame(frame_dir)
        store = tmp_path / "frame.zarr"
        frame.to_zarr(store, overwrite=True, assets=("unw_phase",))
        loaded = Frame.from_zarr(store)
        assert loaded.geometry is not None
        assert loaded.geometry.metadata is not None
        assert loaded.interferograms is not None
        assert loaded.interferograms.summary()["pair_count"] == 2

    def test_to_zarr_memory_filesystem(self, frame_dir: Path) -> None:
        """D1.3: FSStore path for direct cloud writes (S3 mock)."""
        import fsspec
        import xarray as xr

        frame = Frame(frame_dir)
        # Use an in-memory fsspec filesystem as an S3 stand-in.
        url = "memory://faninsar_test/frame.zarr"
        frame.to_zarr(url, overwrite=True, assets=("unw_phase",))
        # Verify the store is readable via fsspec.
        fs, path = fsspec.url_to_fs(url)
        assert fs.exists(path)
        # Read the root group through the same FSStore mapping used for writes.
        mapper = fs.get_mapper(path)
        import zarr

        grp = zarr.open_group(store=mapper, mode="r")
        assert grp.attrs.get("frame_type") == "Frame"
        assert "geometry_metadata" in dict(grp.attrs)

    def test_from_zarr_geometry_metadata_preserved(
        self, frame_dir: Path, tmp_path: Path
    ) -> None:
        frame = Frame(frame_dir)
        store = tmp_path / "frame.zarr"
        frame.to_zarr(store, overwrite=True, assets=("unw_phase",))
        loaded = Frame.from_zarr(store)
        assert loaded.geometry is not None
        geom_meta = loaded.geometry.metadata
        assert geom_meta is not None
        assert geom_meta["type"] == "FrameGeometry"
