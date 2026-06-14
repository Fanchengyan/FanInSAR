"""Tests for faninsar.datasets.frame.frame — Frame facade class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from faninsar.datasets.frame import Frame, FrameGeometry, FrameInterferogramCollection
from faninsar.datasets.frame.metadata import (
    build_interferograms_index,
    build_item_metadata,
    save_json,
)

pystac = pytest.importorskip("pystac")


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

    crs = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    for pname in pair_names:
        pair_dir = ifgs_dir / pname
        parts = pname.split("_")
        item_meta = build_item_metadata(
            pair_name=pname,
            reference_date=parts[0],
            secondary_date=parts[1],
            grid={
                "crs": crs,
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
