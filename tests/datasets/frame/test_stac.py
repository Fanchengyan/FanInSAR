"""Tests for faninsar.datasets.frame.stac — STAC catalog generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from faninsar.datasets.frame.geometry import FrameGeometry
from faninsar.datasets.frame.interferogram import FrameInterferogramCollection
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

    inc = np.random.default_rng(42).uniform(20.0, 60.0, shape).astype(np.float32)
    inc_path = tmp_path / "incidence.tif"
    _write_tiff(inc_path, bounds, inc)

    FrameGeometry.from_rasters(
        out_dir=tmp_path / "frame",
        incidence=inc_path,
        overwrite=True,
    )
    return tmp_path / "frame"


@pytest.fixture
def ifg_collection(frame_dir: Path) -> FrameInterferogramCollection:
    """Build a FrameInterferogramCollection from synthetic pair data."""
    bounds = (10.0, 45.0, 11.0, 46.0)
    shape = (10, 10)
    ifgs_dir = frame_dir / "interferograms"
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

    return FrameInterferogramCollection(ifgs_dir)


@pytest.fixture
def full_frame_dir(frame_dir: Path) -> Path:
    """A frame directory with BOTH geometry/ and interferograms/ on disk.

    Unlike ``frame_dir`` (geometry only) and ``ifg_collection`` (returns the
    collection object), this fixture returns the path to a fully-populated
    frame dir, suitable for ``Frame(path)`` round-trip tests.
    """
    bounds = (10.0, 45.0, 11.0, 46.0)
    shape = (10, 10)
    ifgs_dir = frame_dir / "interferograms"
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

    return frame_dir


# Geometry-only STAC

class TestGeometryToStac:
    def test_returns_catalog(self, frame_dir: Path) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        catalog = geom.to_stac(catalog_id="test-catalog")
        assert isinstance(catalog, pystac.Catalog)
        assert catalog.id == "test-catalog"

    def test_has_geometry_collection(self, frame_dir: Path) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        catalog = geom.to_stac()
        children = list(catalog.get_children())
        ids = [c.id for c in children]
        assert "geometry" in ids

    def test_geometry_item_has_assets(
        self, frame_dir: Path
    ) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        catalog = geom.to_stac()
        geom_col = catalog.get_child("geometry")
        assert geom_col is not None
        items = list(geom_col.get_items())
        assert len(items) == 1
        item = items[0]
        assert "incidence" in item.assets

    def test_geometry_item_has_projection(
        self, frame_dir: Path
    ) -> None:
        from pystac.extensions.projection import ProjectionExtension

        geom = FrameGeometry(frame_dir / "geometry")
        catalog = geom.to_stac()
        geom_col = catalog.get_child("geometry")
        item = next(iter(geom_col.get_items()))
        proj = ProjectionExtension.ext(item)
        assert proj.epsg is not None
        assert proj.shape is not None

    def test_saves_to_directory(self, frame_dir: Path, tmp_path: Path) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        out = tmp_path / "stac_out"
        catalog = geom.to_stac(output_dir=out)
        assert (out / "catalog.json").exists()
        assert catalog is not None


# Full STAC (geometry + interferograms)

class TestFullStac:
    def test_returns_catalog_with_both_collections(
        self, frame_dir: Path, ifg_collection: FrameInterferogramCollection
    ) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        catalog = geom.to_stac(ifgs=ifg_collection)
        children = list(catalog.get_children())
        ids = [c.id for c in children]
        assert "geometry" in ids
        assert "interferograms" in ids

    def test_ifg_collection_has_items(
        self, frame_dir: Path, ifg_collection: FrameInterferogramCollection
    ) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        catalog = geom.to_stac(ifgs=ifg_collection)
        ifg_col = catalog.get_child("interferograms")
        assert ifg_col is not None
        items = list(ifg_col.get_items())
        assert len(items) == 2

    def test_ifg_item_has_pair_properties(
        self, frame_dir: Path, ifg_collection: FrameInterferogramCollection
    ) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        catalog = geom.to_stac(ifgs=ifg_collection)
        ifg_col = catalog.get_child("interferograms")
        item = next(iter(ifg_col.get_items()))
        assert "frame:pair_name" in item.properties
        assert "frame:reference_date" in item.properties
        assert "frame:secondary_date" in item.properties

    def test_ifg_item_has_datetime_range(
        self, frame_dir: Path, ifg_collection: FrameInterferogramCollection
    ) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        catalog = geom.to_stac(ifgs=ifg_collection)
        ifg_col = catalog.get_child("interferograms")
        item = next(iter(ifg_col.get_items()))
        assert item.common_metadata.start_datetime is not None
        assert item.common_metadata.end_datetime is not None

    def test_ifg_item_has_assets(
        self, frame_dir: Path, ifg_collection: FrameInterferogramCollection
    ) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        catalog = geom.to_stac(ifgs=ifg_collection)
        ifg_col = catalog.get_child("interferograms")
        item = next(iter(ifg_col.get_items()))
        assert "unw_phase" in item.assets
        assert "coherence" in item.assets

    def test_saves_full_catalog(
        self,
        frame_dir: Path,
        ifg_collection: FrameInterferogramCollection,
        tmp_path: Path,
    ) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        out = tmp_path / "stac_full"
        geom.to_stac(ifgs=ifg_collection, output_dir=out)
        assert (out / "catalog.json").exists()
        # Should have geometry and interferograms subdirectories
        assert (out / "geometry").is_dir()
        assert (out / "interferograms").is_dir()


# Ifg-only STAC

class TestIfgToStac:
    def test_ifg_to_stac(self, ifg_collection: FrameInterferogramCollection) -> None:
        catalog = ifg_collection.to_stac(catalog_id="ifg-only")
        assert isinstance(catalog, pystac.Catalog)
        assert catalog.id == "ifg-only"
        children = list(catalog.get_children())
        ids = [c.id for c in children]
        assert "interferograms" in ids
        # No geometry collection
        assert "geometry" not in ids

    def test_ifg_to_stac_with_geometry(
        self, frame_dir: Path, ifg_collection: FrameInterferogramCollection
    ) -> None:
        geom = FrameGeometry(frame_dir / "geometry")
        catalog = ifg_collection.to_stac(geometry=geom)
        children = list(catalog.get_children())
        ids = [c.id for c in children]
        assert "geometry" in ids
        assert "interferograms" in ids


# Edge cases

class TestStacEdgeCases:
    def test_no_geometry_no_ifgs_raises(self, frame_dir: Path) -> None:
        from faninsar.datasets.frame.stac import build_stac_catalog

        with pytest.raises(ValueError, match="At least one"):
            build_stac_catalog()

    def test_custom_temporal_extent(
        self, frame_dir: Path
    ) -> None:
        from datetime import UTC, datetime

        geom = FrameGeometry(frame_dir / "geometry")
        start = datetime(2015, 1, 1, tzinfo=UTC)
        end = datetime(2025, 12, 31, tzinfo=UTC)
        catalog = geom.to_stac(temporal_extent=(start, end))
        geom_col = catalog.get_child("geometry")
        interval = geom_col.extent.temporal.intervals[0]
        assert interval[0] == start
        assert interval[1] == end


# Frame.from_stac round-trip

class TestFrameFromStac:
    """Frame.from_stac() reconstructs a Frame from a to_stac() catalog."""

    def test_missing_catalog_raises(self, tmp_path: Path) -> None:
        from faninsar.datasets.frame import Frame

        with pytest.raises(FileNotFoundError):
            Frame.from_stac(tmp_path / "nonexistent_catalog.json")

    def test_round_trip_summary_equivalence(
        self, full_frame_dir: Path, tmp_path: Path
    ) -> None:
        """to_stac -> from_stac yields a Frame with equivalent summary."""
        from faninsar.datasets.frame import Frame

        original = Frame(full_frame_dir)
        original_summary = original.summary()

        out_dir = tmp_path / "stac_out"
        original.to_stac(output_dir=out_dir)
        catalog_json = out_dir / "catalog.json"
        assert catalog_json.exists()

        # Pass frame_root explicitly because the STAC catalog lives in a
        # separate output tree.
        loaded = Frame.from_stac(catalog_json, frame_root=full_frame_dir)

        # Frame root resolves to the original frame dir.
        assert loaded.root == original.root

        # Sub-objects are wired (regardless of which dir they resolve to).
        assert loaded.geometry is not None
        assert loaded.interferograms is not None

        # Summary is structurally equivalent (same keys, same pair count,
        # same asset keys per pair).
        loaded_summary = loaded.summary()
        assert set(loaded_summary.keys()) == set(original_summary.keys())
        assert (
            loaded_summary["interferograms"]["pair_count"]
            == original_summary["interferograms"]["pair_count"]
        )
        assert (
            loaded_summary["interferograms"]["pairs"]
            == original_summary["interferograms"]["pairs"]
        )
        # assets_by_pair keys match
        orig_assets = original_summary["interferograms"]["assets_by_pair"]
        loaded_assets = loaded_summary["interferograms"]["assets_by_pair"]
        assert set(orig_assets.keys()) == set(loaded_assets.keys())
        for pair in orig_assets:
            assert set(orig_assets[pair]) == set(loaded_assets[pair])

    def test_geometry_only_round_trip(self, frame_dir: Path, tmp_path: Path) -> None:
        """The bare frame_dir fixture (geometry only, no interferograms/)
        round-trips correctly.
        """
        from faninsar.datasets.frame import Frame

        # frame_dir only has geometry/ — confirm no interferograms/ on disk.
        assert not (frame_dir / "interferograms").exists()

        original = Frame(frame_dir)
        assert original.geometry is not None
        assert original.interferograms is None

        out_dir = tmp_path / "stac_geom_only"
        original.to_stac(output_dir=out_dir)

        loaded = Frame.from_stac(out_dir / "catalog.json", frame_root=frame_dir)
        assert loaded.geometry is not None
        assert loaded.interferograms is None

    def test_open_remote_raises_not_implemented(self) -> None:
        from faninsar.datasets.frame import Frame

        with pytest.raises(NotImplementedError, match="M5"):
            Frame.open_remote("https://example.com/catalog.json")
