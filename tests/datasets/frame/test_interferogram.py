"""Tests for faninsar.datasets.frame.interferogram."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from faninsar.core import Pairs
from faninsar.datasets.frame.exceptions import GridMismatchError, PairNotFoundError
from faninsar.datasets.frame.geometry import FrameGeometry
from faninsar.datasets.frame.interferogram import FrameInterferogramCollection


def _write_tiff(
    path: Path,
    bounds: tuple[float, float, float, float],
    data: np.ndarray,
    crs: str = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
    nodata: float = -9999.0,
    dtype: str = "float32",
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
        dtype=dtype,
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

    # Geometry rasters
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
def ifg_collection(frame_dir: Path, tmp_path: Path) -> FrameInterferogramCollection:
    """Build a FrameInterferogramCollection directly from synthetic pair data."""
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

    from faninsar.datasets.frame.metadata import (
        build_interferograms_index,
        build_item_metadata,
        save_json,
    )

    for pname in pair_names:
        pair_dir = ifgs_dir / pname
        parts = pname.split("_")
        item_meta = build_item_metadata(
            pair_name=pname,
            reference_date=parts[0],
            secondary_date=parts[1],
            grid={
                "crs": 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
                "width": shape[1],
                "height": shape[0],
                "transform": [0.1, 0.0, 10.0, 0.0, -0.1, 46.0],
                "bounds": [10.0, 45.0, 11.0, 46.0],
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


class TestFrameInterferogramCreation:
    def test_creates_ifgs_dir(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        assert ifg_collection.root.is_dir()

    def test_creates_pair_dirs(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        pairs_obj = ifg_collection.pairs()
        assert len(pairs_obj) == 2

    def test_creates_pair_assets(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        pairs_obj = ifg_collection.pairs()
        for pname in pairs_obj.to_names():
            assert ifg_collection.exists(pname, "unw_phase")
            assert ifg_collection.exists(pname, "coherence")

    def test_creates_item_json(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        pairs_obj = ifg_collection.pairs()
        for pname in pairs_obj.to_names():
            item = ifg_collection.item(pname)
            assert item["type"] == "FrameInterferogramItem"
            assert item["pair_name"] == pname
            assert "reference_date" in item
            assert "secondary_date" in item
            assert "assets" in item
            assert "grid" in item

    def test_creates_interferograms_index_json(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        assert ifg_collection.index_metadata is not None
        idx = ifg_collection.index_metadata
        assert idx["type"] == "FrameInterferogramIndex"
        assert idx["pair_count"] == 2
        assert len(idx["pairs"]) == 2

    def test_temporal_baseline_in_item(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        """item.json should contain temporal_baseline_days."""
        pairs_obj = ifg_collection.pairs()
        pname = pairs_obj.to_names()[0]
        item = ifg_collection.item(pname)
        assert "temporal_baseline_days" in item
        assert item["temporal_baseline_days"] is not None


class TestValueRangeMetadata:
    """D0.2: value_ranges recorded in item.json for cross-processor coherence."""

    def test_value_ranges_serialized(self) -> None:
        from faninsar.datasets.frame.metadata import build_item_metadata

        item = build_item_metadata(
            pair_name="20200101_20200201",
            reference_date="20200101",
            secondary_date="20200201",
            grid={"width": 10, "height": 10},
            assets={"coherence": {"href": "coherence.cog.tif"}},
            value_ranges={"coherence": (0.0, 255.0)},  # LiCSAR-style
        )
        assert "value_ranges" in item
        assert item["value_ranges"]["coherence"] == [0.0, 255.0]

    def test_value_ranges_omitted_when_none(self) -> None:
        from faninsar.datasets.frame.metadata import build_item_metadata

        item = build_item_metadata(
            pair_name="20200101_20200201",
            reference_date="20200101",
            secondary_date="20200201",
            grid={"width": 10, "height": 10},
            assets={"coherence": {"href": "coherence.cog.tif"}},
        )
        assert "value_ranges" not in item

    def test_hyp3_vs_licsar_coherence_ranges(self) -> None:
        """HyP3 coherence is 0-1; LiCSAR is 0-255 — both must be recordable."""
        from faninsar.datasets.frame.metadata import build_item_metadata

        base = dict(
            pair_name="20200101_20200201",
            reference_date="20200101",
            secondary_date="20200201",
            grid={"width": 10, "height": 10},
            assets={"coherence": {"href": "coherence.cog.tif"}},
        )
        hyp3 = build_item_metadata(**base, value_ranges={"coherence": (0.0, 1.0)})
        licsar = build_item_metadata(**base, value_ranges={"coherence": (0.0, 255.0)})
        assert hyp3["value_ranges"]["coherence"] == [0.0, 1.0]
        assert licsar["value_ranges"]["coherence"] == [0.0, 255.0]


class TestFrameInterferogramPairs:
    def test_pairs_returns_pairs_object(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        pairs_obj = ifg_collection.pairs()
        assert isinstance(pairs_obj, Pairs)
        assert len(pairs_obj) == 2


class TestFrameInterferogramOpen:
    def test_open_returns_dataarray(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        pairs_obj = ifg_collection.pairs()
        pname = pairs_obj.to_names()[0]
        da = ifg_collection.open(pname, "unw_phase")
        assert da.ndim == 2
        assert da.shape == (10, 10)

    def test_open_with_chunks(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        pairs_obj = ifg_collection.pairs()
        pname = pairs_obj.to_names()[0]
        da = ifg_collection.open(pname, "coherence", chunks={"y": 5, "x": 5})
        assert hasattr(da.data, "dask")

    def test_open_missing_pair_raises(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        with pytest.raises(PairNotFoundError):
            ifg_collection.open("99990101_99990102", "unw_phase")


class TestFrameInterferogramStack:
    def test_open_stack(self, ifg_collection: FrameInterferogramCollection) -> None:
        stack = ifg_collection.open_stack("coherence")
        assert "pair" in stack.dims
        assert stack.sizes["pair"] == 2

    def test_open_stack_subset(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        pairs_obj = ifg_collection.pairs()
        subset = pairs_obj[:1]
        stack = ifg_collection.open_stack("unw_phase", pairs=subset)
        assert stack.sizes["pair"] == 1


class TestFrameInterferogramZarrStack:
    """D1.1: to_zarr_stack writes a persistent (pair, y, x) Zarr cube."""

    def test_to_zarr_stack_writes_cube(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        import xarray as xr

        out = ifg_collection.to_zarr_stack("unw_phase", overwrite=True)
        assert out.exists()
        # Read back lazily.
        ds = xr.open_zarr(str(out), consolidated=False)
        assert "unw_phase" in ds
        da = ds["unw_phase"]
        assert da.dims == ("pair", "y", "x") or da.dims == ("pair", "band", "y", "x")
        assert da.sizes["pair"] == 2

    def test_zarr_stack_is_dask_backed(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        import xarray as xr

        ifg_collection.to_zarr_stack("unw_phase", overwrite=True)
        out = ifg_collection.zarr_stack_path("unw_phase")
        da = xr.open_zarr(str(out), consolidated=False, chunks="auto")["unw_phase"]
        # dask-backed when chunks requested
        assert da.chunks is not None

    def test_zarr_stack_overwrite_false_raises(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        ifg_collection.to_zarr_stack("unw_phase", overwrite=True)
        with pytest.raises(FileExistsError):
            ifg_collection.to_zarr_stack("unw_phase", overwrite=False)

    def test_zarr_stack_correct_values(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        """Zarr cube values must match the per-pair COG reads."""
        import xarray as xr

        ifg_collection.to_zarr_stack("unw_phase", overwrite=True)
        out = ifg_collection.zarr_stack_path("unw_phase")
        cube = xr.open_zarr(str(out), consolidated=False)["unw_phase"].compute()

        pairs_obj = ifg_collection.pairs()
        for i, pname in enumerate(pairs_obj.to_names()):
            direct = ifg_collection.open(pname, "unw_phase", masked=False)
            # squeeze possible band dim
            if direct.ndim == 3:
                direct = direct.squeeze("band", drop=True)
            np.testing.assert_allclose(
                np.asarray(cube.isel(pair=i).values),
                np.asarray(direct.values),
                equal_nan=True,
            )


class TestFrameInterferogramOpenStackLazy:
    """D1.2: open_stack reads lazily from Zarr cube when present."""

    def test_open_stack_uses_zarr_when_present(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        # No cube yet: eager path returns a non-dask-backed array by default.
        eager = ifg_collection.open_stack("unw_phase")
        # Write the cube; now open_stack must read from Zarr (dask-backed).
        ifg_collection.to_zarr_stack("unw_phase", overwrite=True)
        lazy = ifg_collection.open_stack("unw_phase", chunks="auto")
        assert lazy.sizes["pair"] == eager.sizes["pair"]
        assert lazy.chunks is not None  # dask-backed

    def test_open_stack_eager_fallback_without_cube(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        # No Zarr cube present -> eager concat of COGs.
        assert not ifg_collection.zarr_stack_path("unw_phase").exists()
        stack = ifg_collection.open_stack("unw_phase")
        assert stack.sizes["pair"] == 2

    def test_open_stack_subset_via_zarr(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        ifg_collection.to_zarr_stack("unw_phase", overwrite=True)
        pairs_obj = ifg_collection.pairs()
        subset = pairs_obj[:1]
        stack = ifg_collection.open_stack("unw_phase", pairs=subset)
        assert stack.sizes["pair"] == 1


class TestFrameInterferogramSummary:
    def test_summary(self, ifg_collection: FrameInterferogramCollection) -> None:
        s = ifg_collection.summary()
        assert s["pair_count"] == 2
        assert len(s["pairs"]) == 2
        assert "assets_by_pair" in s


class TestFrameInterferogramExists:
    def test_exists_true(self, ifg_collection: FrameInterferogramCollection) -> None:
        pairs_obj = ifg_collection.pairs()
        pname = pairs_obj.to_names()[0]
        assert ifg_collection.exists(pname, "unw_phase") is True

    def test_exists_false(self, ifg_collection: FrameInterferogramCollection) -> None:
        pairs_obj = ifg_collection.pairs()
        pname = pairs_obj.to_names()[0]
        # wrapped_phase was not created
        assert ifg_collection.exists(pname, "wrapped_phase") is False


class TestFrameInterferogramPath:
    def test_path_returns_expected(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        pairs_obj = ifg_collection.pairs()
        pname = pairs_obj.to_names()[0]
        p = ifg_collection.path(pname, "unw_phase")
        assert p.name == "unw_phase.cog.tif"
        assert p.parent.name == pname


class TestParquetIndex:
    """to_parquet / from_parquet round-trip and double-write."""

    def test_to_parquet_writes_file(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        pytest.importorskip("pyarrow")
        out = ifg_collection.to_parquet()
        assert out.exists()
        assert out.name == "interferograms_index.parquet"

    def test_to_parquet_round_trip_equivalence(
        self, ifg_collection: FrameInterferogramCollection, tmp_path: Path
    ) -> None:
        """from_parquet yields a collection with the same index content."""
        pytest.importorskip("pyarrow")
        ifg_collection.to_parquet()
        loaded = FrameInterferogramCollection.from_parquet(
            ifg_collection.root / "interferograms_index.parquet"
        )
        orig_idx = ifg_collection.index_metadata
        loaded_idx = loaded.index_metadata
        assert loaded_idx is not None
        assert orig_idx is not None
        assert loaded_idx["pair_count"] == orig_idx["pair_count"]
        assert loaded_idx["pairs"] == orig_idx["pairs"]
        assert loaded_idx["assets_by_pair"] == orig_idx["assets_by_pair"]
        assert loaded_idx["common_grid"] == orig_idx["common_grid"]

    def test_index_prefers_parquet_when_present(
        self, ifg_collection: FrameInterferogramCollection
    ) -> None:
        """When parquet exists, index_metadata is read from parquet."""
        pytest.importorskip("pyarrow")
        ifg_collection.to_parquet()
        # Re-open so __init__ picks up the parquet.
        reopened = FrameInterferogramCollection(ifg_collection.root)
        assert reopened.index_metadata is not None
        assert reopened.index_metadata["pair_count"] == 2

    def test_from_parquet_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            FrameInterferogramCollection.from_parquet(
                tmp_path / "nope.parquet"
            )
