"""Tests for faninsar.datasets.frame.mintpy — MintPy HDF5 export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from faninsar.datasets.frame import Frame, build_mintpy
from faninsar.datasets.frame.discovery.mintpy import MintPyDiscoverer


class TestToMintpy:
    """Frame.to_mintpy / build_mintpy round-trip tests."""

    def test_writes_inputs_layout(self, frame_dir: Path, tmp_path: Path) -> None:
        frame = Frame(frame_dir)
        out = frame.to_mintpy(output_dir=tmp_path / "mintpy", overwrite=True)

        assert out == tmp_path / "mintpy"
        assert (out / "inputs" / "ifgramStack.h5").exists()
        assert (out / "inputs" / "geometryGeo.h5").exists()
        assert (out / "mintpy_project.json").exists()

    def test_default_output_under_root(self, frame_dir: Path) -> None:
        frame = Frame(frame_dir)
        out = frame.to_mintpy(overwrite=True)
        assert out == frame_dir / "mintpy"
        assert (out / "inputs" / "ifgramStack.h5").exists()

    def test_ifgram_stack_datasets(self, frame_dir: Path, tmp_path: Path) -> None:
        frame = Frame(frame_dir)
        out = frame.to_mintpy(output_dir=tmp_path / "mintpy", overwrite=True)

        with h5py.File(out / "inputs" / "ifgramStack.h5", "r") as f:
            assert f.attrs["FILE_TYPE"] == "ifgramStack"
            assert "date" in f
            assert "bperp" in f
            assert "dropIfgram" in f
            assert "unwrapPhase" in f
            assert "coherence" in f

            # date shape (m, 2) and S8 encoding
            assert f["date"].shape == (2, 2)
            assert f["date"].dtype == np.dtype("S8")
            # First pair is 20191115_20200314
            assert f["date"][0, 0] == b"20191115"
            assert f["date"][0, 1] == b"20200314"

            # dropIfgram all True by default
            assert np.all(f["dropIfgram"][:])

            # unwrapPhase shape (m, L, W)
            assert f["unwrapPhase"].shape == (2, 10, 10)
            assert f["unwrapPhase"].dtype == np.float32
            assert f["unwrapPhase"].attrs["UNIT"] == "radian"

            # bperp zeros when item.json has no baseline field
            assert f["bperp"].shape == (2,)
            assert np.all(f["bperp"][:] == 0.0)

    def test_geometry_datasets(self, frame_dir: Path, tmp_path: Path) -> None:
        frame = Frame(frame_dir)
        out = frame.to_mintpy(output_dir=tmp_path / "mintpy", overwrite=True)

        with h5py.File(out / "inputs" / "geometryGeo.h5", "r") as f:
            assert f.attrs["FILE_TYPE"] == "geometry"
            assert "incidenceAngle" in f
            assert f["incidenceAngle"].shape == (10, 10)
            assert f["incidenceAngle"].dtype == np.float32
            # Geocoded geo attrs present
            assert "X_FIRST" in f.attrs
            assert "Y_FIRST" in f.attrs
            assert "X_STEP" in f.attrs
            assert "Y_STEP" in f.attrs

    def test_round_trip_discovery(self, frame_dir: Path, tmp_path: Path) -> None:
        """The exported directory should be discoverable by MintPyDiscoverer."""
        frame = Frame(frame_dir)
        out = frame.to_mintpy(output_dir=tmp_path / "mintpy", overwrite=True)

        discoverer = MintPyDiscoverer()
        geom = discoverer.discover_geometry_product(out)
        assert geom.is_dir()

        pairs = discoverer.discover_pairs(out)
        assert len(pairs) == 1
        assert (pairs[0] / "ifgramStack.h5").exists()

    def test_overwrite_false_raises(self, frame_dir: Path, tmp_path: Path) -> None:
        frame = Frame(frame_dir)
        out = tmp_path / "mintpy"
        frame.to_mintpy(output_dir=out, overwrite=True)
        with pytest.raises(FileExistsError):
            frame.to_mintpy(output_dir=out, overwrite=False)

    def test_no_interferograms_raises(self, frame_dir: Path, tmp_path: Path) -> None:
        import shutil

        geom_only = tmp_path / "geom_only"
        shutil.copytree(frame_dir, geom_only)
        shutil.rmtree(geom_only / "interferograms")
        frame = Frame(geom_only)
        with pytest.raises(FileNotFoundError):
            frame.to_mintpy()

    def test_build_mintpy_direct(self, frame_dir: Path, tmp_path: Path) -> None:
        frame = Frame(frame_dir)
        out = build_mintpy(
            geometry=frame.geometry,
            interferograms=frame.interferograms,
            output_dir=tmp_path / "direct",
            overwrite=True,
        )
        assert (out / "inputs" / "ifgramStack.h5").exists()
        assert (out / "inputs" / "geometryGeo.h5").exists()

    def test_build_mintpy_no_geometry(self, frame_dir: Path, tmp_path: Path) -> None:
        """build_mintpy should work without geometry (ifgramStack only)."""
        frame = Frame(frame_dir)
        out = build_mintpy(
            geometry=None,
            interferograms=frame.interferograms,
            output_dir=tmp_path / "no_geom",
            overwrite=True,
        )
        assert (out / "inputs" / "ifgramStack.h5").exists()
        assert not (out / "inputs" / "geometryGeo.h5").exists()


class TestFromMintpy:
    """D1.5: Frame.from_mintpy loads MintPy HDF5 without COG materialization."""

    def test_from_mintpy_round_trip_pair_count(
        self, frame_dir: Path, tmp_path: Path
    ) -> None:
        """build_mintpy(frame) -> from_mintpy(result): pair count matches."""
        frame = Frame(frame_dir)
        out = frame.to_mintpy(output_dir=tmp_path / "mintpy", overwrite=True)

        loaded = Frame.from_mintpy(out, output_dir=tmp_path / "loaded", overwrite=True)
        assert loaded.interferograms is not None
        orig_pairs = frame.interferograms.pairs()
        loaded_pairs = loaded.interferograms.pairs()
        assert len(loaded_pairs) == len(orig_pairs)

    def test_from_mintpy_no_cog_files(self, frame_dir: Path, tmp_path: Path) -> None:
        """from_mintpy must not write per-pair COG files (Zarr cubes only)."""
        frame = Frame(frame_dir)
        out = frame.to_mintpy(output_dir=tmp_path / "mintpy", overwrite=True)
        loaded = Frame.from_mintpy(out, output_dir=tmp_path / "loaded", overwrite=True)

        # No .tif / .cog.tif under interferograms/.
        ifgs_root = loaded.interferograms.root
        cogs = list(ifgs_root.rglob("*.tif"))
        assert cogs == []
        # Zarr cubes present instead.
        assert (ifgs_root / "unw_phase.zarr").exists()

    def test_from_mintpy_open_stack_lazy(
        self, frame_dir: Path, tmp_path: Path
    ) -> None:
        """open_stack returns a dask-backed (pair, y, x) DataArray."""
        frame = Frame(frame_dir)
        out = frame.to_mintpy(output_dir=tmp_path / "mintpy", overwrite=True)
        loaded = Frame.from_mintpy(out, output_dir=tmp_path / "loaded", overwrite=True)

        stack = loaded.interferograms.open_stack("unw_phase", chunks="auto")
        assert stack.dims == ("pair", "y", "x")
        assert stack.chunks is not None  # dask-backed
        assert stack.sizes["pair"] == 2

    def test_from_mintpy_geometry_loaded(self, frame_dir: Path, tmp_path: Path) -> None:
        """Geometry assets are loaded from geometryGeo.h5."""
        frame = Frame(frame_dir)
        out = frame.to_mintpy(output_dir=tmp_path / "mintpy", overwrite=True)
        loaded = Frame.from_mintpy(out, output_dir=tmp_path / "loaded", overwrite=True)
        assert loaded.geometry is not None
        assert loaded.geometry.metadata is not None

    def test_from_mintpy_missing_h5_raises(self, tmp_path: Path) -> None:
        """Missing ifgramStack.h5 raises FileNotFoundError."""
        bad = tmp_path / "not_mintpy"
        (bad / "inputs").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            Frame.from_mintpy(bad)
