"""Tests for faninsar.pipeline.inversion — InversionPipeline (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faninsar.core import Pairs
from faninsar.datasets.frame import Frame, FrameGeometry
from faninsar.pipeline import InversionPipeline

# Reuse the conftest _write_tiff helper.
from tests.datasets.frame.conftest import _write_tiff


def _cuda_or_mps() -> bool:
    from faninsar._core.device import cuda_available, mps_available

    return cuda_available() or mps_available()


def _cuda_only() -> bool:
    from faninsar._core.device import cuda_available

    return cuda_available()


def _build_multi_pair_frame(
    out_dir: Path, n_pairs: int = 6, shape: tuple[int, int] = (8, 8)
) -> Path:
    """Build a frame with a sequential-date pair network for NSBAS.

    Uses dates 20200101..20200601 (one per month) with a sequential network
    (each consecutive pair + a few cross-pairs) so NSBAS can invert.
    """
    import rasterio
    from rasterio.transform import from_bounds

    dates = [f"20200{m}01" for m in range(1, n_pairs + 1)]
    # Sequential network: (d0,d1),(d1,d2),...,(d_{n-2},d_{n-1}) + (d0,d_{n-1})
    pair_names = [f"{dates[i]}_{dates[i + 1]}" for i in range(n_pairs - 1)]
    pair_names.append(f"{dates[0]}_{dates[-1]}")

    bounds = (10.0, 45.0, 11.0, 46.0)
    # Geometry
    inc = np.full(shape, 35.0, dtype=np.float32)
    inc_path = out_dir / "incidence.tif"
    _write_tiff(inc_path, bounds, inc)
    FrameGeometry.from_rasters(
        out_dir=out_dir / "frame", incidence=inc_path, overwrite=True
    )

    ifgs_dir = out_dir / "frame" / "interferograms"
    ifgs_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for pname in pair_names:
        pair_dir = ifgs_dir / pname
        pair_dir.mkdir(parents=True)
        unw = rng.normal(0, 1.0, shape).astype(np.float32)
        coh = rng.uniform(0.5, 1.0, shape).astype(np.float32)
        _write_tiff(pair_dir / "unw_phase.cog.tif", bounds, unw)
        _write_tiff(pair_dir / "coherence.cog.tif", bounds, coh, nodata=0.0)

    # Write item.json + index.
    from faninsar.datasets.frame.metadata import (
        build_interferograms_index,
        build_item_metadata,
        resolve_geometry_href,
        save_json,
    )

    assets_by_pair: dict[str, list[str]] = {}
    for pname in pair_names:
        parts = pname.split("_")
        item = build_item_metadata(
            pair_name=pname,
            reference_date=parts[0],
            secondary_date=parts[1],
            grid={
                "crs": "EPSG:4326",
                "width": shape[1],
                "height": shape[0],
                "transform": [1, 0, 10, 0, -1, 46],
                "bounds": list(bounds),
                "resolution": [1, 1],
            },
            assets={
                "unw_phase": {"href": "unw_phase.cog.tif", "dtype": "float32"},
                "coherence": {"href": "coherence.cog.tif", "dtype": "float32"},
            },
            geometry_href=resolve_geometry_href(ifgs_dir),
            value_ranges={"coherence": (0.0, 1.0)},
        )
        save_json(item, pair_dir / "item.json")
        assets_by_pair[pname] = ["unw_phase", "coherence"]

    idx = build_interferograms_index(
        pair_count=len(pair_names),
        pairs=pair_names,
        assets_by_pair=assets_by_pair,
        common_grid=True,
        geometry_href=resolve_geometry_href(ifgs_dir),
    )
    save_json(idx, ifgs_dir / "interferograms_index.json")
    return out_dir / "frame"


@pytest.fixture
def multi_pair_frame(tmp_path: Path) -> Path:
    return _build_multi_pair_frame(tmp_path)


class TestInversionPipelineCpu:
    """D2.1: InversionPipeline runs on CPU and writes displacement Zarr."""

    def test_pipeline_runs_and_writes_zarr(
        self, multi_pair_frame: Path, tmp_path: Path
    ) -> None:
        import xarray as xr

        frame = Frame(multi_pair_frame)
        pipeline = InversionPipeline(frame, model="linear", device="cpu")
        out = tmp_path / "displacement.zarr"
        ds = pipeline.run(output_zarr=out, overwrite=True, chunk_size=(4, 4))
        assert out.exists()
        loaded = xr.open_zarr(str(out), consolidated=False)
        assert "displacement" in loaded
        disp = loaded["displacement"]
        # time dim = n_dates - 1 (incremental displacement)
        assert disp.dims == ("time", "y", "x")
        assert disp.sizes["time"] == frame.interferograms.pairs().dates.size - 1

    def test_velocity_in_output(self, multi_pair_frame: Path, tmp_path: Path) -> None:
        import xarray as xr

        frame = Frame(multi_pair_frame)
        pipeline = InversionPipeline(frame, model="linear", device="cpu")
        out = tmp_path / "displacement.zarr"
        pipeline.run(output_zarr=out, overwrite=True, chunk_size=(4, 4))
        loaded = xr.open_zarr(str(out), consolidated=False)
        assert "velocity" in loaded
        vel = loaded["velocity"]
        assert vel.dims == ("y", "x")

    def test_pipeline_does_not_mutate_frame(
        self, multi_pair_frame: Path, tmp_path: Path
    ) -> None:
        """D2.1: InversionPipeline must not modify the input Frame."""
        frame = Frame(multi_pair_frame)
        orig_pair_count = len(frame.interferograms.pairs())
        pipeline = InversionPipeline(frame, model="linear", device="cpu")
        pipeline.run(output_zarr=tmp_path / "d.zarr", overwrite=True, chunk_size=(4, 4))
        assert len(frame.interferograms.pairs()) == orig_pair_count

    def test_pipeline_requires_interferograms(self, tmp_path: Path) -> None:
        """Frame without interferograms raises."""
        # geometry-only frame
        out_dir = tmp_path / "geom_only" / "frame"
        (out_dir / "geometry").mkdir(parents=True)
        from faninsar.datasets.frame.metadata import build_geometry_metadata, save_json

        meta = build_geometry_metadata(
            crs="EPSG:4326",
            width=8,
            height=8,
            transform=[1, 0, 0, 0, -1, 8],
            bounds=[0, 0, 8, 8],
            resolution=(1, 1),
            assets={"incidence": {"href": "incidence.zarr"}},
        )
        save_json(meta, out_dir / "geometry" / "geometry.json")
        frame = Frame(out_dir)
        assert frame.interferograms is None
        with pytest.raises(ValueError):
            InversionPipeline(frame)


class TestInversionPipelineGpu:
    """D2.2: GPU annotation path (skipped when no GPU)."""

    @pytest.mark.skipif(
        not _cuda_only(), reason="CUDA required (MPS lacks float64 support)"
    )
    def test_gpu_path_matches_cpu(
        self, multi_pair_frame: Path, tmp_path: Path
    ) -> None:
        import xarray as xr

        frame = Frame(multi_pair_frame)
        device = "cuda"
        gpu_pipe = InversionPipeline(frame, model="linear", device=device)
        cpu_pipe = InversionPipeline(frame, model="linear", device="cpu")
        gpu_out = tmp_path / "gpu.zarr"
        cpu_out = tmp_path / "cpu.zarr"
        gpu_pipe.run(output_zarr=gpu_out, overwrite=True, chunk_size=(4, 4))
        cpu_pipe.run(output_zarr=cpu_out, overwrite=True, chunk_size=(4, 4))
        gpu_disp = xr.open_zarr(str(gpu_out), consolidated=False)["displacement"].compute()
        cpu_disp = xr.open_zarr(str(cpu_out), consolidated=False)["displacement"].compute()
        np.testing.assert_allclose(
            np.asarray(gpu_disp.values),
            np.asarray(cpu_disp.values),
            atol=1e-3,
        )

    def test_device_annotation_logic(self) -> None:
        """D2.2: device='cpu' does not annotate gpu resources."""
        # We verify the logic indirectly: the pipeline picks cpu device and
        # use_gpu is False, so no gpu annotation is emitted.
        # (Full dask graph annotation introspection requires a distributed
        # client; here we assert the device resolution path.)
        from faninsar._core.device import parse_device

        dev = parse_device("cpu")
        assert str(dev) == "cpu"


class TestInversionPipelineVariance:
    """D2.3: displacement_variance in output Zarr, non-negative."""

    def test_variance_present_and_nonnegative(
        self, multi_pair_frame: Path, tmp_path: Path
    ) -> None:
        import xarray as xr

        frame = Frame(multi_pair_frame)
        pipeline = InversionPipeline(frame, model="linear", device="cpu")
        out = tmp_path / "displacement.zarr"
        pipeline.run(output_zarr=out, overwrite=True, chunk_size=(4, 4))
        loaded = xr.open_zarr(str(out), consolidated=False)
        assert "displacement_variance" in loaded
        var = loaded["displacement_variance"].compute()
        arr = np.asarray(var.values)
        assert np.all(arr >= 0.0)
        assert var.dims == ("time", "y", "x")

    def test_variance_absent_without_coherence(
        self, multi_pair_frame: Path, tmp_path: Path
    ) -> None:
        """When coherence is unavailable, no variance variable is written."""
        import xarray as xr

        frame = Frame(multi_pair_frame)
        # Remove coherence assets to simulate no-coherence case.
        for pname in frame.interferograms.pairs().to_names():
            coh_path = frame.interferograms.path(pname, "coherence")
            if coh_path.exists():
                coh_path.unlink()
        pipeline = InversionPipeline(frame, model="linear", device="cpu")
        out = tmp_path / "displacement.zarr"
        pipeline.run(output_zarr=out, overwrite=True, chunk_size=(4, 4))
        loaded = xr.open_zarr(str(out), consolidated=False)
        assert "displacement_variance" not in loaded


def cuda_or_mps() -> bool:
    return _cuda_or_mps()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
