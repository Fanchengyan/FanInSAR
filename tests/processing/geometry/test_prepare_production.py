"""Tests for PROPOSAL-0031 production geometry helper."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.processing.geometry import ConstantDEM
from faninsar.processing.geometry import prepare_production as prepare_mod
from faninsar.processing.geometry.backend_dispatch import DispatchError
from faninsar.processing.geometry.native_v2.builder import (
    NativeBackend,
    NativeBuilder,
    NativeBuildRequest,
    NativeOperation,
)
from faninsar.processing.geometry.orbit import OrbitInterpolator
from faninsar.processing.geometry.prepare_production import (
    _dem_native_arrays,
    _native_lock_is_open,
    _public_raster_dem_native_arrays,
    prepare_production_geometry,
    run_geo2rdr,
    run_rdr2geo,
)
from faninsar.processing.geometry.transforms import RadarGeometryModel
from faninsar.processing.geometry.v2 import Operation


@pytest.fixture(autouse=True)
def _clear_production_residency() -> Iterator[None]:
    """Isolate prepared-cache and GPU DEM table across tests."""
    prepare_mod._PREPARED_CACHE.clear()
    prepare_mod._GPU_DEM_TABLE.clear()
    yield
    prepare_mod._PREPARED_CACHE.clear()
    prepare_mod._GPU_DEM_TABLE.clear()


def _model() -> RadarGeometryModel:
    """Build a tiny circular-orbit radar model for helper tests."""
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    r_sat = 7_071_000.0
    v_sat = 7_000.0
    omega = v_sat / r_sat
    vectors = []
    for i in range(3):
        t = t0 + timedelta(seconds=(i - 1))
        angle = omega * (i - 1)
        pos = (r_sat * np.cos(angle), r_sat * np.sin(angle), 0.0)
        vel = (-v_sat * np.sin(angle), v_sat * np.cos(angle), 0.0)
        vectors.append(OrbitStateVector(time=t, position_m=pos, velocity_m_s=vel))
    orbit = OrbitMetadata(
        reference_frame="ECR",
        source="test",
        vectors=tuple(vectors),
    )
    return RadarGeometryModel(
        orbit=OrbitInterpolator.from_orbit(orbit),
        sensing_start=t0,
        azimuth_time_interval_s=1.0,
        starting_slant_range_m=700_000.0,
        range_spacing_m=10.0,
        wavelength_m=0.056,
        look_direction="right",
    )


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux /proc is required")
def test_native_lock_probe_distinguishes_live_and_stale_lock(tmp_path: Path) -> None:
    """A live FileBaton descriptor is retained while a stale marker is removable."""
    lock = tmp_path / "lock"
    stream = lock.open("w", encoding="utf-8")
    try:
        assert _native_lock_is_open(lock)
    finally:
        stream.close()
    assert not _native_lock_is_open(lock)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux /proc is required")
def test_stale_native_graph_isolated_without_deleting_racing_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale graph is renamed as one unit so a racing compiler stays isolated."""
    build_dir = tmp_path / "geo2rdr"
    build_dir.mkdir()
    (build_dir / "build.ninja").write_text("nvcc = /usr/bin/nvcc\n", encoding="utf-8")
    (build_dir / "geo2rdr_cuda.cuda.o").write_bytes(b"stale")
    monkeypatch.setattr(prepare_mod, "_native_build_dir", lambda _operation: build_dir)
    prepare_mod._scrub_stale_system_nvcc_ninja(
        Path("/opt/pixi/bin/nvcc"),
        NativeOperation.GEO2RDR,
    )
    assert not build_dir.exists()
    isolated = tuple(tmp_path.glob("geo2rdr.stale-*"))
    assert len(isolated) == 1
    assert (isolated[0] / "build.ninja").is_file()
    assert (isolated[0] / "geo2rdr_cuda.cuda.o").is_file()
    assert (isolated[0] / "lock").is_file()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux /proc is required")
def test_failed_stale_graph_rename_releases_claimed_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed isolation rename does not leave a blocking FileBaton marker."""
    build_dir = tmp_path / "geo2rdr"
    build_dir.mkdir()
    (build_dir / "build.ninja").write_text("nvcc = /usr/bin/nvcc\n", encoding="utf-8")
    monkeypatch.setattr(prepare_mod, "_native_build_dir", lambda _operation: build_dir)

    def fail_rename(_source: Path, _target: Path) -> Path:
        raise OSError("injected rename failure")

    monkeypatch.setattr(type(build_dir), "rename", fail_rename)
    prepare_mod._scrub_stale_system_nvcc_ninja(
        Path("/opt/pixi/bin/nvcc"),
        NativeOperation.GEO2RDR,
    )
    assert not (build_dir / "lock").exists()


def test_cuda_geo_plan_is_supported_for_build() -> None:
    """PROPOSAL-0031 flips CUDA geo2rdr/rdr2geo plan.supported for prepare."""
    builder = NativeBuilder()
    for operation in (NativeOperation.GEO2RDR, NativeOperation.RDR2GEO):
        plan = builder.plan(NativeBuildRequest(operation, NativeBackend.CUDA))
        assert plan.supported is True
        assert plan.unsupported_reason == ""


def test_prepare_production_geometry_requires_device() -> None:
    """Callers must pass device; there is no CPU default."""
    params = inspect.signature(prepare_production_geometry).parameters
    assert params["device"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="device"):
        prepare_production_geometry(  # type: ignore[misc]
            Operation.GEO2RDR,
            _model(),
            shape=(1, 1),
        )


def test_prepare_production_geometry_cpu_roundtrip() -> None:
    """CPU helper prepares and execute_geometry runs without UUID/executor."""
    model = _model()
    prepared = prepare_production_geometry(
        Operation.RDR2GEO,
        model,
        device="cpu",
        shape=(2, 2),
    )
    az = np.zeros((2, 2), dtype=np.float64)
    rg = np.zeros((2, 2), dtype=np.float64)
    result = run_rdr2geo(model, az, rg, device="cpu", max_iter=8)
    assert result.latitude_deg.shape == (2, 2)
    assert result.converged.dtype == bool
    geo = run_geo2rdr(
        model,
        result.latitude_deg,
        result.longitude_deg,
        np.where(np.isfinite(result.height_m), result.height_m, 0.0),
        device="cpu",
        max_iter=8,
    )
    assert geo.range_index.shape == (2, 2)
    _ = prepared


def test_cuda_prepare_requests_compile_identity(monkeypatch) -> None:
    """PROPOSAL-0020/0025: CUDA production prepare registers Compile."""
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []

    def fake_prepare(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(compile=kwargs.get("compile"))

    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.prepare_geometry",
        fake_prepare,
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production._require_device",
        lambda _device: SimpleNamespace(type="cuda"),
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production._cuda_uuids",
        lambda _resolved: ("gpu-uuid", None),
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production._try_load_cuda_module",
        lambda _op: None,
    )
    prepared = prepare_production_geometry(
        Operation.RDR2GEO,
        _model(),
        device="cuda",
        shape=(2, 2),
    )
    assert prepared.compile is True
    assert calls
    assert calls[0]["compile"] is True
    assert calls[0]["compile_performance_eligible"] is True


def test_dem_native_arrays_constant_height() -> None:
    """ConstantDEM becomes a coarse global raster for the six-point stencil."""
    values, metadata, bounds = _dem_native_arrays(ConstantDEM(12.5))
    assert values.shape[0] >= 6 and values.shape[1] >= 6
    assert np.allclose(values, 12.5)
    assert metadata.shape == (4,)
    assert bounds.tolist() == [12.5, 12.5]


def test_projected_raster_dem_geometry_view_does_not_materialize_epsg4326(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Projected public DEMs use the private sampler branch directly."""
    from affine import Affine

    from faninsar.processing.geometry import GridSpec, RasterDEM

    source = RasterDEM(
        array=np.arange(64, dtype=np.float32).reshape(8, 8),
        grid=GridSpec(
            "EPSG:32632",
            Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4200000.0),
            shape=(8, 8),
        ),
    )

    def fail_materialization(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("projected geometry view must not call to_raster")

    monkeypatch.setattr(RasterDEM, "to_raster", fail_materialization)
    values, metadata, bounds = _public_raster_dem_native_arrays(source)
    assert values.shape == source.shape
    assert metadata.shape == (4,)
    assert np.all(np.isfinite(bounds))


def test_cuda_rdr2geo_prepare_registers_native_dem(monkeypatch) -> None:
    """PROPOSAL-0026: CUDA rdr2geo prepare binds DEM context for Native."""
    from types import SimpleNamespace

    captured: list[dict[str, object]] = []

    def fake_manifest(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(backend="native"), {
            "look_right": True,
            "dem_values": object(),
            "dem_metadata": object(),
            "dem_height_bounds": object(),
        }

    def fake_prepare(*_args, **kwargs):
        captured.append({"prepare": kwargs})
        return SimpleNamespace(compile=kwargs.get("compile"))

    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production._rdr2geo_native_manifest",
        fake_manifest,
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.prepare_geometry",
        fake_prepare,
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production._require_device",
        lambda _device: SimpleNamespace(type="cuda"),
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production._cuda_uuids",
        lambda _resolved: ("gpu-uuid", None),
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production._try_load_cuda_module",
        lambda _op: object(),
    )
    dem = ConstantDEM(4.0)
    prepared = prepare_production_geometry(
        Operation.RDR2GEO,
        _model(),
        device="cuda",
        shape=(4, 4),
        dem=dem,
    )
    assert captured[0]["dem"] is dem
    prepare_kwargs = captured[1]["prepare"]
    assert prepare_kwargs["native_executor"] is not None
    assert prepare_kwargs["native_context_inputs"]["look_right"] is True
    assert prepare_kwargs["compile"] is False
    assert prepared.compile is False


def test_cpu_prepare_does_not_request_compile(monkeypatch) -> None:
    """CPU production prepare stays Eager so CI does not pay torch.compile."""
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []

    def fake_prepare(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(compile=kwargs.get("compile"))

    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.prepare_geometry",
        fake_prepare,
    )
    prepared = prepare_production_geometry(
        Operation.RDR2GEO,
        _model(),
        device="cpu",
        shape=(2, 2),
    )
    assert prepared.compile is False
    assert calls[0]["compile"] is False
    assert calls[0]["compile_performance_eligible"] is False


def test_prepare_production_geometry_rejects_mps() -> None:
    """MPS fails closed after Newton deletion."""
    pytest.importorskip("torch")
    import torch

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        with pytest.raises(DispatchError):
            prepare_production_geometry(
                Operation.GEO2RDR,
                _model(),
                device="mps",
                shape=(1,),
            )
        return
    with pytest.raises(DispatchError, match="mps"):
        prepare_production_geometry(
            Operation.GEO2RDR,
            _model(),
            device="mps",
            shape=(1,),
        )


def test_prepared_cache_hits_on_model_digest_not_object_id(monkeypatch) -> None:
    """Two models with the same operational digest reuse PreparedGeometry."""
    from dataclasses import replace
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []

    def fake_prepare(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(compile=False, token=len(calls))

    monkeypatch.setattr(prepare_mod, "prepare_geometry", fake_prepare)
    first_model = _model()
    second_model = replace(first_model)
    assert first_model is not second_model
    first = prepare_production_geometry(
        Operation.GEO2RDR,
        first_model,
        device="cpu",
        shape=(3, 3),
    )
    second = prepare_production_geometry(
        Operation.GEO2RDR,
        second_model,
        device="cpu",
        shape=(3, 3),
    )
    assert first is second
    assert len(calls) == 1


def test_prepared_cache_misses_on_sensing_start_or_look(monkeypatch) -> None:
    """Bursts that share orbit+DEM but differ in timing or look miss."""
    from dataclasses import replace
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []

    def fake_prepare(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(compile=False, token=len(calls))

    monkeypatch.setattr(prepare_mod, "prepare_geometry", fake_prepare)
    model = _model()
    prepare_production_geometry(
        Operation.GEO2RDR, model, device="cpu", shape=(3, 3)
    )
    shifted = replace(
        model,
        sensing_start=model.sensing_start + timedelta(seconds=1),
    )
    prepare_production_geometry(
        Operation.GEO2RDR, shifted, device="cpu", shape=(3, 3)
    )
    looked = replace(model, look_direction="left")
    prepare_production_geometry(
        Operation.GEO2RDR, looked, device="cpu", shape=(3, 3)
    )
    assert len(calls) == 3


def test_prepared_cache_misses_on_dem_digest(monkeypatch) -> None:
    """A DEM content change misses the prepared cache."""
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []

    def fake_prepare(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(compile=False, token=len(calls))

    monkeypatch.setattr(prepare_mod, "prepare_geometry", fake_prepare)
    model = _model()
    prepare_production_geometry(
        Operation.RDR2GEO,
        model,
        device="cpu",
        shape=(2, 2),
        dem=ConstantDEM(0.0),
    )
    prepare_production_geometry(
        Operation.RDR2GEO,
        model,
        device="cpu",
        shape=(2, 2),
        dem=ConstantDEM(12.0),
    )
    assert len(calls) == 2


def test_gpu_dem_table_aliases_across_prepared_shapes(monkeypatch) -> None:
    """Shape miss does not re-upload DEM tensors keyed by dem_digest+device."""
    from types import SimpleNamespace

    original_arrays = prepare_mod._dem_native_arrays
    array_calls: list[int] = []

    def counting_arrays(dem):
        array_calls.append(1)
        return original_arrays(dem)

    monkeypatch.setattr(prepare_mod, "_dem_native_arrays", counting_arrays)

    def fake_manifest(**kwargs):
        import torch

        from faninsar.processing.geometry.torch_backends_v2 import _dem_digest

        dem = kwargs["dem"]
        resident = prepare_mod._resident_gpu_dem(
            _dem_digest(dem),
            "cuda-uuid:gpu-uuid",
            torch.device("cpu"),
            dem,
        )
        return SimpleNamespace(backend="native"), {
            "look_right": True,
            **resident,
        }

    captured: list[dict[str, object]] = []

    def fake_prepare(*_args, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(compile=kwargs.get("compile"))

    monkeypatch.setattr(prepare_mod, "_rdr2geo_native_manifest", fake_manifest)
    monkeypatch.setattr(prepare_mod, "prepare_geometry", fake_prepare)
    monkeypatch.setattr(
        prepare_mod,
        "_require_device",
        lambda _device: SimpleNamespace(type="cuda"),
    )
    monkeypatch.setattr(prepare_mod, "_cuda_uuids", lambda _resolved: ("gpu-uuid", None))
    monkeypatch.setattr(prepare_mod, "_try_load_cuda_module", lambda _op: object())
    dem = ConstantDEM(4.0)
    model = _model()
    prepare_production_geometry(
        Operation.RDR2GEO, model, device="cuda", shape=(4, 4), dem=dem
    )
    prepare_production_geometry(
        Operation.RDR2GEO, model, device="cuda", shape=(8, 8), dem=dem
    )
    assert len(array_calls) == 1
    first_dem = captured[0]["native_context_inputs"]["dem_values"]
    second_dem = captured[1]["native_context_inputs"]["dem_values"]
    assert first_dem is second_dem


def test_rdr2geo_without_dem_context_does_not_admit_native(monkeypatch) -> None:
    """Native DEM-bound identity is refused without DEM raster context."""
    from types import SimpleNamespace

    def fake_manifest(**kwargs):
        return SimpleNamespace(backend="native"), {"look_right": True}

    captured: list[dict[str, object]] = []

    def fake_prepare(*_args, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(compile=kwargs.get("compile"))

    monkeypatch.setattr(prepare_mod, "_rdr2geo_native_manifest", fake_manifest)
    monkeypatch.setattr(prepare_mod, "prepare_geometry", fake_prepare)
    monkeypatch.setattr(
        prepare_mod,
        "_require_device",
        lambda _device: SimpleNamespace(type="cuda"),
    )
    monkeypatch.setattr(prepare_mod, "_cuda_uuids", lambda _resolved: ("gpu-uuid", None))
    monkeypatch.setattr(prepare_mod, "_try_load_cuda_module", lambda _op: object())
    prepare_production_geometry(
        Operation.RDR2GEO,
        _model(),
        device="cuda",
        shape=(4, 4),
        dem=ConstantDEM(4.0),
    )
    assert captured[0]["native_executor"] is None
    assert captured[0]["native_context_inputs"] is None
    assert captured[0]["compile"] is True


def test_native_rdr2geo_registers_dem_once_per_digest(monkeypatch) -> None:
    """Second prepare of the same digest does not re-register DEM context."""
    from types import SimpleNamespace

    original_arrays = prepare_mod._dem_native_arrays
    array_calls: list[int] = []

    def counting_arrays(dem):
        array_calls.append(1)
        return original_arrays(dem)

    monkeypatch.setattr(prepare_mod, "_dem_native_arrays", counting_arrays)
    manifest_calls: list[int] = []

    def fake_manifest(**kwargs):
        import torch

        from faninsar.processing.geometry.torch_backends_v2 import _dem_digest

        manifest_calls.append(1)
        dem = kwargs["dem"]
        resident = prepare_mod._resident_gpu_dem(
            _dem_digest(dem),
            "cuda-uuid:gpu-uuid",
            torch.device("cpu"),
            dem,
        )
        return SimpleNamespace(backend="native"), {
            "look_right": True,
            **resident,
        }

    def fake_prepare(*_args, **kwargs):
        return SimpleNamespace(compile=kwargs.get("compile"))

    monkeypatch.setattr(prepare_mod, "_rdr2geo_native_manifest", fake_manifest)
    monkeypatch.setattr(prepare_mod, "prepare_geometry", fake_prepare)
    monkeypatch.setattr(
        prepare_mod,
        "_require_device",
        lambda _device: SimpleNamespace(type="cuda"),
    )
    monkeypatch.setattr(prepare_mod, "_cuda_uuids", lambda _resolved: ("gpu-uuid", None))
    monkeypatch.setattr(prepare_mod, "_try_load_cuda_module", lambda _op: object())
    dem = ConstantDEM(7.0)
    model = _model()
    first = prepare_production_geometry(
        Operation.RDR2GEO, model, device="cuda", shape=(4, 4), dem=dem
    )
    second = prepare_production_geometry(
        Operation.RDR2GEO, _model(), device="cuda", shape=(4, 4), dem=dem
    )
    assert first is second
    assert len(manifest_calls) == 1
    assert len(array_calls) == 1
