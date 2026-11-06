"""Public explicit preparation contracts (PROPOSAL-0073)."""
from pathlib import Path

import pytest

from faninsar.stack.config import StackConfig
from faninsar.stack.session import Stack
from faninsar.stack.preparation import (
    PreparedInputs, PreparationFailure, DataPreparationError,
)


class LocalMission(Stack):
    def _prepare_inputs(self, context):
        return PreparedInputs(sources={"20200101": (Path(self.request["source"]),)})


def test_configured_mission_has_one_explicit_preparation_hook(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"qualified local fixture")
    stack = LocalMission.configure(
        config=StackConfig(work_dir=tmp_path / "work"),
        request={"kind": "fixture", "source": str(source)},
    )
    assert not stack.data_ready
    report = stack.prepare_data()
    assert report.outcome == "complete"
    assert report.ready == ("20200101",)
    assert stack.data_ready
    assert stack.catalog.dates == ("20200101",)
    assert not hasattr(Stack, "prepare")
    assert not hasattr(Stack, "prepare_scenes")
    with pytest.raises(TypeError):
        Stack(config=stack.config)


class SubsetMission(Stack):
    def _prepare_inputs(self, context):
        sources = {}
        failures = []
        for target, source in self.request["sources"].items():
            if Path(source).exists():
                sources[target] = (Path(source),)
            else:
                failures.append(PreparationFailure(target, "source", "missing", FileNotFoundError(target)))
        return PreparedInputs(sources=sources, requested=tuple(self.request["sources"]), failures=tuple(failures))


def test_partial_strict_retry_and_recovery_preserve_original_request(tmp_path):
    sources = {target: tmp_path / target for target in ("20200101", "20200113", "20200125")}
    for target in ("20200101", "20200113"):
        sources[target].write_bytes(b"qualified")
    stack = SubsetMission.configure(config=StackConfig(work_dir=tmp_path / "work"), request={"sources": sources})
    report = stack.prepare_data()
    assert report.outcome == "partial"
    assert report.requested == ("20200101", "20200113", "20200125")
    assert report.ready == ("20200101", "20200113")
    assert report.failed[0].target == "20200125"
    assert stack.data_ready
    stack._ensure_prepared()
    committed = (tmp_path / "work" / "stack.json").read_bytes()
    config = stack.config
    with pytest.raises(DataPreparationError) as raised:
        stack.prepare_data(on_error="raise")
    assert isinstance(raised.value.__cause__, FileNotFoundError)
    assert stack.config is config
    assert stack.preparation_report is report
    assert stack.catalog.dates == ("20200101", "20200113")
    assert (tmp_path / "work" / "stack.json").read_bytes() == committed
    sources["20200125"].write_bytes(b"repaired")
    assert stack.prepare_data().outcome == "complete"
    assert stack.catalog.dates == ("20200101", "20200113", "20200125")
    assert stack.reference == "20200101"


def test_zero_ready_does_not_publish_success(tmp_path):
    stack = SubsetMission.configure(config=StackConfig(work_dir=tmp_path / "work"), request={"sources": {"20200101": tmp_path / "missing"}})
    report = stack.prepare_data()
    assert report.outcome == "failed"
    assert not stack.data_ready
    assert not (tmp_path / "work" / "stack.json").exists()
    assert (tmp_path / "work" / "inputs" / "request.json").is_file()


def test_s1_constructor_and_request_only_restore_do_no_source_work(tmp_path, monkeypatch):
    from faninsar.missions.s1 import S1Stack
    from faninsar.processing.geometry import ConstantDEM

    def forbidden(*args, **kwargs):
        pytest.fail("constructor/open performed source I/O")
    monkeypatch.setattr("faninsar.missions.s1.safe.open_safe_product", forbidden)
    source = tmp_path / "S1A_20200101.SAFE"
    stack = S1Stack.from_safes([source], work_dir=tmp_path / "work", dem=ConstantDEM(0))
    assert not stack.data_ready
    report = stack.prepare_data(offline=True)
    assert report.outcome == "failed"
    opened = Stack.open(stack.config.work_dir)
    assert type(opened) is S1Stack
    assert not opened.data_ready
    assert tuple(Path(path) for path in opened.request["paths"]) == (source,)


def test_nisar_one_survivor_undated_restore_and_retry(tmp_path, monkeypatch):
    from faninsar.missions.nisar import NISARStack
    from faninsar.processing.geometry import ConstantDEM
    from tests.missions.nisar.test_stack import _result
    from types import SimpleNamespace

    good, unavailable = tmp_path / "undated.h5", tmp_path / "later.h5"
    good.write_bytes(b"local RSLC")
    calls = []
    def open_product(sensor, uri, **kwargs):
        calls.append(uri)
        if not Path(uri).exists():
            raise FileNotFoundError(uri)
        return SimpleNamespace(filename=uri)
    monkeypatch.setattr("faninsar.missions.nisar.stack.NisarSensor.open_product", open_product)
    monkeypatch.setattr("faninsar.missions.nisar.stack.NisarSensor.to_slc_product", lambda sensor, handle, **kwargs: _result(Path(handle.filename), "20240101" if Path(handle.filename) == good else "20240113"))
    stack = NISARStack.from_rslc([good, unavailable], work_dir=tmp_path / "work", dem=ConstantDEM(0))
    assert calls == []
    report = stack.prepare_data()
    assert report.outcome == "partial"
    assert report.ready == ("20240101",)
    assert stack.data_ready
    assert len(stack.pairs) == 0
    calls.clear()
    opened = Stack.open(stack.config.work_dir)
    assert calls == []
    assert opened.catalog.dates == ("20240101",)
    assert opened.channel == ("B", "HH")
    assert opened.products["20240101"].acquisition_id == "20240101"
    unavailable.write_bytes(b"recovered RSLC")
    assert opened.prepare_data().outcome == "complete"
    assert opened.reference == "20240101"
    assert opened.catalog.dates == ("20240101", "20240113")


def test_missing_explicit_reference_is_later_prerequisite_and_recovers(tmp_path):
    from faninsar import Pairs
    sources = {target: tmp_path / target for target in ("20200101", "20200113", "20200125")}
    for target in ("20200101", "20200113"):
        sources[target].write_bytes(b"qualified")
    stack = SubsetMission.configure(config=StackConfig(work_dir=tmp_path / "work"), request={"sources": sources, "topology": {"reference": "20200125", "pairs": ("20200101_20200125",)}})
    report = stack.prepare_data()
    assert report.outcome == "partial"
    assert stack.data_ready
    assert stack.reference == "20200125"
    assert any("explicit reference" in item for item in report.prerequisites)
    assert any("endpoints" in item for item in report.prerequisites)
    with pytest.raises(RuntimeError, match="explicit reference"):
        stack._ensure_prepared()
    sources["20200125"].write_bytes(b"recovered")
    assert stack.prepare_data().outcome == "complete"
    assert stack.reference == "20200125"


def test_workspace_guard_is_exclusive_and_recovers_after_process_exit(tmp_path):
    import subprocess
    import sys
    from faninsar.stack.preparation import writer_guard
    root = tmp_path / "work"
    with writer_guard(root):
        with pytest.raises(RuntimeError, match="another preparation writer"):
            with writer_guard(root):
                pytest.fail("two writers acquired one workspace")
    child = subprocess.run([sys.executable, "-c", "from pathlib import Path; import sys, os; from faninsar.stack.preparation import writer_guard; lock=writer_guard(Path(sys.argv[1])); lock.__enter__(); os._exit(0)", str(root)], check=False, cwd=Path(__file__).parents[2])
    assert child.returncode == 0
    with writer_guard(root):
        pass


def test_s1_missing_required_eof_affects_only_dependent_acquisition(tmp_path, monkeypatch):
    from faninsar.missions.s1 import S1Stack
    from faninsar.stack.search import MissionPreparedInputs
    from faninsar.processing.geometry import ConstantDEM
    paths = []
    for target in ("20200101", "20200113"):
        path = tmp_path / f"S1A_{target}.SAFE"
        path.mkdir()
        paths.append(path)
    eof = tmp_path / "qualified.EOF"
    eof.write_bytes(b"orbit fixture")
    monkeypatch.setattr(S1Stack, "_bind_precise_orbits", lambda self: None)
    monkeypatch.setattr(S1Stack, "_track_metadata", lambda self: None)
    monkeypatch.setattr(S1Stack, "_local_mission_inputs", lambda self: MissionPreparedInputs(orbit_paths_by_acquisition={"20200101": eof} if self.catalog.dates == ("20200101",) else {}))
    stack = S1Stack.from_safes(paths, work_dir=tmp_path / "work", dem=ConstantDEM(0))
    report = stack.prepare_data(offline=True)
    assert report.requested == ("20200101", "20200113")
    assert report.ready == ("20200101",)
    assert report.failed[0].target == "20200113"
    assert stack.data_ready
    embedded = S1Stack.from_safes(paths, work_dir=tmp_path / "embedded", dem=ConstantDEM(0), extra={"orbit_auto_fetch": False})
    assert embedded.prepare_data(offline=True).outcome == "complete"


def test_pending_request_and_attempt_records_never_persist_secrets(tmp_path):
    stack = SubsetMission.configure(config=StackConfig(work_dir=tmp_path / "work", extra={"password": "do-not-persist", "endpoint": "https://user:secret@example.com/data?token=secret"}), request={"sources": {"20200101": tmp_path / "missing"}, "access_token": "private-token"})
    stack.prepare_data()
    text = (tmp_path / "work" / "inputs" / "request.json").read_text()
    assert "do-not-persist" not in text
    assert "private-token" not in text
    assert "user:secret" not in text
    assert "token=secret" not in text


@pytest.mark.parametrize("strict", [False, True])
def test_offline_source_dem_cache_miss_never_reaches_transport(tmp_path, monkeypatch, strict):
    from affine import Affine
    from faninsar.processing.geometry import DEM, GridSpec
    source = tmp_path / "source"
    source.write_bytes(b"local")
    grid = GridSpec("EPSG:4326", Affine(0.001, 0, 10, 0, -0.001, 50), height=2, width=2)
    stack = LocalMission.configure(config=StackConfig(work_dir=tmp_path / "work", grid=grid, dem=DEM.from_source("glo30", cache_dir=tmp_path / "empty")), request={"source": source})
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("offline preparation reached DEM network fetch_plan")
    monkeypatch.setattr("faninsar.processing.geometry.dem_transport.fetch_plan", forbidden)
    if strict:
        with pytest.raises(DataPreparationError) as raised:
            stack.prepare_data(offline=True, on_error="raise")
        assert raised.value.report.outcome == "failed"
    else:
        assert stack.prepare_data(offline=True).outcome == "failed"
    assert calls == []
    assert not stack.data_ready


def test_offline_source_dem_reuses_real_cached_tiles(tmp_path, monkeypatch):
    import numpy as np
    import rasterio
    from affine import Affine
    from faninsar.processing.geometry import DEM, GridSpec
    from faninsar.processing.geometry.dem_sources import get_dem_source
    from faninsar.processing.geometry.providers.p0030_adapter import _bounds
    grid = GridSpec("EPSG:4326", Affine(0.001, 0, 10.5, 0, -0.001, 50.5), height=2, width=2)
    cache = tmp_path / "cache"
    entry = get_dem_source("glo30")
    for tile in entry.plan(_bounds(grid)).tiles:
        path = cache / tile.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=Affine(0.01, 0, 10, 0, -0.01, 51)) as dataset:
            dataset.write(np.full((128, 128), 42, dtype="float32"), 1)
        with path.open("ab") as padded:
            padded.write(b"\0" * max(0, tile.min_bytes - path.stat().st_size))
    source = tmp_path / "source"
    source.write_bytes(b"local")
    def forbidden(*args, **kwargs):
        pytest.fail("offline cache hit attempted transport")
    monkeypatch.setattr("faninsar.processing.geometry.dem_transport.fetch_plan", forbidden)
    monkeypatch.setattr("faninsar.processing.geometry.providers.p0030_adapter._head_status", forbidden)
    # Local normalized datum fixture keeps this test about real source-cache lookup.
    monkeypatch.setattr("faninsar.processing.geometry.datum.convert_heights", lambda values, *args, **kwargs: values)
    stack = LocalMission.configure(config=StackConfig(work_dir=tmp_path / "work", grid=grid, dem=DEM.from_source("glo30", cache_dir=cache)), request={"source": source})
    assert stack.prepare_data(offline=True).outcome == "complete"
    assert stack.data_ready


def test_strict_retry_preserves_dem_product_and_identity_bytes(tmp_path):
    from affine import Affine
    from faninsar.processing.geometry import DEM, GridSpec
    from faninsar.stack.preparation import writer_guard
    sources = {target: tmp_path / target for target in ("20200101", "20200113")}
    sources["20200101"].write_bytes(b"ready")
    grid = GridSpec("EPSG:4326", Affine(0.001, 0, 10, 0, -0.001, 50), height=2, width=2)
    stack = SubsetMission.configure(config=StackConfig(work_dir=tmp_path / "work", grid=grid, dem=DEM.from_constant(7)), request={"sources": sources})
    stack.prepare_data()
    artifact = stack.config.work_dir / "dem.tif"
    stack.config.dem.to_raster(grid).save(artifact)
    product = stack.config.work_dir / "coreg" / "product.bin"
    product.parent.mkdir()
    product.write_bytes(b"committed scientific payload")
    identity = stack.config.work_dir / "stack.json"
    before = {path: path.read_bytes() for path in (artifact, product, identity)}
    with pytest.raises(DataPreparationError):
        stack.prepare_data(on_error="raise", offline=True)
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert stack.data_ready
    assert stack.config.grid == grid


def test_cancellation_joins_real_writer_before_workspace_release(tmp_path):
    import threading
    from faninsar.stack.preparation import writer_guard
    cleanup_started, release_writer = threading.Event(), threading.Event()
    root = tmp_path / "work"
    artifact = tmp_path / "qualified.bin"
    artifact.write_bytes(b"prior durable artifact")
    class CancelMission(Stack):
        def _prepare_inputs(self, context):
            def writer():
                cleanup_started.set()
                release_writer.wait(5)
                (root / "writer-finished").write_bytes(b"durable checkpoint")
            thread = threading.Thread(target=writer)
            thread.start()
            try:
                raise KeyboardInterrupt()
            finally:
                thread.join()
    stack = CancelMission.configure(config=StackConfig(work_dir=root), request={"source": artifact})
    caught = []
    def prepare():
        try:
            stack.prepare_data()
        except KeyboardInterrupt:
            caught.append(True)
    task = threading.Thread(target=prepare)
    task.start()
    try:
        assert cleanup_started.wait(5)
        with pytest.raises(RuntimeError, match="another preparation writer"):
            with writer_guard(root):
                pytest.fail("lock released before actual writer joined")
    finally:
        release_writer.set()
        task.join(5)
    assert not task.is_alive()
    assert caught == [True]
    assert artifact.read_bytes() == b"prior durable artifact"
    assert (root / "writer-finished").is_file()
    assert not (root / "stack.json").exists()
    with writer_guard(root):
        pass


def test_recovered_member_retains_grid_and_committed_dem_artifact(tmp_path):
    from affine import Affine
    from faninsar.processing.geometry import DEM, GridSpec
    sources = {target: tmp_path / target for target in ("20200101", "20200113")}
    sources["20200101"].write_bytes(b"ready")
    grid = GridSpec("EPSG:4326", Affine(0.001, 0, 10, 0, -0.001, 50), height=2, width=2)
    stack = SubsetMission.configure(config=StackConfig(work_dir=tmp_path / "work", grid=grid, dem=DEM.from_constant(7)), request={"sources": sources})
    assert stack.prepare_data().outcome == "partial"
    dem = stack.config.work_dir / "dem.tif"
    stack.config.dem.to_raster(grid).save(dem)
    before = dem.read_bytes()
    sources["20200113"].write_bytes(b"repaired")
    assert stack.prepare_data(offline=True).outcome == "complete"
    assert stack.config.grid == grid
    assert stack.reference == "20200101"
    assert dem.read_bytes() == before


def test_offline_geoid_miss_cannot_use_live_transport(tmp_path, monkeypatch):
    from affine import Affine
    from faninsar.processing.geometry import DEM, GridSpec
    from faninsar.processing.geometry import fetch as geoid_fetch
    from dataclasses import replace
    source = tmp_path / "source"
    source.write_bytes(b"local")
    monkeypatch.setenv("FANINSAR_GEOID_CACHE", str(tmp_path / "geoid-empty"))
    def forbidden(*args, **kwargs):
        pytest.fail("offline datum conversion attempted geoid acquisition")
    monkeypatch.setattr(geoid_fetch, "_requests_transport", forbidden)
    grid = GridSpec("EPSG:4326", Affine(0.001, 0, 10, 0, -0.001, 50), height=2, width=2)
    stack = LocalMission.configure(config=StackConfig(work_dir=tmp_path / "work", grid=grid, dem=DEM.from_constant(7, vertical_datum="egm2008")), request={"source": source})
    # A source raster requiring datum conversion exercises the geoid boundary directly.
    raster_path = tmp_path / "orthometric.tif"
    stack._request_config.dem.to_raster(GridSpec("EPSG:4326", Affine(0.001, 0, 9.99, 0, -0.001, 50.01), height=30, width=30), vertical_datum="egm2008").save(raster_path)
    stack._request_config.dem = DEM.from_raster(raster_path, vertical_datum="egm2008")
    assert stack.prepare_data(offline=True).outcome == "failed"


@pytest.mark.parametrize("failure_policy", [None, "warn", "raise"])
def test_expanded_recovered_coverage_uses_new_dem_preserving_prior_bytes(tmp_path, monkeypatch, failure_policy):
    import numpy as np
    from affine import Affine
    from faninsar.processing.geometry import DEM, GridSpec, RasterDEM, SourceDEM
    from faninsar.stack.preparation import PreparedInputs
    first, recovered = tmp_path / "first", tmp_path / "recovered"
    first.write_bytes(b"ready")
    initial = {f"202001{day:02d}": tmp_path / f"source-{day}" for day in range(1, 10)}
    for path in initial.values():
        path.write_bytes(b"qualified")
    public_grid = GridSpec("EPSG:4326", Affine(0.01, 0, 10, 0, -0.01, 40.1), height=10, width=10)
    class CoverageMission(Stack):
        def _prepare_inputs(self, context):
            sources = {key: (path,) for key, path in initial.items()}
            failures = (PreparationFailure("20200113", "source", "missing", FileNotFoundError()),)
            footprint = (9.9, 39.9, 10.2, 40.2)
            if recovered.exists():
                sources["20200113"] = (recovered,)
                failures = ()
                footprint = (9.5, 39.5, 10.5, 40.5)
            from faninsar.stack.search import MissionPreparedInputs
            return PreparedInputs(sources=sources, requested=(*initial, "20200113"), failures=failures, mission=MissionPreparedInputs(selected_footprints=(footprint,)))
    monkeypatch.setattr(SourceDEM, "to_raster", lambda self, grid, **kwargs: RasterDEM(array=np.full(grid.shape, 12, dtype="float32"), grid=grid, vertical_datum="ellipsoidal"))
    stack = CoverageMission.configure(config=StackConfig(work_dir=tmp_path / "work", grid=public_grid, dem=DEM.from_source("glo30", cache_dir=tmp_path / "cache")), request={"source": first})
    assert stack.prepare_data().outcome == "partial"
    paths = tuple(path for path in stack.config.work_dir.iterdir() if path.name in {"dem.tif", "dem_wgs84.tif", "dem.json", "stack.json"})
    product = stack.config.work_dir / "committed-product.bin"
    product.write_bytes(b"prior scientific payload with original lineage")
    before = {path: path.read_bytes() for path in (*paths, product)}
    old_science = stack._coreg_science_config()

    recovered.write_bytes(b"recovered")
    old_geographic = stack._stack_wgs84_raster_path()
    original_report = stack.preparation_report
    if failure_policy is not None:
        original_materialize = SourceDEM.to_raster
        def unavailable(*args, **kwargs):
            raise OSError("injected expanded DEM failure before commit")
        monkeypatch.setattr(SourceDEM, "to_raster", unavailable)
        if failure_policy == "raise":
            with pytest.raises(DataPreparationError):
                stack.prepare_data(on_error="raise")
        else:
            assert stack.prepare_data().outcome == "failed"
        assert stack.preparation_report is original_report
        assert stack.data_ready
        assert stack.catalog.dates == tuple(initial)
        assert all(path.read_bytes() == payload for path, payload in before.items())
        monkeypatch.setattr(SourceDEM, "to_raster", original_materialize)
    assert stack.prepare_data(on_error="raise").outcome == "complete"
    assert all(path.read_bytes() == payload for path, payload in before.items() if path.name != "stack.json")
    assert stack.catalog.dates == (*initial, "20200113")
    assert stack.config.grid == public_grid
    assert stack._stack_wgs84_raster_path() != old_geographic
    assert stack._stack_wgs84_raster_path().is_file()
    assert stack._dem_wgs84.grid.bounds[0] <= 9.5
    assert stack._dem_wgs84.grid.bounds[2] >= 10.5
    # Expanded science is not silently relabeled as the old product lineage.
    assert stack._coreg_science_config() != old_science
    from faninsar.missions import base as registry
    monkeypatch.setitem(registry._MISSION_STACKS, "CoverageMission", (__name__, "_coverage_mission"))
    monkeypatch.setattr(__import__(__name__, fromlist=["x"]), "_coverage_mission", CoverageMission, raising=False)
    opened = Stack.open(stack.config.work_dir)
    assert opened.catalog.dates == (*initial, "20200113")
    assert opened._stack_wgs84_raster_path() == stack._stack_wgs84_raster_path()
    assert opened._dem_wgs84.grid == stack._dem_wgs84.grid


@pytest.mark.parametrize("damage", ["delete", "truncate"])
def test_complete_reuse_repairs_deleted_required_dem(tmp_path, monkeypatch, damage):
    import numpy as np
    from affine import Affine
    from faninsar.processing.geometry import DEM, GridSpec, RasterDEM, SourceDEM
    source = tmp_path / "source"
    source.write_bytes(b"qualified")
    grid = GridSpec("EPSG:4326", Affine(.001, 0, 10, 0, -.001, 50), height=2, width=2)
    calls = []
    def materialize(self, grid, **kwargs):
        calls.append(True)
        return RasterDEM(array=np.full(grid.shape, 7, dtype="float32"), grid=grid, vertical_datum="ellipsoidal")
    monkeypatch.setattr(SourceDEM, "to_raster", materialize)
    stack = LocalMission.configure(config=StackConfig(work_dir=tmp_path / "work", grid=grid, dem=DEM.from_source("glo30", cache_dir=tmp_path / "cache")), request={"source": source})
    assert stack.prepare_data(offline=True).complete
    for _ in range(2):
        if damage == "delete":
            stack._stack_dem_raster_path().unlink()
        else:
            stack._stack_dem_raster_path().write_bytes(b"corrupt")
        assert not stack.data_ready
        count = len(calls)
        assert stack.prepare_data(offline=True).complete
        assert len(calls) > count
        assert stack._stack_dem_raster_path().is_file()
        assert stack.data_ready


def test_changed_source_cannot_reuse_complete_report(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"qualified")
    stack = LocalMission.configure(config=StackConfig(work_dir=tmp_path / "work"), request={"source": source})
    stack.prepare_data()
    source.write_bytes(b"truncated")
    assert not stack.data_ready
    source.write_bytes(b"x")
    assert stack.prepare_data(offline=True).outcome == "failed"
    assert not stack.data_ready


@pytest.mark.parametrize("kind", ["orbit", "aux"])
@pytest.mark.parametrize("damage", ["delete", "truncate"])
def test_auxiliary_damage_invalidates_ready_and_only_dependent_target(tmp_path, kind, damage):
    from faninsar.stack.search import MissionPreparedInputs
    paths = {target: tmp_path / target for target in ("20200101", "20200113")}
    aux = tmp_path / "required-auxiliary"
    for path in (*paths.values(), aux):
        path.write_bytes(b"qualified")
    class BoundMission(Stack):
        def _prepare_inputs(self, context):
            bindings = {"orbit_paths_by_acquisition" if kind == "orbit" else "aux_cal_paths_by_acquisition": {"20200101": aux}}
            return PreparedInputs(sources={key: (path,) for key, path in paths.items()}, requested=tuple(paths), mission=MissionPreparedInputs(**bindings))
    stack = BoundMission.configure(config=StackConfig(work_dir=tmp_path / "work"), request={"sources": paths})
    assert stack.prepare_data().complete
    if damage == "delete":
        aux.unlink()
    else:
        aux.write_bytes(b"x")
    assert not stack.data_ready
    report = stack.prepare_data(offline=True)
    assert report.outcome == "partial"
    assert report.ready == ("20200113",)
    assert report.failed[0].target == "20200101"
    assert stack.data_ready


def test_missing_dem_cache_cannot_fall_back_to_stale_in_memory_raster(tmp_path, monkeypatch):
    import numpy as np
    from affine import Affine
    from faninsar.processing.geometry import DEM, GridSpec, RasterDEM, SourceDEM
    source = tmp_path / "source"
    source.write_bytes(b"qualified")
    grid = GridSpec("EPSG:4326", Affine(.001, 0, 10, 0, -.001, 50), height=2, width=2)
    original = SourceDEM.to_raster
    monkeypatch.setattr(SourceDEM, "to_raster", lambda self, grid, **kwargs: RasterDEM(array=np.full(grid.shape, 7, dtype="float32"), grid=grid, vertical_datum="ellipsoidal"))
    stack = LocalMission.configure(config=StackConfig(work_dir=tmp_path / "work", grid=grid, dem=DEM.from_source("glo30", cache_dir=tmp_path / "empty-cache")), request={"source": source})
    stack.prepare_data(offline=True)
    stack._stack_dem_raster_path().write_bytes(b"corrupted")
    assert not stack.data_ready
    monkeypatch.setattr(SourceDEM, "to_raster", original)
    monkeypatch.setattr("faninsar.processing.geometry.dem_transport.fetch_plan", lambda *args, **kwargs: pytest.fail("offline repair attempted network"))
    for _ in range(2):
        assert stack.prepare_data(offline=True).outcome == "failed"
        assert not stack.data_ready


def test_damaged_shared_artifact_keeps_product_only_open_local(tmp_path, monkeypatch):
    import numpy as np
    from affine import Affine
    from faninsar.processing.geometry import DEM, GridSpec, RasterDEM, SourceDEM
    from faninsar.missions import base as registry
    monkeypatch.setitem(registry._MISSION_STACKS, "LocalMission", (__name__, "LocalMission"))
    source = tmp_path / "source"
    source.write_bytes(b"qualified")
    grid = GridSpec("EPSG:4326", Affine(.001, 0, 10, 0, -.001, 50), height=2, width=2)
    monkeypatch.setattr(SourceDEM, "to_raster", lambda self, grid, **kwargs: RasterDEM(array=np.full(grid.shape, 7, dtype="float32"), grid=grid, vertical_datum="ellipsoidal"))
    stack = LocalMission.configure(config=StackConfig(work_dir=tmp_path / "work", grid=grid, dem=DEM.from_source("glo30", cache_dir=tmp_path / "cache")), request={"source": source})
    stack.prepare_data(offline=True)
    from tests.support.scene_stores import write_test_scene
    write_test_scene(stack.config.work_dir, date_id="20200101", reference_id="20200101", payload=np.ones((2,2), dtype="complex64"))
    stack._stack_dem_raster_path().write_bytes(b"broken")
    opened = Stack.open(stack.config.work_dir)
    assert not opened.data_ready
    assert "20200101" in opened.scenes
    assert opened.prepare_data(offline=True).complete
    assert opened.data_ready
