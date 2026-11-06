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
