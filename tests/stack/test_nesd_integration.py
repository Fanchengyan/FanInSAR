"""Session integration of durable NESD measurement and final correction.

These tests exercise the public Stack seams: per-edge measurement commits,
resume after a later edge failure, graph qualification before apply, blocked
date exclusion, effective per-date correction reuse, strict scene baselines,
the shared writer lock and interrupted-attempt reconciliation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.support.local_stack import LocalStack
from tests.support.scene_stores import write_test_scene

from faninsar import Pairs
from faninsar.missions.s1 import S1Stack
from faninsar.processing.coregistration.misreg_network import MisregArc
from faninsar.stack.nesd_recovery import PairMeasurementStore
from faninsar.stack.preparation import PreparedInputs
from faninsar.stack.provider import StackSceneProvider
from faninsar.stack.reporting import ReportWriter
from faninsar.stack.search import MissionPreparedInputs
from faninsar.stack.source_integrity import establish_source_integrity


def _write_safe_payload(source: Path) -> None:
    """Give a fixture SAFE directory one minimal declared scientific tree."""
    (source / "manifest.safe").write_bytes(b"<safe/>")
    for name in ("annotation", "measurement", "calibration"):
        (source / name).mkdir(exist_ok=True)
    (source / "annotation" / "a.xml").write_bytes(b"<annotation/>")
    (source / "measurement" / "a.tiff").write_bytes(b"measurement")
    (source / "calibration" / "a.xml").write_bytes(b"<calibration/>")


def _stack(
    tmp_path: Path,
    days: tuple[str, ...] = ("20160101", "20160113"),
    *,
    name: str = "out",
    misreg_pairs: list[str] | None = None,
) -> S1Stack:
    """Build a prepared NESD Stack with minimal SAFE source trees."""
    paths = []
    for day in days:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir(exist_ok=True)
        _write_safe_payload(path)
        paths.append(path)
    stack = LocalStack.from_paths(
        paths,
        work_dir=tmp_path / name,
        coreg_mode="nesd",
        pair_max_interval=10,
        pair_max_days=120,
    )
    stack.prepare_data()
    stack._date_track_key = lambda _date: "1A"  # type: ignore[method-assign]
    stack._ensure_precise_orbits = lambda: None  # type: ignore[method-assign]
    if misreg_pairs is not None:
        stack._inputs_state = replace(
            stack._inputs_state,
            misreg_pairs=Pairs.from_names(misreg_pairs),
        )
    return stack


def _arc(
    primary: str, secondary: str, *, az: float | None, rg: float | None
) -> MisregArc:
    """Return one finite pair measurement."""
    return MisregArc(
        primary=primary,
        secondary=secondary,
        azimuth_shift_px=az,
        range_shift_px=rg,
        azimuth_sigma_px=0.01 if az is not None else 1.0,
        range_sigma_px=0.01 if rg is not None else 1.0,
        method="overlap",
        n_valid=10,
    )


def _install_measure(
    monkeypatch: pytest.MonkeyPatch,
    result: Any,
) -> list[str]:
    """Install one measured-edge callback and return its call log."""
    calls: list[str] = []

    def fake_measure(_primary: object, _secondary: object, **_kwargs: object) -> Any:
        pair = f"{Path(str(_primary)).name.split('_')[5][:8]}_" + (
            f"{Path(str(_secondary)).name.split('_')[5][:8]}"
        )
        calls.append(pair)
        return result(pair)

    monkeypatch.setattr(
        "faninsar.missions.s1.measure.measure_s1_pair_misreg", fake_measure
    )
    return calls


def _forbid_pair_engine(stack: S1Stack) -> list[str]:
    """Fail if measure or apply reaches the full pair engine."""
    calls: list[str] = []

    def produce_pair(*_args: object, **_kwargs: object) -> None:
        calls.append("produce_pair")
        message = "NESD integration must not use the full pair engine"
        raise AssertionError(message)

    stack.scene_provider = StackSceneProvider(name="TEST", produce_pair=produce_pair)
    return calls


class _FakeReferenceContext(dict):
    """Reference context stub that still accepts ``context.dem`` assignment."""


def _install_apply(
    stack: S1Stack,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    *,
    fail_on: str | None = None,
    events: list[str] | None = None,
) -> None:
    """Replace Reference preparation and secondary apply with scene writers."""

    def fake_prepare(_reference: object, **kwargs: object) -> _FakeReferenceContext:
        output = Path(str(kwargs["output_scene"]))
        if not (output / "item.json").is_file():
            write_test_scene(
                stack.config.work_dir,
                date_id=stack.reference,
                reference_id=stack.reference,
                payload=np.ones((2, 2), dtype=np.complex64),
                tag="IW1_b0",
            )
        return _FakeReferenceContext(prepare=kwargs)

    def fake_secondary(
        _context: object,
        secondary_path: object,
        *,
        output_scene: Path,
        **kwargs: object,
    ) -> None:
        date_id = Path(str(secondary_path)).name.split("_")[5][:8]
        calls.append(date_id)
        if events is not None:
            events.append(f"coreg:{date_id}")
        if fail_on is not None and date_id == fail_on:
            message = f"injected apply failure for {date_id}"
            raise RuntimeError(message)
        write_test_scene(
            stack.config.work_dir,
            date_id=date_id,
            reference_id=stack.reference,
            payload=np.ones((2, 2), dtype=np.complex64),
            tag="IW1_b0",
            properties=kwargs.get("properties"),
            inputs=kwargs.get("inputs"),
        )

    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.prepare_s1_reference", fake_prepare
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.coregister_s1_secondary", fake_secondary
    )


def test_later_edge_failure_preserves_committed_edges_and_rerun_retries_only_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _stack(
        tmp_path,
        ("20160101", "20160113", "20160125"),
        misreg_pairs=["20160101_20160113", "20160113_20160125"],
    )
    _forbid_pair_engine(stack)
    fail = {"value": True}

    def measured(pair: str) -> MisregArc:
        if pair == "20160113_20160125" and fail["value"]:
            message = "injected overlap failure"
            raise RuntimeError(message)
        return _arc(*pair.split("_"), az=0.01, rg=0.02)

    calls = _install_measure(monkeypatch, measured)
    stack.measure_misreg()

    assert calls == ["20160101_20160113", "20160113_20160125"]
    assert [arc.pair_id for arc in stack.arcs] == ["20160101_20160113"]
    assert stack._misreg_measure_failures == {
        "20160113_20160125": "RuntimeError: injected overlap failure"
    }
    store = PairMeasurementStore(stack._misreg_store().root)
    spec = stack._misreg_edge_spec("20160101", "20160113")
    assert store.load("20160101_20160113", spec) is not None

    fail["value"] = False
    stack.measure_misreg()
    assert calls == [
        "20160101_20160113",
        "20160113_20160125",
        "20160113_20160125",
    ]
    assert [arc.pair_id for arc in stack.arcs] == [
        "20160101_20160113",
        "20160113_20160125",
    ]


def test_changed_endpoint_invalidates_only_its_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _stack(
        tmp_path,
        ("20160101", "20160113", "20160125"),
        misreg_pairs=["20160101_20160113", "20160101_20160125"],
    )
    _forbid_pair_engine(stack)
    calls = _install_measure(
        monkeypatch, lambda pair: _arc(*pair.split("_"), az=0.01, rg=None)
    )
    stack.measure_misreg()
    assert len(calls) == 2

    changed = Path("/new/orbits/20160113.EOF")
    stack.config.extra["orbit_paths"] = {"20160113": changed}
    stack.measure_misreg()

    assert calls == [
        "20160101_20160113",
        "20160101_20160125",
        "20160101_20160113",
    ]
    assert [arc.pair_id for arc in stack.arcs] == [
        "20160101_20160113",
        "20160101_20160125",
    ]


def test_blocked_disconnected_date_is_excluded_from_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _stack(
        tmp_path,
        ("20160101", "20160113", "20160125"),
        misreg_pairs=["20160101_20160113", "20160113_20160125"],
    )
    _forbid_pair_engine(stack)

    def measured(pair: str) -> MisregArc:
        if pair == "20160113_20160125":
            message = "injected overlap failure"
            raise RuntimeError(message)
        return _arc(*pair.split("_"), az=0.01, rg=0.02)

    _install_measure(monkeypatch, measured)
    stack.measure_misreg()
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls)

    stack.coregister_scenes()

    assert calls == ["20160113"]
    assert stack.date_misreg is not None
    assert "20160125" in stack._misreg_blocked_dates()
    latest = stack.coregistration_report
    assert latest is not None
    blocked = [r for r in latest.records if r.status == "blocked"]
    assert any(r.unit_id == "20160125" for r in blocked)


def test_quality_rejected_azimuth_blocks_target_despite_valid_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured azimuth quality rejection blocks the date instead of zeroing it."""
    stack = _stack(tmp_path)
    _forbid_pair_engine(stack)
    rejected = replace(
        _arc("20160101", "20160113", az=None, rg=0.02),
        azimuth_omission="quality_rejected",
    )
    _install_measure(monkeypatch, lambda _pair: rejected)
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls)

    stack.measure_misreg()
    assert stack._nesd_skipped_no_windows is False
    stack.coregister_scenes()

    assert calls == []
    assert stack.date_misreg is not None
    assert stack.date_misreg.azimuth_px == {}
    assert stack.date_misreg.range_px["20160113"] == pytest.approx(0.02)
    assert stack._misreg_blocked_dates() == ("20160113",)
    latest = stack.coregistration_report
    assert latest is not None
    blocked = [r for r in latest.records if r.status == "blocked"]
    assert any(
        r.unit_id == "20160113" and r.reason_code == "nesd_quality_rejected"
        for r in blocked
    )


def test_equal_effective_correction_reuses_scene_but_new_correction_reapplies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _stack(tmp_path)
    _forbid_pair_engine(stack)
    _install_measure(monkeypatch, lambda pair: _arc(*pair.split("_"), az=0.01, rg=0.02))
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls)

    stack.measure_misreg()
    stack.coregister_scenes()
    assert calls == ["20160113"]
    first_scene = stack.scenes["20160113"]
    historical_solution = first_scene.inputs["misreg"]

    # A new aggregate solution id with the same effective correction must not
    # invalidate the consumed scene.
    stack.date_misreg = replace(
        stack.date_misreg,
        metadata={**dict(stack.date_misreg.metadata), "provenance_note": "new"},
    )
    stack._persist_misreg()
    stack._misreg_solution = None
    stack._misreg_solution_key = None
    stack.coregister_scenes()
    assert calls == ["20160113"]
    assert stack.scenes["20160113"].write_id == first_scene.write_id
    assert stack.scenes["20160113"].inputs["misreg"] == historical_solution

    # A changed measured correction re-solves and reapplies exactly one scene.
    stack.arcs = [replace(stack.arcs[0], range_shift_px=0.25, range_sigma_px=0.01)]
    stack._misreg_solution = None
    stack._misreg_solution_key = None
    stack.coregister_scenes()
    assert calls == ["20160113", "20160113"]


def test_interrupted_apply_rerun_skips_already_committed_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _stack(tmp_path, ("20160101", "20160113", "20160125"))
    stack.config.coreg_mode = "geometry"
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls, fail_on="20160125")

    with pytest.raises(RuntimeError, match="injected apply failure"):
        stack.coregister_scenes()
    assert calls == ["20160113", "20160125"]

    _install_apply(stack, monkeypatch, calls)
    stack.coregister_scenes()
    assert calls == ["20160113", "20160125", "20160125"]


def test_strict_verification_recomputes_unverifiable_legacy_scene_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _stack(tmp_path)
    stack.config.coreg_mode = "geometry"
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls)
    stack.coregister_scenes()
    assert calls == ["20160113"]

    # Strict source qualification requires an explicit trusted baseline. Record
    # one real baseline per prepared local source; verification never fabricates
    # a baseline from the current bytes.
    root = stack.config.work_dir / "integrity" / "sources"
    for paths in stack.catalog.paths.values():
        for source in paths:
            establish_source_integrity(root, source)

    stack.verification = "strict"
    stack.coregister_scenes()
    assert calls == ["20160113", "20160113"]

    stack.coregister_scenes()
    assert calls == ["20160113", "20160113"]


def test_second_writer_is_rejected_while_a_stage_owns_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    stack = _stack(tmp_path)
    other = _stack(tmp_path, name="other")
    other.config.work_dir = stack.config.work_dir
    _install_measure(monkeypatch, lambda pair: _arc(*pair.split("_"), az=0.01, rg=0.01))
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with stack._writer_locked(stage="test"):
            held.set()
            assert release.wait(timeout=10)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(hold)
        assert held.wait(timeout=10)
        try:
            with pytest.raises(RuntimeError, match="another preparation writer"):
                other.measure_misreg()
        finally:
            release.set()
        future.result(timeout=10)


def test_unfinished_attempt_is_reconciled_into_the_next_locked_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _stack(tmp_path)
    stack.config.coreg_mode = "geometry"
    abandoned = ReportWriter(
        stack.config.work_dir,
        intent_revision=stack.intent_revision,
        mode="geometry",
        grid="radar",
        reference=stack.reference,
        verification="fast",
    )
    abandoned.record(
        stage="coregistration",
        unit_id="20160113",
        status="pending",
    )
    interrupted_id = abandoned.report.attempt_id

    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls)
    stack.coregister_scenes()

    latest = stack.coregistration_report
    assert latest is not None
    assert latest.attempt_id != interrupted_id
    assert latest.scope == {"reconciled_interrupted_attempts": (interrupted_id,)}


def test_reference_only_selection_still_commits_the_reference_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A catalog with only the Reference ready still commits its anchor."""
    stack = _stack(tmp_path, ("20160101",))
    stack.config.coreg_mode = "geometry"
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls)

    stack.coregister_scenes()

    report = stack.coregistration_report
    assert report is not None
    assert calls == []
    reference = stack.reference_scene
    assert reference is not None
    assert (reference.root / "item.json").is_file()
    assert any(
        record.stage == "reference" and record.status == "completed"
        for record in report.records
    )


def _install_streaming_prepare(
    stack: S1Stack, dates: tuple[str, ...], order: list[str]
) -> None:
    """Emit one qualified source per callback, then the terminal full set."""
    full_sources = {date: stack.catalog.paths_for(date) for date in dates}

    def streaming_prepare(_self: S1Stack, context: Any) -> PreparedInputs:
        assert context.on_ready is not None
        for date in dates:
            order.append(f"prepare:{date}")
            context.on_ready(
                PreparedInputs(
                    sources={date: full_sources[date]},
                    requested=dates,
                    mission=MissionPreparedInputs(),
                )
            )
        order.append("prepare:terminal")
        return PreparedInputs(
            sources=full_sources,
            requested=dates,
            mission=MissionPreparedInputs(),
        )

    stack._prepare_inputs = streaming_prepare.__get__(stack)  # type: ignore[method-assign]


def test_combined_entry_coregisters_a_ready_date_before_terminal_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public combined entry consumes the streaming callback, not a batch."""
    stack = _stack(tmp_path)
    stack.config.coreg_mode = "geometry"
    stack._request_config = replace(stack._request_config, coreg_mode="geometry")
    stack.preparation_report = None
    (stack.config.work_dir / "inputs" / "request.json").unlink(missing_ok=True)
    order: list[str] = []
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls, events=order)
    dates = tuple(stack.catalog.dates)
    _install_streaming_prepare(stack, dates, order)

    report = stack.prepare_and_coregister()

    assert report is not None
    assert calls == ["20160113"]
    assert "coreg:20160113" in order
    assert order.index("coreg:20160113") < order.index("prepare:terminal")


def test_combined_selection_skips_unselected_and_forces_each_date_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit selection bounds callback work and is forced only once."""
    stack = _stack(tmp_path, ("20160101", "20160113", "20160125", "20160206"))
    stack.config.coreg_mode = "geometry"
    stack._request_config = replace(stack._request_config, coreg_mode="geometry")
    stack.preparation_report = None
    (stack.config.work_dir / "inputs" / "request.json").unlink(missing_ok=True)
    order: list[str] = []
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls, events=order)
    _install_streaming_prepare(stack, tuple(stack.catalog.dates), order)

    report = stack.prepare_and_coregister(
        dates=["20160113", "20160125"], overwrite=True
    )

    assert report is not None
    assert calls == ["20160113", "20160125"]
    assert "coreg:20160206" not in order


def test_combined_empty_selection_authorizes_no_scene_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit ``dates=[]`` skips even the streaming coregistration stage."""
    stack = _stack(tmp_path, ("20160101", "20160113", "20160125"))
    stack.config.coreg_mode = "geometry"
    stack._request_config = replace(stack._request_config, coreg_mode="geometry")
    stack.preparation_report = None
    (stack.config.work_dir / "inputs" / "request.json").unlink(missing_ok=True)
    order: list[str] = []
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls, events=order)
    _install_streaming_prepare(stack, tuple(stack.catalog.dates), order)

    report = stack.prepare_and_coregister(dates=[], overwrite=True)

    assert report is not None
    assert calls == []
    assert not any(event.startswith("coreg:") for event in order)


def test_combined_selection_on_a_configured_stack_resolves_reference_after_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dates`` is not resolved against a Reference before preparation."""
    from faninsar.processing.geometry import ConstantDEM

    paths = []
    for day in ("20160101", "20160113", "20160125"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir(exist_ok=True)
        paths.append(path)
    stack = LocalStack.from_safes(paths, work_dir=tmp_path / "out", dem=ConstantDEM(0))
    assert stack._inputs_state is None
    stack._ensure_precise_orbits = lambda: None  # type: ignore[method-assign]
    stack._date_track_key = lambda _date: "1A"  # type: ignore[method-assign]
    order: list[str] = []
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls, events=order)
    sources = {path.name.split("_")[5][:8]: (path,) for path in paths}
    requested = tuple(sources)

    def streaming_prepare(_self: S1Stack, context: Any) -> PreparedInputs:
        assert context.on_ready is not None
        for date in requested:
            context.on_ready(
                PreparedInputs(
                    sources={date: sources[date]},
                    requested=requested,
                    mission=MissionPreparedInputs(),
                )
            )
        return PreparedInputs(
            sources=sources,
            requested=requested,
            mission=MissionPreparedInputs(),
        )

    stack._prepare_inputs = streaming_prepare.__get__(stack)  # type: ignore[method-assign]

    report = stack.prepare_and_coregister(dates=["20160113"], overwrite=True)

    assert report is not None
    assert calls == ["20160113"]


def test_combined_nesd_measures_only_edges_with_two_ready_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The NESD callback measures an edge only once both endpoints arrived."""
    stack = _stack(tmp_path, ("20160101", "20160113", "20160125"))
    stack.preparation_report = None
    (stack.config.work_dir / "inputs" / "request.json").unlink(missing_ok=True)
    order: list[str] = []
    measured: list[str] = []

    def fake_measure(
        primary: object, secondary: object, **_kwargs: object
    ) -> MisregArc:
        pair = (
            f"{Path(str(primary)).name.split('_')[5][:8]}_"
            f"{Path(str(secondary)).name.split('_')[5][:8]}"
        )
        order.append(f"measure:{pair}")
        measured.append(pair)
        return _arc(*pair.split("_"), az=0.01, rg=0.02)

    monkeypatch.setattr(
        "faninsar.missions.s1.measure.measure_s1_pair_misreg", fake_measure
    )
    apply_calls: list[str] = []
    _install_apply(stack, monkeypatch, apply_calls)
    dates = tuple(stack.catalog.dates)
    _install_streaming_prepare(stack, dates, order)

    report = stack.prepare_and_coregister()

    assert report is not None
    for pair in measured:
        first, second = pair.split("_")
        assert order.index(f"prepare:{first}") < order.index(f"measure:{pair}")
        assert order.index(f"prepare:{second}") < order.index(f"measure:{pair}")
    assert set(measured) == {
        pair
        for first, second in stack._misreg_planned_pairs()
        for pair in (f"{first}_{second}",)
    }
    assert all(f"prepare:{date}" in order for date in dates)
    assert apply_calls == ["20160113", "20160125"]


def test_combined_callback_failure_preserves_the_original_error_and_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nested scene failure must surface unchanged from the combined entry."""
    stack = _stack(tmp_path)
    stack.config.coreg_mode = "geometry"
    stack._request_config = replace(stack._request_config, coreg_mode="geometry")
    stack.preparation_report = None
    (stack.config.work_dir / "inputs" / "request.json").unlink(missing_ok=True)
    order: list[str] = []
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls, fail_on="20160113", events=order)
    _install_streaming_prepare(stack, tuple(stack.catalog.dates), order)

    with pytest.raises(RuntimeError) as raised:
        stack.prepare_and_coregister()

    assert str(raised.value) == "injected apply failure for 20160113"
    latest = stack.coregistration_report
    assert latest is not None
    assert latest.outcome == "failed"
    assert not latest.unfinished
    assert [
        (record.stage, record.unit_id)
        for record in latest.records
        if record.status == "failed"
    ] == [
        ("coregistration", "20160113"),
        ("coregistration", "coregister_scenes"),
    ]


def test_report_write_failure_never_replaces_the_original_callback_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken diagnostic writer cannot mask the scientific failure."""
    from faninsar.stack import reporting

    stack = _stack(tmp_path)
    stack.config.coreg_mode = "geometry"
    stack._request_config = replace(stack._request_config, coreg_mode="geometry")
    stack.preparation_report = None
    (stack.config.work_dir / "inputs" / "request.json").unlink(missing_ok=True)
    order: list[str] = []
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls, fail_on="20160113", events=order)
    _install_streaming_prepare(stack, tuple(stack.catalog.dates), order)

    real_record = reporting.ReportWriter.record

    def broken_failed_record(_self: object, **kwargs: object) -> None:
        if kwargs.get("status") == "failed":
            message = "injected report write failure"
            raise reporting.ReportWriteError(message)
        return real_record(_self, **kwargs)

    def broken_finish(_self: object, **_kwargs: object) -> None:
        message = "injected report finalization failure"
        raise reporting.ReportWriteError(message)

    monkeypatch.setattr(reporting.ReportWriter, "record", broken_failed_record)
    monkeypatch.setattr(reporting.ReportWriter, "finish", broken_finish)

    with pytest.raises(RuntimeError) as raised:
        stack.prepare_and_coregister()

    assert str(raised.value) == "injected apply failure for 20160113"


def test_standalone_coregistration_failure_finishes_its_own_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A direct scene pass still finalizes the attempt it owns."""
    stack = _stack(tmp_path)
    stack.config.coreg_mode = "geometry"
    stack._request_config = replace(stack._request_config, coreg_mode="geometry")
    calls: list[str] = []
    _install_apply(stack, monkeypatch, calls, fail_on="20160113")

    with pytest.raises(RuntimeError, match="injected apply failure for 20160113"):
        stack.coregister_scenes()

    report = stack.coregistration_report
    assert report is not None
    assert report.outcome == "failed"
    assert not report.unfinished
