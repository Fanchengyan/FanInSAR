"""Per-date S1 qualification events overlap real acquisition futures (P0088).

``_acquire_dates`` must hand each completed date to the controlling thread so
that date can be qualified (SLC + EOF + AUX) and published to science while
the remaining acquisition futures keep running.  These tests drive the real
thread pool and real ``wait`` loop; only remote materialization and mission
binding are fakes, so no network is touched.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import faninsar.missions.s1.preparation as preparation_mod
import faninsar.missions.s1.stack as s1_stack_mod
from faninsar.missions.s1.stack import S1Stack
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.preparation import PreparedInputs
from tests.missions.s1.test_acquire_stage_parallel import (
    _context,
    _dates,
    _make_stack,
    _Recorder,
    _search,
)


def _source_tree(tmp_path: Path, targets: tuple[str, ...]) -> dict[str, Path]:
    """Create local SAFE-like directories whose names carry each date."""
    sources = {}
    for target in targets:
        path = tmp_path / f"{target}.SAFE"
        path.mkdir()
        sources[target] = path
    return sources


def _serialized_search(targets: tuple[str, ...]) -> dict[str, Any]:
    """Return the stable serialized search shape used for plan identity."""
    return {
        "items": [
            {
                "acquisition": {
                    "start": (
                        f"{target[:4]}-{target[4:6]}-{target[6:]}T00:00:00+00:00"
                    ),
                    "acquisition_id": target,
                }
            }
            for target in targets
        ]
    }


def _streaming_context(events: list[Any], **kwargs: Any) -> SimpleNamespace:
    """Return a context with the per-date observer the mission must call."""
    context = _context(**kwargs)
    context.on_ready = lambda inputs: events.append(inputs)
    return context


def test_ready_date_is_published_before_another_download_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The controlling thread runs a ready callback while a peer still fetches."""
    recorder = _Recorder()
    release_second = threading.Event()
    second_waited = threading.Event()
    sources = _source_tree(tmp_path, ("20161201", "20161202"))

    def block_second() -> None:
        second_waited.set()
        assert release_second.wait(5), "ready callback never released the peer"
        # The callback must run while this acquisition is still parked.

    stack = _make_stack(
        recorder,
        tmp_path,
        {"20161202": block_second},
        source_paths=sources,
    )
    monkeypatch.setattr(
        preparation_mod,
        "restore_search",
        lambda record, spatial: _search(_dates(2)),
    )
    monkeypatch.setattr(s1_stack_mod, "_configure_s1_bindings", lambda *a, **k: None)

    published: list[tuple[str, ...]] = []
    context = _streaming_context([])
    context.on_ready = lambda inputs: (
        published.append(tuple(inputs.sources)),
        release_second.set(),
    )

    result = preparation_mod.prepare_s1_inputs(stack, context)

    assert second_waited.is_set()
    assert release_second.is_set(), "date 20161201 was not published mid-acquisition"
    assert published[:1] == [("20161201",)], (
        "the first publish must contain only the date that finished first"
    )
    assert published[1] == ("20161202",)
    assert second_waited.is_set()
    assert result.failures == ()
    assert dict(result.sources) == {target: (path,) for target, path in sources.items()}
    assert result.requested == ("20161201", "20161202")


def test_partial_events_carry_the_full_requested_target_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every per-date handoff names the complete request, not the ready subset."""
    recorder = _Recorder()
    sources = _source_tree(tmp_path, ("20161201", "20161202", "20161203"))
    stack = _make_stack(recorder, tmp_path, {}, source_paths=sources)
    monkeypatch.setattr(
        preparation_mod,
        "restore_search",
        lambda record, spatial: _search(_dates(3)),
    )
    monkeypatch.setattr(s1_stack_mod, "_configure_s1_bindings", lambda *a, **k: None)
    events: list[PreparedInputs] = []
    context = _streaming_context(events)

    preparation_mod.prepare_s1_inputs(stack, context)

    assert {tuple(event.sources) for event in events} == {
        ("20161201",),
        ("20161202",),
        ("20161203",),
    }
    assert len(events) == 3
    assert {event.requested for event in events} == {
        ("20161201", "20161202", "20161203")
    }


def test_unqualified_date_never_reaches_the_science_observer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing EOF/AUX dependency suppresses that date's publish only."""
    recorder = _Recorder()
    sources = _source_tree(tmp_path, ("20161201", "20161202"))
    stack = _make_stack(recorder, tmp_path, {}, source_paths=sources)
    monkeypatch.setattr(
        preparation_mod,
        "restore_search",
        lambda record, spatial: _search(_dates(2)),
    )
    monkeypatch.setattr(s1_stack_mod, "_configure_s1_bindings", lambda *a, **k: None)

    def mission_inputs(self: Any) -> SimpleNamespace:
        if self.catalog.dates == ("20161202",):
            raise FileNotFoundError("required covering EOF is unavailable")
        return SimpleNamespace(
            selected_footprints=(),
            orbit_paths_by_acquisition={},
            aux_cal_paths_by_acquisition={},
            provenance={},
        )

    monkeypatch.setattr(type(stack), "_local_mission_inputs", mission_inputs)
    events: list[PreparedInputs] = []
    context = _streaming_context(events)

    result = preparation_mod.prepare_s1_inputs(stack, context)

    assert [tuple(event.sources) for event in events] == [("20161201",)]
    assert result.failures[0].target == "20161202"
    assert result.failures[0].phase == "mission_inputs"
    assert dict(result.sources) == {"20161201": (sources["20161201"],)}


def test_already_qualified_source_is_not_acquired_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A restart publishes a committed date without repeating its download."""
    recorder = _Recorder()
    sources = _source_tree(tmp_path, ("20161201", "20161202"))
    stack = _make_stack(
        recorder,
        tmp_path,
        {},
        source_paths={"20161202": sources["20161202"]},
    )
    stack._inputs_state = SimpleNamespace()
    stack.catalog = SceneCatalog({"20161201": (sources["20161201"],)})
    monkeypatch.setattr(
        preparation_mod,
        "restore_search",
        lambda record, spatial: _search(_dates(2)),
    )
    monkeypatch.setattr(s1_stack_mod, "_configure_s1_bindings", lambda *a, **k: None)
    events: list[PreparedInputs] = []
    context = _streaming_context(events)
    context.request["search"] = _serialized_search(("20161201", "20161202"))
    context.science = {}
    context.previous_document = {"source": dict(context.request), "science": {}}

    result = preparation_mod.prepare_s1_inputs(stack, context)

    assert recorder.calls == {"20161202": 1}, (
        "the committed date must not be downloaded again"
    )
    assert [tuple(event.sources) for event in events] == [
        ("20161201",),
        ("20161202",),
    ]
    assert dict(result.sources) == {target: (path,) for target, path in sources.items()}
    assert result.requested == ("20161201", "20161202")


def test_same_date_extended_roi_replans_instead_of_reusing_old_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An additive ROI/burst change for a committed date triggers materialize."""
    recorder = _Recorder()
    sources = _source_tree(tmp_path, ("20161201", "20161202"))
    stack = _make_stack(recorder, tmp_path, {}, source_paths=sources)
    stack._inputs_state = SimpleNamespace()
    stack.catalog = SceneCatalog({"20161201": (sources["20161201"],)})
    monkeypatch.setattr(
        preparation_mod,
        "restore_search",
        lambda record, spatial: _search(_dates(2)),
    )
    monkeypatch.setattr(s1_stack_mod, "_configure_s1_bindings", lambda *a, **k: None)

    def context_for(current, previous):
        context = _streaming_context([])
        context.request["search"] = _serialized_search(("20161201", "20161202"))
        context.science = current
        context.previous_document = {
            "source": dict(context.request),
            "science": previous,
        }
        return context

    preparation_mod.prepare_s1_inputs(
        stack, context_for({"roi": "old"}, {"roi": "old"})
    )
    assert recorder.calls == {"20161202": 1}, (
        "an unchanged per-date plan reuses the committed source"
    )

    recorder.calls.clear()
    result = preparation_mod.prepare_s1_inputs(
        stack, context_for({"roi": "extended"}, {"roi": "old"})
    )
    assert recorder.calls == {"20161201": 1, "20161202": 1}, (
        "the same date must be re-materialized for the extended ROI"
    )
    assert result.requested == ("20161201", "20161202"), (
        "plan changes cannot shrink the durable target"
    )
    assert dict(result.sources) == {target: (path,) for target, path in sources.items()}


def test_extend_roi_same_date_replans_through_the_public_prepare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real ``extend(roi=...)`` re-materializes the already-committed date."""
    from faninsar.data.query import BoundingBox
    from faninsar.missions.s1.s1search import S1Search
    from faninsar.processing.geometry import DEM
    from faninsar.stack.search import MissionPreparedInputs
    from tests.missions.s1.test_from_roi import ROI, _cdse_item

    calls: list[tuple[str, ...]] = []
    source = tmp_path / "20161207.SAFE"
    source.mkdir()

    def materialize(cls: Any, search: Any, **kwargs: Any) -> Any:
        calls.append(tuple(item.item_id for item in search.items))
        return SimpleNamespace(request={"paths": [source]})

    monkeypatch.setattr(S1Stack, "_materialize_from_search", classmethod(materialize))
    monkeypatch.setattr(s1_stack_mod, "_configure_s1_bindings", lambda *a, **k: None)
    monkeypatch.setattr(S1Stack, "_bind_precise_orbits", lambda self: None)
    monkeypatch.setattr(
        S1Stack,
        "_local_mission_inputs",
        lambda self: MissionPreparedInputs(),
    )
    monkeypatch.setattr(S1Stack, "_track_metadata", lambda self: None)

    stack = S1Stack.from_search(
        S1Search(
            items=(_cdse_item(),),
            catalog="sentinel1-cdse",
            spatial=ROI,
            query={},
        ),
        work_dir=tmp_path / "work",
        dem=DEM.from_constant(0),
        extra={"orbit_auto_fetch": False, "aux_cal_auto_fetch": False},
    )
    assert stack.prepare_data().complete
    assert len(calls) == 1

    stack.extend(roi=BoundingBox(100.2, 44.2, 100.6, 44.6, crs="EPSG:4326"))
    assert stack.prepare_data().complete
    assert len(calls) == 2, (
        "the same date must be materialized again after the additive ROI"
    )
    assert stack.catalog.dates == ("20161207",)
    assert stack.request["kind"] == "search"
    assert len(stack.request["search"]["items"]) == 1, (
        "the durable request must remain the full selected source, not the "
        "replanned subset"
    )


def test_pipeline_coreg_callback_runs_while_peer_download_is_blocked(
    tmp_path: Path,
) -> None:
    """The real completion loop publishes and runs science before peer release."""
    from affine import Affine

    from faninsar.processing.geometry import DEM, GridSpec
    from faninsar.stack.config import StackConfig
    from faninsar.stack.preparation_pipeline import run_preparation_pipeline
    from faninsar.stack.search import MissionPreparedInputs

    release_peer = threading.Event()
    peer_blocked = threading.Event()
    timeline: list[tuple[Any, ...]] = []
    sources = _source_tree(tmp_path, ("20161201", "20161202"))

    class StreamingMission(S1Stack):
        def _prepare_inputs(self, context: Any) -> PreparedInputs:
            requested = tuple(self.request["targets"])
            pending = {
                target: (Path(path),)
                for target, path in self.request["sources"].items()
                if self._inputs_state is None
                or target not in self.catalog.paths
                or not all(item.exists() for item in self.catalog.paths_for(target))
            }

            def acquire(target: str, items: tuple[Path, ...]) -> list[Path]:
                if target == "20161202":
                    peer_blocked.set()
                    assert release_peer.wait(5), "coreg callback never released peer"
                    timeline.append(("end", target))
                return list(items)

            def complete(target: str, result: Any) -> None:
                timeline.append(("acquired", target))
                assert context.on_ready is not None
                context.on_ready(
                    PreparedInputs(
                        sources={target: tuple(result)},
                        requested=requested,
                        mission=MissionPreparedInputs(),
                    )
                )

            from faninsar.missions.s1.preparation import _acquire_dates

            _acquire_dates(acquire, pending, context, on_complete=complete)
            return PreparedInputs(
                sources={
                    target: (Path(path),)
                    for target, path in self.request["sources"].items()
                },
                requested=requested,
            )

        def _track_metadata(self) -> Any:
            return None

    stack = StreamingMission.configure(
        config=StackConfig(
            work_dir=tmp_path / "work",
            grid=GridSpec(
                "EPSG:4326", Affine(0.001, 0, 10, 0, -0.001, 50), height=2, width=2
            ),
            dem=DEM.from_constant(0),
        ),
        request={
            "kind": "fixture",
            "targets": ("20161201", "20161202"),
            "sources": sources,
            "topology": {},
        },
    )

    def coreg(stack: Any, dates: tuple[str, ...]) -> None:
        timeline.append(("coreg", dates))
        release_peer.set()

    report = run_preparation_pipeline(stack, on_ready=coreg)

    assert peer_blocked.is_set()
    assert ("coreg", ("20161201",)) in timeline
    assert timeline.index(("coreg", ("20161201",))) < timeline.index(
        ("end", "20161202")
    ), "coreg must run while the peer acquisition is still blocked"
    assert report.complete
    assert stack.catalog.dates == ("20161201", "20161202")
    assert stack.reference == "20161201"
