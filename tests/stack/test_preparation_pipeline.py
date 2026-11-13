"""Incremental preparation publication and bounded overlap (P0088).

The mission double emits one qualified date at a time through the real
``PreparationContext`` observer.  The shared preparation pass must install an
incremental durable snapshot and invoke the caller's ``on_ready`` callback on
the controlling thread before the mission finishes its later dates, while
keeping the original full request and deterministic reference intact.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from affine import Affine

from faninsar.missions import base as mission_registry
from faninsar.processing.geometry import DEM, GridSpec
from faninsar.stack.config import StackConfig
from faninsar.stack.preparation import (
    DataPreparationError,
    PreparationFailure,
    PreparedInputs,
    reentrant_writer_guard,
    writer_guard,
)
from faninsar.stack.preparation_pipeline import run_preparation_pipeline
from faninsar.stack.search import MissionPreparedInputs
from faninsar.stack.session import Stack

STATE: dict[str, Any] = {}


class PipelineMission(Stack):
    """Emit one date at a time through the shared preparation observer."""

    def _prepare_inputs(self, context):
        state = STATE
        requested = tuple(self.request["targets"])
        order = tuple(state.get("order") or requested)
        sources = {}
        failures = []
        for target in order:
            blocker = state.get("block", {}).get(target)
            if blocker is not None:
                state["events"].append(("block", target))
                assert blocker.wait(5), f"{target} never unblocked"
            path = Path(self.request["sources"][target])
            ready = (
                self._inputs_state is not None
                and target in self.catalog.paths
                and all(item.exists() for item in self.catalog.paths_for(target))
            )
            if not ready:
                state["acquisitions"].append(target)
            if not path.exists():
                failures.append(
                    PreparationFailure(
                        target, "source", "missing", FileNotFoundError(target)
                    )
                )
                state["events"].append(("failed", target))
                continue
            sources[target] = (path,)
            partial = PreparedInputs(
                sources={target: (path,)},
                requested=requested,
                mission=MissionPreparedInputs(),
            )
            if context.on_ready is not None:
                context.on_ready(partial)
            state["events"].append(("qualified", target))
        return PreparedInputs(
            sources=sources,
            requested=requested,
            failures=tuple(failures),
        )


@pytest.fixture(autouse=True)
def pipeline_mission(monkeypatch: pytest.MonkeyPatch) -> type[PipelineMission]:
    """Register the mission for local reopen and reset its shared state."""
    monkeypatch.setitem(
        mission_registry._MISSION_STACKS,
        "PipelineMission",
        (__name__, "_pipeline_mission"),
    )
    monkeypatch.setattr(
        __import__(__name__, fromlist=["x"]),
        "_pipeline_mission",
        PipelineMission,
        raising=False,
    )
    STATE.clear()
    STATE.update(
        {
            "events": [],
            "acquisitions": [],
            "block": {},
            "order": None,
        }
    )
    return PipelineMission


def _grid() -> GridSpec:
    return GridSpec("EPSG:4326", Affine(0.001, 0, 10, 0, -0.001, 50), height=2, width=2)


def _stack(
    tmp_path: Path,
    targets: tuple[str, ...],
    *,
    present: tuple[str, ...] | None = None,
    topology: dict[str, Any] | None = None,
) -> tuple[Stack, dict[str, Path]]:
    present = targets if present is None else present
    sources = {target: tmp_path / f"source-{target}" for target in targets}
    for target in present:
        sources[target].write_bytes(b"qualified")
    stack = PipelineMission.configure(
        config=StackConfig(
            work_dir=tmp_path / "work", grid=_grid(), dem=DEM.from_constant(0)
        ),
        request={
            "kind": "fixture",
            "targets": targets,
            "sources": sources,
            "topology": topology or {},
        },
    )
    return stack, sources


def _ready_callback(work: Path):
    def callback(stack: Stack, dates: tuple[str, ...]) -> None:
        STATE["events"].append(
            (
                "ready",
                dates,
                tuple(stack.catalog.dates),
                stack.reference,
                stack.grid,
            )
        )
        for date in dates:
            (work / f"science-{date}.bin").write_bytes(b"local coreg product")

    return callback


def test_first_date_publishes_before_the_mission_finishes(
    tmp_path: Path, pipeline_mission: type[PipelineMission]
) -> None:
    """The callback sees a cumulative snapshot, not the final all-dates set."""
    stack, sources = _stack(tmp_path, ("20200101", "20200113"))
    report = run_preparation_pipeline(
        stack,
        on_ready=_ready_callback(stack.config.work_dir),
    )

    events = STATE["events"]
    ready = [event for event in events if event[0] == "ready"]
    assert report.complete
    assert ready[0][1] == ("20200101",)
    assert ready[0][2] == ("20200101",), "first publish must be incremental"
    assert ready[0][3] == "20200101"
    assert ready[0][4] == _grid(), "fixed grid is shared with the callback"
    assert ready[1][1] == ("20200113",)
    assert ready[1][2] == ("20200101", "20200113")
    assert ready[1][3] == "20200101"
    assert stack.catalog.dates == ("20200101", "20200113")
    assert dict(stack.catalog.paths) == {
        target: (path,) for target, path in sources.items()
    }
    assert events.index(
        ("ready", ("20200101",), ("20200101",), "20200101", _grid())
    ) < (events.index(("qualified", "20200101")))
    assert [event for event in events if event[0] == "qualified"] == [
        ("qualified", "20200101"),
        ("qualified", "20200113"),
    ]


def test_reference_is_frozen_from_the_full_request_not_completion_order(
    tmp_path: Path, pipeline_mission: type[PipelineMission]
) -> None:
    """A later secondary arriving first cannot become the reference."""
    STATE["order"] = ("20200113", "20200101")
    stack, _ = _stack(tmp_path, ("20200101", "20200113"))
    report = run_preparation_pipeline(
        stack,
        on_ready=_ready_callback(stack.config.work_dir),
    )

    events = STATE["events"]
    ready = [event for event in events if event[0] == "ready"]
    assert report.complete
    assert len(ready) == 1, "secondaries wait until the reference is qualified"
    assert ready[0][1] == ("20200101", "20200113")
    assert ready[0][2] == ("20200101", "20200113")
    assert stack.reference == "20200101"
    assert ready[0][3] == "20200101"


def test_missing_explicit_reference_blocks_science_without_fallback(
    tmp_path: Path, pipeline_mission: type[PipelineMission]
) -> None:
    """An unavailable explicit reference never becomes an available date."""
    stack, _ = _stack(
        tmp_path,
        ("20200101", "20200113"),
        topology={"reference": "20200125"},
    )
    events: list[Any] = []
    report = run_preparation_pipeline(
        stack,
        on_ready=_ready_callback(stack.config.work_dir),
    )

    events = STATE["events"]
    assert not [event for event in events if event[0] == "ready"]
    assert stack.reference == "20200125"
    assert report.ready == ("20200101", "20200113")
    assert any("explicit reference" in item for item in report.prerequisites)


def test_late_failure_preserves_earlier_published_product_and_intent(
    tmp_path: Path, pipeline_mission: type[PipelineMission]
) -> None:
    """An unavailable later date cannot roll back an earlier commit."""
    stack, sources = _stack(
        tmp_path,
        ("20200101", "20200113"),
        present=("20200101",),
    )
    work = stack.config.work_dir
    report = run_preparation_pipeline(
        stack,
        on_error="warn",
        on_ready=_ready_callback(work),
    )

    assert report.outcome == "partial"
    assert (work / "science-20200101.bin").read_bytes() == b"local coreg product"
    assert stack.catalog.dates == ("20200101",)
    assert stack.reference == "20200101"
    # The original full target stays durable; only completion is partial.
    assert stack.request["targets"] == ("20200101", "20200113")
    legacy = json.loads((work / "inputs" / "request.json").read_text())
    assert legacy["source"]["targets"] == {
        "faninsar:tuple": list(stack.request["targets"])
    }
    identity = json.loads((work / "stack.json").read_text())
    assert identity["request"]["targets"] == list(stack.request["targets"])


def test_raise_policy_keeps_the_incremental_commit(
    tmp_path: Path, pipeline_mission: type[PipelineMission]
) -> None:
    """A strict failure still preserves the already published date."""
    stack, _ = _stack(
        tmp_path,
        ("20200101", "20200113"),
        present=("20200101",),
    )
    work = stack.config.work_dir
    with pytest.raises(DataPreparationError):
        run_preparation_pipeline(
            stack,
            on_error="raise",
            on_ready=_ready_callback(work),
        )

    assert (work / "science-20200101.bin").is_file()
    assert stack.catalog.dates == ("20200101",)
    assert stack.reference == "20200101"


def test_reopen_reuses_qualified_source_without_reacquisition(
    tmp_path: Path, pipeline_mission: type[PipelineMission]
) -> None:
    """A restart publishes committed dates locally and acquires only the rest."""
    stack, sources = _stack(
        tmp_path,
        ("20200101", "20200113"),
        present=("20200101",),
    )
    work = stack.config.work_dir
    run_preparation_pipeline(
        stack,
        on_ready=_ready_callback(work),
    )
    sources["20200113"].write_bytes(b"recovered")

    opened = Stack.open(work)
    STATE["acquisitions"] = []
    STATE["events"] = []
    report = run_preparation_pipeline(
        opened,
        on_ready=_ready_callback(work),
    )

    assert report.complete
    assert STATE["acquisitions"] == ["20200113"], (
        "the committed date must not be downloaded again"
    )
    ready = [event for event in STATE["events"] if event[0] == "ready"]
    assert [event[1] for event in ready] == [("20200101",), ("20200113",)]
    assert opened.catalog.dates == ("20200101", "20200113")
    assert opened.reference == "20200101"
    assert (work / "science-20200101.bin").is_file()


def test_shared_dem_failure_blocks_callback_until_recovery(
    tmp_path: Path,
    pipeline_mission: type[PipelineMission],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed shared DEM never publishes a premature ready callback."""
    stack, _ = _stack(tmp_path, ("20200101", "20200113"))
    original = Stack._ensure_raster_dem
    calls = {"count": 0}

    def flaky(self: Stack) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectionError("shared DEM fetch failed")
        return original(self)

    monkeypatch.setattr(Stack, "_ensure_raster_dem", flaky)
    report = run_preparation_pipeline(
        stack,
        on_ready=_ready_callback(stack.config.work_dir),
    )

    events = STATE["events"]
    ready = [event for event in events if event[0] == "ready"]
    assert calls["count"] >= 2, "the second date must retry shared inputs"
    assert len(ready) == 1, "the failed first attempt must not call on_ready"
    assert ready[0][1] == ("20200101", "20200113")
    assert ready[0][2] == ("20200101", "20200113")
    assert events.index(("qualified", "20200101")) < events.index(ready[0])
    assert report.complete
    assert stack.data_ready
    assert (stack.config.work_dir / "science-20200101.bin").is_file()
    assert (stack.config.work_dir / "science-20200113.bin").is_file()


def test_reference_uses_full_acquisition_id_order_not_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unparseable start still pins the reference from the full request."""
    from faninsar.stack.preparation import _resolve_networks

    monkeypatch.setattr(
        "faninsar.stack.session._pairs_from_factory",
        lambda dates, **kwargs: tuple(dates),
    )
    stack = SimpleNamespace(
        request={"kind": "search", "topology": {}},
        _inputs_state=None,
    )
    sources = {"S1A_B": (tmp_path / "b",), "S1A_A": (tmp_path / "a",)}
    _, _, reference, _ = _resolve_networks(stack, sources, requested=("S1A_A", "S1A_B"))
    assert reference == "S1A_A", "completion order must not choose the reference"

    non_search = SimpleNamespace(
        request={"kind": "rslc", "topology": {}},
        _inputs_state=None,
    )
    path_sources = {
        "20240113": (tmp_path / "later.h5",),
        "20240101": (tmp_path / "earlier.h5",),
    }
    _, _, reference, _ = _resolve_networks(
        non_search, path_sources, requested=("/source/later.h5",)
    )
    assert reference == "20240101", "path membership stays a derived catalog view"


def test_reentrant_writer_guard_is_same_thread_only(tmp_path: Path) -> None:
    """Nested preparation re-enters on one thread; peers and processes still block."""
    root = tmp_path / "work"
    failures: list[BaseException] = []

    def peer() -> None:
        try:
            with reentrant_writer_guard(root):
                pass
        except BaseException as error:  # noqa: BLE001 - asserted below
            failures.append(error)

    with reentrant_writer_guard(root):
        with reentrant_writer_guard(root):
            pass
        thread = threading.Thread(target=peer)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert failures and "another preparation writer" in str(failures[0])
        with pytest.raises(RuntimeError, match="another preparation writer"):
            with writer_guard(root):
                pass
    with reentrant_writer_guard(root):
        pass
