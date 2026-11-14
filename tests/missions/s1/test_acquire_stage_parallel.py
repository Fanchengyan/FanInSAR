"""Bounded parallel acquire stage and the deferred rate-limit pass.

Contract (user-approved narrow amendment to PROPOSAL-0061/P0070 execution
semantics, 2026-11-12):

- Regular acquire stage: a budget-bounded number of measurement tasks
  (four by default) for CDSE burst requests whose every pending date carries
  exactly one searched item; every other request keeps the serial per-date
  loop.  The shared fetcher's own budgeted ceiling still governs HTTP
  concurrency; the library never claims a global measurement cap it does
  not enforce.
- Workers run under a copied context per submit, see the shared
  operation-scoped fetcher, and never mutate the shared stack.
- A CDSE 429 retry allowance exhausted by the shared fetcher (``reason``
  ``rate_limit_exhausted``) defers a target to a final pass that runs
  after every regular date has drained: the shared fetcher is first
  reconfigured to zero retries / one worker / reset counters through its
  own loop, then each deferred target re-runs once at one measurement at
  a time.  Any 429 in the deferred pass is a terminal
  :class:`PreparationFailure`; checksum and auth failures are never
  deferred.
- Main-thread cancellation stops new admission, cancels queued tasks and
  unsettled bridges, joins the workers, and only then closes the shared
  session — no minute-long retry backoff can block the unwind.
- A CDSE rediscovery that lacks an ASF-selected parent measurement fails
  only that date (``rediscovery_missing_parent``); its siblings still
  download and qualify, and ``on_error='raise'`` refuses the reduced set
  only after every date has drained.

All tests are offline: ``_materialize_from_search`` is a fake on the
stack class, so no network is touched.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

import faninsar.missions.s1.acquisition as acquisition_mod
import faninsar.missions.s1.preparation as preparation_mod
import faninsar.missions.s1.stack as s1_stack_mod
from faninsar.missions.s1.preparation import prepare_s1_inputs
from faninsar.remote.errors import RemoteAccessError

_SENTINEL = contextvars.ContextVar("acquire_stage_test_sentinel", default=None)


@dataclass
class _FakeRequestConfig:
    roi: object = None
    polarizations: tuple = ("VV",)
    coreg_mode: str = "golden"
    extra: dict = field(default_factory=dict)


@dataclass
class _FakeSearch:
    items: tuple
    catalog: str = "fake"
    spatial: object = None


def _catalog_item(day: int) -> SimpleNamespace:
    return SimpleNamespace(
        acquisition=SimpleNamespace(start=datetime(2016, 12, day, tzinfo=UTC))
    )


def _context(provider: str = "cdse", granularity: str = "burst") -> SimpleNamespace:
    return SimpleNamespace(
        budget=None,
        request={
            "kind": "search",
            "provider": provider,
            "granularity": granularity,
            "orbit_group": None,
            "orbit_paths": None,
            "aux_cal_paths": None,
            "search": {},
        },
        offline=False,
    )


class _Recorder:
    """Thread-safe event log shared by fake workers and assertions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[tuple[Any, ...]] = []
        self.calls: dict[str, int] = {}
        self.peak = 0
        self.active = 0

    def append(self, event: tuple[Any, ...]) -> None:
        with self._lock:
            self.events.append(event)

    def enter(self, target: str) -> None:
        with self._lock:
            self.calls[target] = self.calls.get(target, 0) + 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.events.append(("start", target))

    def exit(self, target: str) -> None:
        with self._lock:
            self.active -= 1
            self.events.append(("end", target))


def _make_stack(
    recorder: _Recorder,
    tmp_path: Any,
    behaviors: dict[str, Any],
    *,
    pace: Any = None,
    source_paths: dict[str, Any] | None = None,
):
    """Build a fake stack whose per-date materialization follows ``behaviors``.

    ``behaviors`` maps a date target to an exception instance or a list
    of per-attempt outcomes (``None`` means success); missing targets
    succeed.  ``pace`` runs inside each recorded attempt between
    the enter and exit events, so a barrier there proves real overlap.
    The fake records the shared operation identity and the context
    sentinel inside each worker.
    """

    class _FakeStack:
        _request_config = _FakeRequestConfig(
            extra={"polarizations": ("VV",), "orbit_auto_fetch": False}
        )
        _inputs_state = None
        config = SimpleNamespace(work_dir=tmp_path)
        catalog = SimpleNamespace(paths={}, paths_for=lambda target: [])

        def __init__(self, config: Any = None, catalog: Any = None) -> None:
            if config is not None:
                self._request_config = config
                self.catalog = catalog

        def _bind_precise_orbits(self) -> None:
            return None

        def _local_mission_inputs(self) -> SimpleNamespace:
            return SimpleNamespace(
                selected_footprints=(),
                orbit_paths_by_acquisition={},
                aux_cal_paths_by_acquisition={},
                provenance={},
            )

        @classmethod
        def _materialize_from_search(cls, search: Any, **kwargs: Any) -> Any:
            del kwargs
            target = search.items[0].acquisition.start.strftime("%Y%m%d")
            operation = acquisition_mod._burst_transfer.get()
            recorder.enter(target)
            recorder.append(("context", target, _SENTINEL.get(), id(operation)))
            try:
                if pace is not None:
                    pace()
                behavior = behaviors.get(target)
                if isinstance(behavior, list):
                    behavior = behavior.pop(0)
                if isinstance(behavior, BaseException):
                    raise behavior
                if callable(behavior):
                    behavior()
            finally:
                recorder.exit(target)
            paths = [] if source_paths is None else [source_paths[target]]
            return SimpleNamespace(request={"paths": paths})

    return _FakeStack()


def _dates(count: int) -> tuple:
    return tuple(_catalog_item(day) for day in range(1, count + 1))


def _search(items: tuple) -> _FakeSearch:
    return _FakeSearch(items=items)


def _rate_limit_error() -> RemoteAccessError:
    return RemoteAccessError(
        "rate_limit_exhausted", "range object exhausted its 429 retry allowance"
    )


_MISSING_PARENT = (
    "CDSE rediscovery is missing ASF-selected measurements "
    "[('S1A_IW_SLC__1SDV_20150428T105157_20150428T105224_045935_055FA1_87A1',"
    "'IW1', 'VV')]; refusing to apply another swath or date's burst hints"
)


class TestRegularParallelStage:
    """At most four concurrent measurement tasks on eligible requests."""

    def test_four_measurement_tasks_run_concurrently_not_serial(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Eight single-measurement dates overlap in waves of exactly four."""
        recorder = _Recorder()
        barrier = threading.Barrier(4, timeout=1)
        stack = _make_stack(recorder, tmp_path, {}, pace=barrier.wait)
        items = _dates(8)
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(items)
        )
        result = prepare_s1_inputs(stack, _context())
        assert result.failures == ()
        assert recorder.calls == {
            item.acquisition.start.strftime("%Y%m%d"): 1 for item in items
        }, "every date acquired exactly once"
        assert recorder.peak == 4, (
            f"four concurrent measurement tasks expected, peak was {recorder.peak}"
        )

    def test_workers_inherit_prepare_context_and_shared_operation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Each submit copies the context; workers see one shared operation."""
        recorder = _Recorder()
        stack = _make_stack(recorder, tmp_path, {})
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(_dates(4))
        )
        _SENTINEL.set("prepare-context")
        prepare_s1_inputs(stack, _context())
        contexts = [event for event in recorder.events if event[0] == "context"]
        assert len(contexts) == 4
        assert {event[2] for event in contexts} == {"prepare-context"}, (
            "workers must run under a copied prepare context"
        )
        operation_ids = {event[3] for event in contexts}
        assert len(operation_ids) == 1 and None not in operation_ids, (
            "every worker must observe the same shared CDSE burst operation"
        )

    def test_worker_limit_uses_operation_budget(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A caller can lower active date tasks without changing range limits."""
        recorder = _Recorder()
        barrier = threading.Barrier(2, timeout=1)
        stack = _make_stack(recorder, tmp_path, {}, pace=barrier.wait)
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(_dates(6))
        )
        context = _context()
        context.budget = SimpleNamespace(max_active_files=2)
        result = prepare_s1_inputs(stack, context)
        assert result.failures == ()
        assert len(recorder.calls) == 6
        assert recorder.peak == 2

    @pytest.mark.parametrize(
        ("provider", "granularity"), [("asf", "burst"), ("cdse", "scene")]
    )
    def test_other_requests_keep_serial_acquisition(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        provider: str,
        granularity: str,
    ) -> None:
        """Only eligible CDSE burst requests enter the date worker pool."""
        recorder = _Recorder()
        stack = _make_stack(recorder, tmp_path, {"20161202": _rate_limit_error()})
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(_dates(3))
        )
        result = prepare_s1_inputs(stack, _context(provider, granularity))
        assert recorder.peak == 1
        assert recorder.calls["20161202"] == 1
        assert [failure.reason for failure in result.failures] == ["rate_limit_exhausted"]

    def test_multi_item_dates_keep_serial_semantics_without_deferral(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A multi-measurement date disables the pool: serial, immediate fail.

        The generic library must not claim a global four-measurement cap
        it does not enforce: dates carrying several searched items stay on
        today's serial loop, and a rate-limit exhaustion there is an
        immediate failure, not a deferral.
        """
        recorder = _Recorder()
        stack = _make_stack(recorder, tmp_path, {"20161202": _rate_limit_error()})
        items = (*_dates(1), _catalog_item(2), _catalog_item(2))
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(items)
        )
        result = prepare_s1_inputs(stack, _context())
        assert recorder.peak == 1, "multi-item dates must stay serial"
        assert recorder.calls["20161202"] == 1, "no deferred re-run off the gate"
        reasons = {failure.target: failure.reason for failure in result.failures}
        assert reasons == {"20161202": "rate_limit_exhausted"}


class TestDeferredFinalPass:
    """Rate-limit exhaustion defers; the final pass runs after full drain."""

    def test_deferred_target_reruns_once_after_every_regular_date_drained(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """One deferred date re-runs serially behind a policy reconfigure."""
        recorder = _Recorder()
        stack = _make_stack(recorder, tmp_path, {"20161202": [_rate_limit_error(), None]})
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(_dates(3))
        )
        original_configure = preparation_mod._reconfigure_deferred_pass_policy

        def record_configure(operation: Any) -> None:
            recorder.append(("configure",))
            original_configure(operation)

        monkeypatch.setattr(
            preparation_mod, "_reconfigure_deferred_pass_policy", record_configure
        )
        result = prepare_s1_inputs(stack, _context())
        assert recorder.calls["20161202"] == 2, "exactly one deferred re-run"
        events = recorder.events
        configure_at = events.index(("configure",))
        retry_at = events.index(("start", "20161202"), configure_at + 1)
        for target in ("20161201", "20161203"):
            end_at = events.index(("end", target))
            assert end_at < configure_at, (
                f"{target} must drain before the deferred pass reconfigures"
            )
        assert configure_at < retry_at, "reconfigure precedes the first re-run"
        assert result.failures == ()
        assert result.requested == ("20161201", "20161202", "20161203")

    def test_deferred_success_is_qualified_without_dropping_failed_requests(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Drain order cannot remove successful or failed dates from the report."""
        import faninsar.missions.s1.stack as stack_mod

        recorder = _Recorder()
        source_paths = {
            target: tmp_path / f"{target}.SAFE"
            for target in ("20161201", "20161202", "20161203")
        }
        for path in source_paths.values():
            path.mkdir()
        stack = _make_stack(
            recorder,
            tmp_path,
            {
                "20161202": [_rate_limit_error(), None],
                "20161203": RemoteAccessError("auth_expired", "expired token"),
            },
            source_paths=source_paths,
        )
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(_dates(3))
        )
        monkeypatch.setattr(stack_mod, "_configure_s1_bindings", lambda *a, **k: None)
        result = prepare_s1_inputs(stack, _context())
        assert result.requested == ("20161201", "20161202", "20161203")
        assert dict(result.sources) == {
            target: (source_paths[target],) for target in ("20161201", "20161202")
        }
        assert [(failure.target, failure.reason) for failure in result.failures] == [
            ("20161203", "auth_expired")
        ]
        assert recorder.calls == {"20161201": 1, "20161202": 2, "20161203": 1}

    def test_final_pass_rate_limit_is_terminal_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A 429 in the deferred pass fails the target for the report."""
        recorder = _Recorder()
        stack = _make_stack(recorder, tmp_path, {"20161202": _rate_limit_error()})
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(_dates(2))
        )
        result = prepare_s1_inputs(stack, _context())
        assert recorder.calls["20161202"] == 2
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.target == "20161202"
        assert failure.phase == "acquisition"
        assert failure.reason == "rate_limit_exhausted"
        assert result.requested == ("20161201", "20161202"), (
            "a failed deferred target stays in the requested contract"
        )

    @pytest.mark.parametrize(
        "error",
        [
            RemoteAccessError("auth_expired", "token refresh failed twice"),
            ValueError("every Sentinel-1 input failed download verification"),
        ],
    )
    def test_non_rate_limit_failures_are_never_deferred(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, error: Exception
    ) -> None:
        """Auth and checksum errors keep the immediate single-attempt failure."""
        recorder = _Recorder()
        stack = _make_stack(recorder, tmp_path, {"20161202": error})
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(_dates(2))
        )
        configure_calls: list[Any] = []
        monkeypatch.setattr(
            preparation_mod,
            "_reconfigure_deferred_pass_policy",
            lambda operation: configure_calls.append(operation),
        )
        result = prepare_s1_inputs(stack, _context())
        assert recorder.calls["20161202"] == 1, "non-rate-limit failure is not retried"
        assert configure_calls == [], "no deferred pass without a 429 exhaustion"
        expected = getattr(error, "reason", "checksum_mismatch")
        assert [failure.reason for failure in result.failures] == [expected]


class TestRediscoveryParentBoundary:
    """A missing CDSE parent is one date's failure, never a global abort."""

    def test_missing_parent_fails_only_its_date_and_peers_still_qualify(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Independent dates download while the unpairable date is reported."""
        recorder = _Recorder()
        targets = ("20161201", "20161202", "20161203")
        sources = {}
        for target in targets:
            path = tmp_path / f"{target}.SAFE"
            path.mkdir()
            sources[target] = path
        error = ValueError(_MISSING_PARENT)
        stack = _make_stack(
            recorder,
            tmp_path,
            {"20161201": error},
            source_paths=sources,
        )
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda *_: _search(_dates(3))
        )
        monkeypatch.setattr(
            s1_stack_mod, "_configure_s1_bindings", lambda *_a, **_k: None
        )
        context = _context()
        # One date task at a time proves the peer dates are admitted only
        # because the missing parent settles as a failure instead of raising.
        context.budget = SimpleNamespace(max_active_files=1)

        result = prepare_s1_inputs(stack, context)

        assert recorder.calls == dict.fromkeys(targets, 1), (
            "every selected date must still be attempted"
        )
        assert dict(result.sources) == {
            target: (sources[target],) for target in ("20161202", "20161203")
        }, "the downloaded siblings are not discarded with the failed date"
        assert result.requested == targets, "the request contract cannot shrink"
        assert [
            (failure.target, failure.phase, failure.reason)
            for failure in result.failures
        ] == [("20161201", "acquisition", "rediscovery_missing_parent")]
        assert result.failures[0].cause is error

    def test_unrelated_value_error_still_aborts_preparation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Only the known rediscovery seam is converted to a per-date failure."""
        recorder = _Recorder()
        stack = _make_stack(
            recorder, tmp_path, {"20161201": ValueError("unexpected defect")}
        )
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda *_: _search(_dates(3))
        )
        context = _context()
        context.budget = SimpleNamespace(max_active_files=1)

        with pytest.raises(ValueError, match="unexpected defect"):
            prepare_s1_inputs(stack, context)
        assert recorder.calls == {"20161201": 1}, "the defect must not be swallowed"

    def test_strict_preparation_raises_partial_report_after_every_date_drained(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """``on_error='raise'`` refuses the reduced set after the drain."""
        from faninsar.missions.s1.s1search import S1Search
        from faninsar.missions.s1.stack import S1Stack
        from faninsar.processing.geometry import DEM
        from faninsar.remote.records import AcquisitionMetadata, CatalogItem
        from faninsar.stack.preparation import DataPreparationError
        from faninsar.stack.search import MissionPreparedInputs
        from tests.missions.s1.test_from_roi import ROI

        targets = ("20161201", "20161202", "20161203")
        sources = {}
        for target in targets:
            path = tmp_path / f"{target}.SAFE"
            path.mkdir()
            sources[target] = path
        attempts: list[str] = []

        def item(target: str) -> Any:
            start = datetime.strptime(target, "%Y%m%d").replace(tzinfo=UTC)
            return CatalogItem(
                provider="CDSE",
                catalog="sentinel1-cdse",
                collection="SENTINEL-1",
                item_id=target,
                acquisition=AcquisitionMetadata(
                    acquisition_id=target, start=start, end=start
                ),
                assets={},
                raw_metadata={"polarization": "VV"},
            )

        def materialize(_cls: Any, search: Any, **_kwargs: Any) -> Any:
            target = search.items[0].acquisition.start.strftime("%Y%m%d")
            attempts.append(target)
            if target == "20161201":
                raise ValueError(_MISSING_PARENT)
            return SimpleNamespace(request={"paths": [sources[target]]})

        def noop(*_args: Any) -> None:
            return None

        def mission_inputs(_self: Any) -> Any:
            return MissionPreparedInputs()

        monkeypatch.setattr(
            S1Stack, "_materialize_from_search", classmethod(materialize)
        )
        monkeypatch.setattr(
            s1_stack_mod, "_configure_s1_bindings", lambda *_a, **_k: None
        )
        monkeypatch.setattr(S1Stack, "_bind_precise_orbits", noop)
        monkeypatch.setattr(S1Stack, "_local_mission_inputs", mission_inputs)
        monkeypatch.setattr(S1Stack, "_track_metadata", noop)
        stack = S1Stack.from_search(
            S1Search(
                items=tuple(item(target) for target in targets),
                catalog="sentinel1-cdse",
                spatial=ROI,
                query={},
            ),
            work_dir=tmp_path / "work",
            dem=DEM.from_constant(0),
            extra={"orbit_auto_fetch": False, "aux_cal_auto_fetch": False},
        )

        with pytest.raises(DataPreparationError) as raised:
            stack.prepare_data(on_error="raise")

        report = raised.value.report
        assert sorted(attempts) == sorted(targets), (
            "strict failure may only surface after every date settled"
        )
        assert report.outcome == "partial"
        assert report.requested == targets
        assert report.ready == ("20161202", "20161203")
        assert [(failure.target, failure.reason) for failure in report.failed] == [
            ("20161201", "rediscovery_missing_parent")
        ]
        assert not stack.data_ready, "a strict failure cannot commit a reduced set"


class TestCancellationSafety:
    """Cancellation joins workers before the shared session closes."""

    def test_main_thread_sigint_cancels_bridges_before_join(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A real SIGINT unblocks the main thread while bridge callers park."""
        recorder = _Recorder()
        parked = threading.Event()
        cancelled = threading.Event()

        def park_on_bridge() -> None:
            operation = acquisition_mod._burst_transfer.get()

            async def park() -> None:
                parked.set()
                try:
                    await asyncio.sleep(3600)
                finally:
                    cancelled.set()

            operation._bridge(park())

        stack = _make_stack(recorder, tmp_path, {}, pace=park_on_bridge)
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(_dates(8))
        )

        def interrupt_main() -> None:
            if parked.wait(timeout=1):
                os.kill(os.getpid(), signal.SIGINT)

        interrupter = threading.Thread(target=interrupt_main, name="sigint-probe")
        interrupter.start()
        started = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt):
                prepare_s1_inputs(stack, _context())
        finally:
            interrupter.join(timeout=1)
        assert time.monotonic() - started < 2
        assert cancelled.is_set()
        assert recorder.active == 0, "all workers must join before return"
        assert sum(recorder.calls.values()) <= 4, "cancellation stops date admission"
        assert not any(
            "faninsar-s1" in thread.name for thread in threading.enumerate()
        )

    def test_interrupt_cancels_queued_tasks_and_parked_bridges_promptly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """SIGINT-class unwind must not wait out a parked retry backoff.

        One worker raises KeyboardInterrupt; another parks on an
        operation-loop bridge the way a 429 backoff does.  The unwind
        cancels queued futures and the parked bridge, joins the workers,
        and the wrapper close then stops the loop — all promptly.
        """
        recorder = _Recorder()
        parked = threading.Event()
        cancelled = threading.Event()

        def interrupt() -> None:
            assert parked.wait(timeout=1), "second worker never reached the bridge"
            raise KeyboardInterrupt("simulated SIGINT")

        def park_on_bridge() -> None:
            operation = acquisition_mod._burst_transfer.get()
            assert operation is not None

            async def park() -> None:
                parked.set()
                try:
                    await asyncio.sleep(3600)
                finally:
                    cancelled.set()

            operation._bridge(park())

        behaviors = {
            "20161201": interrupt,
            "20161202": park_on_bridge,
        }
        stack = _make_stack(recorder, tmp_path, behaviors)
        monkeypatch.setattr(
            preparation_mod, "restore_search", lambda record, spatial: _search(_dates(3))
        )
        started = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            prepare_s1_inputs(stack, _context())
        elapsed = time.monotonic() - started
        assert elapsed < 2, f"unwind blocked {elapsed:.1f}s behind parked bridge work"
        assert cancelled.is_set(), "parked bridge was never cancelled"
        alive = [
            thread.name
            for thread in threading.enumerate()
            if "faninsar-s1" in thread.name
        ]
        assert alive == [], f"operation threads left running: {alive}"


class TestOperationPolicySeams:
    """The shared fetcher policy changes bridge through the operation loop."""

    def test_operation_configures_regular_retry_allowance_after_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lazy open applies the ten-retry regular-pass allowance."""
        import faninsar.remote.providers.cdse as cdse_mod

        configs: list[dict[str, Any]] = []

        class _FakeFetcher:
            def configure_rate_limit_policy(self, **kwargs: Any) -> None:
                configs.append(kwargs)

        class _FakeCM:
            async def __aenter__(self) -> _FakeFetcher:
                return _FakeFetcher()

            async def __aexit__(self, *args: Any) -> None:
                return None

        monkeypatch.setattr(cdse_mod, "cdse_range_fetcher", lambda *a, **k: _FakeCM())
        operation = acquisition_mod._CdseBurstTransfer()
        operation._bridge(operation._validate_and_open((), operation._policy))
        operation.close()
        assert configs == [{"max_retries_per_object": 10}]

    def test_deferred_pass_policy_bridges_configure_zero_one_reset(self) -> None:
        """The final-pass policy is zero retries, one worker, counters reset."""
        calls: list[dict[str, Any]] = []

        class _SpyFetcher:
            def configure_rate_limit_policy(self, **kwargs: Any) -> None:
                calls.append(kwargs)

        class _StubOperation:
            def __init__(self, fetcher: Any) -> None:
                self._fetcher = fetcher

            def _bridge(self, coro: Any) -> Any:
                return asyncio.run(coro)

        acquisition_mod._reconfigure_deferred_pass_policy(_StubOperation(_SpyFetcher()))
        assert calls == [
            {
                "max_retries_per_object": 0,
                "max_workers": 1,
                "reset_counters": True,
            }
        ]

    def test_deferred_pass_policy_skips_fetcher_without_seam(self) -> None:
        """Test doubles without the configuration seam stay untouched."""

        class _StubOperation:
            _fetcher = object()

            def _bridge(self, coro: Any) -> Any:
                raise AssertionError("must not bridge without a policy seam")

        acquisition_mod._reconfigure_deferred_pass_policy(_StubOperation())

    def test_cancel_pending_stops_admission_and_cancels_inflight_bridge(
        self,
    ) -> None:
        """cancel_pending keeps the loop alive for workers but stops spans."""
        operation = acquisition_mod._CdseBurstTransfer()
        settled = threading.Event()

        async def park() -> None:
            settled.set()
            await asyncio.sleep(3600)

        def call_bridge() -> None:
            try:
                operation._bridge(park())
            except BaseException as error:
                from concurrent.futures import CancelledError

                assert isinstance(error, CancelledError)

        bridge = threading.Thread(target=call_bridge, name="park-probe")
        bridge.start()
        deadline = time.monotonic() + 10
        while not settled.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert settled.is_set(), "parked coroutine never started"
        operation.cancel_pending()
        assert operation._stop.is_set(), "new span admission must stop"
        bridge.join(timeout=1)
        assert not bridge.is_alive(), "bridge caller not released by cancel"
        operation.close()
        assert not any(
            "faninsar-s1" in thread.name for thread in threading.enumerate()
        ), "operation loop thread left running"
