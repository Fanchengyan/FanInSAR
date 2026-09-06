"""Tests for finite resource admission and process-tree telemetry."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from faninsar.processing.runtime.resources import (
    ProcessTreeAdmission,
    ProcessTreeMemoryWatchdog,
    ProcessTreeSampler,
    ResourceAdmissionError,
    ResourceAdmissionLedger,
    ResourceBudget,
    bootstrap_worker_runtime,
    estimate_formation_resources,
    estimate_spatial_irls_resources,
    reserve_estimate,
)

if TYPE_CHECKING:
    from pathlib import Path


def _budget() -> ResourceBudget:
    """Return a small deterministic test budget."""
    return ResourceBudget(
        max_files=4,
        max_chunks=8,
        max_encoded_bytes=100,
        max_decoded_bytes=200,
        max_temporary_bytes=100,
        max_workers=2,
        max_processes=3,
        disk_reserve_bytes=1,
        max_rss_bytes=4 * 1024 * 1024 * 1024,
        max_device_bytes=256,
    )


def test_resource_ledger_reserves_and_releases_atomically(tmp_path: Path) -> None:
    """Reservations account decoded bytes before a second request."""
    ledger = ResourceAdmissionLedger(_budget(), tmp_path)
    with ledger.reserve(decoded_bytes=150, temporary_bytes=10, files=1) as reservation:
        assert ledger.usage.decoded_bytes == 150
        with pytest.raises(ResourceAdmissionError):
            ledger.reserve(decoded_bytes=51)
        reservation.release()
        assert ledger.usage == ledger.usage.__class__()


def test_resource_ledger_rejects_negative_and_disk_overcommit(tmp_path: Path) -> None:
    """Invalid requests fail before the ledger counters change."""
    ledger = ResourceAdmissionLedger(_budget(), tmp_path)
    with pytest.raises(ResourceAdmissionError):
        ledger.reserve(decoded_bytes=-1)
    with pytest.raises(ResourceAdmissionError):
        ledger.reserve(decoded_bytes=201)
    assert ledger.usage.decoded_bytes == 0


def test_builtin_formation_estimate_is_reserved_before_work(tmp_path: Path) -> None:
    """Built-in formation accounting is checked through the shared ledger."""
    estimate = estimate_formation_resources(
        shape=(8, 8),
        multilook=(2, 2),
        coherence_window=(3, 3),
        phase_filter=None,
    )
    budget = ResourceBudget(
        max_files=8,
        max_chunks=8,
        max_encoded_bytes=estimate.usage.encoded_bytes,
        max_decoded_bytes=estimate.usage.decoded_bytes,
        max_temporary_bytes=estimate.usage.temporary_bytes,
        max_workers=1,
        max_processes=1,
        disk_reserve_bytes=1,
        max_rss_bytes=4 * 1024 * 1024 * 1024,
        max_device_bytes=estimate.usage.device_bytes + 1,
    )
    ledger = ResourceAdmissionLedger(budget, tmp_path)
    with reserve_estimate(ledger, estimate):
        assert ledger.usage.decoded_bytes == estimate.usage.decoded_bytes
    assert ledger.usage.decoded_bytes == 0


def test_spatial_irls_estimate_charges_component_dct_workspace() -> None:
    """The second-stage solver estimate includes the documented DCT buffers."""
    estimate = estimate_spatial_irls_resources(
        shape=(5, 7),
        active_edges=20,
        component_bbox_areas=(6, 12),
        max_iter=2,
        cg_max_iter=3,
        phase_itemsize=4,
    )
    assert estimate.usage.device_bytes == 4 * (6 * 18 + 4 * 12)
    assert estimate.work > 0


def test_process_tree_sampler_and_watchdog_record_complete_sample() -> None:
    """The sampler includes the current process and watchdog accepts its baseline."""
    snapshot = ProcessTreeSampler().sample()
    assert snapshot.root_pid > 0
    assert snapshot.process_ids
    watchdog = ProcessTreeMemoryWatchdog(
        max_rss_bytes=snapshot.rss_bytes + 64 * 1024 * 1024,
        interval_seconds=0.1,
    )
    current = watchdog.sample()
    assert current.root_pid == snapshot.root_pid
    assert current.process_ids
    assert current.rss_bytes <= watchdog.max_rss_bytes


def test_process_tree_watchdog_fails_closed_on_rss_limit() -> None:
    """An over-limit sample prevents publication rather than merely logging."""
    watchdog = ProcessTreeMemoryWatchdog(max_rss_bytes=1, interval_seconds=0.1)
    with pytest.raises(ResourceAdmissionError):
        watchdog.sample()


def test_process_tree_admission_reserves_workers_and_stops_watchdog(
    tmp_path: Path,
) -> None:
    """A worker admission reserves process slots before execution."""
    baseline = ProcessTreeSampler().sample().rss_bytes
    budget = replace(_budget(), max_rss_bytes=baseline + 64 * 1024 * 1024)
    with ProcessTreeAdmission(budget, tmp_path, workers=1, files=2) as admission:
        assert admission.ledger is not None
        assert admission.ledger.usage.workers == 1
        assert admission.ledger.usage.processes == 2
        assert admission.watchdog is not None
        assert admission.watchdog.samples
    assert admission.reservation is None
    assert admission.ledger is not None
    assert admission.ledger.usage.workers == 0


def test_bootstrap_worker_runtime_sets_caps_before_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker initializer sets numerical caps and its bootstrap marker."""
    fake_torch = SimpleNamespace(
        set_num_threads=lambda value: setattr(fake_torch, "threads", value),
        set_num_interop_threads=lambda value: setattr(
            fake_torch, "interop_threads", value
        ),
    )
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "FANINSAR_WORKER_BOOTSTRAPPED",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    bootstrap_worker_runtime(2)

    assert fake_torch.threads == 2
    assert fake_torch.interop_threads == 2
    assert sys.modules["torch"] is fake_torch
    assert os.environ["FANINSAR_WORKER_BOOTSTRAPPED"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "2"


def test_bootstrap_worker_runtime_rejects_invalid_cap() -> None:
    """A non-positive worker cap fails closed before numerical work."""
    with pytest.raises(ResourceAdmissionError, match="thread cap"):
        bootstrap_worker_runtime(0)
