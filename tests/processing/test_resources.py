"""Tests for finite resource admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from faninsar.processing.runtime.resources import (
    ResourceAdmissionError,
    ResourceAdmissionLedger,
    ResourceBudget,
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
