"""Tests for processing memory budget enforcement."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.processing.runtime.memory import (
    MemoryWatchdog,
    close_memmap,
    release_memmap_pages,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_watchdog_records_available_memory_and_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A memory sample records both process RSS and system availability."""
    monkeypatch.setattr(
        "faninsar.processing.runtime.memory.live_rss_bytes",
        lambda: 512 * 1024 * 1024,
    )
    monkeypatch.setattr(
        "faninsar.processing.runtime.memory.available_memory_bytes",
        lambda: 8 * 1024 * 1024 * 1024,
    )
    watchdog = MemoryWatchdog(limit_mib=1024.0, minimum_available_mib=2048.0)

    snapshot = watchdog.sample("tile:0", collect=False)
    output = tmp_path / "memory.jsonl"
    watchdog.write_jsonl(output)

    assert snapshot.available_mib == pytest.approx(8192.0)
    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert record["label"] == "tile:0"
    assert record["available_mib"] == pytest.approx(8192.0)


def test_watchdog_stops_before_system_memory_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low available system memory trips the guard before an OS-level OOM."""
    monkeypatch.setattr(
        "faninsar.processing.runtime.memory.live_rss_bytes",
        lambda: 512 * 1024 * 1024,
    )
    monkeypatch.setattr(
        "faninsar.processing.runtime.memory.available_memory_bytes",
        lambda: 1024 * 1024 * 1024,
    )
    watchdog = MemoryWatchdog(limit_mib=4096.0, minimum_available_mib=2048.0)

    with pytest.raises(MemoryError, match="available"):
        watchdog.sample("tile:0", collect=False)


def test_dynamic_watchdog_refreshes_limit_when_available_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A low-available start must not freeze a limit that later free memory would allow.

    Regression for IW1_b4 geo: start with ~4 GiB available froze limit ~2409 MiB;
    at coregister available recovered to ~13 GiB but live=2511 still tripped the
    frozen ceiling.
    """
    live_mib = {"value": 1256.0}
    available_mib = {"value": 4351.0}

    monkeypatch.setattr(
        "faninsar.processing.runtime.memory.live_rss_bytes",
        lambda: int(live_mib["value"] * 1024 * 1024),
    )
    monkeypatch.setattr(
        "faninsar.processing.runtime.memory.available_memory_bytes",
        lambda: int(available_mib["value"] * 1024 * 1024),
    )

    watchdog = MemoryWatchdog.for_current_system(
        reserve_mib=3072.0,
        minimum_available_mib=3584.0,
        maximum_process_mib=7168.0,
    )
    # Initial freeze under low available ≈ live + max(1024, 4351-3072) = 2535.
    assert watchdog.limit_mib == pytest.approx(1256.0 + (4351.0 - 3072.0))

    snap0 = watchdog.sample("geo_pipeline:start", collect=False)
    assert snap0.live_mib == pytest.approx(1256.0)

    # Later stage: process grows modestly, system free memory recovers.
    live_mib["value"] = 2511.0
    available_mib["value"] = 13788.0
    snap1 = watchdog.sample("geo_coregister:0:128", collect=False)
    assert snap1.live_mib == pytest.approx(2511.0)
    # Effective limit should track recovered available (capped at max process).
    assert watchdog.limit_mib == pytest.approx(7168.0)
    assert not watchdog.killed


def test_release_memmap_pages_preserves_flushed_data(tmp_path: Path) -> None:
    """Releasing resident mapped pages preserves the disk-backed array."""
    path = tmp_path / "field.float32"
    field = np.memmap(path, mode="w+", dtype=np.float32, shape=(32, 16))
    field[:] = np.arange(field.size, dtype=np.float32).reshape(field.shape)
    expected = np.asarray(field).copy()

    release_memmap_pages(field)
    reopened = np.memmap(path, mode="r", dtype=np.float32, shape=field.shape)

    np.testing.assert_array_equal(reopened, expected)


def test_close_memmap_allows_data_to_be_reopened(tmp_path: Path) -> None:
    """Closing a mapped intermediate releases it without deleting its data."""
    path = tmp_path / "field.float32"
    field = np.memmap(path, mode="w+", dtype=np.float32, shape=(8, 4))
    field[:] = 7.0

    close_memmap(field)
    reopened = np.memmap(path, mode="r", dtype=np.float32, shape=(8, 4))

    np.testing.assert_array_equal(reopened, 7.0)
