"""Burst-archive array writes recover from short ``tofile`` writes."""

from __future__ import annotations

import errno
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.missions.s1 import processing as s1_processing
from faninsar.missions.s1.processing import _archive_tofile

if TYPE_CHECKING:
    from pathlib import Path


class _ShortWriteArray(np.ndarray):
    """Array subclass whose ``tofile`` simulates a short write."""

    def tofile(self, *_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "simulated short write")


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the ENOSPC backoff so tests fail fast."""
    monkeypatch.setattr(s1_processing, "_ARCHIVE_WRITE_RETRIES", 1)
    monkeypatch.setattr(s1_processing, "_ARCHIVE_RETRY_SECONDS", 0.0)


def test_writes_payload_directly(tmp_path: Path) -> None:
    """A healthy ``tofile`` write stores the exact bytes."""
    payload = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = str(tmp_path / "unit.f64")

    _archive_tofile(payload, path, "tag")

    written = np.fromfile(path, dtype=np.float64).reshape(3, 4)
    assert written.tolist() == payload.tolist()


def test_falls_back_to_chunked_write_on_short_tofile(
    tmp_path: Path,
) -> None:
    """A short-write OSError falls back to an explicit chunked rewrite."""
    payload = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = str(tmp_path / "unit.f64")

    _archive_tofile(payload.view(_ShortWriteArray), path, "tag")

    written = np.fromfile(path, dtype=np.float64).reshape(3, 4)
    assert written.tolist() == payload.tolist()


def test_enospc_recovers_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient ENOSPC chunk failure is retried until it succeeds."""
    payload = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = str(tmp_path / "unit.f64")
    real_write = os.write
    state = {"calls": 0}

    def flaky_write(_fd: int, _data: bytes) -> int:
        state["calls"] += 1
        if state["calls"] < 3:
            raise OSError(errno.ENOSPC, "simulated transient full")
        return real_write(_fd, _data)

    monkeypatch.setattr(s1_processing, "_ARCHIVE_WRITE_RETRIES", 3)
    monkeypatch.setattr(os, "write", flaky_write)
    _archive_tofile(payload.view(_ShortWriteArray), path, "tag")

    assert state["calls"] == 3
    written = np.fromfile(path, dtype=np.float64).reshape(3, 4)
    assert written.tolist() == payload.tolist()


def test_persistent_enospc_falls_back_to_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wedged file is completed through a sibling payload and renamed."""
    payload = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = str(tmp_path / "unit.f64")
    real_write = os.write
    state = {"calls": 0}

    def rejecting_write(fd: int, data: bytes) -> int:
        state["calls"] += 1
        if state["calls"] == 1:
            raise OSError(errno.ENOSPC, "simulated wedged file")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", rejecting_write)
    _archive_tofile(payload.view(_ShortWriteArray), path, "tag")

    assert state["calls"] == 2
    written = np.fromfile(path, dtype=np.float64).reshape(3, 4)
    assert written.tolist() == payload.tolist()
