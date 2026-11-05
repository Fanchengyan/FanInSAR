"""DownloadProgress reports streamed HTTP body bytes with tqdm."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class _FakeBar:
    """Minimal tqdm stand-in that records update and total changes."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.n = 0
        self.total = kwargs.get("total", 0)
        self.closed = False
        self.postfix: dict[str, object] = {}
        self.updates: list[int] = []

    def update(self, n: int) -> None:
        self.n += int(n)
        self.updates.append(int(n))

    def set_postfix(
        self, ordered_dict: dict[str, object] | None = None, **kwargs: object
    ) -> None:
        self.postfix = dict(ordered_dict or {})
        self.postfix.update(kwargs)

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_add_bytes_updates_tqdm_before_span_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP body chunks move the bar immediately; span completion is only postfix."""
    from faninsar.remote.fetch.progress import DownloadProgress

    monkeypatch.setattr("faninsar.remote.fetch.progress.tqdm", _FakeBar)
    progress = DownloadProgress(
        label="CDSE ROI",
        total_bytes=13 * 1024 * 1024,
        total_spans=1,
    )
    bar = progress._bar
    assert isinstance(bar, _FakeBar)
    assert bar.total == 13 * 1024 * 1024
    progress.span_start("row-0")
    progress.add_bytes(4 * 1024 * 1024)
    assert bar.n == 4 * 1024 * 1024
    assert bar.updates == [4 * 1024 * 1024]
    assert bar.postfix["spans"] == "0/1"
    progress.add_job(nbytes=1024 * 1024, nspans=1)
    assert bar.total == 14 * 1024 * 1024
    progress.span_done("row-0")
    assert bar.n == 4 * 1024 * 1024
    assert bar.postfix["spans"] == "1/2"
    progress.finish()
    assert bar.closed
