"""Download progress reporting for byte-range span fetches.

:class:`DownloadProgress` observes a batch of span downloads and drives a
``tqdm`` bar from streamed HTTP body bytes — total size, bytes received,
and rate — so a long ``from_search`` download is visible as data arrives
instead of only when a span or file flush completes.  It only observes;
it never changes the fetch behavior.

The downloader feeds it three events: :meth:`add_bytes` as body bytes
stream in, :meth:`span_done` when each span finishes, and
:meth:`add_job` when streaming planning discovers more spans.
"""

from __future__ import annotations

from tqdm import tqdm


class DownloadProgress:
    """Track streamed HTTP body bytes on a ``tqdm`` bar.

    Parameters
    ----------
    label : str
        Human-readable unit being downloaded (e.g. the SAFE or item name),
        shown at the front of the bar.
    total_bytes : int
        Expected total body bytes across all spans in the batch.
    total_spans : int
        Number of spans in the batch.
    log_interval : float, optional
        Minimum seconds between bar redraws (``tqdm`` ``mininterval``).

    Examples
    --------
    >>> progress = DownloadProgress(label="scene", total_bytes=1024, total_spans=1)
    >>> progress.add_bytes(512)
    >>> progress.span_done("row-0-10")
    >>> progress.finish()

    """

    def __init__(
        self,
        *,
        label: str,
        total_bytes: int,
        total_spans: int,
        log_interval: float = 0.5,
    ) -> None:
        """Initialize progress tracking for one span batch."""
        self.label = label
        self.total_bytes = max(0, int(total_bytes))
        self.total_spans = max(0, int(total_spans))
        self.log_interval = float(log_interval)
        self._done_bytes = 0
        self._done_spans = 0
        self._in_flight: set[str] = set()
        self._bar = tqdm(
            total=self.total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"download {label}",
            mininterval=self.log_interval,
            dynamic_ncols=True,
        )
        self._set_postfix()

    def add_bytes(self, n: int) -> None:
        """Count ``n`` HTTP body bytes received (called per streamed chunk)."""
        n = int(n)
        self._done_bytes += n
        self._bar.update(n)

    def add_job(self, *, nbytes: int, nspans: int = 1) -> None:
        """Enlarge expected totals when another owner queues measurement spans.

        Streaming planning discovers jobs after the first downloads have
        already started, so the reporter cannot freeze totals at construction.

        Parameters
        ----------
        nbytes : int
            Additional expected body bytes.
        nspans : int, optional
            Additional span count. Default is 1.

        """
        self.total_bytes += max(0, int(nbytes))
        self.total_spans += max(0, int(nspans))
        self._bar.total = self.total_bytes
        self._set_postfix()
        self._bar.refresh()

    def span_start(self, label: str) -> None:
        """Note a span as in flight (for the postfix hint)."""
        self._in_flight.add(label)
        self._set_postfix()

    def span_done(self, label: str) -> None:
        """Record a finished span; byte counts already moved in :meth:`add_bytes`."""
        self._in_flight.discard(label)
        self._done_spans += 1
        self._set_postfix()

    def finish(self) -> None:
        """Close the bar when the batch completes."""
        self._set_postfix()
        self._bar.close()

    def _set_postfix(self) -> None:
        postfix = {"spans": f"{self._done_spans}/{self.total_spans}"}
        if self._in_flight:
            sample = sorted(self._in_flight)[:2]
            postfix["in-flight"] = ", ".join(sample)
        self._bar.set_postfix(postfix, refresh=False)
