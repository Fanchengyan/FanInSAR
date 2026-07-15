"""Process RSS and wallclock helpers for stage-level profiling.

Used by production instrumentation and offline profiling scripts. Peak and
live RSS are reported in mebibytes (MiB). Wallclock is reported in seconds.
On macOS ``resource.getrusage`` returns bytes for ``ru_maxrss``; on Linux it
returns kibibytes.
"""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = setup_logger(__name__)


def _maxrss_bytes() -> int:
    """Return peak RSS in bytes for this process (platform-normalised)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(ru)
    # Linux and most Unix: kibibytes
    return int(ru) * 1024


def live_rss_bytes() -> int:
    """Return current RSS in bytes (psutil if available, else max-RSS proxy)."""
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return _maxrss_bytes()


def bytes_to_mib(n: float) -> float:
    """Convert bytes to mebibytes."""
    return float(n) / (1024.0 * 1024.0)


@dataclass
class MemorySnapshot:
    """One labelled RSS sample, optionally with wallclock seconds."""

    label: str
    live_mib: float
    peak_mib: float
    wall_s: float | None = None


@dataclass
class MemoryWatchdog:
    """Collect stage RSS / wallclock samples and abort if live RSS exceeds a limit.

    Parameters
    ----------
    limit_mib : float
        Kill threshold for live RSS in MiB. Default 16384 (16 GiB).
    records : list
        Mutable sample log.

    """

    limit_mib: float = 16384.0
    records: list[MemorySnapshot] = field(default_factory=list)
    killed: bool = False

    def sample(
        self,
        label: str,
        *,
        collect: bool = True,
        wall_s: float | None = None,
    ) -> MemorySnapshot:
        """Record live/peak RSS; raise :class:`MemoryError` if over limit."""
        if collect:
            gc.collect()
        snap = MemorySnapshot(
            label=label,
            live_mib=bytes_to_mib(live_rss_bytes()),
            peak_mib=bytes_to_mib(_maxrss_bytes()),
            wall_s=wall_s,
        )
        self.records.append(snap)
        if wall_s is None:
            logger.info(
                "MEM %-40s live=%.0f MiB peak=%.0f MiB",
                label,
                snap.live_mib,
                snap.peak_mib,
            )
        else:
            logger.info(
                "MEM %-40s live=%.0f MiB peak=%.0f MiB wall=%.3fs",
                label,
                snap.live_mib,
                snap.peak_mib,
                wall_s,
            )
        if snap.live_mib > self.limit_mib:
            self.killed = True
            msg = (
                f"RSS watchdog kill at '{label}': "
                f"live={snap.live_mib:.0f} MiB > limit={self.limit_mib:.0f} MiB"
            )
            logger.error(msg)
            raise MemoryError(msg)
        return snap

    def format_table(self) -> str:
        """Return a TSV summary of all samples."""
        lines = ["label\tlive_mib\tpeak_mib\twall_s"]
        for r in self.records:
            wall = "" if r.wall_s is None else f"{r.wall_s:.4f}"
            lines.append(f"{r.label}\t{r.live_mib:.1f}\t{r.peak_mib:.1f}\t{wall}")
        if self.killed:
            lines.append("WATCHDOG_KILLED\t1\t1\t")
        return "\n".join(lines) + "\n"

    def run_stage(
        self,
        label: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Sample before/after a stage callable with wallclock and return result."""
        self.sample(f"{label}:before")
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        wall = time.perf_counter() - t0
        self.sample(f"{label}:after", wall_s=wall)
        return result
