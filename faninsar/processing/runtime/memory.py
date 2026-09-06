"""Process RSS and wallclock helpers for stage-level profiling.

Used by production instrumentation and offline profiling scripts. Peak and
live RSS are reported in mebibytes (MiB). Wallclock is reported in seconds.
On macOS ``resource.getrusage`` returns bytes for ``ru_maxrss``; on Linux it
returns kibibytes.
"""

from __future__ import annotations

import gc
import json
import mmap
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

logger = setup_logger(__name__)


def _maxrss_bytes() -> int:
    """Return peak RSS in bytes for this process (platform-normalised)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(ru)
    # Linux and most Unix: kibibytes
    return int(ru) * 1024


def live_rss_bytes() -> int:
    """Return current process RSS in bytes."""
    return int(psutil.Process(os.getpid()).memory_info().rss)


def available_memory_bytes() -> int:
    """Return currently available system memory in bytes."""
    return int(psutil.virtual_memory().available)


def release_memmap_pages(array: np.memmap) -> None:
    """Flush a memmap and release clean resident pages when supported.

    Parameters
    ----------
    array : numpy.memmap
        Disk-backed array whose clean pages may be evicted.

    """
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise"):
        mapping.madvise(mmap.MADV_DONTNEED)


def close_memmap(array: np.memmap) -> None:
    """Flush and close a disk-backed array mapping.

    Parameters
    ----------
    array : numpy.memmap
        Mapping to close while preserving its backing file.

    """
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def bytes_to_mib(n: float) -> float:
    """Convert bytes to mebibytes."""
    return float(n) / (1024.0 * 1024.0)


@dataclass
class MemorySnapshot:
    """One labelled RSS sample, optionally with wallclock seconds."""

    label: str
    live_mib: float
    peak_mib: float
    available_mib: float
    wall_s: float | None = None


@dataclass
class MemoryWatchdog:
    """Collect stage RSS / wallclock samples and abort if live RSS exceeds a limit.

    Parameters
    ----------
    limit_mib : float
        Kill threshold for live RSS in MiB when ``dynamic_limit`` is False.
        With dynamic limits this is only the initial estimate and is refreshed
        on every :meth:`sample`.
    minimum_available_mib : float
        Immediate stop threshold for system available memory.
    reserve_mib : float, optional
        System memory kept outside the processing budget when dynamic.
    maximum_process_mib : float, optional
        Absolute process RSS ceiling when dynamic.
    dynamic_limit : bool
        When True (default for :meth:`for_current_system`), recompute the RSS
        ceiling from current live RSS and available memory on every sample so
        a low-available start does not permanently freeze a too-tight limit.
    records : list
        Mutable sample log.

    """

    limit_mib: float = 16384.0
    minimum_available_mib: float = 2048.0
    reserve_mib: float | None = None
    maximum_process_mib: float | None = None
    dynamic_limit: bool = False
    records: list[MemorySnapshot] = field(default_factory=list)
    killed: bool = False
    record_path: Path | None = None

    @classmethod
    def for_current_system(
        cls,
        *,
        reserve_mib: float = 3072.0,
        minimum_available_mib: float = 3584.0,
        maximum_process_mib: float = 7168.0,
        record_path: str | Path | None = None,
    ) -> MemoryWatchdog:
        """Build a watchdog with a dynamic RSS ceiling from system memory.

        The initial limit is estimated from current RSS and available memory,
        but unlike a frozen ceiling it is refreshed on every sample using the
        same ``reserve_mib`` / ``maximum_process_mib`` policy.  This avoids
        false kills when the process starts under temporary memory pressure
        and free memory later recovers (e.g. after other bursts finish).

        Parameters
        ----------
        reserve_mib : float, optional
            System memory kept outside the processing budget.
        minimum_available_mib : float, optional
            Immediate stop threshold for system available memory.
        maximum_process_mib : float, optional
            Absolute process RSS ceiling.
        record_path : str or pathlib.Path, optional
            JSONL path refreshed after every memory sample.

        Returns
        -------
        MemoryWatchdog
            Watchdog with a conservative dynamic RSS ceiling.

        """
        live_mib = bytes_to_mib(live_rss_bytes())
        available_mib = bytes_to_mib(available_memory_bytes())
        limit_mib = cls._compute_dynamic_limit(
            live_mib=live_mib,
            available_mib=available_mib,
            reserve_mib=reserve_mib,
            maximum_process_mib=maximum_process_mib,
        )
        return cls(
            limit_mib=limit_mib,
            minimum_available_mib=minimum_available_mib,
            reserve_mib=reserve_mib,
            maximum_process_mib=maximum_process_mib,
            dynamic_limit=True,
            record_path=None if record_path is None else Path(record_path),
        )

    @staticmethod
    def _compute_dynamic_limit(
        *,
        live_mib: float,
        available_mib: float,
        reserve_mib: float,
        maximum_process_mib: float,
    ) -> float:
        """Return process RSS ceiling from live RSS and free system memory."""
        additional_budget = max(1024.0, available_mib - reserve_mib)
        return min(maximum_process_mib, live_mib + additional_budget)

    def effective_limit_mib(self, *, live_mib: float, available_mib: float) -> float:
        """Return the RSS kill threshold for the current sample.

        When ``dynamic_limit`` is enabled the ceiling tracks free system memory
        so a temporary low-available start cannot permanently starve a later
        stage that has enough headroom.
        """
        if (
            self.dynamic_limit
            and self.reserve_mib is not None
            and self.maximum_process_mib is not None
        ):
            return self._compute_dynamic_limit(
                live_mib=live_mib,
                available_mib=available_mib,
                reserve_mib=self.reserve_mib,
                maximum_process_mib=self.maximum_process_mib,
            )
        return self.limit_mib

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
            available_mib=bytes_to_mib(available_memory_bytes()),
            wall_s=wall_s,
        )
        effective_limit = self.effective_limit_mib(
            live_mib=snap.live_mib,
            available_mib=snap.available_mib,
        )
        # Keep limit_mib in sync for logging / external inspection.
        self.limit_mib = effective_limit
        self.records.append(snap)
        if self.record_path is not None:
            self.write_jsonl(self.record_path)
        if wall_s is None:
            logger.info(
                "MEM %-40s live=%.0f MiB peak=%.0f MiB available=%.0f MiB "
                "limit=%.0f MiB",
                label,
                snap.live_mib,
                snap.peak_mib,
                snap.available_mib,
                effective_limit,
            )
        else:
            logger.info(
                "MEM %-40s live=%.0f MiB peak=%.0f MiB available=%.0f MiB "
                "limit=%.0f MiB wall=%.3fs",
                label,
                snap.live_mib,
                snap.peak_mib,
                snap.available_mib,
                effective_limit,
                wall_s,
            )
        rss_exceeded = snap.live_mib > effective_limit
        system_low = snap.available_mib < self.minimum_available_mib
        if rss_exceeded or system_low:
            self.killed = True
            msg = (
                f"RSS watchdog kill at '{label}': "
                f"live={snap.live_mib:.0f} MiB, limit={effective_limit:.0f} MiB, "
                f"available={snap.available_mib:.0f} MiB, "
                f"minimum_available={self.minimum_available_mib:.0f} MiB"
            )
            logger.error(msg)
            raise MemoryError(msg)
        return snap

    def format_table(self) -> str:
        """Return a TSV summary of all samples."""
        lines = ["label\tlive_mib\tpeak_mib\tavailable_mib\twall_s"]
        for r in self.records:
            wall = "" if r.wall_s is None else f"{r.wall_s:.4f}"
            lines.append(
                f"{r.label}\t{r.live_mib:.1f}\t{r.peak_mib:.1f}\t"
                f"{r.available_mib:.1f}\t{wall}"
            )
        if self.killed:
            lines.append("WATCHDOG_KILLED\t1\t1\t1\t")
        return "\n".join(lines) + "\n"

    def write_jsonl(self, path: str | Path) -> Path:
        """Write collected memory samples as newline-delimited JSON.

        Parameters
        ----------
        path : str or pathlib.Path
            Output JSONL file.

        Returns
        -------
        pathlib.Path
            Written path.

        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            for record in self.records:
                stream.write(
                    json.dumps(
                        {
                            "label": record.label,
                            "live_mib": record.live_mib,
                            "peak_mib": record.peak_mib,
                            "available_mib": record.available_mib,
                            "wall_s": record.wall_s,
                        }
                    )
                    + "\n"
                )
        return output

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
