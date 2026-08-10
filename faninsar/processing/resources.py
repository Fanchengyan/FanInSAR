"""Finite resource admission and process-tree telemetry for processing runs."""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

import psutil

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from faninsar.processing.contracts.prepared_geometry import ResourceLimits


class ResourceAdmissionError(RuntimeError):
    """Raised when a run cannot be admitted under its finite resource budget."""


def bootstrap_worker_runtime(thread_cap: int = 1) -> None:
    """Apply deterministic BLAS/Torch thread caps in a worker initializer.

    Parameters
    ----------
    thread_cap : int, optional
        Positive upper bound for numerical worker threads.

    Raises
    ------
    ResourceAdmissionError
        If the cap is invalid, Torch cannot be imported, or Torch rejects the
        cap in a fresh worker.

    Notes
    -----
    The function is intentionally a small top-level callable so it can be
    passed as a ``ProcessPoolExecutor`` initializer under the ``spawn`` start
    method.  It sets environment variables before the worker task imports the
    Sentinel-1 readers and performs the corresponding Torch runtime setup.

    """
    if not isinstance(thread_cap, int) or thread_cap <= 0:
        _raise_resource_error("worker thread cap must be a positive integer")
    value = str(thread_cap)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[variable] = value
    os.environ["FANINSAR_WORKER_BOOTSTRAPPED"] = "1"
    try:
        import torch

        torch.set_num_threads(thread_cap)
        torch.set_num_interop_threads(thread_cap)
    except (ImportError, RuntimeError) as error:
        logger.exception("worker numerical runtime bootstrap failed")
        _raise_resource_error(f"worker numerical runtime bootstrap failed: {error}")


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Finite decoded, temporary, process, and disk budget for one run."""

    max_files: int
    max_chunks: int
    max_encoded_bytes: int
    max_decoded_bytes: int
    max_temporary_bytes: int
    max_workers: int
    max_processes: int
    disk_reserve_bytes: int
    max_rss_bytes: int
    max_device_bytes: int

    @classmethod
    def from_limits(cls, limits: ResourceLimits) -> ResourceBudget:
        """Build a runtime budget from the authenticated contract limits."""
        return cls(
            max_files=limits.max_files,
            max_chunks=limits.max_chunks,
            max_encoded_bytes=limits.max_encoded_bytes,
            max_decoded_bytes=limits.max_decoded_bytes,
            max_temporary_bytes=limits.max_temporary_bytes,
            max_workers=limits.max_workers,
            max_processes=limits.max_processes,
            disk_reserve_bytes=limits.disk_reserve_bytes,
            max_rss_bytes=limits.max_rss_bytes,
            max_device_bytes=limits.max_device_bytes,
        )

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive administrator-approved limits."""
        for name in (
            "max_files",
            "max_chunks",
            "max_encoded_bytes",
            "max_decoded_bytes",
            "max_temporary_bytes",
            "max_workers",
            "max_processes",
            "disk_reserve_bytes",
            "max_rss_bytes",
            "max_device_bytes",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                _raise_resource_error(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Current reservation totals held by a ledger."""

    files: int = 0
    chunks: int = 0
    encoded_bytes: int = 0
    decoded_bytes: int = 0
    temporary_bytes: int = 0
    workers: int = 0
    processes: int = 0
    device_bytes: int = 0


@dataclass
class ResourceReservation:
    """Releasable reservation returned by :class:`ResourceAdmissionLedger`."""

    _ledger: ResourceAdmissionLedger = field(repr=False)
    _amounts: ResourceUsage = field(repr=False)
    _released: bool = field(default=False, init=False)

    def release(self) -> None:
        """Release this reservation exactly once."""
        if not self._released:
            self._ledger._release(self._amounts)
            self._released = True

    def __enter__(self) -> Self:
        """Return this reservation for a context-managed admission."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the reservation when leaving its context."""
        self.release()


@dataclass
class ProcessTreeAdmission:
    """Combine finite reservations with fail-closed process-tree telemetry.

    Parameters
    ----------
    budget : ResourceBudget
        Authenticated finite limits for the processing run.
    root : pathlib.Path or str
        Existing local work directory used for the reservation ledger.
    workers : int
        Number of worker slots admitted for the run.
    files : int, optional
        Number of output payload files reserved before worker submission.
    root_pid : int, optional
        Process-tree root.  Defaults to the current process.

    Notes
    -----
    The admission is deliberately explicit.  Existing callers that do not
    provide a validated ``ResourceLimits`` profile retain their historical
    behavior, but they are not eligible to claim the resource gate.

    """

    budget: ResourceBudget
    root: str | Path
    workers: int
    files: int = 0
    root_pid: int | None = None
    ledger: ResourceAdmissionLedger | None = field(default=None, init=False)
    reservation: ResourceReservation | None = field(default=None, init=False)
    watchdog: ProcessTreeMemoryWatchdog | None = field(default=None, init=False)

    @classmethod
    def from_limits(
        cls,
        limits: ResourceLimits,
        root: str | Path,
        *,
        workers: int,
        files: int = 0,
        root_pid: int | None = None,
    ) -> ProcessTreeAdmission:
        """Build an admission from one validated neutral resource profile."""
        return cls(
            budget=ResourceBudget.from_limits(limits),
            root=root,
            workers=workers,
            files=files,
            root_pid=root_pid,
        )

    def __enter__(self) -> Self:
        """Reserve resources and start the complete-tree watchdog."""
        if self.workers <= 0:
            _raise_resource_error("admitted worker count must be positive")
        if self.files < 0:
            _raise_resource_error("reserved file count must be non-negative")
        self.ledger = ResourceAdmissionLedger(self.budget, self.root)
        try:
            self.reservation = self.ledger.reserve(
                files=self.files,
                workers=self.workers,
                processes=self.workers + 1,
            )
            self.watchdog = ProcessTreeMemoryWatchdog(
                max_rss_bytes=self.budget.max_rss_bytes,
                root_pid=self.root_pid,
            )
            self.watchdog.start()
        except BaseException:
            if self.reservation is not None:
                self.reservation.release()
                self.reservation = None
            raise
        return self

    def __exit__(self, *_: object) -> None:
        """Stop telemetry and release reservations after worker completion."""
        try:
            if self.watchdog is not None:
                self.watchdog.stop()
        finally:
            if self.reservation is not None:
                self.reservation.release()
                self.reservation = None


class ResourceAdmissionLedger:
    """Thread-safe in-process resource reservation ledger.

    Parameters
    ----------
    budget : ResourceBudget
        Authenticated finite limits for this run.
    root : pathlib.Path or str
        Existing local work directory used for disk-free preflight.

    Notes
    -----
    The ledger is intentionally conservative: it reserves decoded and
    temporary bytes before an array/container is opened.  A durable
    cross-process ledger still belongs to the persistent-cache activation
    gate; this class prevents accidental over-admission within one run.

    """

    def __init__(self, budget: ResourceBudget, root: str | Path) -> None:
        """Create a zero-use ledger after validating the work directory."""
        self.budget = budget
        self.root = Path(root)
        if self.root.is_symlink() or not self.root.is_dir():
            _raise_resource_error("resource ledger root must be a real directory")
        self._lock = threading.RLock()
        self._usage = ResourceUsage()

    @property
    def usage(self) -> ResourceUsage:
        """Return a consistent snapshot of current reservations."""
        with self._lock:
            return self._usage

    def reserve(
        self,
        *,
        files: int = 0,
        chunks: int = 0,
        encoded_bytes: int = 0,
        decoded_bytes: int = 0,
        temporary_bytes: int = 0,
        workers: int = 0,
        processes: int = 0,
        device_bytes: int = 0,
    ) -> ResourceReservation:
        """Atomically reserve bounded resources before opening or decoding.

        Raises
        ------
        ResourceAdmissionError
            If a request is negative, exceeds a configured limit, or leaves
            less than the configured disk reserve.

        """
        amounts = ResourceUsage(
            files=files,
            chunks=chunks,
            encoded_bytes=encoded_bytes,
            decoded_bytes=decoded_bytes,
            temporary_bytes=temporary_bytes,
            workers=workers,
            processes=processes,
            device_bytes=device_bytes,
        )
        self._validate_request(amounts)
        with self._lock:
            candidate = _add_usage(self._usage, amounts)
            checks = {
                "files": (candidate.files, self.budget.max_files),
                "chunks": (candidate.chunks, self.budget.max_chunks),
                "encoded_bytes": (
                    candidate.encoded_bytes,
                    self.budget.max_encoded_bytes,
                ),
                "decoded_bytes": (
                    candidate.decoded_bytes,
                    self.budget.max_decoded_bytes,
                ),
                "temporary_bytes": (
                    candidate.temporary_bytes,
                    self.budget.max_temporary_bytes,
                ),
                "workers": (candidate.workers, self.budget.max_workers),
                "processes": (candidate.processes, self.budget.max_processes),
                "device_bytes": (candidate.device_bytes, self.budget.max_device_bytes),
            }
            for name, (used, limit) in checks.items():
                if used > limit:
                    self._reject(f"resource admission exceeds {name}: {used} > {limit}")
            free_bytes = shutil.disk_usage(self.root).free
            if free_bytes - candidate.temporary_bytes < self.budget.disk_reserve_bytes:
                self._reject("resource admission would violate the disk reserve")
            self._usage = candidate
            return ResourceReservation(self, amounts)

    @staticmethod
    def _validate_request(amounts: ResourceUsage) -> None:
        """Reject negative or non-integer reservation requests."""
        for name in amounts.__dataclass_fields__:
            value = getattr(amounts, name)
            if not isinstance(value, int) or value < 0:
                _raise_resource_error(f"{name} must be a non-negative integer")

    def _release(self, amounts: ResourceUsage) -> None:
        """Release one reservation without allowing underflow."""
        with self._lock:
            candidate = _subtract_usage(self._usage, amounts)
            if any(
                getattr(candidate, name) < 0
                for name in ResourceUsage.__dataclass_fields__
            ):
                _raise_resource_error("resource ledger underflow")
            self._usage = candidate

    @staticmethod
    def _reject(message: str) -> None:
        """Log and raise an admission failure."""
        logger.error("resource admission rejected: %s", message)
        raise ResourceAdmissionError(message)


def _add_usage(left: ResourceUsage, right: ResourceUsage) -> ResourceUsage:
    """Add two immutable usage records."""
    return ResourceUsage(
        **{
            name: getattr(left, name) + getattr(right, name)
            for name in ResourceUsage.__dataclass_fields__
        }
    )


def _subtract_usage(left: ResourceUsage, right: ResourceUsage) -> ResourceUsage:
    """Subtract two immutable usage records."""
    return ResourceUsage(
        **{
            name: getattr(left, name) - getattr(right, name)
            for name in ResourceUsage.__dataclass_fields__
        }
    )


@dataclass(frozen=True, slots=True)
class ProcessTreeSnapshot:
    """One complete parent-plus-descendant RSS observation."""

    root_pid: int
    process_ids: tuple[tuple[int, float], ...]
    rss_bytes: int
    sampled_at: float


class ProcessTreeSamplingError(RuntimeError):
    """Raised when a complete process-tree sample cannot be established."""


class ProcessTreeSampler:
    """Sample parent and descendant RSS with PID-start identity checks."""

    def __init__(self, root_pid: int | None = None) -> None:
        """Create a sampler rooted at the current process by default."""
        self.root_pid = os.getpid() if root_pid is None else root_pid

    def sample(self) -> ProcessTreeSnapshot:
        """Return one complete RSS sample or fail closed."""
        try:
            root = psutil.Process(self.root_pid)
            processes = [root, *root.children(recursive=True)]
            identities: list[tuple[int, float]] = []
            total_rss = 0
            for process in processes:
                with process.oneshot():
                    create_time = process.create_time()
                    rss = process.memory_info().rss
                identities.append((process.pid, create_time))
                total_rss += int(rss)
        except (psutil.Error, OSError) as error:
            logger.exception("complete process-tree RSS sample failed")
            _raise_sampling_error(error)
        return ProcessTreeSnapshot(
            root_pid=self.root_pid,
            process_ids=tuple(sorted(identities)),
            rss_bytes=total_rss,
            sampled_at=time.time(),
        )


@dataclass
class ProcessTreeMemoryWatchdog:
    """Fail-closed monitor for summed parent and descendant RSS."""

    max_rss_bytes: int
    interval_seconds: float = 0.1
    root_pid: int | None = None
    samples: list[ProcessTreeSnapshot] = field(default_factory=list)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(
        default_factory=threading.Event,
        init=False,
        repr=False,
    )
    _error: BaseException | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the <=100 ms sampling interval and finite RSS limit."""
        if self.max_rss_bytes <= 0:
            _raise_resource_error("max_rss_bytes must be positive")
        if not 0 < self.interval_seconds <= 0.1:
            _raise_resource_error("process-tree interval must be between 0 and 0.1s")

    def sample(self) -> ProcessTreeSnapshot:
        """Take one sample and reject an over-limit or incomplete observation."""
        snapshot = ProcessTreeSampler(self.root_pid).sample()
        self.samples.append(snapshot)
        if snapshot.rss_bytes > self.max_rss_bytes:
            self._fail(
                ResourceAdmissionError(
                    f"process-tree RSS exceeded limit: {snapshot.rss_bytes} > "
                    f"{self.max_rss_bytes}"
                )
            )
            self.raise_if_failed()
        return snapshot

    def start(self) -> None:
        """Start a daemon monitor after an immediate baseline sample."""
        if self._thread is not None and self._thread.is_alive():
            return
        self.sample()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="faninsar-process-tree-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitor and surface any recorded failure."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        """Raise the first telemetry or RSS failure, if any."""
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        """Sample until stopped, converting all gaps into a failed state."""
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sample()
            except BaseException as error:
                self._fail(error)
                return

    def _fail(self, error: BaseException) -> None:
        """Record the first monitor failure and stop further publication."""
        if self._error is None:
            self._error = error
            logger.error("process-tree watchdog failed closed: %s", error)
        self._stop.set()


def _raise_resource_error(message: str) -> None:
    """Raise a resource admission error after the caller has logged context."""
    logger.error("resource admission rejected: %s", message)
    raise ResourceAdmissionError(message)


def _raise_sampling_error(cause: BaseException) -> None:
    """Raise a telemetry error while preserving its original cause."""
    message = "process-tree telemetry is unavailable; publication must stop"
    raise ProcessTreeSamplingError(message) from cause


__all__ = [
    "ProcessTreeAdmission",
    "ProcessTreeMemoryWatchdog",
    "ProcessTreeSampler",
    "ProcessTreeSamplingError",
    "ProcessTreeSnapshot",
    "ResourceAdmissionError",
    "ResourceAdmissionLedger",
    "ResourceBudget",
    "ResourceReservation",
    "ResourceUsage",
    "bootstrap_worker_runtime",
]
