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


MAX_INPUT_PIXELS = 2**28
"""Largest admitted two-dimensional raster size in pixels."""

MAX_ESTIMATED_WORK = 2**36
"""Largest admitted deterministic element-operation estimate."""


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


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Deterministic byte and work estimate for a built-in processing stage.

    Parameters
    ----------
    usage : ResourceUsage
        Byte-class reservation required before the stage allocates arrays.
    work : int
        Conservative element-operation upper bound.  This value is checked
        against :data:`MAX_ESTIMATED_WORK`; it is not a performance estimate.

    Notes
    -----
    Estimates describe only FanInSAR built-in numerical paths.  A custom
    runtime Python strategy remains trusted code and owns its own allocation
    behaviour.

    """

    usage: ResourceUsage
    work: int


def _checked_integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Return a bounded integer or raise the sole resource admission error."""
    if type(value) is not int or value < minimum:
        _raise_resource_error(f"{name} must be an integer >= {minimum}")
    return value


def _checked_add(*values: int, name: str) -> int:
    """Add non-negative integer terms without exceeding the work ceiling."""
    total = 0
    for value in values:
        _checked_integer(value, name)
        total += value
        if total > MAX_ESTIMATED_WORK:
            _raise_resource_error(f"{name} exceeds the hard work ceiling")
    return total


def _checked_product(*values: int, name: str) -> int:
    """Multiply non-negative integer terms with a fixed hard ceiling."""
    product = 1
    for value in values:
        _checked_integer(value, name)
        if value and product > MAX_ESTIMATED_WORK // value:
            _raise_resource_error(f"{name} exceeds the hard work ceiling")
        product *= value
    return product


def _ceil_dividend(dividend: int, divisor: int, *, name: str) -> int:
    """Return a checked positive ceiling division result."""
    _checked_integer(dividend, name, minimum=1)
    _checked_integer(divisor, name, minimum=1)
    return 1 + (dividend - 1) // divisor


def _ceil_log2(value: int) -> int:
    """Return ``ceil(log2(value))`` for a positive integer."""
    _checked_integer(value, "logarithm argument", minimum=1)
    return (value - 1).bit_length()


def _checked_shape(shape: tuple[int, int]) -> tuple[int, int, int]:
    """Validate a raster shape and return ``(height, width, pixels)``."""
    if not isinstance(shape, tuple) or len(shape) != 2:
        _raise_resource_error("resource estimate shape must be a two-dimensional tuple")
    height = _checked_integer(shape[0], "shape height", minimum=1)
    width = _checked_integer(shape[1], "shape width", minimum=1)
    pixels = _checked_product(height, width, name="input pixels")
    if pixels > MAX_INPUT_PIXELS:
        _raise_resource_error("input pixels exceed the hard admission ceiling")
    return height, width, pixels


def _built_in_filter_work(
    *,
    output_pixels: int,
    input_shape: tuple[int, int],
    phase_filter: object | None,
) -> tuple[int, int]:
    """Return deterministic work and device bytes for one built-in filter.

    Custom filters intentionally contribute neither term: they are trusted
    runtime Python and are outside the deterministic admission promise.
    """
    if phase_filter is None:
        return 0, 0
    try:
        from faninsar.processing.interferometry.phase_filter import (
            BoxcarFilter,
            GaussianFilter,
            GoldsteinWerner,
        )
    except ImportError:  # pragma: no cover - imports are package-local
        return 0, 0
    if isinstance(phase_filter, GoldsteinWerner):
        patch = _checked_integer(phase_filter.patch_size, "Goldstein patch", minimum=1)
        step = patch // 2
        height, width = input_shape
        patches = _checked_product(
            _ceil_dividend(height, step, name="Goldstein rows"),
            _ceil_dividend(width, step, name="Goldstein columns"),
            name="Goldstein patches",
        )
        patch_area = _checked_product(patch, patch, name="Goldstein patch area")
        work = _checked_product(
            2,
            patches,
            patch_area,
            1 + _ceil_log2(patch_area),
            name="Goldstein work",
        )
        # Input, overlap-add output, support, and one FFT patch's complex
        # working buffers.  complex64 is the narrow on-device Stack contract.
        device = _checked_add(
            _checked_product(output_pixels, 17, name="Goldstein device bytes"),
            _checked_product(patch_area, 32, name="Goldstein patch bytes"),
            name="Goldstein device bytes",
        )
        return work, device
    if isinstance(phase_filter, BoxcarFilter):
        kernel = _checked_product(*phase_filter.window, name="boxcar kernel area")
        return (
            _checked_product(output_pixels, kernel, name="boxcar work"),
            _checked_product(output_pixels, 25, name="boxcar device bytes"),
        )
    if isinstance(phase_filter, GaussianFilter):
        import math

        radii = tuple(
            math.ceil(phase_filter.truncate * sigma) for sigma in phase_filter.sigma
        )
        kernel = _checked_product(
            2 * radii[0] + 1,
            2 * radii[1] + 1,
            name="Gaussian kernel area",
        )
        return (
            _checked_product(output_pixels, kernel, name="Gaussian work"),
            _checked_product(output_pixels, 25, name="Gaussian device bytes"),
        )
    return 0, 0


def estimate_formation_resources(
    *,
    shape: tuple[int, int],
    multilook: tuple[int, int],
    coherence_window: tuple[int, int] | None,
    phase_filter: object | None,
) -> ResourceEstimate:
    """Estimate the built-in IFG formation live set before raster processing.

    Parameters
    ----------
    shape : tuple[int, int]
        Full-resolution ``(azimuth, range)`` scene grid.
    multilook : tuple[int, int]
        Positive output look factors.
    coherence_window : tuple[int, int] or None
        Centered first-stage coherence support, or direct output-block MLE.
    phase_filter : object or None
        Filter strategy.  Only built-in filters add a deterministic filter
        workspace; custom trusted strategies are excluded.

    Returns
    -------
    ResourceEstimate
        Decoded/staged, temporary, device, and work reservations.

    Raises
    ------
    ResourceAdmissionError
        If dimensions or the conservative work estimate exceed hard limits.

    """
    height, width, pixels = _checked_shape(shape)
    if not isinstance(multilook, tuple) or len(multilook) != 2:
        _raise_resource_error("multilook must be a two-dimensional tuple")
    azimuth = _checked_integer(multilook[0], "multilook azimuth", minimum=1)
    range_ = _checked_integer(multilook[1], "multilook range", minimum=1)
    block_area = _checked_product(azimuth, range_, name="multilook area")
    output_rows = _ceil_dividend(height, azimuth, name="output rows")
    output_cols = _ceil_dividend(width, range_, name="output columns")
    output_pixels = _checked_product(output_rows, output_cols, name="output pixels")
    formation_work = _checked_product(
        2, output_pixels, block_area, name="formation work"
    )
    if coherence_window is None:
        coherence_work = _checked_product(
            output_pixels, block_area, name="direct coherence work"
        )
    else:
        if not isinstance(coherence_window, tuple) or len(coherence_window) != 2:
            _raise_resource_error("coherence_window must be a two-dimensional tuple")
        window_area = _checked_product(
            _checked_integer(coherence_window[0], "coherence azimuth", minimum=1),
            _checked_integer(coherence_window[1], "coherence range", minimum=1),
            name="coherence window area",
        )
        coherence_work = _checked_add(
            _checked_product(pixels, window_area, name="window coherence work"),
            _checked_product(output_pixels, block_area, name="look coherence work"),
            name="coherence work",
        )
    filter_work, filter_device = _built_in_filter_work(
        output_pixels=output_pixels,
        input_shape=(output_rows, output_cols),
        phase_filter=phase_filter,
    )
    work = _checked_add(
        formation_work, coherence_work, filter_work, name="formation work"
    )
    # Two complex source rasters, output complex/float layers, and the
    # persisted boolean support.  Counts are deliberately conservative.
    decoded = _checked_add(
        _checked_product(pixels, 16, name="formation decoded bytes"),
        _checked_product(output_pixels, 21, name="formation decoded bytes"),
        name="formation decoded bytes",
    )
    staged = _checked_product(output_pixels, 21, name="formation staged bytes")
    temporary = _checked_add(
        _checked_product(pixels, 24, name="formation temporary bytes"),
        _checked_product(output_pixels, 24, name="formation temporary bytes"),
        filter_device,
        name="formation temporary bytes",
    )
    return ResourceEstimate(
        usage=ResourceUsage(
            files=5,
            chunks=1,
            encoded_bytes=staged,
            decoded_bytes=decoded,
            temporary_bytes=temporary,
            device_bytes=filter_device,
        ),
        work=work,
    )


def estimate_unwrap_decode_resources(
    *,
    shape: tuple[int, int],
    coherence_present: bool,
) -> ResourceEstimate:
    """Estimate Dataset decode and device conversion before unwrapping.

    The concrete Stack Dataset reads complex IFG, wrapped phase, amplitude,
    validity mask, and an optional float32 coherence layer.  Only wrapped
    phase, optional coherence, and the support mask cross to the Stack device.
    """
    _, _, pixels = _checked_shape(shape)
    decoded_per_pixel = 8 + 4 + 4 + 1 + (4 if coherence_present else 0)
    decoded = _checked_product(pixels, decoded_per_pixel, name="unwrap decoded bytes")
    device = _checked_product(
        pixels,
        4 + 1 + (4 if coherence_present else 0),
        name="unwrap device bytes",
    )
    return ResourceEstimate(
        usage=ResourceUsage(
            files=5 if coherence_present else 4,
            chunks=1,
            decoded_bytes=decoded,
            temporary_bytes=decoded,
            device_bytes=device,
        ),
        work=0,
    )


def estimate_spatial_irls_resources(
    *,
    shape: tuple[int, int],
    active_edges: int,
    component_bbox_areas: tuple[int, ...],
    max_iter: int,
    cg_max_iter: int,
    phase_itemsize: int = 4,
) -> ResourceEstimate:
    """Estimate component-aware DCT/PCG workspace after support decoding.

    ``component_bbox_areas`` contains the bounding-box area of every active
    connected component.  The reservation follows the documented
    ``S * (6*B + 4*M)`` DCT/PCG live-set bound.
    """
    _, _, pixels = _checked_shape(shape)
    edges = _checked_integer(active_edges, "active edges")
    outer = _checked_integer(max_iter, "IRLS max_iter", minimum=1)
    inner = _checked_integer(cg_max_iter, "IRLS cg_max_iter", minimum=1)
    itemsize = _checked_integer(phase_itemsize, "phase itemsize", minimum=1)
    if not component_bbox_areas:
        _raise_resource_error("IRLS requires at least one component")
    areas = tuple(
        _checked_integer(area, "component bounding-box area", minimum=1)
        for area in component_bbox_areas
    )
    total_area = _checked_add(*areas, name="component bounding-box area")
    largest_area = max(areas)
    workspace = _checked_product(
        itemsize,
        _checked_add(
            _checked_product(6, total_area, name="IRLS workspace"),
            _checked_product(4, largest_area, name="IRLS workspace"),
            name="IRLS workspace",
        ),
        name="IRLS workspace bytes",
    )
    dct_work = _checked_product(
        4,
        total_area,
        1 + _ceil_log2(max(1, largest_area)),
        name="IRLS DCT work",
    )
    inner_work = _checked_add(
        _checked_product(12, edges, name="IRLS edge work"),
        _checked_product(8, pixels, name="IRLS pixel work"),
        dct_work,
        name="IRLS inner work",
    )
    work = _checked_add(
        _checked_product(2, pixels, name="IRLS initial work"),
        _checked_product(
            outer,
            _checked_add(
                _checked_product(8, edges, name="IRLS outer edge work"),
                _checked_product(8, pixels, name="IRLS outer pixel work"),
                _checked_product(inner, inner_work, name="IRLS iterative work"),
                name="IRLS outer work",
            ),
            name="IRLS solver work",
        ),
        name="IRLS solver work",
    )
    return ResourceEstimate(
        usage=ResourceUsage(
            chunks=len(areas),
            temporary_bytes=workspace,
            device_bytes=workspace,
        ),
        work=work,
    )


def reserve_estimate(
    ledger: ResourceAdmissionLedger,
    estimate: ResourceEstimate,
) -> ResourceReservation:
    """Reserve all byte classes represented by one checked estimate."""
    return ledger.reserve(**{
        name: getattr(estimate.usage, name)
        for name in ResourceUsage.__dataclass_fields__
    })


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
    "MAX_ESTIMATED_WORK",
    "MAX_INPUT_PIXELS",
    "ProcessTreeAdmission",
    "ProcessTreeMemoryWatchdog",
    "ProcessTreeSampler",
    "ProcessTreeSamplingError",
    "ProcessTreeSnapshot",
    "ResourceAdmissionError",
    "ResourceAdmissionLedger",
    "ResourceBudget",
    "ResourceEstimate",
    "ResourceReservation",
    "ResourceUsage",
    "bootstrap_worker_runtime",
    "estimate_formation_resources",
    "estimate_spatial_irls_resources",
    "estimate_unwrap_decode_resources",
    "reserve_estimate",
]
