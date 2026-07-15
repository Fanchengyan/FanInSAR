"""End-to-end stack pipeline test on real Sentinel-1 SLCs with memory monitoring.

Runs the three local SAFE scenes through ``run_stack_pipeline`` while sampling
RSS of the current Python process. If RSS exceeds ``--max-gib`` (default 10),
the run is killed and the peak memory + stage logs are reported so the cause
can be localized.

Usage::

    uv run python scripts/e2e_stack_memory_check.py --max-gib 10
    uv run python scripts/e2e_stack_memory_check.py --executor dask-torch --device cpu
    uv run python scripts/e2e_stack_memory_check.py --executor dask-torch --device cuda  # on A100

This is a manual integration script (not a pytest) so it can run outside the
test suite and emit live memory telemetry.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import psutil

SLC_ROOT = Path("/Volumes/DATA2/TEST_sentinel-1/sentinel-slc")


def _rss_gib(proc: psutil.Process) -> float:
    """Return current RSS in GiB."""
    try:
        return proc.memory_info().rss / (1024**3)
    except psutil.NoSuchProcess:
        return 0.0


def _peak_rss_gib(proc: psutil.Process) -> float:
    """Peak RSS across children + self in GiB."""
    peak = _rss_gib(proc)
    for child in proc.children(recursive=True):
        try:
            peak = max(peak, _rss_gib(child))
        except psutil.NoSuchProcess:
            continue
    # also check children-peak via psutil
    try:
        for child in proc.children(recursive=True):
            peak = max(peak, child.memory_info().rss / (1024**3))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return peak


class MemoryGuard:
    """Background thread that samples RSS and kills the process on OOM."""

    def __init__(self, max_gib: float, interval_s: float = 1.0) -> None:
        self.max_gib = max_gib
        self.interval_s = interval_s
        self.peak = 0.0
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid())

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = _peak_rss_gib(self._proc)
            self.peak = max(self.peak, rss)
            self.samples.append((time.time(), rss))
            if rss > self.max_gib:
                print(
                    f"\n!!! MEMORY LIMIT EXCEEDED: {rss:.2f} GiB > {self.max_gib} GiB !!!",
                    file=sys.stderr,
                )
                print(f"Peak so far: {self.peak:.2f} GiB", file=sys.stderr)
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(self.interval_s)

    def report(self) -> str:
        if not self.samples:
            return "no samples"
        lines = [
            f"Peak RSS: {self.peak:.2f} GiB",
            f"Samples: {len(self.samples)}",
        ]
        if len(self.samples) > 10:
            mid = len(self.samples) // 2
            lines.append(
                f"Start: {self.samples[0][1]:.2f} GiB | "
                f"Mid: {self.samples[mid][1]:.2f} GiB | "
                f"End: {self.samples[-1][1]:.2f} GiB"
            )
        return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-gib", type=float, default=10.0)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument(
        "--executor",
        choices=["serial", "dask-torch"],
        default="serial",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--unwrap-method", default="irls")
    parser.add_argument("--multilook", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--no-invert", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    scenes = sorted(SLC_ROOT.glob("S1A_IW_SLC*.zip"))
    if len(scenes) < 3:
        print(f"ERROR: need >=3 SLC scenes, found {len(scenes)} in {SLC_ROOT}")
        return 1

    # Import after arg parse to keep memory baseline clean
    from faninsar.processing.pipeline import run_stack_pipeline

    out_dir = Path(args.output_dir) if args.output_dir else Path("_e2e_stack_out")
    out_dir.mkdir(exist_ok=True)

    guard = MemoryGuard(max_gib=args.max_gib, interval_s=1.0)
    guard.start()

    t0 = time.time()
    try:
        # Try passing executor/device/unwrap_method; fall back if knobs not yet wired
        try:
            result = run_stack_pipeline(
                scenes[:3],
                output_dir=out_dir,
                height=args.height,
                width=args.width,
                multilook=tuple(args.multilook),
                invert_timeseries=not args.no_invert,
                executor=args.executor,
                device=args.device,
                unwrap_method=args.unwrap_method,
            )
        except TypeError:
            print(
                "[warn] run_stack_pipeline does not accept executor/device/unwrap_method yet; "
                "falling back to default (serial/cpu/irls)."
            )
            result = run_stack_pipeline(
                scenes[:3],
                output_dir=out_dir,
                height=args.height,
                width=args.width,
                multilook=tuple(args.multilook),
                invert_timeseries=not args.no_invert,
            )
    finally:
        guard.stop()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"COMPLETED in {elapsed:.1f}s")
    print(f"Scenes: {len(result.scene_ids)} | Pairs: {len(result.pair_results)}")
    for state in result.pair_results:
        print(f"  pair {state.pair_id}: unwrapped={state.unwrapped_phase is not None}")
    if result.timeseries is not None:
        print(f"Timeseries shape: {result.timeseries.cumulative.shape}")
    print("-" * 60)
    print("MEMORY REPORT:")
    print(guard.report())
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
