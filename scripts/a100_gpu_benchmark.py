"""A100 GPU benchmark: serial-CPU vs dask-torch-CUDA for the full stack pipeline.

Compares wall time and memory across window sizes and executor paths on an
A100 (or any CUDA) server. Reports speedup and peak GPU VRAM to evaluate
whether the P7 Lanczos CUDA path delivers a large win end-to-end.

Usage (on the A100 box, with the 3 SLC files copied locally)::

    uv run python scripts/a100_gpu_benchmark.py \\
        --slc-root /path/to/TEST_sentinel-1/sentinel-slc \\
        --max-gib 40 \\
        --windows 256 512 1024

Outputs a markdown table to stdout and writes ``a100_benchmark.json``.

Run BOTH executors per window size and compute speedup = serial / dask_torch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def gpu_vram_gib() -> float:
    """Current GPU VRAM allocated in GiB (CUDA only)."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**3)


def peak_gpu_vram_gib() -> float:
    if not HAS_TORCH or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_gpu_stats() -> None:
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def run_one(
    scenes: list[Path],
    out_dir: Path,
    *,
    height: int,
    width: int,
    executor: str,
    device: str,
    unwrap_method: str,
    max_gib: float,
) -> dict:
    """Run one stack pipeline configuration and return timing + memory."""
    from faninsar.processing.pipeline import run_stack_pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    reset_gpu_stats()
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss / (1024**3)

    t0 = time.time()
    try:
        result = run_stack_pipeline(
            scenes[:3],
            output_dir=out_dir,
            height=height,
            width=width,
            multilook=(2, 2),
            invert_timeseries=True,
            executor=executor,
            device=device,
            unwrap_method=unwrap_method,
            invert_device=device,
        )
        elapsed = time.time() - t0
        peak_rss = max(rss0, proc.memory_info().rss / (1024**3))
        peak_vram = peak_gpu_vram_gib()
        ok = True
        n_pairs = len(result.pair_results)
        ts_shape = result.timeseries.cumulative.shape if result.timeseries else None
    except Exception as e:  # noqa: BLE001
        elapsed = time.time() - t0
        peak_rss = proc.memory_info().rss / (1024**3)
        peak_vram = peak_gpu_vram_gib()
        ok = False
        n_pairs = 0
        ts_shape = None
        print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    return {
        "executor": executor,
        "device": device,
        "unwrap_method": unwrap_method,
        "height": height,
        "width": width,
        "elapsed_s": round(elapsed, 2),
        "peak_rss_gib": round(peak_rss, 2),
        "peak_vram_gib": round(peak_vram, 3),
        "ok": ok,
        "n_pairs": n_pairs,
        "ts_shape": list(ts_shape) if ts_shape else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slc-root",
        default="/Volumes/DATA2/TEST_sentinel-1/sentinel-slc",
        help="Directory containing S1A*.zip SAFE files",
    )
    parser.add_argument("--max-gib", type=float, default=40.0)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="Square window sizes (height=width) to benchmark",
    )
    parser.add_argument(
        "--unwrap-method",
        default="irls",
        choices=["irls", "dct_irls"],
    )
    parser.add_argument("--out-json", default="a100_benchmark.json")
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Skip the dask-torch/cuda run (CPU-only box)",
    )
    args = parser.parse_args()

    slc_root = Path(args.slc_root)
    scenes = sorted(slc_root.glob("S1A_IW_SLC*.zip"))
    if len(scenes) < 3:
        print(f"ERROR: need >=3 SLC scenes in {slc_root}, found {len(scenes)}")
        return 1

    has_cuda = HAS_TORCH and torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results: list[dict] = []
    base_out = Path("a100_bench_out")

    for win in args.windows:
        print(f"\n=== Window {win}x{win} ===")
        # 1. serial CPU baseline
        print(f"  [serial/cpu] running...")
        r_serial = run_one(
            scenes,
            base_out / f"serial_{win}",
            height=win,
            width=win,
            executor="serial",
            device="cpu",
            unwrap_method=args.unwrap_method,
            max_gib=args.max_gib,
        )
        results.append(r_serial)
        print(
            f"  serial/cpu: {r_serial['elapsed_s']}s "
            f"RSS={r_serial['peak_rss_gib']}GiB ok={r_serial['ok']}"
        )

        # 2. dask-torch CUDA
        if not args.skip_cuda and has_cuda:
            print(f"  [dask-torch/cuda] running...")
            r_cuda = run_one(
                scenes,
                base_out / f"cuda_{win}",
                height=win,
                width=win,
                executor="dask-torch",
                device="cuda",
                unwrap_method=args.unwrap_method,
                max_gib=args.max_gib,
            )
            results.append(r_cuda)
            print(
                f"  dask-torch/cuda: {r_cuda['elapsed_s']}s "
                f"RSS={r_cuda['peak_rss_gib']}GiB "
                f"VRAM={r_cuda['peak_vram_gib']}GiB ok={r_cuda['ok']}"
            )
            if r_serial["ok"] and r_cuda["ok"] and r_cuda["elapsed_s"] > 0:
                speedup = r_serial["elapsed_s"] / r_cuda["elapsed_s"]
                print(f"  SPEEDUP: {speedup:.2f}x")
        elif not has_cuda:
            print("  [dask-torch/cuda] SKIPPED (no CUDA)")

    # Write JSON
    Path(args.out_json).write_text(json.dumps(results, indent=2))

    # Markdown table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print("| window | executor | device | time(s) | RSS(GiB) | VRAM(GiB) | speedup |")
    print("|--------|----------|--------|---------|----------|-----------|---------|")
    for r in results:
        speedup = ""
        if r["executor"] == "dask-torch" and r["device"] == "cuda":
            # find matching serial
            serial = next(
                (
                    x
                    for x in results
                    if x["executor"] == "serial"
                    and x["height"] == r["height"]
                ),
                None,
            )
            if serial and serial["ok"] and r["ok"] and r["elapsed_s"] > 0:
                speedup = f"{serial['elapsed_s'] / r['elapsed_s']:.2f}x"
        print(
            f"| {r['height']} | {r['executor']} | {r['device']} | "
            f"{r['elapsed_s']} | {r['peak_rss_gib']} | "
            f"{r['peak_vram_gib']} | {speedup} |"
        )
    print(f"\nJSON written to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())