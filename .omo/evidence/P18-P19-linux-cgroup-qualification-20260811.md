# P18/P19 Linux cgroup qualification packet

Date: 2026-08-11

Host: `cryogpu-hk` (Linux, 36 CPUs, 2 x RTX 2080 Ti, CUDA 12.8 runtime).
The current product tree was synchronized to
`/home/fancy/tmp/faninsar-stac-p18linux-20260811`; the three frozen SAFE ZIPs
were copied with SHA-256 verification. Runs used a user systemd scope with:

```text
MemoryMax=15032385536 (14 GiB)
MemorySwapMax=0
TasksMax=256
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

## Linux focused regression

The transaction, IFG, scene, Stack, pipeline, timeseries, and unwrap suite
completed with `80 passed, 1 warning` inside the 14-GiB scope.

The Linux run exposed and fixed two portability defects:

1. staging and IFG roots were created with the process umask instead of an
   explicit private mode;
2. descriptor-backed generation paths were interpreted as `/dev/fd` symlinks
   by Linux `pathlib` operations.

The transaction layer now enforces a private staging umask, creates managed
roots as `0700`, hardens caller-owned managed roots to `0700`, and provides
descriptor-relative `stat/is_file/is_dir/is_symlink/rglob` behavior. macOS
focused regression after the fix: `59 passed`; Linux focused regression:
`80 passed`.

## Real Linux runs

| Case | Result | Elapsed | cgroup result |
|---|---|---:|---|
| Radar geometry, 3 dates, IW1 burst 0 | PASS | 262.5968 s | `/usr/bin/time` peak 4.35 GiB; first monitor wrapper did not persist cgroup peak |
| Radar network, 3 dates, IW1 burst 0 | PASS | 480.9643 s | cgroup peak `8,511,827,968` B; `MemoryMax=15,032,385,536` B; no swap/OOM |
| Geo geometry, 3 dates, IW1 burst 0 | FAIL-CLOSED | — | reached temporal SBAS gate; no converged pixels, no OOM; publication correctly rejected |
| Radar network, 2 dates, IW1 burst 0 | PASS | 234.8048 s | direct cgroup peak `7,960,080,384` B; maximum sampler gap `54 ms` |
| Geo geometry, 2 dates, IW1 burst 0 | PASS | 65.3000 s | direct cgroup peak `2,834,051,072` B; maximum sampler gap `63 ms` |

The Radar network run produced the complete three-date chain and persisted
three IFGs, three unwrap artifacts, a timeseries generation, and a Stack
parent generation. The Geo run did not publish a scientifically unqualified
timeseries; its failure is the intended temporal quality gate, not a resource
failure.

The two-date direct-monitor runs demonstrate the same hard cap and sub-100-ms
telemetry on both Radar and Geo while producing complete SBAS outputs. The
three-date Geo case remains a real-data convergence issue rather than a
resource issue: its two-date subset converges and publishes successfully, but
the three-date temporal network has no pixels that satisfy the strict quality
gate.

The first shell monitor sampled cgroup state approximately every 40 ms but
reported one 121-ms scheduling gap. The hard cgroup ceiling is authoritative;
the monitor was subsequently changed to read `memory.current`/`memory.peak`
directly instead of invoking `systemctl` for every sample. The direct-monitor
two-date Radar and Geo runs then measured 54 ms and 63 ms maximum gaps.

## Interpretation

Linux now closes the hard-memory enforcement gap for an actual current Stack
run: the process tree is placed in a 14-GiB cgroup and the observed Radar
network peak is below the cap. It does not close the separate old-vs-new full
Stack performance comparison or the human science decision on temporal
closure/residual thresholds. P18/P19 therefore remain `implementing`.
