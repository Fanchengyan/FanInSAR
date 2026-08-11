# P18/P19 current-head execution result

Date: 2026-08-11

## Completed automatically

- Raw SAFE -> coregistration -> persisted scenes -> IFG -> spatial unwrap ->
  temporal IRLS -> rank-aware SBAS -> radians/metres Zarr is integrated and
  passed on the frozen three-date/three-burst Radar Pair candidate.
- Pair reuse is exact and faster: three interleaved frozen-burst runs measured
  +32.303628% total median improvement and +41.707580% product-stage
  improvement, with byte-identical per-burst outputs, +371,015,680 B RSS delta
  (within the +512 MiB gate), 6,518,194,176 B peak, and external sampling gap
  85.709 ms.
- The memory-lifetime fix reduced the raw three-date/three-burst candidate from
  11,304,828,928 B to 8,764,243,968 B peak RSS while preserving 52,274 valid
  SBAS pixels and all hashes.
- Radar and Geo public ROI/no-ROI cases all passed on real SAFE inputs. The
  latest four-case run completed in 150.76 s with peak RSS 7,267,057,664 B.
- Each of those four cases now publishes a Stack parent generation binding the
  exact ordered IFG, unwrap and SBAS generation IDs/digests. A fresh process
  reopened all four parent generations successfully.
- IFG/unwrap/time-series transactions now use immutable generations, bounded
  CURRENT pointers, root locks, leases, quotas, descriptor-relative no-follow
  reads, hardlink/symlink rejection, root replacement protection, and
  fail-closed corruption/partial checks.
- Current focused regression: 79 passed, one third-party deprecation warning.
  Changed-file Ruff and diff checks pass.

Relevant commits: `7964d3a`, `18abf3a`, `f7d6f37`, `531ddda`, `09ebf0e`,
`0a576db`, `225d8c4`.

## Still open and why

1. **Same-corpus breadth:** the full acceptance packet still asks for the
   three-date/three-burst Geo and current-code Radar geometry/network matrices.
   The available host has already terminated larger Geo runs under memory
   pressure; the bounded two-date Radar/Geo matrix is not a substitute.
2. **Complete old/new Stack performance:** the isolated Pair gate passes, but
   there is no three-repetition rolled-back-versus-current full Radar and Geo
   Stack packet covering cold/warm preparation, IFG-only reruns, unwrap/SBAS,
   Torch allocations, disk bytes, and child attribution.
3. **Absolute hard memory gate:** macOS evidence observes process-tree RSS, but
   cannot prove the required descendant-covering Linux cgroup/container hard
   ceiling of 14 GiB.
4. **Full real fault/lineage campaign:** bounded hostile artifact tests pass;
   the proposal-wide multi-date crash, stale-writer, reader/GC race, and
   operation-hash phase-lineage campaign remains to be executed.
5. **Physical SBAS policy:** algebraic rank/reconstruction gates pass, but the
   retained/converged fraction and physical closure/residual limits require an
   approved science error budget or independent oracle. No threshold was
   invented to force activation.
6. **Waymark governance:** official typed P18/P19 gate events and activation
   remain intentionally absent. Both proposals stay `implementing` until the
   open acceptance items are independently verified.

## Authoritative artifacts

- `.omo/evidence/P18-P19-qualified-raw-safe-full-chain-20260811.md`
- `.omo/evidence/P18-P19-real-roi-matrix-20260811.md`
- `.omo/evidence/P18-P19-qualification-matrix-synthesis-20260811.md`
- `/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/roi-matrix-stack-generation-retry-20260811/summary.json`
