# PROPOSAL-0018 / PROPOSAL-0019 qualification matrix synthesis

Date: 2026-08-11

## Decision

**Overall result: IMPLEMENTED FUNCTIONAL CHAIN, NOT YET VERIFIED OR ACTIVE.**

The current product commit is
`225d8c4e340abcdfbdb42da5d6b73ca7ffcfa98a`. A real, same-frame,
three-acquisition/three-burst Radar Pair Stack has completed from raw SAFE ZIPs
through coregistration, persisted scenes, three IFGs, spatial unwrap, temporal
reconciliation, rank-aware SBAS, transactional Zarr publication, and
fresh-process cold reopen. This closes the question of whether SBAS is
integrated and whether the functional full chain can run.

It does **not** close the conjunctive P18/P19 qualification contract. The
remaining proposal-level blockers are the remaining three-date/three-burst
same-corpus breadth in Geo and non-Pair Radar modes, complete old/new Stack
resource comparisons and descendant-covering
14 GiB hard limit, the complete parent/child transaction and phase-lineage
campaign, and a human-approved physical temporal quality policy or independent
scientific oracle. No official gate event should be issued from this packet.

## Evidence precedence

This synthesis uses the most recent immutable or generation-bound evidence.
The following older records are retained as history but are not current
qualification results:

- pre-fix IFGs under `seventh-batch-stack-3acq-radar/ifg/ml_8x20` and
  `eighth-batch-stack-radar-multilook/ifg/ml_2x10` are superseded because the
  declared multilook operation had not been applied;
- the machine report `temporal-sbas-quality-20260811.json` inspected old
  pre-generation artifacts and reported phase-reconstruction failures; the
  current implementation and fresh generation-bound summaries supersede it and
  pass the exact algebraic invariants;
- mutable top-level `summary.json` files from early Radar/Geo downstream runs
  are convenience records only. Qualification uses the immutable `runs/`
  summaries and their hashes.

## Consolidated qualification matrix

`PASS (bounded)` means the stated sub-scope is proven but the wider proposal
gate remains open. `OPEN` means no activation claim is permitted.

| Area | Current status | Exact evidence and boundary |
|---|---|---|
| Frozen real corpus/configuration | **PASS** | Same-frame 2016-12-07, 2016-12-31, 2017-01-24 SAFE hashes and IW1 burst selection are frozen in `qualified-raw-safe-sbas-20260811/frozen-corpus.json`. |
| Signed raw-SAFE qualified-path exercise | **PASS (candidate only)** | Public Stack wrapper, Radar/Pair, 3 dates x 3 bursts x 3 IFGs, 52,274 finite time-series pixels. Candidate-prefixed events use an isolated local authority and are not Waymark qualification events. Fresh summary SHA-256 `6947f59043441a93bf133e044559a33599c25f516dba0dc4fef1f62fac088f1a`; cold-reopen SHA-256 `5250e675043779d4bfe4243c35a4bc4c079da33142cc74dfce9694cb7e2fd1af`. |
| SBAS integration | **PASS** | Real Radar and Geo persisted-scene chains publish phase-radian and LOS-metre time series; the raw candidate additionally proves the public raw-to-SBAS path. This is functional integration, not scientific activation. |
| Radar Pair Stack, same corpus | **PASS** | The raw candidate covers the required 3 dates/3 pairs/3 IW1 bursts and cold reopen at current commit. |
| Radar Geometry Stack | **PASS (bounded), proposal gate OPEN** | Public raw 3-date/3-pair burst-0 run passed at commit `2c4cc76` with 20,560 finite time-series pixels. A current same-corpus three-burst Geometry run paired with the Pair run is not in the packet. |
| Radar Network residual semantics | **PASS (bounded), proposal gate OPEN** | Current public two-date/burst-0 network run measured Ampcor range `+0.1378309873 px` and ESD azimuth `-0.0043831856 px`; Ampcor azimuth was diagnostic and not applied. The full 3-date, multi-burst connected-network campaign is still absent. |
| Geo Geometry | **PASS (bounded), proposal gate OPEN** | Current public two-date/burst-0 Geo Geometry run passed on EPSG:32647/40 m with 6,916 finite pixels; process-tree peak was 2,845,786,112 B. Three dates x three bursts is not covered. |
| Geo Pair / Geo breadth | **PASS (fragmented), proposal gate OPEN** | Earlier real evidence covers 3 dates x 1 burst and 2 dates x 3 bursts with exact regenerated IFGs. It does not form the required current-code 3 dates x 3 bursts Geometry/Pair same-corpus matrix. |
| Multilook/filter correctness | **PASS** | After commit `da675da`, Radar and Geo derived IFGs at multiple looks/Goldstein settings are bitwise equal to the unchanged Pair oracle, with zero new coreg calls during regeneration. The incorrectly labelled pre-fix directories are ineligible. |
| Independent holdout | **PASS (bounded)** | Radar Pair IW1 burst 3 measured ESD azimuth `0.0157305607 px`; corrected derived products after the multilook fix are bitwise equal to the direct oracle. This is a two-date Pair holdout, which satisfies the proposal's “stack/pair” holdout wording but does not add Geo breadth. |
| ROI/no-ROI | **PASS (bounded)** | Current public wrapper completed raw SAFE to SBAS for Radar and Geo, both with and without ROI, on the same two-date corpus. Radar valid pixels were 47,104 (no ROI) and 21,882 (ROI); Geo valid pixels were 6,916 and 2,475. All four transactional time-series products and their new Stack parent generations cold-opened with exact IFG/unwrap/SBAS bindings. Summary SHA-256: `56b65af390a3bdbc111f58ad6e2b994bb342c5a2ecfbb1e5540aa26f692f75a`; process peak RSS 7,267,057,664 B. This closes the real public-surface ROI gate, not the missing three-date/three-burst Geo breadth. |
| P18 three-repetition Pair performance | **PASS (isolated Pair gate)** | Frozen three-burst packet at commit `7964d3a`: reference median 53.643706 s, prepared median 36.314843 s, total improvement 32.303628%, product-stage improvement 41.707580%, exact per-burst hashes, peak 6,518,194,176 B, prepared-minus-reference peak +371,015,680 B (within +512 MiB), and external maximum sampling gap 85.709 ms. Summary SHA-256 `640a7de9f78504ec3c74d3d068c84544d0043a4dc1e566e478847f2e5fd88307`; supervisor SHA-256 `2a2e335e7d82c504a7cccd8d9f5a37ca12c9f9b497a2160868f1cb57975fdec9`. Changes from `7964d3a` to current HEAD are temporal-quality code, not the prepared-geometry reuse kernel. |
| Complete Stack/Geo performance packet | **OPEN** | No complete rolled-back-vs-current 3-repetition Radar Stack and Geo Stack comparison includes preparation, IFG, unwrap, SBAS, cold/warm, call counts, Torch peaks and disk attribution. Single-path public timings do not satisfy an old/new gate. |
| Operational memory | **PASS on the corrected raw candidate** | Commit `f7d6f37` releases each completed Pair state before the next acquisition. The exact 3-date/3-burst rerun retained 52,274 valid pixels and reduced maximum RSS from 11,304,828,928 B to 8,764,243,968 B (8.16 GiB), below the 10 GiB target. |
| Absolute memory enforcement | **OPEN** | macOS process-tree telemetry proves observed RSS only. P18 requires a verified descendant-covering cgroup/container hard limit at or below 14 GiB; this cannot be claimed on the current host. |
| Derived IFG/unwrap/time-series transactions | **PASS** | Commits `7964d3a`, `09ebf0e` and `0a576db` add immutable generations, bounded `CURRENT`, root locks, leases, quotas, descriptor-relative no-follow reads, corruption rejection and cold reopen. Focused result: 32 transaction/IFG/time-series tests passed. |
| Complete Stack transaction/fault contract | **PASS (bounded)** | The new Stack parent generation binds the exact ordered IFG, unwrap and SBAS generation IDs/digests and was reopened in a fresh interpreter for all four real ROI cases. Symlinked namespaces, hardlinks, root replacement, pinned readers and malformed/partial/tampered child sets fail closed. A larger multi-date crash campaign remains open. |
| Exact temporal algebra/rank gate | **PASS** | Current code requires publishable pixels, full rank, integer cycle correction, exact `input + 2πk` reconstruction and criteria/report identity on resume. Fresh Geo and Radar summaries report zero integer/reconstruction error for published pixels. |
| Physical temporal/SBAS accuracy gate | **OPEN — human science policy required** | Geo retained 77.8958% of full-rank pixels; Radar retained 41.3618%. Modulo-closure p95 is 2.9892 rad (Geo) / 2.8156 rad (Radar); SBAS residual p95 is 2.0584 rad / 2.0034 rad. Algebra cannot determine a universal acceptable physical threshold. |
| Phase/residual exactly-once lineage | **PASS in focused tests, proposal campaign OPEN** | Typed state and fail-closed tests exist, but the complete real multi-mode phase-transition packet with operation IDs and input/output payload hashes has not been assembled. |
| Immutable fresh/reopen evidence | **PASS for raw candidate** | Fresh and reopen summaries are separate, exclusive files with hashes. Earlier mutable summaries remain non-qualifying historical records. |
| Focused regressions/lint | **PASS** | Latest focused processing suite reported 120 passed, one third-party warning; changed processing files pass Ruff and `git diff --check`. |
| Full repository regression | **OPEN for a clean all-green claim** | Latest run: 1,179 passed, 23 skipped, 22 failed; 21 are outside this processing delta and one RSS watchdog passes isolated. This is useful context, not an all-green qualification result. |
| Official gate events / activation | **CORRECTLY ABSENT** | `P18-provider-qualified`, `P19-stack-correctness-verified`, `P18-stack-qualified`, and `P19-stack-qualified-activation` must remain unissued while any blocking row above is open. |

## Resource snapshots that are valid but not complete comparisons

| Public real case | Scope | Result | Wall / peak process-tree RSS |
|---|---|---:|---:|
| `public-raw-radar-geometry-b0` | 3 dates, 3 pairs, Radar Geometry, burst 0 | PASS | 62.17 s supervisor wall / 5,603,442,688 B; maximum sample interval 102.82 ms, therefore the <=100 ms telemetry gate is not claimed |
| `public-radar-network-n2-b0` | 2 dates, Radar Network, burst 0 | PASS | 57.67 s / 9,446,342,656 B; maximum interval 55.95 ms |
| `public-geo-geometry-n2-b0-auto-grid` | 2 dates, Geo Geometry, burst 0 | PASS | 41.44 s / 2,845,786,112 B; maximum interval 93.04 ms |
| raw qualified candidate after memory fix | 3 dates, 3 bursts, Radar Pair, SBAS | PASS functional and 10 GiB operating RSS | 298.56 s process wall / 8,764,243,968 B |

These runs prove public-surface behavior and provide memory diagnostics. They
do not compare old and new Stack implementations and therefore cannot close the
P19 runtime gate.

## What can continue automatically

The following items need no new scientific decision and should continue as
execution work:

1. complete the same-current-commit 3-date/3-burst Geometry/Pair matrix in both
   Radar and Geo, plus the 3-date connected Network run;
2. obtain a Linux descendant-covering cgroup/container hard-cap run; the macOS
   candidate now passes the 10 GiB observed RSS target but cannot prove cgroup
   enforcement;
3. run complete three-repetition old/new Radar Stack and Geo Stack packets,
   including cold/warm, IFG-only zero-coreg, unwrap/SBAS, Torch and disk fields;
4. execute the proposal-wide transaction crash/stale-writer/reader/GC matrix
   and collect one complete real phase-transition lineage trace;
5. run the hard-ceiling campaign in a Linux cgroup v2/container environment
   that demonstrably covers the parent and all descendants;
6. rerun focused and broad tests from a clean checkout of the final landed
   commit and then perform independent Waymark verification.

## What genuinely requires human input

Only the following science/governance choices should stop automatic execution:

1. **Physical temporal quality policy:** approve minimum retained/converged
   fraction and physical closure/SBAS-residual limits, or name an independent
   trusted oracle and comparison tolerance. The measured distributions must be
   evaluated against that policy; a threshold must not be invented to make the
   current data pass.
2. **Activation decision after independent verification:** once every blocking
   criterion passes, the decision owner chooses the close-out Notes and permits
   the official typed gate events. Candidate authority records cannot be
   promoted into official events.

All other open rows are engineering execution or environment acquisition, not
questions that require changing interpolation precision or scientific kernels.

## Authoritative evidence paths

- `.omo/evidence/P18-P19-qualified-raw-safe-full-chain-20260811.md`
- `.omo/evidence/P18-P19-temporal-sbas-quality-20260811.md`
- `.omo/evidence/P18-P19-derived-artifact-transactions-20260811.md`
- `.omo/evidence/P18-P19-real-qualification-matrix-20260811.md`
- `.omo/evidence/P18-formal-three-burst-performance-20260811.md`
- `/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/qualified-raw-safe-sbas-20260811/`
- `/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/current-qualification-20260811/`
- `/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/qualification-matrix-20260811/`
- `/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/roi-matrix-after-public-plumbing-retry-20260811/`
