# PROPOSAL-0018 / PROPOSAL-0019 execution checklist

Date: 2026-08-11

## Frozen qualification corpus

- Three real Sentinel-1 SAFE acquisitions from the same first-frame footprint:
  2016-12-07, 2016-12-31, and 2017-01-24. An initially selected middle-frame
  2016 pair was rejected during preflight because the available 2017 scene
  belongs to the first frame; it was replaced before the full campaign.
- Radar reference configuration: IW1 bursts 0, 1, and 2; Pair mode;
  Torch CPU; multilook 16 x 40; Goldstein alpha 0.
- Canonical source hashes and exact paths are stored in
  `qualified-raw-safe-sbas-20260811/frozen-corpus.json` under the DATA2
  campaign root.
- Candidate activation records are isolated test evidence. They must never be
  submitted as official Waymark gate events before every blocking gate passes.

## Ordered remaining gates

1. **Artifact transactions** — implement immutable generations, atomic CURRENT
   publication, root lock/lease, quota checks, corruption rejection, and crash
   recovery for IFG, unwrap, and time-series artifacts.
2. **Temporal scientific gate** — derive and test deterministic convergence,
   closure, retained-pixel, connected-network, and SBAS residual diagnostics.
   Do not invent an arbitrary scientific threshold merely to pass.
3. **Raw SAFE full chain** — run the signed candidate qualified path from the
   frozen SAFE inputs through coregistration, IFG, unwrap, SBAS, and Zarr.
4. **Qualification matrix** — exercise Radar and Geo, geometry/pair/network,
   ROI/no-ROI, multilook/filter, holdout, reopen, missing/corrupt payload, and
   exact provenance on the frozen corpus wherever scientifically applicable.
5. **Performance/resource packet** — compare complete old/reference and new
   Stack paths with repeated runs, process-tree sampling at <=100 ms, peak RSS,
   Torch peaks, disk bytes, and cold/warm timings. Preserve exactness.
6. **Immutable evidence** — each run writes an exclusive run-scoped summary;
   cold reopen may not overwrite the fresh-run record.
7. **Independent verification** — rerun focused and broad regression tests,
   Ruff, diff checks, real artifact reopen, and Waymark verification. Official
   qualification events may be issued only if all blocking gates pass.

## Current status

- Frozen same-frame corpus/config: complete.
- Artifact transaction implementation: complete at commits `7964d3a`,
  `09ebf0e`, `0a576db`, and `225d8c4`; real Radar/Geo ROI cases now publish
  and cold-open complete Stack parent generations binding IFG, unwrap and SBAS.
- Temporal algebraic scientific gate: complete at commit `18abf3a`; an
  oracle/error-budget-backed physical closure threshold remains a human policy
  choice rather than an implementation default.
- Raw SAFE signed qualification-candidate full chain: PASS, three dates x
  three bursts x three IFGs, 52,274 valid SBAS pixels.
- Transactional fresh-process cold reopen: PASS.
- Qualification matrix: bounded Radar/Geo ROI and no-ROI public runs complete;
  remaining breadth is three-date/three-burst Geo and full current-code
  geometry/network coverage.
- Performance/resource packet: isolated three-burst Pair improvement is PASS
  (+32.3036% median, exact bytes, <=100 ms external sampling); complete old/new
  Stack repetitions and Linux descendant-covering 14 GiB enforcement remain
  open.
- Transaction fault/security packet: bounded hostile-path and parent-generation
  tests PASS; multi-date crash/GC/phase-lineage campaign remains open.
- Official Waymark activation: intentionally not issued while blocking gates
  remain.
