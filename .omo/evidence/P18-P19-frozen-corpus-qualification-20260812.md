# P18/P19 frozen-corpus qualification ledger

Date: 2026-08-12

This is the consolidated real-data ledger. It records both passes and the
remaining scope limits; no missing experiment is inferred from a neighboring
campaign.

## Corpus and functional paths

- Radar raw SAFE, three dates, three IW1 bursts, three pair IFGs, spatial and
  temporal processing, SBAS Zarr output, and fresh generation reopen: PASS.
- Radar ROI and Geo ROI current-code runs: PASS with footprint-compatible
  grids; an incompatible Geo grid rejects before publication.
- Geo two-date raw pair: PASS. Three-date/three-burst Geo network has a
  current-code reopen of persisted artifacts and a separate raw public run,
  but not one fresh raw same-corpus end-to-end campaign.
- Radar multilook/filter derived IFGs, holdout, and network residual arcs:
  PASS with zero coregistration calls during derived-product formation.
- Focused product/transaction/Stack/unwrap/SBAS/reference tests: PASS (131
  passed, 1 skipped); changed processing surfaces pass Ruff and diff checks.

## Reference and performance

- ISCE2 same-grid wrapped oracle: PASS; filtered complex correlation 0.98493,
  filtered-core correlation 0.99010, phase RMSE 0.1581 rad.
- ISCE2 geometry offset parity: PASS; range difference 0.0387 px and azimuth
  difference 0.0153 px under the 0.1 px bound.
- Prepared-vs-reference three-burst Pair performance: PASS; +32.3036% median
  total speedup, exact output hashes, external process-tree maximum interval
  85.7 ms, and peak 6.518 GB.
- A fresh pre-P18 direct Pair baseline is 72.1626 s versus 69.9482 s for the
  current direct path, with exact common arrays. This control run is not the
  two-pass prepared-reuse benchmark.
- A fresh three-date Stack control-flow comparison is recorded, but the older
  commit already contains early prepared-field code and the current run adds
  transactional publication. It is not a valid complete old/current dual-
  domain performance gate.

## Fault and governance

- Real three-date generation reopen and tamper/partial/payload/symlink matrix:
  PASS; see the transaction fault packet.
- Linux 14 GiB cgroup and process-tree evidence: PASS in the existing Linux
  qualification packet.
- Full raw SAFE crash/restart, stale-writer/reader-GC, source-mutation, and
  operation-lineage replay: OPEN.
- Complete fresh same-corpus Radar+Geo geometry/pair/network/ROI/holdout
  matrix: OPEN because the available third SAFE is a different frame for the
  alternate corpus.
- Official typed P18 → P19 → Stack gate events: intentionally absent until
  the open evidence is closed and independently verified.

## Current status

The implementation and SBAS integration are functional for the qualified
persisted-scene and raw Radar scopes, and the lossless prepared Pair path has
measured performance improvement. P18/P19 remain `implementing`; this ledger
is evidence consolidation, not a status promotion.
