# P18/P19 continuous execution TODO — 2026-08-12

This checklist is the execution ledger for the current P18/P19 qualification
pass. It is updated continuously; unchecked items are not claimed as complete.

## Governance and contract

- [x] Load FanInSAR project registration and P18/P19 sidecar proposals.
- [x] Confirm closure removal is present in P18/P19 decisions and product tests.
- [x] Confirm proposal statuses remain `implementing` until official gates exist.

## Functional product paths

- [x] Run current raw SAFE → Radar Stack → IFG → unwrap → SBAS → Zarr.
- [x] Reopen and validate the complete Radar parent generation and child hashes.
- [x] Run current raw SAFE → Geo pair path.
- [x] Reopen current three-date/three-burst Geo network artifacts.
- [x] Run real Radar ROI path.
- [x] Run real Geo ROI path with a footprint-compatible grid.
- [x] Run focused regression, transaction, stack, unwrap, SBAS, and reference tests.
- [x] Run Ruff and diff checks for the changed processing surfaces.

## Reference and performance evidence

- [x] Compare FanInSAR/ISCE2 range and azimuth geometry offsets.
- [x] Record existing three-run prepared-vs-reference Pair performance evidence.
- [x] Record process-tree, RSS, Torch, and disk evidence already available.
- [x] Produce a same-crop/same-filter ISCE2 phase/coherence oracle; the
      unwrapped/displacement residual is retained as a diagnostic because
      multi-looking and unwrapping are not expected to close exactly.
- [ ] Produce a complete fresh old-vs-current Radar and Geo Stack packet with
      cold, warm, IFG-only, SBAS, process-tree, Torch, and disk measurements.
      A same-corpus old/current direct Pair and Stack control-flow packet was
      run and recorded, but it is not a qualified dual-domain comparison.

## Fault and lifecycle qualification

- [x] Run bounded manifest, payload, partial, symlink, mixed-generation, and
      reopen fail-closed checks.
- [ ] Run the complete real multi-date crash/restart, stale-writer/reader-GC,
      source-mutation, and exactly-once operation-lineage campaign. Bounded
      real-generation and hostile artifact checks pass; a full multi-date
      fault campaign still needs an isolated writer/reader harness.
- [x] Reconcile all real-data matrix and holdout evidence into one
      frozen-corpus qualification packet; open scope and corpus mismatches are
      recorded explicitly rather than treated as passes.
- [ ] Submit the final independent Waymark verification report. The fresh
      verifier has produced an `INCONCLUSIVE` report, but Waymark refuses to
      start a new verification round while P18 is `implementing` rather than
      `review`.
- [ ] Issue the official typed P18 → P19 → Stack activation gate chain. This
      remains intentionally absent because the verifier still reports open
      same-corpus dual-domain, crash/source-mutation, and full Stack
      performance criteria.

## Stop rule

Stop only after every item is checked, or when an unchecked item requires an
explicit governance/science decision that cannot be inferred from the existing
proposal and evidence.
