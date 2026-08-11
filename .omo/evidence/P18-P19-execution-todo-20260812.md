# P18/P19 continuous execution TODO — 2026-08-12

This checklist is the execution ledger for the current P18/P19 qualification
pass. It is updated continuously; unchecked items are not claimed as complete.

## Governance and contract

- [x] Load FanInSAR project registration and P18/P19 sidecar proposals.
- [x] Confirm closure removal is present in P18/P19 decisions and product tests.
- [x] Enter the P18/P19 Waymark review round and submit all four lanes for both
      proposals. The round is synthesized; verification/activation remains
      gated by the review findings and is not self-issued.

## Functional product paths

- [x] Run current raw SAFE → Radar Stack → IFG → unwrap → SBAS → Zarr.
- [x] Re-run the raw Radar chain on a same-frame three-date corpus with three
      bursts and verify the complete parent generation after process exit.
- [x] Reopen and validate the complete Radar parent generation and child hashes.
- [x] Run current raw SAFE → Geo pair path.
- [x] Run a fresh same-frame three-date/three-burst raw Geo Stack → SBAS path
      and verify its complete parent generation after process exit.
- [x] Reopen current three-date/three-burst Geo network artifacts.
- [x] Re-run the same Geo output in warm-resume mode and record the reduced
      resume cost without recomputing coregistration.
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
- [x] Produce the local executable old-vs-current performance packet.  The
      same-frame Radar Stack comparison is end-to-end (old `97f14c8` versus
      current), and the Geo comparison is stage-matched direct Pair because
      the historical old Stack has no GeoGrid-aware API.  Cold/warm behavior,
      exactness, process-tree, RSS, Torch, and disk evidence are linked from
      `P18-P19-local-performance-packet-20260812.md`.  The strict historical
      old-vs-current *Stack Geo* variant is recorded as not executable rather
      than fabricated.

## Fault and lifecycle qualification

- [x] Run bounded manifest, payload, partial, symlink, mixed-generation, and
      reopen fail-closed checks.
- [x] Run the bounded manifest, payload, partial, symlink, reopen, and
      generation-transaction checks. The bounded transaction campaign passed
      fresh-process reopen, reader pin/GC, CURRENT tamper, crash-before-commit,
      concurrent-writer, immutable-source mutation, and IFG→unwrap binding
      checks. The full real multi-date fault matrix remains open and is not
      claimed from the bounded campaign; see
      `P18-P19-transaction-lineage-campaign-20260812.md`.
- [x] Persist a complete scientific operation trace for residual, carrier,
      and geometric-phase transitions (operation IDs plus input/output payload
      hashes). Radar three-burst and corrected Geo three-date/burst-0 lineage
      runs now emit ordered state/payload-hash chains; see
      `P18-P19-scientific-operation-lineage-20260812.md`. The Geo fix owns
      lineage inputs before memmap cleanup and keeps the normal path zero-copy.
- [x] Reconcile all real-data matrix and holdout evidence into one
      frozen-corpus qualification packet; open scope and corpus mismatches are
      recorded explicitly rather than treated as passes.
- [x] Run a fresh current-code three-date/three-burst Radar+Geo matrix for
      geometry and network modes. All four combinations produced the expected
      three-pair set and complete parent generations; see
      `P18-P19-current-four-mode-matrix-20260812.md`.
- [x] Submit the local final verification packet after the transaction and
      performance evidence.  It is retained as
      `P18-P19-local-final-review-20260812.md`; the prior independent Waymark
      report remains preserved as the governance record and the historical
      old-Geo-Stack scope limitation is explicit.
- [x] Re-run independent review after the current four-mode matrix. The
      reviewer closed `P19-REAL-MODE-MATRIX`; the latest disposition is in
      `P18-P19-final-matrix-delta-gate-review-20260812.md`.
- [x] Submit and synthesize the local P18/P19 Review round. Security recommends
      acceptance; architecture, feasibility, and adversarial lanes recommend
      revision for the remaining strict full-Stack and fault-evidence scope.
- [ ] Complete three interleaved old/current full-Stack repetitions for both
      domains. Radar has only partial repeated evidence, and the historical
      old implementation has no GeoGrid-aware Stack API. Closing this item
      requires either a compatible historical Geo adapter or an explicit
      reviewed criterion revision to the stage-matched Geo Pair comparison.
- [ ] Issue the official typed P18 → P19 → Stack activation gate chain. This
      remains intentionally absent because strict repeated full-Stack
      performance and runtime scientific-operation lineage are not authority
      events that a local test may self-issue.

## Stop rule

Stop only after every item is checked, or when an unchecked item requires an
explicit governance/science decision that cannot be inferred from the existing
proposal and evidence.
