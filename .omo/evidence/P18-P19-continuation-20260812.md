# P18/P19 continuation checkpoint — 2026-08-12

This checkpoint records the continuous execution pass after the previous
INCONCLUSIVE verification. It does not promote either proposal or issue a
Waymark gate event.

## Newly closed by execution

1. Fresh same-frame Radar: three SAFE dates, three IW1 bursts, three pair
   IFGs, spatial unwrap, temporal SBAS, immutable parent generation, and fresh
   parent/child reopen all passed. See
   `P18-P19-raw-same-frame-radar-sbas-20260812.md`.
2. Fresh same-frame Geo: the same three dates and bursts, EPSG:32647 40 m
   common grid, three pair IFGs, spatial unwrap, temporal SBAS, immutable
   parent generation, and fresh parent/child reopen all passed. Peak RSS was
   8,791,343,104 bytes. See
   `P18-P19-raw-same-frame-geo-sbas-20260812.md`.
3. Geo warm resume passed in 13.745 s with the same pair and manifest
   identities; no coregistration was recomputed.
4. The fresh Geo run measured peak RSS `8,791,343,104` bytes; its warm resume
   measured `1,420,394,496` bytes. These are product observations, not a
   claim of the separate Linux hard-cap packet.
5. Focused P18/P19 regression remains green: 105 passed, 1 skipped. Changed
   processing Ruff checks and `git diff --check` pass.

## Explicitly not treated as failures

- Temporal phase closure is not a hard gate. Multi-looking and independent
  unwrapping can leave non-zero closure; acceptance uses the approved
  same-crop/reference/ISCE2 product oracle and valid-pixel semantics.
- The attempt to write under `/tmp` was rejected because `/tmp` contains a
  symlink component. Re-running under an owner-controlled repository path
  passed; this confirms the intended fail-closed path defense.
- Linux SSH `cryogpu-hk` was unavailable from this environment (`Operation not
  permitted`), so Linux-only measurements were not fabricated.

## Remaining gates

- A complete fresh old-vs-current Radar **and** Geo Stack performance packet
  with identical cold/warm/IFG-only/SBAS/process-tree/Torch/disk methodology.
  The isolated prepared-vs-reference Pair packet already passes (+32.3036%
  median total, exact hashes), but it is not the full dual-domain Stack gate.
- A full multi-date crash/restart, stale-writer/reader-GC, source-mutation,
  and exactly-once operation-lineage campaign. Unit-level transaction and
  hostile-artifact checks pass.
- Independent Waymark verification while the proposals are in `review`, then
  the human-owned typed gate chain. Current proposal status remains
  `implementing` by design.
