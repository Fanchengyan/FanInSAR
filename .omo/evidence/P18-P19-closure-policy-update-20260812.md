# P18/P19 closure-policy update — 2026-08-12

## Decision

Temporal loop closure and SBAS residual magnitude are not qualification gates
for multilooked InSAR products. They may be recorded for research diagnostics,
but a non-zero value is not a failure condition. This supersedes earlier gate
reports that listed a physical closure/residual threshold as an open blocker.

Qualification uses the frozen ISCE2 behavior oracle with the same source roles,
grid, DEM policy, multilook factors and filter settings. The manifest-defined
comparisons cover phase, coherence, range/azimuth offsets, geometry,
geolocation and displacement. Structural checks remain mandatory: complete
pair set, matching grid and lineage, finite/rank-valid published pixels, and
non-corrupt artifacts.

## Implemented product changes

- Removed closure and SBAS-residual threshold fields from
  `StackQualityCriteria`.
- Removed closure/residual failure branches from Stack quality evaluation.
- Kept diagnostic distributions in reports without using them for pass/fail.
- Removed the closure gate from the frozen reference manifest and made the
  closure metric family diagnostic-only in the reference reporter.
- Normalized legacy persisted quality criteria on resume so removed closure
  fields do not invalidate otherwise compatible artifacts.

Commits: `1baafd1`, `88356a9`. Proposal records: sidecar commit `5613da7`.

## Verification

- Unwrap, Stack and reference suites: 120 passed, 1 skipped.
- Additional Stack/session/reference regression: 67 passed.
- Ruff, format and diff checks passed.

## Remaining qualification work

This change removes one invalid scientific gate; it does not by itself mark
P18/P19 verified. The remaining evidence must be regenerated against the
ISCE2 oracle on the frozen real corpus, with matching grid/look/filter settings,
and must cover the required Radar/geo modes, ROI/no-ROI cases, holdout and
full-stack resource packet. Official Waymark qualification events remain
unissued until that comparison packet and the independent transaction/runtime
verification are complete.
