# P18/P19 scientific operation lineage — 2026-08-12

## Scope

This packet records the local real-data lineage qualification after fixing a
Geo memmap lifetime fault. It is an implementation/evidence packet, not an
activation event. Temporal loop closure after multilooking is intentionally
not a gate; the accepted comparison policy uses the ISCE2 oracle instead.

## Radar result

The real three-date, three-burst Radar network run is recorded under
`.qualification-runs/p18-p19-lineage-radar-20260812/lineage-summary.json`.
Nine scene units contain the ordered `deramp` and
`apply_carrier_residual_range_phase` transitions. Every record has an
operation ID, input/output payload digest, and chained input/output state
digest. Operation IDs are unique within each unit.

## Geo result

The corrected real three-date Geo network probe used IW1 burst 0 from the
same-frame first-burst corpus and EPSG:32647 (`1857 x 2422`, 40 m). It
completed all three pair arcs and published three scene manifests:

- run root: `.qualification-runs/p18-p19-lineage-geo-b0-r4-20260812/`;
- summary: `lineage-summary.json`;
- scene units: 3;
- records: 6;
- ordered operations: `deramp`,
  `apply_geo_carrier_residual_geometric_phase`;
- payload and state hashes: present for every record;
- unique operation IDs: true within every unit;
- peak RSS: `9,330,999,296` bytes (`/usr/bin/time -l`).

The first three attempts terminated with a SIGSEGV after Geo memmap cleanup.
The root cause was lineage hashing a memmap after `stage_interferogram` had
closed it. The production fix makes an owning copy only when lineage is
enabled, then retains the normal zero-copy path when lineage is disabled.
The corrected run completed without a crash.

## Verification

Focused local regression after the fix:

```text
107 passed, 1 skipped, 6 warnings
```

`git diff --check` is clean. The repository's existing Ruff invocation is not
available in `.venv`; `uvx ruff` reports two pre-existing PLW0108 findings in
the ROI transformer lambdas at `production.py:3272` and `production.py:3277`.

After the review hardening, a fresh local Radar IW1 burst-0 Pair run completed
at `.qualification-runs/p18-p19-phase-radar-b0-r5-20260812/` with
`record_scientific_lineage=True`, CPU Torch, and no unwrap. It produced four
ordered transitions (`deramp`, carrier/residual application, IFG formation,
range-screen flattening), a non-null typed `PhaseState`, and a persisted
operation/state/payload digest chain. The run exited 0 in 22.30 seconds with
maximum RSS `6,295,633,920` bytes. A direct runtime regression also confirmed
that repeating the residual application is rejected by the typed state machine.

The lineage payload has since been changed to bounded 4 MiB C-order chunks;
this keeps the digest stable for contiguous arrays while avoiding one full
temporary copy for strided or memory-mapped inputs. The CUDA Goldstein path
now supplies the required `window=32` argument.

## Remaining qualification boundary

This closes the real Radar/Geo scientific-lineage implementation gap for the
tested scopes, but it does not issue a P18/P19 activation event. The strict
old/current full-Stack three-repetition performance packet remains open (the
historical old Geo Stack has no GeoGrid-aware API), as does the governance
event chain. Those are reported as qualification blockers rather than
silently inferred from this probe.
