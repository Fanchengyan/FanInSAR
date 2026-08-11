# P18/P19 current real-data matrix

Date: 2026-08-11

This packet records the executable same-corpus matrix at product HEAD
`2be103330d51891c234d1ac186ea037a70adc7d8`. It is evidence for the
implementation status only; it does not issue a Waymark qualification event.

## Frozen inputs

The three first-frame IW1 Sentinel-1 SAFE ZIPs were used for every three-date
case. Their SHA-256 digests are recorded by the campaign driver under
`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/current-qualification-20260811/`.

## Three-date real runs

| Domain | Mode | Bursts | Status | Elapsed | Finite time-series pixels | Shape |
|---|---|---:|---|---:|---:|---|
| Radar | geometry | IW1/b0 | PASS | 50.9036 s | 20,560 | `[3, 93, 534]` |
| Radar | network | IW1/b0 | PASS | 104.2600 s | 20,663 | `[3, 93, 534]` |
| Geo | geometry | IW1/b0 | PASS | 53.2463 s | 3,771 | `[3, 116, 112]` |
| Geo | network | IW1/b0 | PASS | 110.7295 s | 3,900 | `[3, 116, 112]` |
| Geo | geometry | IW1/b0–b2 | PASS | 125.6937 s | 12,600 | `[3, 232, 121]` |
| Geo | network | IW1/b0–b2 | PASS | 275.7728 s | 12,611 | `[3, 232, 121]` |

Every case produced three pair artifacts, three persisted unwrapped pair
artifacts, a bound SBAS artifact, and a Stack parent generation. A fresh
interpreter reopened all six `STACK_CURRENT` roots and verified the parent
manifest, pair ordering, and timeseries generation binding.

## Existing two-date ROI matrix

The current wrapper also has a same-corpus two-date matrix covering Radar and
Geo with and without ROI. All four cases reopened successfully after the
transaction and Stack-generation changes. The latest packet is
`P18-P19-real-roi-matrix-20260811.md`.

## Performance evidence

The isolated three-burst Radar prepared-vs-reference packet remains the
authoritative performance subset: median total improvement `32.303628%`,
product-stage improvement `41.707580%`, byte-identical outputs, prepared RSS
increase `371,015,680` bytes (within the `+512 MiB` relative gate), peak tree
RSS `6,518,194,176` bytes, and external sampling maximum gap `85.709 ms`.

This is a Pair/coregistration differential, not a repeated old-vs-new full
Stack packet. Full Stack cold/warm three-repetition resource comparison and a
Linux descendant-covering hard 14-GiB cgroup gate remain open.

## Full-suite result

The full repository run completed with `1173 passed, 24 skipped, 33 failed,
4 errors`. The errors are remote ASF tests blocked by unavailable network/
Earthdata authentication. The failures are outside the focused P18/P19
surface (colormap/geobox/xarray/capability/sampler compatibility) plus the
provider/resource telemetry tests whose macOS `psutil` process enumeration is
denied by the execution environment. The P18/P19 focused suite remains
`79 passed, 1 warning`.

## Status

P18 and P19 remain `implementing`. Remaining non-automatable or incomplete
gates are: a full Stack old/new resource packet, Linux hard-memory enforcement,
an approved physical temporal-closure/SBAS residual oracle, and a complete
real crash/reopen/lease-GC campaign. No precision change was made.
