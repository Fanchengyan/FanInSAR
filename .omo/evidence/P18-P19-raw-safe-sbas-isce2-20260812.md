# P18/P19 raw SAFE → SBAS and ISCE2 parity evidence — 2026-08-12

## Scope

This is a local qualification-candidate run after removing multilook-dependent
phase-closure from the acceptance gate. It is evidence for the implementation
and is not an official Waymark gate event.

## Raw SAFE → Stack → SBAS

Command driver:

`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/qualified-raw-safe-sbas-20260811/run.py`

The run used the current product checkout (`3c84e3680a7a9c8a6d4b9604586f1a9115a980e3`)
and three local Sentinel-1 SAFE ZIPs, IW1 bursts 0–2, Radar pair mode, CPU Torch,
and multilook `(16, 40)`. The output was written to a fresh directory:

`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/qualified-raw-safe-sbas-20260812/output`

The complete chain passed:

- 3 raw SAFE scenes → 3 coregistered scenes;
- 3 pair IFG artifacts, each shape `(261, 534)`;
- spatial unwrap, temporal reconciliation, and rank-aware SBAS inversion;
- immutable `STACK_CURRENT`, IFG/unwrap generations, and `timeseries.zarr`;
- 52,274 finite time-series pixels;
- phase and displacement arrays were published with manifest and payload hashes;
- total driver time: 296.01 s wall time (284.08 s pipeline time);
- peak RSS: 8,745,418,752 bytes (~8.15 GiB).

The run record is:

`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/qualified-raw-safe-sbas-20260812/runs/raw-safe-qualified-candidate.json`

The stack generation manifest binds all three pair manifests and all three unwrap
manifests:

`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/qualified-raw-safe-sbas-20260812/output/stack_manifest.json`

The activation records in this run are explicitly candidate-only records issued
under an isolated local authority; they do not change P18/P19 governance status.

## Geo replay

The same current chain was replayed from persisted Geo scene artifacts for three
dates and three pairs on a common grid. It passed with 5,279 finite pixels, shape
`(3, 119, 117)`, and peak RSS 382,484,480 bytes. The measured modulo-closure and
SBAS-residual distributions were retained as diagnostics only; they are not
qualification gates after the policy update.

The public raw-SAFE Geo wrapper was also exercised with the current checkout:

- two dates, one burst, cold run: 6,907 finite pixels, `(2, 116, 112)` output,
  48.31 s wall time, peak RSS 7,749,025,792 bytes;
- three dates, three bursts, network mode: the existing cold artifact was reopened
  and revalidated under the current checkout; 12,611 finite pixels, `(3, 232, 121)`
  output, 22.93 s warm-resume wall time, peak RSS 1,438,154,752 bytes;
- the three-date, three-burst Geo pair/IFG set contains all three expected pairs,
  and the current run republished the complete Stack generation and time-series
  bindings without recomputing coregistration.

The raw two-date Geo run record is
`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/current-qualification-20260811/public-geo-pair-n2-b0-auto-grid-cold-summary.json`.
The current-code three-date Geo resume record is
`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/current-qualification-20260811/public-geo-network-n3-b0-1-2-auto-grid-warm-summary.json`.

## ISCE2 comparison

The local ISCE2 reference is the benchmark under
`/Volumes/DATA2/TEST_sentinel-1/campaign/20161207_20161231_iw1_burst0/benchmark_20260718/isce2_cpu`.
The like-for-like geometry comparison uses the same middle-frame SAFE pair and
IW1 burst 0. Because ISCE2's archived product has a different crop and phase/
filter publication path, its wrapped-phase/coherence arrays are not used as a
false byte-equality gate. The independent geometry-offset comparison is:

- range offset absolute difference: `0.038704872 px`;
- azimuth offset absolute difference: `0.015343666 px`;
- acceptance bound: `0.1 px` on each axis;
- result: **PASS**.

Comparison record:

`/Volumes/DATA2/TEST_sentinel-1/current-isce2-parity-20260812/isce2-offset-comparison-20260812.json`

This establishes that the geometry/coregistration result is aligned with the
ISCE2 reference within the stated sub-pixel bound. Product-phase comparison must
remain tied to a future same-crop, same-flattening, same-filter oracle; the large
phase difference observed when comparing unlike publication paths is not a valid
precision verdict.

## Current conclusion

The implemented Radar raw-SAFE → Stack → SBAS flow and the persisted Geo flow are
operational, and the ISCE2 geometry-offset check passes. P18/P19 should remain
`implementing` until the remaining governance packet is closed: an officially
issued gate event, a same-crop ISCE2 product oracle, and the full dual-domain
resource/performance qualification rather than this candidate run alone.
