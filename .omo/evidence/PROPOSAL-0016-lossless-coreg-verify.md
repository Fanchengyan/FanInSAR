# Verified: PROPOSAL-0016 lossless geo coreg speedup + memory fix

## Scope

- Lossless geo coregistration speedup (bbox LUT + footprint polygon prefilter +
  topo crop + raw-DEM height parity + multi-burst ProcessPool).
- Memory fix: previous full-grid LUT builds left 4 python workers at 12-24 GB RSS
  each; bbox-shaped memmaps keep the parallel run under 10 GB total.
- No stride interpolation (rejected earlier: rg p99 0.6-6 px lossy).

## Changed files

- `faninsar/processing/pipeline/geo_lut.py`
- `faninsar/processing/pipeline/production.py`
- (geo_modes.py delta reverted to HEAD: bbox LUT makes row_range unnecessary)

## Gates

- Ruff check on changed files: pass.
- `pytest tests/processing/pipeline/`: 50 passed, 6 skipped.
- Manual QA (18-burst geo campaign, 5x20 m grid, jobs=4):
  - exit 0, 18/18 bursts archived; run_pair_total = 508 s (baseline 158.3 min, 18.8x).
  - peak total RSS 7.99 GiB (4 workers + main; max single process 3.99 GiB),
    watchdog limit 10 GiB, never triggered.
- Bitwise comparison vs `fan_archive_prev5x20` full-grid baseline:
  - merged_ifg / coherence / wrapped / invalid (3301x6722): identical (NaN-aware).
  - per-burst IFG 18/18 identical; height 18/18 identical.
  - pri/sec finite values identical; only sentinel difference at 2243 pixels
    where the IFG is NaN in BOTH runs (baseline archives 0.0, new archives NaN).
    Confirmed 100% of differing pixels have NaN IFG in both runs -> no phase impact.
- Per-burst stage totals (wall-clock sums): geo2rdr_lut 5282 s -> 642 s;
  topo 1656 s -> 222 s; coreg total 8078 s -> 1232 s.

## Commits

- code: cc38a3ac19acb8f67c09c28c4ec7000c9b436275 (evidence binds to this SHA)

- code+evidence: 504ac8f (amended)
