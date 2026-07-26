# Proposal note — FanInSAR ≈ ISCE2 parity (memory + precision)

**Date:** 2026-07-26  
**Purpose:** Condense prior campaign `reports/**`, `.omo/ulw-research`, and `.omo` plans/evidence into a ranked work queue with status, then drive the residual science/runtime items needed for **basic** FanInSAR–ISCE2 agreement on the fixed Sentinel-1 campaign.

**Scope of “basic parity” (this note’s bar):**

| Metric | Target | Notes |
|--------|--------|--------|
| Pre-unwrap complex / wrapped phase on overlapping valid pixels | Order-of-magnitude competitive with dem-fixed ISCE2 (not bulk-ramp theater) | Prefer \|complex coh\| ≫ historical ~0.03; wrap circ_std(Fan−ISCE2) ideally ≲ 1 rad on pin-masked valid set after fair DEM-flat |
| Residual dual_std (geo, pin fixed) | Competitive with dem-fixed ISCE2 (~1.0 rad on 20161207–20161231 IW1 b0) | Fan historically ~2 rad; insardev ~2.8 |
| Peak RSS production path | **≤ 14 GiB** | `MemoryWatchdog` hard kill; single IW burst path previously ~6 GiB after chunked Lanczos |
| Fair oracle | dem-fixed ISCE2 products | **not** old `oracle_isce2_from_slc` (hgt ≈ −500 m) |
| Pin | row=**175**, col=**2276** | `reports/2026-07-16-three-way-unw-geocode-compare/common_ref_point.json` |

Non-goals: MintPy/NISAR full rewrite, bit-identity with topsApp internals, completing every historical arch-greenfield Phase-7 todo, multi-burst InSAR.dev oracle when priors are missing.

---

## 1. Source distillation (what the campaigns proved)

### 1.1 Architecture / feasibility (ULW research)

- **Hybrid radar-first default** for S1 TOPS until geocode-first is proven equivalent; keep typed stage products and provenance.  
  Source: `.omo/ulw-research/20260711-155830/SYNTHESIS.md`
- Python-first means owned algorithms + NumPy/SciPy/PyTorch/GDAL-class deps — not “no C anywhere.” ISCE3/GMTSAR/SNAPHU remain **offline oracles** only for validation when desired.
- Unwrapping is a portfolio (IRLS experimental; SNAPHU production-capable); time series is more than matrix inversion.

### 1.2 Early production green (MVP)

- Burst-window SAFE → coreg → ifg → Goldstein → IRLS → Zarr/STAC path landed under master plan todos 1–25.  
  Source: `.omo/evidence/final-verification/ORCHESTRATION_COMPLETE.md`, `.omo/plans/faninsar-python-insar-master-plan.md`

### 1.3 Memory (chunked Lanczos)

- Full-frame Lanczos temps ~31 GiB killed hosts; **chunked Lanczos + float32 offsets + cos/sin deramp + tiled ESD bilinear pre-align** brought peak under **14 GiB** (~5–9 GiB on real IW1 burst).  
  Sources: `reports/2026-07-13-s1-production-memory/report.md`, `reports/2026-07-14-dask-torch-numpy/`, `reports/2026-07-15-lanczos-dask-array/`

### 1.4 Coreg offset sign (pure-noise ifg)

- Geometry offsets used the wrong sign relative to `resample_complex` (`source = output − offset`). Fix restored mean γ ~0.24 (vs ~0.07 noise) on 20161207–20161231.  
  Source: `reports/2026-07-15-s1-three-slc-ifg-compare/ROOT_CAUSE_AND_FIX.md`

### 1.5 Fan vs ISCE2 / InSAR.dev — not unwrap, not “missing one flatten click”

| Finding | Evidence |
|---------|----------|
| Fan post-geo ≈ Fan dense_geo wrap | Same radar DEM-flat ifg; only geocode path differs (`reports/2026-07-17-session-handoff.md`) |
| Fan vs ISCE2 amp corr high, complex coh ~0.03 | Formation/coreg path, not SNAPHU (`reports/2026-07-17-unify-preunwrap-snaphu/`) |
| Old ISCE2 DEM broken | VRTRawRasterBand read LZW GeoTIFF as raw BSQ → hgt≈−500 m; **dem-fixed oracle** is truth (`reports/2026-07-17-isce2-dem-fixed-rerun/`) |
| Residual dual_std (geo, pin) dem-fixed | Fan post ~1.98, Fan dense ~2.79, **ISCE2 ~1.01**, insardev ~2.78 |
| Radar wrap circ_std(Fan−ISCE2) | ~1.22 rad after dem-fixed re-run |
| A/B on ESD/stride/amp/poly | No material dual_std change (`reports/2026-07-17-dual-modes-plan/root_cause_analysis.md`) |
| Offsets Fan ≈ ISCE2 numerically | mean \|Δrg\| ~0.002 px → ~1 rad phase at 528 rad/px — **resample / carrier coupling** is the precision bottleneck |
| ISCE `Resamp_slc` | Per-pixel local deramp → SINC → local reramp **inside** C++; Fan historically whole-image reramp then Lanczos |

### 1.6 Residual topo / DEM extent (full frame)

- Range-offset flatten during coreg is first-order only; **must still subtract DEM residual** on the ifg (`estimate_residual_topographic_scale` / residual `topo + range_offset_phase`).  
  Sources: `fix_burst_merge.md`, campaign diagnosis under `/Volumes/DATA2/TEST_sentinel-1/arch-greenfield-e2e-fullframe-20260721/`
- IW3 (and some edge bursts) need DEM to **lon ~101.5**; limited DEM → failed rdr2geo → stripes.

### 1.7 Burst merge

- **No reference system uses Hanning/cosine feathering.** ISCE2 uses simple `avg`/`top` after **ESD before merge**. InSAR.dev geo merge uses equal-weight dissolve + robust 3-step `align(degree=1)`.  
  Source: `reports/2026-07-23-burst-merge-strategies/report.md`  
- FanInSAR now exposes a method menu; default **`insardev_ramp`**; `faninsar_weighted` is opt-in non-reference.

### 1.8 Session handoff (fixed corpus)

- Pair: **20161207–20161231**, IW1 burst0 ~1494×20k  
- Shared geo grid: UTM EPSG:32647, ~20×80 m, shape (550, 4477)  
- Pin: **(175, 2276)** — never auto-argmax in residual plots  
- Preferred products: `out/oracle_isce2_dem_fixed/`, Fan `true_unified` / `fan_complete_burst_geo`  
  Source: `reports/2026-07-17-session-handoff.md`

---

## 2. Ranked proposals (work queue)

Status codes: **done** | **partial** | **blocked** | **not-started**.  
Statuses below are the **draft matrix at note authoring**; live audit lives in implementer scratch `proposal_status.md` and §4 updates.

| ID | Proposal | Priority for basic ISCE2 parity | Status (live tree) | Primary evidence / code |
|----|----------|----------------------------------|--------------------|-------------------------|
| P0 | Chunked Lanczos + production MemoryWatchdog ≤14 GiB | **Required** (runtime) | **done** | Peak RSS ≈ 6.2 GiB (`reports/2026-07-13-s1-production-memory/`); watchdog 14 GiB no kill |
| P1 | Coreg geometry offset sign + refine roll convention | **Required** (science) | **done** | `coreg/dense_geometry.py`, `geometry_coreg.py`, ROOT_CAUSE report |
| P2 | Dem-fixed ISCE2 oracle + fair pre-unwrap compare harness | **Required** (measurement) | **done** | `reports/2026-07-17-isce2-dem-fixed-rerun/`; DATA2 `faninsar-radar-parity-20260718` |
| P3 | Pinned common ref for residual panels | Required (fair metrics) | **done** | `common_ref_point.json` row=175 col=2276; `common_ref.py` |
| P4 | Residual DEM topo after range-offset flatten | **Required** (science) | **done** | `stage_flatten` residual path + `estimate_residual_topographic_scale` + unit tests |
| P5 | Carrier-aware secondary restore (deramp → Lanczos → analytical reramp) | **Required** (precision) | **done** | Production `stage_coregister` + integer/fractional phase tests |
| P6 | True per-tap carrier-coupled SINC (ISCE Resamp_slc clone) | Nice-to-have / next precision tier | **not-started** (deferred: P5 meets bar) | Dual-modes §11 mid-term |
| P7 | Merge method menu; default ≠ Hanning | Required for multi-burst seams | **done** | `merge/methods.py` default `insardev_ramp` |
| P8 | Expand DEM to lon ~98–101.5 for IW3 | Required for **full-frame** parity only | **partial** | Campaign DEM partial; eastern IW3 still DEM-limited offline |
| P9 | ESD before multi-burst merge (ISCE2 order) | Secondary for single-burst parity | **partial** | ESD in radar coreg; multi-burst merge not topsApp-order clone |
| P10 | Fair Fan-vs-ISCE2 campaign on frozen pin/mask meeting §0 bar | **Required** (acceptance) | **done** (radar ML 2×10) | complex coh **0.928**, wrap RMSE **0.113** rad, unw residual std **0.303** rad vs ISCE2 |
| P11 | Dual package / Phase-7 arch cleanup | Out of parity bar | **partial** (not blocking) | greenfield F1 dual packages remain |
| P12 | Multi-burst InSAR.dev full-frame oracle | Out of bar if priors missing | **blocked** | Session handoff §4 |

---

## 3. Causal chain still between Fan and ISCE2 (single-burst)

```
TOPS azimuth carrier ~0.17 rad/px
    → external whole-image reramp + finite Lanczos support
    → sub-pixel phase error ~O(0.002 px) × ~528 rad/px ≈ 1 rad
    → lower ifg coh (Fan ~0.4 vs ISCE ~0.67 on comparable ML)
    → larger unwrap residual dual_std (~2 vs ~1)

Mitigation shipped / in flight:
  P5  deramped-domain Lanczos + analytical carrier at fractional source coords
  P4  residual DEM topo after range-offset screen
  P0  memory so production path is runnable under 14 GiB
```

Merge cosmetics (Hanning) do **not** close this gap; dual-modes A/B already ruled out ESD/stride/amp knobs as primary dual_std drivers on this scene.

---

## 4. Implementation queue for this goal (only parity-critical)

1. **Docs:** land this Proposal note (`docs:` commit).  
2. **Tests:** unit-test `estimate_residual_topographic_scale`; strengthen fractional-shift coverage for `resample_complex_deramped_reramp` if needed (`test:`).  
3. **Science path:** confirm production `stage_coregister` uses P5 path + `stage_flatten` residual topo (P4); fix any regression (`fix:`/`feat:`).  
4. **Metrics:** run or re-capture Fan production vs dem-fixed ISCE2 on pin/mask; record peak RSS + wrap/complex metrics; update §2 P10.  
5. **Full-frame only if env allows:** DEM expand (P8); otherwise document blocked eastern bursts and prove parity on DEM-covered subswaths.

---

## 5. Absolute path index (sources)

### Reports (repo)

- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-13-s1-production-memory/report.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-13-complex-resampling-kernel/report.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-14-dask-torch-numpy/report.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-15-s1-three-slc-ifg-compare/ROOT_CAUSE_AND_FIX.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-15-lanczos-dask-array/report.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-16-three-way-unw-geocode-compare/` (pin + residual baseline)
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-17-session-handoff.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-17-dual-modes-plan/root_cause_analysis.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-17-isce2-dem-fixed-rerun/report.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-17-unify-preunwrap-snaphu/report.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-17-true-unified/report.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/reports/2026-07-23-burst-merge-strategies/report.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/fix_burst_merge.md`

### Research / plans / evidence (`.omo`)

- `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/ulw-research/20260711-155830/SYNTHESIS.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/plans/faninsar-python-insar-master-plan.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/plans/faninsar-radar-geo-pipeline-rebuild.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/plans/coreg-resampler-paths.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/final-verification/ORCHESTRATION_COMPLETE.md`
- `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/production-s1/learnings.md`

### Campaigns (DATA2, when mounted)

- `/Volumes/DATA2/TEST_sentinel-1/arch-greenfield-e2e-fullframe-20260722-clean/`
- `/Volumes/DATA2/TEST_sentinel-1/faninsar-radar-parity-20260718/`
- `/Volumes/DATA2/TEST_sentinel-1/sentinel-slc/` (SAFE fixtures)

---

## 6. Status log (updated as commits land)

| Date | Change |
|------|--------|
| 2026-07-26 | Proposal note authored from reports + ulw-research + campaign handoffs. |
| 2026-07-26 | P3 pin restored; P4 residual topo landed + tests; P5 fractional phase gate; P7 merge menu default `insardev_ramp`. |
| 2026-07-26 | P10 bar **met** on DATA2 `faninsar-radar-parity-20260718` radar ML(2,10): complex coh 0.928, wrap RMSE 0.113 rad, unw residual std 0.303 rad; peak RSS 6.2 GiB (P0). P6 deferred. P8 full-frame DEM still partial. |
