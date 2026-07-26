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
| P6 | True per-tap / neighborhood carrier-coupled SINC + Doppler (ISCE `Resamp_slc`-class) | **Next precision tier** (skill gap S2–S4,S8) | **not-started** | Dual-modes root cause; skill §5.2; optional after basic bar |
| P6b | Fix / annotate `sar-resampling-kernels` skill (SK-1…SK-5) | Docs / agent hygiene | **not-started** | Note §5; skill over-claims Lanczos≡ISCE and under-specifies TOPS coupling |
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
  P6  (open) true neighborhood carrier+Doppler inside sinc — ISCE Resamp_slc class
```

Merge cosmetics (Hanning) do **not** close this gap; dual-modes A/B already ruled out ESD/stride/amp knobs as primary dual_std drivers on this scene.

**Skill note:** Choosing Lanczos a=4 for complex is **aligned** with open-source sinc-family practice; the remaining gap is **how** TOPS carrier/Doppler enter the resampler (§5), not “should we use bilinear instead.”

---

## 4. Implementation queue for this goal (only parity-critical)

1. **Docs:** land this Proposal note (`docs:` commit).  
2. **Tests:** unit-test `estimate_residual_topographic_scale`; strengthen fractional-shift coverage for `resample_complex_deramped_reramp` if needed (`test:`).  
3. **Science path:** confirm production `stage_coregister` uses P5 path + `stage_flatten` residual topo (P4); fix any regression (`fix:`/`feat:`).  
4. **Metrics:** run or re-capture Fan production vs dem-fixed ISCE2 on pin/mask; record peak RSS + wrap/complex metrics; update §2 P10.  
5. **Full-frame only if env allows:** DEM expand (P8); otherwise document blocked eastern bursts and prove parity on DEM-covered subswaths.

---

## 5. Resampling / interpolation skill audit (vs open-source)

**Skill under review:** global agent skill `sar-resampling-kernels`
(`~/.agents/skills/sar-resampling-kernels/SKILL.md` + `references/kernel_evidence.md`).

**Verdict:** The skill’s **core rule is correct** — full-bandwidth complex SLC/ifg must use a **sinc-family** kernel; bilinear on raw complex is a production bug; multilook is boxcar complex average (SNAP `MultilookOp`); never average wrapped phase alone. That matches ISCE2, NISAR GSLC practice, SNAP multilook docs, GMTSAR “multilook then geocode,” and Hanssen & Bamler (1999).

**However, several skill claims are incomplete or misleading relative to production open-source TOPS paths.** Agents that treat the skill as a complete ISCE2 clone will under-specify secondary SLC resampling. Problems below should be treated as **known skill gaps**, not as “FanInSAR is already ISCE-identical because it uses Lanczos a=4.”

### 5.1 What the open-source stacks actually do

| System | Complex SLC / fine resamp | Geocode of complex | Multilook | Notes |
|--------|---------------------------|--------------------|-----------|-------|
| **ISCE2** `Resamp_slc` | **SINC** (default if complex); **local** deramp (range/az carrier poly) + **Doppler poly** + SINC + local reramp **inside** C resampler | `Geocodable`: `cpx`/`amp` → **sinc**; `cor`/`unw` → **nearest** | Radar looks then products | topsApp order: fine resamp **before** burst ifg / merge |
| **ISCE2** geocode | — | sinc for complex; **nearest** for coh/unw by default | — | Nearest on unw/cor is pipeline caution, not “bilinear is wrong” |
| **ISCE3 / NISAR GSLC** | Windowed sinc (raised-cosine ~8-tap) for phase-preserving geo | Complex path is sinc-family | Product-defined | Not a S1 TOPS burst-merge reference |
| **GMTSAR** | Radar-domain processing; geocode often after looks | GMT `grdresample` often **bilinear** on **already multilooked** grids | Complex boxcar in radar | Bilinear OK only after bandwidth ≪ Nyquist |
| **SNAP** | Coreg/resample tools use sinc-class options for complex | Varies by operator | **MultilookOp**: average I/Q, **never** phase band alone | Multilook ≠ interpolation |
| **InSAR.dev Core** | Geo-domain stack; `align` + `dissolve` on geocoded bursts | Equal-weight circular mean in overlap (no Hanning) | Product-dependent | Closest analog to Fan **geo merge**, not to ISCE2 radar fine resamp |

Sources: local ISCE2 `Resamp_slc.py` / `Geocodable.py`; campaign reports `2026-07-13-complex-resampling-kernel`, `2026-07-17-dual-modes-plan/root_cause_analysis.md`, `2026-07-23-burst-merge-strategies`.

### 5.2 Skill problems (ranked)

| # | Skill claim / omission | Open-source reality | Severity for Fan≈ISCE2 |
|---|------------------------|---------------------|------------------------|
| **S1** | Treats **Lanczos a=4** as interchangeable with “ISCE2/ISCE3/NISAR production standard” | ISCE2 keys **SINC** (truncated sinc), not the Lanczos window formula; NISAR cites **raised-cosine windowed sinc**. Same **family**, not bit-identical kernel | Medium — OK default, must not claim identity |
| **S2** | Emphasizes kernel **family** (sinc vs bilinear) but under-specifies **TOPS carrier coupling** | ISCE2 `Resamp_slc` does **per-output-pixel neighborhood deramp → SINC → reramp** + Doppler poly **inside** the resampler. Dual-modes A/B showed offsets already match; residual dual_std gap tracked **resample/carrier coupling**, not kernel choice alone | **High** — primary remaining precision path (P6) |
| **S3** | Pitfall A2: “prefer analytic carrier at fractional coords after deramped remap” | That is an improvement over interpolating a carrier **plane**, but still **not** ISCE’s local neighborhood deramp-inside-kernel. Whole-image deramp → Lanczos → analytic reramp is a **proxy**, not a clone | **High** |
| **S4** | Recommends **FFT fractional shift** for “ISCE2 dense_offsets → affine fit → resample” | Global FFT shift is exact only for a **constant** (dx, dy). TOPS fine offsets are **spatially varying**; ISCE applies poly/dense offsets inside `Resamp_slc`, not one global FFT | **High** if followed literally for S1 |
| **S5** | Calls **M1 (geocode-SLC → ifg)** the “gold standard” | **topsApp S1 production is radar-first** (coreg/resamp SLC → ifg → merge → optional geocode). Geocode-first is a product branch (e.g. GSLC), not the ISCE2 S1 default; campaigns showed bad geocode_first residuals when mis-specified | Medium — skill over-promotes M1 for S1 parity |
| **S6** | Recommends **bilinear** for unwrapped phase / coherence | ISCE2 geocode defaults **nearest** for `unw`/`cor` (avoid side-lobe ringing into QC). Bilinear is defensible for smooth residual maps; it is a **deliberate divergence**, not “what ISCE does” | Low–medium |
| **S7** | DEM → B-spline 4–5 (GAMMA) | ISCE2 DEM sampling in geometry is not the same as Fan’s geocode of products; fine for DEM *as smooth field*, but do not confuse with Resamp_slc | Low |
| **S8** | Skill is silent on **Doppler poly** at fine resamp | ISCE2 sets Doppler poly on `Resamp_slc`; Fan’s Python Lanczos path has no equivalent | Medium for high-PRF / residual az |
| **S9** | Skill correct that no reference merge uses Hanning | Confirmed ISCE2 `avg`/`top`, InSAR.dev equal-weight dissolve (`reports/2026-07-23-burst-merge-strategies`) | Skill OK here |

### 5.3 FanInSAR code vs skill vs ISCE2 (current)

| Path | Fan (HEAD) | Skill says | ISCE2-like? |
|------|------------|------------|-------------|
| Radar secondary fine resamp | `resample_complex_deramped_reramp` (deramp whole → Lanczos → analytic carrier) | Complex → Lanczos; analytic carrier preferred | **Partial** — sinc family yes; **not** neighborhood-coupled Resamp_slc |
| ESD pre-align | bilinear (`order=1`) allowed | bilinear only if already band-limited | Acceptable as **coarse** pre-align only |
| Geo complex geocode | `lanczos_resample` in `geocode.py` / `geocode_raster.py` | Lanczos a=4 | Good enough vs Geocodable sinc |
| Geo real (coh) | bilinear | bilinear | OK (ISCE often nearest) |
| Hard labels | nearest | nearest | Yes |
| Some **campaign scripts** under `reports/2026-07-23-*` | still `Resampling.bilinear` for dump plots | should be Lanczos for complex | **Script debt** — do not treat plot dumps as production kernel |

### 5.4 Implications for this Proposal note

1. **Do not “fix” the skill by switching away from Lanczos to bilinear** — that would regress against every open-source complex path.
2. **Do treat skill S2–S4 as open algorithm work** (aligns with P6: true per-tap / neighborhood carrier-coupled SINC + Doppler, or optional ISCE `Resamp_slc` backend). Until then, document residual Fan–ISCE phase gap as **resample coupling**, not “wrong kernel family.”
3. **Prefer radar-first (ISCE2-like) for S1 parity campaigns**; use geocode-first only with full carrier-aware SLC geocode semantics (ULW synthesis), not generic Lanczos warp of focused SLC.
4. **Update skill later** (out of minimal parity code path): (a) state Lanczos ≈ windowed-sinc family, not ISCE kernel identity; (b) add mandatory TOPS subsection “carrier+Doppler inside resampler”; (c) restrict FFT shift to constant offset only; (d) demote M1 gold-standard wording for S1 topsApp parity.

### 5.5 Proposed skill-fix backlog (docs-only until code lands)

| ID | Action |
|----|--------|
| SK-1 | Add TOPS / `Resamp_slc` subsection: neighborhood deramp → sinc → reramp + Doppler poly; cite dual-modes root cause |
| SK-2 | Narrow FFT-shift “use when” to constant or affine-global shifts; forbid as drop-in for dense TOPS offsets |
| SK-3 | Replace “M1 gold standard” with “M1 valid for GSLC-class products; S1 topsApp parity = radar-first M2-like” |
| SK-4 | Document ISCE2 nearest-for-unw/cor as optional QC-preserving mode vs bilinear residual maps |
| SK-5 | Note separable Lanczos a=4 is an engineering choice; production oracles may use non-separable truncated sinc / raised-cosine |

---

## 6. Absolute path index (sources)

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

## 7. Status log (updated as commits land)

| Date | Change |
|------|--------|
| 2026-07-26 | Proposal note authored from reports + ulw-research + campaign handoffs. |
| 2026-07-26 | P3 pin restored; P4 residual topo landed + tests; P5 fractional phase gate; P7 merge menu default `insardev_ramp`. |
| 2026-07-26 | P10 bar **met** on DATA2 `faninsar-radar-parity-20260718` radar ML(2,10): complex coh 0.928, wrap RMSE 0.113 rad, unw residual std 0.303 rad; peak RSS 6.2 GiB (P0). P6 deferred. P8 full-frame DEM still partial. |
| 2026-07-26 | Fixed `stage_flatten` UnboundLocalError (`estimate_residual_azimuth_ramp` module-level import). **HEAD re-run** `run_production_pair` ML(2,10) → `/Volumes/DATA2/TEST_sentinel-1/head-parity-20260726/`: residual DEM topo applied (span 6.78 rad); complex coh **0.800**, wrap RMSE **0.567** rad, unw residual std **0.676** rad, peak RSS **3.96 GiB**, wall ~78 s. Production flatten tests green. |
| 2026-07-26 | §5: audited `sar-resampling-kernels` skill vs ISCE2/SNAP/GMTSAR/InSAR.dev — core complex→sinc rule OK; **gaps** on TOPS carrier-coupled Resamp_slc, FFT-shift misuse, M1-vs-radar-first wording (SK-1…SK-5). |
