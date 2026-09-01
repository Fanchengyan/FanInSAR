# KNOW-0005 — AC-11 real-data ionosphere campaign (Alaska ALOS-2 WB1, 2026-08-28/29)

- **Status**: accepted finding + incident record
- **Scope**: PROPOSAL-0036 AC-11 three-way campaign on host A100; alosStack
  reference lane, ISCE3 split_main_band lane, FanInSAR split-spectrum lane
- **Campaign root**: `/DATA/DATA6/fancy/ion-campaign-20260828/` (A100)

## FanInSAR algorithm weakness located + wrapped-difference fix (2026-08-30)

Root cause of the 157-rad phantom screen: FanInSAR unwraps EACH subband
separately (IRLS) and subtracts. At subband SNR ~0.1 the IRLS solution of
each band is an arbitrary smooth surface; the difference does not cancel the
arbitrary low-frequency parts, and the 78x amplification turns them into the
monotonic 157-rad ramp.

Fix (low-SNR-safe estimator, validated in `wrapped_diff_1607.py`): estimate
the dispersive difference DIRECTLY from the wrapped phases,
`d = wrap(phi_low - phi_high)` (the ~0.03-rad true difference never wraps;
non-dispersive content cancels without any unwrapping), then smooth and
amplify. Result: the phantom drops from 157 rad to 9.1 rad (mostly a
constant offset) — profile correlation with the true 2-rad trend stays
insignificant (-0.16). Even the algorithmically correct estimator cannot
recover the signal at subband SNR 0.1: the fix is necessary but the pair
remains unestimable. Candidate PROPOSAL-0036 amendment: add a low-SNR
wrapped-difference mode (or coherence-gated unwrap rejection) to
`estimate_ionosphere`.

Final figures (compare-1607/): `ion_1607_before_after_comparison.png`
(3 methods x before | screen | after; ISCE3 needs the same post-filter as
FanInSAR — the raw solve output is per-pixel noise, smoothed it matches
FanInSAR's screen) and `ion_1607_final_comparison.png` (three screens on one
0-25 rad scale). Earlier per-method divergence (157 vs 20 vs 6 rad spans)
was entirely the unwrapping path, not the solve. Superseded figures with
retracted claims were deleted; the surviving set is these two figures.

## Coastal roughness: adaptive vs fixed smoothing (2026-08-30)

Observation (user): FanInSAR/ISCE3 screens vary rapidly near the coastline;
alosStack varies uniformly. Measured (local 5x5 std, coastal = land within
10 px of ocean): FanInSAR coast/interior ratio **1.56**, ISCE3 **1.47**,
alosStack **1.15**. Cause: alosStack's adaptive Gaussian grows its window
(11-301 px) where the coherence-derived std is high, keeping output noise
uniform (and preserving detail in coherent areas: interior local std 0.071
vs 0.049 for the fixed sigma=8 kernels, which over-smooth the interior and
under-smooth coasts). FanInSAR/ISCE3 campaign drivers use one fixed sigma=8
pass. Candidate PROPOSAL-0036 amendment: adaptive-window smoothing (or
coherence-scaled sigma) for the ion screen post-filter.

External adaptive smoothing implemented (`adaptive_smooth_1607.py`) — this
step is NOT part of ISCE3 (its solve is only the 2x2; production pipelines
do multilook+plain filtering) and works on any ion screen. TWO lessons:
(a) naive coherence-based window assignment fails (the looks.py proxy is
inflated and mis-ranks coastal pixels); (b) the screen must be DETRENDED
(order-2 polynomial removed) before the roughness-driven window assignment —
a large kernel on a trending field reproduces the gradient, which the
local-std metric counts as roughness (coastal ratio worsened to 8.6x without
detrending, 5.9x at sigma=32 on the raw screen). With detrend + roughness-
driven windows: FanInSAR ratio 1.21, ISCE3 1.35 (alosStack native 1.15).
Figure: `compare-1607/ion_1607_before_after_reference.png` (uses the
adaptive-smoothed FanInSAR/ISCE3 screens).

## FanInSAR payload-split vs alosStack corrected-subband inputs (2026-08-30)

Persistent FanInSAR-vs-ISCE3 mismatch after the snaphu-config fixes: the
FanInSAR screen still carried a strong diagonal ramp (std 42.7 rad) with
circular fringes after correction. Root cause located: FanInSAR split its
subbands from the raw resampled payload SLCs, which still contain the
residual co-registration (range-offset) phase ramp; alosStack/ISCE3 use the
"diff" subband IFGs, i.e. the co-registration-corrected interferogram
(cmd_2 applies `ifg * exp(-j*4*pi*rg_offset*B/lambda)` before splitting).
The 2x2 solve is identical — the discrepancy was entirely the subband INPUT
field. Fix (campaign): compute the FanInSAR screen from the same corrected
alosStack subband .int files with the same bounded snaphu (smooth, nlooks
2560, ocean-masked) — screen std 42.7 -> 13.7 rad, matching ISCE3 (5.2) and
alosStack in structure; red diagonal and circular fringes gone
(`ion_1607_before_after_reference.png`, three rows now consistent).
Product-level follow-up (PROPOSAL-0036): pipe the co-registration phase
correction into FanInSAR's ion estimation (its payload path lacks the
rg-offset ramp removal alosStack applies before splitting).

## FanInSAR switched to mandatory snaphu unwrap (2026-08-30, per review)

Driver default is now `unwrap_method="snaphu"` (CLI-switchable, Stack's
unwrap-backend class; the wrapped-difference no-unwrap variant is withdrawn).
Result on the summer pair: median 322.75 rad, std 511 rad — the per-band
snaphu solutions disagree at multiples of cycles between bands, and the
78x-amplified band difference is a jagged +-6-rad 2-cycle mess
(`ion_1607_before_after_reference.png`, top row). The IRLS run was smoother
(std 42) but equally non-physical; alosStack survives because its weighted
fit+adaptive filter + cor-gating suppresses the same disagreement to a
12-22 rad gradient. Reading: per-band unwrap -> subtract is fragile at
subband SNR ~0.1 regardless of tool; the wrapped-difference (no-unwrap)
estimator remains the most robust estimator form for low-SNR pairs, but per
the product decision FanInSAR's standard path now unwraps with snaphu (and
high-coherence pairs, where the unwrap is meaningful, will benefit).

## Reference-point phase calibration (2026-08-30, per review request)

The three screens carry different arbitrary constants ( FanInSAR +21.05,
ISCE3 +20.03, alosStack +18.65 rad at the reference block — a 2.4-rad spread),
which rotated each method's corrected phase differently and made the wrapped
results incomparable. Fix: pick the coherent 10x10 block whose raw-phase mean
is closest to 0 (rows 264-274, cols 0-10, mean +1.155 rad), subtract each
screen's mean over that block, then correct. All corrected phases now share
one absolute convention; figure:
`compare-1607/ion_1607_before_after_reference.png`. The screen column shows
all three methods at −5..0 rad (south) rising to ~0 (north) — consistent
structures; the corrected phases are directly comparable.

## Unwrap-tool experiment + full reconciliation (2026-08-30)

With ALL fixes in (ocean mask inside the estimation, bounded unwraps, same
post-filter, shared p3-p98 color scale), the four screens converge: pairwise
correlations (constants removed) fan-vs-isce3(snaphu) +0.90, fan-vs-isce3(irls)
+0.97, fan-vs-alos +0.96, isce3(snaphu)-vs-irls +0.95, isce3-vs-alos +0.92/+0.99.
FanInSAR's fixed estimator uses NO unwrap (wrapped-difference); the IRLS and
snaphu unwraps only feed the ISCE3 arm (whose solve requires unwrapped
inputs). FanInSAR integrates snaphu as its default backend; FanInSAR snaphu
with ocean-masked coherence gives bounded unwraps (std 0.93 rad) where
alosStack's own snaphu invocation produced 1e7-rad garbage. Same ISCE3 solve
fed snaphu vs IRLS unwraps: std 12.5 vs 5.2 rad, mutual r=+0.95 — the unwrap
tool matters at the margin, but every variant tracks the same pattern.
CONCLUSION: the 2x2 solves are identical; the FanInSAR-vs-ISCE3 divergence
was entirely the unwrap path + masking order + missing post-filter, all now
fixed. Figure: `compare-1607/ion_1607_unwrap_tool_experiment.png`.

## Water-mask ordering matters (2026-08-30)

The FanInSAR fix and the ISCE3 arm initially smoothed over the FULL grid and
masked the ocean afterwards: coastal land pixels averaged in decorrelated
ocean values -> abrupt value jumps at the coastline (alosStack does not show
this because its chain zeroes the ocean weights BEFORE the fit/filter and
normalises by valid pixels only). Moving the water mask INSIDE the
estimation (ocean coherence zeroed before the IRLS unwrap; land-only
smoothing kernels) removed the coastal jumps and dropped the ISCE3 screen
std from 91.9 to 5.2 rad; pairwise screen correlations improved to
fan-isce3 +0.30, alos-isce3 +0.30, alos-fan +0.96. Rule: mask -> estimate,
never estimate -> mask.

## ISCE3 arm investigation and fix (2026-08-30)

The ISCE3 arm produced +-1e8 rad "screens" because its INPUT was garbage:
alosStack's snaphu subband unwraps (`ion/ion_cal/{lower,upper}_80rlks_448alks.unw`)
contain **min 617, max 5e7, std 4.8e6 rad** (the wrapped .int are normal,
std ~2.1 rad) — snaphu integrated ~1e5 cycles of noise; the ISCE3 solve
(dispersive = 78x(phi_low - phi_high)) amplified that to 1e8. Fix: re-unwrap
the wrapped subband .int with the bounded FanInSAR IRLS unwrapper
(`unwrap_subbands_1607.py`, std ~1.2 rad) and feed those to ISCE3
(`run_isce3_ion.py` prefers the IRLS npys).

With clean inputs ISCE3 gives a near-constant screen (median 17, std 8.6,
span 20 rad) — while FanInSAR's screen on its own subbands is a 157-rad
monotonic ramp, and alosStack's is a 12-22 rad gradient. Three-way screen
correlations: fan-vs-isce3 +0.19, alos-vs-isce3 +0.18, alos-vs-fan +0.91;
the true IFG trend is 2.0 rad and NO screen tracks it. The three chains
differ only in subband generation + unwrapping (the solves are the same
2x2 linear system), so at subband SNR ~0.1 the unwrap path alone decides
the answer by orders of magnitude. This confirms the earlier correction:
nothing physical can be estimated on this target; implementation-vs-algorithm
cannot be adjudicated without subband coherence >= ~0.3.

## CORRECTION (2026-08-30, after review challenge): the summer "cross-validation" was NOT physical

The initial positive reading of the summer pair was wrong. Challenged on the
trend mismatch, a direct azimuth-profile check (verify_trend_1607.py) showed:

- The interferogram's TRUE large-scale trend (azimuth-mean, complex-averaged,
  unwrapped) spans only **2.0 rad (0.3 fringes)**.
- The FanInSAR screen spans **156.9 rad — 77x the true trend**; profile
  correlation with the IFG trend = **-0.25**. The alosStack screen spans
  6.0 rad (3x), profile r = -0.36. **Neither screen tracks the IFG.**
- The screen-to-screen r=+0.91 is explained by both pipelines estimating from
  the SAME subband phase-difference field: the split amplification for this
  geometry is f0/(2*df) = 1236.5/(2*7.93) = **78x**, matching the measured
  77x amplitude ratio. At true subband coherence ~0.03 (subband SNR ~0.1)
  the unwrap integrates the amplified noise into smooth phantom screens.
- Two earlier metrics were artifacts and are retracted: (a) the "0.13 rad
  low-passed residual" — low-passing a WRAPPED phase aliases any sub-cycle
  trend to ~0, so it could not distinguish perfect correction from aliasing;
  (b) the "cross-implementation validation" — agreement between two
  estimators sharing one noisy input is necessary, not evidence of physical
  correctness.

Definitive visual: `compare-1607/ion_1607_method_comparison.png` column 3
(screen wrapped to +-pi) — the FanInSAR wrapped screen is spatially aliased
speckle, not the raw-phase fringes; `ion_1607_screen_evaluation.png` panel 1
(profiles).

Standing conclusion for AC-11: on Alaska North Slope tundra with WB1 splits,
the iono signal is ~2 rad full-band (subband difference ~0.03 rad vs 1.8 rad
speckle, SNR ~0.1) — **split-spectrum estimation is not viable at any tuning;
this is a target/band limitation, not an implementation defect**. Validation
requires subband coherence ~>=0.3 (e.g. S1 IW or a higher-coherence target).

## Fair solve-core comparison + FanInSAR==ISCE3 proof (2026-08-31)

After the user asked "if FanInSAR replicates ISCE3 with the same inputs and
post-processing, why do the comparison figures not match?", the comparison
chain itself was audited. Two artifacts in the comparison scripts made
FanInSAR and ISCE3 look different when they were in fact identical:

1. **Different unwrap inputs.** `faninsar_from_alos_subbands.py` re-ran snaphu
   itself, while the ISCE3 npy was built from `lower/upper_unw_snaphu.npy`
   (the shared `unwrap_subbands_snaphu.py` outputs). Two independent snaphu
   runs differ by an arbitrary absolute-phase constant -> a 4.44 rad screen
   offset (median 17.99 vs 13.62 rad) at r=0.9994.
2. **Simplified coefficient in the fan script.** The old script used
   `AMP*(low-high)` with AMP=1236.5e6/(2*7.93e6)=77.96 (pure difference), while
   ISCE3 uses the exact 2x2 solve `m21*low + m22*high` (m21=78.18, m22=-77.68,
   sum=0.50 — a small common-mode term). `adaptive_smooth_1607.py` also fed
   FanInSAR a wrapped-difference smoothed field instead of the solve output.

Fix: FanInSAR now consumes the SAME `lower/upper_unw_snaphu.npy` files and
runs the exact `solve_2x2_low_high` (faninsar/processing/atmosphere/estimation.py
— identical a,b,c,d,det,m21,m22 and identical output as ISCE3's
`estimate_iono_low_high` in isce3/atmosphere/split_band_estimation.py:894).

Result — FanInSAR vs ISCE3: **r=1.000000, max|diff|=1.1e-5 rad** (float32
round-off). The remaining honest difference is the solve-core variant:

- alosStack (`computeIonosphere`: cor^20 weighting + relative unwrap-error
  adjustment + its own initOnly snaphu): raw r vs pure 2x2 = 0.23; after one
  common alosStack-style post-process (polyfit-2 -> detrend -> roughness-
  adaptive smoothing sigma 4-32 -> re-add fit -> reference calibration) the
  three converge: alosStack vs FanInSAR **r=+0.906**, alosStack vs ISCE3
  **r=+0.906**, FanInSAR vs ISCE3 **r=+1.000** (identical).
- The 0.854 vs 0.906 asymmetry seen before the fix was the same 4.44 rad
  offset + independent per-method adaptive thresholds; it is gone.

Key lesson: **"same algorithm" comparisons must share the unwrap inputs AND
the exact coefficient formula; an independent snaphu re-run silently breaks
pixel-level agreement at the 78x amplification factor.**

Deliverables (compare-1607/): `ion_1607_fair_solve_comparison.png` (3x3
before/screen/after, per-panel HistColorbar, ocean-masked, reference block
calibrated) and `ion_1607_before_after_reference.png` — FanInSAR and ISCE3
rows are now pixel-identical by construction.

## alosStack-style weighted+adjusted solve core (PROPOSAL-0042, 2026-08-31)

Per owner request, FanInSAR gained a second, optional solve core that
reproduces alosStack's `computeIonosphere` (adjFlag=1) exactly:
`solve_guided_split` in
`faninsar/processing/atmosphere/estimation.py`, selected by
`IonosphereEstimationConfig.solve_core: "isce3" | "alosstack"` (default
`"isce3"`), with `cor_order_adj=20` (the alosStack coherence-power
weight). ISCE3-style pure 2x2 is retained. The alosStack time-series
inversion already exists as `invert_ionosphere_network` (ion_ls.py
semantics, PROPOSAL-0036).

**Validation against alosStack raw solve on 160702-160730**
(inputs: alosStack's OWN `lower/upper_80rlks_448alks.unw` — read as
`reshape(279*2,71)[1::2,:]` — and `diff_80rlks_448alks.cor`; reference
`ion_80rlks_448alks.ion`):

- First implementation: r=+0.9841, slope=0.9838 (1.6% scaling + wrong
  cycles on low-coherence land pixels).
- Root cause: `solve_guided_split` gated BOTH the weighted surface fit
  AND the per-pixel integer-cycle adjustment with the coherence weight mask
  (`wgt>0`), but alosStack applies the cycle adjustment to ALL pixels with
  `lowerUnw != 0` (`flag2`), independent of coherence. Restricting the
  adjustment left wrong 2π cycles on low-coherence pixels.
- Fix: separate masks — `fit_valid = isfinite(diff) & (wgt>0)` gates the
  weighted polyfit only; `adj_mask = isfinite(low) & (low != 0.0)` gates the
  cycle adjustment, matching alosStack `flag2`.
- After fix: raw solve r=+1.000000, slope=1.000000, median|diff|=5.7e-6 rad;
  post-processed screens r=+1.000000, median|diff|=3.8e-6 rad — bit-level
  agreement. Regression test
  `test_zero_coherence_pixels_still_get_cycle_adjustment` added; 49
  atmosphere tests pass, lint clean.

The alosStack frequencies are `fl = f0 - B/3`, `fu = f0 + B/3` from
`c/λ` and `rangeBandwidth` (0.2424525 m / 11.9 MHz for this campaign) —
identical to the ISCE3 summary values already used.

Deliverables: `compare-1607/ion_1607_alosstack_solve_validation.png` (3
panels: alosStack reference, FanInSAR solve, difference ≈ 0),
`alosstack_solve_validation.json` (raw), and the FanInSAR raw solve npy.

## Summer-pair rerun (2026-08-30, campaign `logs-1607`/`compare-1607`)

> **Note:** the "positive cross-validation" claims in this section are
> retracted — see CORRECTION above. The screen-to-screen r=+0.91 reflects a
> shared noisy input, and the 0.13/1.24 rad residuals were wrap-aliasing
> artifacts.

Reran the chain on **160702-160730** (28 d, mid-July, peak summer) per the
season hypothesis: summer tundra (no snow, no freeze-thaw) should be far more
coherent than winter.

Results:

- **TRUE land coherence 0.109** (identical estimator): across all three pairs
  the full-band ion-grid coherence is invariant at 0.109-0.114. Coherence on
  this target is a fixed property of Alaska North Slope tundra in WB1,
  independent of season and baseline. The season hypothesis is rejected at
  the coherence level.
- **Yet the summer campaign delivered the first positive cross-validation.**
  FanInSAR published 12,954 finite pixels (65%) with a physically scaled
  screen (std 42 rad, NE +150 -> SW -100 rad large-scale TEC gradient).
  After aligning the sign, alosStack vs FanInSAR: **r = +0.91, median |diff|
  29.5 rad**, inland blocks r up to +0.94. The two independent
  implementations measure the same ionospheric pattern.
- **Sign convention finding**: the FanInSAR dispersive/ion screen carries the
  OPPOSITE sign of the alosStack ion definition (raw r = -0.906). Product
  consumers applying cross-implementation corrections must pin this
  convention (PROPOSAL-0036 follow-up).
- The alosStack screen is the same pattern attenuated ~10-25x (std 1.5-5 rad
  vs 42) by its proxy-cor-weighted fit+filter pipeline; the ISCE3 arm stays
  ungated (~1e8 rad scale) and remains unusable without a mask.

Verdict: AC-11 chain validated end-to-end on real data with a genuine
cross-implementation agreement at ~30 rad median difference (sign-aligned),
despite subband coherence ~0.03 - provided coherence gating + water mask +
smoothing are active (the FanInSAR defaults tuned in this campaign).

## Winter-pair rerun (2026-08-29, campaign `logs-1501`/`compare-1501`)

Reran the identical chain on **150117-150228** (42 d, Jan-Feb, both deep
winter, frozen tundra + stable snow) from the same 43-scene frame-2200 WB1
archive, to test the "shorter winter baseline restores coherence" hypothesis.

Result: **hypothesis rejected**. TRUE land coherence (identical numpy
estimator, ion rows 93-124): **0.113 vs 0.114** for the 84-day freeze-up pair.
Decorrelation on this target is NOT baseline- or season-driven — it is a
fixed property of Alaska North Slope tundra in WB1 (SNR/volume-scattering
limited). FanInSAR low-passed screens (sigma 2/4/8) still do not correlate
with the alosStack smooth-fit surface (global r -0.09..-0.14, blocks
-0.68..+0.72); the alosStack screen collapses to an ~11-rad-std polynomial
surface; the ISCE3 arm remains ungated (~1e7 rad scale).

Drivers are now date-parameterized (`--ref-date/--sec-date`, `--pair`); the
chain is one command (`aux/run_1501_all.sh` + `aux/arms_1501.sh`). Two more
campaign-setup nits: the arm workdir must contain the driver copy and the
compare outdir must contain `compare_ion_screens.py`.

Conclusion: AC-11 quality judgement needs a higher-coherence target or band
(e.g. S1 IW), not a different ALOS-2 WB1 date pair over this tundra scene.

## Outcome

The three-way chain is mechanically verified end to end on real data:
alosStack `est_slc_offset → resample → rdr2geo/geo2rdr → cmd_2 → cmd_3 →
ion_ls/ion_correct`, FanInSAR out-of-product converter → public store APIs →
`estimate_ionosphere` → published artifact (7,388 finite ion-grid pixels),
ISCE3 pixi lane, and the water-masked per-block three-way comparison
(`compare/ion_screens.png`, `compare/compare_report.json`, grid 279x71).

The scientific screens are noise-dominated and the pair is below the practical
limit for split-spectrum ionosphere estimation:

- 20140913-20141206 (94 d, Alaska North Slope tundra, freeze-up).
- TRUE ion-grid coherence measured from the raw resampled SLCs: ~0.12
  full-band median on land, ~0.03 per subband (7.93 MHz split, WB1 s3).
- FanInSAR vs ISCE3: pearson r = -0.006, median |diff| 3.7e5 rad (ISCE3
  screen ungated, ~1e6 rad scale).
- FanInSAR vs alosStack: global r = -0.28 (blocks -0.47..+0.69), median
  |diff| 417 rad — both screens noise/floor-dominated.
- alosStack screen: smooth 2-D-fit surface, 0..-1750 rad, band-stratified.

Recommendation: re-run AC-11 on a shorter-baseline (e.g. 24-46 d) ALOS-2 WB1
pair or an S1 14+ MHz pair before using the screens for threshold judgement.

## Incident: wrong-region aux data (root cause of the full redo)

The first campaign pass used Patagonia (S38_S35 W072_W068) DEM/SWBD for the
Alaska scene: rdr2geo diverged everywhere (hgt = -500 fill), the
`estimate_slc_offset` land gate emptied ("land too small for estimating slc
offsets at frame 2200, swath 3"), and every downstream offset/resample/geometry
product was contaminated. Fixed by rebuilding aux from primary sources:

- DEM: 24 AWS GLO-30 tiles (copernicus-dem-30m mirror) mosaicked to
  `demLat_N68_N72_Lon_W154_W147.dem.wgs84` (1-arcsec, ocean tiles absent from
  the mirror zero-filled). Tile lon spacing is 1/2/3 arcsec by latitude;
  exact integer-second lattice upsample, no interpolation at samples.
- Water mask: SWBD does not exist >60 deg N; built from GSHHG `gshhs_f.b`
  (44-byte big-endian header, level = flag & 0xF, polygons west of -180 stored
  in 0..360 and normalised before bbox prefilter), painted 0=land/255=water.
  Land fraction 62.1%, Beaufort Sea strip 0.0% land, Brooks Range strip 99.9%.
- ISCE raw products need `.vrt` companions even when fabricated from a custom
  xml; hand-written `VRTRawRasterBand` VRTs satisfy the DataAccessor.

`geom_check` after the redo: hgt diverged 0.0000%, lat/lon exactly on the
North Slope bounds, radar-grid water 34.2%.

## alosStack tool pathologies found and fixed on A100 (files backed up `.bak-20260828`)

All in the installed `isce2` env (not the FanInSAR product tree); each is a
performance/robustness guard, not a semantics change unless noted:

1. `imageMath.py`/`looks.py` missing from PATH under plain
   `conda activate isce2` — generated cmd scripts exit 0 even when their
   commands fail (no `set -e`); orchestrators need explicit product gates.
2. `ion_filt.py` win2 table built `gaussian(size, size/2.0)` — a size x size
   2-D matrix per iteration up to 10001 x 10001 (~800 MB) x 45k iterations,
   ~2.6 TB of first-touch churn = the multi-hour hang in every pass.
   Replaced with the separable analytic form (identical to 1.6e-15).
3. `polyfit_2d` called per pixel inside `adaptive_gaussian`: replaced the
   (n x 6) dgelsd call with cached-H normal equations (identical to 6e-8;
   per-call allocations caused a second fault storm).
4. `numberOfLooks` for the subband noise model is 0 in standalone cmd-mode
   runs (`azimuthBandwidth` absent from XML-loaded tracks) -> all weights
   vanish -> all-zero screens. Added a total-looks fallback.
5. alosStack ion-grid `.cor` (from `looks.py` on an already-looked int) is a
   homogeneity proxy, NOT true coherence — inflated (median 0.31 vs true
   0.12). Its hard-coded `corThresholdAdj = 0.97` zeroed every screen; the
   campaign relaxed it to 0.20 (documented deviation; as-is output archived
   as `aux/filt_ion_gated097.ion` — all zeros).
6. `ion_check.py` needs `mdx` (not installed); non-fatal, cmd scripts exit 0.

## FanInSAR campaign-driver fixes (drivers live under the campaign root)

- Pair directories are `140913-141206` (hyphen) and ion products live under
  `ion/ion_cal/`; several drivers assumed underscores + `f1_2200/s3`.
- Secondary-date SLCs must be read from `dates_resampled/<date>/...` (the
  driver reused the reference date's directory).
- `Stack` session uses full date ids (`20140913`); short-id keys silently
  failed the qualified-scene check and made both scenes secondary-role.
- `estimate_ionosphere` caches on `ion_manifest.json` unless
  `overwrite=True`; three threshold changes silently returned the first
  cached screen. Also `write_ionosphere_artifact` did not receive the
  overwrite intent (plumbed `replace_existing=overwrite` in the campaign
  checkout of faninsar-src — upstream gap, not a workspace-tree change).
- `valid_mask` must be passed already downsampled to the ion grid; the driver
  now block-majority-reduces the ml1 water mask (and actually passes it).
- Default `coherence_threshold=0.5` assumes S1-quality pairs; for this pair
  0.02 keeps the coherent land majority. Recorded in the artifact
  method_parameters.

## Comparison infra

`compare_ion_screens.py` (campaign root `compare/`): GSHHG land mask sampled
on the ion grid, per-block (3x3) pearson/median-diff plus global, screens PNG.
AlosStack reference screen for the comparison is the relaxed-gate run.
