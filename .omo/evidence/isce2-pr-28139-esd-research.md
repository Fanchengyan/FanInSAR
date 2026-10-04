# ISCE2 PR 28139 "topsStack: fix ESD source coordinates and phase response": primary-source evidence

**Scope.** Evidence gathered on 2026-09-30 from the live GitHub PR
[isce-framework/isce2#28139](https://github.com/isce-framework/isce2/pull/28139)
(state OPEN, not merged), the pre-PR file trees on `main`
(`main` HEAD `b7702cfbd571681a133ebc5c3b23a10c78d52706`; last commit touching
`overlap_withDEM.py` is `e03042f4e5899019da413d6e7e79c4627b07482c`), the PR head
commit `d4dfb51fc9e1953e56140c14451a519011898301`, and the ISCE2 zerodop/TOPS
sources those scripts depend on. All claims below are cited inline.

## TL;DR

- The PR modifies two shipping files (`contrib/stack/topsStack/overlap_withDEM.py`,
  `contrib/stack/topsStack/Stack.py`) and adds two new test files
  (`contrib/stack/topsStack/tests/test_esd.py`, `tests/conftest.py`); totals
  +174/-24 across 4 files
  ([PR view](https://github.com/isce-framework/isce2/pull/28139),
  [diff](https://github.com/isce-framework/isce2/pull/28139/files)).
- It is a **bug fix + fidelity upgrade inside the existing ESD method**, not a new
  method: the pair estimator `estimateAzimuthMisreg.py` is untouched; what changes
  is (a) the coordinate frame in which the per-date azimuth phase response is
  evaluated, (b) the phase-response model itself (TOPS steering carrier, reference
  time, Doppler centroid), (c) use of the mean of the two dates' responses, and
  (d) source-burst indexing via the stack reference.
- The allegedly faulty coordinate logic (`y` pre-offset by the overlap crop origin,
  `yy = 0.0` fallback, the old `frequencySeparation` combination) exists **verbatim
  since the file entered the repo on 2019-01-16**
  ([commit 9f9a2b0](https://github.com/isce-framework/isce2/commit/9f9a2b0805beffd80a6539ca247fdaa8898fd8a9));
  only two later commits touch the file, both cosmetic/mechanical (2020-04-13,
  2020-07-02). So if it is a bug, it is ~7.7 years old.
- Independent verification against the geo2rdr kernel supports the author's core
  claim: `geo2rdr.f90` writes `azoff = (tline - tstart)/dtaz - (line-1)`, i.e. the
  offset image stores *absolute radar line minus DEM row*, so adding the local row
  recovers the absolute full-burst line and adding the crop origin `topStart` again
  double-counts it ([geo2rdr.f90 L374-375](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/components/zerodop/geo2rdr/src/geo2rdr.f90#L374-L375)).
- Reception is **skeptical but not hostile**: one comment from maintainer
  piyushrpt (Piyush Agram) questioning whether the bug is real, arguing the
  bookkeeping "would be consistent - else this would have created issues across
  more dates and more stacks", and noting that in the one branch he checked the
  `y + yy` sum is unchanged ([comment](https://github.com/isce-framework/isce2/pull/28139#issuecomment-5883244977)).
  That observation is correct only for the missing-offsets fallback branch; for the
  offsets-present branch the sum does change (see Section 2/6). No reviews, no CI
  checks reported.
- Risks: single-stack validation, scalar mean-of-two-dates approximation, breaking
  runconfig change (new required `--stack_reference_dir`), ESD + azimuth inversion
  must be rerun for existing stacks.

## 1. What the PR changes, file by file

Diff obtained with `gh pr diff 28139 --repo isce-framework/isce2`; head commit
`d4dfb51fc9e1953e56140c14451a519011898301`
([PR](https://github.com/isce-framework/isce2/pull/28139)).

### `contrib/stack/topsStack/overlap_withDEM.py` (+53/-24)

This script builds the ESD ("Enhanced Spectral Diversity") inputs: overlap
interferograms, their coherence, and a per-pixel "frequency response" file that the
estimator divides the overlap phase by
([estimator usage](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/estimateAzimuthMisreg.py#L61-L106)).

- New required CLI argument `--stack_reference_dir` (L24), "Full-burst stack
  reference used by overlap resampling"
  ([head L21-25](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L21-L25)).
- `overlapSpectralSeparation(...)` gains four keyword parameters:
  `referenceShift=0.0, secondaryShift=0.0, referenceIsStack=False,
  secondaryIsStack=False` (L59), and its docstring changes from "Estimate
  separation in frequency due to unit pixel misregistration" to "Estimate the mean
  phase response (rad/source pixel)".
- Coordinate fix: the row grid for all four burst blocks (reference top/bottom,
  secondary top/bottom) changes from crop-relative-to-full-burst indices to local
  overlap indices. Before
  ([old L85](https://github.com/isce-framework/isce2/blob/e03042f4e5899019da413d6e7e79c4627b07482c/contrib/stack/topsStack/overlap_withDEM.py#L85),
  identical at
  [2019 L85](https://github.com/isce-framework/isce2/blob/9f9a2b0805beffd80a6539ca247fdaa8898fd8a9/contrib/stack/topsStack/overlap_withDEM.py#L85)):

  ```python
  y = np.arange(topStart, topStart+overlapLen)[:,None] * np.ones((overlapLen, topBurstIfg.numberOfSamples))
  ```

  After ([head L87](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L87)):

  ```python
  y = np.arange(overlapLen)[:,None] * np.ones((overlapLen, topBurstIfg.numberOfSamples))
  ```

- Fallback when the geo2rdr `.off` files are absent: the old code silently used
  `yy = 0.0; xx = 0.0`
  ([old L93-95](https://github.com/isce-framework/isce2/blob/e03042f4e5899019da413d6e7e79c4627b07482c/contrib/stack/topsStack/overlap_withDEM.py#L93-L95));
  the new code raises `ValueError('Missing overlap offsets for a resampled date')`
  unless the date *is* the stack reference (`referenceIsStack` /
  `secondaryIsStack`), in which case the crop origin is computed analytically,
  e.g. `yy = (topBurstIfg.sensingStart - referenceTop.sensingStart).total_seconds()
  / referenceTop.azimuthTimeInterval` and the matching range expression
  ([head L94-99](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L94-L99)).
  This is the "unresampled stack reference still needs its explicit crop origin"
  case from the PR description.
- Phase-response model: after `tm1 = (azi - (numberOfLines//2)) * dt`, the new code
  applies a burst-start alignment shift and a Doppler-centroid reference-time
  correction ([head L112-114](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L112-L114)):

  ```python
  tm1 = (azi - (referenceTop.numberOfLines//2) + referenceShift) * referenceTop.azimuthTimeInterval
  fm1 = referenceTop.doppler(rng)
  tm1 -= referenceTop.doppler(referenceTop.startingRange) / referenceTop.azimuthFMRate(referenceTop.startingRange) - fm1 / Ka
  ```

  (and the analogous `tm2`, `ts1`, `ts2` blocks). The `eta_ref` term is exactly the
  `eta_ref` computed in ISCE2's own carrier estimator
  `computeAzimuthCarrier`
  ([TOPSSwathSLCProduct.py L165-206](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/components/isceobj/Sensor/TOPS/TOPSSwathSLCProduct.py#L165-L206)).
- Response combination. Before
  ([old L205](https://github.com/isce-framework/isce2/blob/e03042f4e5899019da413d6e7e79c4627b07482c/contrib/stack/topsStack/overlap_withDEM.py#L205)):

  ```python
  frequencySeparation =  -Ktm2*tm2 + Ktm1*tm1  + Kts1*ts1 - Kts2*ts2 +  fm2 - fm1 + fs1 -fs2
  ```

  After ([head L218-224](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L218-L224)):

  ```python
  # For nearly equal date responses, DD phase / mean response gives the relative shift.
  fm1 = 2 * np.pi * referenceTop.azimuthTimeInterval * (Ktm1*tm1 + fm1)
  fm2 = 2 * np.pi * referenceBot.azimuthTimeInterval * (Ktm2*tm2 + fm2)
  fs1 = 2 * np.pi * secondaryTop.azimuthTimeInterval * (Kts1*ts1 + fs1)
  fs2 = 2 * np.pi * secondaryBot.azimuthTimeInterval * (Kts2*ts2 + fs2)
  frequencySeparation = (fm1 - fm2 + fs1 - fs2) / 2
  ```

  i.e. each date's top-minus-bottom response is formed as a proper rad/sample
  carrier derivative `2*pi*dt*(Kt*t + fdc)`, and the two dates are **averaged**
  instead of the old raw Hz-term sum. Correspondingly the writer drops the
  `* 2 * np.pi * azimuthTimeInterval` conversion because the value is already in
  rad/source sample ([head L410](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L410)
  vs [old L382](https://github.com/isce-framework/isce2/blob/e03042f4e5899019da413d6e7e79c4627b07482c/contrib/stack/topsStack/overlap_withDEM.py#L382)).
- Burst indexing. Old `main()` indexed the full source bursts as
  `mind = ii - minReference`, `sind = ii - minSecondary`
  ([old L308-311](https://github.com/isce-framework/isce2/blob/e03042f4e5899019da413d6e7e79c4627b07482c/contrib/stack/topsStack/overlap_withDEM.py#L308-L311));
  new code loads the stack reference swath, derives common-burst limits and
  relative burst-start shifts, and indexes as `mind = ii - 1 + mOffset`,
  `sind = ii - 1 + sOffset` "Index into full source reference/secondary"
  ([head L294-297](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L294-L297),
  [head L337-338](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L337-L338)),
  using the existing helpers
  [`getCommonBurstLimits`](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/components/isceobj/Sensor/TOPS/TOPSSwathSLCProduct.py#L261-L272)
  and
  [`getRelativeShifts`](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/s1a_isce_utils.py#L50-L66).
  The shifts are passed as `referenceShift=-mShifts[mind]`,
  `secondaryShift=-sShifts[sind]`
  ([head L406-408](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L406-L408)),
  mirroring how `resamp_withCarrier.py` passes `-1.0 * offset` into the carrier
  polynomials when resampling
  ([resamp_withCarrier.py L177-L212](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/resamp_withCarrier.py#L177-L212)).
- `main()` detects whether each ESD-pair date is the stack reference by comparing
  real paths: `referenceIsStack = os.path.realpath(inps.reference) ==
  os.path.realpath(inps.stack_reference_dir)` (and likewise for the secondary)
  ([head L276-277](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L276-L277)).

### `contrib/stack/topsStack/Stack.py` (+1/-0)

One line added to the `overlap_withDEM` runconfig section, writing the stack
reference directory into the generated config so `overlap_withDEM.py` can find the
full-burst reference products
([head L134-140](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/Stack.py#L134-L140)):

```python
self.f.write('stack_reference_dir : ' + os.path.join(self.work_dir, 'reference') + '\n')
```

### `contrib/stack/topsStack/tests/test_esd.py` (+116, new)

A regression test
([head file](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/tests/test_esd.py)):

- `test_original_four_blocks_match_carrier_phase_derivatives` builds four synthetic
  bursts (SimpleNamespace metadata with plausible S-1-like values: dt=0.002 s,
  steering rate 0.028 rad/s, lambda 0.056 m, per-burst Doppler offsets), parametrized
  over `stack_reference in [None, 0, 1]` (both dates, only reference, only secondary
  being the stack reference), and asserts that `overlapSpectralSeparation` output
  equals the mean of numerical derivatives of the analytic carrier phase
  `pi*rate*((row-752+shift)*dt - eta_ref)**2 + 2*pi*dop*row*dt` — i.e. the code is
  pinned to d(phase)/d(sample) of the TOPS carrier model.
- `test_generated_config_reaches_existing_parsers` checks the new
  `stack_reference_dir` key round-trips through `Stack.config(...).overlap_withDEM`
  into `overlap.cmdLineParse`, alongside `estimateAzimuthMisreg` parsing.

### `contrib/stack/topsStack/tests/conftest.py` (+4, new)

Inserts the `topsStack` directory into `sys.path` so the flat scripts (`Stack`,
`estimateAzimuthMisreg`, `overlap_withDEM`, `SentinelWrapper`) can be imported by
the tests ([head file](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/tests/conftest.py)).

## 2. The OLD behavior and the alleged double-count

Pre-PR file: [overlap_withDEM.py at `e03042f4e5`](https://github.com/isce-framework/isce2/blob/e03042f4e5899019da413d6e7e79c4627b07482c/contrib/stack/topsStack/overlap_withDEM.py)
(byte-identical ESD logic to the 2019 original,
[9f9a2b08](https://github.com/isce-framework/isce2/blob/9f9a2b0805beffd80a6539ca247fdaa8898fd8a9/contrib/stack/topsStack/overlap_withDEM.py),
modulo the 2020 master/slave->reference/secondary rename).

What the old code did, per overlap burst pair `ii/ii+1`:

```python
topStart = int(np.round((referenceBot.sensingStart - referenceTop.sensingStart).total_seconds()/dt)) + referenceBot.firstValidLine
botStart = referenceBot.firstValidLine
...
y = np.arange(topStart, topStart+overlapLen)[:,None] * ...   # rows pre-offset by crop origin
...
if os.path.exists(azMasTop) and os.path.exists(rgMasTop):
      yy = np.memmap(azMasTop, ...)   # geo2rdr azimuth offsets for the reference date
else:
      yy = 0.0; xx = 0.0              # silent fallback
azi = y + yy
```

([old L72-99](https://github.com/isce-framework/isce2/blob/e03042f4e5899019da413d6e7e79c4627b07482c/contrib/stack/topsStack/overlap_withDEM.py#L72-L99).)

The PR's claim is that the geo2rdr offset images *already* encode the crop origin,
so adding `topStart` again evaluates the carrier at an azimuth time shifted by the
crop offset. Verification against the kernel that writes those files:

- `geo2rdr.py` runs the kernel per **full secondary burst** (`burstTop =
  secondary.bursts[sBurst]`), configuring `grdr.setSensingStart(info.sensingStart)`
  with that burst's own start, against the reference overlap geometry
  (`geom_referenceDir/overlap/IW*/lat_II_II+2.rdr`)
  ([geo2rdr.py L51-100, L257-292](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/geo2rdr.py#L51-L100)).
- The kernel writes, per DEM/geometry row `line`:
  `azoff(pixel) = ((tline - tstart)/dtaz) - 1.0d0*(line-1)`
  ([geo2rdr.f90 L374-375](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/components/zerodop/geo2rdr/src/geo2rdr.f90#L374-L375)),
  where `(tline - tstart)/dtaz` is the **absolute line number in that date's full
  burst**. So `azoff = absolute_full_burst_line - local_overlap_row`, and
  `local_row + azoff = absolute_full_burst_line`.
- Consistent with this, `resamp_withCarrier.py` uses the same `.off` images as
  residual index images into the full secondary burst SLC (`rObj.imageIn` is the
  secondary burst image, `rObj.residualAzimuthImage` is the `.off`) to produce
  coregistered overlaps ([resamp_withCarrier.py L67-100, L177-212](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/resamp_withCarrier.py#L67-L100)).

Consequences for the old code:

- Offsets-present branch: `azi_old = (topStart + local_row) + (absolute_line -
  local_row) = topStart + absolute_line` — the crop origin is counted twice; the
  correct full-burst azimuth coordinate is `absolute_line`. The PR's `y = arange(overlapLen)`
  restores that. This is the "double-counting" the PR description states:
  "Geo2rdr offsets already map local overlap pixels into the full source burst.
  Adding the crop origin again evaluates the response at the wrong azimuth
  coordinate" ([PR body](https://github.com/isce-framework/isce2/pull/28139)).
- Missing-offsets branch (the stack-reference date, for which geo2rdr is never run
  against itself — the stack workflow generates overlap geo2rdr configs only for
  secondaries, see `config_overlap_geo2rdr_<secondary>` in
  [Stack.py L805-819](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/Stack.py#L805-L819)):
  the old `yy = 0.0` gave `azi = topStart + local_row`, which was accidentally
  correct. The maintainer's "you just move it to `yy` but their sum remains
  unchanged as you have reintroduced `topStart`" observation
  ([comment](https://github.com/isce-framework/isce2/pull/28139#issuecomment-5883244977))
  is exactly right for this branch (old sum `topStart + r + 0` = new sum
  `r + crop_origin`), but it does not apply to the offsets-present branch, where
  the new sum is smaller by `topStart`.

Around the faulty lines, the old code also used these coordinates in the carrier
terms `tm1 = (azi - (numberOfLines//2)) * dt`, `Ka = azimuthFMRate(rng)`,
`Kt = Ks/(1 - Ks/Ka)` with `Ks = 2*Vs*steeringRate/lambda`
([old L101-109](https://github.com/isce-framework/isce2/blob/e03042f4e5899019da413d6e7e79c4627b07482c/contrib/stack/topsStack/overlap_withDEM.py#L101-L109)),
and combined them into the raw-Hz expression quoted in Section 1, then scaled by
`2*pi*dt` on write
([old L378-382](https://github.com/isce-framework/isce2/blob/e03042f4e5899019da413d6e7e79c4627b07482c/contrib/stack/topsStack/overlap_withDEM.py#L378-L382)).

## 3. Is the faulty code long-standing? Yes — since 2019-01-16

`gh api "repos/isce-framework/isce2/commits?path=contrib/stack/topsStack/overlap_withDEM.py&per_page=30"`
returns exactly three commits
([API](https://api.github.com/repos/isce-framework/isce2/commits?path=contrib/stack/topsStack/overlap_withDEM.py&per_page=30)):

| Date | Commit | Author | Message |
|---|---|---|---|
| 2019-01-16T19:40:08Z | `9f9a2b0805beffd80a6539ca247fdaa8898fd8a9` | Eric Gurrola | "Adding all files" (topsStack imported into isce2; `overlapSpectralSeparation` is annotated "# Added by Heresh Fattahi") |
| 2020-04-13T19:40:32Z | `23f2ffaa1eeac93bb039f7259817810c4c3c0b12` | Ryan Burns | "Use exist_ok argument to makedirs when appropriate (#112)" |
| 2020-07-02T19:40:49Z | `e03042f4e5899019da413d6e7e79c4627b07482c` | Paul A Rosen | "diversity: replaced oppressive language with neutral definitions..." (master/slave -> reference/secondary rename) |

The three load-bearing old lines are present verbatim at the same line numbers in
the 2019 blob (`y = np.arange(topStart, ...)`, `yy = 0.0`,
`frequencySeparation = -Ktm2*tm2 + Ktm1*tm1 + Kts1*ts1 - Kts2*ts2 + fm2 - fm1 + fs1 -fs2`;
diff of the 2019 blob vs the 2020 blob shows only the rename and an `makedirs`
change). So the coordinate logic the PR corrects has been in the repo unchanged
since topsStack was contributed on 2019-01-16 — ~7.7 years. This cuts both ways:
it means the defect was never caught upstream (which the maintainer leans on), and
it means any behavioral change now propagates to every reprocessed stack.

## 4. Bug fix or new method? Bug fix + fidelity upgrade within the existing ESD method

The ESD method is unchanged: for each adjacent-burst overlap, form the
top-minus-bottom double-difference interferogram, and estimate azimuth
misregistration by dividing its phase slope by the per-pixel "frequency response"
of the TOPS carrier. The consumer
`estimateAzimuthMisreg.py` (`off = np.angle(ifg) / freq`) is not touched by the PR
([estimator L61-L106](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/estimateAzimuthMisreg.py#L61-L106)),
and the PR body states "The existing pair estimator, network inversion and output
formats remain unchanged" ([PR body](https://github.com/isce-framework/isce2/pull/28139)).

What changed is the *content* of the response file and the coordinates at which it
is evaluated:

1. **Coordinate mapping** (Section 2): evaluate each date's carrier at the pixel's
   true line in that date's full burst, instead of crop-origin + full-burst line.
2. **Phase-response model**: old code summed raw Hz terms without a reference-time
   correction; new code evaluates `2*pi*dt*(Kt*(eta - eta_ref) + fdc)`, where
   `Kt = Ks/(1 - Ks/Ka)` is the TOPS steering-carrier FM rate, `eta_ref =
   fdc(startingRange)/Ka(startingRange) - fdc(rng)/Ka(rng)` is the Doppler-centroid
   reference time, and the burst-start shift from `getRelativeShifts` aligns the
   time axis with how the burst was resampled. This is the same model as ISCE2's
   own `computeAzimuthCarrier`, which forms the resampling carrier as
   `carr = np.pi * Kt * ((eta - eta_ref)**2)`
   ([TOPSSwathSLCProduct.py L165-206](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/components/isceobj/Sensor/TOPS/TOPSSwathSLCProduct.py#L165-L206)),
   plus the Doppler term that `computeAzimuthCarrier` handles separately via
   `dpoly` (`coeffs = [2*np.pi*val*azimuthTimeInterval ...]`).
3. **Mean of two dates**: `frequencySeparation = (fm1 - fm2 + fs1 - fs2)/2` —
   per the code comment, "For nearly equal date responses, DD phase / mean response
   gives the relative shift", and per the PR body, "This scalar approximation
   assumes similar responses for the two dates"
   ([head L218-224](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py#L218-L224),
   [PR body](https://github.com/isce-framework/isce2/pull/28139)).

**What the "phase response" means physically.** After focusing, a TOPS burst
retains an azimuth carrier from antenna steering: with steering FM rate `Ks = 2*Vs*
steeringRate/lambda` and processed FM rate `Ka`, the residual quadratic phase is
`phi(eta) = pi*Kt*(eta - eta_ref)^2` plus a Doppler-centroid ramp `2*pi*fdc*eta`
([computeAzimuthCarrier](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/components/isceobj/Sensor/TOPS/TOPSSwathSLCProduct.py#L165-L206);
the PR's test uses the same two-term phase). A small azimuth misregistration
`delta_eta` therefore contributes phase `~ dphi/deta * delta_eta`; the instantaneous
"response" frequency `dphi/deta = 2*pi*(Kt*(eta - eta_ref) + fdc)` sweeps through
the burst (S-1 scale: `Ks ~= 7000 Hz/s` with the test's `2*7000*0.028/0.056`
parameters, `dt ~= 2 ms`), and its top-vs-bottom difference across the burst
overlap is what converts a measured overlap phase ramp into a sub-pixel azimuth
shift. The `2*pi*dt` factors throughout the old and new code are just the
unit conversion between Hz and rad/sample at the PRF; with real Sentinel-1
`dt ~= 2.06 ms` (test value 0.002 s), the per-sample conversion is the
`2*pi/2.3ms`-type factor noted in the research question.

## 5. Review status and reception

- **State**: OPEN, not merged (`mergedAt: null`), created 2026-09-28T06:16:36Z,
  single commit `d4dfb51fc9e1953e56140c14451a519011898301` by WU-Pengzhan
  (Wu Pengzhan), last updated 2026-09-29
  ([PR view](https://github.com/isce-framework/isce2/pull/28139)).
- **Reviews**: none submitted — `gh api repos/isce-framework/isce2/pulls/28139/reviews`
  returns `[]`.
- **Review comments** (line comments): none — `pulls/28139/comments` returns `[]`.
- **CI checks**: `gh pr checks 28139 --repo isce-framework/isce2` reports
  "no checks reported on the 'fix/topsstack-esd' branch" — no CI evidence either way.
- **Issue comments**: exactly one, from **piyushrpt** (Piyush Agram, ISCE
  maintainer, CONTRIBUTOR association) on 2026-09-29
  ([permalink](https://github.com/isce-framework/isce2/pull/28139#issuecomment-5883244977)).
  Points, verbatim where load-bearing:
  - Asks for "a summary diagram or description" and whether the changes "affect
    only those 2 dates in your stack".
  - Skeptical the bug is real: "The geo2rdr offset bookkeeping would be consistent
    - else this would have created issues across more dates and more stacks. We
    have not encountered this issue and we haven't heard of such issue yet from
    other users even though this data is more than a year old."
  - Suggests a workflow-misuse explanation: "Is it possible, you have been working
    with a stack in a modified state that is not consistent with the workflow."
  - Challenges the coordinate change with the `y + yy` argument: "what ever change
    you make to `y` - you just move it to `yy` but their sum remains unchanged as
    you have reintroduced `topStart`" — correct only for the missing-offsets
    fallback branch (see Section 2), and asks for verification "across more stacks
    by others".
  - No other maintainer has weighed in; no requested changes filed formally.

## 6. Reasonableness assessment

**Physically sound core.** The central coordinate claim independently checks out
against primary sources: the geo2rdr kernel writes `azoff = absolute_line - DEM_row`
([geo2rdr.f90 L374-375](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/components/zerodop/geo2rdr/src/geo2rdr.f90#L374-L375)),
and the consumer adds the local row back to recover absolute lines
([resamp_withCarrier.py](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/resamp_withCarrier.py#L67-L100)).
Under that semantics, the old `y = arange(topStart, ...)` + `yy` does over-offset
azimuth by the crop origin in the offsets-present branch. The new response formula
`2*pi*dt*(Kt*t + fdc)` with the `eta_ref` correction is the sample-domain
derivative of ISCE2's own TOPS carrier `pi*Kt*(eta - eta_ref)^2` plus Doppler ramp
([computeAzimuthCarrier](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/components/isceobj/Sensor/TOPS/TOPSSwathSLCProduct.py#L165-L206)),
and the regression test pins the implementation to numerical derivatives of that
analytic phase across the stack-reference/none/secondary parametrizations
([test_esd.py](https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/tests/test_esd.py)).
The `-1.0 * offset` burst-shift handling mirrors the existing resampling code path
([resamp_withCarrier.py L195, L212](https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/resamp_withCarrier.py#L195-L212)),
so the ESD response is now evaluated consistently with how the overlaps were
actually resampled.

**Risks and open questions.**

- **The maintainer's empirical counter-argument is unresolved.** If the crop-origin
  double-count were large in practice, seven years of topsStack processing should
  have shown systematic ESD/burst-boundary artifacts widely; piyushrpt reports none
  and suspects an inconsistent (modified) stack state
  ([comment](https://github.com/isce-framework/isce2/pull/28139#issuecomment-5883244977)).
  The PR author's single-stack evidence (31 acquisitions, 87 ESD pairs, 219
  overlaps; before/after burst-boundary phase-jump images; reprocessed pairs
  `20250116_20250209` and `20250317_20250329`) is suggestive but is one sensor
  track and not yet an independent replication
  ([PR body](https://github.com/isce-framework/isce2/pull/28139)).
- **Scalar mean-of-two-dates approximation**: the new `(fm1 - fm2 + fs1 - fs2)/2`
  assumes the two dates' responses are similar (author acknowledges this in the PR
  body). The old code instead used a date-*difference* combination; both cannot be
  exactly consistent with `off = angle(ifg)/freq` inversion under large
  date-dependent Doppler differences, so the change alters estimator calibration
  even where coordinates were fine.
- **The fallback branch changes from silent to error**: dates that previously ran
  with missing `.off` files and `yy=0` will now raise
  `ValueError('Missing overlap offsets for a resampled date')` unless recognized as
  the stack reference by `os.path.realpath` comparison — stricter, but a behavior
  change for unusual directory layouts.
- **Breaking config change**: `--stack_reference_dir` is a new *required* argument
  and `Stack.py` writes a new config key; existing runconfigs must be regenerated,
  and the PR body instructs users to "regenerate the run/config files and rerun all
  ESD pair observations and azimuth inversion before fine coregistration"
  ([PR body](https://github.com/isce-framework/isce2/pull/28139)).
- **Burst indexing change** (`mind = ii - 1 + mOffset` vs `ii - minReference`) is a
  second, independent behavioral change bundled with the phase fix; it alters which
  source bursts feed the response when overlaps do not start at the swath's first
  burst. The maintainer's remark that the diff "looks like getting bookkeeping from
  other scripts like extracting overlaps and common burst limits" reads this as
  duplication of existing logic rather than a fix
  ([comment](https://github.com/isce-framework/isce2/pull/28139#issuecomment-5883244977)).
- **Process risk**: no CI runs, no formal review, unmerged 2 days after opening;
  the PR needs cross-stack validation and a clear mathematical write-up (the
  maintainer's explicit ask) before it is likely to land. FanInSAR-relevant
  takeaway: if adopting topsStack ESD semantics, note that the two candidate
  response definitions (crop-origin-adjusted coordinates and mean-of-dates
  response) are still contested upstream.

## 7. Does FanInSAR share this problem? (local analysis, this repo)

Checked on branch `faninsar-stac` at commit `7b98f31`, 2026-09-30. File-level
references: `faninsar/processing/coregistration/{esd.py,geometry_coreg.py,offsets.py,geometric_phase.py}`,
`faninsar/processing/coregistration/tops/{deramp.py,carrier.py}`, `faninsar/missions/s1/processing.py`.

**Conclusion: the specific PR-28139 bug cannot exist in the current native
FanInSAR stack (branch `faninsar-stac`), because it never replicated the buggy
code path. Historical caveat: the pre-2026-06 FanInSAR `faninsar/isce2/`
wrapper DID drive ISCE2 topsStack with overlap ESD, so stacks processed with
that wrapper inherited the upstream bug (PR status pending, see §5).** Details:

1. **Different ESD method.** topsStack's ESD is overlap-based
   (`overlap_withDEM.py` → `estimateAzimuthMisreg.py` → ESD network inversion):
   it *predicts* a per-pixel azimuth phase response from geo2rdr misalignment
   and evaluates it at overlap pixels mapped back into each date's source
   burst — exactly the coordinate bookkeeping that was wrong since 2019.
   FanInSAR's `estimate_azimuth_shift_esd` (`esd.py:94`) is a *dual-look
   split-spectrum* estimator (Prats-Iraola-style, per its docstring): it
   bandpass-filters the azimuth spectrum into lower/upper looks, forms
   per-look interferograms, and converts the circular-mean differential phase
   via `Δφ = 2π·Δf·Δaz` (`esd.py:200-209`). There is no overlap→source-burst
   coordinate mapping and no modeled phase response to mis-place; the buggy
   code path simply does not exist.

2. **ESD runs in the deramped domain.** `stage_deramp`
   (`missions/s1/processing.py:1508`) deramps *full native bursts* with each
   date's own annotation carrier (`carrier_from_swath`); the ESD window is a
   crop of that deramped array (`processing.py:2056-2084`), so window-local
   coordinates cannot bias the differential phase measurement.

3. **The analogous "evaluate the carrier at true source coordinates"
   discipline is already implemented.** `resample_complex_deramped_reramp`
   (`offsets.py:2488`) reramps the secondary carrier *analytically* at source
   fractional coordinates `output_index - offset`, adding the window origin
   exactly once (`carrier_row = src_row + row0`, `offsets.py:2591`) with a
   native-height-aware centre row; `missions/s1/processing.py:2433-2443` passes
   `row0/col0/native_height` explicitly. Its docstring even documents the
   failure mode of the naive variant (interpolating an integer-grid carrier
   plane with `map_coordinates(order=1)` "collapsed to 65 rad").

4. **One dormant hazard worth noting (not wired into the pipeline).** The
   legacy helper `restore_original_domain_secondary`
   (`coregistration/tops/deramp.py:311`) reramps by interpolating an
   integer-grid carrier plane with `map_coordinates(order=1)` — the exact
   approach the newer `offsets.py` docstring warns collapses the ~0.17 rad/px
   azimuth carrier — and it is crop-naive (`row0/native_height` unsupported).
   `grep` shows it is exported from `coregistration/tops/__init__.py` but never
   called by `missions/s1/processing.py`. Consider deprecating or removing it
   before anyone wires it in.

5. **Fidelity caveat vs the PR's upgraded response model.** FanInSAR's
   dual-look ESD is self-contained (no carrier-response model needed), so it
   does not inherit the PR's contested modeling choices (crop-origin-adjusted
   coordinates, mean-of-two-dates scalar response) — which upstream is still
   debating. Its trade-off is different: a single scalar azimuth shift per
   burst window (or one forced common residual across bursts, topsApp
   `secondaryAzimuthCorrection` style, `processing.py:2009-2013`), relying on
   the two dates sharing near-identical Doppler geometry after deramping —
   acceptable for same-orbit S1 coregistration, but not a per-overlap
   measurement like topsStack's.

6. **History: overlap ESD existed via the removed ISCE2 wrapper.** Until
   commit `987a933` (2026-06-14, "refactor: remove isce2 and isce3 processing
   modules"), FanInSAR shipped `faninsar/isce2/` — a wrapper that drove ISCE2
   topsStack as an external tool with `coreg_method="NESD"` (default),
   `esd_coherence_threshold=0.85`, and NESD overlap connections
   (`git show 987a933^:faninsar/isce2/workflows/slc_stack.py`,
   `tops_workflow.py`). Stacks processed through that wrapper used ISCE2's
   overlap-ESD path — including the 2019-era coordinate logic this PR
   contests. The native stack that replaced it (introduced in `e17ffc7`,
   2026-07-15) implements only the dual-look split-spectrum estimator.
   Residual artifact of the transition: `faninsar/stack/config.py:32` still
   declares `EsdMethod = Literal["auto", "splitband", "overlap"]` and writes
   it into runconfigs, but no code dispatches on `"splitband"`/`"overlap"` —
   the only estimator call is `estimate_azimuth_shift_esd`, and
   `stack/session.py` `measure_misreg` gates ESD with the tautology
   `esd_on = method != "auto" or True` (always True). The config vocabulary
   is a placeholder for a never-built method dispatch.

## Sources

- PR page and description: https://github.com/isce-framework/isce2/pull/28139
- PR files/diff: https://github.com/isce-framework/isce2/pull/28139/files (fetched via `gh pr diff 28139 --repo isce-framework/isce2`)
- Maintainer comment: https://github.com/isce-framework/isce2/pull/28139#issuecomment-5883244977
- Reviews API (empty): https://api.github.com/repos/isce-framework/isce2/pulls/28139/reviews
- Issue comments API: https://api.github.com/repos/isce-framework/isce2/issues/28139/comments
- PR head files (sha `d4dfb51fc9e1953e56140c14451a519011898301`):
  - https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py
  - https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/Stack.py
  - https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/tests/test_esd.py
  - https://github.com/isce-framework/isce2/blob/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/tests/conftest.py
  - raw: https://raw.githubusercontent.com/isce-framework/isce2/d4dfb51fc9e1953e56140c14451a519011898301/contrib/stack/topsStack/overlap_withDEM.py
- Pre-PR file (last commit to touch it, `e03042f4e5899019da413d6e7e79c4627b07482c`, 2020-07-02):
  - https://github.com/isce-framework/isce2/blob/e03042f4e5899019da413d6e7e79c4627b07482c/contrib/stack/topsStack/overlap_withDEM.py
  - raw: https://raw.githubusercontent.com/isce-framework/isce2/main/contrib/stack/topsStack/overlap_withDEM.py
- Original 2019 file (`9f9a2b0805beffd80a6539ca247fdaa8898fd8a9`, 2019-01-16):
  - https://github.com/isce-framework/isce2/blob/9f9a2b0805beffd80a6539ca247fdaa8898fd8a9/contrib/stack/topsStack/overlap_withDEM.py
- File history API: https://api.github.com/repos/isce-framework/isce2/commits?path=contrib/stack/topsStack/overlap_withDEM.py&per_page=30
- 2020 commits: https://github.com/isce-framework/isce2/commit/23f2ffaa1eeac93bb039f7259817810c4c3c0b12 , https://github.com/isce-framework/isce2/commit/e03042f4e5899019da413d6e7e79c4627b07482c
- geo2rdr kernel (offset-image semantics), `main` HEAD `b7702cfbd571681a133ebc5c3b23a10c78d52706`:
  - https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/components/zerodop/geo2rdr/src/geo2rdr.f90
  - raw: https://raw.githubusercontent.com/isce-framework/isce2/main/components/zerodop/geo2rdr/src/geo2rdr.f90
- topsStack geo2rdr driver: https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/geo2rdr.py
- Resampling consumer of the same offsets: https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/resamp_withCarrier.py
  - raw: https://raw.githubusercontent.com/isce-framework/isce2/main/contrib/stack/topsStack/resamp_withCarrier.py
- `getRelativeShifts`: https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/s1a_isce_utils.py
- TOPS carrier / burst limits: https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/components/isceobj/Sensor/TOPS/TOPSSwathSLCProduct.py
- ESD estimator (untouched by PR): https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/estimateAzimuthMisreg.py
- Stack.py on main (overlap_withDEM + overlap geo2rdr config): https://github.com/isce-framework/isce2/blob/b7702cfbd571681a133ebc5c3b23a10c78d52706/contrib/stack/topsStack/Stack.py
- CI checks: `gh pr checks 28139 --repo isce-framework/isce2` -> "no checks reported on the 'fix/topsstack-esd' branch"
