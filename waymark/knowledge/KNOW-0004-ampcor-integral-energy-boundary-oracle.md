---
kind: knowledge
n: 4
title: "Use float64 integral-image energy with a selective same-device FFT oracle"
status: active
type: architecture
scope: project
authors:
  - agent
tags:
  - ampcor
  - integral-image
  - float64
  - fft
  - numerical-safety
  - oracle
created: 2026-08-18
updated: 2026-08-18
derived_from:
  - PROPOSAL-0028
  - NOTE-0211
  - NOTE-0215
related_notes:
  - NOTE-0211
  - NOTE-0215
related_knowledge: []
maturity: raw
---

# Summary

For Ampcor local energy, a float64 2-D integral image removes redundant
all-ones and secondary-square FFTs while retaining one Torch computation
contract. Prefix arithmetic changes the summation order, so a conservative
per-window cancellation guard must selectively route risky windows to an
admitted same-device Torch FFT reference. NumPy remains an independent
accuracy/benchmark oracle only, never a production executor or runtime
fallback.

# Problem pattern

Reduced-precision prefix or correlation arithmetic can change cull membership,
ties, and surviving patches even when ordinary inputs look faster. A global
metric can miss a cancellation-risk window, while always recomputing FFTs
throws away the integral-image benefit. P28’s fixed matrix rejected every
tested float32 layer: `prefix_f32/correlation64` failed 14/212 membership
comparisons, `prefix64/correlation32/NCC32` 26/212, full float32 18/212, and
the mixed `prefix64/correlation32/NCC64` 36/212.

# Recommended approach

- Keep local-energy prefixes, reductions, peak/tie decisions, and cull logic
  in float64. Compare direct energy to an independent scalar oracle with the
  declared `1e-10 + 1e-12*abs(oracle)` tolerance.
- Evaluate cancellation over each output window and its halo with checked
  float64 arithmetic. Unknown, uniform, overflowed, or near-threshold risk
  must not authorize prefix execution.
- For a risky tile/window, preflight the workspace and rerun the affected
  computation on the same device with Torch FFT, including correlation, energy,
  NCC, peak/tie, SNR, subpixel, and cull decisions. If FFT admission fails,
  fail closed rather than changing device or precision.
- Preserve ordered membership, counts, finite masks, deterministic lowest-index
  ties, and aggregate parity. Release prefix temporaries before admitting the
  FFT packet when lifetimes cannot coexist.
- Use NumPy only outside production for independent scalar checks and benchmark
  comparison; do not add a NumPy runtime fallback.

P28’s Epoch2 evidence exercised high-dynamic whole-batch FFT fallback and a
boundary-small two-phase oracle with parity passing. This validates the guarded
boundary for the recorded Ampcor fixtures; it does not establish universal
coverage or speed.

# Applicability

Use for sliding-window sums or energy terms where prefix arithmetic is faster
but changes rounding order, especially numerical image and SAR correlation
pipelines with strict membership or threshold semantics.

# Limitations and non-generalizations

- Guard thresholds, tile sizes, workspace bounds, and tolerances require
  calibration against representative and adversarial data; they are not
  universal constants.
- Float32 speed measurements are diagnostic only when membership drifts; they
  do not qualify a fast mode or mixed-precision fallback.
- The FFT oracle is a same-device safety path, not permission to rewrite every
  FFT stage or to hide memory-admission failures.
- Inherited strict NumPy failure cases remain reportable limitations; passing
  Torch-vs-oracle checks on one fixture is not a global scientific guarantee.

# Verification checklist

1. Compare float64 integral energy with an independent scalar oracle over
   ordinary, periodic, low-texture, uniform, threshold, and real fixtures.
2. Test cancellation guard decisions over actual halos, including unknown and
   near-threshold windows and checked overflow/memory behavior.
3. Exercise same-device Torch FFT fallback and fail-closed admission/runtime
   failures; verify no NumPy, cross-device, or reduced-precision retry.
4. Compare ordered membership/counts, patch and aggregate values, finite masks,
   and deterministic tie behavior against the Torch reference.
5. Run the float32 and mixed-precision matrix as a rejection check; retain all
   membership mismatches in the evidence packet.
6. Measure end-to-end public-call timing and memory only after correctness and
   boundary-oracle gates pass.

# Provenance

The integral-image design, guard, and NumPy-oracle boundary come from
PROPOSAL-0028. The rejected float32 membership matrix and independent
precision/performance evidence are recorded in NOTE-0211; the final boundary
fallback and Epoch2 parity evidence are recorded in NOTE-0215. Claims are
limited to the validated P28 implementation epoch and fixtures.
