---
kind: knowledge
n: 3
title: "Dispatch native kernels only for exact qualified profiles with Torch fallback"
status: active
type: architecture
scope: project
authors:
  - agent
tags:
  - native
  - torch
  - dispatch
  - fallback
  - qualification
  - compile
  - cuda
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

Native dispatch should be an exact, profile-local qualification decision. Key
the registry by shape, batch/search profile, host/device, runtime/compiler,
thread or launch profile, dtype, prepared generation, and source/ABI identity.
Select native only after correctness and warm end-to-end performance pass;
otherwise use the same-device Torch path. Prepared `torch.compile` is an
optional candidate whose setup cost must reach measured repeated-shape
break-even. Mature correlation FFT/cuFFT work remains in Torch rather than
being rewritten merely to expand native scope.

# Problem pattern

A native result qualified for one exact shape or batch profile can regress on a
nearby profile, use a stale binary, or violate numerical behavior. Dispatching
on a partial key silently broadens the qualification claim. Compile setup can
erase a small steady-state gain, and replacing a mature FFT implementation
adds risk without addressing the dominant public-call cost. Falling through to
NumPy, another device, reduced precision, or an unprepared compile breaks the
single Torch execution contract.

# Recommended approach

- Maintain a per-profile registry with source, ABI, runtime, device, shape,
  dtype, and generation checks. Revalidate the generation immediately before
  native entry and quarantine failed candidates.
- Measure and retain every eager, prepared-compile, and native cell. Enable
  native only for exact profiles with ordered membership/count, finite-mask,
  tolerance, determinism, memory, and positive end-to-end timing evidence.
- On a profile mismatch or pre-dispatch native failure, use the admitted
  same-device Torch fallback. If native execution has begun and crashes or
  poisons the CUDA context, propagate the failure and restart the worker or
  process; same-process recovery is not promised.
- Amortize compile preparation over the intended repeated shape and select it
  only when measured total cost reaches break-even. Keep mature Torch FFT/cuFFT
  stages as the reference and safety path while native NCC or other targeted
  work is qualified.

P28 Epoch2 qualified exact profiles `(search8, batch16)` and `(search16,
batch32)`. Its fresh default-dispatch probe made native calls for the exact
profiles and used Torch for a batch mismatch; measured native speedups were
1.16144x and 1.15562x on that A100 fixture. These results validate the dispatch
policy, not a universal native speedup.

# Applicability

Use for optional CPU/CUDA extensions, prepared kernels, and other accelerated
backends that share a Torch public contract and must coexist with a portable
reference path.

# Limitations and non-generalizations

- Qualification is bounded by the frozen source, fixture, runtime, host,
  device, shape, dtype, and launch profile; a nearby profile needs its own
  evidence.
- Compile break-even depends on call count and preparation/cache lifetime; a
  compile win in one campaign does not justify a default elsewhere.
- Torch fallback covers pre-dispatch recoverable failures only. Native crashes
  or poisoned contexts require process-level recovery.
- This record does not claim that every mature FFT or cuFFT stage should remain
  forever in Torch; replacement requires a separate end-to-end qualification.

# Verification checklist

1. Freeze and record all dispatch-key fields, source digest, ABI, and runtime.
2. Test exact-profile selection, mismatch-to-Torch behavior, stale generation,
   and candidate quarantine.
3. Compare eager, compile, and native correctness before timing warm repeats.
4. Separate build, preparation, first call, and warm execution; calculate
   compile break-even for the intended repeated-shape workload.
5. Verify same-device fallback, no NumPy/reduced-precision/cross-device retry,
   and process-restart behavior after native faults.
6. Keep mature FFT/cuFFT reference stages in the measured public-call budget.

# Provenance

The general dispatch and fallback guidance is distilled from PROPOSAL-0028;
the exact-profile and mismatch evidence comes from NOTE-0215, with the
complete six-mode qualification boundary and fallback behavior recorded in
NOTE-0211. All P28 ratios are scoped to their recorded A100 fixture and
profiles.
