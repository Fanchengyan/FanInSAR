---
kind: knowledge
n: 2
title: "Profile the complete public call before native optimization"
status: active
type: process
scope: project
authors:
  - agent
tags:
  - profiling
  - performance
  - amdahl
  - native
  - diagnostics
  - ampcor
created: 2026-08-18
updated: 2026-08-18
derived_from:
  - PROPOSAL-0028
  - NOTE-0212
related_notes:
  - NOTE-0212
related_knowledge: []
maturity: raw
---

# Summary

Native optimization decisions must use warmed, end-to-end public-call
measurements. Amdahl accounting must count every repeated per-batch event and
divide by the number of timed public calls; diagnostic copies and correctness
capture must run outside those repeats. Stage rows are nested observations,
not an additive partition of the public-call time.

# Problem pattern

Optimizing one fast-looking stage can produce little or no public speedup when
NCC/peak work, transfers, staging, boundary handling, culling, cleanup, or
output conversion dominate. Counting one batch event as one public call
understates its cost. Capturing membership with `detach().cpu().numpy()` in a
timed repeat also adds non-product D2H work and can create a false bottleneck.

# Recommended approach

- Define the timing boundary around the complete public NumPy-in/Torch-or-native-out call.
- Warm up, synchronize the relevant device, and report repeated-call medians
  with build, preparation, first-call, and steady-state phases separated.
- Count all main batches for each public call before deriving stage fractions or
  Amdahl bounds. Keep stage measurements labeled as nested observations.
- Run membership capture, CPU diagnostics, and independent oracle comparisons
  in a separate correctness call; retain only product transfers in timed runs.
- Require ordered membership, counts, finite masks, tolerances, and aggregate
  parity before accepting a performance result.

For P28, the corrected A100 profile reduced the energy-only bounds to 1.030x
and 1.008x for `s8/b16` and `s16/b32`, while complete eager-to-native medians
improved only 1.070x and 1.015x. These are evidence for the measurement rule,
not universal speedup targets.

# Applicability

Use this pattern for staged numerical pipelines, especially batched SAR
operations where device transfers, reductions, boundary oracles, and output
assembly surround a candidate native kernel.

# Limitations and non-generalizations

- Amdahl bounds are only as reliable as the declared public-call boundary and
  event counts; they do not predict another shape, host, or workload.
- Nested stage timings must not be summed unless the instrumentation proves
  that the stages are exclusive.
- A corrected profiler supersedes an accounting-defective report for decisions,
  but the old report may remain as historical provenance.
- This record does not establish that native optimization is worthwhile for
  every profile or authorize changing product dispatch without qualification.

# Verification checklist

1. Freeze source, fixture, runtime, device, shape, dtype, and thread/launch profile.
2. Time the complete warmed public call with synchronized repeats.
3. Count every batch event and divide by timed public-call repeats.
4. Keep diagnostics and oracle copies outside timed execution.
5. Record build, preparation, first-call, warm, transfer, memory, and output
   phases without treating nested rows as an exclusive sum.
6. Gate any native default on end-to-end positive improvement plus numerical,
   membership, finite-mask, and memory parity.

# Provenance

This record distills the corrected P28 A100 profiler and audit in NOTE-0212,
with the public-call gates and baselines specified by PROPOSAL-0028. The
corrected report is authoritative for the P28 native-pipeline epoch; the
reported ratios are host-, fixture-, and profile-specific.
