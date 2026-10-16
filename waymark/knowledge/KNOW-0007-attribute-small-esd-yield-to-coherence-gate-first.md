---
kind: knowledge
n: 7
title: "Attribute small ESD n_valid to the coherence gate before suspecting geometry"
status: active
type: process
scope: project
authors:
  - agent
tags:
  - esd
  - nesd
  - coherence
  - diagnostics
  - cpu-cuda-parity
created: 2026-10-16
updated: 2026-10-16
derived_from:
  - PROPOSAL-0058
  - NOTE-0366
related_notes:
  - NOTE-0366
related_knowledge:
  - KNOW-0006
maturity: confirmed
---

# Summary

A small overlap-ESD `n_valid` is usually the coherence gate doing its job,
not a resampling, DEM, or device defect. Burst-edge overlap strips are
geometrically degraded: multilooked coherence medians of 0.12-0.16 with a
0.85 gate leave only 0.03-0.11 percent of cells, i.e. tens of cells from a
~34k-cell strip. Log the coherence distribution whenever an ESD result is
reported so the yield is attributable.

# Problem pattern

Comparing a CUDA run (n_valid 9-32) against stored CPU arcs (182-836)
suggested "CUDA loses valid pixels". The stored arcs came from an older code
state with different overlap geometry, and two pairs had been dropped
entirely by the invalid-line bug (KNOW-0006). With one fixed code base the
two devices agree within one cell on every pair.

# Recommended approach

- Log per-window ESD diagnostics: finite-cell count, coherence p50/p90, and
  fraction at or above the gate, alongside n_valid.
- Before attributing a device difference, re-measure both devices with the
  identical commit on the identical inputs; never compare against cached arcs
  from another code state.
- Treat date-constant differences below the sigma floor (0.01 px here) as
  sampling noise of tiny per-arc cell counts, not a bias.

# Verification checklist

1. Same commit, same SAFE set, CPU and CUDA: per-pair n_valid agreement
   within a few cells.
2. Coherence distribution (p50/p90/frac>=thr) logged for every window.
3. Recovered pairs produce observations on both devices.

