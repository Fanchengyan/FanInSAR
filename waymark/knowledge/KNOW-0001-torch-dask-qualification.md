---
kind: knowledge
n: 1
title: "Torch/Dask acceleration requires measured CPU gates and explicit GPU qualification"
status: active
type: architecture
scope: project
authors:
  - agent
tags:
  - torch
  - dask
  - cuda
  - performance
  - rss
  - gpu-binding
  - esd
created: 2026-08-13
updated: 2026-08-13
derived_from:
  - PROPOSAL-0021
  - NOTE-0120
  - NOTE-0121
  - NOTE-0122
related_notes: []
related_knowledge: []
maturity: validated
---

# Summary

When migrating NumPy SAR algorithms to Torch, preserve a NumPy-in/NumPy-out
boundary and make acceleration a measured, per-kernel decision. CPU migration
must be evaluated end to end, distributed GPU execution must use an explicitly
trusted Dask client with fail-closed worker binding, and ESD needs a separate
memory strategy because it is a global reduction rather than a tiled image
kernel.

# Problem pattern

Replacing a NumPy kernel with Torch can improve CPU and CUDA throughput, but
unmeasured promotion can introduce slower execution, numerical drift,
unschedulable Dask graphs, or unsafe assumptions about GPU identity. ESD also
has global FFT/reduction behavior that does not fit ordinary block scheduling.

# Recommended approach

## CPU performance and memory gates

- Benchmark the complete NumPy-in/NumPy-out call after warmup on representative
  shapes.
- Record median runtime, peak RSS, dtype/parity metrics, and the gate decision
  in machine-readable output with logs separated from the JSON payload.
- Use a declared RSS limit such as
  `torch_peak <= max(1.25 * numpy_peak, numpy_peak + 512 MiB)` and retain the
  timing threshold with the artifact.
- Keep automatic CUDA promotion behind an authoritative per-kernel registry;
  kernels that fail speed or numerical qualification remain on the CPU
  reference path.
- A selected Dask client without GPU workers falls back to local eager Torch;
  only qualified kernels may submit remote GPU graphs. Explicit MPS and
  unqualified geographic reramp paths remain on the NumPy CPU reference path.
- A representative CPU timing gate is Torch/NumPy `<= 1.10x`; a representative
  CUDA promotion gate is `>= 1.5x` including transfer and scheduling overhead.
  CUDA memory must remain below 75% of the selected worker's available VRAM.

## Explicit Dask and GPU binding

- Public execution functions accept only an explicitly injected Dask client;
  ambient clients must not cause SAR data submission or graph execution.
- Submit GPU work only after checking the selected client's advertised worker
  resources and validating one visible CUDA ordinal plus a unique physical GPU
  identity per worker.
- Forward `device="cuda"` explicitly into scheduled kernels and fail closed on
  malformed resources or inconsistent identity after GPU admission. A client
  without GPU workers is not GPU admission and uses the documented local eager
  fallback; a worker that was admitted as GPU but cannot bind CUDA fails rather
  than silently falling back to CPU.
- Treat Dask transport authentication, authorization, and trusted-worker
  configuration as deployment prerequisites; resource metadata is not an
  authorization mechanism.

## ESD memory boundary

- Submit ESD as one `resources={"gpu": 1}` whole-array task to the selected GPU
  worker and gather only its small `ESDResult` object.
- Bound intermediate FFT memory inside the kernel with range-column chunks
  (512 columns is the validated default in this project).
- Qualify ESD independently from tiled image kernels and record the large-case
  chunk size, runtime, peak VRAM, and finite-result check. The validated
  acceptance case is 4000x15000 on a worker with at least 8 GiB VRAM.

# Applicability

Use this pattern for NumPy/SciPy SAR algorithms being moved to Torch CPU or
CUDA while retaining xarray/Dask-compatible public boundaries. It is most
useful when a pipeline supports both local execution and explicitly selected
remote workers.

# Limitations and non-generalizations

- Qualification measurements are environment- and shape-dependent; passing on
  one GPU server does not establish a universal speedup.
- The validated ESD chunk size is a starting point, not a promise for every
  input shape or GPU memory size.
- This record does not authorize ambient cluster discovery or replace cluster
  security configuration.
- MPS, geometry inversion, and unrelated kernels require their own numerical
  and performance evidence.

# Verification checklist

1. Compare warmed end-to-end NumPy and Torch timings on representative shapes.
2. Capture and parse peak RSS; fail the gate when the declared limit is missed.
3. Compare numerical outputs and dtypes against the NumPy reference.
4. Test explicit-client routing, ambient-client isolation, and CPU fallback.
5. Validate worker resource/ordinal/physical-identity binding before data
   submission.
6. For ESD, verify whole-array submission, internal chunking, peak VRAM, and a
   finite result on a representative large case.

# Provenance

The record distills the design and its implementation/verification evidence
for PROPOSAL-0021. Evidence included focused regression tests, a
machine-readable CPU RSS benchmark, and real-worker CUDA/Dask qualification
artifacts. The guidance is not a universal performance guarantee.
