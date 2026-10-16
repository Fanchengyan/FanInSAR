---
kind: knowledge
n: 8
title: "Keep one persisted WGS84 lon/lat DEM view per Stack for all geometry backends"
status: active
type: architecture
scope: project
authors:
  - agent
tags:
  - dem
  - geometry
  - wgs84
  - stack
  - persistence
  - cuda
  - torch
created: 2026-10-16
updated: 2026-10-16
derived_from:
  - PROPOSAL-0041
  - PROPOSAL-0058
  - NOTE-0366
related_notes:
  - NOTE-0366
related_knowledge: []
maturity: confirmed
---

# Summary

Materialize the Stack DEM once on the authoritative product grid
(`work_dir/dem/stack.tif`) and once as a native-spacing geographic crop
(`work_dir/dem/stack_wgs84.tif`). Every geometry backend (native CUDA,
Torch, NumPy) consumes one shared private lon/lat view built from the
WGS84 file instead of rebuilding its own geographic DEM. Reopen reloads the
persisted crop when the recorded product/provider still match, avoiding a
tens-of-minutes Copernicus mosaic rebuild.

# Problem pattern

Each backend previously built a private geographic view from the UTM product
raster, with per-backend resampling differences and edge handling. Reloading
a persisted raster also failed silently when identity checks compared path
provenance, so Stack rematerialized from source tiles on every reopen.

# Recommended approach

- Persist both rasters side by side under `work_dir/dem/`; keep the sidecar
  JSON recording product, provider, and both identities.
- Reopen accepts the geographic file when the source recipe still matches;
  do not compare content identity against in-memory rasters because
  RasterDEM identity includes path provenance.
- Save rasters atomically (temporary file plus replace) and write the CRS as
  WKT so reloaded files keep their projection in every reader.
- Backends consume one `lonlat_dem_view` composition; no backend reprojects
  the product grid itself.

# Verification checklist

1. Reopen does not call SourceDEM.to_raster again when the persisted files
   exist and the recipe matches.
2. Both files open with correct CRS and no zero-filled edges.
3. A full NESD campaign reuses the geographic DEM (log line "Reusing
   geographic DEM") on both a CPU and a CUDA host.
