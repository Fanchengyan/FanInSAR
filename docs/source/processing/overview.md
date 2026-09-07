# Processing overview

Stack processing stages (explicit, not a black box):

1. **READ** — open SAFE, build TOPS carrier from annotation, read complex burst window  
2. **DERAMP** — remove TOPS azimuth carrier  
3. **COREG** — geometry coarse shift (orbits) + amplitude correlation refine + resample + reramp  
4. **IFG** — complex interferogram, multilook, Goldstein filter  
5. **UNWRAP** — SNAPHU (the default spatial unwrapper)
6. **GEOCODE** — `rdr2geo` (+ DEM if provided) for unwrapped phase and coherence  
7. **WRITE** — Zarr (radar layers + `geocoded/` group) and STAC item  

## Stack API

```python
from faninsar.missions import S1Stack

stack = S1Stack.from_safes(
    ["acquisitions/20240101.SAFE", "acquisitions/20240113.SAFE"],
    work_dir="out/stack",
    reference="20240101",
)

# Each stage is explicit and resumable.
stack.prepare_scenes()
stack.coregister_scenes()
stack.form_interferograms(multilook=(2, 8))
stack.unwrap()
result = stack.analyze_time_series()
```

`Stack.unwrap()` uses SNAPHU by default. The Torch-native `SpatialIRLS`
backend remains available as an explicit developer/experimental strategy:

```python
from faninsar.processing.unwrapping import SpatialIRLS

stack.unwrap(SpatialIRLS())
```

For an existing interferogram collection, construct a path-based `Network`
instance instead. `Network` opens its persisted products internally and exposes
the same time-series analysis seam; it does not read SAFE/RSLC sources.
